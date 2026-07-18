/**
 * Workbench: canvas viewport over the working DemGrid with pan/zoom, a hover
 * elevation readout, and interactive centerline editing (draw, drag, insert,
 * delete vertices) in grid space.
 */

import type { DemGrid } from './grid';
import { sampleNearest } from './grid';
import type { XY } from './centerline';

export type WorkbenchMode = 'pan' | 'draw' | 'edit';

export interface WorkbenchCallbacks {
  onHover?: (x: number, y: number, elev: number) => void;
  onCenterlineChange?: (coords: XY[]) => void;
  onModeChange?: (mode: WorkbenchMode) => void;
}

const VERTEX_HIT_PX = 8;
const SEGMENT_HIT_PX = 6;

export class Workbench {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private backing: HTMLCanvasElement | null = null;
  private grid: DemGrid | null = null;

  /** view transform: screen = (gridPx - viewX/viewY) * scale */
  private scale = 1;
  private viewX = 0;
  private viewY = 0;

  mode: WorkbenchMode = 'pan';
  centerline: XY[] = [];
  auxLines: Array<{ coords: XY[]; highlight: boolean }> = [];
  showCenterline = true;

  private draggingView = false;
  private dragVertex = -1;
  private hoverVertex = -1;
  private lastMouse: [number, number] = [0, 0];
  private cb: WorkbenchCallbacks;

  constructor(canvas: HTMLCanvasElement, cb: WorkbenchCallbacks = {}) {
    this.canvas = canvas;
    this.cb = cb;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('no 2d context');
    this.ctx = ctx;
    this.bind();
  }

  setGrid(grid: DemGrid): void {
    this.grid = grid;
    this.backing = document.createElement('canvas');
    this.backing.width = grid.width;
    this.backing.height = grid.height;
    this.fitView();
  }

  /** Push a freshly rendered RGBA frame for the current grid. */
  setImage(rgba: Uint8ClampedArray<ArrayBuffer>): void {
    if (!this.grid || !this.backing) return;
    const ctx = this.backing.getContext('2d')!;
    ctx.putImageData(new ImageData(rgba, this.grid.width, this.grid.height), 0, 0);
    this.render();
  }

  setMode(mode: WorkbenchMode): void {
    this.mode = mode;
    this.canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
    this.cb.onModeChange?.(mode);
    this.render();
  }

  setCenterline(coords: XY[]): void {
    this.centerline = coords.map((c) => [c[0], c[1]]);
    this.cb.onCenterlineChange?.(this.centerline);
    this.render();
  }

  fitView(): void {
    if (!this.grid) return;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
    this.canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
    const sx = this.canvas.width / this.grid.width;
    const sy = this.canvas.height / this.grid.height;
    this.scale = Math.min(sx, sy) * 0.98;
    this.viewX = this.grid.width / 2 - this.canvas.width / 2 / this.scale;
    this.viewY = this.grid.height / 2 - this.canvas.height / 2 / this.scale;
    this.render();
  }

  // ---- coordinate mapping -------------------------------------------------

  private screenToGridPx(sx: number, sy: number): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    const px = ((sx - r.left) * devicePixelRatio) / this.scale + this.viewX;
    const py = ((sy - r.top) * devicePixelRatio) / this.scale + this.viewY;
    return [px, py];
  }

  private gridPxToScreen(px: number, py: number): [number, number] {
    return [(px - this.viewX) * this.scale, (py - this.viewY) * this.scale];
  }

  private worldToGridPx(x: number, y: number): [number, number] {
    const g = this.grid!;
    return [(x - g.originX) / g.res, (g.originY - y) / g.res];
  }

  private gridPxToWorld(px: number, py: number): [number, number] {
    const g = this.grid!;
    return [g.originX + px * g.res, g.originY - py * g.res];
  }

  // ---- events -------------------------------------------------------------

  private bind(): void {
    const c = this.canvas;
    c.addEventListener('mousedown', (e) => this.onDown(e));
    window.addEventListener('mousemove', (e) => this.onMove(e));
    window.addEventListener('mouseup', () => this.onUp());
    c.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    c.addEventListener('dblclick', (e) => this.onDblClick(e));
    c.addEventListener('contextmenu', (e) => this.onContextMenu(e));
    window.addEventListener('keydown', (e) => this.onKey(e));
    new ResizeObserver(() => this.fitView()).observe(c);
  }

  private hitVertex(sx: number, sy: number): number {
    if (!this.grid || !this.showCenterline) return -1;
    for (let i = 0; i < this.centerline.length; i++) {
      const [px, py] = this.worldToGridPx(this.centerline[i][0], this.centerline[i][1]);
      const [vx, vy] = this.gridPxToScreen(px, py);
      const r = this.canvas.getBoundingClientRect();
      const mx = (sx - r.left) * devicePixelRatio;
      const my = (sy - r.top) * devicePixelRatio;
      if (Math.hypot(vx - mx, vy - my) <= VERTEX_HIT_PX * devicePixelRatio) return i;
    }
    return -1;
  }

  /** Index of segment whose interior is near the cursor, or -1. */
  private hitSegment(sx: number, sy: number): number {
    if (!this.grid || this.centerline.length < 2) return -1;
    const r = this.canvas.getBoundingClientRect();
    const mx = (sx - r.left) * devicePixelRatio;
    const my = (sy - r.top) * devicePixelRatio;
    for (let i = 0; i < this.centerline.length - 1; i++) {
      const a = this.gridPxToScreen(...this.worldToGridPx(this.centerline[i][0], this.centerline[i][1]));
      const b = this.gridPxToScreen(...this.worldToGridPx(this.centerline[i + 1][0], this.centerline[i + 1][1]));
      const vx = b[0] - a[0];
      const vy = b[1] - a[1];
      const len2 = vx * vx + vy * vy;
      if (len2 === 0) continue;
      let t = ((mx - a[0]) * vx + (my - a[1]) * vy) / len2;
      t = Math.max(0, Math.min(1, t));
      const dx = mx - (a[0] + t * vx);
      const dy = my - (a[1] + t * vy);
      if (Math.hypot(dx, dy) <= SEGMENT_HIT_PX * devicePixelRatio && t > 0.02 && t < 0.98) return i;
    }
    return -1;
  }

  private onDown(e: MouseEvent): void {
    if (!this.grid) return;
    this.lastMouse = [e.clientX, e.clientY];
    if (e.button === 1 || this.mode === 'pan' || e.shiftKey) {
      this.draggingView = true;
      this.canvas.style.cursor = 'grabbing';
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;

    if (this.mode === 'edit') {
      const vi = this.hitVertex(e.clientX, e.clientY);
      if (vi >= 0) {
        this.dragVertex = vi;
        return;
      }
      const si = this.hitSegment(e.clientX, e.clientY);
      if (si >= 0) {
        const [px, py] = this.screenToGridPx(e.clientX, e.clientY);
        const w = this.gridPxToWorld(px, py);
        this.centerline.splice(si + 1, 0, [w[0], w[1]]);
        this.dragVertex = si + 1;
        this.cb.onCenterlineChange?.(this.centerline);
        this.render();
        return;
      }
      this.draggingView = true; // fall back to panning
    } else if (this.mode === 'draw') {
      const [px, py] = this.screenToGridPx(e.clientX, e.clientY);
      const w = this.gridPxToWorld(px, py);
      this.centerline.push([w[0], w[1]]);
      this.cb.onCenterlineChange?.(this.centerline);
      this.render();
    }
  }

  private onMove(e: MouseEvent): void {
    if (!this.grid) return;
    if (this.draggingView) {
      const dx = (e.clientX - this.lastMouse[0]) * devicePixelRatio;
      const dy = (e.clientY - this.lastMouse[1]) * devicePixelRatio;
      this.viewX -= dx / this.scale;
      this.viewY -= dy / this.scale;
      this.lastMouse = [e.clientX, e.clientY];
      this.render();
      return;
    }
    if (this.dragVertex >= 0) {
      const [px, py] = this.screenToGridPx(e.clientX, e.clientY);
      const w = this.gridPxToWorld(px, py);
      this.centerline[this.dragVertex] = [w[0], w[1]];
      this.cb.onCenterlineChange?.(this.centerline);
      this.render();
      return;
    }
    // hover
    const inCanvas = e.target === this.canvas;
    if (inCanvas) {
      const [px, py] = this.screenToGridPx(e.clientX, e.clientY);
      const w = this.gridPxToWorld(px, py);
      const elev = sampleNearest(this.grid, w[0], w[1]);
      this.cb.onHover?.(w[0], w[1], elev);
      if (this.mode === 'edit') {
        const vi = this.hitVertex(e.clientX, e.clientY);
        if (vi !== this.hoverVertex) {
          this.hoverVertex = vi;
          this.render();
        }
        this.canvas.style.cursor = vi >= 0 ? 'move' : this.hitSegment(e.clientX, e.clientY) >= 0 ? 'copy' : 'crosshair';
      }
    }
  }

  private onUp(): void {
    if (this.draggingView) {
      this.draggingView = false;
      this.canvas.style.cursor = this.mode === 'pan' ? 'grab' : 'crosshair';
    }
    this.dragVertex = -1;
  }

  private onWheel(e: WheelEvent): void {
    if (!this.grid) return;
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.0015);
    const [px, py] = this.screenToGridPx(e.clientX, e.clientY);
    this.scale = Math.max(0.05, Math.min(64, this.scale * factor));
    // keep the point under the cursor fixed
    const r = this.canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * devicePixelRatio;
    const my = (e.clientY - r.top) * devicePixelRatio;
    this.viewX = px - mx / this.scale;
    this.viewY = py - my / this.scale;
    this.render();
  }

  private onDblClick(e: MouseEvent): void {
    if (this.mode === 'draw') {
      e.preventDefault();
      this.setMode('edit');
    }
  }

  private onContextMenu(e: MouseEvent): void {
    if (this.mode !== 'edit') return;
    const vi = this.hitVertex(e.clientX, e.clientY);
    if (vi >= 0) {
      e.preventDefault();
      this.centerline.splice(vi, 1);
      this.hoverVertex = -1;
      this.cb.onCenterlineChange?.(this.centerline);
      this.render();
    }
  }

  private onKey(e: KeyboardEvent): void {
    if ((e.key === 'Delete' || e.key === 'Backspace') && this.mode === 'edit' && this.hoverVertex >= 0) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      e.preventDefault();
      this.centerline.splice(this.hoverVertex, 1);
      this.hoverVertex = -1;
      this.cb.onCenterlineChange?.(this.centerline);
      this.render();
    }
    if (e.key === 'Escape' && this.mode === 'draw') this.setMode('edit');
  }

  // ---- drawing ------------------------------------------------------------

  render(): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = '#14161a';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.backing && this.grid) {
      ctx.imageSmoothingEnabled = this.scale < 1;
      ctx.setTransform(this.scale, 0, 0, this.scale, -this.viewX * this.scale, -this.viewY * this.scale);
      ctx.drawImage(this.backing, 0, 0);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.drawLines(ctx);
    }
    ctx.restore();
  }

  private drawLines(ctx: CanvasRenderingContext2D): void {
    if (!this.grid) return;
    for (const aux of this.auxLines) {
      this.strokeLine(ctx, aux.coords, aux.highlight ? 'rgba(255,210,80,0.9)' : 'rgba(255,255,255,0.35)', 1.5);
    }
    if (this.showCenterline && this.centerline.length > 0) {
      this.strokeLine(ctx, this.centerline, '#ff4d6a', 2.5);
      // vertices
      const dpr = devicePixelRatio;
      for (let i = 0; i < this.centerline.length; i++) {
        const [px, py] = this.worldToGridPx(this.centerline[i][0], this.centerline[i][1]);
        const [x, y] = this.gridPxToScreen(px, py);
        ctx.beginPath();
        ctx.arc(x, y, (i === this.hoverVertex ? 6 : 4) * dpr, 0, Math.PI * 2);
        ctx.fillStyle = i === this.hoverVertex ? '#ffd24d' : '#ffffff';
        ctx.strokeStyle = '#ff4d6a';
        ctx.lineWidth = 1.5 * dpr;
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  private strokeLine(ctx: CanvasRenderingContext2D, coords: XY[], style: string, widthPx: number): void {
    if (coords.length < 2) return;
    ctx.beginPath();
    for (let i = 0; i < coords.length; i++) {
      const [px, py] = this.worldToGridPx(coords[i][0], coords[i][1]);
      const [x, y] = this.gridPxToScreen(px, py);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = style;
    ctx.lineWidth = widthPx * devicePixelRatio;
    ctx.stroke();
  }
}
