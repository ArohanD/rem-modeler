/**
 * Centerline operations: densify, snap-to-channel, smooth, sample.
 * Ports of densify_line / snap_centerline_to_channel /
 * create_points_from_centerline / sample_elevations_along_line from the
 * Python pipeline, operating on the in-memory DemGrid.
 */

import type { DemGrid } from './grid';
import { sampleNearest } from './grid';

export type XY = [number, number];

export function lineLength(coords: XY[]): number {
  let len = 0;
  for (let i = 1; i < coords.length; i++) {
    len += Math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1]);
  }
  return len;
}

/** Add intermediate points so no segment exceeds maxSegment. */
export function densifyLine(coords: XY[], maxSegment: number): XY[] {
  if (coords.length < 2) return coords.slice();
  const out: XY[] = [coords[0]];
  for (let i = 1; i < coords.length; i++) {
    const [x0, y0] = coords[i - 1];
    const [x1, y1] = coords[i];
    const seg = Math.hypot(x1 - x0, y1 - y0);
    if (seg > maxSegment) {
      const nSeg = Math.ceil(seg / maxSegment);
      for (let j = 1; j < nSeg; j++) {
        const t = j / nSeg;
        out.push([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]);
      }
    }
    out.push(coords[i]);
  }
  return out;
}

/** Evenly spaced points along the line (plus the endpoint). */
export function pointsAlongLine(coords: XY[], spacing: number): XY[] {
  const total = lineLength(coords);
  if (total === 0 || coords.length < 2) return coords.slice();
  const out: XY[] = [];
  let target = 0;
  let acc = 0;
  let i = 1;
  let prev = coords[0];
  while (i < coords.length) {
    const cur = coords[i];
    const seg = Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);
    while (acc + seg >= target) {
      const t = seg === 0 ? 0 : (target - acc) / seg;
      out.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
      target += spacing;
      if (target > total) break;
    }
    acc += seg;
    prev = cur;
    i++;
  }
  const last = coords[coords.length - 1];
  const tail = out[out.length - 1];
  if (!tail || Math.hypot(tail[0] - last[0], tail[1] - last[1]) > 1e-6) out.push([last[0], last[1]]);
  return out;
}

/** 5-point moving average, keeping the first/last two points fixed. */
export function smoothLine(coords: XY[]): XY[] {
  if (coords.length <= 5) return coords.slice();
  const out: XY[] = coords.slice(0, 2);
  for (let i = 2; i < coords.length - 2; i++) {
    let sx = 0, sy = 0;
    for (let k = -2; k <= 2; k++) {
      sx += coords[i + k][0];
      sy += coords[i + k][1];
    }
    out.push([sx / 5, sy / 5]);
  }
  out.push(coords[coords.length - 2], coords[coords.length - 1]);
  return out;
}

export interface SnapParams {
  /** search radius in meters */
  searchRadius: number;
  /** densification spacing in meters */
  pointSpacing: number;
  /** only consider pixels within this elevation range, if set */
  elevRange?: [number, number] | null;
}

/**
 * Snap each (densified) centerline point to the lowest valid elevation within
 * searchRadius, then smooth. Points with no valid pixels in range keep their
 * original position.
 */
export function snapCenterlineToChannel(grid: DemGrid, coords: XY[], params: SnapParams): XY[] {
  const dense = densifyLine(coords, params.pointSpacing);
  const searchPx = Math.max(1, Math.round(params.searchRadius / grid.res));
  const [lo, hi] = params.elevRange ?? [-Infinity, Infinity];
  const snapped: XY[] = [];

  for (const [x, y] of dense) {
    const col = Math.floor((x - grid.originX) / grid.res);
    const row = Math.floor((grid.originY - y) / grid.res);
    const r0 = Math.max(0, row - searchPx);
    const r1 = Math.min(grid.height - 1, row + searchPx);
    const c0 = Math.max(0, col - searchPx);
    const c1 = Math.min(grid.width - 1, col + searchPx);
    let bestVal = Infinity;
    let bestR = -1, bestC = -1;
    for (let r = r0; r <= r1; r++) {
      const base = r * grid.width;
      for (let c = c0; c <= c1; c++) {
        const v = grid.data[base + c];
        if (Number.isNaN(v) || v < lo || v > hi) continue;
        if (v < bestVal) { bestVal = v; bestR = r; bestC = c; }
      }
    }
    if (bestR < 0) {
      snapped.push([x, y]);
    } else {
      snapped.push([
        grid.originX + (bestC + 0.5) * grid.res,
        grid.originY - (bestR + 0.5) * grid.res,
      ]);
    }
  }

  return smoothLine(snapped);
}

/** Sample grid elevations at evenly spaced stations along the line. */
export function sampleElevations(grid: DemGrid, coords: XY[], spacing: number): Array<[number, number, number]> {
  const pts = pointsAlongLine(coords, spacing);
  const out: Array<[number, number, number]> = [];
  for (const [x, y] of pts) {
    const v = sampleNearest(grid, x, y);
    if (!Number.isNaN(v)) out.push([x, y, v]);
  }
  return out;
}

/**
 * Cap the number of interpolation stations by widening spacing if needed.
 * Returns the effective spacing to keep the TPS solve tractable (O(n^3)).
 */
export function effectiveSpacing(coords: XY[], requestedSpacing: number, maxPoints: number): number {
  const len = lineLength(coords);
  const n = Math.floor(len / requestedSpacing) + 1;
  if (n <= maxPoints) return requestedSpacing;
  return len / (maxPoints - 1);
}
