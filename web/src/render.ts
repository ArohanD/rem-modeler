/**
 * Raster rendering: hillshade (port of compute_hillshade) and composited
 * RGBA views for the explore and REM stages.
 */

import type { DemGrid } from './grid';

/** Hillshade 0..255; NaN cells get 128. Port of the numpy implementation. */
export function computeHillshade(
  grid: DemGrid,
  azimuthDeg = 315,
  altitudeDeg = 45,
  zFactor = 1,
): Uint8Array {
  const { data, width, height, res } = grid;
  const out = new Uint8Array(width * height);
  const az = (azimuthDeg * Math.PI) / 180;
  const alt = (altitudeDeg * Math.PI) / 180;
  const sinAlt = Math.sin(alt);
  const cosAlt = Math.cos(alt);

  const get = (r: number, c: number): number => {
    const v = data[Math.max(0, Math.min(height - 1, r)) * width + Math.max(0, Math.min(width - 1, c))];
    return Number.isNaN(v) ? 0 : v;
  };

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const v = data[r * width + c];
      if (Number.isNaN(v)) { out[r * width + c] = 128; continue; }
      // central differences, matching np.gradient axis order (x = d/drow)
      const gx = ((get(r + 1, c) - get(r - 1, c)) * zFactor) / (2 * res);
      const gy = ((get(r, c + 1) - get(r, c - 1)) * zFactor) / (2 * res);
      const slope = Math.PI / 2 - Math.atan(Math.hypot(gx, gy));
      const aspect = Math.atan2(-gx, gy);
      let shaded = sinAlt * Math.sin(slope) + cosAlt * Math.cos(slope) * Math.cos(az - aspect);
      shaded = ((shaded + 1) / 2) * 255;
      out[r * width + c] = shaded < 0 ? 0 : shaded > 255 ? 255 : shaded;
    }
  }
  return out;
}

type ColorStop = [number, number, number];

function buildLut(stops: ColorStop[]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  const n = stops.length - 1;
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * n;
    const k = Math.min(n - 1, Math.floor(t));
    const f = t - k;
    for (let ch = 0; ch < 3; ch++) {
      lut[i * 3 + ch] = Math.round(stops[k][ch] + f * (stops[k + 1][ch] - stops[k][ch]));
    }
  }
  return lut;
}

// matplotlib YlGnBu anchor colors (light -> dark)
const YLGNBU = buildLut([
  [255, 255, 217], [237, 248, 177], [199, 233, 180], [127, 205, 187],
  [65, 182, 196], [29, 145, 192], [34, 94, 168], [37, 52, 148], [8, 29, 88],
]);

// muted terrain ramp for the explore view
const TERRAIN = buildLut([
  [84, 110, 84], [140, 160, 120], [200, 196, 148], [222, 206, 170], [240, 235, 220],
]);

export interface ExploreParams {
  elevMin: number; // display stretch
  elevMax: number;
  rangeLo: number | null; // river highlight range
  rangeHi: number | null;
  hillshadeAlpha: number; // 0..1 blend of hillshade over color
  hillshade: Uint8Array | null;
}

/** Compose the explore view into an RGBA buffer. */
export function renderExplore(grid: DemGrid, p: ExploreParams, out: Uint8ClampedArray): void {
  const { data } = grid;
  const n = grid.width * grid.height;
  const span = p.elevMax - p.elevMin || 1;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    const o = i * 4;
    if (Number.isNaN(v)) {
      out[o] = 24; out[o + 1] = 26; out[o + 2] = 30; out[o + 3] = 255;
      continue;
    }
    let t = (v - p.elevMin) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const li = Math.round(t * 255) * 3;
    let r = TERRAIN[li], g = TERRAIN[li + 1], b = TERRAIN[li + 2];
    if (p.rangeLo != null && p.rangeHi != null && v >= p.rangeLo && v <= p.rangeHi) {
      // river highlight: strong blue
      r = 40; g = 110; b = 235;
    }
    if (p.hillshade && p.hillshadeAlpha > 0) {
      const h = p.hillshade[i] / 255;
      const a = p.hillshadeAlpha;
      const shade = 1 - a + a * h * 1.6; // let highlights brighten a bit
      r = Math.min(255, r * shade); g = Math.min(255, g * shade); b = Math.min(255, b * shade);
    }
    out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
  }
}

export interface RemParams {
  vmax: number; // meters above water surface at the dark end
  hillshade: Uint8Array | null;
  hillshadeAlpha: number;
}

/** Compose the REM view (YlGnBu reversed-intensity like the Python tool). */
export function renderRem(rem: Float32Array, p: RemParams, out: Uint8ClampedArray): void {
  const n = rem.length;
  const vmax = p.vmax || 1;
  for (let i = 0; i < n; i++) {
    const v = rem[i];
    const o = i * 4;
    if (Number.isNaN(v)) {
      out[o] = 24; out[o + 1] = 26; out[o + 2] = 30; out[o + 3] = 255;
      continue;
    }
    // matplotlib: imshow(rem, cmap YlGnBu, vmin=0, vmax) maps 0 -> light.
    // Water (0) should be dark blue, so invert like the classic REM styling.
    let t = 1 - v / vmax;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const li = Math.round(t * 255) * 3;
    let r = YLGNBU[li], g = YLGNBU[li + 1], b = YLGNBU[li + 2];
    if (p.hillshade && p.hillshadeAlpha > 0) {
      const h = p.hillshade[i] / 255;
      const a = p.hillshadeAlpha;
      const shade = 1 - a + a * h * 1.5;
      r = Math.min(255, r * shade); g = Math.min(255, g * shade); b = Math.min(255, b * shade);
    }
    out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
  }
}
