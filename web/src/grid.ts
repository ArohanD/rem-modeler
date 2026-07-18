/**
 * DemGrid: an in-memory single-band float raster on a regular projected grid.
 *
 * Convention matches GDAL/rasterio: originX/originY is the outer corner of the
 * top-left pixel, res is the pixel size in CRS units (square pixels, north-up).
 * NoData is represented as NaN throughout the app.
 */

export interface DemGrid {
  data: Float32Array;
  width: number;
  height: number;
  /** pixel size in CRS units (meters for UTM) */
  res: number;
  /** X of the left edge of column 0 */
  originX: number;
  /** Y of the top edge of row 0 */
  originY: number;
  /** EPSG code of the projected CRS */
  epsg: number;
}

/** World coords of a pixel center. */
export function pixelToWorld(g: DemGrid, col: number, row: number): [number, number] {
  return [g.originX + (col + 0.5) * g.res, g.originY - (row + 0.5) * g.res];
}

/** Fractional pixel coords (col, row) for a world coordinate. */
export function worldToPixel(g: DemGrid, x: number, y: number): [number, number] {
  return [(x - g.originX) / g.res - 0.5, (g.originY - y) / g.res - 0.5];
}

/** Nearest-neighbor sample; NaN outside the grid. */
export function sampleNearest(g: DemGrid, x: number, y: number): number {
  const col = Math.round((x - g.originX) / g.res - 0.5);
  const row = Math.round((g.originY - y) / g.res - 0.5);
  if (col < 0 || row < 0 || col >= g.width || row >= g.height) return NaN;
  return g.data[row * g.width + col];
}

/** Bilinear sample; falls back to nearest when neighbors are NaN. */
export function sampleBilinear(g: DemGrid, x: number, y: number): number {
  const [fc, fr] = worldToPixel(g, x, y);
  const c0 = Math.floor(fc);
  const r0 = Math.floor(fr);
  if (c0 < -1 || r0 < -1 || c0 >= g.width || r0 >= g.height) return NaN;
  const c1 = Math.min(g.width - 1, c0 + 1);
  const r1 = Math.min(g.height - 1, r0 + 1);
  const cc0 = Math.max(0, c0);
  const rr0 = Math.max(0, r0);
  const tx = Math.min(1, Math.max(0, fc - c0));
  const ty = Math.min(1, Math.max(0, fr - r0));
  const v00 = g.data[rr0 * g.width + cc0];
  const v10 = g.data[rr0 * g.width + c1];
  const v01 = g.data[r1 * g.width + cc0];
  const v11 = g.data[r1 * g.width + c1];
  if (Number.isNaN(v00) || Number.isNaN(v10) || Number.isNaN(v01) || Number.isNaN(v11)) {
    return sampleNearest(g, x, y);
  }
  const a = v00 + (v10 - v00) * tx;
  const b = v01 + (v11 - v01) * tx;
  return a + (b - a) * ty;
}

export interface GridStats {
  min: number;
  max: number;
  /** approximate percentiles from a subsample */
  p2: number;
  p50: number;
  p98: number;
  validFraction: number;
}

/** Stats from a subsample of at most ~200k pixels (fast on any grid size). */
export function gridStats(g: DemGrid): GridStats {
  const n = g.width * g.height;
  const stride = Math.max(1, Math.floor(n / 200_000));
  const vals: number[] = [];
  let valid = 0;
  let seen = 0;
  for (let i = 0; i < n; i += stride) {
    seen++;
    const v = g.data[i];
    if (!Number.isNaN(v)) {
      valid++;
      vals.push(v);
    }
  }
  vals.sort((a, b) => a - b);
  const pick = (p: number) => (vals.length ? vals[Math.min(vals.length - 1, Math.floor((p / 100) * vals.length))] : NaN);
  return {
    min: vals.length ? vals[0] : NaN,
    max: vals.length ? vals[vals.length - 1] : NaN,
    p2: pick(2),
    p50: pick(50),
    p98: pick(98),
    validFraction: seen ? valid / seen : 0,
  };
}

/**
 * Bilinearly resample a coarse grid of values (covering exactly the same
 * extent as the target grid) up to targetW x targetH.
 */
export function upsampleBilinear(
  coarse: Float32Array,
  cw: number,
  ch: number,
  targetW: number,
  targetH: number,
): Float32Array {
  const out = new Float32Array(targetW * targetH);
  const sx = cw / targetW;
  const sy = ch / targetH;
  for (let r = 0; r < targetH; r++) {
    // map target pixel center into coarse pixel space
    const fy = (r + 0.5) * sy - 0.5;
    const r0 = Math.max(0, Math.min(ch - 1, Math.floor(fy)));
    const r1 = Math.min(ch - 1, r0 + 1);
    const ty = Math.min(1, Math.max(0, fy - r0));
    for (let c = 0; c < targetW; c++) {
      const fx = (c + 0.5) * sx - 0.5;
      const c0 = Math.max(0, Math.min(cw - 1, Math.floor(fx)));
      const c1 = Math.min(cw - 1, c0 + 1);
      const tx = Math.min(1, Math.max(0, fx - c0));
      const v00 = coarse[r0 * cw + c0];
      const v10 = coarse[r0 * cw + c1];
      const v01 = coarse[r1 * cw + c0];
      const v11 = coarse[r1 * cw + c1];
      const a = v00 + (v10 - v00) * tx;
      const b = v01 + (v11 - v01) * tx;
      out[r * targetW + c] = a + (b - a) * ty;
    }
  }
  return out;
}
