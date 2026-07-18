/**
 * Memory / size budgeting for the browser REM pipeline.
 *
 * The working grid is the single biggest memory consumer. For a grid of
 * N pixels the app holds, worst case, roughly:
 *
 *   DEM            Float32   4 B/px
 *   water surface  Float32   4 B/px
 *   REM            Float32   4 B/px
 *   hillshade      Uint8     1 B/px
 *   display RGBA   Uint8x4   4 B/px  (ImageData)
 *   backing canvas Uint8x4   4 B/px  (browser-side copy of the ImageData)
 *   ------------------------------------------------
 *   total                   ~21 B/px  -> budgeted at 24 B/px for slack
 *
 * We derive a pixel budget from navigator.deviceMemory when the browser
 * exposes it (Chrome/Edge; capped at 8 by spec) and fall back to a
 * conservative default elsewhere. Tiers:
 *
 *   deviceMemory >= 8  -> 512 MB working budget -> ~22 Mpx -> maxDim 4096
 *   deviceMemory >= 4  -> 384 MB                -> ~16 Mpx -> maxDim 4096
 *   unknown            -> 320 MB                -> ~13 Mpx -> maxDim 3584
 *   deviceMemory <= 2  -> 192 MB                ->  ~8 Mpx -> maxDim 2560
 *
 * Independent of memory, the bounding box itself is capped: beyond ~40 km a
 * side, the ground resolution needed to fit the pixel budget exceeds ~16 m/px
 * and a "relative elevation" of a 1 m-lidar river stops being meaningful.
 * We warn beyond 15 km (>= ~8 m/px on default budgets).
 */

export const BYTES_PER_PIXEL_BUDGET = 24;
export const HARD_MAX_DIM = 4096; // canvas + typed-array practicality ceiling
export const HARD_MAX_BBOX_KM = 40;
export const WARN_BBOX_KM = 15;
/** Never fetch finer than the source (1 m); never coarser than this. */
export const MAX_RES_M = 32;

export interface CapsProfile {
  budgetMB: number;
  maxPixels: number;
  maxDim: number;
  deviceMemoryGB: number | null;
}

export function capsProfile(deviceMemoryGB?: number | null): CapsProfile {
  const dm = deviceMemoryGB ?? null;
  let budgetMB: number;
  if (dm == null) budgetMB = 320;
  else if (dm >= 8) budgetMB = 512;
  else if (dm >= 4) budgetMB = 384;
  else budgetMB = 192;
  const maxPixels = Math.floor((budgetMB * 1024 * 1024) / BYTES_PER_PIXEL_BUDGET);
  const maxDim = Math.min(HARD_MAX_DIM, Math.floor(Math.sqrt(maxPixels * 1.3)));
  return { budgetMB, maxPixels, maxDim, deviceMemoryGB: dm };
}

export interface GridPlan {
  /** chosen ground resolution, m/px */
  res: number;
  width: number;
  height: number;
  pixels: number;
  /** estimated peak working-set MB at BYTES_PER_PIXEL_BUDGET */
  estimatedMB: number;
  /** true when the bbox exceeds the hard cap and must be rejected */
  rejected: boolean;
  /** human-readable warnings (soft limits) */
  warnings: string[];
}

/** Snap a required resolution up to the next power-of-two multiple of 1 m
 *  (1, 2, 4, 8, 16, 32) so reads align with COG overview levels. */
export function snapResolution(required: number): number {
  let r = 1;
  while (r < required && r < MAX_RES_M) r *= 2;
  return r;
}

/**
 * Plan the working grid for a bbox of extentX x extentY meters under a caps
 * profile. Chooses the finest COG-aligned resolution that fits both the
 * per-dimension and total-pixel budgets.
 */
export function planGrid(extentX: number, extentY: number, caps: CapsProfile): GridPlan {
  const warnings: string[] = [];
  const maxSideKm = Math.max(extentX, extentY) / 1000;
  if (extentX <= 0 || extentY <= 0) {
    return { res: 1, width: 0, height: 0, pixels: 0, estimatedMB: 0, rejected: true, warnings: ['Empty selection'] };
  }
  if (maxSideKm > HARD_MAX_BBOX_KM) {
    return {
      res: MAX_RES_M,
      width: 0,
      height: 0,
      pixels: 0,
      estimatedMB: 0,
      rejected: true,
      warnings: [
        `Selection is ${maxSideKm.toFixed(1)} km on its longest side; the maximum is ${HARD_MAX_BBOX_KM} km. ` +
          'Draw a smaller area.',
      ],
    };
  }

  const required = Math.max(
    1,
    extentX / caps.maxDim,
    extentY / caps.maxDim,
    Math.sqrt((extentX * extentY) / caps.maxPixels),
  );
  const res = snapResolution(required);
  const width = Math.max(1, Math.ceil(extentX / res));
  const height = Math.max(1, Math.ceil(extentY / res));
  const pixels = width * height;
  const estimatedMB = (pixels * BYTES_PER_PIXEL_BUDGET) / (1024 * 1024);

  if (maxSideKm > WARN_BBOX_KM) {
    warnings.push(
      `Large area (${maxSideKm.toFixed(1)} km): working resolution drops to ${res} m/px; ` +
        'subtle floodplain detail will be lost.',
    );
  } else if (res > 4) {
    warnings.push(`Working resolution is ${res} m/px on this device; smaller areas give finer detail.`);
  }

  return { res, width, height, pixels, estimatedMB, rejected: false, warnings };
}
