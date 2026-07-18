/**
 * Thin-plate-spline interpolation of scattered (x, y, z) samples — the
 * browser port of scipy's RBFInterpolator(kernel='thin_plate_spline',
 * smoothing=1.0) used by the Python pipeline.
 *
 * Model: s(x) = sum_i w_i * phi(|x - x_i|) + c0 + c1*x + c2*y,
 * phi(r) = r^2 * ln(r), with the standard KKT system
 *
 *   [ K + lambda*I   P ] [w]   [z]
 *   [ P^T            0 ] [c] = [0]
 *
 * Coordinates are shifted/scaled to a unit box before solving (same trick
 * scipy uses) so the dense solve stays well-conditioned.
 */

export interface TpsModel {
  weights: Float64Array; // n kernel weights followed by 3 polynomial coeffs
  xs: Float64Array;
  ys: Float64Array;
  shiftX: number;
  shiftY: number;
  scale: number;
}

function phi(r2: number): number {
  // r^2 * ln(r) = 0.5 * r^2 * ln(r^2); define phi(0) = 0
  return r2 > 0 ? 0.5 * r2 * Math.log(r2) : 0;
}

/** Solve A x = b in place via Gaussian elimination with partial pivoting. */
export function solveDense(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best === 0) throw new Error('Singular interpolation system (duplicate points?)');
    if (piv !== col) {
      for (let k = col; k < n; k++) {
        const t = A[col * n + k]; A[col * n + k] = A[piv * n + k]; A[piv * n + k] = t;
      }
      const t = b[col]; b[col] = b[piv]; b[piv] = t;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      if (f === 0) continue;
      A[r * n + col] = 0;
      for (let k = col + 1; k < n; k++) A[r * n + k] -= f * A[col * n + k];
      b[r] -= f * b[col];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

export function tpsFit(
  points: Array<[number, number, number]>,
  smoothing = 1.0,
): TpsModel {
  const n = points.length;
  if (n < 3) throw new Error('Need at least 3 points for TPS');

  // normalize domain to unit box for conditioning
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const scale = Math.max(maxX - minX, maxY - minY) || 1;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const zs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = (points[i][0] - minX) / scale;
    ys[i] = (points[i][1] - minY) / scale;
    zs[i] = points[i][2];
  }

  const m = n + 3;
  const A = new Float64Array(m * m);
  const b = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const dx = xs[i] - xs[j];
      const dy = ys[i] - ys[j];
      const v = phi(dx * dx + dy * dy);
      A[i * m + j] = v;
      A[j * m + i] = v;
    }
    A[i * m + i] += smoothing;
    // polynomial columns
    A[i * m + n] = 1;
    A[i * m + n + 1] = xs[i];
    A[i * m + n + 2] = ys[i];
    A[n * m + i] = 1;
    A[(n + 1) * m + i] = xs[i];
    A[(n + 2) * m + i] = ys[i];
    b[i] = zs[i];
  }

  const weights = solveDense(A, b, m);
  return { weights, xs, ys, shiftX: minX, shiftY: minY, scale };
}

export function tpsEvaluate(model: TpsModel, x: number, y: number): number {
  const n = model.xs.length;
  const px = (x - model.shiftX) / model.scale;
  const py = (y - model.shiftY) / model.scale;
  let s = model.weights[n] + model.weights[n + 1] * px + model.weights[n + 2] * py;
  for (let i = 0; i < n; i++) {
    const dx = px - model.xs[i];
    const dy = py - model.ys[i];
    s += model.weights[i] * phi(dx * dx + dy * dy);
  }
  return s;
}

/**
 * Evaluate the model over a regular grid (row-major, top row first).
 * Calls onProgress with [0..1] between row chunks so a worker can report.
 */
export function tpsEvaluateGrid(
  model: TpsModel,
  originX: number,
  originY: number,
  res: number,
  width: number,
  height: number,
  onProgress?: (frac: number) => void,
): Float32Array {
  const out = new Float32Array(width * height);
  for (let r = 0; r < height; r++) {
    const y = originY - (r + 0.5) * res;
    for (let c = 0; c < width; c++) {
      out[r * width + c] = tpsEvaluate(model, originX + (c + 0.5) * res, y);
    }
    if (onProgress && (r & 15) === 0) onProgress(r / height);
  }
  onProgress?.(1);
  return out;
}
