/// <reference lib="webworker" />
/**
 * Web Worker that fits the thin-plate spline and evaluates it on the coarse
 * grid, posting progress along the way. Keeps the O(n^3) solve and the
 * grid evaluation off the main thread.
 */

import { tpsFit, tpsEvaluateGrid } from './tps';

export interface TpsRequest {
  points: Array<[number, number, number]>;
  originX: number;
  originY: number;
  res: number;
  width: number;
  height: number;
  smoothing: number;
}

export type TpsResponse =
  | { kind: 'progress'; frac: number }
  | { kind: 'done'; surface: Float32Array; width: number; height: number }
  | { kind: 'error'; message: string };

self.onmessage = (ev: MessageEvent<TpsRequest>) => {
  const req = ev.data;
  try {
    post({ kind: 'progress', frac: 0 });
    const model = tpsFit(req.points, req.smoothing);
    post({ kind: 'progress', frac: 0.05 });
    const surface = tpsEvaluateGrid(
      model,
      req.originX,
      req.originY,
      req.res,
      req.width,
      req.height,
      (f) => post({ kind: 'progress', frac: 0.05 + 0.95 * f }),
    );
    (self as unknown as Worker).postMessage(
      { kind: 'done', surface, width: req.width, height: req.height } satisfies TpsResponse,
      [surface.buffer],
    );
  } catch (e) {
    post({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};

function post(msg: TpsResponse): void {
  (self as unknown as Worker).postMessage(msg);
}
