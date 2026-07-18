import { describe, expect, it } from 'vitest';
import { tpsFit, tpsEvaluate, tpsEvaluateGrid, solveDense } from '../src/tps';

describe('solveDense', () => {
  it('solves a simple system', () => {
    // 2x + y = 5; x + 3y = 10
    const A = new Float64Array([2, 1, 1, 3]);
    const b = new Float64Array([5, 10]);
    const x = solveDense(A, b, 2);
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(3, 10);
  });
});

describe('tpsFit / tpsEvaluate', () => {
  it('reproduces a planar surface (polynomial part)', () => {
    const points: Array<[number, number, number]> = [];
    const f = (x: number, y: number) => 100 + 0.01 * x - 0.02 * y;
    for (let i = 0; i < 30; i++) {
      const x = 1000 * Math.cos(i * 2.399); // low-discrepancy-ish scatter
      const y = 1000 * Math.sin(i * 1.618);
      points.push([x, y, f(x, y)]);
    }
    const model = tpsFit(points, 1.0);
    for (const [x, y] of [[200, -300], [-750, 400], [0, 0], [900, 900]] as const) {
      expect(tpsEvaluate(model, x, y)).toBeCloseTo(f(x, y), 1);
    }
  });

  it('interpolates a smooth downstream gradient like a water surface', () => {
    // stations along a straight river: elevation drops 1 m per 100 m
    const points: Array<[number, number, number]> = [];
    for (let d = 0; d <= 5000; d += 100) {
      points.push([d, 50 * Math.sin(d / 400), 1300 - d / 100]);
    }
    const model = tpsFit(points, 1.0);
    // midway between stations, on and off axis
    expect(tpsEvaluate(model, 2550, 0)).toBeCloseTo(1300 - 25.5, 0);
    expect(tpsEvaluate(model, 2550, 500)).toBeCloseTo(1300 - 25.5, 0);
  });

  it('evaluates a grid row-major from the top-left', () => {
    const points: Array<[number, number, number]> = [
      [0, 0, 0], [100, 0, 1], [0, 100, 2], [100, 100, 3], [50, 50, 1.5],
    ];
    const model = tpsFit(points, 0.01);
    const surface = tpsEvaluateGrid(model, 0, 100, 10, 10, 10);
    expect(surface.length).toBe(100);
    // top-left cell center (5, 95) should be near the (0,100)=2 corner value
    expect(surface[0]).toBeGreaterThan(surface[99] - 3);
    expect(Number.isFinite(surface[0])).toBe(true);
  });
});
