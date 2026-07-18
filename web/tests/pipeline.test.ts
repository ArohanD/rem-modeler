/**
 * End-to-end numeric pipeline on the synthetic valley: perturbed centerline
 * -> snap to channel -> sample stations -> TPS water surface -> REM.
 * Asserts the REM is ~0 along the channel and positive on the valley walls.
 */

import { describe, expect, it } from 'vitest';
import { makeSyntheticDem } from './synthetic';
import { snapCenterlineToChannel, sampleElevations, densifyLine, pointsAlongLine, lineLength, effectiveSpacing } from '../src/centerline';
import { tpsFit, tpsEvaluateGrid } from '../src/tps';
import { upsampleBilinear, sampleNearest } from '../src/grid';
import type { XY } from '../src/centerline';

describe('centerline utilities', () => {
  it('densifies so no segment exceeds the max length', () => {
    const dense = densifyLine([[0, 0], [100, 0]], 30);
    for (let i = 1; i < dense.length; i++) {
      expect(Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1])).toBeLessThanOrEqual(30 + 1e-9);
    }
  });

  it('spaces points evenly along the line', () => {
    const pts = pointsAlongLine([[0, 0], [100, 0], [100, 100]], 25);
    expect(lineLength(pts)).toBeCloseTo(200, 6);
    expect(pts.length).toBe(9); // 0,25,...,200
  });

  it('widens spacing to cap station count', () => {
    const line: XY[] = [[0, 0], [100000, 0]];
    const spacing = effectiveSpacing(line, 10, 1200);
    expect(Math.floor(100000 / spacing) + 1).toBeLessThanOrEqual(1200);
  });
});

describe('synthetic pipeline', () => {
  const { grid, channel, waterElevAtY } = makeSyntheticDem(512, 512, 2);

  it('snap pulls a laterally offset centerline into the channel', () => {
    // shift the true channel 30 m east
    const offset: XY[] = channel.map(([x, y]) => [x + 30, y]);
    const snapped = snapCenterlineToChannel(grid, offset, {
      searchRadius: 60,
      pointSpacing: 20,
    });
    // sampled elevations along the snapped line should sit near the channel
    // bottom, i.e. below the water elevation at that latitude (bank is higher)
    const samples = sampleElevations(grid, snapped, 20);
    expect(samples.length).toBeGreaterThan(20);
    let nearBottom = 0;
    for (const [, y, z] of samples) {
      if (z <= waterElevAtY(y) - 0.5) nearBottom++;
    }
    expect(nearBottom / samples.length).toBeGreaterThan(0.8);
  });

  it('TPS water surface + REM recovers height-above-river', () => {
    const stations = sampleElevations(grid, channel, 25);
    const model = tpsFit(stations, 1.0);

    // evaluate on a 4x coarse grid, then upsample — same shape as the app
    const cw = grid.width / 4;
    const ch = grid.height / 4;
    const coarse = tpsEvaluateGrid(model, grid.originX, grid.originY, grid.res * 4, cw, ch);
    const ws = upsampleBilinear(coarse, cw, ch, grid.width, grid.height);

    const rem = new Float32Array(grid.data.length);
    for (let i = 0; i < rem.length; i++) rem[i] = grid.data[i] - ws[i];

    // Along the channel the REM should be near -channelDepth..0; sample a few
    for (const [x, y] of channel.filter((_, i) => i % 5 === 0)) {
      const col = Math.round((x - grid.originX) / grid.res - 0.5);
      const row = Math.round((grid.originY - y) / grid.res - 0.5);
      const v = rem[row * grid.width + col];
      expect(v).toBeGreaterThan(-6);
      expect(v).toBeLessThan(2);
    }

    // On the valley wall (100 px east of channel) REM should be clearly positive
    let positives = 0;
    let count = 0;
    for (const [x, y] of channel.filter((_, i) => i % 5 === 0)) {
      const v = remAt(rem, grid.width, grid, x + 200, y);
      if (!Number.isNaN(v)) {
        count++;
        if (v > 2) positives++;
      }
    }
    expect(positives / count).toBeGreaterThan(0.9);
  });

  function remAt(rem: Float32Array, width: number, g: typeof grid, x: number, y: number): number {
    const col = Math.round((x - g.originX) / g.res - 0.5);
    const row = Math.round((g.originY - y) / g.res - 0.5);
    if (col < 0 || col >= width || row < 0 || row >= g.height) return NaN;
    return rem[row * width + col];
  }

  it('nearest sampling matches direct indexing', () => {
    const v = sampleNearest(grid, grid.originX + 10.5 * grid.res, grid.originY - 20.5 * grid.res);
    expect(v).toBe(grid.data[20 * grid.width + 10]);
  });
});
