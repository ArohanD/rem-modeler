/**
 * Synthetic river-valley DEM for tests: a sloping valley with a meandering
 * incised channel. The analytic channel path and water elevation are known,
 * so tests can assert the pipeline recovers them.
 */

import type { DemGrid } from '../src/grid';

export interface SyntheticDem {
  grid: DemGrid;
  /** channel path in world coords, upstream (top) to downstream (bottom) */
  channel: Array<[number, number]>;
  /** water elevation at a given row */
  waterElevAtY: (y: number) => number;
}

export function makeSyntheticDem(width = 512, height = 512, res = 2): SyntheticDem {
  const originX = 300000;
  const originY = 4370000;
  const epsg = 26911;
  const data = new Float32Array(width * height);

  const midCol = width / 2;
  const amplitudePx = width * 0.15;
  const wavelengthPx = height / 2.5;
  const slopePerPx = 0.02; // downstream gradient, m per pixel of row
  const baseElev = 1300;
  const valleyDepth = 30;
  const channelDepth = 3;
  const channelWidthPx = 6;

  const channelColAtRow = (r: number): number =>
    midCol + amplitudePx * Math.sin((2 * Math.PI * r) / wavelengthPx);

  // water surface falls linearly downstream (increasing row)
  const waterAtRow = (r: number): number => baseElev + (height - r) * slopePerPx;

  for (let r = 0; r < height; r++) {
    const cc = channelColAtRow(r);
    const water = waterAtRow(r);
    for (let c = 0; c < width; c++) {
      const dist = Math.abs(c - cc);
      // valley walls: parabolic rise away from channel
      const wall = valleyDepth * Math.pow(dist / (width / 2), 1.6);
      // channel: incised notch around the path
      const notch = dist < channelWidthPx ? channelDepth * (1 - dist / channelWidthPx) : 0;
      const noise = 0.15 * Math.sin(r * 0.7) * Math.cos(c * 0.9);
      data[r * width + c] = water + wall - notch + noise;
    }
  }

  const channel: Array<[number, number]> = [];
  for (let r = 4; r < height - 4; r += 8) {
    const c = channelColAtRow(r);
    channel.push([originX + (c + 0.5) * res, originY - (r + 0.5) * res]);
  }

  return {
    grid: { data, width, height, res, originX, originY, epsg },
    channel,
    waterElevAtY: (y: number) => waterAtRow((originY - y) / res - 0.5),
  };
}
