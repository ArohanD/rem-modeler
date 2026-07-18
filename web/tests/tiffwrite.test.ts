/**
 * Round-trip: our minimal GeoTIFF writer must be readable by geotiff.js with
 * intact georeferencing, EPSG code, and float data (including NaN nodata).
 */

import { describe, expect, it } from 'vitest';
import { fromArrayBuffer } from 'geotiff';
import { writeGeoTiff } from '../src/tiffwrite';
import { gridFromArrayBuffer } from '../src/cog';
import { makeSyntheticDem } from './synthetic';
import type { DemGrid } from '../src/grid';

describe('writeGeoTiff', () => {
  const grid: DemGrid = {
    data: new Float32Array([1.5, 2.5, NaN, 4.5, 5.5, 6.5]),
    width: 3,
    height: 2,
    res: 2,
    originX: 500000,
    originY: 4400000,
    epsg: 26911,
  };

  it('round-trips through geotiff.js', async () => {
    const buf = writeGeoTiff(grid);
    const tiff = await fromArrayBuffer(buf);
    const img = await tiff.getImage(0);

    expect(img.getWidth()).toBe(3);
    expect(img.getHeight()).toBe(2);
    expect(img.getOrigin()[0]).toBeCloseTo(500000);
    expect(img.getOrigin()[1]).toBeCloseTo(4400000);
    expect(Math.abs(img.getResolution()[0])).toBeCloseTo(2);
    const geoKeys = (img as unknown as { geoKeys: Record<string, number> }).geoKeys;
    expect(geoKeys.ProjectedCSTypeGeoKey).toBe(26911);

    const rasters = await img.readRasters();
    const band = rasters[0] as Float32Array;
    expect(band[0]).toBeCloseTo(1.5);
    expect(band[1]).toBeCloseTo(2.5);
    expect(Number.isNaN(band[2])).toBe(true);
    expect(band[5]).toBeCloseTo(6.5);
  });

  it('round-trips through our own reader (gridFromArrayBuffer)', async () => {
    const g2 = await gridFromArrayBuffer(writeGeoTiff(grid));
    expect(g2.width).toBe(grid.width);
    expect(g2.height).toBe(grid.height);
    expect(g2.epsg).toBe(26911);
    expect(g2.res).toBeCloseTo(2);
    expect(g2.data[4]).toBeCloseTo(5.5);
    expect(Number.isNaN(g2.data[2])).toBe(true);
  });

  it('handles a full synthetic DEM', async () => {
    const { grid: dem } = makeSyntheticDem(128, 96, 2);
    const g2 = await gridFromArrayBuffer(writeGeoTiff(dem));
    expect(g2.width).toBe(128);
    expect(g2.height).toBe(96);
    expect(g2.data[50 * 128 + 60]).toBeCloseTo(dem.data[50 * 128 + 60], 4);
  });
});
