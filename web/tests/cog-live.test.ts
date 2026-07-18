/**
 * Live integration test against real USGS 1 m COG tiles on prd-tnm S3.
 * Network-dependent, so it only runs when COG_LIVE=1 is set:
 *
 *   COG_LIVE=1 npm test
 *
 * (In proxied environments, undici's EnvHttpProxyAgent picks up HTTPS_PROXY;
 * point NODE_EXTRA_CA_CERTS at the proxy CA bundle if TLS is intercepted.)
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { emptyGrid, mosaicFromUrls, probeEpsg } from '../src/cog';
import { gridStats } from '../src/grid';

const TILES = [
  'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x30y436_NV_WestCentral_EarthMRI_2020_D20.tif',
  'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x31y436_NV_WestCentral_EarthMRI_2020_D20.tif',
];

describe.skipIf(!process.env.COG_LIVE)('live COG mosaic (prd-tnm S3)', () => {
  beforeAll(async () => {
    if (process.env.HTTPS_PROXY || process.env.https_proxy) {
      const { fetch: proxiedFetch, EnvHttpProxyAgent } = await import('undici');
      const agent = new EnvHttpProxyAgent();
      globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) =>
        proxiedFetch(input as never, { ...(init as object), dispatcher: agent } as never)) as unknown as typeof fetch;
    }
  });

  it('probes the CRS of a real tile', async () => {
    const epsg = await probeEpsg(TILES[0]);
    // NV_WestCentral is published in a NAD83 UTM zone 11 variant
    expect([26911, 6340, 32611]).toContain(epsg);
  }, 60_000);

  it('mosaics a window spanning two tiles at overview resolution', async () => {
    const epsg = (await probeEpsg(TILES[0]))!;
    // 2 km x 1 km straddling the x30/x31 tile boundary (easting 310000);
    // tile yNNN indices give the TOP edge, so y436 spans 4350000..4360000
    const target = emptyGrid(309000, 4356000, 4, 500, 250, epsg);
    const reports = await mosaicFromUrls(TILES, target);

    expect(reports.filter((r) => r.status === 'ok').length, JSON.stringify(reports)).toBe(2);
    const stats = gridStats(target);
    expect(stats.validFraction).toBeGreaterThan(0.95);
    // west-central Nevada terrain: plausible elevation band
    expect(stats.min).toBeGreaterThan(500);
    expect(stats.max).toBeLessThan(4000);
    expect(stats.max - stats.min).toBeGreaterThan(1); // actual relief, not a constant fill
  }, 120_000);
});
