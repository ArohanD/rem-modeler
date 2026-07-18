/**
 * DEM discovery/loading providers for an arbitrary WGS84 bounding box.
 *
 *  - TNM API: asks the USGS National Map catalog which 1 m COG tiles
 *    intersect the bbox, then streams them with cog.ts.
 *  - 3DEP ImageServer: single exportImage request returning a float32
 *    GeoTIFF mosaic at exactly the working grid size (covers all of the US
 *    at whatever the best available source is, server-side resampled).
 *
 * Both endpoints are public. Errors are surfaced so the UI can offer the
 * direct-URL / local-file fallbacks.
 */

const TNM_API = 'https://tnmaccess.nationalmap.gov/api/v1/products';
const IMAGE_SERVER = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer';

export const MAX_TILES = 40;

export interface DiscoveryResult {
  urls: string[];
  total: number;
  truncated: boolean;
}

interface TnmItem {
  downloadURL?: string;
  urls?: Record<string, string>;
  format?: string;
}

/** Find USGS 1 m DEM COG URLs intersecting a WGS84 bbox via the TNM API. */
export async function discoverOneMeterTiles(
  bbox: [number, number, number, number],
  fetchImpl: typeof fetch = fetch,
): Promise<DiscoveryResult> {
  const urls = new Set<string>();
  let offset = 0;
  let total = Infinity;
  while (offset < total && urls.size < MAX_TILES * 2) {
    const params = new URLSearchParams({
      datasets: 'Digital Elevation Model (DEM) 1 meter',
      bbox: bbox.join(','),
      prodFormats: 'GeoTIFF',
      outputFormat: 'JSON',
      max: '100',
      offset: String(offset),
    });
    const resp = await fetchImpl(`${TNM_API}?${params.toString()}`);
    if (!resp.ok) throw new Error(`TNM API ${resp.status} ${resp.statusText}`);
    const json = (await resp.json()) as { total?: number; items?: TnmItem[] };
    total = json.total ?? 0;
    const items = json.items ?? [];
    if (items.length === 0) break;
    for (const it of items) {
      const u = it.urls?.['GeoTIFF'] ?? it.downloadURL;
      if (u && /\/Elevation\/1m\//.test(u) && u.toLowerCase().endsWith('.tif')) urls.add(u);
    }
    offset += items.length;
  }
  const list = [...urls];
  return {
    urls: list.slice(0, MAX_TILES),
    total: list.length,
    truncated: list.length > MAX_TILES,
  };
}

/**
 * Fetch a float32 GeoTIFF mosaic for a projected bbox from the 3DEP
 * ImageServer, sized to the working grid. Returns the raw TIFF bytes.
 */
export async function fetchImageServerTiff(
  projBbox: [number, number, number, number],
  epsg: number,
  width: number,
  height: number,
  fetchImpl: typeof fetch = fetch,
): Promise<ArrayBuffer> {
  const params = new URLSearchParams({
    f: 'image',
    bbox: projBbox.join(','),
    bboxSR: String(epsg),
    imageSR: String(epsg),
    size: `${width},${height}`,
    format: 'tiff',
    pixelType: 'F32',
    interpolation: 'RSP_BilinearInterpolation',
  });
  const resp = await fetchImpl(`${IMAGE_SERVER}/exportImage?${params.toString()}`);
  if (!resp.ok) throw new Error(`3DEP ImageServer ${resp.status} ${resp.statusText}`);
  const ct = resp.headers.get('content-type') ?? '';
  const buf = await resp.arrayBuffer();
  if (ct.includes('json')) {
    // ArcGIS reports errors as JSON with HTTP 200
    const text = new TextDecoder().decode(buf);
    throw new Error(`3DEP ImageServer error: ${text.slice(0, 300)}`);
  }
  return buf;
}
