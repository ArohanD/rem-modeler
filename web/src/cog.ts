/**
 * COG mosaic reader: fills a target DemGrid from one or more remote
 * Cloud-Optimized GeoTIFFs (USGS 1 m DEM tiles) using HTTP range requests.
 * Replaces the GDAL BuildVRT + Translate merge step of the Python pipeline —
 * only the requested window, at the overview level matching the working
 * resolution, is ever downloaded.
 */

import { fromUrl, fromArrayBuffer, type GeoTIFF, type GeoTIFFImage } from 'geotiff';
import type { DemGrid } from './grid';

export interface TileReport {
  url: string;
  status: 'ok' | 'skipped-crs' | 'skipped-outside' | 'error';
  detail?: string;
}

export interface MosaicProgress {
  (done: number, total: number, note: string): void;
}

function imageEpsg(img: GeoTIFFImage): number | null {
  const gk = (img as unknown as { geoKeys?: Record<string, number> }).geoKeys;
  return gk?.ProjectedCSTypeGeoKey ?? null;
}

/** Pick the image (full-res or overview) whose resolution best matches
 *  targetRes without being coarser. */
async function pickImage(tiff: GeoTIFF, targetRes: number): Promise<GeoTIFFImage> {
  const count = await tiff.getImageCount();
  const base = await tiff.getImage(0);
  let chosen = base;
  let chosenRes = Math.abs(base.getResolution()[0]);
  for (let i = 1; i < count; i++) {
    const img = await tiff.getImage(i);
    const res = Math.abs(img.getResolution(base)[0]);
    if (res <= targetRes && res > chosenRes) {
      chosen = img;
      chosenRes = res;
    }
  }
  return chosen;
}

/**
 * Read the part of `tiff` that overlaps `target` and paste it into cells that
 * are still NaN (first tile wins where tiles overlap).
 */
export async function pasteTiff(tiff: GeoTIFF, target: DemGrid): Promise<TileReport['status']> {
  const base = await tiff.getImage(0);
  const epsg = imageEpsg(base);
  if (epsg !== null && epsg !== target.epsg) return 'skipped-crs';

  const [tMinX, tMinY, tMaxX, tMaxY] = base.getBoundingBox();
  const gMinX = target.originX;
  const gMaxY = target.originY;
  const gMaxX = gMinX + target.width * target.res;
  const gMinY = gMaxY - target.height * target.res;

  const ix0 = Math.max(tMinX, gMinX);
  const ix1 = Math.min(tMaxX, gMaxX);
  const iy0 = Math.max(tMinY, gMinY);
  const iy1 = Math.min(tMaxY, gMaxY);
  if (ix0 >= ix1 || iy0 >= iy1) return 'skipped-outside';

  // snap the intersection outward to target pixel boundaries
  const col0 = Math.max(0, Math.floor((ix0 - gMinX) / target.res));
  const col1 = Math.min(target.width, Math.ceil((ix1 - gMinX) / target.res));
  const row0 = Math.max(0, Math.floor((gMaxY - iy1) / target.res));
  const row1 = Math.min(target.height, Math.ceil((gMaxY - iy0) / target.res));
  const cols = col1 - col0;
  const rows = row1 - row0;
  if (cols <= 0 || rows <= 0) return 'skipped-outside';

  // world bbox of the pasted block
  const wx0 = gMinX + col0 * target.res;
  const wx1 = gMinX + col1 * target.res;
  const wy1 = gMaxY - row0 * target.res;
  const wy0 = gMaxY - row1 * target.res;

  const img = await pickImage(tiff, target.res);
  const imgResX = Math.abs(img.getResolution(base)[0]);
  const imgResY = Math.abs(img.getResolution(base)[1]) || imgResX;
  const [oX, oY] = base.getOrigin();

  const win = [
    Math.max(0, Math.floor((wx0 - oX) / imgResX)),
    Math.max(0, Math.floor((oY - wy1) / imgResY)),
    Math.min(img.getWidth(), Math.ceil((wx1 - oX) / imgResX)),
    Math.min(img.getHeight(), Math.ceil((oY - wy0) / imgResY)),
  ] as [number, number, number, number];
  if (win[2] <= win[0] || win[3] <= win[1]) return 'skipped-outside';

  const nodata = base.getGDALNoData();
  const rasters = await img.readRasters({
    window: win,
    width: cols,
    height: rows,
    samples: [0],
    resampleMethod: 'nearest',
    fillValue: NaN,
  });
  const block = rasters[0] as Float32Array | Float64Array | Int16Array;

  for (let r = 0; r < rows; r++) {
    const targetBase = (row0 + r) * target.width + col0;
    const blockBase = r * cols;
    for (let c = 0; c < cols; c++) {
      const ti = targetBase + c;
      if (!Number.isNaN(target.data[ti])) continue; // first tile wins
      let v = block[blockBase + c] as number;
      if (v === nodata || !Number.isFinite(v) || v < -1e30 || v > 1e30) v = NaN;
      target.data[ti] = v;
    }
  }
  return 'ok';
}

/** Build a NaN-initialized target grid. */
export function emptyGrid(
  originX: number,
  originY: number,
  res: number,
  width: number,
  height: number,
  epsg: number,
): DemGrid {
  const data = new Float32Array(width * height);
  data.fill(NaN);
  return { data, width, height, res, originX, originY, epsg };
}

/** Open each COG URL and paste it into the target grid. */
export async function mosaicFromUrls(
  urls: string[],
  target: DemGrid,
  onProgress?: MosaicProgress,
): Promise<TileReport[]> {
  const reports: TileReport[] = [];
  let done = 0;
  for (const url of urls) {
    onProgress?.(done, urls.length, `Reading ${url.split('/').pop()}`);
    try {
      const tiff = await fromUrl(url, { allowFullFile: false });
      const status = await pasteTiff(tiff, target);
      reports.push({ url, status });
    } catch (e) {
      reports.push({ url, status: 'error', detail: e instanceof Error ? e.message : String(e) });
    }
    done++;
    onProgress?.(done, urls.length, '');
  }
  return reports;
}

/**
 * Inspect the first COG to learn its CRS (used to pick the working UTM zone
 * when the source tiles dictate it).
 */
export async function probeEpsg(url: string): Promise<number | null> {
  const tiff = await fromUrl(url);
  const img = await tiff.getImage(0);
  return imageEpsg(img);
}

/** Load a complete (small) GeoTIFF from a buffer as its own DemGrid. */
export async function gridFromArrayBuffer(buf: ArrayBuffer): Promise<DemGrid> {
  const tiff = await fromArrayBuffer(buf);
  const img = await tiff.getImage(0);
  const width = img.getWidth();
  const height = img.getHeight();
  const [oX, oY] = img.getOrigin();
  const res = Math.abs(img.getResolution()[0]);
  const epsg = imageEpsg(img) ?? 0;
  const nodata = img.getGDALNoData();
  const rasters = await img.readRasters({ samples: [0] });
  const src = rasters[0] as ArrayLike<number>;
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) {
    let v = src[i];
    if (v === nodata || !Number.isFinite(v) || v < -1e30 || v > 1e30) v = NaN;
    data[i] = v;
  }
  return { data, width, height, res, originX: oX, originY: oY, epsg };
}
