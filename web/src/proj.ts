/** CRS helpers built on proj4. The app works internally in a UTM zone. */

import proj4 from 'proj4';

export function utmZoneForLon(lon: number): number {
  return Math.max(1, Math.min(60, Math.floor((lon + 180) / 6) + 1));
}

/** proj4 definition string for a (northern-hemisphere) UTM zone on NAD83/WGS84. */
function utmDef(zone: number, south: boolean): string {
  return `+proj=utm +zone=${zone}${south ? ' +south' : ''} +datum=NAD83 +units=m +no_defs`;
}

/**
 * Resolve an EPSG code to a proj4 definition. Covers the codes that USGS 1 m
 * DEMs and common exports actually use:
 *   269xx  NAD83 UTM zone xx (north)
 *   63xx   NAD83(2011) UTM (6330 + zone = zone N)
 *   326xx / 327xx  WGS84 UTM north / south
 * NAD83 vs WGS84 datum differences (~1 m) are ignored — acceptable for a PoC.
 */
export function defForEpsg(epsg: number): string | null {
  if (epsg >= 26901 && epsg <= 26923) return utmDef(epsg - 26900, false);
  if (epsg >= 6330 && epsg <= 6348) return utmDef(epsg - 6329, false);
  if (epsg >= 32601 && epsg <= 32660) return utmDef(epsg - 32600, false);
  if (epsg >= 32701 && epsg <= 32760) return utmDef(epsg - 32700, true);
  if (epsg === 4326) return '+proj=longlat +datum=WGS84 +no_defs';
  if (epsg === 3857) return '+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs';
  return null;
}

export function epsgForUtmZone(zone: number): number {
  return 26900 + zone; // NAD83 UTM north — matches USGS 1m products
}

export interface BboxTransform {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Forward-project a WGS84 lon/lat bbox into the given EPSG, densifying edges. */
export function projectBbox(lonLat: BboxTransform, epsg: number): BboxTransform {
  const def = defForEpsg(epsg);
  if (!def) throw new Error(`Unsupported EPSG:${epsg}`);
  const fwd = proj4('+proj=longlat +datum=WGS84 +no_defs', def).forward;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const N = 16;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const pts = [
      [lonLat.minX + t * (lonLat.maxX - lonLat.minX), lonLat.minY],
      [lonLat.minX + t * (lonLat.maxX - lonLat.minX), lonLat.maxY],
      [lonLat.minX, lonLat.minY + t * (lonLat.maxY - lonLat.minY)],
      [lonLat.maxX, lonLat.minY + t * (lonLat.maxY - lonLat.minY)],
    ];
    for (const p of pts) {
      const [x, y] = fwd(p as [number, number]);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  return { minX, minY, maxX, maxY };
}

export function transformer(fromEpsg: number, toEpsg: number): (xy: [number, number]) => [number, number] {
  const from = defForEpsg(fromEpsg);
  const to = defForEpsg(toEpsg);
  if (!from || !to) throw new Error(`Unsupported EPSG ${fromEpsg} -> ${toEpsg}`);
  const conv = proj4(from, to);
  return (xy) => conv.forward(xy) as [number, number];
}
