/**
 * OpenStreetMap waterway fetch via Overpass — port of query_osm_waterways +
 * build_linestrings_from_osm. Tries multiple public Overpass endpoints.
 */

export interface Waterway {
  name: string;
  /** WGS84 lon/lat coordinates */
  coords: Array<[number, number]>;
}

const ENDPOINTS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
];

interface OsmElement {
  type: 'node' | 'way';
  id: number;
  lat?: number;
  lon?: number;
  nodes?: number[];
  tags?: Record<string, string>;
}

export async function fetchWaterways(
  bbox: [number, number, number, number],
  fetchImpl: typeof fetch = fetch,
): Promise<Waterway[]> {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const bboxStr = `${minLat},${minLon},${maxLat},${maxLon}`;
  const query = `
[out:json][timeout:60];
(
  way["waterway"="river"](${bboxStr});
  way["waterway"="stream"](${bboxStr});
  way["waterway"="canal"](${bboxStr});
);
out body;
>;
out skel qt;
`;

  let lastError: unknown = null;
  for (const url of ENDPOINTS) {
    try {
      const resp = await fetchImpl(url, {
        method: 'POST',
        body: new URLSearchParams({ data: query }),
      });
      if (!resp.ok) {
        lastError = new Error(`Overpass ${resp.status} ${resp.statusText}`);
        continue;
      }
      const json = (await resp.json()) as { elements?: OsmElement[] };
      return assembleWaterways(json.elements ?? []);
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('All Overpass endpoints failed');
}

export function assembleWaterways(elements: OsmElement[]): Waterway[] {
  const nodes = new Map<number, [number, number]>();
  const ways: OsmElement[] = [];
  for (const el of elements) {
    if (el.type === 'node' && el.lon !== undefined && el.lat !== undefined) {
      nodes.set(el.id, [el.lon, el.lat]);
    } else if (el.type === 'way') {
      ways.push(el);
    }
  }

  const out: Waterway[] = [];
  for (const way of ways) {
    const coords: Array<[number, number]> = [];
    for (const id of way.nodes ?? []) {
      const c = nodes.get(id);
      if (c) coords.push(c);
    }
    if (coords.length >= 2) {
      out.push({ name: way.tags?.name ?? `way ${way.id}`, coords });
    }
  }

  // merge ways that share a name AND an endpoint, so long rivers arrive whole
  return mergeByNameAndEndpoint(out);
}

function keyOf(c: [number, number]): string {
  return `${c[0].toFixed(7)},${c[1].toFixed(7)}`;
}

function mergeByNameAndEndpoint(ways: Waterway[]): Waterway[] {
  const byName = new Map<string, Waterway[]>();
  for (const w of ways) {
    const list = byName.get(w.name) ?? [];
    list.push({ name: w.name, coords: [...w.coords] });
    byName.set(w.name, list);
  }

  const merged: Waterway[] = [];
  for (const [name, list] of byName) {
    const pool = list.map((w) => w.coords);
    let progress = true;
    while (progress && pool.length > 1) {
      progress = false;
      outer: for (let i = 0; i < pool.length; i++) {
        for (let j = i + 1; j < pool.length; j++) {
          const a = pool[i];
          const b = pool[j];
          const joined = tryJoin(a, b);
          if (joined) {
            pool.splice(j, 1);
            pool[i] = joined;
            progress = true;
            break outer;
          }
        }
      }
    }
    for (const coords of pool) merged.push({ name, coords });
  }
  return merged;
}

function tryJoin(a: Array<[number, number]>, b: Array<[number, number]>): Array<[number, number]> | null {
  const aStart = keyOf(a[0]);
  const aEnd = keyOf(a[a.length - 1]);
  const bStart = keyOf(b[0]);
  const bEnd = keyOf(b[b.length - 1]);
  if (aEnd === bStart) return [...a, ...b.slice(1)];
  if (aEnd === bEnd) return [...a, ...b.slice(0, -1).reverse()];
  if (aStart === bEnd) return [...b, ...a.slice(1)];
  if (aStart === bStart) return [...b.slice(1).reverse(), ...a];
  return null;
}
