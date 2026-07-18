/**
 * Minimal GeoTIFF writer: single-band float32, uncompressed, striped,
 * little-endian, georeferenced via ModelPixelScale + ModelTiepoint and a
 * GeoKeyDirectory carrying the projected EPSG code. NaN is declared as
 * nodata via the GDAL_NODATA ASCII tag, which GDAL/QGIS/rasterio honor.
 */

import type { DemGrid } from './grid';

interface TagEntry {
  tag: number;
  type: number; // 2=ASCII, 3=SHORT, 4=LONG, 12=DOUBLE
  values: number[] | string;
}

const TYPE_SIZES: Record<number, number> = { 2: 1, 3: 2, 4: 4, 12: 8 };

export function writeGeoTiff(grid: DemGrid): ArrayBuffer {
  const { width, height, res, originX, originY, epsg } = grid;
  const stripByteCount = width * height * 4;

  const geoKeys = [
    1, 1, 0, 4, // version, rev, minor, number of keys
    1024, 0, 1, 1, // GTModelTypeGeoKey = projected
    1025, 0, 1, 1, // GTRasterTypeGeoKey = PixelIsArea
    3072, 0, 1, epsg, // ProjectedCSTypeGeoKey
    3076, 0, 1, 9001, // ProjLinearUnitsGeoKey = meter
  ];

  const tags: TagEntry[] = [
    { tag: 256, type: 4, values: [width] },
    { tag: 257, type: 4, values: [height] },
    { tag: 258, type: 3, values: [32] },
    { tag: 259, type: 3, values: [1] }, // no compression
    { tag: 262, type: 3, values: [1] }, // BlackIsZero
    { tag: 273, type: 4, values: [0] }, // StripOffsets (patched below)
    { tag: 277, type: 3, values: [1] },
    { tag: 278, type: 4, values: [height] },
    { tag: 279, type: 4, values: [stripByteCount] },
    { tag: 284, type: 3, values: [1] },
    { tag: 339, type: 3, values: [3] }, // SampleFormat = IEEE float
    { tag: 33550, type: 12, values: [res, res, 0] }, // ModelPixelScale
    { tag: 33922, type: 12, values: [0, 0, 0, originX, originY, 0] }, // ModelTiepoint
    { tag: 34735, type: 3, values: geoKeys }, // GeoKeyDirectory
    { tag: 42113, type: 2, values: 'nan\0' }, // GDAL_NODATA
  ];
  tags.sort((a, b) => a.tag - b.tag);

  // layout: 8-byte header | IFD | overflow values | pixel data
  const ifdOffset = 8;
  const ifdSize = 2 + tags.length * 12 + 4;
  let overflowOffset = ifdOffset + ifdSize;
  const overflows: Array<{ offset: number; entry: TagEntry }> = [];
  for (const t of tags) {
    const count = typeof t.values === 'string' ? t.values.length : t.values.length;
    const bytes = count * TYPE_SIZES[t.type];
    if (bytes > 4) {
      if (overflowOffset % 2 === 1) overflowOffset += 1;
      overflows.push({ offset: overflowOffset, entry: t });
      overflowOffset += bytes;
    }
  }
  let dataOffset = overflowOffset;
  if (dataOffset % 4 !== 0) dataOffset += 4 - (dataOffset % 4);

  const total = dataOffset + stripByteCount;
  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // header
  view.setUint8(0, 0x49); view.setUint8(1, 0x49); // 'II' little-endian
  view.setUint16(2, 42, true);
  view.setUint32(4, ifdOffset, true);

  // patch strip offset now that layout is known
  const stripTag = tags.find((t) => t.tag === 273)!;
  stripTag.values = [dataOffset];

  const writeValues = (t: TagEntry, at: number) => {
    if (typeof t.values === 'string') {
      for (let i = 0; i < t.values.length; i++) u8[at + i] = t.values.charCodeAt(i);
      return;
    }
    for (let i = 0; i < t.values.length; i++) {
      if (t.type === 3) view.setUint16(at + i * 2, t.values[i], true);
      else if (t.type === 4) view.setUint32(at + i * 4, t.values[i], true);
      else view.setFloat64(at + i * 8, t.values[i], true);
    }
  };

  // IFD
  view.setUint16(ifdOffset, tags.length, true);
  let entryAt = ifdOffset + 2;
  for (const t of tags) {
    const count = typeof t.values === 'string' ? t.values.length : t.values.length;
    const bytes = count * TYPE_SIZES[t.type];
    view.setUint16(entryAt, t.tag, true);
    view.setUint16(entryAt + 2, t.type, true);
    view.setUint32(entryAt + 4, count, true);
    if (bytes <= 4) {
      writeValues(t, entryAt + 8);
    } else {
      const ov = overflows.find((o) => o.entry === t)!;
      view.setUint32(entryAt + 8, ov.offset, true);
      writeValues(t, ov.offset);
    }
    entryAt += 12;
  }
  view.setUint32(entryAt, 0, true); // next IFD = none

  // pixel data (little-endian float32)
  const out = new Float32Array(buf, dataOffset, width * height);
  out.set(grid.data);

  return buf;
}
