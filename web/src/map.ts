/**
 * Bounding-box picker: a MapLibre map with an OSM basemap and a drag-to-draw
 * rectangle tool. Emits the bbox in WGS84 (minLon, minLat, maxLon, maxLat).
 */

import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { FeatureCollection } from 'geojson';

export type LonLatBbox = [number, number, number, number];

export class BboxPicker {
  readonly map: maplibregl.Map;
  private drawing = false;
  private armed = false;
  private start: [number, number] | null = null;
  private bbox: LonLatBbox | null = null;
  private onChange: (bbox: LonLatBbox) => void;

  constructor(container: HTMLElement, onChange: (bbox: LonLatBbox) => void) {
    this.onChange = onChange;
    this.map = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [-119.35, 39.35], // Carson River, NV — the repo's tutorial area
      zoom: 9,
    });

    this.map.on('load', () => {
      this.map.addSource('bbox', { type: 'geojson', data: this.bboxGeojson() });
      this.map.addLayer({
        id: 'bbox-fill',
        type: 'fill',
        source: 'bbox',
        paint: { 'fill-color': '#ff4d6a', 'fill-opacity': 0.12 },
      });
      this.map.addLayer({
        id: 'bbox-line',
        type: 'line',
        source: 'bbox',
        paint: { 'line-color': '#ff4d6a', 'line-width': 2 },
      });
    });

    this.map.on('mousedown', (e) => {
      if (!this.armed) return;
      e.preventDefault();
      this.drawing = true;
      this.start = [e.lngLat.lng, e.lngLat.lat];
    });
    this.map.on('mousemove', (e) => {
      if (!this.drawing || !this.start) return;
      this.setBboxInternal(this.start, [e.lngLat.lng, e.lngLat.lat]);
    });
    this.map.on('mouseup', (e) => {
      if (!this.drawing || !this.start) return;
      this.drawing = false;
      this.armed = false;
      this.map.getCanvas().style.cursor = '';
      this.setBboxInternal(this.start, [e.lngLat.lng, e.lngLat.lat]);
      this.start = null;
      if (this.bbox) this.onChange(this.bbox);
    });
  }

  /** Next drag on the map draws a new rectangle. */
  armDraw(): void {
    this.armed = true;
    this.map.getCanvas().style.cursor = 'crosshair';
  }

  setBbox(bbox: LonLatBbox): void {
    this.bbox = bbox;
    this.refresh();
  }

  getBbox(): LonLatBbox | null {
    return this.bbox;
  }

  private setBboxInternal(a: [number, number], b: [number, number]): void {
    this.bbox = [
      Math.min(a[0], b[0]),
      Math.min(a[1], b[1]),
      Math.max(a[0], b[0]),
      Math.max(a[1], b[1]),
    ];
    this.refresh();
  }

  private refresh(): void {
    const src = this.map.getSource('bbox') as maplibregl.GeoJSONSource | undefined;
    src?.setData(this.bboxGeojson());
  }

  private bboxGeojson(): FeatureCollection {
    if (!this.bbox) return { type: 'FeatureCollection', features: [] };
    const [w, s, e, n] = this.bbox;
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'Polygon',
            coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
          },
        },
      ],
    };
  }
}
