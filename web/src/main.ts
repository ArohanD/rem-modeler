/**
 * App orchestration: bbox selection -> DEM load (capped) -> explore ->
 * centerline (OSM / manual, edit, snap) -> water surface -> REM -> export.
 */

import './style.css';
import { BboxPicker, type LonLatBbox } from './map';
import { capsProfile, planGrid, type GridPlan } from './caps';
import { projectBbox, utmZoneForLon, epsgForUtmZone, transformer } from './proj';
import { discoverOneMeterTiles, fetchImageServerTiff, MAX_TILES } from './discovery';
import { emptyGrid, mosaicFromUrls, probeEpsg, gridFromArrayBuffer } from './cog';
import { gridStats, upsampleBilinear, type DemGrid } from './grid';
import { computeHillshade, renderExplore, renderRem } from './render';
import { Workbench } from './workbench';
import {
  snapCenterlineToChannel,
  sampleElevations,
  effectiveSpacing,
  lineLength,
  type XY,
} from './centerline';
import { fetchWaterways, type Waterway } from './osm';
import { writeGeoTiff } from './tiffwrite';
import type { TpsRequest, TpsResponse } from './tps.worker';
import TpsWorker from './tps.worker?worker';

const MAX_TPS_POINTS = 1200;
const COARSE_MAX_DIM = 512;

// ---------------------------------------------------------------------------
// state

const caps = capsProfile((navigator as { deviceMemory?: number }).deviceMemory ?? null);

interface AppState {
  bbox: LonLatBbox | null;
  plan: GridPlan | null;
  grid: DemGrid | null;
  hillshade: Uint8Array | null;
  rem: Float32Array | null;
  waterSurface: Float32Array | null;
  centerlinePreSnap: XY[] | null;
  waterways: Array<Waterway & { projected: XY[] }>;
  view: 'dem' | 'rem';
  hsAlpha: number;
  hsZ: number;
  hsAlt: number;
  rangeLo: number | null;
  rangeHi: number | null;
  remVmax: number;
  stretchLo: number;
  stretchHi: number;
}

const state: AppState = {
  bbox: null,
  plan: null,
  grid: null,
  hillshade: null,
  rem: null,
  waterSurface: null,
  centerlinePreSnap: null,
  waterways: [],
  view: 'dem',
  hsAlpha: 0.6,
  hsZ: 3,
  hsAlt: 45,
  rangeLo: null,
  rangeHi: null,
  remVmax: 5,
  stretchLo: 0,
  stretchHi: 1,
};

// ---------------------------------------------------------------------------
// dom helpers

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const logEl = $('#log'.slice(1));
function log(msg: string, isError = false): void {
  const div = document.createElement('div');
  div.textContent = msg;
  if (isError) div.className = 'err';
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  if (isError) console.error(msg);
  else console.log(msg);
}

const progressEl = $('progress');
const progressBar = $('progress-bar');
const progressText = $('progress-text');
function showProgress(text: string, frac: number | null): void {
  progressEl.hidden = false;
  progressText.textContent = text;
  progressBar.style.width = `${Math.round((frac ?? 0) * 100)}%`;
}
function hideProgress(): void {
  progressEl.hidden = true;
}

function setStepEnabled(id: string, enabled: boolean): void {
  $(id).classList.toggle('disabled', !enabled);
}

// ---------------------------------------------------------------------------
// workbench + map

const hoverReadout = $('hover-readout');
const workbenchCanvas = $('workbench') as HTMLCanvasElement;
const workbench = new Workbench(workbenchCanvas, {
  onHover: (x, y, elev) => {
    hoverReadout.hidden = false;
    hoverReadout.textContent = Number.isNaN(elev)
      ? `${x.toFixed(0)}, ${y.toFixed(0)}  (no data)`
      : `${x.toFixed(0)}, ${y.toFixed(0)}  ${elev.toFixed(2)} m`;
  },
  onCenterlineChange: (coords) => {
    setStepEnabled('step-rem', coords.length >= 2 && state.grid !== null);
  },
  onModeChange: (mode) => {
    $('btn-draw-line').classList.toggle('active', mode === 'draw');
  },
});

const picker = new BboxPicker($('map-container'), (bbox) => {
  state.bbox = bbox;
  syncBboxInputs();
  updatePlanReadout();
});

function showWorkbench(): void {
  $('map-container').style.display = 'none';
  workbenchCanvas.hidden = false;
  ($('btn-back-map') as HTMLButtonElement).hidden = false;
  workbench.fitView();
}
function showMap(): void {
  $('map-container').style.display = '';
  workbenchCanvas.hidden = true;
  hoverReadout.hidden = true;
  ($('btn-back-map') as HTMLButtonElement).hidden = true;
  picker.map.resize();
}
$('btn-back-map').addEventListener('click', showMap);

// ---------------------------------------------------------------------------
// step 1: bbox + plan + load

function syncBboxInputs(): void {
  if (!state.bbox) return;
  const [w, s, e, n] = state.bbox;
  ($('bbox-w') as HTMLInputElement).value = w.toFixed(4);
  ($('bbox-s') as HTMLInputElement).value = s.toFixed(4);
  ($('bbox-e') as HTMLInputElement).value = e.toFixed(4);
  ($('bbox-n') as HTMLInputElement).value = n.toFixed(4);
}

for (const id of ['bbox-w', 'bbox-s', 'bbox-e', 'bbox-n']) {
  $(id).addEventListener('change', () => {
    const w = parseFloat(($('bbox-w') as HTMLInputElement).value);
    const s = parseFloat(($('bbox-s') as HTMLInputElement).value);
    const e = parseFloat(($('bbox-e') as HTMLInputElement).value);
    const n = parseFloat(($('bbox-n') as HTMLInputElement).value);
    if ([w, s, e, n].some(Number.isNaN) || w >= e || s >= n) return;
    state.bbox = [w, s, e, n];
    picker.setBbox(state.bbox);
    updatePlanReadout();
  });
}

$('btn-draw-bbox').addEventListener('click', () => picker.armDraw());

function currentEpsg(): number {
  if (!state.bbox) return epsgForUtmZone(11);
  return epsgForUtmZone(utmZoneForLon((state.bbox[0] + state.bbox[2]) / 2));
}

function planForBbox(epsg: number): GridPlan | null {
  if (!state.bbox) return null;
  const proj = projectBbox(
    { minX: state.bbox[0], minY: state.bbox[1], maxX: state.bbox[2], maxY: state.bbox[3] },
    epsg,
  );
  return planGrid(proj.maxX - proj.minX, proj.maxY - proj.minY, caps);
}

function updatePlanReadout(): void {
  const el = $('plan-readout');
  if (!state.bbox) {
    el.textContent = 'No area selected yet.';
    return;
  }
  const plan = planForBbox(currentEpsg());
  state.plan = plan;
  if (!plan) return;
  if (plan.rejected) {
    el.innerHTML = `<span class="err">${plan.warnings.join(' ')}</span>`;
    return;
  }
  const lines = [
    `Working grid: ${plan.width} × ${plan.height} px @ ${plan.res} m/px`,
    `Estimated working set: ~${plan.estimatedMB.toFixed(0)} MB ` +
      `(budget ${caps.budgetMB} MB${caps.deviceMemoryGB ? `, device reports ${caps.deviceMemoryGB} GB RAM` : ''})`,
  ];
  el.innerHTML =
    lines.join('\n') +
    (plan.warnings.length ? `\n<span class="warn">${plan.warnings.join('\n')}</span>` : '');
}

($('source-select') as HTMLSelectElement).addEventListener('change', () => {
  const v = ($('source-select') as HTMLSelectElement).value;
  ($('source-urls-row') as HTMLElement).hidden = v !== 'urls';
  ($('source-file-row') as HTMLElement).hidden = v !== 'file';
});

$('btn-load').addEventListener('click', () => {
  loadDem().catch((e) => {
    hideProgress();
    log(`Load failed: ${e instanceof Error ? e.message : e}`, true);
  });
});

async function loadDem(): Promise<void> {
  const source = ($('source-select') as HTMLSelectElement).value;

  if (source === 'file') {
    const input = $('source-file') as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) throw new Error('Choose a GeoTIFF file first.');
    showProgress('Reading local GeoTIFF…', 0.2);
    const grid = await capGrid(await gridFromArrayBuffer(await file.arrayBuffer()));
    finishLoad(grid);
    return;
  }

  if (!state.bbox) throw new Error('Draw a bounding box first.');

  if (source === 'imageserver') {
    const epsg = currentEpsg();
    const plan = planForBbox(epsg);
    if (!plan || plan.rejected) throw new Error(plan?.warnings.join(' ') ?? 'Invalid selection.');
    const proj = projectBbox(
      { minX: state.bbox[0], minY: state.bbox[1], maxX: state.bbox[2], maxY: state.bbox[3] },
      epsg,
    );
    showProgress('Requesting mosaic from 3DEP ImageServer…', 0.2);
    const buf = await fetchImageServerTiff(
      [proj.minX, proj.minY, proj.maxX, proj.maxY],
      epsg,
      plan.width,
      plan.height,
    );
    showProgress('Decoding…', 0.8);
    const grid = await gridFromArrayBuffer(buf);
    if (!grid.epsg) grid.epsg = epsg;
    if (!grid.res || !Number.isFinite(grid.originX)) {
      // some ArcGIS deployments strip georeferencing from f=image output
      grid.res = plan.res;
      grid.originX = proj.minX;
      grid.originY = proj.maxY;
    }
    finishLoad(grid);
    return;
  }

  // COG paths: tnm discovery or user-provided URLs
  let urls: string[];
  if (source === 'urls') {
    urls = ($('source-urls') as HTMLTextAreaElement).value
      .split('\n')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (urls.length === 0) throw new Error('Paste at least one COG URL.');
  } else {
    showProgress('Querying USGS TNM catalog…', 0.05);
    const found = await discoverOneMeterTiles(state.bbox);
    if (found.urls.length === 0) {
      throw new Error(
        'No USGS 1 m tiles found for this area. Try the 3DEP ImageServer source instead.',
      );
    }
    if (found.truncated) {
      log(`Area intersects ${found.total} tiles; loading the first ${MAX_TILES}.`, true);
    }
    urls = found.urls;
    log(`TNM: ${urls.length} tile(s) intersect the area.`);
  }

  showProgress('Probing tile CRS…', 0.1);
  const tileEpsg = (await probeEpsg(urls[0])) ?? currentEpsg();
  const plan = planForBbox(tileEpsg);
  if (!plan || plan.rejected) throw new Error(plan?.warnings.join(' ') ?? 'Invalid selection.');
  state.plan = plan;

  const proj = projectBbox(
    { minX: state.bbox[0], minY: state.bbox[1], maxX: state.bbox[2], maxY: state.bbox[3] },
    tileEpsg,
  );
  const target = emptyGrid(proj.minX, proj.maxY, plan.res, plan.width, plan.height, tileEpsg);

  const reports = await mosaicFromUrls(urls, target, (done, total, note) =>
    showProgress(note || `Reading tiles ${done}/${total}`, 0.1 + (0.85 * done) / total),
  );
  for (const r of reports) {
    if (r.status !== 'ok') log(`${r.status}: ${r.url.split('/').pop()} ${r.detail ?? ''}`, r.status === 'error');
  }
  if (reports.every((r) => r.status !== 'ok')) throw new Error('No tile could be read.');
  finishLoad(target);
}

/** Downsample an oversized local grid to fit the pixel budget. */
async function capGrid(grid: DemGrid): Promise<DemGrid> {
  const px = grid.width * grid.height;
  if (px <= caps.maxPixels) return grid;
  const factor = Math.ceil(Math.sqrt(px / caps.maxPixels));
  log(`Local file is ${grid.width}×${grid.height}; downsampling ×${factor} to fit the memory budget.`);
  const w = Math.floor(grid.width / factor);
  const h = Math.floor(grid.height / factor);
  const data = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      data[r * w + c] = grid.data[r * factor * grid.width + c * factor];
    }
  }
  return { data, width: w, height: h, res: grid.res * factor, originX: grid.originX, originY: grid.originY, epsg: grid.epsg };
}

function finishLoad(grid: DemGrid): void {
  state.grid = grid;
  state.rem = null;
  state.waterSurface = null;
  state.view = 'dem';
  const stats = gridStats(grid);
  if (stats.validFraction === 0) {
    hideProgress();
    throw new Error('Loaded grid contains no valid elevation data for this area.');
  }
  state.stretchLo = stats.p2;
  state.stretchHi = stats.p98;
  log(
    `DEM loaded: ${grid.width}×${grid.height} @ ${grid.res} m/px, EPSG:${grid.epsg}, ` +
      `elev ${stats.min.toFixed(1)}–${stats.max.toFixed(1)} m, ${(stats.validFraction * 100).toFixed(0)}% valid`,
  );
  showProgress('Computing hillshade…', 0.95);
  // let the progress paint before the synchronous hillshade pass
  setTimeout(() => {
    state.hillshade = computeHillshade(grid, 315, state.hsAlt, state.hsZ);
    workbench.setGrid(grid);
    redraw();
    hideProgress();
    showWorkbench();
    setStepEnabled('step-explore', true);
    setStepEnabled('step-centerline', true);
    setStepEnabled('step-rem', workbench.centerline.length >= 2);
  }, 30);
}

// ---------------------------------------------------------------------------
// rendering

let rgba: Uint8ClampedArray<ArrayBuffer> | null = null;

function redraw(): void {
  const grid = state.grid;
  if (!grid) return;
  const n = grid.width * grid.height * 4;
  if (!rgba || rgba.length !== n) rgba = new Uint8ClampedArray(n);
  if (state.view === 'rem' && state.rem) {
    renderRem(state.rem, { vmax: state.remVmax, hillshade: state.hillshade, hillshadeAlpha: state.hsAlpha }, rgba);
  } else {
    renderExplore(
      grid,
      {
        elevMin: state.stretchLo,
        elevMax: state.stretchHi,
        rangeLo: state.rangeLo,
        rangeHi: state.rangeHi,
        hillshadeAlpha: state.hsAlpha,
        hillshade: state.hillshade,
      },
      rgba,
    );
  }
  workbench.setImage(rgba);
}

// step 2 controls
$('range-lo').addEventListener('change', () => {
  state.rangeLo = parseFloatOrNull(($('range-lo') as HTMLInputElement).value);
  redraw();
});
$('range-hi').addEventListener('change', () => {
  state.rangeHi = parseFloatOrNull(($('range-hi') as HTMLInputElement).value);
  redraw();
});
$('hs-alpha').addEventListener('input', () => {
  state.hsAlpha = parseFloat(($('hs-alpha') as HTMLInputElement).value);
  redraw();
});
for (const [id, key] of [['hs-z', 'hsZ'], ['hs-alt', 'hsAlt']] as const) {
  $(id).addEventListener('change', () => {
    state[key] = parseFloat(($(id) as HTMLInputElement).value);
    if (!state.grid) return;
    showProgress('Recomputing hillshade…', 0.5);
    setTimeout(() => {
      state.hillshade = computeHillshade(state.grid!, 315, state.hsAlt, state.hsZ);
      redraw();
      hideProgress();
    }, 30);
  });
}

$('btn-range-auto').addEventListener('click', () => {
  if (!state.grid || workbench.centerline.length < 2) {
    log('Auto range needs a centerline (step 3) to sample along.', true);
    return;
  }
  const samples = sampleElevations(state.grid, workbench.centerline, Math.max(state.grid.res * 2, 10));
  if (samples.length < 3) return;
  const zs = samples.map((s) => s[2]).sort((a, b) => a - b);
  const lo = zs[Math.floor(zs.length * 0.02)];
  const hi = zs[Math.floor(zs.length * 0.98)];
  const margin = Math.max(0.5, (hi - lo) * 0.05);
  state.rangeLo = Math.floor((lo - margin) * 10) / 10;
  state.rangeHi = Math.ceil((hi + margin) * 10) / 10;
  ($('range-lo') as HTMLInputElement).value = String(state.rangeLo);
  ($('range-hi') as HTMLInputElement).value = String(state.rangeHi);
  redraw();
});

function parseFloatOrNull(s: string): number | null {
  const v = parseFloat(s);
  return Number.isNaN(v) ? null : v;
}

// ---------------------------------------------------------------------------
// step 3: centerline

$('btn-draw-line').addEventListener('click', () => {
  workbench.setMode(workbench.mode === 'draw' ? 'edit' : 'draw');
});

$('btn-osm').addEventListener('click', () => {
  fetchOsmCenterlines().catch((e) => {
    hideProgress();
    log(`OSM fetch failed: ${e instanceof Error ? e.message : e}. You can draw the centerline manually.`, true);
  });
});

async function fetchOsmCenterlines(): Promise<void> {
  if (!state.bbox || !state.grid) throw new Error('Load a DEM first.');
  showProgress('Querying OpenStreetMap (Overpass)…', 0.3);
  const ways = await fetchWaterways(state.bbox);
  hideProgress();
  if (ways.length === 0) {
    log('No OSM waterways found in this area — draw the centerline manually.', true);
    return;
  }
  const toGrid = transformer(4326, state.grid.epsg);
  state.waterways = ways
    .map((w) => ({ ...w, projected: w.coords.map((c) => toGrid(c)) as XY[] }))
    .sort((a, b) => lineLength(b.projected) - lineLength(a.projected));

  const sel = $('osm-select') as HTMLSelectElement;
  sel.innerHTML = '';
  state.waterways.forEach((w, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `${w.name} (${(lineLength(w.projected) / 1000).toFixed(1)} km)`;
    sel.appendChild(opt);
  });
  ($('osm-pick-row') as HTMLElement).hidden = false;
  log(`OSM: ${ways.length} waterway(s) found.`);
  applyWaterway(0);
}

function applyWaterway(index: number): void {
  const w = state.waterways[index];
  if (!w) return;
  workbench.auxLines = state.waterways
    .filter((_, i) => i !== index)
    .map((o) => ({ coords: o.projected, highlight: false }));
  workbench.setCenterline(w.projected);
  workbench.setMode('edit');
  state.centerlinePreSnap = null;
}

($('osm-select') as HTMLSelectElement).addEventListener('change', (e) => {
  applyWaterway(parseInt((e.target as HTMLSelectElement).value, 10));
});

$('btn-snap').addEventListener('click', () => {
  if (!state.grid || workbench.centerline.length < 2) return;
  const spacing = parseFloat(($('snap-spacing') as HTMLInputElement).value) || 25;
  const radius = parseFloat(($('snap-radius') as HTMLInputElement).value) || 100;
  const range: [number, number] | null =
    state.rangeLo != null && state.rangeHi != null ? [state.rangeLo, state.rangeHi] : null;
  state.centerlinePreSnap = workbench.centerline.map((c) => [c[0], c[1]]);
  showProgress('Snapping centerline to channel…', 0.5);
  setTimeout(() => {
    const snapped = snapCenterlineToChannel(state.grid!, state.centerlinePreSnap!, {
      searchRadius: radius,
      pointSpacing: spacing,
      elevRange: range,
    });
    workbench.setCenterline(snapped);
    hideProgress();
    log(`Snapped centerline: ${snapped.length} points${range ? ' (constrained to river range)' : ''}.`);
  }, 30);
});

$('btn-unsnap').addEventListener('click', () => {
  if (state.centerlinePreSnap) {
    workbench.setCenterline(state.centerlinePreSnap);
    state.centerlinePreSnap = null;
  }
});

// ---------------------------------------------------------------------------
// step 4: REM

$('btn-rem').addEventListener('click', () => {
  generateRem().catch((e) => {
    hideProgress();
    log(`REM generation failed: ${e instanceof Error ? e.message : e}`, true);
  });
});

function generateRem(): Promise<void> {
  const grid = state.grid;
  if (!grid || workbench.centerline.length < 2) return Promise.resolve();

  const requestedSpacing = parseFloat(($('snap-spacing') as HTMLInputElement).value) || 25;
  const spacing = effectiveSpacing(workbench.centerline, requestedSpacing, MAX_TPS_POINTS);
  if (spacing > requestedSpacing) {
    log(`Station spacing widened to ${spacing.toFixed(0)} m to cap the interpolation at ${MAX_TPS_POINTS} points.`);
  }
  const points = sampleElevations(grid, workbench.centerline, spacing);
  if (points.length < 3) return Promise.reject(new Error('Too few valid elevation samples along the centerline.'));
  log(`Interpolating water surface from ${points.length} stations…`);

  const factor = Math.max(1, Math.ceil(Math.max(grid.width, grid.height) / COARSE_MAX_DIM));
  const cw = Math.max(2, Math.ceil(grid.width / factor));
  const ch = Math.max(2, Math.ceil(grid.height / factor));
  const req: TpsRequest = {
    points,
    originX: grid.originX,
    originY: grid.originY,
    res: grid.res * factor,
    width: cw,
    height: ch,
    smoothing: 1.0,
  };

  return new Promise<void>((resolve, reject) => {
    const worker = new TpsWorker();
    worker.onmessage = (ev: MessageEvent<TpsResponse>) => {
      const msg = ev.data;
      if (msg.kind === 'progress') {
        showProgress('Interpolating water surface…', msg.frac);
      } else if (msg.kind === 'done') {
        worker.terminate();
        showProgress('Building REM…', 1);
        setTimeout(() => {
          const ws = upsampleBilinear(msg.surface, msg.width, msg.height, grid.width, grid.height);
          const rem = new Float32Array(grid.width * grid.height);
          for (let i = 0; i < rem.length; i++) {
            const d = grid.data[i];
            rem[i] = Number.isNaN(d) ? NaN : d - ws[i];
          }
          state.waterSurface = ws;
          state.rem = rem;
          state.view = 'rem';
          redraw();
          hideProgress();
          let lo = Infinity, hi = -Infinity;
          for (let i = 0; i < rem.length; i += 7) {
            const v = rem[i];
            if (!Number.isNaN(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
          }
          log(`REM ready. Range ${lo.toFixed(1)} to ${hi.toFixed(1)} m relative to water surface.`);
          resolve();
        }, 30);
      } else {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.postMessage(req);
  });
}

$('rem-vmax').addEventListener('input', () => {
  state.remVmax = parseFloat(($('rem-vmax') as HTMLInputElement).value);
  $('vmax-label').textContent = String(state.remVmax);
  if (state.view === 'rem') redraw();
});

$('btn-show-dem').addEventListener('click', () => {
  state.view = state.view === 'rem' ? 'dem' : 'rem';
  $('btn-show-dem').textContent = state.view === 'rem' ? 'Show DEM' : 'Show REM';
  redraw();
});

function download(blob: Blob, filename: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$('btn-export-tif').addEventListener('click', () => {
  if (!state.grid || !state.rem) return;
  const buf = writeGeoTiff({ ...state.grid, data: state.rem });
  download(new Blob([buf], { type: 'image/tiff' }), 'rem.tif');
  log(`Exported rem.tif (${(buf.byteLength / 1024 / 1024).toFixed(1)} MB, EPSG:${state.grid.epsg}).`);
});

$('btn-export-png').addEventListener('click', () => {
  if (!state.grid || !rgba) return;
  const c = document.createElement('canvas');
  c.width = state.grid.width;
  c.height = state.grid.height;
  c.getContext('2d')!.putImageData(new ImageData(rgba, c.width, c.height), 0, 0);
  c.toBlob((b) => b && download(b, state.view === 'rem' ? 'rem.png' : 'dem.png'), 'image/png');
});

// ---------------------------------------------------------------------------
// boot

updatePlanReadout();
log(
  `Memory budget: ${caps.budgetMB} MB working set → max grid ${caps.maxDim}×${caps.maxDim} ` +
    `(${(caps.maxPixels / 1e6).toFixed(0)} Mpx).`,
);

// debug/e2e hooks
declare global {
  interface Window {
    __poc: {
      state: AppState;
      caps: typeof caps;
      loadDemBuffer: (buf: ArrayBuffer) => Promise<void>;
      setBbox: (bbox: LonLatBbox) => void;
      setCenterline: (coords: XY[]) => void;
      getCenterline: () => XY[];
      snap: () => void;
      generateRem: () => Promise<void>;
      exportTiff: () => ArrayBuffer | null;
    };
  }
}

window.__poc = {
  state,
  caps,
  loadDemBuffer: async (buf: ArrayBuffer) => {
    const grid = await capGrid(await gridFromArrayBuffer(buf));
    finishLoad(grid);
    await new Promise((r) => setTimeout(r, 120)); // wait out the hillshade timeout
  },
  setBbox: (bbox) => {
    state.bbox = bbox;
    picker.setBbox(bbox);
    syncBboxInputs();
    updatePlanReadout();
  },
  setCenterline: (coords) => {
    workbench.setCenterline(coords);
    workbench.setMode('edit');
  },
  getCenterline: () => workbench.centerline,
  snap: () => $('btn-snap').click(),
  generateRem,
  exportTiff: () => (state.grid && state.rem ? writeGeoTiff({ ...state.grid, data: state.rem }) : null),
};
