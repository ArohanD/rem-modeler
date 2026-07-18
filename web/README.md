# REM Modeler — browser proof of concept

A fully client-side port of the Python REM pipeline: pick an arbitrary
bounding box on a map, stream the DEM straight from USGS cloud storage,
edit the river centerline in the browser, and generate/export a Relative
Elevation Model — no backend, no GDAL install.

```
cd web
npm install
npm run dev      # http://localhost:5173
```

## Workflow (mirrors src/ in the repo root)

| Python step | Browser equivalent |
|---|---|
| `merge_tifs` (GDAL VRT + Translate) | not needed — window/overview reads from COGs (`src/cog.ts`) |
| `interactive_min_max` | hover readout + river range inputs, live blue highlight |
| `interactive_hillshade` | hillshade sliders (`src/render.ts`, port of `compute_hillshade`) |
| `query_osm_waterways` | same Overpass query from the browser (`src/osm.ts`) |
| `snap_centerline_to_channel` | `src/centerline.ts` — densify → per-point window argmin → smooth |
| *(new)* | full in-browser centerline editing: drag / insert / delete vertices, or draw from scratch |
| `RBFInterpolator(thin_plate_spline, smoothing=1)` | `src/tps.ts` in a Web Worker, evaluated on a coarse grid and bilinearly upsampled (same shape as the Python code) |
| REM = DEM − surface, save GeoTIFF | typed-array subtraction + minimal GeoTIFF writer (`src/tiffwrite.ts`) |

## Data sources (step 1 dropdown)

- **USGS 1 m COG tiles (TNM discovery)** — queries the TNM catalog for 1 m
  DEM tiles intersecting the bbox, then range-reads only the needed windows
  at the overview level matching the working resolution. The `prd-tnm`
  bucket serves CORS (`Access-Control-Allow-Origin: *`) and the tiles are
  true COGs (512-px tiles, 6-level overview pyramids), verified directly.
- **USGS 3DEP ImageServer** — one `exportImage` request returning a float32
  GeoTIFF mosaic sized to the working grid; covers the whole US where 1 m
  lidar is absent.
- **Direct COG URL(s)** — paste tile URLs (e.g. from
  `data/usgs_standard_export/rasters_USGS1m/Original_USGS1mTiles_URLs.txt`).
- **Local GeoTIFF** — drag in any projected DEM; oversized files are
  downsampled to the memory budget.

## Memory & size caps (why the browser won't crash)

Everything hinges on the working grid. Peak cost is ~21 bytes/pixel
(DEM + water surface + REM float32 layers, hillshade, and two RGBA
buffers), budgeted at **24 B/px** for slack. `src/caps.ts` derives a
budget from `navigator.deviceMemory` where available:

| device memory | working budget | pixel cap | max grid |
|---|---|---|---|
| ≥ 8 GB | 512 MB | ~22 Mpx | 4096 × 4096 |
| ≥ 4 GB | 384 MB | ~16 Mpx | 4096 × 4096 |
| unknown (Firefox/Safari) | 320 MB | ~13 Mpx | 3584² |
| ≤ 2 GB | 192 MB | ~8 Mpx | 2560² |

The bbox then *degrades resolution instead of growing memory*: the working
resolution is the smallest power-of-two meter value (1, 2, 4, … 32 — chosen
to align with COG overview levels, so coarser never costs more bandwidth)
that fits the grid inside both the per-dimension and total-pixel caps.
Independently:

- **Hard bbox cap: 40 km** on the longest side — beyond that even 4096 px
  means > 10 m/px and "height above the river" loses meaning for 1 m lidar.
- **Soft warning at 15 km** — the UI shows the resolution penalty before
  loading, in a live readout (grid size, m/px, estimated MB, budget).
- **Interpolation cap: 1200 stations** — the O(n³) thin-plate solve stays
  ≲ 1 s; station spacing widens automatically with a log message.
- **TPS evaluation grid ≤ 512 px** per side (like the Python code's 10×
  coarsening), bilinearly upsampled.

Every allowed bbox is provably within budget — `tests/caps.test.ts` sweeps
1–40 km and asserts `pixels ≤ maxPixels` and `estimatedMB ≤ budgetMB`, and
the Playwright test re-checks it in a real browser.

Full-resolution export of large areas is deliberately out of scope for the
PoC; the production path would stream 512-px COG blocks through the
pipeline and write the output incrementally (File System Access API), which
turns the cap from "resolution" into "time".

## Tests

```
npm test                 # unit + numeric pipeline on a synthetic valley
COG_LIVE=1 npm test      # + live mosaic from real USGS tiles on prd-tnm S3
npm run build
npm run test:e2e         # Playwright: full UI flow, worker, export, caps
```

The synthetic-valley test (`tests/pipeline.test.ts`) carves a meandering
channel with a known water surface, offsets the centerline 30 m, and
asserts snap recovers the channel and the REM is ≈ 0 along it and positive
on the walls. The e2e test drives the same scenario through the real UI
(file load → snap button → worker → GeoTIFF export) and fails on any page
error.

If Playwright's downloaded browser build is unavailable (sandboxes/CI),
point at a pre-installed one: `PW_CHROMIUM_PATH=/path/to/chrome npm run test:e2e`.

## Verified vs. assumed

Verified in this environment:
- `prd-tnm` S3: CORS enabled, byte ranges, true COG layout (read a real
  tile header), live two-tile mosaic (`tests/cog-live.test.ts`).
- Full pipeline numerics + UI flow (unit + e2e, all green).

Assumed (blocked by this sandbox's egress policy, verify on first real
deploy):
- CORS headers on `tnmaccess.nationalmap.gov` (TNM discovery) and
  `elevation.nationalmap.gov` (ImageServer). Overpass CORS is proven by
  overpass-turbo. If TNM turns out to be CORS-blocked, the fallbacks
  (ImageServer / direct URLs / local file) and a precomputed static tile
  index are the mitigation paths.

## Known PoC limitations

- Single UTM zone per load (tiles in other zones are skipped with a log).
- Overlapping tiles resolve first-wins, not newest-first like GDAL VRT.
- No skeletonization ("brush" mode) — draw + snap covers the manual path.
- Hillshade recompute is main-thread (fine ≤ 16 Mpx; would move to a worker
  or shader next).
