'use strict';

/* ==========================================================================
   Aftershock Forecast Studio — prototype dashboard behaviour
   Transparent ETAS-inspired baseline, rendered entirely in the browser.
   ========================================================================== */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const REGIONS = {
  japan: {
    name: 'Japan Trench',
    latLabel: '38.2°N',
    lonLabel: '142.1°E',
    angle: 17,
    faultLength: 92,
    seed: 1847,
    setting: 'schematic subduction fault, 17°',
    preset: { magnitude: 6.4, depth: 28, elapsed: 6, window: '7d' }
  },
  chile: {
    name: 'Chile Margin',
    latLabel: '33.0°S',
    lonLabel: '72.0°W',
    angle: -27,
    faultLength: 78,
    seed: 2711,
    setting: 'schematic subduction fault, −27°',
    preset: { magnitude: 7.2, depth: 35, elapsed: 12, window: '7d' }
  },
  california: {
    name: 'California',
    latLabel: '35.8°N',
    lonLabel: '120.5°W',
    angle: -7,
    faultLength: 54,
    seed: 3499,
    setting: 'schematic strike-slip fault, −7°',
    preset: { magnitude: 5.8, depth: 12, elapsed: 2, window: '24h' }
  },
  catalog: {
    name: 'USGS catalog event',
    latLabel: '—',
    lonLabel: '—',
    angle: 0,
    faultLength: 60,
    seed: 991,
    setting: 'catalog-derived spatial kernel',
    preset: { magnitude: 6.5, depth: 20, elapsed: 6, window: '7d' }
  }
};

const USGS_FEEDS = {
  'live-day': {
    label: 'Live · past day',
    url: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv',
    period: 'past day'
  },
  'live-week': {
    label: 'Live · past week',
    url: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv',
    period: 'past week'
  },
  'live-month': {
    label: 'Live · past month',
    url: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv',
    period: 'past month'
  },
  bundled: {
    label: 'Bundled snapshot · past week',
    url: 'assets/usgs-all-week.csv',
    period: 'captured 26 Nov 2026'
  }
};

const NEEDS_STORAGE_KEY = 'earthquake-aftershock-forecast.needs.v1';
const MAX_CATALOG_ROWS = 20000;
const CATALOG_FETCH_MS = 14000;
const ASSET_VERSION = '20261127k';
const MODEL_URL = `assets/ml-model.json?v=${ASSET_VERSION}`;
const COMPARISON_BASE = 'assets/comparison/';

const PRESETS = Object.fromEntries(
  Object.entries(REGIONS).map(([key, region]) => [key, { region: key, ...region.preset }])
);

const WINDOWS = {
  '24h': { hours: 24, days: 1, label: '24-hour', short: '24 h' },
  '7d': { hours: 168, days: 7, label: '7-day', short: '7 days' },
  '30d': { hours: 720, days: 30, label: '30-day', short: '30 days' }
};

const MAP = {
  xMin: -60,
  xMax: 60,
  yMin: -37.5,
  yMax: 37.5,
  cell: 3,
  cols: 40,
  rows: 25
};

const CELL_COUNT = MAP.cols * MAP.rows;
const CELL_AREA = MAP.cell * MAP.cell;
const HIGH_RISK_THRESHOLD = 0.5;

const OMORI_C = 0.05;
const OMORI_P = 1.1;
const OMORI_NORM = omoriIntegral(0, 30);

const SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const BANDS = [
  { min: 0, max: 5, label: '0–5 km' },
  { min: 5, max: 10, label: '5–10 km' },
  { min: 10, max: 20, label: '10–20 km' },
  { min: 20, max: 40, label: '20–40 km' },
  { min: 40, max: Infinity, label: '40+ km' }
];

const HEAT_STOPS = [
  [0, [244, 249, 249]],
  [0.08, [226, 239, 239]],
  [0.2, [192, 220, 220]],
  [0.34, [234, 218, 194]],
  [0.48, [238, 190, 150]],
  [0.62, [220, 145, 95]],
  [0.76, [190, 96, 55]],
  [0.9, [151, 57, 32]],
  [1, [103, 31, 20]]
];

const HEAT_LUT = Array.from({ length: 201 }, (_, index) => heatColorAt(index / 200));

const state = {
  region: 'catalog',
  magnitude: 6.5,
  depth: 20,
  elapsed: 6,
  window: '7d',
  layers: { grid: true, events: true, fault: true },
  engine: 'ml',
  mlModel: null,
  mlStatus: 'Loading local Random Forest…',
  catalog: { rows: [], meta: null },
  mainshockId: null,
  catalogAnalysis: null,
  comparisonMeta: null,
  comparisonSummary: null,
  result: null,
  hover: null,
  forecastTimer: 0,
  drawRaf: 0
};

const dom = {};

let methodTrigger = null;

/* --------------------------------------------------------------- bootstrap --- */

function init() {
  cacheDom();
  bindEvents();
  syncControlValues();
  syncControlLabels();
  updatePresetActive();
  initNeeds();
  performForecast();
  loadCatalog();
  loadMlModel();
  loadComparison();
  updateMethodDialog();
}

function cacheDom() {
  dom.sourcePill = $('#sourcePill');
  dom.headerContext = $('#headerContext');
  dom.catalogFeed = $('#catalogFeed');
  dom.loadCatalog = $('#loadCatalog');
  dom.catalogFile = $('#catalogFile');
  dom.catalogStatus = $('#catalogStatus');
  dom.engineSelect = $('#engineSelect');
  dom.engineName = $('#engineName');
  dom.engineVersion = $('#engineVersion');
  dom.engineMetrics = $('#engineMetrics');
  dom.engineStatus = $('#engineStatus');
  dom.methodValidation = $('#methodValidation');
  dom.mainshockSelect = $('#mainshockSelect');
  dom.mainshockHelp = $('#mainshockHelp');
  dom.dataBanner = $('#dataBanner');
  dom.dataBannerTitle = $('#dataBannerTitle');
  dom.dataBannerSub = $('#dataBannerSub');
  dom.exportObservedCsv = $('#exportObservedCsv');
  dom.evidenceSummary = $('#evidenceSummary');
  dom.observedBody = $('#observedBody');
  dom.evidenceNote = $('#evidenceNote');

  dom.comparisonEvent = $('#comparisonEvent');
  dom.comparisonHelp = $('#comparisonHelp');
  dom.comparisonAiScore = $('#comparisonAiScore');
  dom.comparisonManualScore = $('#comparisonManualScore');
  dom.comparisonAftershocks = $('#comparisonAftershocks');
  dom.comparisonAiImg = $('#comparisonAiImg');
  dom.comparisonManualImg = $('#comparisonManualImg');
  dom.comparisonActualImg = $('#comparisonActualImg');
  dom.comparisonSummary = $('#comparisonSummary');

  dom.needsChecks = $$('#needsChecks input[data-need]');
  dom.needsNotes = $('#needsNotes');
  dom.saveNeeds = $('#saveNeeds');
  dom.exportNeedsCsv = $('#exportNeedsCsv');
  dom.clearNeeds = $('#clearNeeds');
  dom.printCard = $('#printCard');
  dom.needsStatus = $('#needsStatus');

  dom.form = $('#scenarioForm');
  dom.region = $('#region');
  dom.magnitude = $('#magnitude');
  dom.magnitudeOut = $('#magnitudeOut');
  dom.depth = $('#depth');
  dom.elapsed = $('#elapsed');
  dom.windowInputs = $$('input[name="window"]', dom.form);
  dom.coords = $('#coords');
  dom.presetButtons = $$('.preset');

  dom.runButton = $('#runButton');
  dom.runLabel = $('#runLabel');
  dom.scenarioTitle = $('#scenarioTitle');
  dom.scenarioSub = $('#scenarioSub');
  dom.srStatus = $('#srStatus');

  dom.statPeak = $('#statPeak');
  dom.statPeakNote = $('#statPeakNote');
  dom.statExpected = $('#statExpected');
  dom.statExpectedNote = $('#statExpectedNote');
  dom.statHigh = $('#statHigh');
  dom.statHighNote = $('#statHighNote');
  dom.statZone = $('#statZone');
  dom.statZoneNote = $('#statZoneNote');

  dom.priorityBody = $('#priorityBody');
  dom.priorityNote = $('#priorityNote');
  dom.rateBars = $('#rateBars');
  dom.rateMax = $('#rateMax');
  dom.rateNote = $('#rateNote');

  dom.mapWrap = $('#mapWrap');
  dom.mapCanvas = $('#mapCanvas');
  dom.mapTooltip = $('#mapTooltip');
  dom.mapFootText = $('#mapFootText');

  dom.layerGrid = $('#layerGrid');
  dom.layerEvents = $('#layerEvents');
  dom.layerFault = $('#layerFault');

  dom.exportPng = $('#exportPng');
  dom.exportCsv = $('#exportCsv');

  dom.methodDialog = $('#methodDialog');
  dom.methodOpen = $('#methodOpen');
  dom.methodOpenInline = $('#methodOpenInline');
  dom.methodOpenFooter = $('#methodOpenFooter');
}

function bindEvents() {
  dom.form.addEventListener('input', handleControlChange);
  dom.form.addEventListener('change', handleControlChange);
  dom.form.addEventListener('submit', (event) => {
    event.preventDefault();
    window.clearTimeout(state.forecastTimer);
    performForecast();
  });

  dom.catalogFeed.addEventListener('change', () => loadCatalog());
  dom.loadCatalog.addEventListener('click', () => loadCatalog());
  dom.engineSelect.addEventListener('change', () => {
    state.engine = dom.engineSelect.value === 'baseline' ? 'baseline' : 'ml';
    updateEngineBox();
    performForecast();
  });
  dom.catalogFile.addEventListener('change', handleCatalogFile);
  dom.mainshockSelect.addEventListener('change', handleMainshockChange);
  dom.exportObservedCsv.addEventListener('click', exportObservedCsv);
  dom.comparisonEvent.addEventListener('change', updateComparison);

  dom.saveNeeds.addEventListener('click', () => saveNeeds(false));
  dom.exportNeedsCsv.addEventListener('click', exportNeedsCsv);
  dom.clearNeeds.addEventListener('click', clearNeeds);
  dom.printCard.addEventListener('click', () => window.print());

  dom.needsChecks.forEach((input) => {
    input.addEventListener('change', () => saveNeeds(true));
  });
  dom.needsNotes.addEventListener('change', () => saveNeeds(true));

  [dom.depth, dom.elapsed].forEach((input) => {
    input.addEventListener('blur', () => {
      readFormState();
      syncControlValues();
      syncControlLabels();
      updatePresetActive();
      scheduleForecast();
    });
  });

  dom.presetButtons.forEach((button) => {
    button.addEventListener('click', () => applyPreset(button.dataset.preset));
  });

  [dom.layerGrid, dom.layerEvents, dom.layerFault].forEach((input) => {
    input.addEventListener('change', () => {
      state.layers.grid = dom.layerGrid.checked;
      state.layers.events = dom.layerEvents.checked;
      state.layers.fault = dom.layerFault.checked;
      updateMapFoot();
      scheduleDraw();
    });
  });

  dom.mapCanvas.addEventListener('pointermove', handlePointerMove);
  dom.mapCanvas.addEventListener('pointerdown', handlePointerMove);
  dom.mapCanvas.addEventListener('pointerleave', hideTooltip);
  dom.mapWrap.addEventListener('pointerleave', hideTooltip);
  window.addEventListener('scroll', hideTooltip, { passive: true });

  dom.exportPng.addEventListener('click', exportPng);
  dom.exportCsv.addEventListener('click', exportCsv);

  [dom.methodOpen, dom.methodOpenInline, dom.methodOpenFooter].forEach((button) => {
    button.addEventListener('click', openMethod);
  });

  $$('[data-close-dialog]', dom.methodDialog).forEach((button) => {
    button.addEventListener('click', closeMethod);
  });

  dom.methodDialog.addEventListener('click', (event) => {
    if (event.target === dom.methodDialog) closeMethod();
  });

  dom.methodDialog.addEventListener('close', () => {
    if (methodTrigger) {
      methodTrigger.focus();
      methodTrigger = null;
    }
  });

  const redraw = () => scheduleDraw();
  if ('ResizeObserver' in window) {
    const resizeObserver = new ResizeObserver(redraw);
    resizeObserver.observe(dom.mapWrap);
  }
  window.addEventListener('resize', redraw);
}

/* --------------------------------------------------------------- controls --- */

function handleControlChange(event) {
  if (event.target === dom.region && REGIONS[dom.region.value]) {
    state.region = dom.region.value;
    if (state.region === 'catalog' && state.mainshockId) {
      applyMainshockToScenario();
      return;
    }
  }

  readFormState();
  syncControlLabels();
  updatePresetActive();
  scheduleForecast();
}

function readFormState() {
  const regionKey = REGIONS[dom.region.value] ? dom.region.value : state.region;
  const magnitude = clamp(numberOr(dom.magnitude.value, state.magnitude), 3, 9);
  const depth = clamp(Math.round(numberOr(dom.depth.value, state.depth)), 0, 700);
  const elapsed = clamp(Math.round(numberOr(dom.elapsed.value, state.elapsed)), 0, 720);
  const checkedWindow = dom.windowInputs.find((input) => input.checked);

  state.region = regionKey;
  state.magnitude = Math.round(magnitude * 10) / 10;
  state.depth = depth;
  state.elapsed = elapsed;
  if (checkedWindow && WINDOWS[checkedWindow.value]) {
    state.window = checkedWindow.value;
  }

  return state;
}

function syncControlValues() {
  dom.region.value = state.region;
  if (dom.engineSelect) dom.engineSelect.value = state.engine;
  dom.magnitude.value = state.magnitude.toFixed(1);
  dom.depth.value = String(state.depth);
  dom.elapsed.value = String(state.elapsed);
  dom.windowInputs.forEach((input) => {
    input.checked = input.value === state.window;
  });
}

function syncControlLabels() {
  const region = REGIONS[state.region];
  dom.magnitudeOut.textContent = `M ${state.magnitude.toFixed(1)}`;

  if (state.region === 'catalog' && state.catalogAnalysis) {
    const main = state.catalogAnalysis.main;
    dom.coords.textContent =
      `${formatLatLon(main.lat, main.lon)} · ${formatDayTime(main.time)} UTC · ${shortPlace(main.place, 60)}`;
  } else {
    dom.coords.textContent = `${region.latLabel}, ${region.lonLabel} · ${region.setting}`;
  }
}

function updatePresetActive() {
  let activeKey = null;

  Object.entries(PRESETS).forEach(([key, preset]) => {
    if (
      preset.region === state.region &&
      Math.abs(preset.magnitude - state.magnitude) < 0.051 &&
      Math.abs(preset.depth - state.depth) < 0.5 &&
      Math.abs(preset.elapsed - state.elapsed) < 0.5 &&
      preset.window === state.window
    ) {
      activeKey = key;
    }
  });

  dom.presetButtons.forEach((button) => {
    button.classList.toggle('is-active', button.dataset.preset === activeKey);
  });
}

function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;

  Object.assign(state, {
    region: preset.region,
    magnitude: preset.magnitude,
    depth: preset.depth,
    elapsed: preset.elapsed,
    window: preset.window
  });

  syncControlValues();
  syncControlLabels();
  updatePresetActive();
  performForecast();
}

function scheduleForecast() {
  window.clearTimeout(state.forecastTimer);
  state.forecastTimer = window.setTimeout(performForecast, 95);
}

/* --------------------------------------------------------------- USGS CSV --- */

async function loadCatalog() {
  const key = USGS_FEEDS[dom.catalogFeed.value] ? dom.catalogFeed.value : 'live-week';
  const feed = USGS_FEEDS[key];

  dom.loadCatalog.disabled = true;
  setCatalogStatus('loading', `Loading ${feed.label}…`);
  setDataBanner('loading', `Loading ${feed.label} CSV…`, 'The bundled USGS snapshot is used if this feed cannot be reached.');

  try {
    const text = await fetchText(feed.url);
    applyCatalog(text, {
      kind: key === 'bundled' ? 'bundled' : 'live',
      feedKey: key,
      label: feed.label,
      url: feed.url,
      period: feed.period
    });
  } catch (error) {
    if (key !== 'bundled') {
      setCatalogStatus('fallback', 'Live feed unavailable — loading the bundled USGS CSV snapshot.');
      setDataBanner('fallback', 'Live feed unavailable — using bundled USGS CSV.', 'The forecast is now based on a saved USGS snapshot.');
      try {
        const bundled = USGS_FEEDS.bundled;
        const text = await fetchText(bundled.url);
        applyCatalog(text, {
          kind: 'bundled',
          feedKey: 'bundled',
          label: bundled.label,
          url: bundled.url,
          period: bundled.period,
          fallback: true,
          error: error.message
        });
      } catch (fallbackError) {
        setCatalogStatus(
          'error',
          'Live feed and bundled CSV could not be loaded. Start a local server or import a USGS CSV file.'
        );
        setDataBanner('error', 'CSV catalog unavailable', 'Use a local server or import a USGS-format CSV file.');
      }
    } else {
      setCatalogStatus(
        'error',
        'Bundled CSV could not be loaded. Start a local server or import a USGS CSV file.'
      );
      setDataBanner('error', 'Bundled CSV unavailable', 'Use a local server or import a USGS-format CSV file.');
    }
  } finally {
    dom.loadCatalog.disabled = false;
  }
}

async function fetchText(url) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), CATALOG_FETCH_MS);

  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.text();
  } finally {
    window.clearTimeout(timer);
  }
}

/* ------------------------------------------------------------ ML estimator --- */

async function loadMlModel() {
  state.mlStatus = 'Loading local Random Forest…';
  updateEngineBox();

  try {
    const response = await fetch(MODEL_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const model = await response.json();
    if (!Array.isArray(model.trees) || !Array.isArray(model.features)) {
      throw new Error('unexpected model format');
    }

    state.mlModel = model;
    state.engine = 'ml';
    state.mlStatus = `Random Forest loaded · ${model.trees.length} trees`;
  } catch (error) {
    state.mlModel = null;
    state.engine = 'baseline';
    state.mlStatus = `ML model unavailable (${error.message}) — baseline active`;
  }

  updateEngineBox();
  updateMethodDialog();
  performForecast();
}

function updateEngineBox() {
  if (!dom.engineSelect) return;

  const model = state.mlModel;
  const usingMl = state.engine === 'ml' && Boolean(model);
  dom.engineSelect.value = state.engine;
  dom.engineName.textContent = usingMl ? 'Trained Random Forest' : 'Catalog-calibrated baseline';
  dom.engineVersion.textContent = usingMl ? `v${model.version}` : 'v0.2';
  dom.engineStatus.textContent = state.mlStatus;

  if (usingMl && model.validation) {
    const validation = model.validation;
    dom.engineMetrics.textContent =
      `Time-ordered holdout · ${validation.samples.test.toLocaleString()} samples · ` +
      `MAE ${validation.ml.mae} vs baseline ${validation.baseline.mae} M3+ events.`;
  } else {
    dom.engineMetrics.textContent =
      'Transparent ETAS-inspired baseline. Switch to the Random Forest when it is loaded.';
  }
}

function updateMethodDialog() {
  if (!dom.methodValidation) return;

  const model = state.mlModel;
  if (!model || !model.validation) {
    dom.methodValidation.textContent =
      'The bundled Random Forest could not be loaded in this browser, so the baseline remains active.';
    return;
  }

  const validation = model.validation;
  dom.methodValidation.textContent =
    `Bundled model: ${model.algorithm}, ${model.nodeCount.toLocaleString()} nodes. ` +
    `Time-ordered holdout (${validation.samples.test.toLocaleString()} event-window samples, ` +
    `${validation.mainshocks.test} mainshocks): MAE ${validation.ml.mae} M≥3 events ` +
    `(baseline ${validation.baseline.mae}), Brier ${validation.ml.brier} (baseline ${validation.baseline.brier}). ` +
    'This same-split comparison is a prototype check, not an accuracy claim or operational validation.';
}

function predictMlCount(features) {
  const model = state.mlModel;
  if (!model || !Array.isArray(model.trees)) return null;

  let total = 0;
  for (const tree of model.trees) {
    let node = 0;
    let guard = 0;
    while (tree.l[node] !== -1 && guard < 256) {
      const value = features[tree.f[node]];
      if (!Number.isFinite(value)) break;
      node = value <= tree.t[node] ? tree.l[node] : tree.r[node];
      guard += 1;
    }
    total += tree.v[node] || 0;
  }

  const mean = total / model.trees.length;
  return clamp(Math.expm1(clamp(mean, 0, 20)), 0.02, 600);
}

async function handleCatalogFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  setCatalogStatus('loading', `Reading ${file.name}…`);
  setDataBanner('loading', `Reading ${file.name}…`, 'Checking USGS CSV columns and earthquake rows.');

  try {
    const text = typeof file.text === 'function' ? await file.text() : await readFileText(file);
    applyCatalog(text, {
      kind: 'imported',
      feedKey: 'imported',
      label: `Imported · ${file.name}`,
      url: 'local file',
      period: 'imported CSV'
    });
  } catch (error) {
    setCatalogStatus('error', `Could not read ${file.name}: ${error.message}`);
    setDataBanner('error', 'CSV import failed', 'Choose a USGS Earthquake Catalog CSV with the standard column headings.');
  }

  event.target.value = '';
}

function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('file reader error'));
    reader.readAsText(file);
  });
}

function applyCatalog(text, meta) {
  const rows = parseUsgsCsv(text);
  if (!rows.length) throw new Error('No earthquake rows found');

  state.catalog = {
    rows,
    meta: {
      ...meta,
      count: rows.length,
      latestTime: rows[0].time,
      loadedAt: new Date()
    }
  };

  populateMainshocks(rows);

  const keepCurrent = state.mainshockId && rows.some((row) => row.id === state.mainshockId);
  const firstId = dom.mainshockSelect.options.length ? dom.mainshockSelect.options[0].value : null;
  dom.mainshockSelect.value = keepCurrent ? state.mainshockId : firstId;
  state.mainshockId = dom.mainshockSelect.value;
  dom.mainshockSelect.disabled = false;

  applyMainshockToScenario();
  updateCatalogLabels();
}

function populateMainshocks(rows) {
  const options = rows
    .slice()
    .sort((a, b) => b.mag - a.mag || b.time - a.time)
    .slice(0, 15)
    .map((row) => {
      const option = document.createElement('option');
      option.value = row.id;
      option.textContent = `M${row.mag.toFixed(1)} · ${shortPlace(row.place, 42)} · ${formatDayTime(row.time)} UTC`;
      return option;
    });

  dom.mainshockSelect.replaceChildren(...options);
}

function handleMainshockChange() {
  state.mainshockId = dom.mainshockSelect.value;
  applyMainshockToScenario();
}

function applyMainshockToScenario() {
  const main = findMainshock();
  if (!main) return;

  const analysis = analyzeCatalog(main);
  state.mainshockId = main.id;
  state.catalogAnalysis = analysis;
  state.region = 'catalog';

  const region = REGIONS.catalog;
  region.name = `USGS · ${shortPlace(main.place, 34)}`;
  region.latLabel = formatLatLon(main.lat, main.lon);
  region.lonLabel = '';
  region.angle = analysis.angle;
  region.faultLength = analysis.faultLength;
  region.setting = `catalog trend ${Math.round(analysis.angle)}° · ${Math.round(analysis.radiusKm)} km search`;

  state.magnitude = Math.round(main.mag * 10) / 10;
  state.depth = clamp(Math.round(main.depth), 0, 700);
  const referenceTime = state.catalog.meta ? state.catalog.meta.latestTime : main.time;
  state.elapsed = clamp(Math.round((referenceTime - main.time) / 3600000), 0, 720);

  dom.region.value = 'catalog';
  syncControlValues();
  syncControlLabels();
  updatePresetActive();
  performForecast();

  dom.mainshockHelp.textContent =
    `${shortPlace(main.place, 54)} · ${analysis.observed.length} observed events within ${Math.round(analysis.radiusKm)} km ` +
    `over ${formatHours(analysis.availableHours)} of catalog time.`;
}

function findMainshock() {
  if (!state.catalog.rows.length) return null;
  return state.catalog.rows.find((row) => row.id === state.mainshockId) || state.catalog.rows[0];
}

function analyzeCatalog(main) {
  const meta = state.catalog.meta;
  const referenceTime = meta ? meta.latestTime : main.time;
  const availableHours = clamp((referenceTime - main.time) / 3600000, 0, 720);
  const radiusKm = clamp(60 + (main.mag - 5) * 45, 80, 220);
  const observed = [];

  state.catalog.rows.forEach((row) => {
    if (row.id === main.id || row.time <= main.time || row.type !== 'earthquake') return;

    const hours = (row.time - main.time) / 3600000;
    if (hours > availableHours || hours > 720) return;

    const local = localKm(row, main);
    const distance = Math.hypot(local.x, local.y);
    if (distance > radiusKm) return;

    observed.push({ row, x: local.x, y: local.y, distance, hours });
  });

  observed.sort((a, b) => a.hours - b.hours);
  const observedM3 = observed.filter((event) => event.row.mag >= 3).length;

  const magnitudeFactor = Math.pow(10, 0.8 * (main.mag - 5));
  const depthFactor = 1 + clamp(main.depth, 0, 700) / 500;
  const baselineAvailable =
    availableHours > 0
      ? 12 * magnitudeFactor * depthFactor * (omoriIntegral(0, availableHours / 24) / OMORI_NORM)
      : 0;

  const calibrated = observedM3 > 0 && baselineAvailable > 0;
  const productivityScale = calibrated
    ? clamp(observedM3 / Math.max(0.25, baselineAvailable), 0.2, 5)
    : 1;

  const trend = inferTrend(observed, main);

  return {
    main,
    radiusKm,
    availableHours,
    observed,
    observedM3,
    calibrated,
    productivityScale,
    angle: trend.angle,
    faultLength: trend.faultLength,
    strength: trend.strength
  };
}

/* Events available at the selected forecast start time. Later catalog rows are
   deliberately excluded so the outlook does not use its own future. */
function observedThrough(analysis, elapsedHours) {
  if (!analysis || !analysis.observed) return [];
  return analysis.observed.filter((event) => event.hours <= elapsedHours + 0.001);
}

function productivityScaleAt(analysis, elapsedHours) {
  if (!analysis) return 1;

  const early = observedThrough(analysis, elapsedHours);
  const earlyM3 = early.filter((event) => event.row.mag >= 3).length;
  const main = analysis.main;
  const magnitudeFactor = Math.pow(10, 0.8 * (main.mag - 5));
  const depthFactor = 1 + clamp(main.depth, 0, 700) / 500;
  const baselineEarly =
    elapsedHours > 0
      ? 12 * magnitudeFactor * depthFactor * (omoriIntegral(0, elapsedHours / 24) / OMORI_NORM)
      : 0;

  if (elapsedHours <= 0 || earlyM3 <= 0 || baselineEarly <= 0) return 1;
  return clamp(earlyM3 / Math.max(0.25, baselineEarly), 0.2, 5);
}

/* Feature order must match tools/train_ml.py and model.features exactly. */
function computeMlFeatures(input, analysis) {
  const windowHours = WINDOWS[input.window].hours;
  const magnitude = input.magnitude;
  const depth = clamp(input.depth, 0, 700);
  const main = analysis ? analysis.main : { mag: magnitude, depth };
  const radius = analysis
    ? analysis.radiusKm
    : clamp(60 + (magnitude - 5) * 45, 80, 220);
  const elapsed = clamp(input.elapsed, 0, 720);

  const early = observedThrough(analysis, elapsed);
  const earlyM3 = early.filter((event) => event.row.mag >= 3).length;
  const earlyAll = early.length;
  const earlyMaxMag = early.length
    ? Math.max(...early.map((event) => event.row.mag))
    : main.mag;
  const trend = analysis
    ? inferTrend(early, main)
    : { angle: 0, faultLength: 60, strength: 0 };

  const magnitudeFactor = Math.pow(10, 0.8 * (magnitude - 5));
  const depthFactor = 1 + depth / 500;
  const omoriFraction = omoriIntegral(elapsed / 24, (elapsed + windowHours) / 24) / OMORI_NORM;
  const baselineExpected = 3 + 12 * magnitudeFactor * depthFactor * omoriFraction;
  const baselineEarly =
    elapsed > 0
      ? 12 * magnitudeFactor * depthFactor * (omoriIntegral(0, elapsed / 24) / OMORI_NORM)
      : 0;
  const productivityScale =
    elapsed > 0 && earlyM3 > 0 && baselineEarly > 0
      ? clamp(earlyM3 / Math.max(0.25, baselineEarly), 0.2, 5)
      : 1;
  const baselinePrediction = baselineExpected * productivityScale;
  const theta = (trend.angle * Math.PI) / 180;

  return [
    magnitude,
    depth,
    elapsed,
    windowHours,
    radius,
    Math.log10(magnitudeFactor),
    depthFactor,
    omoriFraction,
    Math.log1p(earlyM3),
    Math.log1p(earlyAll),
    earlyM3 / Math.max(elapsed, 1),
    Math.max(0, earlyMaxMag - magnitude),
    productivityScale,
    trend.faultLength / 100,
    Math.cos(theta),
    Math.sin(theta),
    trend.strength,
    Math.log1p(baselineExpected),
    Math.log1p(baselinePrediction)
  ];
}

function inferTrend(observed, main) {
  if (observed.length < 3) {
    return {
      angle: main.mag >= 7 ? 12 : -6,
      faultLength: clamp(40 + (main.mag - 5) * 10, 40, 140),
      strength: 0
    };
  }

  const sample = observed.slice(0, 80);
  let meanX = 0;
  let meanY = 0;
  sample.forEach((event) => {
    meanX += event.x;
    meanY += event.y;
  });
  meanX /= sample.length;
  meanY /= sample.length;

  let covXX = 0;
  let covYY = 0;
  let covXY = 0;
  sample.forEach((event) => {
    const dx = event.x - meanX;
    const dy = event.y - meanY;
    covXX += dx * dx;
    covYY += dy * dy;
    covXY += dx * dy;
  });
  covXX /= sample.length;
  covYY /= sample.length;
  covXY /= sample.length;

  const angle = (0.5 * Math.atan2(2 * covXY, covXX - covYY) * 180) / Math.PI;
  const half = (covXX + covYY) / 2;
  const radius = Math.sqrt(Math.pow((covXX - covYY) / 2, 2) + covXY * covXY);
  const eigenMax = half + radius;
  const eigenMin = Math.max(0, half - radius);
  const faultLength = clamp(Math.sqrt(Math.max(0, eigenMax)) * 4, 40, 150);
  const strength = eigenMax + eigenMin > 0 ? (eigenMax - eigenMin) / (eigenMax + eigenMin) : 0;

  return { angle, faultLength, strength };
}

function localKm(row, main) {
  const latScale = 111.32;
  const lonScale = 111.32 * Math.max(0.05, Math.cos((main.lat * Math.PI) / 180));
  return {
    x: (row.lon - main.lon) * lonScale,
    y: (row.lat - main.lat) * latScale
  };
}

function parseUsgsCsv(text) {
  const table = parseCsvRows(text.replace(/^\uFEFF/, ''));
  if (table.length < 2) return [];

  const header = table[0].map((name) => name.trim().toLowerCase());
  const index = (name) => header.indexOf(name);
  const required = ['time', 'latitude', 'longitude', 'depth', 'mag'];
  if (required.some((name) => index(name) === -1)) {
    throw new Error('missing USGS columns');
  }

  const idx = {
    time: index('time'),
    lat: index('latitude'),
    lon: index('longitude'),
    depth: index('depth'),
    mag: index('mag'),
    magType: index('magtype'),
    place: index('place'),
    type: index('type'),
    status: index('status'),
    net: index('net'),
    id: index('id'),
    url: index('url')
  };

  const cell = (cells, key) => (idx[key] >= 0 && cells[idx[key]] !== undefined ? cells[idx[key]] : '');

  const rows = [];
  for (let rowIndex = 1; rowIndex < table.length; rowIndex += 1) {
    const cells = table[rowIndex];
    const time = Date.parse(cell(cells, 'time'));
    const lat = Number(cell(cells, 'lat'));
    const lon = Number(cell(cells, 'lon'));
    const depth = Number(cell(cells, 'depth'));
    const mag = Number(cell(cells, 'mag'));
    const type = cell(cells, 'type').toLowerCase();

    if (![time, lat, lon, depth, mag].every(Number.isFinite)) continue;
    if (type && type !== 'earthquake') continue;

    rows.push({
      time,
      lat,
      lon,
      depth,
      mag,
      magType: cell(cells, 'magType'),
      place: cell(cells, 'place'),
      type: type || 'earthquake',
      status: cell(cells, 'status'),
      net: cell(cells, 'net'),
      id: cell(cells, 'id') || `row-${rowIndex}`,
      url: cell(cells, 'url')
    });
  }

  rows.sort((a, b) => b.time - a.time);
  return rows.slice(0, MAX_CATALOG_ROWS);
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];

    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field.replace(/\r$/, ''));
      if (row.some((value) => value.trim() !== '')) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }

  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ''));
    if (row.some((value) => value.trim() !== '')) rows.push(row);
  }

  return rows;
}

function setCatalogStatus(kind, message) {
  dom.catalogStatus.dataset.status = kind;
  dom.catalogStatus.textContent = message;
}

function setDataBanner(status, title, sub) {
  dom.dataBanner.dataset.status = status;
  dom.dataBannerTitle.textContent = title;
  dom.dataBannerSub.textContent = sub;
}

function updateCatalogLabels() {
  const meta = state.catalog.meta;
  if (!meta) return;

  const count = meta.count.toLocaleString();
  const latest = formatDayTime(meta.latestTime);
  const kindLabel =
    meta.kind === 'live' ? 'USGS live' : meta.kind === 'bundled' ? 'USGS snapshot' : 'Imported CSV';

  setCatalogStatus(
    meta.fallback ? 'fallback' : meta.kind,
    `${kindLabel} · ${meta.label} · ${count} events · latest ${latest} UTC`
  );

  const dot = dom.sourcePill.querySelector('.status-dot');
  dom.sourcePill.replaceChildren(dot, document.createTextNode(` ${kindLabel} · prototype`));
  dom.headerContext.textContent = `${count} CSV events · ${meta.period}`;

  setDataBanner(
    meta.fallback ? 'fallback' : meta.kind,
    meta.fallback
      ? 'Live feed unavailable — using the bundled USGS CSV snapshot.'
      : `${kindLabel} loaded · ${count} earthquake rows`,
    meta.kind === 'imported'
      ? 'The selected event and observed aftershocks come from your imported CSV.'
      : `Latest row: ${latest} UTC. Select a mainshock to recompute the outlook.`
  );

  dom.exportObservedCsv.disabled = false;
}

/* --------------------------------------------------------------- forecast --- */

function performForecast() {
  readFormState();
  syncControlLabels();
  updatePresetActive();

  const result = buildForecast(state);
  state.result = result;
  state.hover = null;
  dom.mapTooltip.hidden = true;

  const region = REGIONS[state.region];
  const windowInfo = WINDOWS[state.window];
  const analysis = state.region === 'catalog' ? state.catalogAnalysis : null;
  const title = analysis ? shortPlace(analysis.main.place, 52) : region.name;
  const start = state.elapsed === 0 ? 'mainshock time' : `+${state.elapsed} h`;

  dom.scenarioTitle.textContent = `${title} · M${state.magnitude.toFixed(1)}`;
  dom.scenarioSub.textContent = analysis
    ? `${windowInfo.label} estimate from ${start} · ${region.setting}`
    : `${windowInfo.label} forecast starting ${formatElapsed(state.elapsed)} · 3 km grid`;
  const engineLabel = result.engine === 'ml' ? 'Random Forest' : 'baseline';
  dom.runLabel.textContent =
    `${analysis ? 'Catalog outlook' : 'Scenario run'} · ${engineLabel} · ${timeLabel()}`;

  renderSummary(result);
  renderPriority(result);
  renderRate(result);
  renderEvidence(analysis);
  updateMapFoot();
  updateCanvasLabel(result);

  document.title = `${title} M${state.magnitude.toFixed(1)} · Aftershock Forecast Studio`;

  dom.srStatus.textContent =
    `Outlook updated for ${title}, magnitude ${state.magnitude.toFixed(1)} using the ` +
    `${result.engine === 'ml' ? 'trained Random Forest' : 'catalog-calibrated baseline'}. ` +
    `Peak cell estimate ${formatPct(result.peak)} percent. ` +
    `${Math.round(result.expectedTotal).toLocaleString()} estimated M3 or greater aftershocks in the window. ` +
    `${analysis ? `${observedThrough(analysis, state.elapsed).length} observed catalog events available before the forecast start.` : 'Synthetic context events shown.'}`;

  scheduleDraw();
}

function buildSpatialWeights(input, region, analysis) {
  const theta = (region.angle * Math.PI) / 180;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  const sigma = (6.5 + (input.magnitude - 5) * 3.0) * (0.9 + input.depth / 220);
  const ridgeWidth = Math.max(1.4, 2.0 + (input.magnitude - 5) * 1.0);
  const faultHalf = (region.faultLength / 2) * (0.55 + (input.magnitude - 5) * 0.10);

  const prior = new Float64Array(CELL_COUNT);
  let priorSum = 0;

  for (let row = 0; row < MAP.rows; row += 1) {
    for (let col = 0; col < MAP.cols; col += 1) {
      const index = row * MAP.cols + col;
      const x = MAP.xMin + (col + 0.5) * MAP.cell;
      const y = MAP.yMax - (row + 0.5) * MAP.cell;
      const along = x * cosTheta + y * sinTheta;
      const across = -x * sinTheta + y * cosTheta;

      const core = Math.exp(
        -(
          (along * along) / (2 * Math.pow(sigma * 1.5, 2)) +
          (across * across) / (2 * Math.pow(sigma * 0.6, 2))
        )
      );
      const ridge =
        Math.exp(-(across * across) / (2 * ridgeWidth * ridgeWidth)) *
        Math.exp(-(along * along) / (2 * Math.pow(faultHalf * 1.15, 2)));

      const noiseA = valueNoise(x * 0.075, y * 0.075, region.seed) - 0.5;
      const noiseB = valueNoise(x * 0.21, y * 0.21, region.seed + 19) - 0.5;
      const texture = noiseA * 0.65 + noiseB * 0.35;

      const value = Math.max(0.00001, 0.92 * core + 0.78 * ridge + 0.06 + texture * 0.075);
      prior[index] = value;
      priorSum += value;
    }
  }

  const observedKernel = new Float64Array(CELL_COUNT);
  let observedSum = 0;
  const earlyObserved = observedThrough(analysis, input.elapsed);

  if (earlyObserved.length >= 2) {
    const samples = earlyObserved
      .slice()
      .sort((a, b) => b.row.mag - a.row.mag)
      .slice(0, 80);

    samples.forEach((event) => {
      const sigmaKm = clamp(4 + Math.max(0, 4 - event.row.mag) * 1.2 + (event.row.mag >= 5 ? 2 : 0), 4, 12);
      const strength = 0.5 + Math.max(0, event.row.mag) / 10;
      const colCenter = Math.floor((event.x - MAP.xMin) / MAP.cell);
      const rowCenter = Math.floor((MAP.yMax - event.y) / MAP.cell);
      const radiusCells = Math.ceil((sigmaKm * 3) / MAP.cell);

      for (
        let row = Math.max(0, rowCenter - radiusCells);
        row <= Math.min(MAP.rows - 1, rowCenter + radiusCells);
        row += 1
      ) {
        const y = MAP.yMax - (row + 0.5) * MAP.cell;
        for (
          let col = Math.max(0, colCenter - radiusCells);
          col <= Math.min(MAP.cols - 1, colCenter + radiusCells);
          col += 1
        ) {
          const x = MAP.xMin + (col + 0.5) * MAP.cell;
          const d2 = (x - event.x) * (x - event.x) + (y - event.y) * (y - event.y);
          const contribution = strength * Math.exp(-d2 / (2 * sigmaKm * sigmaKm));
          const index = row * MAP.cols + col;
          observedKernel[index] += contribution;
          observedSum += contribution;
        }
      }
    });
  }

  const useObserved = observedSum > 0 && earlyObserved.length >= 3;
  const weights = new Float64Array(CELL_COUNT);

  for (let index = 0; index < CELL_COUNT; index += 1) {
    const priorWeight = prior[index] / priorSum;
    if (useObserved) {
      weights[index] = 0.62 * priorWeight + 0.38 * (observedKernel[index] / observedSum);
    } else {
      weights[index] = priorWeight;
    }
  }

  return { weights, spatialSource: useObserved ? 'observed + prior' : 'prior' };
}

function buildForecast(input) {
  const region = REGIONS[input.region];
  const windowInfo = WINDOWS[input.window];
  const magnitude = input.magnitude;
  const depth = input.depth;
  const elapsed = input.elapsed;
  const analysis = input.region === 'catalog' ? state.catalogAnalysis : null;
  const spatial = buildSpatialWeights(input, region, analysis);
  const weights = spatial.weights;

  const magnitudeFactor = Math.pow(10, 0.8 * (magnitude - 5));
  const depthFactor = 1 + depth / 500;
  const temporalFactor =
    omoriIntegral(elapsed / 24, (elapsed + windowInfo.hours) / 24) / OMORI_NORM;
  const productivityScale = productivityScaleAt(analysis, elapsed);
  const baselineExpected = 3 + 12 * magnitudeFactor * temporalFactor * depthFactor;
  const mlPrediction =
    state.engine === 'ml' && state.mlModel
      ? predictMlCount(computeMlFeatures(input, analysis))
      : null;
  const engine = mlPrediction === null ? 'baseline' : 'ml';
  const expectedTotal = clamp(mlPrediction === null ? baselineExpected * productivityScale : mlPrediction, 0.02, 600);

  let maxWeight = 0;
  let peakIndex = 0;
  const probabilities = new Float64Array(CELL_COUNT);

  for (let index = 0; index < CELL_COUNT; index += 1) {
    const probability = 1 - Math.exp(-expectedTotal * weights[index]);
    probabilities[index] = Math.min(0.995, probability);
    if (weights[index] > maxWeight) {
      maxWeight = weights[index];
      peakIndex = index;
    }
  }

  const peak = probabilities[peakIndex];
  const peakRow = Math.floor(peakIndex / MAP.cols);
  const peakCol = peakIndex % MAP.cols;
  const peakX = MAP.xMin + (peakCol + 0.5) * MAP.cell;
  const peakY = MAP.yMax - (peakRow + 0.5) * MAP.cell;

  let highCells = 0;
  for (let index = 0; index < CELL_COUNT; index += 1) {
    if (probabilities[index] >= HIGH_RISK_THRESHOLD) highCells += 1;
  }

  const pastTemporal = elapsed > 0 ? omoriIntegral(0, elapsed / 24) / OMORI_NORM : 0;
  const syntheticPastCount = Math.round(12 * magnitudeFactor * pastTemporal * depthFactor);

  let events;
  let pastCount;
  if (analysis && analysis.observed.length) {
    events = analysis.observed.slice(0, 180).map((event) => ({
      x: event.x,
      y: event.y,
      index: cellIndex(event.x, event.y),
      row: event.row
    }));
    pastCount = analysis.observed.length;
  } else {
    pastCount = syntheticPastCount;
    events = sampleEvents(weights, region, input, Math.min(pastCount, 180));
  }

  const shownEvents = events.length;
  const zones = buildZones(weights, expectedTotal);
  const rawRates = buildDailyRates(magnitudeFactor, depthFactor, elapsed, engine === 'ml' ? 1 : productivityScale);
  const rawWindowSum = rawRates.slice(0, windowInfo.days).reduce((sum, rate) => sum + rate, 0);
  const rateScale = rawWindowSum > 0 ? expectedTotal / rawWindowSum : 1;
  const dailyRates = rawRates.map((rate) => rate * rateScale);

  return {
    engine,
    regionKey: input.region,
    magnitude,
    depth,
    elapsed,
    window: input.window,
    windowHours: windowInfo.hours,
    windowDays: windowInfo.days,
    weights,
    probabilities,
    expectedTotal,
    baselineExpected: baselineExpected * productivityScale,
    mlPrediction,
    peak,
    peakIndex,
    peakCell: {
      distance: Math.hypot(peakX, peakY),
      x: peakX,
      y: peakY
    },
    highCells,
    highArea: highCells * CELL_AREA,
    pastCount,
    shownEvents,
    events,
    zones,
    topZone: zones[0] || null,
    dailyRates,
    analysis,
    spatialSource: spatial.spatialSource
  };
}

function buildZones(weights, expectedTotal) {
  const zonesByKey = new Map();

  for (let row = 0; row < MAP.rows; row += 1) {
    for (let col = 0; col < MAP.cols; col += 1) {
      const index = row * MAP.cols + col;
      const x = MAP.xMin + (col + 0.5) * MAP.cell;
      const y = MAP.yMax - (row + 0.5) * MAP.cell;
      const distance = Math.hypot(x, y);
      const bearing = (Math.atan2(x, y) * 180) / Math.PI;
      const normalizedBearing = (bearing + 360) % 360;
      const sectorIndex = Math.floor(((normalizedBearing + 22.5) % 360) / 45);
      const bandIndex = distanceBandIndex(distance);
      const key = `${sectorIndex}-${bandIndex}`;

      if (!zonesByKey.has(key)) {
        zonesByKey.set(key, {
          sectorIndex,
          bandIndex,
          mass: 0,
          cellCount: 0
        });
      }

      const zone = zonesByKey.get(key);
      zone.mass += weights[index];
      zone.cellCount += 1;
    }
  }

  const zones = Array.from(zonesByKey.values()).map((zone) => {
    const probability = 1 - Math.exp(-expectedTotal * zone.mass);
    const expected = expectedTotal * zone.mass;
    return {
      ...zone,
      direction: SECTORS[zone.sectorIndex],
      bandLabel: BANDS[zone.bandIndex].label,
      probability,
      expected,
      area: zone.cellCount * CELL_AREA,
      level: zoneLevel(probability)
    };
  });

  zones.sort((a, b) => b.probability - a.probability || b.expected - a.expected);
  return zones;
}

function buildDailyRates(magnitudeFactor, depthFactor, elapsed, scale = 1) {
  const productivity = 12 * magnitudeFactor * depthFactor * scale;
  const rates = [];

  for (let day = 1; day <= 30; day += 1) {
    const start = (elapsed + (day - 1) * 24) / 24;
    const end = (elapsed + day * 24) / 24;
    rates.push((productivity * omoriIntegral(start, end)) / OMORI_NORM);
  }

  return rates;
}

function sampleEvents(weights, region, input, count) {
  if (count <= 0) return [];

  const cumulative = new Float64Array(CELL_COUNT);
  let running = 0;
  for (let index = 0; index < CELL_COUNT; index += 1) {
    running += weights[index];
    cumulative[index] = running;
  }

  let seed =
    region.seed ^
    Math.round(input.magnitude * 10) * 374761393 ^
    Math.round(input.depth * 7) * 668265263 ^
    Math.round(input.elapsed * 13) * 1274126177 ^
    WINDOWS[input.window].hours * 2246822519;
  seed >>>= 0;

  const random = mulberry32(seed);
  const events = [];

  for (let eventIndex = 0; eventIndex < count; eventIndex += 1) {
    const target = random() * running;
    let low = 0;
    let high = CELL_COUNT - 1;

    while (low < high) {
      const mid = (low + high) >> 1;
      if (cumulative[mid] < target) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }

    const row = Math.floor(low / MAP.cols);
    const col = low % MAP.cols;
    const x = MAP.xMin + (col + random()) * MAP.cell;
    const y = MAP.yMax - (row + random()) * MAP.cell;

    events.push({ x, y, index: low });
  }

  return events;
}

function zoneLevel(probability) {
  if (probability >= 0.6) {
    return { key: 'high', label: 'High' };
  }
  if (probability >= 0.3) {
    return { key: 'mid', label: 'Medium' };
  }
  return { key: 'low', label: 'Low' };
}

/* --------------------------------------------------------------- rendering --- */

function renderSummary(result) {
  const windowInfo = WINDOWS[state.window];
  const topZone = result.topZone;

  dom.statPeak.textContent = formatPct(result.peak);
  dom.statPeakNote.textContent = result.analysis
    ? `Peak cell ${result.peakCell.distance.toFixed(1)} km from USGS mainshock`
    : `Peak cell ${result.peakCell.distance.toFixed(1)} km from mainshock`;

  dom.statExpected.textContent = Math.round(result.expectedTotal).toLocaleString();
  dom.statExpectedNote.textContent = result.engine === 'ml'
    ? `Random Forest estimate · ${windowInfo.short} window`
    : result.analysis
      ? `Catalog-scaled baseline · ${windowInfo.short} window`
      : `Baseline estimate · ${windowInfo.short} window`;

  dom.statHigh.textContent = Math.round(result.highArea).toLocaleString();
  dom.statHighNote.textContent = result.highCells
    ? `${result.highCells} cell${result.highCells === 1 ? '' : 's'} at or above 50%`
    : 'No cells at or above 50%';

  if (topZone) {
    dom.statZone.textContent = `${topZone.direction} · ${topZone.bandLabel}`;
    dom.statZoneNote.textContent =
      `P(≥1) ${formatPct(topZone.probability)}% · ${formatCount(topZone.expected)} expected`;
  } else {
    dom.statZone.textContent = '—';
    dom.statZoneNote.textContent = 'No priority zone available';
  }
}

function renderPriority(result) {
  const rows = result.zones.slice(0, 5).map((zone, index) => {
    const row = document.createElement('tr');

    const zoneCell = document.createElement('td');
    const name = document.createElement('div');
    name.className = 'zone-name';

    const rank = document.createElement('span');
    rank.className = 'zone-rank';
    rank.textContent = String(index + 1).padStart(2, '0');

    const copy = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = `${zone.direction} · ${zone.bandLabel}`;

    const sub = document.createElement('span');
    sub.className = 'zone-sub';
    sub.textContent = `${zone.cellCount} cells · ${Math.round(zone.area)} km²`;

    copy.append(title, sub);
    name.append(rank, copy);
    zoneCell.append(name);

    const probabilityCell = document.createElement('td');
    const probability = document.createElement('span');
    probability.className = `prob-pill ${probabilityClass(zone.probability)}`;
    probability.textContent = `${formatPct(zone.probability)}%`;
    probabilityCell.append(probability);

    const expectedCell = document.createElement('td');
    expectedCell.className = 'expected-cell';
    expectedCell.textContent = formatCount(zone.expected);

    const levelCell = document.createElement('td');
    const level = document.createElement('span');
    level.className = `level level-${zone.level.key}`;
    level.textContent = zone.level.label;
    levelCell.append(level);

    row.append(zoneCell, probabilityCell, expectedCell, levelCell);
    return row;
  });

  if (!rows.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.textContent = 'No zones could be ranked for this scenario.';
    row.append(cell);
    rows.push(row);
  }

  dom.priorityBody.replaceChildren(...rows);
  dom.priorityNote.textContent =
    'Labels use scenario-relative thresholds (High ≥60%, Medium ≥30%). They are not validated alert levels.';
}

function renderRate(result) {
  const maxRate = Math.max(...result.dailyRates);
  const windowDays = WINDOWS[state.window].days;
  const bars = result.dailyRates.map((rate, index) => {
    const bar = document.createElement('div');
    bar.className = `bar${index < windowDays ? ' is-window' : ''}`;
    const height = maxRate > 0 ? Math.max(2, (rate / maxRate) * 100) : 2;
    bar.style.setProperty('--h', `${height.toFixed(1)}%`);
    bar.title = `Day ${index + 1} · ~${formatRate(rate)} modelled M≥3 events`;
    bar.setAttribute('aria-hidden', 'true');
    return bar;
  });

  dom.rateBars.replaceChildren(...bars);
  dom.rateBars.setAttribute(
    'aria-label',
    `Expected daily M3 or greater aftershock rate over 30 days. Peak about ${formatRate(maxRate)} per day.`
  );
  dom.rateMax.textContent = `Peak ${formatRate(maxRate)}/day`;

  const span = windowDays === 1 ? 'day 1' : `days 1–${windowDays}`;
  dom.rateNote.textContent =
    `Selected window covers ${span}. Bars show modelled M≥3 counts per day, not observations.`;
}

function renderEvidence(analysis) {
  if (!analysis) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.textContent = 'Choose a USGS catalog event to see its observed aftershock rows.';
    row.append(cell);
    dom.observedBody.replaceChildren(row);
    dom.evidenceSummary.textContent = 'No catalog event selected yet.';
    dom.evidenceNote.textContent = 'USGS CSV rows appear here once a catalog is loaded.';
    return;
  }

  dom.evidenceSummary.textContent =
    `M${analysis.main.mag.toFixed(1)} ${shortPlace(analysis.main.place, 64)} · ` +
    `${analysis.observed.length} observed earthquakes within ${Math.round(analysis.radiusKm)} km over ${formatHours(analysis.availableHours)} · ` +
    `${analysis.observedM3} at M≥3 · productivity scale ${analysis.productivityScale.toFixed(2)}×` +
    `${analysis.calibrated ? '' : ' (prior only)'}.`;

  const sorted = analysis.observed
    .slice()
    .sort((a, b) => b.row.mag - a.row.mag || b.hours - a.hours)
    .slice(0, 8);

  const rows = sorted.map((event) => {
    const row = document.createElement('tr');
    const cells = [
      formatDayTime(event.row.time),
      `M${event.row.mag.toFixed(1)}`,
      `${Math.round(event.row.depth)} km`,
      `${event.distance.toFixed(1)} km`
    ].map((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      return cell;
    });

    const placeCell = document.createElement('td');
    const place = document.createElement('span');
    place.className = 'evidence-place';
    place.textContent = shortPlace(event.row.place, 80);
    place.title = event.row.place || 'Unnamed location';
    placeCell.append(place);

    const linkCell = document.createElement('td');
    if (/^https?:\/\//.test(event.row.url)) {
      const link = document.createElement('a');
      link.className = 'evidence-link';
      link.href = event.row.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'Open ↗';
      linkCell.append(link);
    } else {
      linkCell.textContent = '—';
    }

    row.append(...cells, placeCell, linkCell);
    return row;
  });

  if (!rows.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.textContent = 'No observed earthquakes met the distance and time filters for this event.';
    row.append(cell);
    rows.push(row);
  }

  dom.observedBody.replaceChildren(...rows);
  dom.evidenceNote.textContent =
    `Rows are observed USGS CSV records after the mainshock, not model output. The outlook uses only rows at or before the selected forecast start (${state.elapsed} h) as a prototype signal; earthquakes beyond that time are not used. Only events within ${Math.round(analysis.radiusKm)} km are listed.`;
}

function updateMapFoot() {
  if (!state.result) return;

  const result = state.result;
  const isCatalog = Boolean(result.analysis);
  const noun = isCatalog ? 'observed USGS events' : 'synthetic early events';

  if (!state.layers.events) {
    dom.mapFootText.textContent = `Forecast heatmap · ${noun} hidden`;
    return;
  }

  if (result.shownEvents <= 0) {
    dom.mapFootText.textContent = `Forecast heatmap · no ${noun} to show`;
    return;
  }

  if (result.shownEvents < result.pastCount) {
    dom.mapFootText.textContent =
      `Forecast heatmap · ${result.shownEvents.toLocaleString()} of ~${result.pastCount.toLocaleString()} ${noun} shown`;
    return;
  }

  dom.mapFootText.textContent =
    `Forecast heatmap · ${result.shownEvents.toLocaleString()} ${noun} plotted; ${result.spatialSource} spatial weights`;
}

function updateCanvasLabel(result) {
  const region = REGIONS[state.region];
  const evidence = result.analysis
    ? `${result.shownEvents} observed USGS events plotted.`
    : 'Synthetic early events shown for context.';
  dom.mapCanvas.setAttribute(
    'aria-label',
    `${region.name} probability estimate for a magnitude ${result.magnitude.toFixed(1)} mainshock using the ` +
    `${result.engine === 'ml' ? 'trained Random Forest' : 'catalog-calibrated baseline'}. ` +
    `Peak cell estimate ${formatPct(result.peak)} percent. ` +
    `${result.highCells} grid cells at or above 50 percent. ${evidence}`
  );
}

/* --------------------------------------------------------------- map canvas --- */

function scheduleDraw() {
  if (state.drawRaf) return;
  state.drawRaf = window.requestAnimationFrame(() => {
    state.drawRaf = 0;
    drawMap();
  });
}

function sizeCanvas() {
  const canvas = dom.mapCanvas;
  if (!canvas) return null;

  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return null;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(rect.width * dpr);
  const pixelHeight = Math.round(rect.height * dpr);

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const context = canvas.getContext('2d');
  if (!context) return null;

  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function getPlot(width, height) {
  const compact = width < 560;
  const padding = {
    left: compact ? 40 : 54,
    right: compact ? 16 : 26,
    top: compact ? 56 : 60,
    bottom: 42
  };

  const availableWidth = Math.max(80, width - padding.left - padding.right);
  const availableHeight = Math.max(80, height - padding.top - padding.bottom);
  const scale = Math.min(
    availableWidth / (MAP.xMax - MAP.xMin),
    availableHeight / (MAP.yMax - MAP.yMin)
  );
  const plotWidth = (MAP.xMax - MAP.xMin) * scale;
  const plotHeight = (MAP.yMax - MAP.yMin) * scale;
  const x = padding.left + (availableWidth - plotWidth) / 2;
  const y = padding.top + (availableHeight - plotHeight) / 2;

  return { x, y, w: plotWidth, h: plotHeight, scale, width, height, compact };
}

function worldToScreen(x, y, plot) {
  return {
    x: plot.x + ((x - MAP.xMin) / (MAP.xMax - MAP.xMin)) * plot.w,
    y: plot.y + ((MAP.yMax - y) / (MAP.yMax - MAP.yMin)) * plot.h
  };
}

function drawMap() {
  const sized = sizeCanvas();
  if (!sized) return;

  const { context: ctx, width, height } = sized;
  const plot = getPlot(width, height);
  const result = state.result;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#f7faf9';
  ctx.fillRect(0, 0, width, height);

  if (!result) return;

  drawHeatmap(ctx, plot, result);
  if (state.layers.grid) drawGrid(ctx, plot);
  if (state.layers.fault) drawFault(ctx, plot);
  if (state.layers.events) drawEvents(ctx, plot, result);
  drawMainshock(ctx, plot, result);
  drawHoverCell(ctx, plot);

  drawTitleBlock(ctx, plot, result);
  drawLegend(ctx, plot);
  drawScaleBar(ctx, plot);
  drawCompass(ctx, plot);
}

function drawHeatmap(ctx, plot, result) {
  const cellWidth = plot.w / MAP.cols;
  const cellHeight = plot.h / MAP.rows;

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();

  for (let row = 0; row < MAP.rows; row += 1) {
    for (let col = 0; col < MAP.cols; col += 1) {
      const index = row * MAP.cols + col;
      ctx.fillStyle = heatColor(result.probabilities[index]);
      ctx.fillRect(
        plot.x + col * cellWidth,
        plot.y + row * cellHeight,
        cellWidth + 0.6,
        cellHeight + 0.6
      );
    }
  }

  ctx.restore();
}

function drawGrid(ctx, plot) {
  const cellWidth = plot.w / MAP.cols;
  const cellHeight = plot.h / MAP.rows;

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();

  if (plot.w > 480) {
    ctx.strokeStyle = 'rgba(18, 41, 74, 0.035)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let col = 1; col < MAP.cols; col += 1) {
      const x = plot.x + col * cellWidth;
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
    }
    for (let row = 1; row < MAP.rows; row += 1) {
      const y = plot.y + row * cellHeight;
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(18, 41, 74, 0.11)';
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  for (let x = MAP.xMin; x <= MAP.xMax; x += 10) {
    const point = worldToScreen(x, 0, plot);
    ctx.moveTo(point.x, plot.y);
    ctx.lineTo(point.x, plot.y + plot.h);
  }
  for (let y = MAP.yMin; y <= MAP.yMax; y += 10) {
    const point = worldToScreen(0, y, plot);
    ctx.moveTo(plot.x, point.y);
    ctx.lineTo(plot.x + plot.w, point.y);
  }
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = 'rgba(18, 41, 74, 0.2)';
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

  ctx.fillStyle = '#7d8c97';
  ctx.font = '8.5px "Segoe UI", Arial, sans-serif';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';

  const xTicks = [-40, -20, 0, 20, 40];
  xTicks.forEach((tick) => {
    const point = worldToScreen(tick, 0, plot);
    ctx.fillText(String(tick), point.x, plot.y + plot.h + 7);
  });

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = [-20, 0, 20];
  yTicks.forEach((tick) => {
    const point = worldToScreen(0, tick, plot);
    ctx.fillText(String(tick), plot.x - 7, point.y);
  });

  ctx.fillStyle = '#8e9ba4';
  ctx.font = '8px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('km', plot.x + plot.w, plot.y + plot.h + 25);
}

function drawFault(ctx, plot) {
  const region = REGIONS[state.region];
  const theta = (region.angle * Math.PI) / 180;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  const halfLength = Math.min(72, Math.max(30, region.faultLength * 0.65));

  const start = worldToScreen(-cosTheta * halfLength, -sinTheta * halfLength, plot);
  const end = worldToScreen(cosTheta * halfLength, sinTheta * halfLength, plot);

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();
  ctx.lineCap = 'round';

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.strokeStyle = '#31526f';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(18, 41, 74, 0.36)';
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  for (let distance = -halfLength + 6; distance < halfLength; distance += 8) {
    const center = worldToScreen(cosTheta * distance, sinTheta * distance, plot);
    const tickX = -sinTheta * 1.4;
    const tickY = cosTheta * 1.4;
    ctx.moveTo(center.x - tickX * plot.scale, center.y + tickY * plot.scale);
    ctx.lineTo(center.x + tickX * plot.scale, center.y - tickY * plot.scale);
  }
  ctx.stroke();
  ctx.restore();
}

function drawEvents(ctx, plot, result) {
  if (!result.events.length) return;

  const radius = clamp(plot.w / 470, 1.4, 2.6);

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();

  ctx.lineWidth = 0.75;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.82)';
  ctx.fillStyle = 'rgba(18, 41, 74, 0.72)';

  result.events.forEach((event) => {
    const point = worldToScreen(event.x, event.y, plot);
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  ctx.restore();
}

function drawMainshock(ctx, plot, result) {
  const point = worldToScreen(0, 0, plot);

  ctx.save();
  ctx.beginPath();
  ctx.arc(point.x, point.y, 11, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(196, 112, 58, 0.3)';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  starPath(ctx, point.x, point.y, 5, 8.5, 3.7);
  ctx.fillStyle = '#c4703a';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();

  const label = `M ${result.magnitude.toFixed(1)}`;
  ctx.font = '700 9px "Segoe UI", Arial, sans-serif';
  const labelWidth = ctx.measureText(label).width + 13;
  const labelX = point.x + 10;
  const labelY = point.y - 8;

  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.strokeStyle = 'rgba(18, 41, 74, 0.2)';
  ctx.lineWidth = 0.8;
  pathRoundRect(ctx, labelX, labelY, labelWidth, 16, 4);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#12294a';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, labelX + labelWidth / 2, labelY + 8.3);
  ctx.restore();
}

function drawHoverCell(ctx, plot) {
  if (state.hover === null) return;

  const row = Math.floor(state.hover / MAP.cols);
  const col = state.hover % MAP.cols;
  const cellWidth = plot.w / MAP.cols;
  const cellHeight = plot.h / MAP.rows;
  const x = plot.x + col * cellWidth;
  const y = plot.y + row * cellHeight;

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
  ctx.lineWidth = 2.4;
  ctx.strokeRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);
  ctx.strokeStyle = 'rgba(18, 41, 74, 0.85)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);
  ctx.restore();
}

function drawTitleBlock(ctx, plot, result) {
  const compact = plot.compact;
  const title = `${REGIONS[state.region].name} · M ${result.magnitude.toFixed(1)}`;
  const subtitle =
    result.elapsed === 0
      ? `${WINDOWS[result.window].label} forecast from mainshock`
      : `${WINDOWS[result.window].label} forecast from +${result.elapsed} h`;

  ctx.save();
  ctx.fillStyle = '#c4703a';
  ctx.fillRect(12, 10, 3, compact ? 13 : 15);

  ctx.fillStyle = '#12294a';
  ctx.font = `${compact ? '700 11.5px' : '700 13px'} Georgia, "Times New Roman", serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(title, 21, compact ? 21 : 23);

  ctx.fillStyle = '#63727f';
  ctx.font = `${compact ? '8.5px' : '9.5px'} "Segoe UI", Arial, sans-serif`;
  ctx.fillText(subtitle, 21, compact ? 35 : 38);

  if (state.layers.events && result.shownEvents > 0) {
    ctx.beginPath();
    ctx.arc(23, compact ? 46 : 50, 2.1, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(18, 41, 74, 0.78)';
    ctx.fill();
    ctx.fillStyle = '#7d8c97';
    ctx.font = '8px "Segoe UI", Arial, sans-serif';
    ctx.fillText(
      `${result.shownEvents.toLocaleString()} ${result.analysis ? 'observed USGS events' : 'synthetic early events'}`,
      29,
      compact ? 49 : 53
    );
  }

  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(143, 78, 35, 0.78)';
  ctx.font = '700 8px "Segoe UI", Arial, sans-serif';
  const engineTag = state.result && state.result.engine === 'ml' ? 'RANDOM FOREST' : 'BASELINE';
  ctx.fillText(plot.width > 560 ? `PROTOTYPE · ${engineTag}` : 'PROTOTYPE', plot.width - 14, 20);
  ctx.restore();
}

function drawLegend(ctx, plot) {
  const compact = plot.w < 480;
  const width = compact ? 126 : 150;
  const height = compact ? 44 : 48;
  const x = plot.x + plot.w - width - 10;
  const y = plot.y + plot.h - height - 10;

  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.strokeStyle = 'rgba(18, 41, 74, 0.16)';
  ctx.lineWidth = 0.8;
  pathRoundRect(ctx, x, y, width, height, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#63727f';
  ctx.font = '700 7.5px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('P(≥1 M3+ EVENT)', x + 9, y + 13);

  const gradientX = x + 9;
  const gradientY = y + 20;
  const gradientWidth = width - 18;
  const gradientHeight = 6;
  const gradient = ctx.createLinearGradient(gradientX, gradientY, gradientX + gradientWidth, gradientY);
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((stop) => {
    gradient.addColorStop(stop, heatColor(stop));
  });
  ctx.fillStyle = gradient;
  ctx.fillRect(gradientX, gradientY, gradientWidth, gradientHeight);
  ctx.strokeStyle = 'rgba(18, 41, 74, 0.14)';
  ctx.lineWidth = 0.6;
  ctx.strokeRect(gradientX, gradientY, gradientWidth, gradientHeight);

  ctx.fillStyle = '#8a98a2';
  ctx.font = '7.5px "Segoe UI", Arial, sans-serif';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText('0', gradientX, gradientY + 9);
  ctx.textAlign = 'center';
  ctx.fillText('50', gradientX + gradientWidth / 2, gradientY + 9);
  ctx.textAlign = 'right';
  ctx.fillText('100%', gradientX + gradientWidth, gradientY + 9);
  ctx.restore();
}

function drawScaleBar(ctx, plot) {
  let kilometers = plot.scale > 8 ? 20 : 10;
  let barWidth = kilometers * plot.scale;

  if (barWidth > plot.w * 0.32) {
    kilometers = 10;
    barWidth = kilometers * plot.scale;
  }

  const x = plot.x + 12;
  const y = plot.y + plot.h - 14;

  ctx.save();
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + barWidth, y);
  ctx.stroke();

  ctx.strokeStyle = '#31526f';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + barWidth, y);
  ctx.moveTo(x, y - 4);
  ctx.lineTo(x, y + 4);
  ctx.moveTo(x + barWidth, y - 4);
  ctx.lineTo(x + barWidth, y + 4);
  ctx.stroke();

  ctx.fillStyle = '#4d6375';
  ctx.font = '700 8px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(`${kilometers} km`, x, y - 6);
  ctx.restore();
}

function drawCompass(ctx, plot) {
  const x = plot.x + plot.w - 18;
  const y = plot.y + 20;

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.moveTo(x, y + 15);
  ctx.lineTo(x, y + 1);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(18, 41, 74, 0.7)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(x, y + 15);
  ctx.lineTo(x, y + 1);
  ctx.moveTo(x, y + 1);
  ctx.lineTo(x - 3.5, y + 6);
  ctx.moveTo(x, y + 1);
  ctx.lineTo(x + 3.5, y + 6);
  ctx.stroke();

  ctx.fillStyle = '#31526f';
  ctx.font = '700 8px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('N', x, y - 2);
  ctx.restore();
}

/* --------------------------------------------------------------- tooltip --- */

function handlePointerMove(event) {
  const result = state.result;
  if (!result) return;

  const canvasRect = dom.mapCanvas.getBoundingClientRect();
  const localX = event.clientX - canvasRect.left;
  const localY = event.clientY - canvasRect.top;

  if (localX < 0 || localY < 0 || localX > canvasRect.width || localY > canvasRect.height) {
    hideTooltip();
    return;
  }

  const plot = getPlot(canvasRect.width, canvasRect.height);
  if (
    localX < plot.x ||
    localX > plot.x + plot.w ||
    localY < plot.y ||
    localY > plot.y + plot.h
  ) {
    hideTooltip();
    return;
  }

  const worldX = MAP.xMin + ((localX - plot.x) / plot.w) * (MAP.xMax - MAP.xMin);
  const worldY = MAP.yMax - ((localY - plot.y) / plot.h) * (MAP.yMax - MAP.yMin);
  const col = clamp(Math.floor((worldX - MAP.xMin) / MAP.cell), 0, MAP.cols - 1);
  const row = clamp(Math.floor((MAP.yMax - worldY) / MAP.cell), 0, MAP.rows - 1);
  const index = row * MAP.cols + col;

  if (state.hover !== index) {
    state.hover = index;
    scheduleDraw();
  }

  showTooltip(event, index, worldX, worldY);
}

function showTooltip(event, index, worldX, worldY) {
  const result = state.result;
  if (!result) return;

  const probability = result.probabilities[index];
  const expected = result.expectedTotal * result.weights[index];
  const distance = Math.hypot(worldX, worldY);
  const direction = bearingLabel(worldX, worldY);

  dom.mapTooltip.innerHTML = `
    <p class="tooltip-kicker">3 km grid cell · P(≥1 M3+)</p>
    <p class="tooltip-prob">${formatPct(probability)}<span>%</span></p>
    <div class="tooltip-rows">
      <div class="tooltip-row"><span>Expected in cell</span><b>${formatCount(expected)}</b></div>
      <div class="tooltip-row"><span>Distance</span><b>${distance.toFixed(1)} km</b></div>
      <div class="tooltip-row"><span>Direction</span><b>${direction}</b></div>
    </div>
  `;
  dom.mapTooltip.hidden = false;

  const wrapRect = dom.mapWrap.getBoundingClientRect();
  const tooltipWidth = dom.mapTooltip.offsetWidth;
  const tooltipHeight = dom.mapTooltip.offsetHeight;
  let left = event.clientX - wrapRect.left + 14;
  let top = event.clientY - wrapRect.top + 14;

  if (left + tooltipWidth > wrapRect.width - 8) {
    left = event.clientX - wrapRect.left - tooltipWidth - 14;
  }
  if (top + tooltipHeight > wrapRect.height - 8) {
    top = event.clientY - wrapRect.top - tooltipHeight - 14;
  }

  left = clamp(left, 8, Math.max(8, wrapRect.width - tooltipWidth - 8));
  top = clamp(top, 8, Math.max(8, wrapRect.height - tooltipHeight - 8));

  dom.mapTooltip.style.left = `${left}px`;
  dom.mapTooltip.style.top = `${top}px`;
}

function hideTooltip() {
  if (dom.mapTooltip) dom.mapTooltip.hidden = true;
  if (state.hover !== null) {
    state.hover = null;
    scheduleDraw();
  }
}

/* --------------------------------------------------------------- exports --- */

function exportPng() {
  if (!state.result) return;
  drawMap();

  const filename = `aftershock-map-${state.region}-m${state.magnitude.toFixed(1)}-${state.window}.png`;

  if (dom.mapCanvas.toBlob) {
    dom.mapCanvas.toBlob((blob) => {
      if (blob) {
        downloadBlob(blob, filename);
      } else {
        downloadBlob(dataUrlToBlob(dom.mapCanvas.toDataURL('image/png')), filename);
      }
    }, 'image/png');
  } else {
    downloadBlob(dataUrlToBlob(dom.mapCanvas.toDataURL('image/png')), filename);
  }
}

function exportCsv() {
  if (!state.result) return;

  const result = state.result;
  const region = REGIONS[state.region];
  const windowInfo = WINDOWS[state.window];
  const meta = state.catalog.meta;
  const lines = [
    'Aftershock Forecast Studio - USGS CSV prototype',
    `scenario,${csvEscape(region.name)}`,
    `magnitude,${state.magnitude.toFixed(1)}`,
    `depth_km,${state.depth}`,
    `hours_since_mainshock,${state.elapsed}`,
    `forecast_window,${windowInfo.short}`,
    `model,${result.engine === 'ml' && state.mlModel ? `Random Forest v${state.mlModel.version}` : 'ETAS-inspired baseline v0.2'}`,
    `model_validation,${csvEscape(result.engine === 'ml' && state.mlModel ? state.mlModel.validation.note : 'Transparent baseline; no trained model used.')}`,
    `catalog_source,${csvEscape(meta ? `${meta.kind} · ${meta.label}` : 'none')}`,
    `catalog_url,${csvEscape(meta ? meta.url : '')}`,
    `catalog_events,${meta ? meta.count : 0}`,
    ...mainMetaLines(),
    `spatial_weights,${result.spatialSource || 'prior'}`,
    `note,Model-derived demonstration output. Not validated and not for operational use.`,
    '',
    'east_km,north_km,probability,expected_events,distance_km,bearing_deg'
  ];

  for (let row = 0; row < MAP.rows; row += 1) {
    for (let col = 0; col < MAP.cols; col += 1) {
      const index = row * MAP.cols + col;
      const x = MAP.xMin + (col + 0.5) * MAP.cell;
      const y = MAP.yMax - (row + 0.5) * MAP.cell;
      const distance = Math.hypot(x, y);
      const bearing = ((Math.atan2(x, y) * 180) / Math.PI + 360) % 360;
      lines.push(
        [
          x.toFixed(2),
          y.toFixed(2),
          result.probabilities[index].toFixed(6),
          (result.expectedTotal * result.weights[index]).toFixed(4),
          distance.toFixed(2),
          bearing.toFixed(2)
        ].join(',')
      );
    }
  }

  const filename = `aftershock-grid-${result.engine}-${state.region}-m${state.magnitude.toFixed(1)}-${state.window}.csv`;
  const blob = new Blob([`\uFEFF${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8;' });
  downloadBlob(blob, filename);
}

function mainMetaLines() {
  const analysis = state.result && state.result.analysis;
  if (!analysis) return ['mainshock_id,', 'mainshock_time,', 'mainshock_place,'];
  const main = analysis.main;
  return [
    `mainshock_id,${csvEscape(main.id)}`,
    `mainshock_time,${csvEscape(new Date(main.time).toISOString())}`,
    `mainshock_place,${csvEscape(main.place)}`
  ];
}

function exportObservedCsv() {
  const analysis = state.catalogAnalysis;
  if (!analysis) return;

  const meta = state.catalog.meta;
  const lines = [
    'USGS Earthquake Catalog - observed rows used by the prototype',
    `source,${csvEscape(meta ? `${meta.kind} · ${meta.label}` : 'unknown')}`,
    `catalog_url,${csvEscape(meta ? meta.url : '')}`,
    `mainshock_id,${csvEscape(analysis.main.id)}`,
    `mainshock_mag,${analysis.main.mag.toFixed(1)}`,
    `search_radius_km,${Math.round(analysis.radiusKm)}`,
    `available_hours,${Math.round(analysis.availableHours)}`,
    `observed_m3_count,${analysis.observedM3}`,
    `productivity_scale,${analysis.productivityScale.toFixed(3)}`,
    `note,Observed USGS rows. Not model output.`,
    '',
    'time_utc,latitude,longitude,depth_km,magnitude,mag_type,place,event_id,distance_km,hours_after_mainshock,usgs_url'
  ];

  analysis.observed.forEach((event) => {
    const row = event.row;
    lines.push(
      [
        new Date(row.time).toISOString(),
        row.lat.toFixed(5),
        row.lon.toFixed(5),
        row.depth.toFixed(2),
        row.mag.toFixed(2),
        csvEscape(row.magType),
        csvEscape(row.place),
        csvEscape(row.id),
        event.distance.toFixed(2),
        event.hours.toFixed(2),
        csvEscape(row.url)
      ].join(',')
    );
  });

  const filename = `usgs-observed-${analysis.main.id}-${state.window}.csv`;
  downloadBlob(new Blob([`\uFEFF${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8;' }), filename);
}

function initNeeds() {
  let saved = null;
  try {
    saved = JSON.parse(window.localStorage.getItem(NEEDS_STORAGE_KEY) || 'null');
  } catch (error) {
    saved = null;
  }

  if (saved && Array.isArray(saved.items)) {
    dom.needsChecks.forEach((input) => {
      input.checked = saved.items.includes(input.dataset.need);
    });
    dom.needsNotes.value = typeof saved.notes === 'string' ? saved.notes : '';
    dom.needsStatus.textContent = `Locally saved checklist restored from ${formatDayTime(Date.parse(saved.savedAt))}.`;
  }
}

function collectNeeds() {
  return {
    items: dom.needsChecks.filter((input) => input.checked).map((input) => input.dataset.need),
    notes: dom.needsNotes.value.trim(),
    savedAt: new Date().toISOString()
  };
}

function saveNeeds(silent) {
  const needs = collectNeeds();
  try {
    window.localStorage.setItem(NEEDS_STORAGE_KEY, JSON.stringify(needs));
    dom.needsStatus.textContent = silent
      ? `Autosaved locally at ${formatClock(new Date())} UTC.`
      : `Saved in this browser at ${formatDayTime(new Date())} UTC. Nothing was sent.`;
  } catch (error) {
    dom.needsStatus.textContent = 'This browser did not allow local storage. Use Export checklist CSV instead.';
  }
}

function exportNeedsCsv() {
  const needs = collectNeeds();
  const lines = [
    'Earthquake Aftershock Forecasting local needs checklist',
    `saved_at,${csvEscape(needs.savedAt)}`,
    `note,Local-only export. Not sent to any agency. Do not include sensitive personal information.`,
    '',
    'need,checked'
  ];

  dom.needsChecks.forEach((input) => {
    lines.push(`${csvEscape(input.dataset.need)},${input.checked ? 'yes' : 'no'}`);
  });

  lines.push('', `notes,${csvEscape(needs.notes)}`);

  const filename = `relief-needs-${new Date().toISOString().slice(0, 10)}.csv`;
  downloadBlob(new Blob([`\uFEFF${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8;' }), filename);
  dom.needsStatus.textContent = 'Checklist exported as CSV. Nothing was sent.';
}

function clearNeeds() {
  dom.needsChecks.forEach((input) => {
    input.checked = false;
  });
  dom.needsNotes.value = '';
  try {
    window.localStorage.removeItem(NEEDS_STORAGE_KEY);
    dom.needsStatus.textContent = 'Local checklist cleared.';
  } catch (error) {
    dom.needsStatus.textContent = 'Local checklist cleared in this view.';
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1200);
}

function dataUrlToBlob(dataUrl) {
  const [header, body] = dataUrl.split(',');
  const mime = /:(.*?);/.exec(header)[1];
  const binary = atob(body);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mime });
}

/* --------------------------------------------------------------- dialog --- */

function openMethod(event) {
  methodTrigger = event && event.currentTarget ? event.currentTarget : null;
  updateMethodDialog();
  if (typeof dom.methodDialog.showModal === 'function') {
    dom.methodDialog.showModal();
  } else {
    dom.methodDialog.setAttribute('open', '');
  }
}

function closeMethod() {
  if (typeof dom.methodDialog.close === 'function') {
    dom.methodDialog.close();
  } else {
    dom.methodDialog.removeAttribute('open');
    if (methodTrigger) {
      methodTrigger.focus();
      methodTrigger = null;
    }
  }
}

/* --------------------------------------------------------------- helpers --- */

function numberOr(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function cellIndex(x, y) {
  const col = clamp(Math.floor((x - MAP.xMin) / MAP.cell), 0, MAP.cols - 1);
  const row = clamp(Math.floor((MAP.yMax - y) / MAP.cell), 0, MAP.rows - 1);
  return row * MAP.cols + col;
}

function formatLatLon(lat, lon) {
  const latText = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}`;
  const lonText = `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`;
  return `${latText}, ${lonText}`;
}

function formatDayTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const day = String(date.getUTCDate()).padStart(2, '0');
  return `${day} ${months[date.getUTCMonth()]} ${formatClock(date)}`;
}

function formatClock(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
}

function formatHours(hours) {
  if (!Number.isFinite(hours) || hours <= 0) return '0 h';
  if (hours < 48) return `${Math.round(hours)} h`;
  return `${(hours / 24).toFixed(hours < 240 ? 1 : 0)} days`;
}

function shortPlace(place, maxLength) {
  const text = String(place || 'Unnamed location');
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

function omoriIntegral(startDays, endDays) {
  return (
    (Math.pow(startDays + OMORI_C, 1 - OMORI_P) -
      Math.pow(endDays + OMORI_C, 1 - OMORI_P)) /
    (1 - OMORI_P)
  );
}

function distanceBandIndex(distance) {
  for (let index = 0; index < BANDS.length; index += 1) {
    if (distance < BANDS[index].max) return index;
  }
  return BANDS.length - 1;
}

function bearingLabel(x, y) {
  if (x === 0 && y === 0) return 'Mainshock';
  const bearing = (Math.atan2(x, y) * 180) / Math.PI;
  const normalized = (bearing + 360) % 360;
  const index = Math.floor(((normalized + 22.5) % 360) / 45);
  return SECTORS[index];
}

function probabilityClass(probability) {
  if (probability >= 0.6) return 'prob-high';
  if (probability >= 0.3) return 'prob-mid';
  return 'prob-low';
}

function formatPct(probability) {
  const percent = clamp(probability, 0, 1) * 100;
  if (percent > 0 && percent < 1) return '<1';
  return String(Math.min(99, Math.round(percent)));
}

function formatCount(value) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 100) return Math.round(value).toLocaleString();
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function formatRate(value) {
  if (value >= 100) return Math.round(value).toLocaleString();
  return value.toFixed(1);
}

function formatElapsed(hours) {
  if (hours <= 0) return 'at the mainshock';
  if (hours < 48) return `${hours} h after the mainshock`;
  const days = hours / 24;
  return `${Number.isInteger(days) ? days : days.toFixed(1)} days after the mainshock`;
}

function timeLabel() {
  const now = new Date();
  return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
}

function csvEscape(value) {
  const string = String(value);
  return /[",\n]/.test(string) ? `"${string.replace(/"/g, '""')}"` : string;
}

function heatColor(probability) {
  return HEAT_LUT[Math.round(clamp(probability, 0, 1) * 200)];
}

function heatColorAt(probability) {
  const value = clamp(probability, 0, 1);
  let lower = HEAT_STOPS[0];
  let upper = HEAT_STOPS[HEAT_STOPS.length - 1];

  for (let index = 1; index < HEAT_STOPS.length; index += 1) {
    if (value <= HEAT_STOPS[index][0]) {
      lower = HEAT_STOPS[index - 1];
      upper = HEAT_STOPS[index];
      break;
    }
  }

  const span = upper[0] - lower[0] || 1;
  const amount = (value - lower[0]) / span;
  const rgb = lower[1].map((channel, index) =>
    Math.round(channel + (upper[1][index] - channel) * amount)
  );
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return function random() {
    value = (value + 0x6D2B79F5) >>> 0;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function hash2(x, y, seed) {
  let hash =
    Math.imul(x | 0, 374761393) +
    Math.imul(y | 0, 668265263) +
    Math.imul(seed | 0, 1274126177);
  hash = Math.imul(hash ^ (hash >>> 13), 1274126177);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}

function valueNoise(x, y, seed) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);

  const a = hash2(xi, yi, seed);
  const b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed);
  const d = hash2(xi + 1, yi + 1, seed);

  return (
    a * (1 - u) * (1 - v) +
    b * u * (1 - v) +
    c * (1 - u) * v +
    d * u * v
  );
}

function starPath(ctx, cx, cy, spikes, outerRadius, innerRadius) {
  let rotation = -Math.PI / 2;
  const step = Math.PI / spikes;

  ctx.moveTo(cx + Math.cos(rotation) * outerRadius, cy + Math.sin(rotation) * outerRadius);
  for (let index = 0; index < spikes; index += 1) {
    rotation += step;
    ctx.lineTo(cx + Math.cos(rotation) * innerRadius, cy + Math.sin(rotation) * innerRadius);
    rotation += step;
    ctx.lineTo(cx + Math.cos(rotation) * outerRadius, cy + Math.sin(rotation) * outerRadius);
  }
  ctx.closePath();
}

function pathRoundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

/* ------------------------------------------------------- model comparison --- */

function formatAuc(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : '—';
}

async function loadComparison() {
  try {
    const meta = JSON.parse(await fetchText(`${COMPARISON_BASE}meta.json?v=${ASSET_VERSION}`));
    state.comparisonMeta = meta;
    const ids = Object.keys(meta).sort((a, b) => {
      const magDifference = Number(meta[b].mag) - Number(meta[a].mag);
      return magDifference || String(meta[a].name).localeCompare(String(meta[b].name));
    });

    dom.comparisonEvent.replaceChildren();
    ids.forEach((id) => {
      const option = document.createElement('option');
      option.value = id;
      option.textContent = String(meta[id].name || id);
      dom.comparisonEvent.appendChild(option);
    });
    dom.comparisonEvent.disabled = false;
    updateComparison();
  } catch (error) {
    dom.comparisonHelp.textContent = `Historical comparison unavailable (${error.message}).`;
    return;
  }

  try {
    state.comparisonSummary = JSON.parse(await fetchText(`${COMPARISON_BASE}summary.json?v=${ASSET_VERSION}`));
    updateComparison();
  } catch (error) {
    dom.comparisonSummary.textContent = 'Historical benchmark summary unavailable.';
  }
}

function updateComparison() {
  const meta = state.comparisonMeta;
  const id = dom.comparisonEvent.value;
  if (!meta || !id || !meta[id]) return;

  const event = meta[id];
  const version = ASSET_VERSION;
  dom.comparisonAiScore.textContent = formatAuc(event.auc_nn);
  dom.comparisonManualScore.textContent = formatAuc(event.auc_base);
  dom.comparisonAftershocks.textContent = Number.isFinite(Number(event.n_aftershocks))
    ? Number(event.n_aftershocks).toLocaleString()
    : '—';
  dom.comparisonHelp.textContent = `${event.name} · ${Number(event.n_aftershocks || 0).toLocaleString()} observed aftershocks`;

  const fileId = encodeURIComponent(id);
  dom.comparisonAiImg.src = `${COMPARISON_BASE}${fileId}-ai.png?v=${version}`;
  dom.comparisonManualImg.src = `${COMPARISON_BASE}${fileId}-manual.png?v=${version}`;
  dom.comparisonActualImg.src = `${COMPARISON_BASE}${fileId}-actual.png?v=${version}`;

  const summary = state.comparisonSummary;
  if (summary) {
    dom.comparisonSummary.textContent =
      `Across all ${Number(summary.n_events || 0).toLocaleString()} historical earthquakes: ` +
      `AI average ${formatAuc(summary.auc_nn)}, manual-method average ${formatAuc(summary.auc_base)}, ` +
      `and AI scored higher in ${Math.round(Number(summary.ai_wins || 0) * 100)}% of events.`;
  }
}

/* --------------------------------------------------------------- start --- */

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
