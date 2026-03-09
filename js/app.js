/* ============================================
   Main App Controller - Part 1: Core, Navigation, Data
   ============================================ */
let dm, viz, profiler, dragDrop, mlEngine;
let currentPage = 1, rowsPerPage = 50;
const trainedModels = [];

document.addEventListener('DOMContentLoaded', () => {
    dm = new DataManager(); viz = new VisualizationEngine(); profiler = new DataProfiler(dm, viz);
    dragDrop = new DragDropEngine(); mlEngine = new MLEngine();
    window.dm = dm; window.dragDrop = dragDrop;

    // Render all pages
    Object.entries(Pages).forEach(([key, fn]) => {
        const el = document.getElementById(`page-${key === 'dataSources' ? 'data-sources' : key === 'dataProfiler' ? 'data-profiler' : key === 'dataView' ? 'data-view' : key === 'dataCleaning' ? 'data-cleaning' : key === 'dataTransform' ? 'data-transform' : key === 'columnOps' ? 'column-ops' : key === 'vizBuilder' ? 'viz-builder' : key === 'chartGallery' ? 'chart-gallery' : key === 'controlCharts' ? 'control-charts' : key === 'mlSetup' ? 'ml-setup' : key === 'mlModels' ? 'ml-models' : key === 'mlResults' ? 'ml-results' : key === 'mlPredict' ? 'ml-predict' : key}`);
        if (el) el.innerHTML = fn();
    });

    setTimeout(() => { U.el('splash-screen').classList.add('hidden'); U.el('app').classList.remove('hidden'); }, 2200);
    initNavigation(); initDataSources(); initDataView(); initCleaning(); initColumnOps();
    initVizBuilder(); initDescriptive(); initHypothesis(); initRegression(); initAnova();
    initCorrelation(); initControlCharts(); initCapability(); initPareto(); initTimeSeries();
    initForecasting(); initMLSetup(); initMLModels(); initMLPredict(); initProject(); initTopBar();
    initGallery();
});

/* ---- Navigation ---- */
const PAGE_TITLES = { 'dashboard': 'Dashboard', 'data-sources': 'Data Sources', 'data-profiler': 'Data Profiler', 'data-view': 'Data Viewer', 'data-cleaning': 'Data Cleaning', 'data-transform': 'Transform', 'column-ops': 'Column Studio', 'viz-builder': 'Visual Builder', 'chart-gallery': 'Chart Gallery', 'descriptive': 'Descriptive Statistics', 'hypothesis': 'Hypothesis Testing', 'regression': 'Regression Analysis', 'anova': 'ANOVA', 'correlation': 'Correlation Analysis', 'control-charts': 'Control Charts', 'capability': 'Capability Analysis', 'pareto': 'Pareto Analysis', 'timeseries': 'Time Series', 'forecasting': 'Forecasting', 'ml-setup': 'ML Setup', 'ml-models': 'Model Training', 'ml-results': 'Model Results', 'ml-predict': 'Predict', 'project': 'Project Management' };

function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => item.addEventListener('click', e => { e.preventDefault(); navigateTo(item.dataset.page); }));
    U.on('sidebar-toggle', 'click', () => U.el('sidebar').classList.toggle('collapsed'));
}
function navigateTo(page) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const ni = document.querySelector(`[data-page="${page}"]`); if (ni) ni.classList.add('active');
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pe = U.el(`page-${page}`); if (pe) pe.classList.add('active');
    U.el('page-title').textContent = PAGE_TITLES[page] || page;
    U.el('breadcrumb').innerHTML = `<span>Home</span><i class="fas fa-chevron-right"></i><span>${PAGE_TITLES[page] || page}</span>`;
    if (page === 'data-profiler' && dm.hasData()) profiler.renderProfileReport('profiler-content');

    // Globally update dropdowns so navigating back and forth updates Canvas UI properly
    if (typeof updateVarDropdowns === 'function') updateVarDropdowns();
}
window.navigateTo = navigateTo; window.showToast = U.toast;

/* ---- Top Bar ---- */
function initTopBar() {
    U.on('btn-undo', 'click', () => { if (dm.undo()) { refreshAll(); U.toast('Undo applied', 'success'); } else U.toast('Nothing to undo', 'info'); });
    U.on('btn-redo', 'click', () => { if (dm.redo()) { refreshAll(); U.toast('Redo applied', 'success'); } else U.toast('Nothing to redo', 'info'); });
    U.on('btn-save-project', 'click', () => navigateTo('project'));
    U.on('btn-export', 'click', () => { if (!dm.hasData()) { U.toast('No data', 'warning'); return; } U.download(dm.exportCSV(), 'export.csv'); U.toast('Exported', 'success'); });
    U.on('modal-close', 'click', () => U.closeModal());
    U.el('modal-overlay').addEventListener('click', e => { if (e.target === U.el('modal-overlay')) U.closeModal(); });
}

/* ---- Data Loading ---- */
function onDataLoaded(name) {
    dm.datasetName = name;
    U.toast(`Loaded: ${name} (${dm.data.length} rows × ${dm.columns.length} columns)`, 'success');
    refreshAll();
}
function refreshAll() {
    updateDashboard(); updateVarDropdowns(); dragDrop.setupVariables(dm.columns, dm.columnTypes); renderDataTable();
    U.el('data-status').innerHTML = `<i class="fas fa-database"></i><span>${dm.data.length} rows</span>`;
}
function loadSampleData(type) {
    let r;
    switch (type) {
        case 'manufacturing': r = generateManufacturingData(); break;
        case 'quality': r = generateQualityData(); break;
        case 'timeseries': r = generateTimeSeriesData(); break;
        case 'experiment': r = generateExperimentData(); break;
        case 'classification': r = generateClassificationData(); break;
        default: return;
    }
    dm.data = r.data; dm.columns = Object.keys(r.data[0]); dm.detectColumnTypes();
    onDataLoaded(r.name); navigateTo('data-view');
}
window.loadSampleData = loadSampleData;

/* ---- Data Sources ---- */
function initDataSources() {
    const dz = U.el('drop-zone'), fi = U.el('file-input');
    ['dragenter', 'dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(e => dz.addEventListener(e, () => dz.classList.remove('dragover')));
    dz.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); });
    fi.addEventListener('change', e => { if (e.target.files[0]) processFile(e.target.files[0]); });
    U.on('btn-paste-data', 'click', () => { const el = U.el('paste-area'); el.style.display = el.style.display === 'none' ? '' : 'none'; });
    U.on('btn-paste-load', 'click', () => { const t = U.el('paste-text').value; if (!t.trim()) { U.toast('Paste data first', 'warning'); return; } dm.loadFromClipboard(t); onDataLoaded('Pasted Data'); });
    U.on('btn-url-import', 'click', () => { const el = U.el('url-area'); el.style.display = el.style.display === 'none' ? '' : 'none'; });
    U.on('btn-url-load', 'click', () => { const url = U.el('url-input').value; if (!url) { U.toast('Enter URL', 'warning'); return; } U.toast('Fetching...', 'info'); dm.loadFromURL(url).then(() => onDataLoaded('URL Data')).catch(e => U.toast('Error: ' + e.message, 'error')); });
    U.on('btn-clipboard', 'click', () => { navigator.clipboard.readText().then(t => { dm.loadFromClipboard(t); onDataLoaded('Clipboard Data'); }).catch(() => U.toast('Clipboard access denied', 'error')); });
}
function processFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
        const opts = { delimiter: U.el('delimiter-select').value, header: U.el('header-toggle').checked, skipRows: parseInt(U.el('skip-rows').value) || 0 };
        if (file.name.endsWith('.json')) dm.parseJSON(e.target.result); else dm.parseCSV(e.target.result, opts);
        onDataLoaded(file.name);
    };
    reader.readAsText(file);
}

/* ---- Dashboard ---- */
function updateDashboard() {
    U.html('kpi-rows', dm.data.length.toLocaleString());
    U.html('kpi-cols', dm.columns.length);
    U.html('kpi-models', trainedModels.length);
    U.html('kpi-analyses', dm.analysisCount);
    if (!dm.hasData()) return;
    const profiles = dm.profileAll();
    const q = Math.round(profiles.reduce((s, p) => s + p.qualityPct, 0) / profiles.length * 100);
    U.html('kpi-quality', q + '%');
    const numCols = dm.getNumericColumns().slice(0, 4);
    if (numCols.length) {
        U.hide('dash-overview'); U.el('dash-overview-chart').style.display = '';
        viz.line('dash-overview-chart', dm.data.slice(0, 50).map((_, i) => i + 1), numCols.map(c => ({ label: c, data: dm.getNumericValues(c).slice(0, 50) })), { title: 'Data Snapshot', smooth: true });
    }
    const tc = {}; Object.values(dm.columnTypes).forEach(t => tc[t] = (tc[t] || 0) + 1);
    const te = Object.entries(tc).filter(([, v]) => v > 0);
    if (te.length) viz.pie('dash-types-chart', te.map(e => e[0]), te.map(e => e[1]), { title: 'Variable Types', doughnut: true });
}

/* ---- Data Viewer ---- */
function initDataView() {
    U.on('prev-page', 'click', () => { if (currentPage > 1) { currentPage--; renderDataTable() } });
    U.on('next-page', 'click', () => { const mp = Math.ceil(dm.data.length / rowsPerPage); if (currentPage < mp) { currentPage++; renderDataTable() } });
    U.on('rows-per-page', 'change', e => { rowsPerPage = e.target.value === 'all' ? dm.data.length : parseInt(e.target.value); currentPage = 1; renderDataTable() });
    U.on('btn-add-row', 'click', () => { const row = {}; dm.columns.forEach(c => row[c] = ''); dm.data.push(row); renderDataTable(); U.toast('Row added', 'success') });
    U.on('btn-export-csv', 'click', () => { if (!dm.hasData()) return; U.download(dm.exportCSV(), 'data.csv'); U.toast('Exported', 'success') });
}
function renderDataTable() {
    if (!dm.hasData()) return;
    U.el('data-pagination').style.display = '';
    const thead = U.el('data-table-head'), tbody = U.el('data-table-body');
    thead.innerHTML = '<tr><th>#</th>' + dm.columns.map(c => `<th title="${dm.columnTypes[c]}">${c}</th>`).join('') + '</tr>';
    const start = (currentPage - 1) * rowsPerPage, end = Math.min(start + rowsPerPage, dm.data.length);
    tbody.innerHTML = dm.data.slice(start, end).map((row, i) => {
        return `<tr><td class="row-num">${start + i + 1}</td>${dm.columns.map(c => {
            let v = row[c]; const miss = v == null || v === '';
            return `<td${miss ? ' class="missing"' : ''}>${miss ? '<span style="color:var(--danger)">—</span>' : v}</td>`;
        }).join('')}</tr>`;
    }).join('');
    U.html('page-info', `Page ${currentPage} of ${Math.ceil(dm.data.length / rowsPerPage)}`);
    U.html('data-info-text', `${dm.data.length} rows × ${dm.columns.length} columns`);
}

/* ---- Variable Dropdowns ---- */
function updateVarDropdowns() {
    document.querySelectorAll('.var-dropdown').forEach(sel => {
        const old = sel.value; sel.innerHTML = sel.multiple ? '' : '<option value="">Select variable...</option>';
        dm.columns.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = `${c} (${dm.columnTypes[c]})`; sel.appendChild(o); });
        if (old) sel.value = old;
    });
    ['desc-var-list', 'corr-var-list'].forEach(id => {
        const el = U.el(id); if (!el) return; el.innerHTML = '';
        dm.getNumericColumns().forEach(c => { const l = document.createElement('label'); l.className = 'form-check'; l.innerHTML = `<input type="checkbox" value="${c}" checked> ${c}`; el.appendChild(l); });
    });
    // ML feature list
    const fl = U.el('ml-feature-list'); if (fl) {
        fl.innerHTML = '';
        dm.getNumericColumns().forEach(c => { const l = document.createElement('label'); l.className = 'form-check'; l.innerHTML = `<input type="checkbox" value="${c}" checked class="ml-feature-check"> ${c}`; fl.appendChild(l); });
    }
    // Column ops
    const dcl = U.el('drop-col-list'); if (dcl) {
        dcl.innerHTML = '';
        dm.columns.forEach(c => { const l = document.createElement('label'); l.className = 'form-check'; l.innerHTML = `<input type="checkbox" value="${c}" class="drop-col-check"> ${c}`; dcl.appendChild(l); });
    }
    const fct = U.el('formula-col-tags'); if (fct) {
        fct.innerHTML = dm.columns.map(c => `<span class="formula-tag" onclick="document.getElementById('calc-formula').value+=\` ${c}\`">${c}</span>`).join('');
    }
}

/* ---- Cleaning ---- */
function initCleaning() {
    U.on('btn-fill-missing', 'click', () => { const c = U.el('clean-col').value, m = U.el('clean-method').value; if (!c) { U.toast('Select column', 'warning'); return } dm.fillMissing(c, m); refreshAll(); U.toast(`Applied ${m} to ${c}`, 'success') });
    U.on('btn-remove-dupes', 'click', () => { const before = dm.data.length; dm.removeDuplicates(); refreshAll(); U.toast(`Removed ${before - dm.data.length} duplicates`, 'success') });
    U.on('btn-remove-outliers', 'click', () => { const c = U.el('outlier-col').value, m = U.el('outlier-method').value; if (!c) { U.toast('Select column', 'warning'); return } const b = dm.data.length; dm.removeOutliers(c, m); refreshAll(); U.toast(`Removed ${b - dm.data.length} outliers from ${c}`, 'success') });
    U.on('btn-convert-type', 'click', () => { const c = U.el('convert-col').value, t = U.el('convert-type').value; if (!c) return; dm.convertType(c, t); refreshAll(); U.toast(`Converted ${c} to ${t}`, 'success') });
    U.on('btn-trim', 'click', () => { const c = U.el('trim-col').value; if (!c) return; dm.trimWhitespace(c); refreshAll(); U.toast('Trimmed', 'success') });
    U.on('btn-replace-vals', 'click', () => { const c = U.el('replace-col').value, f = U.el('find-val').value, r = U.el('replace-val').value; if (!c) return; dm.replaceValues(c, f, r); refreshAll(); U.toast('Replaced', 'success') });
}

/* ---- Column Ops ---- */
function initColumnOps() {
    U.on('btn-add-calc-col', 'click', () => {
        const name = U.el('calc-col-name').value.trim(), formula = U.el('calc-formula').value.trim();
        if (!name || !formula) { U.toast('Enter name and formula', 'warning'); return }
        try { dm.addCalculatedColumn(name, formula); refreshAll(); U.toast(`Created column: ${name}`, 'success') } catch (e) { U.toast('Formula error: ' + e.message, 'error') }
    });
    U.on('btn-drop-cols', 'click', () => {
        const cols = [...document.querySelectorAll('.drop-col-check:checked')].map(cb => cb.value);
        if (!cols.length) { U.toast('Select columns', 'warning'); return }
        cols.forEach(c => dm.dropColumn(c)); refreshAll(); U.toast(`Dropped ${cols.length} columns`, 'success');
    });
    U.on('btn-rename-col', 'click', () => {
        const old = U.el('rename-col').value, nw = U.el('rename-new').value.trim();
        if (!old || !nw) { U.toast('Enter names', 'warning'); return }
        dm.renameColumn(old, nw); refreshAll(); U.toast(`Renamed ${old} → ${nw}`, 'success');
    });
}
