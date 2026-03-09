/* ============================================
   Data Manager v2 - Multi-source, Cleaning, Transform
   ============================================ */
class DataManager {
    constructor() {
        this.data = []; this.columns = []; this.columnTypes = {}; this.datasetName = '';
        this.analysisCount = 0; this.history = []; this.historyIndex = -1;
        this.target = null; this.features = [];
    }
    snapshot() { if (this.data.length > 0) { this.history.splice(this.historyIndex + 1); this.history.push(JSON.parse(JSON.stringify(this.data))); this.historyIndex = this.history.length - 1; if (this.history.length > 30) { this.history.shift(); this.historyIndex--; } } }
    undo() { if (this.historyIndex > 0) { this.historyIndex--; this.data = JSON.parse(JSON.stringify(this.history[this.historyIndex])); this.columns = Object.keys(this.data[0] || {}); this.detectColumnTypes(); return true; } return false; }
    redo() { if (this.historyIndex < this.history.length - 1) { this.historyIndex++; this.data = JSON.parse(JSON.stringify(this.history[this.historyIndex])); this.columns = Object.keys(this.data[0] || {}); this.detectColumnTypes(); return true; } return false; }

    parseCSV(text, opts = {}) {
        const d = opts.delimiter === 'auto' ? undefined : opts.delimiter;
        const h = opts.header !== false;
        if (opts.skipRows > 0) text = text.split('\n').slice(opts.skipRows).join('\n');
        const r = Papa.parse(text, { header: h, dynamicTyping: true, skipEmptyLines: true, delimiter: d });
        this.data = r.data; this.columns = r.meta.fields || Object.keys(r.data[0] || {});
        this.detectColumnTypes(); this.snapshot(); return this.data;
    }
    parseJSON(text) {
        const d = JSON.parse(text);
        this.data = Array.isArray(d) ? d : (d.data || d.records || d.results || [d]);
        this.columns = Object.keys(this.data[0] || {}); this.detectColumnTypes(); this.snapshot(); return this.data;
    }
    loadFromClipboard(text) {
        if (text.trim().startsWith('{') || text.trim().startsWith('[')) return this.parseJSON(text);
        return this.parseCSV(text);
    }
    loadFromURL(url) {
        return fetch(url).then(r => r.text()).then(t => { if (url.endsWith('.json')) return this.parseJSON(t); return this.parseCSV(t); });
    }

    detectColumnTypes() {
        this.columnTypes = {};
        this.columns.forEach(col => {
            const vals = this.getColumnValues(col).filter(v => v != null && v !== '');
            if (!vals.length) { this.columnTypes[col] = 'unknown'; return; }
            const sample = vals.slice(0, 100);
            if (sample.every(v => typeof v === 'boolean' || v === 'true' || v === 'false' || v === 0 || v === 1)) { this.columnTypes[col] = 'boolean'; return; }
            const isDate = sample.every(v => !isNaN(Date.parse(String(v))) && isNaN(Number(v)) && String(v).length > 4);
            if (isDate) { this.columnTypes[col] = 'datetime'; return; }
            const numPct = sample.filter(v => typeof v === 'number' || (!isNaN(Number(v)) && v !== '')).length / sample.length;
            if (numPct > 0.8) {
                const uniq = new Set(vals.map(Number).filter(v => !isNaN(v)));
                this.columnTypes[col] = (uniq.size <= 15 && sample.every(v => Number.isInteger(Number(v)))) ? 'discrete' : 'continuous';
            } else { this.columnTypes[col] = 'categorical'; }
        });
    }

    getColumnValues(col) { return this.data.map(r => r[col]); }
    getNumericValues(col) { return this.getColumnValues(col).map(Number).filter(v => !isNaN(v)); }
    getNumericColumns() { return this.columns.filter(c => ['continuous', 'discrete'].includes(this.columnTypes[c])); }
    getCategoricalColumns() { return this.columns.filter(c => this.columnTypes[c] === 'categorical' || this.columnTypes[c] === 'boolean'); }
    getDateColumns() { return this.columns.filter(c => this.columnTypes[c] === 'datetime'); }
    getUniqueValues(col) { return [...new Set(this.getColumnValues(col).filter(v => v != null && v !== ''))]; }
    hasData() { return this.data.length > 0; }
    exportCSV() { return Papa.unparse(this.data); }

    // Data Profiling
    profileColumn(col) {
        const vals = this.getColumnValues(col);
        const n = vals.length;
        const missing = vals.filter(v => v == null || v === '' || (typeof v === 'number' && isNaN(v))).length;
        const valid = n - missing;
        const unique = new Set(vals.filter(v => v != null && v !== '')).size;
        const missingPct = n > 0 ? missing / n : 0;
        const errorPct = missingPct;
        const qualityPct = 1 - errorPct;
        const type = this.columnTypes[col];
        const profile = { col, type, n, valid, missing, missingPct, unique, uniquePct: valid > 0 ? unique / valid : 0, errorPct, qualityPct };
        if (['continuous', 'discrete'].includes(type)) {
            const numVals = this.getNumericValues(col);
            if (numVals.length > 0) {
                const sorted = [...numVals].sort((a, b) => a - b);
                Object.assign(profile, {
                    mean: ss.mean(numVals), std: ss.standardDeviation(numVals), median: ss.median(sorted),
                    min: sorted[0], max: sorted[sorted.length - 1], q1: ss.quantile(sorted, 0.25), q3: ss.quantile(sorted, 0.75),
                    skewness: ss.sampleSkewness(numVals), kurtosis: numVals.length > 3 ? ss.sampleKurtosis(numVals) : 0,
                    zeros: numVals.filter(v => v === 0).length, negatives: numVals.filter(v => v < 0).length,
                    outliers: (() => { const iqr = ss.quantile(sorted, 0.75) - ss.quantile(sorted, 0.25); const lo = ss.quantile(sorted, 0.25) - 1.5 * iqr; const hi = ss.quantile(sorted, 0.75) + 1.5 * iqr; return numVals.filter(v => v < lo || v > hi).length; })()
                });
            }
        } else if (type === 'categorical') {
            const freqs = {};
            vals.filter(v => v != null && v !== '').forEach(v => freqs[v] = (freqs[v] || 0) + 1);
            profile.topValues = Object.entries(freqs).sort((a, b) => b[1] - a[1]).slice(0, 5);
            profile.mode = profile.topValues[0] ? profile.topValues[0][0] : null;
        }
        return profile;
    }

    profileAll() { return this.columns.map(c => this.profileColumn(c)); }

    // Cleaning
    fillMissing(col, method = 'mean') {
        this.snapshot();
        const numVals = this.getNumericValues(col);
        let fillVal;
        if (method === 'mean') fillVal = numVals.length ? ss.mean(numVals) : 0;
        else if (method === 'median') fillVal = numVals.length ? ss.median(numVals) : 0;
        else if (method === 'mode') fillVal = numVals.length ? ss.mode(numVals) : 0;
        else if (method === 'zero') fillVal = 0;
        else if (method === 'ffill') { let last = null; this.data.forEach(r => { if (r[col] != null && r[col] !== '') last = r[col]; else r[col] = last; }); return; }
        else if (method === 'bfill') { let last = null; for (let i = this.data.length - 1; i >= 0; i--) { if (this.data[i][col] != null && this.data[i][col] !== '') last = this.data[i][col]; else this.data[i][col] = last; } return; }
        else if (method === 'drop') { this.data = this.data.filter(r => r[col] != null && r[col] !== ''); return; }
        else fillVal = method;
        this.data.forEach(r => { if (r[col] == null || r[col] === '' || (typeof r[col] === 'number' && isNaN(r[col]))) r[col] = fillVal; });
    }
    removeDuplicates(cols) { this.snapshot(); const seen = new Set(); this.data = this.data.filter(r => { const key = (cols || this.columns).map(c => r[c]).join('|'); if (seen.has(key)) return false; seen.add(key); return true; }); }
    removeOutliers(col, method = 'iqr') {
        this.snapshot();
        const vals = this.getNumericValues(col);
        if (vals.length === 0) return;
        if (method === 'iqr') { const q1 = ss.quantile(vals, 0.25), q3 = ss.quantile(vals, 0.75), iqr = q3 - q1; this.data = this.data.filter(r => { const v = Number(r[col]); return !isNaN(v) && v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr; }); }
        else if (method === 'zscore') { const m = ss.mean(vals), s = ss.standardDeviation(vals); if (s === 0) return; this.data = this.data.filter(r => { const v = Number(r[col]); return !isNaN(v) && Math.abs((v - m) / s) <= 3; }); }
    }
    // Transforms
    normalize(col) {
        this.snapshot();
        const vals = this.getNumericValues(col);
        if (vals.length < 2) return;
        const min = Math.min(...vals), max = Math.max(...vals);
        if (max === min) return;
        this.data.forEach(r => { if (r[col] != null) r[col] = (Number(r[col]) - min) / (max - min); });
    }
    standardize(col) {
        this.snapshot();
        const vals = this.getNumericValues(col);
        if (vals.length < 2) return;
        const m = ss.mean(vals), s = ss.standardDeviation(vals);
        if (s === 0) return;
        this.data.forEach(r => { if (r[col] != null) r[col] = (Number(r[col]) - m) / s; });
    }
    logScale(col) {
        this.snapshot();
        this.data.forEach(r => { const v = Number(r[col]); if (!isNaN(v) && v > 0) r[col] = Math.log10(v); });
    }
    convertType(col, toType) {
        this.snapshot();
        this.data.forEach(r => {
            if (toType === 'number') r[col] = Number(r[col]) || 0;
            else if (toType === 'string') r[col] = String(r[col] ?? '');
            else if (toType === 'date') r[col] = new Date(r[col]).toISOString().slice(0, 19);
        });
        this.detectColumnTypes();
    }
    trimWhitespace(col) { this.snapshot(); this.data.forEach(r => { if (typeof r[col] === 'string') r[col] = r[col].trim(); }); }
    replaceValues(col, find, replace) { this.snapshot(); this.data.forEach(r => { if (r[col] == find || String(r[col]) === String(find)) r[col] = replace; }); }

    // Column Operations
    addCalculatedColumn(name, formula) {
        this.snapshot();
        this.data.forEach((row, idx) => {
            try {
                const ctx = { ...row, ROW: idx, ROWNUM: idx + 1, PI: Math.PI, E: Math.E, abs: Math.abs, sqrt: Math.sqrt, log: Math.log, log10: Math.log10, exp: Math.exp, pow: Math.pow, round: Math.round, ceil: Math.ceil, floor: Math.floor, min: Math.min, max: Math.max, sin: Math.sin, cos: Math.cos, tan: Math.tan };
                const fn = new Function(...Object.keys(ctx), `return ${formula}`);
                row[name] = fn(...Object.values(ctx));
            } catch (e) { row[name] = null; }
        });
        if (!this.columns.includes(name)) this.columns.push(name);
        this.detectColumnTypes();
    }
    dropColumn(col) { this.snapshot(); this.columns = this.columns.filter(c => c !== col); this.data.forEach(r => delete r[col]); delete this.columnTypes[col]; }
    renameColumn(oldName, newName) {
        this.snapshot();
        this.columns = this.columns.map(c => c === oldName ? newName : c);
        this.data.forEach(r => { r[newName] = r[oldName]; delete r[oldName]; });
        this.columnTypes[newName] = this.columnTypes[oldName]; delete this.columnTypes[oldName];
    }
    sortBy(col, asc = true) { this.snapshot(); this.data.sort((a, b) => { let va = a[col], vb = b[col]; if (typeof va === 'number' && typeof vb === 'number') return asc ? va - vb : vb - va; return asc ? String(va || '').localeCompare(String(vb || '')) : String(vb || '').localeCompare(String(va || '')); }); }
    filterData(col, op, val) { return this.data.filter(r => { const v = r[col]; switch (op) { case '==': return v == val; case '!=': return v != val; case '>': return Number(v) > Number(val); case '<': return Number(v) < Number(val); case '>=': return Number(v) >= Number(val); case '<=': return Number(v) <= Number(val); case 'contains': return String(v).toLowerCase().includes(String(val).toLowerCase()); default: return true; } }); }
    applyFilter(col, op, val) { this.snapshot(); this.data = this.filterData(col, op, val); }

    // Column stats for quick view
    getColumnStats(col) {
        const vals = this.getNumericValues(col);
        if (!vals.length) return null;
        const sorted = [...vals].sort((a, b) => a - b);
        return { n: vals.length, mean: ss.mean(vals), std: ss.standardDeviation(vals), median: ss.median(sorted), min: sorted[0], max: sorted[sorted.length - 1], q1: ss.quantile(sorted, 0.25), q3: ss.quantile(sorted, 0.75) };
    }

    // Project Save/Load
    saveProject(name) {
        const project = { name, timestamp: new Date().toISOString(), data: this.data, columns: this.columns, columnTypes: this.columnTypes, datasetName: this.datasetName, target: this.target, features: this.features, analysisCount: this.analysisCount };
        const projects = JSON.parse(localStorage.getItem('industryai_projects') || '{}');
        projects[name] = project;
        localStorage.setItem('industryai_projects', JSON.stringify(projects));
        return project;
    }
    loadProject(name) {
        const projects = JSON.parse(localStorage.getItem('industryai_projects') || '{}');
        const p = projects[name]; if (!p) return false;
        this.data = p.data; this.columns = p.columns; this.columnTypes = p.columnTypes; this.datasetName = p.datasetName; this.target = p.target; this.features = p.features || []; this.analysisCount = p.analysisCount || 0;
        this.snapshot(); return true;
    }
    listProjects() { return Object.values(JSON.parse(localStorage.getItem('industryai_projects') || '{}')); }
    deleteProject(name) { const p = JSON.parse(localStorage.getItem('industryai_projects') || '{}'); delete p[name]; localStorage.setItem('industryai_projects', JSON.stringify(p)); }
    exportProject() { return { data: this.data, columns: this.columns, columnTypes: this.columnTypes, datasetName: this.datasetName, target: this.target, features: this.features }; }
    importProject(json) { const p = JSON.parse(json); this.data = p.data; this.columns = p.columns; this.columnTypes = p.columnTypes; this.datasetName = p.datasetName; this.target = p.target; this.features = p.features || []; this.snapshot(); }
}

// Sample data generators
function generateManufacturingData() {
    const data = []; const machines = ['Machine A', 'Machine B', 'Machine C', 'Machine D']; const operators = ['Op-1', 'Op-2', 'Op-3', 'Op-4', 'Op-5']; const shifts = ['Morning', 'Afternoon', 'Night']; const materials = ['Steel-A', 'Steel-B', 'Aluminum', 'Composite'];
    for (let i = 0; i < 500; i++) {
        const machine = machines[Math.floor(Math.random() * machines.length)]; const baseT = machine === 'Machine A' ? 180 : machine === 'Machine B' ? 195 : machine === 'Machine C' ? 170 : 185;
        const temp = baseT + (Math.random() - .5) * 20 + Math.sin(i / 50) * 5; const pressure = (machine === 'Machine A' ? 45 : machine === 'Machine B' ? 50 : 42) + (Math.random() - .5) * 10;
        const speed = 1200 + (Math.random() - .5) * 200; const vib = 0.5 + Math.random() * 2.5 + (temp > 195 ? 1 : 0);
        const thick = 2.5 + (Math.random() - .5) * .3 + (pressure - 45) * .01; const rough = 0.8 + Math.random() * 1.2 + vib * .1;
        const yld = 85 + Math.random() * 12 - (vib > 2 ? 5 : 0); const defects = Math.max(0, Math.floor(Math.random() * 5 + (vib > 2.5 ? 3 : 0)));
        data.push({ Batch_ID: `B${String(i + 1).padStart(4, '0')}`, Machine: machine, Operator: operators[Math.floor(Math.random() * operators.length)], Shift: shifts[Math.floor(Math.random() * shifts.length)], Material: materials[Math.floor(Math.random() * materials.length)], Temperature: U.round(temp, 1), Pressure: U.round(pressure, 1), Speed_RPM: Math.round(speed), Vibration: U.round(vib, 2), Thickness: U.round(thick, 3), Roughness: U.round(rough, 2), Yield_Pct: U.round(yld, 1), Defects: defects });
    }
    return { data, name: 'Manufacturing Process Data' };
}
function generateQualityData() {
    const data = []; const types = ['Scratch', 'Dent', 'Crack', 'Discoloration', 'Misalignment', 'Burr', 'Contamination']; const lines = ['Line 1', 'Line 2', 'Line 3'];
    for (let i = 0; i < 300; i++) {
        const line = lines[Math.floor(Math.random() * lines.length)]; const sz = 50 + Math.floor(Math.random() * 50); const rate = line === 'Line 1' ? .03 : line === 'Line 2' ? .06 : .04;
        data.push({ Sample_ID: i + 1, Production_Line: line, Inspector: ['Inspector A', 'Inspector B', 'Inspector C'][Math.floor(Math.random() * 3)], Sample_Size: sz, Defective: Math.max(0, Math.floor(sz * (rate + (Math.random() - .5) * .04))), Defect_Type: types[Math.floor(Math.random() * types.length)], Severity: ['Minor', 'Major', 'Critical'][Math.floor(Math.random() * 3)], Pass_Fail: Math.random() > .15 ? 'Pass' : 'Fail' });
    }
    return { data, name: 'Quality Control Data' };
}
function generateTimeSeriesData() {
    const data = []; let t = 175, p = 45, f = 120, pw = 250; const start = new Date('2025-01-01');
    for (let i = 0; i < 1000; i++) {
        const d = new Date(start); d.setHours(d.getHours() + i);
        t += (Math.random() - .5) * 3 + Math.sin(i / 24 * Math.PI * 2) * 2; p += (Math.random() - .5) * 1.5; f += (Math.random() - .5) * 5 + Math.cos(i / 48 * Math.PI * 2) * 3; pw += (Math.random() - .5) * 10;
        if (Math.random() > .98) t += 15; if (Math.random() > .99) p -= 10;
        data.push({ Timestamp: d.toISOString().slice(0, 19).replace('T', ' '), Temperature: U.round(t, 1), Pressure: U.round(p, 1), Flow_Rate: U.round(Math.max(80, f), 1), Power_kW: U.round(Math.max(180, pw), 1), Status: t > 200 || p > 55 || p < 35 ? 'Alarm' : 'Normal' });
    }
    return { data, name: 'Sensor Time Series Data' };
}
function generateExperimentData() {
    const data = []; const factors = [-1, 1]; let idx = 0;
    for (let a of factors) for (let b of factors) for (let c of factors) for (let rep = 0; rep < 4; rep++) {
        data.push({ Run: ++idx, Factor_A: a === -1 ? 'Low' : 'High', Factor_B: b === -1 ? 'Low' : 'High', Factor_C: c === -1 ? 'Low' : 'High', Response: U.round(40 + 8 * a + 3 * b - 5 * c + 2 * a * b - 1.5 * a * c + (Math.random() - .5) * 4, 2) });
    }
    return { data, name: 'DOE Experiment Data' };
}
function generateClassificationData() {
    const data = [];
    for (let i = 0; i < 400; i++) {
        const age = 20 + Math.floor(Math.random() * 40); const exp = Math.max(0, age - 20 - Math.floor(Math.random() * 5)); const rating = U.round(1 + Math.random() * 4, 1); const hours = 30 + Math.floor(Math.random() * 20); const projects = Math.floor(Math.random() * 15);
        const score = age * .01 + exp * .05 + rating * .2 + hours * .01 + projects * .03 + Math.random() * .3;
        data.push({ Employee_ID: `E${String(i + 1).padStart(3, '0')}`, Age: age, Experience_Years: exp, Performance_Rating: rating, Weekly_Hours: hours, Projects_Completed: projects, Department: ['Engineering', 'Sales', 'HR', 'Marketing'][Math.floor(Math.random() * 4)], Promoted: score > 0.65 ? 'Yes' : 'No' });
    }
    return { data, name: 'Employee Classification Data' };
}
