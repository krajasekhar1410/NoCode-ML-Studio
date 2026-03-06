/* ============================================
   App Controller - Part 2: Viz, Stats, ML, Project
   ============================================ */

/* ---- Viz Builder ---- */
function initVizBuilder() {
    document.querySelectorAll('.chart-type-btn').forEach(btn => btn.addEventListener('click', () => { document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active'); }));
    U.on('btn-update-viz', 'click', buildViz);
    U.on('btn-clear-viz', 'click', () => { dragDrop.clearAll(); viz.destroyChart('viz-canvas'); U.el('viz-canvas').style.display = 'none'; U.el('viz-placeholder').style.display = '' });
    U.on('btn-download-viz', 'click', () => { const c = U.el('viz-canvas'); const a = document.createElement('a'); a.href = c.toDataURL('image/png'); a.download = 'chart.png'; a.click() });
}
function buildViz() {
    if (!dm.hasData()) { U.toast('Load data first', 'warning'); return }
    const m = dragDrop.getMappings(), ct = document.querySelector('.chart-type-btn.active')?.dataset.type || 'scatter';
    const opts = { title: U.el('viz-title').value, palette: U.el('viz-palette').value, pointSize: +U.el('viz-point-size').value, trendline: U.el('viz-trendline').checked, smooth: U.el('viz-smooth').checked };
    U.el('viz-placeholder').style.display = 'none'; U.el('viz-canvas').style.display = '';
    try {
        if (ct === 'scatter') { if (!m.x || !m.y) { U.toast('Need X and Y', 'warning'); return } viz.scatter('viz-canvas', dm.getNumericValues(m.x), dm.getNumericValues(m.y), { ...opts, xLabel: m.x, yLabel: m.y }) }
        else if (ct === 'line') { if (!m.x || !m.y) { U.toast('Need X and Y', 'warning'); return } viz.line('viz-canvas', dm.getColumnValues(m.x), [{ label: m.y, data: dm.getNumericValues(m.y) }], { ...opts, xLabel: m.x, yLabel: m.y }) }
        else if (ct === 'bar') { if (!m.x) { U.toast('Need X', 'warning'); return } const cats = dm.getUniqueValues(m.x); const vals = m.y ? cats.map(c => { const rows = dm.data.filter(r => r[m.x] === c); return ss.mean(rows.map(r => Number(r[m.y])).filter(v => !isNaN(v))) }) : cats.map(c => dm.data.filter(r => r[m.x] === c).length); viz.bar('viz-canvas', cats, vals, { ...opts, xLabel: m.x, yLabel: m.y ? `Mean ${m.y}` : 'Count' }) }
        else if (ct === 'histogram') { const v = m.x || m.y; if (!v) { U.toast('Need a variable', 'warning'); return } viz.histogram('viz-canvas', dm.getNumericValues(v), { ...opts, xLabel: v }) }
        else if (ct === 'box') { if (!m.y) { U.toast('Need Y', 'warning'); return } if (m.x) { const cats = dm.getUniqueValues(m.x); viz.boxPlot('viz-canvas', cats.map(c => dm.data.filter(r => r[m.x] === c).map(r => Number(r[m.y])).filter(v => !isNaN(v))), cats.map(String), { ...opts, xLabel: m.x, yLabel: m.y }) } else { viz.boxPlot('viz-canvas', [dm.getNumericValues(m.y)], [m.y], { ...opts, yLabel: m.y }) } }
        else if (ct === 'pie' || ct === 'doughnut') { if (!m.x) { U.toast('Need X', 'warning'); return } const cats = dm.getUniqueValues(m.x); viz.pie('viz-canvas', cats, cats.map(c => dm.data.filter(r => r[m.x] === c).length), { ...opts, doughnut: ct === 'doughnut' }) }
        else if (ct === 'area') { if (!m.x || !m.y) { U.toast('Need X and Y', 'warning'); return } viz.line('viz-canvas', dm.getColumnValues(m.x), [{ label: m.y, data: dm.getNumericValues(m.y) }], { ...opts, xLabel: m.x, yLabel: m.y, fill: true }) }
        else if (ct === 'bubble') { if (!m.x || !m.y || !m.size) { U.toast('Need X, Y, Size', 'warning'); return } const x = dm.getNumericValues(m.x), y = dm.getNumericValues(m.y), s = dm.getNumericValues(m.size); const maxS = Math.max(...s); viz.bubble('viz-canvas', x.map((xi, i) => ({ x: xi, y: y[i], r: 3 + (s[i] / maxS) * 15 })), { ...opts, xLabel: m.x, yLabel: m.y }) }
        else U.toast('Chart type not supported yet', 'info');
        dm.analysisCount++; U.toast('Visualization updated', 'success');
    } catch (e) { U.toast('Error: ' + e.message, 'error') }
}

/* ---- Gallery ---- */
function initGallery() {
    const grid = U.el('gallery-grid'); if (!grid) return;

    const CHARTS = [
        // 1. Basic Statistical
        { name: 'Histogram', cat: 'basic', icon: 'fa-signal', color: '#3b82f6', desc: 'Frequency distribution', type: 'histogram', live: true },
        { name: 'Box Plot', cat: 'basic', icon: 'fa-square', color: '#8b5cf6', desc: 'Quartile spread', type: 'box', live: true },
        { name: 'Density Plot', cat: 'basic', icon: 'fa-water', color: '#06b6d4', desc: 'KDE smooth distribution', type: 'line', live: true },
        { name: 'Dot Plot', cat: 'basic', icon: 'fa-ellipsis-h', color: '#10b981', desc: 'Individual data points', type: 'scatter', live: true },
        { name: 'Probability Plot', cat: 'basic', icon: 'fa-chart-line', color: '#f59e0b', desc: 'Normal Q-Q plot', type: 'scatter', live: true },
        { name: 'Stem-and-Leaf', cat: 'basic', icon: 'fa-list', color: '#ec4899', desc: 'Textual data distribution', type: 'bar', live: false },
        // 2. Time Series
        { name: 'Time Series Plot', cat: 'timeseries', icon: 'fa-chart-line', color: '#3b82f6', desc: 'Sequential observations', type: 'line', live: true },
        { name: 'Multi-Line Series', cat: 'timeseries', icon: 'fa-stream', color: '#06b6d4', desc: 'Multiple trend lines', type: 'line', live: true },
        { name: 'Moving Average', cat: 'timeseries', icon: 'fa-wave-square', color: '#8b5cf6', desc: 'Smoothed trend line', type: 'line', live: true },
        { name: 'Seasonal Decomp.', cat: 'timeseries', icon: 'fa-calendar-alt', color: '#10b981', desc: 'Trend + seasonality + residual', type: 'line', live: false },
        { name: 'Rolling Statistics', cat: 'timeseries', icon: 'fa-redo', color: '#f59e0b', desc: 'Rolling mean & std band', type: 'line', live: false },
        { name: 'Lag Plot', cat: 'timeseries', icon: 'fa-project-diagram', color: '#ef4444', desc: 'Auto-correlation lag', type: 'scatter', live: false },
        { name: 'Cross-Correlation', cat: 'timeseries', icon: 'fa-exchange-alt', color: '#ec4899', desc: 'Two series cross-corr', type: 'bar', live: false },
        // 3. Relationship
        { name: 'Scatter Plot', cat: 'relationship', icon: 'fa-braille', color: '#3b82f6', desc: 'X vs Y with trendline', type: 'scatter', live: true },
        { name: 'Bubble Chart', cat: 'relationship', icon: 'fa-circle', color: '#06b6d4', desc: '3D relationship via size', type: 'bubble', live: true },
        { name: 'Correlation Heatmap', cat: 'relationship', icon: 'fa-th', color: '#f97316', desc: 'Feature correlation matrix', type: 'heatmap', live: true, route: 'correlation' },
        { name: 'Scatter Matrix', cat: 'relationship', icon: 'fa-th-large', color: '#8b5cf6', desc: 'All-pairs scatter grid', type: 'scatter', live: false },
        { name: 'Pair Plot', cat: 'relationship', icon: 'fa-border-all', color: '#10b981', desc: 'Pairwise grid with KDE', type: 'scatter', live: false },
        // 4. SPC
        { name: 'X̄ Chart', cat: 'spc', icon: 'fa-wave-square', color: '#3b82f6', desc: 'Sample mean control', type: 'control', live: true, route: 'control-charts' },
        { name: 'X̄-R Chart', cat: 'spc', icon: 'fa-wave-square', color: '#8b5cf6', desc: 'Mean & Range chart', type: 'control', live: true, route: 'control-charts' },
        { name: 'X̄-S Chart', cat: 'spc', icon: 'fa-wave-square', color: '#06b6d4', desc: 'Mean & Std Dev chart', type: 'control', live: true, route: 'control-charts' },
        { name: 'I-MR Chart', cat: 'spc', icon: 'fa-wave-square', color: '#10b981', desc: 'Individual & Moving Range', type: 'control', live: true, route: 'control-charts' },
        { name: 'P Chart', cat: 'spc', icon: 'fa-percentage', color: '#f59e0b', desc: 'Proportion defective', type: 'bar', live: false },
        { name: 'NP Chart', cat: 'spc', icon: 'fa-hashtag', color: '#ef4444', desc: 'Number defective', type: 'bar', live: false },
        { name: 'C Chart', cat: 'spc', icon: 'fa-bug', color: '#ec4899', desc: 'Count of defects', type: 'line', live: false },
        { name: 'U Chart', cat: 'spc', icon: 'fa-divide', color: '#f97316', desc: 'Defects per unit', type: 'line', live: false },
        // 5. Quality
        { name: 'Pareto Chart', cat: 'quality', icon: 'fa-sort-amount-down', color: '#3b82f6', desc: '80/20 rule analysis', type: 'pareto', live: true, route: 'pareto' },
        { name: 'Fishbone Diagram', cat: 'quality', icon: 'fa-fish', color: '#ef4444', desc: 'Cause & Effect (Ishikawa)', type: 'fishbone', live: false },
        { name: 'Capability Hist.', cat: 'quality', icon: 'fa-bullseye', color: '#8b5cf6', desc: 'Process capability plot', type: 'histogram', live: true, route: 'capability' },
        { name: 'Cp / Cpk Plot', cat: 'quality', icon: 'fa-tachometer-alt', color: '#10b981', desc: 'Capability index gauges', type: 'bar', live: true, route: 'capability' },
        // 6. Multivariate
        { name: 'PCA Score Plot', cat: 'multivariate', icon: 'fa-compress-arrows-alt', color: '#3b82f6', desc: 'PC1 vs PC2 scores', type: 'scatter', live: false },
        { name: 'PCA Loading Plot', cat: 'multivariate', icon: 'fa-arrows-alt', color: '#8b5cf6', desc: 'Variable loadings', type: 'scatter', live: false },
        { name: 'Biplot', cat: 'multivariate', icon: 'fa-expand-arrows-alt', color: '#06b6d4', desc: 'Scores + loadings', type: 'scatter', live: false },
        { name: 'Cluster Dendrogram', cat: 'multivariate', icon: 'fa-sitemap', color: '#10b981', desc: 'Hierarchical clustering', type: 'bar', live: false },
        { name: 'Hotelling T² Chart', cat: 'multivariate', icon: 'fa-chart-area', color: '#f59e0b', desc: 'Multivariate control', type: 'line', live: false },
        // 7. ML & AI
        { name: 'Feature Importance', cat: 'ml', icon: 'fa-chart-bar', color: '#8b5cf6', desc: 'Ranked feature impact', type: 'bar', live: true, route: 'ml-results' },
        { name: 'Residual Plot', cat: 'ml', icon: 'fa-project-diagram', color: '#ef4444', desc: 'Residuals vs fitted', type: 'scatter', live: true, route: 'regression' },
        { name: 'Pred vs Actual', cat: 'ml', icon: 'fa-equals', color: '#10b981', desc: 'Model prediction quality', type: 'scatter', live: true, route: 'ml-results' },
        { name: 'SHAP Summary Plot', cat: 'ml', icon: 'fa-layer-group', color: '#f97316', desc: 'SHAP feature attribution', type: 'bar', live: false },
        { name: 'SHAP Dependence', cat: 'ml', icon: 'fa-bezier-curve', color: '#ec4899', desc: 'SHAP vs feature value', type: 'scatter', live: false },
        { name: 'Partial Dependence', cat: 'ml', icon: 'fa-chart-line', color: '#06b6d4', desc: 'PDP marginal effect', type: 'line', live: false },
        // 8. Time-Series AI
        { name: 'Lag Correlation', cat: 'tsai', icon: 'fa-clock', color: '#3b82f6', desc: 'ACF/PACF lags', type: 'bar', live: true, route: 'timeseries' },
        { name: 'Forecast vs Actual', cat: 'tsai', icon: 'fa-chart-line', color: '#10b981', desc: 'Predicted vs ground truth', type: 'line', live: true, route: 'forecasting' },
        { name: 'Change Point Detect.', cat: 'tsai', icon: 'fa-exclamation', color: '#ef4444', desc: 'Structural break detection', type: 'line', live: false },
        { name: 'Anomaly Timeline', cat: 'tsai', icon: 'fa-radiation', color: '#f97316', desc: 'Anomalies on time axis', type: 'scatter', live: false },
        { name: 'Seasonal Pattern', cat: 'tsai', icon: 'fa-sun', color: '#f59e0b', desc: 'Seasonal cycle plot', type: 'line', live: false },
        // 9. Causal
        { name: 'Causal Graph', cat: 'causal', icon: 'fa-network-wired', color: '#8b5cf6', desc: 'Causal network diagram', type: 'scatter', live: false },
        { name: 'Root Cause Tree', cat: 'causal', icon: 'fa-tree', color: '#10b981', desc: 'Hierarchy of causes', type: 'bar', live: false },
        { name: 'Event Impact Chart', cat: 'causal', icon: 'fa-bolt', color: '#f59e0b', desc: 'Before/after event', type: 'line', live: false },
        { name: 'Causal Effect Plot', cat: 'causal', icon: 'fa-arrows-left-right', color: '#ef4444', desc: 'Treatment effect size', type: 'scatter', live: false },
        // 10. Industrial
        { name: 'Process Heatmap', cat: 'industrial', icon: 'fa-th', color: '#f97316', desc: 'Process stage heatmap', type: 'heatmap', live: false },
        { name: 'Sankey Flow Diagram', cat: 'industrial', icon: 'fa-project-diagram', color: '#3b82f6', desc: 'Flow / energy balance', type: 'bar', live: false },
        { name: 'Sensor Network', cat: 'industrial', icon: 'fa-broadcast-tower', color: '#06b6d4', desc: 'Sensor influence graph', type: 'scatter', live: false },
        { name: 'Process Flow Diag.', cat: 'industrial', icon: 'fa-sitemap', color: '#8b5cf6', desc: 'Interactive PFD', type: 'bar', live: false },
        { name: 'Digital Twin Map', cat: 'industrial', icon: 'fa-robot', color: '#10b981', desc: 'Virtual process model', type: 'scatter', live: false },
        // 11. Dashboards
        { name: 'KPI Dashboard', cat: 'dashboard', icon: 'fa-tachometer-alt', color: '#3b82f6', desc: 'Key performance indicators', type: 'bar', live: true, route: 'dashboard' },
        { name: 'Production Trend', cat: 'dashboard', icon: 'fa-industry', color: '#10b981', desc: 'Production KPIs over time', type: 'line', live: false },
        { name: 'Energy Dashboard', cat: 'dashboard', icon: 'fa-bolt', color: '#f59e0b', desc: 'Energy consumption trends', type: 'line', live: false },
        { name: 'Quality Monitor', cat: 'dashboard', icon: 'fa-check-double', color: '#8b5cf6', desc: 'Quality metrics summary', type: 'bar', live: false },
        { name: 'Alarm Timeline', cat: 'dashboard', icon: 'fa-bell', color: '#ef4444', desc: 'Event & alarm history', type: 'scatter', live: false },
        // 12. Advanced AI
        { name: 'Dynamic Lag Network', cat: 'advanced', icon: 'fa-project-diagram', color: '#8b5cf6', desc: 'Time-lag dependency graph', type: 'scatter', live: false },
        { name: 'Sensor Dependency', cat: 'advanced', icon: 'fa-link', color: '#06b6d4', desc: 'Sensor correlation map', type: 'scatter', live: false },
        { name: 'Feature Drift Chart', cat: 'advanced', icon: 'fa-wind', color: '#f97316', desc: 'Feature distribution shift', type: 'line', live: false },
        { name: 'Stability Radar', cat: 'advanced', icon: 'fa-broadcast-tower', color: '#10b981', desc: 'Process stability spider', type: 'bar', live: false },
        { name: 'AI Insight Dashboard', cat: 'advanced', icon: 'fa-brain', color: '#ec4899', desc: 'Auto-generated AI insights', type: 'bar', live: true, route: 'quick-insights' },
    ];

    const catLabels = {
        basic: '📊 Basic Statistical', timeseries: '📈 Time Series', relationship: '🔗 Relationship',
        spc: '⚙️ SPC', quality: '🏭 Quality', multivariate: '🧬 Multivariate',
        ml: '🤖 ML & AI', tsai: '⏱ Time-Series AI', causal: '🔍 Causal',
        industrial: '🏗 Industrial', dashboard: '📋 Dashboards', advanced: '🚀 Advanced AI'
    };

    CHARTS.forEach((c, i) => {
        const el = document.createElement('div');
        el.className = 'gallery-item';
        el.dataset.category = c.cat;
        const badge = c.live
            ? `<span class="gallery-badge live">Live</span>`
            : `<span class="gallery-badge soon">Coming Soon</span>`;
        el.innerHTML = `
            <div class="gallery-item-preview" id="galp-${i}">
                <canvas id="gal-${i}" style="display:none"></canvas>
                <div class="gallery-icon-placeholder" id="galph-${i}" style="background:${c.color}22;color:${c.color}">
                    <i class="fas ${c.icon}" style="font-size:28px"></i>
                </div>
            </div>
            <div class="gallery-item-info">
                <h4><i class="fas ${c.icon}"></i> ${c.name}</h4>
                <p style="font-size:10px;color:var(--text-muted);margin:2px 0 4px">${c.desc}</p>
                <div style="display:flex;justify-content:space-between;align-items:center">${badge}
                    <span style="font-size:9px;color:var(--text-muted)">${catLabels[c.cat] || c.cat}</span>
                </div>
            </div>`;
        el.addEventListener('click', () => {
            if (c.route) { navigateTo(c.route); return; }
            navigateTo('viz-builder');
            document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.toggle('active', b.dataset.type === c.type));
        });
        grid.appendChild(el);
    });

    // Render live mini-chart previews
    setTimeout(() => {
        const sd = Array.from({ length: 20 }, (_, i) => 20 + Math.random() * 60 + Math.sin(i / 3) * 15);
        const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
        CHARTS.forEach((c, i) => {
            const cv = U.el(`gal-${i}`);
            const ph = U.el(`galph-${i}`);
            if (!cv || !c.live) return;
            cv.style.display = '';
            if (ph) ph.style.display = 'none';
            let cfg;
            const labels = sd.map((_, j) => j + 1);
            const col = colors[i % colors.length];
            if (c.type === 'scatter') cfg = { type: 'scatter', data: { datasets: [{ data: sd.map((v, j) => ({ x: j * 5, y: v })), backgroundColor: col + '99', pointRadius: 3 }] } };
            else if (c.type === 'bubble') cfg = { type: 'bubble', data: { datasets: [{ data: sd.slice(0, 10).map((v, j) => ({ x: j * 10, y: v, r: 3 + Math.random() * 6 })), backgroundColor: col + '88' }] } };
            else if (c.type === 'pie' || c.type === 'doughnut') cfg = { type: c.type, data: { labels: ['A', 'B', 'C', 'D'], datasets: [{ data: [30, 25, 20, 25], backgroundColor: colors.slice(0, 4).map(x => x + 'cc') }] } };
            else if (c.type === 'bar') cfg = { type: 'bar', data: { labels: labels.slice(0, 8), datasets: [{ data: sd.slice(0, 8), backgroundColor: col + 'bb', borderColor: col, borderWidth: 1 }] } };
            else cfg = { type: 'line', data: { labels, datasets: [{ data: sd, borderColor: col, borderWidth: 2, pointRadius: 0, tension: .4, fill: c.type === 'area', backgroundColor: col + '22' }] } };
            cfg.options = {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { legend: { display: false } },
                scales: (c.type === 'pie' || c.type === 'doughnut') ? {} : { x: { display: false }, y: { display: false } }
            };
            try { new Chart(cv, cfg); } catch (e) { cv.style.display = 'none'; if (ph) ph.style.display = ''; }
        });
    }, 300);

    // Filter buttons
    document.querySelectorAll('.gallery-filter').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('.gallery-filter').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        const f = b.dataset.filter;
        document.querySelectorAll('.gallery-item').forEach(it => it.style.display = f === 'all' || it.dataset.category === f ? '' : 'none');
    }));
}


/* ---- Statistics Pages ---- */
function initDescriptive() {
    U.on('btn-run-descriptive', 'click', () => {
        const cols = [...document.querySelectorAll('#desc-var-list input:checked')].map(c => c.value); if (!cols.length) { U.toast('Select variables', 'warning'); return }
        let h = ''; cols.forEach(col => { const s = StatisticsEngine.descriptive(dm.getNumericValues(col)); h += `<div class="result-section"><h4><i class="fas fa-calculator"></i> ${col}</h4><table class="result-table"><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>` + Object.entries({ N: s.n, Mean: U.fmt(s.mean), StdDev: U.fmt(s.std), Variance: U.fmt(s.variance), Median: U.fmt(s.median), Min: U.fmt(s.min), Max: U.fmt(s.max), Range: U.fmt(s.range), Q1: U.fmt(s.q1), Q3: U.fmt(s.q3), IQR: U.fmt(s.iqr), Skewness: U.fmt(s.skewness), Kurtosis: U.fmt(s.kurtosis), 'CV%': U.fmt(s.cv, 2) + '%', '95% CI': U.fmt(s.ci95[0]) + ' – ' + U.fmt(s.ci95[1]) }).map(([k, v]) => `<tr><td>${k}</td><td class="result-value">${v}</td></tr>`).join('') + `</tbody></table><div class="result-chart"><canvas id="dh-${col.replace(/\W/g, '_')}"></canvas></div></div>` });
        U.html('desc-results', h); dm.analysisCount++; cols.forEach(col => viz.histogram(`dh-${col.replace(/\W/g, '_')}`, dm.getNumericValues(col), { title: `Distribution: ${col}`, xLabel: col }));
    })
}
function initHypothesis() {
    U.on('hyp-test-type', 'change', e => { const t = e.target.value; U.el('hyp-var2-group').style.display = ['2-sample-t', 'paired-t', 'chi-square', 'f-test'].includes(t) ? '' : 'none'; U.el('hyp-mu-group').style.display = t === '1-sample-t' ? '' : 'none' });
    U.on('btn-run-hypothesis', 'click', () => { if (!dm.hasData()) { U.toast('Load data', 'warning'); return } const t = U.el('hyp-test-type').value, v1 = U.el('hyp-var1').value; if (!v1) { U.toast('Select variable', 'warning'); return } let r; try { if (t === '1-sample-t') r = StatisticsEngine.oneSampleTTest(dm.getNumericValues(v1), parseFloat(U.el('hyp-mu').value) || 0, U.el('hyp-alternative').value); else if (t === '2-sample-t') { const v2 = U.el('hyp-var2').value; if (!v2) return; r = StatisticsEngine.twoSampleTTest(dm.getNumericValues(v1), dm.getNumericValues(v2), U.el('hyp-alternative').value) } else if (t === 'normality') r = StatisticsEngine.normalityTest(dm.getNumericValues(v1)); else if (t === 'f-test') { const v2 = U.el('hyp-var2').value; if (!v2) return; r = StatisticsEngine.fTest(dm.getNumericValues(v1), dm.getNumericValues(v2)) } else { U.toast('Not implemented', 'info'); return } U.html('hyp-results', `<div class="result-section"><h4><i class="fas fa-flask"></i> ${r.testName}</h4><table class="result-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>${Object.entries(r).filter(([k]) => !['testName', 'conclusion', 'significant'].includes(k)).map(([k, v]) => `<tr><td>${k}</td><td class="result-value">${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`).join('')}</tbody></table><div class="result-summary"><strong>Conclusion:</strong> ${r.conclusion}<br><strong>Significant:</strong> <span class="${r.significant ? 'result-significant' : 'result-not-significant'}">${r.significant ? 'Yes' : 'No'}</span></div></div>`); dm.analysisCount++ } catch (e) { U.toast('Error: ' + e.message, 'error') } });
}
function initRegression() {
    U.on('reg-type', 'change', e => U.el('reg-poly-group').style.display = e.target.value === 'polynomial' ? '' : 'none');
    U.on('btn-run-regression', 'click', () => { if (!dm.hasData()) return; const resp = U.el('reg-response').value, pred = U.el('reg-predictor').value; if (!resp || !pred) { U.toast('Select variables', 'warning'); return } const y = dm.getNumericValues(resp), x = dm.getNumericValues(pred), n = Math.min(x.length, y.length); try { const t = U.el('reg-type').value; let r; if (t === 'polynomial') r = StatisticsEngine.polynomialRegression(x.slice(0, n), y.slice(0, n), +U.el('reg-degree').value); else r = StatisticsEngine.linearRegression(x.slice(0, n), y.slice(0, n)); let h = `<div class="result-section"><h4><i class="fas fa-project-diagram"></i> ${r.equation}</h4><div class="result-summary"><strong>R² = ${r.rSquared.toFixed(4)}</strong>${r.adjR2 ? ` · Adj R² = ${r.adjR2.toFixed(4)}` : ''} · RMSE = ${r.rmse.toFixed(4)}</div>`; if (r.coefficients && r.coefficients[0].se) h += `<table class="result-table"><thead><tr><th>Term</th><th>Estimate</th><th>SE</th><th>t</th><th>P</th></tr></thead><tbody>${r.coefficients.map(c => `<tr><td>${c.term}</td><td class="result-value">${c.est.toFixed(4)}</td><td>${c.se.toFixed(4)}</td><td>${c.t.toFixed(3)}</td><td class="${c.p < .05 ? 'result-significant' : 'result-not-significant'}">${c.p.toFixed(4)}</td></tr>`).join('')}</tbody></table>`; h += `<div class="result-chart"><canvas id="reg-fit"></canvas></div><div class="result-chart"><canvas id="reg-res"></canvas></div></div>`; U.html('reg-results', h); dm.analysisCount++; setTimeout(() => { viz.scatter('reg-fit', x.slice(0, n), y.slice(0, n), { title: 'Fitted Line', xLabel: pred, yLabel: resp, trendline: true }); if (r.residuals) viz.scatter('reg-res', r.yHat, r.residuals, { title: 'Residuals vs Fitted', xLabel: 'Fitted', yLabel: 'Residual' }) }, 100) } catch (e) { U.toast('Error: ' + e.message, 'error') } });
}
function initAnova() { U.on('btn-run-anova', 'click', () => { if (!dm.hasData()) return; const resp = U.el('anova-response').value, f1 = U.el('anova-factor1').value; if (!resp || !f1) return; const cats = dm.getUniqueValues(f1), groups = cats.map(c => dm.data.filter(r => r[f1] === c).map(r => Number(r[resp])).filter(v => !isNaN(v))); const r = StatisticsEngine.oneWayAnova(groups); let h = `<div class="result-section"><h4><i class="fas fa-layer-group"></i> ANOVA Table</h4><table class="result-table"><thead><tr><th>Source</th><th>SS</th><th>DF</th><th>MS</th><th>F</th><th>P</th></tr></thead><tbody>${r.table.map(t => `<tr><td>${t.source}</td><td>${typeof t.ss === 'number' ? t.ss.toFixed(2) : t.ss}</td><td>${t.df}</td><td>${typeof t.ms === 'number' ? t.ms.toFixed(2) : ''}</td><td>${typeof t.f === 'number' ? t.f.toFixed(3) : ''}</td><td class="${typeof t.p === 'number' && t.p < .05 ? 'result-significant' : ''}">${typeof t.p === 'number' ? t.p.toFixed(4) : ''}</td></tr>`).join('')}</tbody></table><div class="result-summary">${r.conclusion}</div><div class="result-chart"><canvas id="anova-box"></canvas></div></div>`; U.html('anova-results', h); dm.analysisCount++; setTimeout(() => viz.boxPlot('anova-box', groups, cats.map(String), { title: `${resp} by ${f1}`, xLabel: f1, yLabel: resp }), 100) }) }
function initCorrelation() {
    // When target changes, rebuild the feature checkboxes (exclude target itself)
    U.on('corr-target', 'change', buildCorrFeatureList);
    U.on('btn-run-correlation', 'click', runTargetCorrelation);
}
function buildCorrFeatureList() {
    const target = U.el('corr-target')?.value;
    const numCols = dm.getNumericColumns().filter(c => c !== target);
    const list = U.el('corr-var-list');
    if (!list) return;
    list.innerHTML = numCols.map(c =>
        `<div class="form-check"><input type="checkbox" class="corr-feature-check" value="${c}" id="cfc-${c.replace(/\W/g, '_')}" checked>
         <label for="cfc-${c.replace(/\W/g, '_')}" style="font-size:12px">${c}</label></div>`
    ).join('');
}
window.buildCorrFeatureList = buildCorrFeatureList;
function runTargetCorrelation() {
    if (!dm.hasData()) { U.toast('Load data first', 'warning'); return; }
    const target = U.el('corr-target')?.value;
    if (!target) { U.toast('Select a target variable', 'warning'); return; }
    const features = [...document.querySelectorAll('.corr-feature-check:checked')].map(c => c.value);
    if (!features.length) { U.toast('Select at least one feature', 'warning'); return; }

    const yVals = dm.getNumericValues(target);

    // Compute correlation of each feature vs target
    const results = features.map(f => {
        const xVals = dm.getNumericValues(f);
        const n = Math.min(xVals.length, yVals.length);
        const c = StatisticsEngine.correlation(xVals.slice(0, n), yVals.slice(0, n));
        return { feature: f, r: c.r, r2: c.rSquared, p: c.pValue, strength: c.strength, direction: c.direction, significant: c.significant, xVals: xVals.slice(0, n), yVals: yVals.slice(0, n) };
    }).sort((a, b) => Math.abs(b.r) - Math.abs(a.r)); // Sort by |r| descending

    // Build HTML
    let h = '';

    // 1. Summary bar chart
    h += `<div class="result-section"><h4><i class="fas fa-chart-bar"></i> Correlation with <strong style="color:var(--accent)">${target}</strong> — Ranked</h4>
        <div class="result-chart" style="height:${Math.max(200, results.length * 40)}px"><canvas id="corr-bar-chart"></canvas></div></div>`;

    // 2. Heatmap row (single row showing each feature r vs target)
    h += `<div class="result-section"><h4><i class="fas fa-th"></i> Heatmap — Features vs ${target}</h4>
        <div class="heatmap-container"><table class="heatmap-table"><thead><tr><th>${target}</th>${results.map(r => `<th title="${r.r.toFixed(4)}">${r.feature}</th>`).join('')}</tr></thead>
        <tbody><tr><th>${target}</th>${results.map(r => `<td style="background:${U.colorScale(r.r)};font-weight:600">${r.r.toFixed(2)}</td>`).join('')}</tr></tbody></table></div></div>`;

    // 3. Ranked table — all metrics vs target only
    h += `<div class="result-section"><h4><i class="fas fa-table"></i> Pairwise Details — vs <strong style="color:var(--accent)">${target}</strong></h4>
        <table class="result-table"><thead><tr>
            <th>Rank</th><th>Feature</th>
            <th>r</th><th>R²</th><th>P-Value</th>
            <th title="VIF = 1/(1-R²)">VIF</th>
            <th>Strength</th>
        </tr></thead><tbody>
        ${results.map((r, i) => {
        const vif = r.r2 >= 0.9999 ? '\u221e' : (1 / (1 - r.r2)).toFixed(3);
        const vifNum = r.r2 < 0.9999 ? 1 / (1 - r.r2) : Infinity;
        const vifColor = vifNum > 10 ? 'var(--danger)' : vifNum > 5 ? 'var(--warning)' : 'var(--success)';
        return `<tr>
                <td><strong>#${i + 1}</strong></td>
                <td><strong>${r.feature}</strong></td>
                <td class="result-value" style="color:${r.r > 0 ? 'var(--success)' : 'var(--danger)'}">${r.r.toFixed(4)}</td>
                <td>${r.r2.toFixed(4)}</td>
                <td class="${r.significant ? 'result-significant' : 'result-not-significant'}">${r.p.toFixed(4)}</td>
                <td style="color:${vifColor};font-weight:600">${vif}</td>
                <td>${r.strength} ${r.direction}</td>
            </tr>`;
    }).join('')}</tbody></table></div>`;

    // 4. Individual scatter plots (top 6)
    const topN = results.slice(0, 6);
    h += `<div class="result-section"><h4><i class="fas fa-braille"></i> Scatter Plots — Top ${topN.length} Features vs ${target}</h4>`;
    h += `<div class="grid-${topN.length <= 2 ? '2' : topN.length <= 4 ? '2' : '3'}" style="gap:16px">`;
    topN.forEach((r, i) => {
        h += `<div class="card"><div class="card-header"><h3 style="font-size:12px">${r.feature} vs ${target} <span style="color:${Math.abs(r.r) > .7 ? 'var(--success)' : Math.abs(r.r) > .4 ? 'var(--warning)' : 'var(--text-muted)'}">r=${r.r.toFixed(3)}</span></h3></div>
              <div class="card-body"><div class="result-chart"><canvas id="corr-sc-${i}"></canvas></div></div></div>`;
    });
    h += `</div></div>`;

    U.html('corr-results', h);
    dm.analysisCount++;

    setTimeout(() => {
        // Bar chart (horizontal sorted by r)
        const barEl = U.el('corr-bar-chart');
        if (barEl) {
            const labels = results.map(r => r.feature);
            const data = results.map(r => r.r);
            const bgColors = data.map(v => v > 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)');
            new Chart(barEl, {
                type: 'bar',
                data: { labels, datasets: [{ label: `r with ${target}`, data, backgroundColor: bgColors, borderColor: bgColors.map(c => c.replace('0.7', '1')), borderWidth: 1 }] },
                options: {
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` r = ${ctx.raw.toFixed(4)}` } } },
                    scales: { x: { min: -1, max: 1, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { display: false } } }
                }
            });
        }
        // Scatter plots per feature
        topN.forEach((r, i) => {
            const el = U.el(`corr-sc-${i}`);
            if (el) viz.scatter(`corr-sc-${i}`, r.xVals, r.yVals, { xLabel: r.feature, yLabel: target, title: '', trendline: true });
        });
    }, 150);
}
function initControlCharts() { U.on('cc-type', 'change', e => { U.el('cc-subgroup-group').style.display = e.target.value === 'xbar-r' ? '' : 'none' }); U.on('btn-run-cc', 'click', () => { if (!dm.hasData()) return; const v = U.el('cc-variable').value; if (!v) { U.toast('Select variable', 'warning'); return } const vals = dm.getNumericValues(v), t = U.el('cc-type').value, sg = +U.el('cc-subgroup-size').value || 5; try { let r; if (t === 'i-mr') r = StatisticsEngine.iMRChart(vals); else r = StatisticsEngine.xbarRChart(vals, sg); let h = ''; if (r.xbar) { h += `<div class="result-section"><h4><i class="fas fa-wave-square"></i> X̄ Chart</h4><div class="result-chart"><canvas id="cc-x"></canvas></div><div class="result-summary">CL=${r.xbar.cl.toFixed(4)} · UCL=${r.xbar.ucl.toFixed(4)} · LCL=${r.xbar.lcl.toFixed(4)}</div></div><div class="result-section"><h4>R Chart</h4><div class="result-chart"><canvas id="cc-r"></canvas></div></div>` } if (r.individuals) { h += `<div class="result-section"><h4><i class="fas fa-wave-square"></i> I Chart</h4><div class="result-chart"><canvas id="cc-i"></canvas></div><div class="result-summary">CL=${r.individuals.cl.toFixed(4)} · UCL=${r.individuals.ucl.toFixed(4)} · LCL=${r.individuals.lcl.toFixed(4)}</div></div><div class="result-section"><h4>MR Chart</h4><div class="result-chart"><canvas id="cc-mr"></canvas></div></div>` } U.html('cc-results', h); dm.analysisCount++; setTimeout(() => { if (r.xbar) { viz.controlChart('cc-x', r.xbar, { title: 'X̄ Chart', yLabel: v }); viz.controlChart('cc-r', r.range, { title: 'R Chart', yLabel: 'Range' }) } if (r.individuals) { viz.controlChart('cc-i', r.individuals, { title: 'I Chart', yLabel: v }); viz.controlChart('cc-mr', r.mr, { title: 'MR Chart', yLabel: 'MR' }) } }, 100) } catch (e) { U.toast('Error: ' + e.message, 'error') } }) }
function initCapability() { U.on('btn-run-capability', 'click', () => { if (!dm.hasData()) return; const v = U.el('cap-variable').value, lsl = +U.el('cap-lsl').value, usl = +U.el('cap-usl').value; if (!v || isNaN(lsl) || isNaN(usl)) { U.toast('Enter variable and limits', 'warning'); return } const vals = dm.getNumericValues(v), r = StatisticsEngine.capability(vals, lsl, usl, +U.el('cap-target').value || (lsl + usl) / 2); const gc = v => v >= 1.33 ? 'gauge-good' : v >= 1 ? 'gauge-warn' : 'gauge-bad'; U.html('cap-results', `<div class="result-section"><h4><i class="fas fa-bullseye"></i> ${v}</h4><div class="capability-gauges"><div class="gauge-card"><div class="gauge-value ${gc(r.cp)}">${r.cp.toFixed(3)}</div><div class="gauge-label">Cp</div></div><div class="gauge-card"><div class="gauge-value ${gc(r.cpk)}">${r.cpk.toFixed(3)}</div><div class="gauge-label">Cpk</div></div><div class="gauge-card"><div class="gauge-value ${gc(r.pp)}">${r.pp.toFixed(3)}</div><div class="gauge-label">Pp</div></div><div class="gauge-card"><div class="gauge-value ${gc(r.ppk)}">${r.ppk.toFixed(3)}</div><div class="gauge-label">Ppk</div></div><div class="gauge-card"><div class="gauge-value">${r.sigmaLevel}σ</div><div class="gauge-label">Sigma</div></div></div><div class="result-summary">PPM Total: ${r.ppmTotal.toLocaleString()} · Rating: <span class="${r.rating === 'Capable' ? 'result-significant' : 'result-not-significant'}">${r.rating}</span></div><div class="result-chart"><canvas id="cap-ch"></canvas></div></div>`); dm.analysisCount++; setTimeout(() => viz.capabilityChart('cap-ch', vals, lsl, usl, r.target, { title: `Capability: ${v}` }), 100) }) }
function initPareto() { U.on('btn-run-pareto', 'click', () => { if (!dm.hasData()) return; const c = U.el('pareto-category').value; if (!c) return; const d = StatisticsEngine.pareto(dm.getColumnValues(c).filter(v => v != null)); U.html('pareto-results', `<div class="result-section"><h4><i class="fas fa-sort-amount-down"></i> ${c}</h4><div class="result-chart"><canvas id="par-ch"></canvas></div><table class="result-table"><thead><tr><th>Category</th><th>Count</th><th>%</th><th>Cum%</th></tr></thead><tbody>${d.map(x => `<tr><td>${x.category}</td><td class="result-value">${x.count}</td><td>${x.pct.toFixed(1)}%</td><td>${x.cumulative.toFixed(1)}%</td></tr>`).join('')}</tbody></table></div>`); dm.analysisCount++; setTimeout(() => viz.paretoChart('par-ch', d, { title: `Pareto: ${c}` }), 100) }) }
function initTimeSeries() { U.on('btn-run-timeseries', 'click', () => { if (!dm.hasData()) return; const v = U.el('ts-value').value; if (!v) return; const vals = dm.getNumericValues(v), a = U.el('ts-analysis').value, w = +U.el('ts-window').value || 7, labels = vals.map((_, i) => i + 1); const ds = [{ label: v, data: vals }]; if (a === 'moving-average') ds.push({ label: `${w}-pt MA`, data: StatisticsEngine.movingAverage(vals, w) }); let h = `<div class="result-section"><h4><i class="fas fa-chart-line"></i> Time Series</h4><div class="result-chart" style="height:380px"><canvas id="ts-ch"></canvas></div></div>`; if (a === 'autocorrelation') { const acf = StatisticsEngine.autocorrelation(vals); h += `<div class="result-section"><h4>ACF</h4><div class="result-chart"><canvas id="ts-acf"></canvas></div></div>` } U.html('ts-results', h); dm.analysisCount++; setTimeout(() => { viz.line('ts-ch', labels, ds, { title: `Time Series: ${v}`, xLabel: 'Observation', yLabel: v }); if (a === 'autocorrelation') { const acf = StatisticsEngine.autocorrelation(vals); viz.bar('ts-acf', acf.acf.map((_, i) => i), acf.acf, { title: 'ACF', xLabel: 'Lag', yLabel: 'Correlation' }) } }, 100) }) }
function initForecasting() {
    U.on('fc-alpha', 'input', e => U.html('fc-alpha-val', parseFloat(e.target.value).toFixed(2)));
    U.on('btn-run-forecast', 'click', () => {
        if (!dm.hasData()) return; const v = U.el('fc-value').value; if (!v) return; const vals = dm.getNumericValues(v), method = U.el('fc-method').value, alpha = +U.el('fc-alpha').value, periods = +U.el('fc-periods').value;
        if (method === 'lstm') { const r = MLEngine.lstmForecast(vals, 10, 50, 8); const allLabels = [...vals.map((_, i) => i + 1), ...r.forecast.map((_, i) => `F${i + 1}`)]; U.html('fc-results', `<div class="result-section"><h4><i class="fas fa-brain"></i> LSTM Forecast</h4><div class="result-chart" style="height:380px"><canvas id="fc-ch"></canvas></div><div class="result-summary">Method: LSTM Neural Network · Lookback: ${r.lookback} · Forecast: ${r.forecast[0].toFixed(2)}</div></div>`); dm.analysisCount++; setTimeout(() => viz.line('fc-ch', allLabels, [{ label: 'Actual', data: [...vals, ...Array(r.forecast.length).fill(null)] }, { label: 'Fitted', data: [...Array(vals.length - r.fitted.length).fill(null), ...r.fitted, ...Array(r.forecast.length).fill(null)] }, { label: 'Forecast', data: [...Array(vals.length).fill(null), ...r.forecast] }], { title: 'LSTM Forecast', xLabel: 'Period', yLabel: v }), 100) }
        else { const { smoothed: sm, forecast: fc } = StatisticsEngine.exponentialSmoothing(vals, alpha, periods); const allLabels = [...vals.map((_, i) => i + 1), ...fc.map((_, i) => `F${i + 1}`)]; U.html('fc-results', `<div class="result-section"><h4><i class="fas fa-chart-line"></i> Forecast</h4><div class="result-chart" style="height:380px"><canvas id="fc-ch"></canvas></div><div class="result-summary">Method: Exponential Smoothing (α=${alpha}) · Forecast: ${fc[0].toFixed(4)}</div></div>`); dm.analysisCount++; setTimeout(() => viz.line('fc-ch', allLabels, [{ label: 'Actual', data: [...vals, ...Array(fc.length).fill(null)] }, { label: 'Smoothed', data: [...sm, ...Array(fc.length).fill(null)] }, { label: 'Forecast', data: [...Array(vals.length).fill(null), ...fc] }], { title: 'Forecast', xLabel: 'Period', yLabel: v }), 100) }
    });
}

/* ---- ML ---- */
function initMLSetup() {
    U.on('ml-test-split', 'input', e => U.html('ml-split-val', e.target.value + '%'));
    U.on('btn-ml-prepare', 'click', () => {
        const target = U.el('ml-target').value; if (!target) { U.toast('Select target', 'warning'); return }
        dm.target = target; dm.features = [...document.querySelectorAll('.ml-feature-check:checked')].map(c => c.value).filter(c => c !== target);
        if (!dm.features.length) { U.toast('Select features', 'warning'); return }
        const pType = U.el('ml-problem-type').value; let problemType = pType;
        if (pType === 'auto') problemType = dm.columnTypes[target] === 'categorical' || dm.columnTypes[target] === 'boolean' || dm.getUniqueValues(target).length <= 10 ? 'classification' : 'regression';
        dm.problemType = problemType;
        U.html('ml-preview', `<div class="result-summary"><strong>Target:</strong> ${target} (${problemType})<br><strong>Features:</strong> ${dm.features.join(', ')}<br><strong>Rows:</strong> ${dm.data.length}<br><strong>Test Split:</strong> ${U.el('ml-test-split').value}%</div><p style="margin-top:12px;color:var(--success)"><i class="fas fa-check-circle"></i> Dataset prepared. Go to Model Training.</p>`);
        U.toast('Dataset prepared', 'success');
        setupModelCards(problemType);
    });
}
function setupModelCards(type) {
    const regModels = [{ name: 'Linear Regression', icon: 'fa-chart-line', desc: 'Ordinary least squares', id: 'linear' }, { name: 'Ridge Regression', icon: 'fa-mountain', desc: 'L2 regularized', id: 'ridge' }, { name: 'Lasso Regression', icon: 'fa-compress-alt', desc: 'L1 regularized', id: 'lasso' }, { name: 'Polynomial', icon: 'fa-bezier-curve', desc: 'Nonlinear polynomial', id: 'poly' }, { name: 'KNN Regression', icon: 'fa-project-diagram', desc: 'K-nearest neighbors', id: 'knn-reg' }, { name: 'Decision Tree', icon: 'fa-sitemap', desc: 'Tree-based regression', id: 'dt-reg' }, { name: 'Random Forest', icon: 'fa-tree', desc: 'Ensemble of trees', id: 'rf-reg' }];
    const clsModels = [{ name: 'Logistic Regression', icon: 'fa-divide', desc: 'Binary/multi classification', id: 'logistic' }, { name: 'KNN Classifier', icon: 'fa-project-diagram', desc: 'K-nearest neighbors', id: 'knn-cls' }, { name: 'Decision Tree', icon: 'fa-sitemap', desc: 'Tree classifier', id: 'dt-cls' }, { name: 'Random Forest', icon: 'fa-tree', desc: 'Ensemble classifier', id: 'rf-cls' }];
    const models = type === 'regression' ? regModels : clsModels;
    const gridId = type === 'regression' ? 'reg-model-grid' : 'cls-model-grid';
    U.el('ml-regression-models').style.display = type === 'regression' ? '' : 'none';
    U.el('ml-classification-models').style.display = type === 'classification' ? '' : 'none';
    U.html(gridId, models.map(m => `<div class="model-card" data-model="${m.id}" onclick="this.classList.toggle('selected')"><div class="model-icon" style="background:var(--gradient-1)"><i class="fas ${m.icon}"></i></div><h4>${m.name}</h4><p>${m.desc}</p></div>`).join(''));
}
function initMLModels() {
    U.on('btn-train-models', 'click', () => trainModels(false));
    U.on('btn-train-all', 'click', () => trainModels(true));
}
function trainModels(all) {
    if (!dm.target || !dm.features.length) { U.toast('Prepare dataset first in ML Setup', 'warning'); return }
    const selected = all ? document.querySelectorAll('.model-card') : document.querySelectorAll('.model-card.selected');
    if (!selected.length) { U.toast('Select models', 'warning'); return }
    const ids = [...selected].map(c => c.dataset.model);
    U.el('training-progress').style.display = ''; U.html('train-status', 'Preparing data...');
    const numCols = dm.features.filter(f => ['continuous', 'discrete'].includes(dm.columnTypes[f]));
    const X = numCols.map(f => dm.getNumericValues(f));
    let y = dm.getColumnValues(dm.target);
    if (dm.problemType === 'classification') { const labels = [...new Set(y)]; y = y.map(v => labels.indexOf(v)); dm.targetLabels = labels } else y = y.map(Number);
    const n = Math.min(y.length, ...X.map(c => c.length)); const XTrim = X.map(c => c.slice(0, n)), yTrim = y.slice(0, n);
    const split = MLEngine.splitData(XTrim, yTrim, +U.el('ml-test-split').value / 100);
    const XTr = split.XTrain[0].map((_, i) => split.XTrain.map(r => r[i])); const XTe = split.XTest[0].map((_, i) => split.XTest.map(r => r[i]));
    const XTrainCols = XTr[0] ? XTr[0].map((_, j) => XTr.map(r => r[j])) : []; const XTestCols = XTe[0] ? XTe[0].map((_, j) => XTe.map(r => r[j])) : [];
    trainedModels.length = 0; let done = 0;
    const bar = U.el('train-progress-bar'); bar.style.width = '0%'; bar.style.animation = 'none';
    const next = () => {
        if (done >= ids.length) { bar.style.width = '100%'; U.html('train-status', `Done! ${trainedModels.length} models trained`); U.toast(`Trained ${trainedModels.length} models`, 'success'); showMLResults(); updateDashboard(); return }
        const id = ids[done]; U.html('train-status', `Training ${id}... (${done + 1}/${ids.length})`); bar.style.width = ((done / ids.length) * 100) + '%';
        setTimeout(() => {
            try {
                let r;
                if (id === 'linear') r = MLEngine.linearRegressionModel(XTrainCols, split.yTrain, XTestCols, split.yTest);
                else if (id === 'ridge') r = MLEngine.ridgeRegression(XTrainCols, split.yTrain, XTestCols, split.yTest, 1);
                else if (id === 'lasso') r = MLEngine.lassoRegression(XTrainCols, split.yTrain, XTestCols, split.yTest, 1);
                else if (id === 'poly') r = MLEngine.polynomialRegressionModel(XTrainCols, split.yTrain, XTestCols, split.yTest, 2);
                else if (id === 'knn-reg') r = MLEngine.knnRegression(XTrainCols, split.yTrain, XTestCols, split.yTest, 5);
                else if (id === 'dt-reg') r = MLEngine.decisionTreeRegression(XTrainCols, split.yTrain, XTestCols, split.yTest, 5);
                else if (id === 'rf-reg') r = MLEngine.randomForestRegression(XTrainCols, split.yTrain, XTestCols, split.yTest, 10, 4);
                else if (id === 'logistic') r = MLEngine.logisticRegression(XTrainCols, split.yTrain, XTestCols, split.yTest);
                else if (id === 'knn-cls') r = MLEngine.knnClassifier(XTrainCols, split.yTrain, XTestCols, split.yTest, 5);
                else if (id === 'dt-cls') r = MLEngine.decisionTreeClassifier(XTrainCols, split.yTrain, XTestCols, split.yTest, 5);
                else if (id === 'rf-cls') r = MLEngine.randomForestClassifier(XTrainCols, split.yTrain, XTestCols, split.yTest, 10, 4);
                if (r) { r.id = id; r.features = numCols; r.testY = split.yTest; trainedModels.push(r) }
            } catch (e) { console.error(id, e) } done++; next()
        }, 50)
    };
    next();
}
function showMLResults() {
    if (!trainedModels.length) { U.html('ml-results-content', '<div class="empty-state"><i class="fas fa-trophy"></i><h3>No Results</h3></div>'); return }
    const isReg = trainedModels[0].type === 'regression';
    let h = '<h3 style="margin-bottom:14px;font-size:15px;font-weight:600"><i class="fas fa-trophy"></i> Model Comparison</h3>';
    if (isReg) {
        h += `<table class="result-table"><thead><tr><th>Model</th><th>Train R²</th><th>Test R²</th><th>Train RMSE</th><th>Test RMSE</th><th>Test MAE</th></tr></thead><tbody>${trainedModels.map(m => `<tr><td><strong>${m.name}</strong></td><td class="result-value">${m.trainR2.toFixed(4)}</td><td class="result-value">${m.testR2.toFixed(4)}</td><td>${m.trainRMSE.toFixed(4)}</td><td>${m.testRMSE.toFixed(4)}</td><td>${m.testMAE.toFixed(4)}</td></tr>`).join('')}</tbody></table>`;
    } else { h += `<table class="result-table"><thead><tr><th>Model</th><th>Train Acc</th><th>Test Acc</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${trainedModels.map(m => `<tr><td><strong>${m.name}</strong></td><td class="result-value">${(m.trainAccuracy * 100).toFixed(1)}%</td><td class="result-value">${(m.testAccuracy * 100).toFixed(1)}%</td><td>${(m.precision * 100).toFixed(1)}%</td><td>${(m.recall * 100).toFixed(1)}%</td><td>${m.f1.toFixed(3)}</td></tr>`).join('')}</tbody></table>` }
    h += `<div class="result-chart" style="margin-top:18px"><canvas id="ml-comparison-chart"></canvas></div>`;
    // Best model
    const best = isReg ? trainedModels.reduce((a, b) => a.testR2 > b.testR2 ? a : b) : trainedModels.reduce((a, b) => a.testAccuracy > b.testAccuracy ? a : b);
    h += `<div class="result-summary"><strong>Best Model:</strong> ${best.name}<br>${isReg ? `Test R² = ${best.testR2.toFixed(4)} · RMSE = ${best.testRMSE.toFixed(4)}` : `Test Accuracy = ${(best.testAccuracy * 100).toFixed(1)}% · F1 = ${best.f1.toFixed(3)}`}</div>`;
    U.html('ml-results-content', h);
    // Populate predict dropdown
    const ps = U.el('predict-model'); if (ps) { ps.innerHTML = trainedModels.map(m => `<option value="${m.id}">${m.name}</option>`).join(''); setupPredictInputs() }
    setTimeout(() => { if (isReg) viz.bar('ml-comparison-chart', trainedModels.map(m => m.name), trainedModels.map(m => m.testR2), { title: 'Test R² Comparison', yLabel: 'R²' }); else viz.bar('ml-comparison-chart', trainedModels.map(m => m.name), trainedModels.map(m => m.testAccuracy * 100), { title: 'Test Accuracy %', yLabel: 'Accuracy %' }) }, 100);
}

/* ---- Predict ---- */
function initMLPredict() { U.on('btn-predict', 'click', makePrediction); U.on('predict-model', 'change', setupPredictInputs) }
function setupPredictInputs() { const m = trainedModels[0]; if (!m) return; U.html('predict-inputs', m.features.map(f => `<div class="form-group"><label>${f}</label><input class="form-control predict-input" data-feature="${f}" type="number" step="any" value="${dm.getNumericValues(f)[0] || 0}"></div>`).join('')) }
function makePrediction() { const mid = U.el('predict-model').value; const m = trainedModels.find(x => x.id === mid); if (!m) { U.toast('Select model', 'warning'); return } const inputs = [...document.querySelectorAll('.predict-input')].map(i => +i.value); try { let pred = m.predict(inputs); if (m.type === 'classification' && dm.targetLabels) pred = dm.targetLabels[pred] || pred; U.html('predict-result', `<div class="metric-card" style="background:var(--bg-card)"><div class="metric-value" style="color:var(--accent)">${typeof pred === 'number' ? pred.toFixed(4) : pred}</div><div class="metric-label">Predicted ${dm.target}</div></div>`); U.toast('Prediction complete', 'success') } catch (e) { U.toast('Error: ' + e.message, 'error') } }

/* ---- Project ---- */
function initProject() {
    U.on('btn-save-proj', 'click', () => { const n = U.el('project-name').value.trim() || 'Untitled'; dm.saveProject(n); U.toast(`Saved: ${n}`, 'success'); refreshProjectList() });
    U.on('btn-export-proj', 'click', () => { U.downloadJSON(dm.exportProject(), `${dm.datasetName || 'project'}.json`) });
    U.on('btn-import-proj', 'click', () => U.el('import-proj-file').click());
    U.el('import-proj-file').addEventListener('change', e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = ev => { dm.importProject(ev.target.result); onDataLoaded(f.name) }; r.readAsText(f) });
    refreshProjectList();
}
function refreshProjectList() { const list = U.el('project-list'); if (!list) return; const projects = dm.listProjects(); list.innerHTML = projects.length ? projects.map(p => `<div class="project-item" onclick="dm.loadProject('${p.name}');onDataLoaded('${p.name}');U.toast('Loaded','success')"><div><strong>${p.name}</strong><br><span style="font-size:10px;color:var(--text-muted)">${p.timestamp ? new Date(p.timestamp).toLocaleString() : ''} · ${p.data?.length || 0} rows</span></div><button class="btn btn-sm btn-danger" onclick="event.stopPropagation();dm.deleteProject('${p.name}');refreshProjectList()"><i class="fas fa-trash"></i></button></div>`).join('') : '<p style="font-size:12px;color:var(--text-muted)">No saved projects</p>' }
window.refreshProjectList = refreshProjectList; window.onDataLoaded = onDataLoaded;
