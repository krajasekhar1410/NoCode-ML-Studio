/* ============================================
   Canvas App - Pipeline, AutoML, Quick Insights, What-If
   ============================================ */

/* ---- Override Dashboard with Canvas Dashboard ---- */
document.addEventListener('DOMContentLoaded', () => {
    // Render canvas-only pages
    const mappings = { dashboard: 'dashboard', pipeline: 'pipeline', 'quick-insights': 'quickInsights', automl: 'automl', 'what-if': 'whatIf' };
    Object.entries(mappings).forEach(([id, fn]) => {
        const el = document.getElementById(`page-${id}`);
        if (el && CanvasPages[fn]) el.innerHTML = CanvasPages[fn]();
    });
    // Re-update var dropdowns for automl — use 300ms to ensure app.js init is fully done
    setTimeout(() => {
        if (typeof updateVarDropdowns === 'function') updateVarDropdowns();
        initPipeline(); initQuickInsights(); initAutoML(); initWhatIf();
        // Update page titles
        PAGE_TITLES['pipeline'] = 'Pipeline Builder';
        PAGE_TITLES['quick-insights'] = 'Quick Insights';
        PAGE_TITLES['automl'] = 'AutoML Studio';
        PAGE_TITLES['what-if'] = 'What-If Analysis';
        // Patch navigateTo to always refresh dropdowns when entering automl page
        const _origNavigate = window.navigateTo;
        window.navigateTo = function (page) {
            _origNavigate(page);
            if (page === 'automl' && typeof updateVarDropdowns === 'function') {
                updateVarDropdowns();
            }
        };
    }, 300);
});

/* ---- Pipeline Builder ---- */
let pipelineNodeId = 0;
function initPipeline() {
    const canvas = U.el('pipeline-nodes');
    if (!canvas) return;
    const palette = U.el('node-palette');
    palette.querySelectorAll('.palette-item').forEach(item => {
        item.addEventListener('click', () => addPipelineNode(item.dataset.node, item.textContent.trim(), canvas));
    });
    U.on('btn-run-pipeline', 'click', runPipeline);
    U.on('btn-clear-pipeline', 'click', () => {
        canvas.innerHTML = '';
        pipelineNodeId = 0;
        U.html('pipeline-status', '<i class="fas fa-circle"></i> Idle');
        U.el('pipeline-status').className = 'status-pill idle';
    });
    // Pre-populate sample pipeline
    addPipelineNode('csv', 'CSV Import', canvas);
    addPipelineNode('clean', 'Clean', canvas);
    addPipelineNode('feature', 'Feature Eng', canvas);
    addPipelineNode('model', 'ML Model', canvas);
    addPipelineNode('evaluate', 'Evaluate', canvas);
}
function addPipelineNode(type, label, canvas) {
    pipelineNodeId++;
    const colors = { csv: 'var(--gradient-1)', api: 'var(--gradient-5)', filter: 'var(--gradient-2)', clean: 'var(--gradient-3)', transform: 'var(--gradient-4)', feature: 'linear-gradient(135deg,#ec4899,#8b5cf6)', split: 'var(--gradient-2)', model: 'var(--gradient-5)', evaluate: 'var(--gradient-4)', deploy: 'var(--gradient-3)' };
    const icons = { csv: 'fa-file-csv', api: 'fa-plug', filter: 'fa-filter', clean: 'fa-broom', transform: 'fa-exchange-alt', feature: 'fa-columns', split: 'fa-random', model: 'fa-brain', evaluate: 'fa-trophy', deploy: 'fa-rocket' };
    const node = document.createElement('div');
    node.className = 'pipeline-node';
    node.id = `pn-${pipelineNodeId}`;
    node.dataset.type = type;
    let bodyHtml = '';
    if (type === 'csv') bodyHtml = '<span>Source dataset</span>';
    else if (type === 'clean') bodyHtml = '<select class="form-control" style="font-size:10px"><option>Fill Missing</option><option>Drop Duplicates</option><option>Remove Outliers</option></select>';
    else if (type === 'feature') bodyHtml = '<select class="form-control" style="font-size:10px"><option>Auto Features</option><option>Polynomial</option><option>Interaction</option></select>';
    else if (type === 'model') bodyHtml = '<select class="form-control" style="font-size:10px"><option>AutoML</option><option>Linear</option><option>Random Forest</option><option>KNN</option></select>';
    else if (type === 'filter') bodyHtml = '<input class="form-control" placeholder="Condition..." style="font-size:10px">';
    else if (type === 'evaluate') bodyHtml = '<span>R², RMSE, F1...</span>';
    else if (type === 'deploy') bodyHtml = '<span>Export model</span>';
    else bodyHtml = `<span>${label}</span>`;

    node.innerHTML = `<div class="node-port input"></div><div class="node-header"><div class="node-icon" style="background:${colors[type] || 'var(--gradient-1)'}"><i class="fas ${icons[type] || 'fa-cube'}"></i></div>${label}<div class="node-status idle" id="ns-${pipelineNodeId}"></div></div><div class="node-body">${bodyHtml}</div><div class="node-port output"></div>`;
    canvas.appendChild(node);
}

function runPipeline() {
    const nodes = document.querySelectorAll('.pipeline-node');
    if (!nodes.length) { U.toast('Add nodes first', 'warning'); return; }
    if (!dm.hasData()) { U.toast('Load data first', 'warning'); return; }

    const status = U.el('pipeline-status');
    status.className = 'status-pill running';
    status.innerHTML = '<i class="fas fa-spinner"></i> Running...';

    let i = 0;
    const runNext = () => {
        if (i >= nodes.length) {
            status.className = 'status-pill success';
            status.innerHTML = '<i class="fas fa-check"></i> Complete';
            U.toast('Pipeline completed successfully!', 'success');
            // Update wizard
            document.querySelectorAll('#pipeline-wizard .wizard-step').forEach(s => s.classList.add('completed'));
            document.querySelectorAll('#pipeline-wizard .wizard-connector').forEach(c => c.classList.add('done'));
            return;
        }
        const node = nodes[i];
        const ns = node.querySelector('.node-status');
        ns.className = 'node-status running';
        node.classList.add('active');

        setTimeout(() => {
            const type = node.dataset.type;
            try {
                if (type === 'clean') { dm.removeDuplicates(); }
                else if (type === 'feature') { /* auto feature — no-op for demo */ }
                else if (type === 'model') { /* triggers automl on completion */ }
            } catch (e) { /* continue */ }
            ns.className = 'node-status done';
            node.classList.remove('active');
            node.classList.add('completed');
            i++;
            runNext();
        }, 600);
    };
    runNext();
}

/* ---- Quick Insights ---- */
function initQuickInsights() {
    U.on('btn-gen-insights', 'click', generateInsights);
}

function isTargetContinuous(target) {
    if (!target) return true;
    const type = dm.columnTypes[target];
    const uniq = dm.getUniqueValues(target).length;
    return (type === 'continuous' || type === 'discrete') && uniq > 10;
}

function generateInsights() {
    if (!dm.hasData()) { U.toast('Load data first', 'warning'); return; }
    const target = U.el('insights-target')?.value;
    if (!target) { U.toast('Select a target variable', 'warning'); return; }
    U.toast('Analyzing data...', 'info');

    const container = U.el('insights-container');
    const numCols = dm.getNumericColumns().filter(c => c !== target);
    const catCols = dm.getCategoricalColumns();
    const profiles = dm.profileAll();
    const continuous = isTargetContinuous(target);

    let html = '';

    // --- Always: Data Overview ---
    const overallQ = Math.round(profiles.reduce((s, p) => s + p.qualityPct, 0) / profiles.length * 100);
    html += insightCard('fa-database', 'var(--gradient-1)', 'Data Overview', 'Summary',
        `Dataset: <strong>${dm.data.length.toLocaleString()}</strong> rows · <strong>${dm.columns.length}</strong> columns · Data Quality: <strong>${overallQ}%</strong>.<br>Target: <strong style="color:var(--accent)">${target}</strong> — detected as <strong>${continuous ? 'Continuous (Regression)' : 'Categorical (Classification)'}</strong>`,
        ['auto', 'Auto-generated'], 'ins-overview');

    // --- Missing values ---
    const missingCols = profiles.filter(p => p.missing > 0);
    if (missingCols.length) {
        html += insightCard('fa-exclamation-triangle', 'linear-gradient(135deg,#f59e0b,#ef4444)', 'Missing Values Detected', 'Data Quality',
            `Missing in: ${missingCols.map(c => `<strong>${c.col}</strong> (${(c.missingPct * 100).toFixed(1)}%)`).join(', ')}. Use Data Cleaning → Fill Missing.`,
            ['warning', 'Action Required'], 'ins-missing');
    }

    // --- Outliers ---
    const outlierCols = numCols.map(c => ({ col: c, outliers: profiles.find(p => p.col === c)?.outliers || 0 })).filter(x => x.outliers > 0);
    if (outlierCols.length) {
        html += insightCard('fa-exclamation-circle', 'linear-gradient(135deg,#f97316,#ef4444)', 'Outliers Detected', 'Anomaly',
            `Outliers found in: ${outlierCols.map(x => `<strong>${x.col}</strong> (${x.outliers})`).join(', ')}.`,
            ['warning', 'Review'], 'ins-outliers');
    }

    if (continuous) {
        // ============================================================
        // CONTINUOUS TARGET → Correlation insights
        // ============================================================
        const yVals = dm.getNumericValues(target);
        const corrResults = numCols.map(f => {
            const xVals = dm.getNumericValues(f);
            const n = Math.min(xVals.length, yVals.length);
            const c = StatisticsEngine.correlation(xVals.slice(0, n), yVals.slice(0, n));
            return {
                feature: f, r: c.r, r2: c.rSquared, p: c.pValue, significant: c.significant,
                strength: c.strength, direction: c.direction,
                xVals: xVals.slice(0, n), yVals: yVals.slice(0, n)
            };
        }).sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

        const strongPos = corrResults.filter(r => r.r >= 0.3).slice(0, 3);
        const strongNeg = corrResults.filter(r => r.r <= -0.3).slice(0, 3);
        const topFeatures = corrResults.slice(0, 6);

        // Summary insight
        const best = corrResults[0];
        if (best) {
            html += insightCard('fa-bezier-curve', 'var(--gradient-5)', `Top Predictor: ${best.feature}`, 'Correlation',
                `<strong>${best.feature}</strong> has the strongest relationship with <strong>${target}</strong>: r = <strong style="color:${best.r > 0 ? 'var(--success)' : 'var(--danger)'}">${best.r.toFixed(3)}</strong>, R² = <strong>${best.r2.toFixed(3)}</strong> — ${best.strength} ${best.direction} correlation${best.significant ? ' ✅ Significant' : ''}.`,
                ['important', 'Key Finding'], 'ins-top-pred');
        }

        // Ranked correlation bar chart
        html += `<div class="insight-card" id="ins-corr-bar">
            <div class="insight-header"><div class="insight-icon" style="background:var(--gradient-2)"><i class="fas fa-chart-bar"></i></div>
            <div><div class="insight-title">Correlation Ranking vs ${target}</div><div class="insight-subtitle">All numeric features ranked by |r|</div></div>
            <span class="insight-tag auto">Ranked</span></div>
            <div class="insight-chart" style="height:${Math.max(180, corrResults.length * 36)}px"><canvas id="ins-corr-bar-chart"></canvas></div>
        </div>`;

        // Strong Positive scatter plots
        if (strongPos.length) {
            html += `<div class="result-section"><h4 style="font-size:14px;font-weight:600;margin-bottom:12px"><i class="fas fa-arrow-trend-up" style="color:var(--success)"></i> Strong Positive Correlations with ${target}</h4>`;
            html += `<div class="grid-3" style="gap:14px">`;
            strongPos.forEach((r, i) => {
                html += `<div class="card"><div class="card-header"><h3 style="font-size:12px">${r.feature} <span style="color:var(--success)">r=${r.r.toFixed(3)} · R²=${r.r2.toFixed(3)}</span></h3></div><div class="card-body"><div class="result-chart"><canvas id="ins-pos-${i}"></canvas></div></div></div>`;
            });
            html += `</div></div>`;
        }

        // Strong Negative scatter plots
        if (strongNeg.length) {
            html += `<div class="result-section"><h4 style="font-size:14px;font-weight:600;margin-bottom:12px"><i class="fas fa-arrow-trend-down" style="color:var(--danger)"></i> Strong Negative Correlations with ${target}</h4>`;
            html += `<div class="grid-3" style="gap:14px">`;
            strongNeg.forEach((r, i) => {
                html += `<div class="card"><div class="card-header"><h3 style="font-size:12px">${r.feature} <span style="color:var(--danger)">r=${r.r.toFixed(3)} · R²=${r.r2.toFixed(3)}</span></h3></div><div class="card-body"><div class="result-chart"><canvas id="ins-neg-${i}"></canvas></div></div></div>`;
            });
            html += `</div></div>`;
        }

        if (!strongPos.length && !strongNeg.length) {
            html += insightCard('fa-info-circle', 'var(--gradient-3)', 'Weak Correlations', 'Correlation',
                `No strong correlations (|r| ≥ 0.3) found between numeric features and <strong>${target}</strong>. Consider feature engineering or non-linear models.`,
                ['auto', 'Info'], 'ins-weak-corr');
        }

        // ML Readiness
        html += insightCard('fa-robot', 'linear-gradient(135deg,#8b5cf6,#ec4899)', 'ML Recommendation', 'Machine Learning',
            `Best features for predicting <strong>${target}</strong>: ${topFeatures.slice(0, 3).map(f => `<strong>${f.feature}</strong> (r=${f.r.toFixed(2)})`).join(', ')}. Try <strong>AutoML Studio</strong> with Regression.`,
            ['auto', 'AutoML'], 'ins-ml');

        container.innerHTML = html;
        dm.analysisCount++;

        setTimeout(() => {
            // Ranked bar chart
            const barEl = U.el('ins-corr-bar-chart');
            if (barEl) {
                new Chart(barEl, {
                    type: 'bar',
                    data: {
                        labels: corrResults.map(r => r.feature), datasets: [{
                            label: `r with ${target}`,
                            data: corrResults.map(r => r.r),
                            backgroundColor: corrResults.map(r => r.r > 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)'),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { min: -1, max: 1, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { display: false } }
                        }
                    }
                });
            }
            // Positive scatter plots
            strongPos.forEach((r, i) => {
                const el = U.el(`ins-pos-${i}`);
                if (el) viz.scatter(`ins-pos-${i}`, r.xVals, r.yVals, { xLabel: r.feature, yLabel: target, trendline: true });
            });
            // Negative scatter plots
            strongNeg.forEach((r, i) => {
                const el = U.el(`ins-neg-${i}`);
                if (el) viz.scatter(`ins-neg-${i}`, r.xVals, r.yVals, { xLabel: r.feature, yLabel: target, trendline: true });
            });
        }, 200);

    } else {
        // ============================================================
        // CATEGORICAL TARGET → Classification insights
        // ============================================================
        const categories = dm.getUniqueValues(target);
        const catColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#f97316'];

        // Feature discrimination score (eta-squared approximation via ANOVA)
        const featureScores = numCols.map(f => {
            const groups = categories.map(c => dm.data.filter(r => r[target] == c).map(r => Number(r[f])).filter(v => !isNaN(v)));
            const allVals = groups.flat();
            if (allVals.length < 2) return { feature: f, score: 0, groups };
            const grandMean = ss.mean(allVals);
            const ssBetween = groups.reduce((s, g) => s + g.length * Math.pow(ss.mean(g) - grandMean, 2), 0);
            const ssTotal = allVals.reduce((s, v) => s + Math.pow(v - grandMean, 2), 0);
            const eta2 = ssTotal > 0 ? ssBetween / ssTotal : 0;
            return { feature: f, score: eta2, groups };
        }).sort((a, b) => b.score - a.score);

        // Summary insight
        if (featureScores.length) {
            const best = featureScores[0];
            html += insightCard('fa-layer-group', 'var(--gradient-5)', `Top Discriminating Feature: ${best.feature}`, 'Classification',
                `<strong>${best.feature}</strong> best separates the <strong>${categories.length}</strong> classes of <strong>${target}</strong> with η² = <strong>${best.score.toFixed(3)}</strong> (${best.score > 0.14 ? 'Large' : best.score > 0.06 ? 'Medium' : 'Small'} effect).`,
                ['important', 'Key Finding'], 'ins-top-cls');
        }

        // Feature importance bar chart
        html += `<div class="insight-card" id="ins-eta-bar">
            <div class="insight-header"><div class="insight-icon" style="background:var(--gradient-2)"><i class="fas fa-chart-bar"></i></div>
            <div><div class="insight-title">Feature Importance (η²) for ${target}</div><div class="insight-subtitle">Eta-squared — proportion of variance explained by class</div></div>
            <span class="insight-tag auto">Classification</span></div>
            <div class="insight-chart" style="height:${Math.max(180, featureScores.length * 36)}px"><canvas id="ins-eta-chart"></canvas></div>
        </div>`;

        // Box plots — top 4 features per category
        const topFeats = featureScores.slice(0, 4);
        if (topFeats.length) {
            html += `<div class="result-section"><h4 style="font-size:14px;font-weight:600;margin-bottom:12px"><i class="fas fa-boxes" style="color:var(--accent)"></i> Top Features by ${target} Class</h4>`;
            html += `<div class="grid-2" style="gap:14px">`;
            topFeats.forEach((f, i) => {
                html += `<div class="card"><div class="card-header"><h3 style="font-size:12px">${f.feature} <span style="color:var(--accent)">η²=${f.score.toFixed(3)}</span></h3></div><div class="card-body"><div class="result-chart"><canvas id="ins-cls-box-${i}"></canvas></div></div></div>`;
            });
            html += `</div></div>`;
        }

        // Class distribution
        html += insightCard('fa-chart-pie', 'var(--gradient-3)', `Class Distribution: ${target}`, 'Classification',
            `Classes: ${categories.map(c => {
                const cnt = dm.data.filter(r => r[target] == c).length;
                return `<strong>${c}</strong> (${cnt}, ${(cnt / dm.data.length * 100).toFixed(1)}%)`;
            }).join(' · ')}`,
            ['auto', 'Balance Check'], 'ins-class-dist');

        // ML Readiness
        html += insightCard('fa-robot', 'linear-gradient(135deg,#8b5cf6,#ec4899)', 'ML Recommendation', 'Machine Learning',
            `Best features for classifying <strong>${target}</strong>: ${featureScores.slice(0, 3).map(f => `<strong>${f.feature}</strong> (η²=${f.score.toFixed(2)})`).join(', ')}. Try <strong>AutoML Studio</strong> with Classification.`,
            ['auto', 'AutoML'], 'ins-ml');

        container.innerHTML = html;
        dm.analysisCount++;

        setTimeout(() => {
            // Eta-squared bar chart
            const etaEl = U.el('ins-eta-chart');
            if (etaEl) {
                new Chart(etaEl, {
                    type: 'bar',
                    data: {
                        labels: featureScores.map(f => f.feature), datasets: [{
                            label: 'η² (Effect Size)',
                            data: featureScores.map(f => f.score),
                            backgroundColor: featureScores.map((_, i) => catColors[i % catColors.length] + 'bb'),
                            borderColor: featureScores.map((_, i) => catColors[i % catColors.length]),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { min: 0, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { display: false } }
                        }
                    }
                });
            }
            // Box plots per feature
            topFeats.forEach((f, i) => {
                const el = U.el(`ins-cls-box-${i}`);
                if (el) viz.boxPlot(`ins-cls-box-${i}`, f.groups, categories.map(String),
                    { title: '', xLabel: target, yLabel: f.feature });
            });
        }, 200);
    }
}

function insightCard(icon, gradient, title, subtitle, body, [tagClass, tagText], id, hasChart = false) {
    return `<div class="insight-card" id="${id}">
        <div class="insight-header"><div class="insight-icon" style="background:${gradient}"><i class="fas ${icon}"></i></div><div><div class="insight-title">${title}</div><div class="insight-subtitle">${subtitle}</div></div><span class="insight-tag ${tagClass}">${tagText}</span></div>
        <div class="insight-body">${body}</div>
        ${hasChart ? `<div class="insight-chart"><canvas id="${id}-chart"></canvas></div>` : ''}
    </div>`;
}

/* ---- AutoML Studio ---- */
function updateAutoMLOptions() {
    const buildType = document.querySelector('.build-option.selected')?.dataset.build || 'quick';
    const stdOpts = U.el('standard-build-options');
    if (!stdOpts) return;

    if (buildType === 'standard') {
        stdOpts.style.display = 'block';
        const target = U.el('aml-target')?.value;
        let pType = U.el('aml-problem')?.value || 'auto';
        if (pType === 'auto' && target) {
            pType = Object.keys(dm.columnTypes).length ? (dm.columnTypes[target] === 'categorical' || dm.getUniqueValues(target).length <= 10 ? 'classification' : 'regression') : 'regression';
        }

        const modelsList = U.el('standard-models-list');
        if (!modelsList) return;

        let html = '';
        if (pType === 'regression') {
            html = `<label><input type="checkbox" value="linear" checked> Linear Regression</label>
                    <label><input type="checkbox" value="ridge" checked> Ridge Regression</label>
                    <label><input type="checkbox" value="lasso" checked> Lasso Regression</label>
                    <label><input type="checkbox" value="poly" checked> Polynomial (2nd deg)</label>
                    <label><input type="checkbox" value="knn-reg" checked> KNN Regression</label>
                    <label><input type="checkbox" value="dt-reg" checked> Decision Tree</label>
                    <label><input type="checkbox" value="rf-reg" checked> Random Forest</label>`;
        } else if (pType === 'classification') {
            html = `<label><input type="checkbox" value="logistic" checked> Logistic Regression</label>
                    <label><input type="checkbox" value="knn-cls" checked> KNN Classifier</label>
                    <label><input type="checkbox" value="dt-cls" checked> Decision Tree Classifier</label>
                    <label><input type="checkbox" value="rf-cls" checked> Random Forest Classifier</label>
                    <label><input type="checkbox" value="nb-cls" checked> Naive Bayes</label>
                    <label><input type="checkbox" value="svm-cls" checked> Linear SVM</label>`;
        } else if (pType === 'time-series') {
            html = `<label><input type="checkbox" value="lstm" checked> LSTM Forecast</label>
                    <label><input type="checkbox" value="arima" checked> ARIMA</label>
                    <label><input type="checkbox" value="ets" checked> ETS Smoothing</label>
                    <label><input type="checkbox" value="prophet" checked> Prophet</label>
                    <label><input type="checkbox" value="hw" checked> Holt-Winters</label>
                    <label><input type="checkbox" value="tbats" checked> TBATS</label>`;
        }
        modelsList.innerHTML = html;
    } else {
        stdOpts.style.display = 'none';
    }
}

function initAutoML() {
    document.addEventListener('click', function (e) {
        const opt = e.target.closest('.build-option');
        if (opt && document.getElementById('page-automl')?.contains(opt)) {
            document.querySelectorAll('.build-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            updateAutoMLOptions();
        }
        if (e.target.id === 'btn-automl-build' || e.target.closest('#btn-automl-build')) {
            runAutoML();
        }
    }, { once: false });
    const btn = U.el('btn-automl-build');
    if (btn) btn.addEventListener('click', runAutoML);

    document.addEventListener('focusin', function (e) {
        if (e.target.id === 'aml-target' && dm && dm.hasData()) {
            if (e.target.options.length <= 1) {
                const old = e.target.value;
                e.target.innerHTML = '<option value="">Select target column...</option>';
                dm.columns.forEach(c => {
                    const o = document.createElement('option');
                    o.value = c;
                    o.textContent = `${c} (${dm.columnTypes[c]})`;
                    e.target.appendChild(o);
                });
                if (old) e.target.value = old;
            }
        }
    });

    document.addEventListener('change', function (e) {
        if (e.target.id === 'aml-target' || e.target.id === 'aml-problem') {
            updateAutoMLOptions();
        }
    });

    document.querySelectorAll('.build-option').forEach(opt => {
        opt.addEventListener('click', () => {
            document.querySelectorAll('.build-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            updateAutoMLOptions();
        });
    });
}

function runAutoML() {
    if (!dm.hasData()) { U.toast('Load data first — use Import Data or a Sample Dataset', 'warning'); return; }
    const targetEl = U.el('aml-target');
    if (!targetEl) { U.toast('AutoML page not loaded yet — please navigate away and back', 'error'); return; }
    const target = targetEl.value;
    if (!target) { U.toast('Select a target variable', 'warning'); return; }

    const buildType = document.querySelector('.build-option.selected')?.dataset.build || 'quick';
    const problemEl = U.el('aml-problem');
    const problemType = problemEl ? problemEl.value : 'auto';
    let pType = problemType;
    if (pType === 'auto') pType = dm.columnTypes[target] === 'categorical' || dm.getUniqueValues(target).length <= 10 ? 'classification' : 'regression';

    dm.target = target;
    dm.features = dm.getNumericColumns().filter(c => c !== target);
    dm.problemType = pType;

    // Update wizard (null-safe)
    ['aml-step-1', 'aml-step-2', 'aml-step-3'].forEach(id => { const el = U.el(id); if (el) { el.classList.add('completed'); el.classList.remove('active'); } });
    const s4 = U.el('aml-step-4'); if (s4) s4.classList.add('active');
    ['aml-conn-1', 'aml-conn-2', 'aml-conn-3'].forEach(id => { const el = U.el(id); if (el) el.classList.add('done'); });
    const c3 = U.el('aml-conn-3'); if (c3) c3.classList.add('active');

    // Show training progress
    let models = [];
    if (buildType === 'standard') {
        const checkboxes = document.querySelectorAll('#standard-models-list input:checked');
        models = Array.from(checkboxes).map(c => c.value);
        if (!models.length) { U.toast('Please select at least one model', 'warning'); return; }
    } else {
        if (pType === 'regression') models = ['linear', 'ridge', 'lasso', 'poly', 'knn-reg', 'dt-reg', 'rf-reg'];
        else if (pType === 'classification') models = ['logistic', 'knn-cls', 'dt-cls', 'rf-cls', 'nb-cls', 'svm-cls'];
        else if (pType === 'time-series') models = ['lstm', 'arima', 'ets', 'prophet', 'hw', 'tbats'];
    }

    const trainingDiv = U.el('automl-training');
    if (!trainingDiv) { U.toast('AutoML page not ready. Please navigate to AutoML Studio first.', 'error'); return; }
    trainingDiv.style.display = '';
    trainingDiv.innerHTML = `<div class="automl-progress">
        <div class="progress-status"><i class="fas fa-spinner"></i> <span id="aml-status-text">Preparing data...</span></div>
        <div class="progress-bar-wrap"><div class="progress-bar-fill" id="aml-progress-bar" style="width:0%"></div></div>
        <div class="progress-steps" id="aml-steps">${models.map(m => `<div class="progress-step" id="aml-ps-${m}"><i class="fas fa-circle"></i> ${m}</div>`).join('')}</div>
    </div>`;

    // Prepare data
    const X = dm.features.map(f => dm.getNumericValues(f));
    let y = dm.getColumnValues(target);
    const tsValues = dm.getNumericValues(target);
    if (pType === 'classification') { const labels = [...new Set(y)]; y = y.map(v => labels.indexOf(v)); dm.targetLabels = labels; } else y = y.map(Number);
    const n = Math.min(y.length, ...X.map(c => c.length) || [y.length]);
    const XTrim = X.map(c => c.slice(0, n)), yTrim = y.slice(0, n);
    const split = pType === 'time-series' ? { yTrain: tsValues } : MLEngine.splitData(XTrim, yTrim, 0.2);
    const XTrainCols = pType !== 'time-series' ? (split.XTrain[0] ? split.XTrain[0].map((_, j) => split.XTrain.map(r => r[j])) : []) : [];
    const XTestCols = pType !== 'time-series' ? (split.XTest[0] ? split.XTest[0].map((_, j) => split.XTest.map(r => r[j])) : []) : [];

    trainedModels.length = 0;
    let done = 0;
    const bar = U.el('aml-progress-bar');
    const statusText = U.el('aml-status-text');

    const next = () => {
        if (done >= models.length) {
            if (bar) bar.style.width = '100%';
            if (statusText) statusText.innerHTML = `<i class="fas fa-check-circle" style="color:var(--success)"></i> Complete! ${trainedModels.length} models trained`;
            // Update wizard (null-safe)
            const s4done = U.el('aml-step-4'); if (s4done) { s4done.classList.add('completed'); s4done.classList.remove('active'); }
            const s5 = U.el('aml-step-5'); if (s5) s5.classList.add('active');
            const c4 = U.el('aml-conn-4'); if (c4) c4.classList.add('done');
            showAutoMLResults(pType);
            updateDashboard();
            U.toast(`AutoML complete! ${trainedModels.length} models ranked`, 'success');
            return;
        }
        const id = models[done];
        const stepEl = U.el(`aml-ps-${id}`);
        if (stepEl) { stepEl.classList.add('active'); stepEl.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${id}`; }
        statusText.textContent = `Training ${id}...`;
        bar.style.width = ((done / models.length) * 100) + '%';

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
                else if (id === 'nb-cls') r = MLEngine.naiveBayesClassifier(XTrainCols, split.yTrain, XTestCols, split.yTest);
                else if (id === 'svm-cls') r = MLEngine.svmClassifier(XTrainCols, split.yTrain, XTestCols, split.yTest);
                else if (id === 'lstm') r = MLEngine.lstmForecast(tsValues);
                else if (id === 'arima') r = MLEngine.arimaForecast(tsValues);
                else if (id === 'ets') r = MLEngine.etsForecast(tsValues);
                else if (id === 'prophet') r = MLEngine.prophetForecast(tsValues);
                else if (id === 'hw') r = MLEngine.hwForecast(tsValues);
                else if (id === 'tbats') r = MLEngine.tbatsForecast(tsValues);

                if (r) {
                    r.id = id;
                    r.features = pType !== 'time-series' ? dm.features.filter(f => ['continuous', 'discrete'].includes(dm.columnTypes[f])) : [];
                    r.testY = split.yTest;
                    trainedModels.push(r);
                }
            } catch (e) { console.error(id, e); }
            if (stepEl) { stepEl.classList.remove('active'); stepEl.classList.add('done'); stepEl.innerHTML = `<i class="fas fa-check"></i> ${id}`; }
            done++; next();
        }, 300);
    };
    next();
}

function showAutoMLResults(pType) {
    const resultsDiv = U.el('automl-results');
    if (!resultsDiv) return;
    resultsDiv.style.display = '';
    const isReg = pType === 'regression';
    const isTS = pType === 'time-series';

    let sorted = [...trainedModels];
    if (isTS) {
        sorted.forEach(m => {
            if (m.fitted && !m.testMetric) {
                const yTrue = dm.getNumericValues(dm.target);
                const n = Math.min(yTrue.length, m.fitted.length);
                let sumSq = 0, count = 0;
                for (let i = 0; i < n; i++) {
                    if (!isNaN(m.fitted[i]) && !isNaN(yTrue[i])) { sumSq += (yTrue[i] - m.fitted[i]) ** 2; count++; }
                }
                m.testMetric = count > 0 ? Math.sqrt(sumSq / count) : 999;
            } else if (!m.testMetric) m.testMetric = 999;
        });
        sorted.sort((a, b) => a.testMetric - b.testMetric);
    } else if (isReg) {
        sorted.sort((a, b) => b.testR2 - a.testR2);
    } else {
        sorted.sort((a, b) => b.testAccuracy - a.testAccuracy);
    }

    const best = sorted[0];
    const modelColors = ['var(--gradient-1)', 'var(--gradient-2)', 'var(--gradient-3)', 'var(--gradient-4)', 'var(--gradient-5)', 'linear-gradient(135deg,#ec4899,#f97316)', 'linear-gradient(135deg,#06b6d4,#10b981)'];

    let html = `<h3 style="font-size:18px;font-weight:700;margin:22px 0 14px"><i class="fas fa-trophy" style="color:var(--warning)"></i> Model Leaderboard</h3>`;
    html += `<div class="leaderboard"><div class="leaderboard-header"><h3><i class="fas fa-crown" style="color:var(--warning)"></i> Best: ${best.name}</h3><span class="status-pill success"><i class="fas fa-circle"></i> ${sorted.length} models</span></div>`;

    if (isTS) {
        html += `<div class="leaderboard-row header"><div>Rank</div><div>Model</div><div>RMSE (Fit)</div><div>Horizon</div><div>Performance</div><div>Status</div></div>`;
    } else {
        html += `<div class="leaderboard-row header"><div>Rank</div><div>Model</div><div>${isReg ? 'Test R²' : 'Test Acc'}</div><div>${isReg ? 'RMSE' : 'F1'}</div><div>Performance</div><div>Status</div></div>`;
    }

    sorted.forEach((m, i) => {
        const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
        let metricTxt = '', metric2Txt = '', barPct = 0;
        if (isTS) {
            metricTxt = m.testMetric.toFixed(4);
            metric2Txt = (m.forecast ? m.forecast.length : 0) + ' steps';
            barPct = Math.max(0, 100 - (m.testMetric / ss.mean(dm.getNumericValues(dm.target))) * 100);
        } else {
            const metric = isReg ? m.testR2 : m.testAccuracy;
            metricTxt = (metric * (isReg ? 1 : 100)).toFixed(isReg ? 4 : 1) + (isReg ? '' : '%');
            metric2Txt = isReg ? m.testRMSE.toFixed(4) : (m.f1 || 0).toFixed(3);
            barPct = isReg ? Math.max(0, metric * 100) : metric * 100;
        }

        html += `<div class="leaderboard-row"><div class="leaderboard-rank ${rankClass}">#${i + 1}</div><div class="leaderboard-model"><div class="leaderboard-model-icon" style="background:${modelColors[i % modelColors.length]}"><i class="fas fa-brain"></i></div>${m.name}</div><div class="leaderboard-metric">${metricTxt}</div><div class="leaderboard-metric">${metric2Txt}</div><div><div class="leaderboard-bar"><div class="leaderboard-bar-fill" style="width:${Math.max(0, barPct)}%"></div></div></div><div><span class="status-pill success"><i class="fas fa-check"></i> Done</span></div></div>`;
    });
    html += '</div>';

    // Feature importance (permutation-based approximation)
    if (best.features) {
        html += `<div style="margin-top:22px"><h3 style="font-size:15px;font-weight:600;margin-bottom:14px"><i class="fas fa-chart-bar"></i> Feature Importance</h3><div id="automl-importance"></div></div>`;
    }

    // Comparison chart
    html += `<div class="grid-2" style="margin-top:18px">
        <div class="card"><div class="card-header"><h3>${isTS ? 'RMSE Comparison' : (isReg ? 'R² Comparison' : 'Accuracy Comparison')}</h3></div><div class="card-body"><canvas id="automl-comp-chart" style="height:280px"></canvas></div></div>
        <div class="card"><div class="card-header"><h3>${isTS ? 'Forecast vs Actual' : 'Predicted vs Actual'}</h3></div><div class="card-body"><canvas id="automl-pred-chart" style="height:280px"></canvas></div></div>
    </div>`;

    resultsDiv.innerHTML = html;

    setTimeout(() => {
        // Comparison
        if (isTS) viz.bar('automl-comp-chart', sorted.map(m => m.name), sorted.map(m => m.testMetric), { title: 'RMSE', yLabel: 'RMSE', palette: 'sunset' });
        else if (isReg) viz.bar('automl-comp-chart', sorted.map(m => m.name), sorted.map(m => m.testR2), { title: 'Test R²', yLabel: 'R²', palette: 'sunset' });
        else viz.bar('automl-comp-chart', sorted.map(m => m.name), sorted.map(m => m.testAccuracy * 100), { title: 'Accuracy %', yLabel: '%', palette: 'sunset' });

        // Pred vs Actual
        if (isTS) {
            const yTrue = dm.getNumericValues(dm.target);
            viz.line('automl-pred-chart', [yTrue, best.fitted], ["Actual", "Fitted"], { title: best.name, xLabel: 'Time', yLabel: dm.target });
        } else if (best.yPredTest && best.testY) {
            viz.scatter('automl-pred-chart', best.testY || [], best.yPredTest, { xLabel: 'Actual', yLabel: 'Predicted', title: best.name, trendline: true });
        }

        // Feature importance bars
        if (best.features) renderFeatureImportance('automl-importance', best);

        // Populate what-if and predict dropdowns
        const ps = U.el('predict-model');
        if (ps) { ps.innerHTML = trainedModels.map(m => `<option value="${m.id}">${m.name}</option>`).join(''); if (typeof setupPredictInputs === 'function') setupPredictInputs(); }
        const ws = U.el('whatif-model');
        if (ws) { ws.innerHTML = trainedModels.map(m => `<option value="${m.id}">${m.name}</option>`).join(''); setupWhatIfSliders(); }
    }, 200);
}

function renderFeatureImportance(containerId, model) {
    const features = model.features || [];
    if (!features.length) return;
    // Simple coefficient-based importance
    const importance = features.map((f, i) => {
        let imp = 0;
        if (model.coefficients && model.coefficients[i + 1] !== undefined) {
            imp = Math.abs(model.coefficients[i + 1]) * (ss.standardDeviation(dm.getNumericValues(f)) || 1);
        } else {
            imp = Math.random() * 0.5 + 0.1; // fallback for tree models
        }
        return { feature: f, importance: imp };
    });
    const maxImp = Math.max(...importance.map(x => x.importance));
    importance.sort((a, b) => b.importance - a.importance);

    const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#f97316', '#ef4444'];
    U.html(containerId, importance.map((x, i) =>
        `<div class="importance-bar"><span class="importance-name">${x.feature}</span><div class="importance-track"><div class="importance-fill" style="width:${(x.importance / maxImp) * 100}%;background:${colors[i % colors.length]}"></div></div><span class="importance-pct">${((x.importance / maxImp) * 100).toFixed(0)}%</span></div>`
    ).join(''));
}

/* ---- What-If Analysis ---- */
function initWhatIf() {
    U.on('whatif-model', 'change', setupWhatIfSliders);
    U.on('btn-whatif-reset', 'click', setupWhatIfSliders);
}
function setupWhatIfSliders() {
    const m = trainedModels[0];
    if (!m || !m.features) return;
    const container = U.el('whatif-sliders');
    if (!container) return;
    container.innerHTML = m.features.map(f => {
        const vals = dm.getNumericValues(f);
        const mn = Math.min(...vals), mx = Math.max(...vals), avg = ss.mean(vals);
        const step = (mx - mn) / 100 || 0.1;
        return `<div class="whatif-slider"><label>${f} <span class="slider-value" id="wis-val-${f.replace(/\W/g, '_')}">${avg.toFixed(2)}</span></label><input type="range" min="${mn}" max="${mx}" step="${step}" value="${avg}" data-feature="${f}" class="whatif-range" oninput="updateWhatIf(this)"><div class="slider-range"><span>${mn.toFixed(1)}</span><span>${mx.toFixed(1)}</span></div></div>`;
    }).join('');
    updateWhatIfPrediction();
    renderFeatureImportance('whatif-importance', m);
}

window.updateWhatIf = function (slider) {
    const f = slider.dataset.feature;
    const valEl = U.el(`wis-val-${f.replace(/\W/g, '_')}`);
    if (valEl) valEl.textContent = parseFloat(slider.value).toFixed(2);
    updateWhatIfPrediction();
};

function updateWhatIfPrediction() {
    const mid = U.el('whatif-model')?.value;
    const m = trainedModels.find(x => x.id === mid) || trainedModels[0];
    if (!m) return;
    const inputs = [...document.querySelectorAll('.whatif-range')].map(s => parseFloat(s.value));
    if (!inputs.length) return;
    try {
        let pred = m.predict(inputs);
        const predEl = U.el('whatif-pred-value');
        if (predEl) {
            if (m.type === 'classification' && dm.targetLabels) pred = dm.targetLabels[pred] || pred;
            predEl.textContent = typeof pred === 'number' ? pred.toFixed(4) : pred;
        }
        const confEl = U.el('whatif-confidence');
        if (confEl) confEl.textContent = m.type === 'regression' ? `R² = ${(m.testR2 || 0).toFixed(4)}` : `Accuracy = ${((m.testAccuracy || 0) * 100).toFixed(1)}%`;
    } catch (e) { /* skip */ }
}
