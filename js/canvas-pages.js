/* ============================================
   Canvas Pages - Pipeline, Quick Insights, AutoML, What-If
   ============================================ */
const CanvasPages = {
    dashboard: () => `
        <div class="notification-bar" id="welcome-bar"><i class="fas fa-sparkles"></i>
            <span>Welcome to <strong>IndustryAI Canvas</strong> — your no-code ML & analytics studio. Start by importing data or loading a sample dataset.</span>
            <span class="dismiss" onclick="this.parentElement.style.display='none'"><i class="fas fa-times"></i></span>
        </div>
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-1)"><i class="fas fa-database"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-rows">0</span><span class="kpi-label">Rows</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-2)"><i class="fas fa-columns"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-cols">0</span><span class="kpi-label">Features</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-3)"><i class="fas fa-check-circle"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-quality">–</span><span class="kpi-label">Data Quality</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-4)"><i class="fas fa-brain"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-models">0</span><span class="kpi-label">Models</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-5)"><i class="fas fa-flask"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-analyses">0</span><span class="kpi-label">Analyses</span></div></div>
        </div>
        <div class="welcome-grid">
            <div class="welcome-card" onclick="navigateTo('data-sources')"><div class="wc-icon" style="background:var(--gradient-1)"><i class="fas fa-cloud-upload-alt"></i></div><h3>Import Data</h3><p>Upload CSV, JSON, paste data, or try sample datasets. Multi-source ingestion.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Get Started</div></div>
            <div class="welcome-card" onclick="navigateTo('pipeline')"><div class="wc-icon" style="background:var(--gradient-5)"><i class="fas fa-sitemap"></i></div><h3>Pipeline Builder</h3><p>Drag-and-drop visual pipeline. Connect data → transform → model → deploy.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Build Pipeline</div></div>
            <div class="welcome-card" onclick="navigateTo('automl')"><div class="wc-icon" style="background:var(--gradient-2)"><i class="fas fa-robot"></i></div><h3>AutoML Studio</h3><p>One-click model building. SageMaker Canvas-style automated ML training.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Launch AutoML</div></div>
            <div class="welcome-card" onclick="navigateTo('quick-insights')"><div class="wc-icon" style="background:var(--gradient-3)"><i class="fas fa-magic"></i></div><h3>Quick Insights</h3><p>AI-powered automatic analysis. Discover patterns, anomalies &amp; recommendations.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Explore</div></div>
            <div class="welcome-card" onclick="navigateTo('viz-builder')"><div class="wc-icon" style="background:var(--gradient-4)"><i class="fas fa-palette"></i></div><h3>Visual Builder</h3><p>Drag-and-drop chart creation. Power BI / R Esquisse style visualization.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Create Visuals</div></div>
            <div class="welcome-card" onclick="navigateTo('what-if')"><div class="wc-icon" style="background:linear-gradient(135deg,#ec4899,#f97316)"><i class="fas fa-sliders-h"></i></div><h3>What-If Analysis</h3><p>Interactive sliders to explore model predictions. Real-time sensitivity.</p><div class="wc-action"><i class="fas fa-arrow-right"></i> Try It</div></div>
        </div>
        <div style="margin-top:22px"><h3 style="font-size:15px;font-weight:600;margin-bottom:12px"><i class="fas fa-bolt"></i> Quick Start — Sample Datasets</h3>
        <div class="sample-grid">
            <button class="sample-card" onclick="loadSampleData('manufacturing')"><div class="sample-icon" style="background:var(--gradient-1)"><i class="fas fa-industry"></i></div><div class="sample-info"><h4>Manufacturing</h4><p>500 rows • 13 features</p><span class="sample-tag">Process</span></div></button>
            <button class="sample-card" onclick="loadSampleData('quality')"><div class="sample-icon" style="background:var(--gradient-3)"><i class="fas fa-check-double"></i></div><div class="sample-info"><h4>Quality Control</h4><p>300 rows • 8 features</p><span class="sample-tag">Inspect</span></div></button>
            <button class="sample-card" onclick="loadSampleData('timeseries')"><div class="sample-icon" style="background:var(--gradient-4)"><i class="fas fa-wave-square"></i></div><div class="sample-info"><h4>Sensor Data</h4><p>1000 rows • IoT stream</p><span class="sample-tag">TimeSeries</span></div></button>
            <button class="sample-card" onclick="loadSampleData('classification')"><div class="sample-icon" style="background:var(--gradient-2)"><i class="fas fa-users"></i></div><div class="sample-info"><h4>Classification</h4><p>400 rows • ML ready</p><span class="sample-tag">ML</span></div></button>
        </div></div>
        <div class="grid-2" style="margin-top:18px">
            <div class="card"><div class="card-header"><h3><i class="fas fa-chart-line"></i> Data Snapshot</h3></div><div class="card-body"><div id="dash-overview" class="empty-state" style="min-height:200px"><i class="fas fa-database"></i><h3>No Data</h3></div><canvas id="dash-overview-chart" style="display:none;height:220px"></canvas></div></div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-chart-pie"></i> Feature Types</h3></div><div class="card-body"><canvas id="dash-types-chart" style="height:220px"></canvas></div></div>
        </div>`,

    pipeline: () => `
        <div class="wizard" id="pipeline-wizard">
            <div class="wizard-step completed" onclick="navigateTo('data-sources')"><div class="step-num"><span>1</span></div><span>Data Source</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step" onclick="navigateTo('data-cleaning')"><div class="step-num"><span>2</span></div><span>Clean & Prep</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step" onclick="navigateTo('column-ops')"><div class="step-num"><span>3</span></div><span>Feature Eng.</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step" onclick="navigateTo('automl')"><div class="step-num"><span>4</span></div><span>Model Build</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step"><div class="step-num"><span>5</span></div><span>Evaluate</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step"><div class="step-num"><span>6</span></div><span>Deploy</span></div>
        </div>
        <h3 style="font-size:15px;font-weight:600;margin-bottom:12px"><i class="fas fa-puzzle-piece"></i> Drag Nodes to Canvas</h3>
        <div class="node-palette" id="node-palette">
            <div class="palette-item" data-node="csv"><i class="fas fa-file-csv" style="color:var(--accent)"></i> CSV Import</div>
            <div class="palette-item" data-node="api"><i class="fas fa-plug" style="color:var(--purple)"></i> API Source</div>
            <div class="palette-item" data-node="filter"><i class="fas fa-filter" style="color:var(--info)"></i> Filter</div>
            <div class="palette-item" data-node="clean"><i class="fas fa-broom" style="color:var(--success)"></i> Clean</div>
            <div class="palette-item" data-node="transform"><i class="fas fa-exchange-alt" style="color:var(--warning)"></i> Transform</div>
            <div class="palette-item" data-node="feature"><i class="fas fa-columns" style="color:var(--pink)"></i> Feature Eng</div>
            <div class="palette-item" data-node="split"><i class="fas fa-random" style="color:var(--accent)"></i> Train/Test Split</div>
            <div class="palette-item" data-node="model"><i class="fas fa-brain" style="color:var(--purple)"></i> ML Model</div>
            <div class="palette-item" data-node="evaluate"><i class="fas fa-trophy" style="color:var(--warning)"></i> Evaluate</div>
            <div class="palette-item" data-node="deploy"><i class="fas fa-rocket" style="color:var(--success)"></i> Deploy</div>
        </div>
        <div class="pipeline-canvas" id="pipeline-canvas">
            <div class="pipeline-toolbar">
                <button class="btn btn-sm btn-primary" id="btn-run-pipeline"><i class="fas fa-play"></i> Run Pipeline</button>
                <button class="btn btn-sm" id="btn-clear-pipeline"><i class="fas fa-trash"></i> Clear</button>
                <div class="toolbar-divider"></div>
                <span style="font-size:11px;color:var(--text-muted)">Drag nodes from palette into canvas</span>
                <span id="pipeline-status" class="status-pill idle" style="margin-left:auto"><i class="fas fa-circle"></i> Idle</span>
            </div>
            <div class="pipeline-nodes" id="pipeline-nodes"></div>
        </div>`,

    quickInsights: () => `
        <div class="notification-bar"><i class="fas fa-robot"></i>
            <span><strong>AI-Powered Insights</strong> — Select a target column for focused pattern detection, correlation analysis, and actionable recommendations.</span>
        </div>
        <div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:20px;flex-wrap:wrap">
            <div class="form-group" style="margin:0;min-width:220px">
                <label style="font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.5px">Target Variable</label>
                <select class="form-control var-dropdown" id="insights-target" style="border-color:var(--accent);margin-top:4px"><option value="">Select target column...</option></select>
            </div>
            <button class="btn btn-primary btn-lg" id="btn-gen-insights"><i class="fas fa-magic"></i> Generate Insights</button>
        </div>
        <div id="insights-container"><div class="empty-state"><i class="fas fa-magic"></i><h3>Quick Insights</h3><p>Select a target column and click Generate Insights</p></div></div>`,

    automl: () => `
        <div class="wizard" id="automl-wizard">
            <div class="wizard-step active" id="aml-step-1"><div class="step-num"><span>1</span></div><span>Select Data</span></div>
            <div class="wizard-connector" id="aml-conn-1"></div>
            <div class="wizard-step" id="aml-step-2"><div class="step-num"><span>2</span></div><span>Choose Target</span></div>
            <div class="wizard-connector" id="aml-conn-2"></div>
            <div class="wizard-step" id="aml-step-3"><div class="step-num"><span>3</span></div><span>Build Type</span></div>
            <div class="wizard-connector" id="aml-conn-3"></div>
            <div class="wizard-step" id="aml-step-4"><div class="step-num"><span>4</span></div><span>Training</span></div>
            <div class="wizard-connector" id="aml-conn-4"></div>
            <div class="wizard-step" id="aml-step-5"><div class="step-num"><span>5</span></div><span>Results</span></div>
        </div>
        <div id="automl-step-content">
            <div class="automl-hero"><h2>AutoML Studio</h2><p>Build production-quality machine learning models with zero code. Just select your target and click build.</p>
                <div class="grid-2" style="max-width:700px;margin:0 auto">
                    <div class="form-group"><label>Target Variable</label><select class="form-control var-dropdown" id="aml-target" style="font-size:14px;padding:12px"><option value="">Select target column...</option></select></div>
                    <div class="form-group"><label>Problem Type</label><select class="form-control" id="aml-problem" style="font-size:14px;padding:12px"><option value="auto">Auto-detect</option><option value="regression">Regression</option><option value="classification">Classification</option></select></div>
                </div>
            </div>
            <h3 style="font-size:15px;font-weight:600;margin-bottom:14px"><i class="fas fa-bolt"></i> Choose Build Type</h3>
            <div class="build-options">
                <div class="build-option selected" data-build="quick" id="opt-quick"><div class="option-icon" style="background:var(--gradient-2)"><i class="fas fa-bolt"></i></div><h3>Quick Build</h3><p>Fast training with top 3 algorithms. Best for rapid prototyping.</p><div class="option-time"><i class="fas fa-clock"></i> ~5 seconds</div></div>
                <div class="build-option" data-build="standard" id="opt-standard"><div class="option-icon" style="background:var(--gradient-5)"><i class="fas fa-microscope"></i></div><h3>Standard Build</h3><p>Full sweep of 7+ algorithms with hyperparameter tuning.</p><div class="option-time"><i class="fas fa-clock"></i> ~15 seconds</div></div>
            </div>
            <div style="text-align:center;margin-top:4px"><button class="btn btn-primary btn-lg" id="btn-automl-build" style="padding:14px 48px;font-size:15px"><i class="fas fa-rocket"></i> Start AutoML Build</button></div>
        </div>
        <div id="automl-training" style="display:none"></div>
        <div id="automl-results" style="display:none"></div>`,

    whatIf: () => `
        <div class="wizard">
            <div class="wizard-step completed"><div class="step-num"><span>1</span></div><span>Train Model</span></div>
            <div class="wizard-connector done"></div>
            <div class="wizard-step active"><div class="step-num"><span>2</span></div><span>What-If Analysis</span></div>
            <div class="wizard-connector"></div>
            <div class="wizard-step"><div class="step-num"><span>3</span></div><span>Generate Report</span></div>
        </div>
        <div class="whatif-container">
            <div class="whatif-panel">
                <h3 style="font-size:14px;font-weight:700;margin-bottom:4px"><i class="fas fa-sliders-h"></i> Feature Controls</h3>
                <p style="font-size:11px;color:var(--text-secondary);margin-bottom:14px">Adjust sliders to see how each feature affects the prediction in real-time.</p>
                <div class="form-group"><label>Model</label><select class="form-control" id="whatif-model"></select></div>
                <div id="whatif-sliders"></div>
                <div class="whatif-prediction" id="whatif-pred-box">
                    <div class="pred-label">Predicted Value</div>
                    <div class="pred-value" id="whatif-pred-value">—</div>
                    <div class="pred-confidence" id="whatif-confidence"></div>
                </div>
                <button class="btn btn-primary btn-block" id="btn-whatif-reset" style="margin-top:12px"><i class="fas fa-undo"></i> Reset to Defaults</button>
            </div>
            <div class="whatif-results" id="whatif-results">
                <div class="whatif-chart-card"><h4><i class="fas fa-chart-bar"></i> Feature Importance</h4><div id="whatif-importance"></div></div>
                <div class="whatif-chart-card"><h4><i class="fas fa-chart-line"></i> Sensitivity Analysis</h4><canvas id="whatif-sensitivity-chart" style="height:250px"></canvas></div>
                <div class="whatif-chart-card"><h4><i class="fas fa-chart-area"></i> Prediction Distribution</h4><canvas id="whatif-dist-chart" style="height:250px"></canvas></div>
            </div>
        </div>`,
};
