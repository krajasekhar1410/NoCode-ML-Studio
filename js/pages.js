/* ============================================
   Page Renderers - All page HTML content
   ============================================ */
const Pages = {
    dashboard: () => `
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-1)"><i class="fas fa-database"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-rows">0</span><span class="kpi-label">Rows</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-2)"><i class="fas fa-columns"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-cols">0</span><span class="kpi-label">Variables</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-3)"><i class="fas fa-check-circle"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-quality">–</span><span class="kpi-label">Data Quality</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-4)"><i class="fas fa-brain"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-models">0</span><span class="kpi-label">Models Trained</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-5)"><i class="fas fa-flask"></i></div><div class="kpi-info"><span class="kpi-value" id="kpi-analyses">0</span><span class="kpi-label">Analyses Run</span></div></div>
        </div>
        <div class="grid-2">
            <div class="card"><div class="card-header"><h3><i class="fas fa-chart-line"></i> Data Overview</h3></div><div class="card-body"><div id="dash-overview" class="empty-state"><i class="fas fa-database"></i><h3>No Data Loaded</h3><p>Import data or load a sample dataset to get started</p></div><canvas id="dash-overview-chart" style="display:none;height:250px"></canvas></div></div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-chart-pie"></i> Variable Types</h3></div><div class="card-body"><canvas id="dash-types-chart" style="height:250px"></canvas></div></div>
        </div>
        <div style="margin-top:18px"><h3 style="font-size:15px;font-weight:600;margin-bottom:12px"><i class="fas fa-rocket"></i> Quick Actions</h3>
        <div class="sample-grid">
            <button class="sample-card" onclick="loadSampleData('manufacturing')"><div class="sample-icon" style="background:var(--gradient-1)"><i class="fas fa-industry"></i></div><div class="sample-info"><h4>Manufacturing</h4><p>500 rows • 13 variables</p><span class="sample-tag">Process Data</span></div></button>
            <button class="sample-card" onclick="loadSampleData('quality')"><div class="sample-icon" style="background:var(--gradient-3)"><i class="fas fa-check-double"></i></div><div class="sample-info"><h4>Quality Control</h4><p>300 rows • 8 variables</p><span class="sample-tag">Inspection</span></div></button>
            <button class="sample-card" onclick="loadSampleData('timeseries')"><div class="sample-icon" style="background:var(--gradient-4)"><i class="fas fa-wave-square"></i></div><div class="sample-info"><h4>Sensor Time Series</h4><p>1000 rows • 6 variables</p><span class="sample-tag">IoT Data</span></div></button>
            <button class="sample-card" onclick="loadSampleData('experiment')"><div class="sample-icon" style="background:var(--gradient-5)"><i class="fas fa-vials"></i></div><div class="sample-info"><h4>DOE Experiment</h4><p>32 rows • 5 variables</p><span class="sample-tag">Factorial</span></div></button>
            <button class="sample-card" onclick="loadSampleData('classification')"><div class="sample-icon" style="background:var(--gradient-2)"><i class="fas fa-users"></i></div><div class="sample-info"><h4>Classification</h4><p>400 rows • 8 variables</p><span class="sample-tag">ML Ready</span></div></button>
        </div></div>`,

    dataSources: () => `
        <div class="drop-zone" id="drop-zone">
            <i class="fas fa-cloud-upload-alt"></i>
            <h3>Drop Files Here</h3>
            <p>Drag and drop CSV, JSON, TSV, or TXT files</p>
            <input type="file" id="file-input" accept=".csv,.json,.tsv,.txt,.xlsx" hidden>
            <button class="btn btn-primary" onclick="document.getElementById('file-input').click()"><i class="fas fa-folder-open"></i> Browse Files</button>
        </div>
        <h3 style="font-size:15px;font-weight:600;margin-bottom:12px"><i class="fas fa-plug"></i> Data Sources</h3>
        <div class="source-grid">
            <button class="source-card" onclick="document.getElementById('file-input').click()"><i class="fas fa-file-csv"></i>CSV / TSV File</button>
            <button class="source-card" onclick="document.getElementById('file-input').click()"><i class="fas fa-file-code"></i>JSON File</button>
            <button class="source-card" id="btn-paste-data"><i class="fas fa-paste"></i>Paste Data</button>
            <button class="source-card" id="btn-url-import"><i class="fas fa-link"></i>URL Import</button>
            <button class="source-card" id="btn-clipboard"><i class="fas fa-clipboard"></i>From Clipboard</button>
            <button class="source-card" onclick="loadSampleData('manufacturing')"><i class="fas fa-flask"></i>Sample Data</button>
        </div>
        <div class="grid-3" style="margin-top:18px">
            <div class="form-group"><label>Delimiter</label><select class="form-control" id="delimiter-select"><option value="auto">Auto-detect</option><option value=",">Comma</option><option value=";">Semicolon</option><option value="&#9;">Tab</option><option value="|">Pipe</option></select></div>
            <div class="form-group"><label>Header Row</label><div class="toggle"><input type="checkbox" id="header-toggle" checked><label for="header-toggle" class="toggle-track"></label><span>First row as header</span></div></div>
            <div class="form-group"><label>Skip Rows</label><input type="number" class="form-control" id="skip-rows" value="0" min="0"></div>
        </div>
        <div id="paste-area" style="display:none" class="card"><div class="card-header"><h3>Paste Data</h3></div><div class="card-body"><textarea class="form-control" id="paste-text" placeholder="Paste CSV, JSON, or tab-delimited data here..." style="min-height:160px;font-family:'JetBrains Mono',monospace"></textarea><button class="btn btn-primary" id="btn-paste-load" style="margin-top:10px"><i class="fas fa-check"></i> Load Data</button></div></div>
        <div id="url-area" style="display:none" class="card" style="margin-top:14px"><div class="card-header"><h3>URL Import</h3></div><div class="card-body"><div class="form-inline"><div class="form-group" style="flex:4"><input class="form-control" id="url-input" placeholder="https://example.com/data.csv"></div><div class="form-group" style="flex:1"><button class="btn btn-primary btn-block" id="btn-url-load"><i class="fas fa-download"></i> Fetch</button></div></div></div></div>`,

    dataProfiler: () => `<div id="profiler-content"><div class="empty-state"><i class="fas fa-microscope"></i><h3>Data Profiler</h3><p>Load data to see detailed quality report with error percentages</p></div></div>`,

    dataView: () => `
        <div style="display:flex;flex-direction:column;height:calc(100vh - var(--topbar-h) - 40px)">
            <div class="data-toolbar">
                <div class="toolbar-left">
                    <button class="btn btn-sm" id="btn-add-row"><i class="fas fa-plus"></i> Add Row</button>
                    <button class="btn btn-sm" id="btn-delete-rows"><i class="fas fa-trash"></i> Delete</button>
                    <div class="toolbar-divider"></div>
                    <button class="btn btn-sm" id="btn-sort-asc"><i class="fas fa-sort-amount-down-alt"></i></button>
                    <button class="btn btn-sm" id="btn-sort-desc"><i class="fas fa-sort-amount-up"></i></button>
                    <div class="toolbar-divider"></div>
                    <button class="btn btn-sm" id="btn-filter-data"><i class="fas fa-filter"></i> Filter</button>
                    <button class="btn btn-sm" id="btn-search-data"><i class="fas fa-search"></i> Search</button>
                </div>
                <div class="toolbar-right">
                    <span id="data-info-text" style="font-size:11px;color:var(--text-secondary)">No data</span>
                    <button class="btn btn-sm btn-primary" id="btn-export-csv"><i class="fas fa-download"></i> Export CSV</button>
                </div>
            </div>
            <div class="data-table-wrapper" id="table-wrapper"><table class="data-table" id="data-table"><thead id="data-table-head"></thead><tbody id="data-table-body"></tbody></table></div>
            <div class="data-pagination" id="data-pagination" style="display:none">
                <button class="btn btn-sm" id="prev-page"><i class="fas fa-chevron-left"></i></button>
                <span id="page-info" style="font-size:12px">Page 1</span>
                <button class="btn btn-sm" id="next-page"><i class="fas fa-chevron-right"></i></button>
                <select class="form-control" id="rows-per-page" style="width:100px"><option value="25">25 rows</option><option value="50" selected>50 rows</option><option value="100">100 rows</option><option value="all">All</option></select>
            </div>
        </div>`,

    dataCleaning: () => `
        <div class="grid-sidebar-wide">
            <div style="overflow-y:auto">
                <div class="cleaning-card"><h4><i class="fas fa-band-aid"></i> Handle Missing Values</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="clean-col"></select></div>
                    <div class="form-group"><label>Method</label><select class="form-control" id="clean-method"><option value="mean">Fill with Mean</option><option value="median">Fill with Median</option><option value="mode">Fill with Mode</option><option value="zero">Fill with Zero</option><option value="ffill">Forward Fill</option><option value="bfill">Backward Fill</option><option value="drop">Drop Missing Rows</option></select></div>
                    <button class="btn btn-primary btn-block" id="btn-fill-missing"><i class="fas fa-medkit"></i> Apply</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-copy"></i> Remove Duplicates</h4>
                    <p style="font-size:11px;color:var(--text-secondary);margin-bottom:10px">Remove duplicate rows based on all columns</p>
                    <button class="btn btn-primary btn-block" id="btn-remove-dupes"><i class="fas fa-broom"></i> Remove Duplicates</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-times-circle"></i> Remove Outliers</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="outlier-col"></select></div>
                    <div class="form-group"><label>Method</label><select class="form-control" id="outlier-method"><option value="iqr">IQR (1.5×)</option><option value="zscore">Z-Score (3σ)</option></select></div>
                    <button class="btn btn-danger btn-block" id="btn-remove-outliers"><i class="fas fa-cut"></i> Remove Outliers</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-exchange-alt"></i> Convert Type</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="convert-col"></select></div>
                    <div class="form-group"><label>To Type</label><select class="form-control" id="convert-type"><option value="number">Numeric</option><option value="string">Text</option><option value="date">DateTime</option></select></div>
                    <button class="btn btn-primary btn-block" id="btn-convert-type"><i class="fas fa-sync"></i> Convert</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-text-width"></i> Trim Whitespace</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="trim-col"></select></div>
                    <button class="btn btn-primary btn-block" id="btn-trim"><i class="fas fa-text-width"></i> Trim</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-pencil-alt"></i> Find & Replace</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="replace-col"></select></div>
                    <div class="form-inline"><div class="form-group"><label>Find</label><input class="form-control" id="find-val"></div><div class="form-group"><label>Replace</label><input class="form-control" id="replace-val"></div></div>
                    <button class="btn btn-primary btn-block" id="btn-replace-vals"><i class="fas fa-exchange-alt"></i> Replace</button>
                </div>
            </div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-eye"></i> Data Preview</h3></div><div class="card-body" style="overflow:auto"><div id="clean-preview">Apply a cleaning operation to see the preview</div></div></div>
        </div>`,

    dataTransform: () => `<div class="grid-sidebar-wide"><div style="overflow-y:auto"><div id="transform-ops"></div></div><div class="card"><div class="card-header"><h3>Preview</h3></div><div class="card-body" style="overflow:auto"><div id="transform-preview">Apply transforms to see results</div></div></div></div>`,

    columnOps: () => `
        <div class="grid-sidebar-wide">
            <div style="overflow-y:auto">
                <div class="cleaning-card"><h4><i class="fas fa-plus-circle"></i> Add Calculated Column</h4>
                    <div class="form-group"><label>Column Name</label><input class="form-control" id="calc-col-name" placeholder="new_column"></div>
                    <div class="formula-editor"><label style="font-size:10px;color:var(--text-secondary);text-transform:uppercase">Formula</label><textarea id="calc-formula" placeholder="e.g. Temperature * 1.8 + 32"></textarea>
                    <p class="formula-help">Use column names directly. Available: abs, sqrt, log, exp, pow, round, ceil, floor, sin, cos, min, max, PI, E, ROW, ROWNUM</p>
                    <div class="formula-tags" id="formula-col-tags"></div></div>
                    <button class="btn btn-primary btn-block" id="btn-add-calc-col"><i class="fas fa-plus"></i> Create Column</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-trash-alt"></i> Drop Columns</h4>
                    <div id="drop-col-list" style="max-height:200px;overflow-y:auto"></div>
                    <button class="btn btn-danger btn-block" id="btn-drop-cols" style="margin-top:10px"><i class="fas fa-trash"></i> Drop Selected</button>
                </div>
                <div class="cleaning-card"><h4><i class="fas fa-edit"></i> Rename Column</h4>
                    <div class="form-group"><label>Column</label><select class="form-control var-dropdown" id="rename-col"></select></div>
                    <div class="form-group"><label>New Name</label><input class="form-control" id="rename-new"></div>
                    <button class="btn btn-primary btn-block" id="btn-rename-col"><i class="fas fa-edit"></i> Rename</button>
                </div>
            </div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-eye"></i> Column Preview</h3></div><div class="card-body" style="overflow:auto"><div id="col-preview">Select operations to see the result</div></div></div>
        </div>`,

    vizBuilder: () => `
        <div class="viz-builder-container">
            <div class="viz-panel">
                <h3><i class="fas fa-th"></i> Variables</h3>
                <input class="form-control" id="var-search-input" placeholder="Search variables..." style="margin-bottom:8px">
                <div id="variable-list"><div style="color:var(--text-muted);font-size:11px;padding:12px">Load data to see variables</div></div>
            </div>
            <div class="viz-panel">
                <h3><i class="fas fa-arrows-alt"></i> Drag Variables to Map</h3>
                <div class="mapping-zone" id="drop-x" data-mapping="x"><span class="zone-label"><i class="fas fa-arrows-alt-h"></i> X Axis</span><span class="zone-content" id="zone-x">Drop variable here</span></div>
                <div class="mapping-zone" id="drop-y" data-mapping="y"><span class="zone-label"><i class="fas fa-arrows-alt-v"></i> Y Axis</span><span class="zone-content" id="zone-y">Drop variable here</span></div>
                <div class="mapping-zone" id="drop-color" data-mapping="color"><span class="zone-label"><i class="fas fa-palette"></i> Color</span><span class="zone-content" id="zone-color">Drop variable here</span></div>
                <div class="mapping-zone" id="drop-size" data-mapping="size"><span class="zone-label"><i class="fas fa-expand"></i> Size</span><span class="zone-content" id="zone-size">Drop variable here</span></div>
                <div class="mapping-zone" id="drop-facet" data-mapping="facet"><span class="zone-label"><i class="fas fa-th-large"></i> Facet</span><span class="zone-content" id="zone-facet">Drop variable here</span></div>
                <div class="mapping-zone" id="drop-label" data-mapping="label"><span class="zone-label"><i class="fas fa-font"></i> Label</span><span class="zone-content" id="zone-label">Drop variable here</span></div>
                <div class="chart-type-selector"><h4>Chart Type</h4><div class="chart-types">
                    <button class="chart-type-btn active" data-type="scatter" title="Scatter"><i class="fas fa-braille"></i></button>
                    <button class="chart-type-btn" data-type="line" title="Line"><i class="fas fa-chart-line"></i></button>
                    <button class="chart-type-btn" data-type="bar" title="Bar"><i class="fas fa-chart-bar"></i></button>
                    <button class="chart-type-btn" data-type="histogram" title="Histogram"><i class="fas fa-signal"></i></button>
                    <button class="chart-type-btn" data-type="box" title="Box"><i class="fas fa-square"></i></button>
                    <button class="chart-type-btn" data-type="pie" title="Pie"><i class="fas fa-chart-pie"></i></button>
                    <button class="chart-type-btn" data-type="area" title="Area"><i class="fas fa-mountain"></i></button>
                    <button class="chart-type-btn" data-type="bubble" title="Bubble"><i class="fas fa-circle"></i></button>
                    <button class="chart-type-btn" data-type="doughnut" title="Doughnut"><i class="fas fa-circle-notch"></i></button>
                    <button class="chart-type-btn" data-type="heatmap" title="Heatmap"><i class="fas fa-th"></i></button>
                </div></div>
                <div class="chart-options">
                    <h4>Options</h4>
                    <div class="form-group"><label>Title</label><input class="form-control" id="viz-title" placeholder="Chart title"></div>
                    <div class="form-group"><label>Palette</label><select class="form-control" id="viz-palette"><option value="default">Default</option><option value="viridis">Viridis</option><option value="plasma">Plasma</option><option value="industrial">Industrial</option><option value="ocean">Ocean</option><option value="sunset">Sunset</option></select></div>
                    <div class="form-inline">
                        <div class="form-group"><label>Opacity</label><input type="range" class="form-range" id="viz-opacity" min="0.1" max="1" step="0.1" value="0.8"></div>
                        <div class="form-group"><label>Point Size</label><input type="range" class="form-range" id="viz-point-size" min="1" max="10" step="1" value="4"></div>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px">
                        <div class="form-check"><input type="checkbox" id="viz-grid" checked> Grid</div>
                        <div class="form-check"><input type="checkbox" id="viz-legend" checked> Legend</div>
                        <div class="form-check"><input type="checkbox" id="viz-trendline"> Trendline</div>
                        <div class="form-check"><input type="checkbox" id="viz-smooth"> Smooth</div>
                    </div>
                </div>
                <div style="display:flex;gap:6px;margin-top:14px">
                    <button class="btn btn-primary" id="btn-update-viz" style="flex:1"><i class="fas fa-play"></i> Build</button>
                    <button class="btn" id="btn-clear-viz"><i class="fas fa-times"></i></button>
                    <button class="btn" id="btn-download-viz"><i class="fas fa-download"></i></button>
                </div>
            </div>
            <div class="viz-preview">
                <div class="viz-preview-header"><h3>Visualization Preview</h3><div class="btn-group"><button class="btn btn-sm btn-icon" id="btn-fullscreen-viz"><i class="fas fa-expand"></i></button></div></div>
                <div class="viz-preview-body" id="viz-preview-body"><div class="viz-placeholder" id="viz-placeholder"><i class="fas fa-chart-area"></i><h3>Build Your Visualization</h3><p>Drag variables to the mapping zones and click Build</p></div><canvas id="viz-canvas" style="display:none"></canvas></div>
            </div>
        </div>`,

    chartGallery: () => `
        <div class="gallery-filters"><button class="gallery-filter active" data-filter="all">All</button><button class="gallery-filter" data-filter="relationship">Relationship</button><button class="gallery-filter" data-filter="trend">Trend</button><button class="gallery-filter" data-filter="distribution">Distribution</button><button class="gallery-filter" data-filter="comparison">Comparison</button><button class="gallery-filter" data-filter="composition">Composition</button></div>
        <div class="gallery-grid" id="gallery-grid"></div>`,

    descriptive: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Select Variables</h3></div><div class="card-body"><div id="desc-var-list"></div><button class="btn btn-primary btn-block" id="btn-run-descriptive" style="margin-top:14px"><i class="fas fa-play"></i> Calculate</button></div></div>
        <div style="overflow-y:auto"><div id="desc-results"><div class="empty-state"><i class="fas fa-calculator"></i><h3>Descriptive Statistics</h3><p>Select variables and click Calculate</p></div></div></div></div>`,

    hypothesis: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Test Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Test Type</label><select class="form-control" id="hyp-test-type"><option value="1-sample-t">1-Sample t-Test</option><option value="2-sample-t">2-Sample t-Test</option><option value="paired-t">Paired t-Test</option><option value="normality">Normality Test</option><option value="chi-square">Chi-Square Test</option><option value="f-test">F-Test</option></select></div>
            <div class="form-group"><label>Variable 1</label><select class="form-control var-dropdown" id="hyp-var1"></select></div>
            <div class="form-group" id="hyp-var2-group" style="display:none"><label>Variable 2</label><select class="form-control var-dropdown" id="hyp-var2"></select></div>
            <div class="form-group" id="hyp-mu-group"><label>Test Value (μ₀)</label><input class="form-control" id="hyp-mu" value="0" type="number"></div>
            <div class="form-group"><label>Significance (α)</label><select class="form-control" id="hyp-alpha"><option value="0.05">0.05</option><option value="0.01">0.01</option><option value="0.10">0.10</option></select></div>
            <div class="form-group"><label>Alternative</label><select class="form-control" id="hyp-alternative"><option value="two-sided">Two-sided</option><option value="less">Less</option><option value="greater">Greater</option></select></div>
            <button class="btn btn-primary btn-block" id="btn-run-hypothesis"><i class="fas fa-play"></i> Run Test</button>
        </div></div><div style="overflow-y:auto"><div id="hyp-results"><div class="empty-state"><i class="fas fa-flask"></i><h3>Hypothesis Testing</h3><p>Configure and run a test</p></div></div></div></div>`,

    regression: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Regression Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Response (Y)</label><select class="form-control var-dropdown" id="reg-response"></select></div>
            <div class="form-group"><label>Predictor (X)</label><select class="form-control var-dropdown" id="reg-predictor"></select></div>
            <div class="form-group"><label>Type</label><select class="form-control" id="reg-type"><option value="linear">Simple Linear</option><option value="polynomial">Polynomial</option><option value="multiple">Multiple</option></select></div>
            <div class="form-group" id="reg-poly-group" style="display:none"><label>Degree</label><input class="form-control" id="reg-degree" type="number" value="2" min="2" max="6"></div>
            <button class="btn btn-primary btn-block" id="btn-run-regression"><i class="fas fa-play"></i> Run Regression</button>
        </div></div><div style="overflow-y:auto"><div id="reg-results"><div class="empty-state"><i class="fas fa-project-diagram"></i><h3>Regression Analysis</h3><p>Set up and run a regression</p></div></div></div></div>`,

    anova: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>ANOVA Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Response</label><select class="form-control var-dropdown" id="anova-response"></select></div>
            <div class="form-group"><label>Factor</label><select class="form-control var-dropdown" id="anova-factor1"></select></div>
            <button class="btn btn-primary btn-block" id="btn-run-anova"><i class="fas fa-play"></i> Run ANOVA</button>
        </div></div><div style="overflow-y:auto"><div id="anova-results"><div class="empty-state"><i class="fas fa-layer-group"></i><h3>ANOVA</h3><p>Set up the analysis</p></div></div></div></div>`,

    correlation: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Select Variables</h3></div><div class="card-body">
            <div class="form-group"><label>Method</label><select class="form-control" id="corr-method"><option value="pearson">Pearson</option><option value="spearman">Spearman</option></select></div>
            <div id="corr-var-list"></div>
            <button class="btn btn-primary btn-block" id="btn-run-correlation" style="margin-top:14px"><i class="fas fa-play"></i> Compute</button>
        </div></div><div style="overflow-y:auto"><div id="corr-results"><div class="empty-state"><i class="fas fa-bezier-curve"></i><h3>Correlation Analysis</h3><p>Select variables and compute</p></div></div></div></div>`,

    controlCharts: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Control Chart Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Chart Type</label><select class="form-control" id="cc-type"><option value="xbar-r">X̄-R Chart</option><option value="i-mr">I-MR Chart</option></select></div>
            <div class="form-group"><label>Variable</label><select class="form-control var-dropdown" id="cc-variable"></select></div>
            <div class="form-group" id="cc-subgroup-group"><label>Subgroup Size</label><input class="form-control" id="cc-subgroup-size" type="number" value="5" min="2"></div>
            <button class="btn btn-primary btn-block" id="btn-run-cc"><i class="fas fa-play"></i> Generate Chart</button>
        </div></div><div style="overflow-y:auto"><div id="cc-results"><div class="empty-state"><i class="fas fa-wave-square"></i><h3>Control Charts</h3><p>Configure and generate</p></div></div></div></div>`,

    capability: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Capability Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Variable</label><select class="form-control var-dropdown" id="cap-variable"></select></div>
            <div class="form-inline"><div class="form-group"><label>LSL</label><input class="form-control" id="cap-lsl" type="number" step="any"></div><div class="form-group"><label>USL</label><input class="form-control" id="cap-usl" type="number" step="any"></div></div>
            <div class="form-group"><label>Target</label><input class="form-control" id="cap-target" type="number" step="any" placeholder="Auto"></div>
            <button class="btn btn-primary btn-block" id="btn-run-capability"><i class="fas fa-play"></i> Calculate</button>
        </div></div><div style="overflow-y:auto"><div id="cap-results"><div class="empty-state"><i class="fas fa-bullseye"></i><h3>Capability Analysis</h3><p>Enter spec limits</p></div></div></div></div>`,

    pareto: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Pareto Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Category Variable</label><select class="form-control var-dropdown" id="pareto-category"></select></div>
            <button class="btn btn-primary btn-block" id="btn-run-pareto"><i class="fas fa-play"></i> Analyze</button>
        </div></div><div style="overflow-y:auto"><div id="pareto-results"><div class="empty-state"><i class="fas fa-sort-amount-down"></i><h3>Pareto Analysis</h3><p>Select a category variable</p></div></div></div></div>`,

    timeseries: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Time Series Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Value Variable</label><select class="form-control var-dropdown" id="ts-value"></select></div>
            <div class="form-group"><label>Analysis</label><select class="form-control" id="ts-analysis"><option value="plot">Time Series Plot</option><option value="moving-average">Moving Average</option><option value="autocorrelation">Autocorrelation</option></select></div>
            <div class="form-group"><label>Window</label><input class="form-control" id="ts-window" type="number" value="7" min="2"></div>
            <button class="btn btn-primary btn-block" id="btn-run-timeseries"><i class="fas fa-play"></i> Analyze</button>
        </div></div><div style="overflow-y:auto"><div id="ts-results"><div class="empty-state"><i class="fas fa-clock"></i><h3>Time Series</h3><p>Configure and analyze</p></div></div></div></div>`,

    forecasting: () => `
        <div class="grid-sidebar"><div class="card" style="overflow-y:auto"><div class="card-header"><h3>Forecast Setup</h3></div><div class="card-body">
            <div class="form-group"><label>Value Variable</label><select class="form-control var-dropdown" id="fc-value"></select></div>
            <div class="form-group"><label>Method</label><select class="form-control" id="fc-method"><option value="ses">Exponential Smoothing</option><option value="lstm">LSTM Neural Network</option></select></div>
            <div class="form-group"><label>Alpha <span id="fc-alpha-val">0.30</span></label><input type="range" class="form-range" id="fc-alpha" min="0.05" max="0.95" step="0.05" value="0.3"></div>
            <div class="form-group"><label>Forecast Periods</label><input class="form-control" id="fc-periods" type="number" value="20" min="1"></div>
            <button class="btn btn-primary btn-block" id="btn-run-forecast"><i class="fas fa-play"></i> Forecast</button>
        </div></div><div style="overflow-y:auto"><div id="fc-results"><div class="empty-state"><i class="fas fa-chart-line"></i><h3>Forecasting</h3><p>Configure and forecast</p></div></div></div></div>`,

    mlSetup: () => `
        <div class="grid-sidebar-wide">
            <div class="card" style="overflow-y:auto"><div class="card-header"><h3><i class="fas fa-crosshairs"></i> Target & Features</h3></div><div class="card-body">
                <div class="form-group"><label>Target Variable (Y)</label><select class="form-control var-dropdown" id="ml-target"></select></div>
                <div class="form-group"><label>Problem Type</label><select class="form-control" id="ml-problem-type"><option value="auto">Auto-detect</option><option value="regression">Regression</option><option value="classification">Classification</option></select></div>
                <div class="form-group"><label>Test Split (%)</label><input type="range" class="form-range" id="ml-test-split" min="10" max="40" step="5" value="20"><span id="ml-split-val">20%</span></div>
                <div class="form-group" style="margin-top:14px"><label>Feature Variables</label><div id="ml-feature-list"></div></div>
                <button class="btn btn-primary btn-block" id="btn-ml-prepare" style="margin-top:14px"><i class="fas fa-check-circle"></i> Prepare Dataset</button>
            </div></div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-eye"></i> Data Preview</h3></div><div class="card-body" style="overflow:auto"><div id="ml-preview"><div class="empty-state"><i class="fas fa-cogs"></i><h3>ML Setup</h3><p>Select target and features, then prepare the dataset</p></div></div></div></div>
        </div>`,

    mlModels: () => `
        <div id="ml-model-content">
            <div class="kpi-row" style="margin-bottom:18px" id="ml-train-kpi"></div>
            <h3 style="font-size:15px;font-weight:600;margin-bottom:12px" id="ml-model-title"><i class="fas fa-brain"></i> Select Models to Train</h3>
            <div id="ml-regression-models" style="display:none"><h4 style="font-size:12px;color:var(--text-secondary);margin-bottom:10px">REGRESSION MODELS</h4>
                <div class="grid-4" id="reg-model-grid"></div></div>
            <div id="ml-classification-models" style="display:none"><h4 style="font-size:12px;color:var(--text-secondary);margin-bottom:10px;margin-top:18px">CLASSIFICATION MODELS</h4>
                <div class="grid-4" id="cls-model-grid"></div></div>
            <div style="margin-top:18px;display:flex;gap:8px"><button class="btn btn-primary btn-lg" id="btn-train-models"><i class="fas fa-play"></i> Train Selected Models</button><button class="btn btn-lg" id="btn-train-all"><i class="fas fa-forward"></i> Train All</button></div>
            <div id="training-progress" style="margin-top:18px;display:none"><div class="loading-bar" style="width:100%;margin-bottom:10px"><div class="loading-bar-fill" id="train-progress-bar" style="animation:none"></div></div><p id="train-status" style="font-size:12px;color:var(--text-secondary)">Training...</p></div>
        </div>`,

    mlResults: () => `<div id="ml-results-content"><div class="empty-state"><i class="fas fa-trophy"></i><h3>No Results Yet</h3><p>Train models first to see comparison and results</p></div></div>`,

    mlPredict: () => `
        <div class="grid-sidebar-wide">
            <div class="card" style="overflow-y:auto"><div class="card-header"><h3><i class="fas fa-magic"></i> Make Predictions</h3></div><div class="card-body">
                <div class="form-group"><label>Model</label><select class="form-control" id="predict-model"></select></div>
                <div id="predict-inputs"></div>
                <button class="btn btn-primary btn-block" id="btn-predict" style="margin-top:14px"><i class="fas fa-magic"></i> Predict</button>
                <div id="predict-result" style="margin-top:18px"></div>
            </div></div>
            <div class="card"><div class="card-header"><h3><i class="fas fa-table"></i> Batch Predictions</h3></div><div class="card-body" style="overflow:auto"><div id="batch-predictions">Select a model and make predictions</div></div></div>
        </div>`,

    project: () => `
        <div class="grid-2">
            <div class="project-card"><h4><i class="fas fa-save"></i> Save Project</h4><p>Save your current work including data, settings, and trained models</p>
                <div class="form-group"><label>Project Name</label><input class="form-control" id="project-name" placeholder="My Analysis Project"></div>
                <button class="btn btn-primary" id="btn-save-proj"><i class="fas fa-save"></i> Save to Browser</button>
                <button class="btn" id="btn-export-proj" style="margin-left:6px"><i class="fas fa-file-export"></i> Export as JSON</button>
            </div>
            <div class="project-card"><h4><i class="fas fa-folder-open"></i> Load Project</h4><p>Load a previously saved project</p>
                <div class="project-list" id="project-list"></div>
                <div style="margin-top:12px"><input type="file" id="import-proj-file" accept=".json" hidden><button class="btn" id="btn-import-proj"><i class="fas fa-file-import"></i> Import from File</button></div>
            </div>
        </div>`,
};
