/* ============================================
   Data Profiler - Power BI-style Quality Report
   ============================================ */
class DataProfiler {
    constructor(dm, viz) { this.dm = dm; this.viz = viz; }

    renderProfileReport(containerId) {
        if (!this.dm.hasData()) { U.html(containerId, '<div class="empty-state"><i class="fas fa-microscope"></i><h3>No Data to Profile</h3><p>Load a dataset first</p></div>'); return; }
        const profiles = this.dm.profileAll();
        const overallQuality = profiles.reduce((s, p) => s + p.qualityPct, 0) / profiles.length;

        // Update sidebar quality
        const qb = U.el('quality-badge');
        if (qb) { qb.style.display = ''; U.el('quality-score').textContent = Math.round(overallQuality * 100); const ring = qb.querySelector('.quality-ring'); ring.style.borderColor = overallQuality > 0.9 ? 'var(--success)' : overallQuality > 0.7 ? 'var(--warning)' : 'var(--danger)'; ring.querySelector('span').style.color = ring.style.borderColor; }

        let html = `<div class="kpi-row">
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-1)"><i class="fas fa-table"></i></div><div class="kpi-info"><span class="kpi-value">${this.dm.data.length.toLocaleString()}</span><span class="kpi-label">Rows</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-2)"><i class="fas fa-columns"></i></div><div class="kpi-info"><span class="kpi-value">${this.dm.columns.length}</span><span class="kpi-label">Columns</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-3)"><i class="fas fa-check-circle"></i></div><div class="kpi-info"><span class="kpi-value">${Math.round(overallQuality * 100)}%</span><span class="kpi-label">Data Quality</span></div><div class="kpi-trend ${overallQuality > .9 ? 'up' : 'down'}"><i class="fas fa-arrow-${overallQuality > .9 ? 'up' : 'down'}"></i></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-4)"><i class="fas fa-exclamation-triangle"></i></div><div class="kpi-info"><span class="kpi-value">${profiles.reduce((s, p) => s + p.missing, 0).toLocaleString()}</span><span class="kpi-label">Missing Values</span></div></div>
            <div class="kpi-card"><div class="kpi-icon" style="background:var(--gradient-5)"><i class="fas fa-chart-bar"></i></div><div class="kpi-info"><span class="kpi-value">${this.dm.getNumericColumns().length}N / ${this.dm.getCategoricalColumns().length}C</span><span class="kpi-label">Numeric / Categorical</span></div></div>
        </div>`;

        // Error distribution chart
        html += `<div class="grid-2" style="margin-bottom:20px"><div class="card"><div class="card-header"><h3><i class="fas fa-chart-bar"></i> Column Quality Distribution</h3></div><div class="card-body"><canvas id="profiler-quality-chart" style="height:220px"></canvas></div></div>`;
        html += `<div class="card"><div class="card-header"><h3><i class="fas fa-chart-pie"></i> Data Type Distribution</h3></div><div class="card-body"><canvas id="profiler-types-chart" style="height:220px"></canvas></div></div></div>`;

        // Column cards
        html += '<h3 style="margin-bottom:14px;font-size:15px;font-weight:600"><i class="fas fa-th-list"></i> Column Profiles</h3>';
        html += '<div class="profiler-grid">';
        profiles.forEach((p, i) => {
            const qClass = p.qualityPct > 0.95 ? 'good' : p.qualityPct > 0.8 ? 'warn' : 'bad';
            const typeClass = ['continuous', 'discrete'].includes(p.type) ? 'numeric' : p.type === 'datetime' ? 'datetime' : 'categorical';
            html += `<div class="profiler-col-card">
                <div class="col-name">${p.col} <span class="col-type ${typeClass}">${p.type}</span></div>
                <div class="error-pct" style="color:${p.errorPct > 0.05 ? 'var(--danger)' : p.errorPct > 0 ? 'var(--warning)' : 'var(--success)'}">
                    <i class="fas fa-${p.errorPct > 0.05 ? 'times-circle' : p.errorPct > 0 ? 'exclamation-circle' : 'check-circle'}"></i>
                    ${(p.errorPct * 100).toFixed(1)}% error · ${(p.qualityPct * 100).toFixed(1)}% valid
                </div>
                <div class="quality-bar"><div class="quality-bar-fill ${qClass}" style="width:${p.qualityPct * 100}%"></div></div>
                <div class="mini-chart"><canvas id="prof-mini-${i}"></canvas></div>
                <div class="col-stats">
                    <div class="col-stat"><span class="stat-label">Valid</span><span class="stat-value">${p.valid}</span></div>
                    <div class="col-stat"><span class="stat-label">Missing</span><span class="stat-value">${p.missing}</span></div>
                    <div class="col-stat"><span class="stat-label">Unique</span><span class="stat-value">${p.unique}</span></div>
                    <div class="col-stat"><span class="stat-label">Unique%</span><span class="stat-value">${(p.uniquePct * 100).toFixed(1)}%</span></div>
                    ${p.mean !== undefined ? `
                    <div class="col-stat"><span class="stat-label">Mean</span><span class="stat-value">${U.fmt(p.mean, 2)}</span></div>
                    <div class="col-stat"><span class="stat-label">Std</span><span class="stat-value">${U.fmt(p.std, 2)}</span></div>
                    <div class="col-stat"><span class="stat-label">Min</span><span class="stat-value">${U.fmt(p.min, 2)}</span></div>
                    <div class="col-stat"><span class="stat-label">Max</span><span class="stat-value">${U.fmt(p.max, 2)}</span></div>
                    <div class="col-stat"><span class="stat-label">Outliers</span><span class="stat-value" style="color:${(p.outliers || 0) > 0 ? 'var(--warning)' : 'inherit'}">${p.outliers || 0}</span></div>
                    <div class="col-stat"><span class="stat-label">Zeros</span><span class="stat-value">${p.zeros || 0}</span></div>
                    ` : ''}
                    ${p.topValues ? p.topValues.slice(0, 3).map(([k, v]) => `<div class="col-stat"><span class="stat-label">${k}</span><span class="stat-value">${v}</span></div>`).join('') : ''}
                </div>
            </div>`;
        });
        html += '</div>';

        U.html(containerId, html);

        // Render charts
        setTimeout(() => {
            // Quality distribution
            this.viz.createChart('profiler-quality-chart', {
                type: 'bar', data: {
                    labels: profiles.map(p => p.col), datasets: [
                        { label: 'Valid %', data: profiles.map(p => p.qualityPct * 100), backgroundColor: profiles.map(p => p.qualityPct > .95 ? '#10b98188' : p.qualityPct > .8 ? '#f59e0b88' : '#ef444488'), borderRadius: 4 }
                    ]
                }, options: { scales: { y: { max: 100, title: { display: true, text: '%' } }, x: {} }, plugins: { legend: { display: false } } }
            });

            // Types chart
            const typeCounts = {};
            Object.values(this.dm.columnTypes).forEach(t => typeCounts[t] = (typeCounts[t] || 0) + 1);
            const typeEntries = Object.entries(typeCounts);
            this.viz.createChart('profiler-types-chart', {
                type: 'doughnut', data: { labels: typeEntries.map(e => e[0]), datasets: [{ data: typeEntries.map(e => e[1]), backgroundColor: ['#3b82f6cc', '#8b5cf6cc', '#10b981cc', '#f59e0bcc', '#ef4444cc'] }] }, options: { plugins: { legend: { position: 'bottom' } } }
            });

            // Mini charts
            profiles.forEach((p, i) => {
                const canvas = U.el(`prof-mini-${i}`);
                if (!canvas) return;
                if (['continuous', 'discrete'].includes(p.type)) {
                    const vals = this.dm.getNumericValues(p.col);
                    const bins = 15; const min = Math.min(...vals), max = Math.max(...vals); const bw = (max - min) / bins || 1;
                    const counts = new Array(bins).fill(0);
                    vals.forEach(v => { let idx = Math.min(Math.floor((v - min) / bw), bins - 1); counts[idx]++; });
                    new Chart(canvas, { type: 'bar', data: { labels: counts.map((_, i) => ''), datasets: [{ data: counts, backgroundColor: '#3b82f666', borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } } });
                } else if (p.topValues) {
                    const top5 = p.topValues.slice(0, 5);
                    new Chart(canvas, { type: 'bar', data: { labels: top5.map(e => e[0]), datasets: [{ data: top5.map(e => e[1]), backgroundColor: '#8b5cf666', borderWidth: 0, borderRadius: 2 }] }, options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } } });
                }
            });
        }, 100);
    }
}
