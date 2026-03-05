/* ============================================
   Visualization Engine v2
   ============================================ */
class VisualizationEngine {
    constructor() {
        this.charts = {};
        this.palettes = {
            default: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'],
            viridis: ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b'],
            plasma: ['#0d0887', '#5302a3', '#8b0aa5', '#b83289', '#db5c68', '#f48849', '#febd2a', '#f0f921'],
            industrial: ['#1a5276', '#2e86c1', '#48c9b0', '#f4d03f', '#e74c3c', '#8e44ad', '#2c3e50', '#95a5a6'],
            ocean: ['#001f3f', '#0074D9', '#7FDBFF', '#39CCCC', '#3D9970', '#2ECC40', '#01FF70', '#FFDC00'],
            sunset: ['#ff6b6b', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd', '#01a3a4', '#f368e0', '#ff9f43']
        };
    }
    getColors(p = 'default', n = 8) { const pal = this.palettes[p] || this.palettes.default; return Array.from({ length: n }, (_, i) => pal[i % pal.length]) }
    destroyChart(id) { if (this.charts[id]) { this.charts[id].destroy(); delete this.charts[id] } }
    createChart(canvasId, config) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId); if (!ctx) return null;
        config.options = config.options || {}; config.options.responsive = true; config.options.maintainAspectRatio = false;
        config.options.plugins = config.options.plugins || {};
        config.options.plugins.legend = config.options.plugins.legend || {};
        config.options.plugins.legend.labels = { color: '#8899b4', font: { family: 'Inter', size: 10 } };
        if (config.options.scales) { Object.values(config.options.scales).forEach(a => { a.ticks = a.ticks || {}; a.ticks.color = '#4e6080'; a.ticks.font = { family: 'Inter', size: 9 }; a.grid = a.grid || {}; a.grid.color = 'rgba(30,45,74,.5)'; if (a.title) { a.title.color = '#8899b4'; a.title.font = { family: 'Inter', size: 10, weight: '600' } } }) }
        this.charts[canvasId] = new Chart(ctx, config); return this.charts[canvasId];
    }
    scatter(id, x, y, opts = {}) {
        const c = this.getColors(opts.palette);
        const cfg = { type: 'scatter', data: { datasets: [{ label: opts.yLabel || 'Y', data: x.map((xi, i) => ({ x: xi, y: y[i] })), backgroundColor: c[0] + (opts.opacity ? Math.round(opts.opacity * 255).toString(16) : 'cc'), pointRadius: opts.pointSize || 4, borderColor: c[0], borderWidth: 1 }] }, options: { scales: { x: { title: { display: true, text: opts.xLabel || 'X' } }, y: { title: { display: true, text: opts.yLabel || 'Y' } } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } };
        if (opts.trendline) { const r = StatisticsEngine.linearRegression(x, y); const xs = [...x].sort((a, b) => a - b); cfg.data.datasets.push({ type: 'line', label: `Trend (R²=${r.rSquared.toFixed(3)})`, data: [{ x: xs[0], y: r.intercept + r.slope * xs[0] }, { x: xs[xs.length - 1], y: r.intercept + r.slope * xs[xs.length - 1] }], borderColor: c[1], borderWidth: 2, pointRadius: 0, borderDash: [5, 5] }) }
        return this.createChart(id, cfg);
    }
    line(id, labels, datasets, opts = {}) {
        const c = this.getColors(opts.palette, datasets.length);
        return this.createChart(id, { type: 'line', data: { labels, datasets: datasets.map((ds, i) => ({ label: ds.label || `Series ${i + 1}`, data: ds.data, borderColor: c[i], backgroundColor: c[i] + '20', borderWidth: 2, pointRadius: opts.pointSize || 2, tension: opts.smooth ? .4 : 0, fill: opts.fill || false })) }, options: { scales: { x: { title: { display: true, text: opts.xLabel || '' } }, y: { title: { display: true, text: opts.yLabel || '' } } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    bar(id, labels, values, opts = {}) {
        const c = this.getColors(opts.palette, labels.length);
        return this.createChart(id, { type: 'bar', data: { labels, datasets: [{ label: opts.label || 'Value', data: values, backgroundColor: c.map(x => x + 'bb'), borderColor: c, borderWidth: 1, borderRadius: 4 }] }, options: { indexAxis: opts.horizontal ? 'y' : 'x', scales: { x: { title: { display: true, text: opts.xLabel || '' } }, y: { title: { display: true, text: opts.yLabel || '' }, beginAtZero: opts.beginAtZero !== false } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    histogram(id, values, opts = {}) {
        const bins = opts.bins || Math.ceil(Math.sqrt(values.length)), min = Math.min(...values), max = Math.max(...values), bw = (max - min) / bins || 1;
        const counts = new Array(bins).fill(0), labels = [];
        values.forEach(v => { let i = Math.min(Math.floor((v - min) / bw), bins - 1); counts[i]++ });
        for (let i = 0; i < bins; i++)labels.push((min + i * bw + bw / 2).toFixed(1));
        const c = this.getColors(opts.palette);
        return this.createChart(id, { type: 'bar', data: { labels, datasets: [{ label: opts.label || 'Frequency', data: counts, backgroundColor: c[0] + 'aa', borderColor: c[0], borderWidth: 1, barPercentage: 1, categoryPercentage: 1, borderRadius: 2 }] }, options: { scales: { x: { title: { display: true, text: opts.xLabel || 'Value' } }, y: { title: { display: true, text: 'Frequency' }, beginAtZero: true } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    boxPlot(id, groups, labels, opts = {}) {
        const c = this.getColors(opts.palette, groups.length);
        const stats = groups.map(g => { const s = [...g].sort((a, b) => a - b); return { min: s[0], q1: ss.quantile(s, .25), median: ss.median(s), q3: ss.quantile(s, .75), max: s[s.length - 1] } });
        return this.createChart(id, { type: 'bar', data: { labels, datasets: [{ label: 'IQR', data: stats.map(s => [s.q1, s.q3]), backgroundColor: c.map(x => x + '88'), borderColor: c, borderWidth: 2, borderRadius: 4, barPercentage: .5 }, { type: 'scatter', label: 'Median', data: stats.map((s, i) => ({ x: i, y: s.median })), backgroundColor: '#fff', pointRadius: 5, borderColor: '#000', borderWidth: 2 }, { type: 'scatter', label: 'Min', data: stats.map((s, i) => ({ x: i, y: s.min })), backgroundColor: c[0], pointRadius: 3, pointStyle: 'triangle' }, { type: 'scatter', label: 'Max', data: stats.map((s, i) => ({ x: i, y: s.max })), backgroundColor: c[1], pointRadius: 3, pointStyle: 'triangle' }] }, options: { scales: { x: { title: { display: true, text: opts.xLabel || '' } }, y: { title: { display: true, text: opts.yLabel || '' } } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    pie(id, labels, values, opts = {}) {
        const c = this.getColors(opts.palette, labels.length);
        return this.createChart(id, { type: opts.doughnut ? 'doughnut' : 'pie', data: { labels, datasets: [{ data: values, backgroundColor: c.map(x => x + 'cc'), borderColor: c, borderWidth: 2 }] }, options: { plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    bubble(id, data, opts = {}) {
        const c = this.getColors(opts.palette);
        return this.createChart(id, { type: 'bubble', data: { datasets: [{ label: opts.label || 'Data', data, backgroundColor: c[0] + '88', borderColor: c[0], borderWidth: 1 }] }, options: { scales: { x: { title: { display: true, text: opts.xLabel || 'X' } }, y: { title: { display: true, text: opts.yLabel || 'Y' } } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    controlChart(id, chartData, opts = {}) {
        const n = chartData.data.length, labels = Array.from({ length: n }, (_, i) => i + 1);
        const ooc = chartData.data.map((v, i) => v > chartData.ucl || v < chartData.lcl);
        return this.createChart(id, { type: 'line', data: { labels, datasets: [{ label: opts.label || 'Value', data: chartData.data, borderColor: '#3b82f6', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: chartData.data.map((v, i) => ooc[i] ? '#ef4444' : '#3b82f6') }, { label: 'CL', data: Array(n).fill(chartData.cl), borderColor: '#10b981', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 }, { label: 'UCL', data: Array(n).fill(chartData.ucl), borderColor: '#ef4444', borderWidth: 2, borderDash: [8, 4], pointRadius: 0 }, { label: 'LCL', data: Array(n).fill(chartData.lcl), borderColor: '#ef4444', borderWidth: 2, borderDash: [8, 4], pointRadius: 0 }] }, options: { scales: { x: { title: { display: true, text: 'Subgroup' } }, y: { title: { display: true, text: opts.yLabel || 'Value' } } }, plugins: { title: { display: !!opts.title, text: opts.title, color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    paretoChart(id, data, opts = {}) {
        const c = this.getColors(opts.palette, data.length);
        return this.createChart(id, { type: 'bar', data: { labels: data.map(d => d.category), datasets: [{ label: 'Count', data: data.map(d => d.count), backgroundColor: c.map(x => x + 'bb'), borderColor: c, borderWidth: 1, borderRadius: 4, yAxisID: 'y' }, { type: 'line', label: 'Cum%', data: data.map(d => d.cumulative), borderColor: '#f59e0b', backgroundColor: '#f59e0b20', borderWidth: 2, pointRadius: 4, yAxisID: 'y1' }] }, options: { scales: { x: {}, y: { title: { display: true, text: 'Count' }, beginAtZero: true, position: 'left' }, y1: { title: { display: true, text: '%' }, min: 0, max: 100, position: 'right', grid: { drawOnChartArea: false } } }, plugins: { title: { display: true, text: opts.title || 'Pareto Chart', color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    capabilityChart(id, values, lsl, usl, target, opts = {}) {
        const bins = 30, mn = Math.min(...values, lsl) - 1, mx = Math.max(...values, usl) + 1, bw = (mx - mn) / bins;
        const counts = new Array(bins).fill(0), labels = [];
        values.forEach(v => { let i = Math.min(Math.floor((v - mn) / bw), bins - 1); counts[i]++ });
        for (let i = 0; i < bins; i++)labels.push((mn + i * bw + bw / 2).toFixed(2));
        const colors = labels.map(l => { const v = parseFloat(l); return (v < lsl || v > usl) ? '#ef444488' : '#3b82f688' });
        return this.createChart(id, { type: 'bar', data: { labels, datasets: [{ label: 'Frequency', data: counts, backgroundColor: colors, borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }] }, options: { scales: { x: { title: { display: true, text: 'Value' } }, y: { title: { display: true, text: 'Freq' }, beginAtZero: true } }, plugins: { title: { display: true, text: opts.title || 'Capability', color: '#e8ecf4', font: { size: 13, weight: '600' } } } } });
    }
    // Heatmap as HTML table
    renderHeatmap(containerId, matrix, labels) {
        let html = '<div class="heatmap-container"><table class="heatmap-table"><thead><tr><th></th>';
        labels.forEach(l => html += `<th>${l}</th>`);
        html += '</tr></thead><tbody>';
        matrix.forEach((row, i) => {
            html += `<tr><th>${labels[i]}</th>`;
            row.forEach(v => {
                const bg = U.colorScale(v);
                html += `<td style="background:${bg}">${v.toFixed(2)}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        U.html(containerId, html);
    }
}
