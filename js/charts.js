/* ============================================
   Chart Gallery Templates
   ============================================ */

const CHART_GALLERY = [
    { name: 'Scatter Plot', type: 'scatter', category: 'relationship', icon: 'fa-braille', desc: 'Explore relationships between two continuous variables' },
    { name: 'Line Chart', type: 'line', category: 'trend', icon: 'fa-chart-line', desc: 'Visualize trends over time or ordered categories' },
    { name: 'Bar Chart', type: 'bar', category: 'comparison', icon: 'fa-chart-bar', desc: 'Compare values across categories' },
    { name: 'Histogram', type: 'histogram', category: 'distribution', icon: 'fa-signal', desc: 'Show the distribution of a single variable' },
    { name: 'Box Plot', type: 'box', category: 'distribution', icon: 'fa-square', desc: 'Display the five-number summary' },
    { name: 'Pie Chart', type: 'pie', category: 'composition', icon: 'fa-chart-pie', desc: 'Show proportions of a whole' },
    { name: 'Area Chart', type: 'area', category: 'trend', icon: 'fa-mountain', desc: 'Filled line chart for cumulative trends' },
    { name: 'Stacked Bar', type: 'stacked-bar', category: 'composition', icon: 'fa-layer-group', desc: 'Compare compositions across categories' },
    { name: 'Doughnut', type: 'doughnut', category: 'composition', icon: 'fa-circle-notch', desc: 'Ring-style proportion chart' },
    { name: 'Bubble Chart', type: 'bubble', category: 'relationship', icon: 'fa-circle', desc: 'Three-dimensional scatter plot' },
    { name: 'Control Chart', type: 'control', category: 'trend', icon: 'fa-wave-square', desc: 'Monitor process stability' },
    { name: 'Pareto Chart', type: 'pareto', category: 'comparison', icon: 'fa-sort-amount-down', desc: 'Identify the vital few causes' },
];

function initChartGallery() {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;
    grid.innerHTML = '';

    CHART_GALLERY.forEach((item, idx) => {
        const el = document.createElement('div');
        el.className = 'gallery-item';
        el.dataset.category = item.category;
        el.innerHTML = `
            <div class="gallery-item-preview">
                <canvas id="gallery-chart-${idx}"></canvas>
            </div>
            <div class="gallery-item-info">
                <h4><i class="fas ${item.icon}"></i> ${item.name}</h4>
                <p>${item.desc}</p>
            </div>
        `;
        el.addEventListener('click', () => {
            if (window.dm && window.dm.hasData()) {
                navigateTo('viz-builder');
                // Set chart type
                document.querySelectorAll('.chart-type-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.type === item.type);
                });
            } else {
                showToast('Load data first to use this chart type', 'warning');
            }
        });
        grid.appendChild(el);
    });

    // Render sample mini charts
    setTimeout(() => renderGallerySamples(), 100);

    // Gallery filters
    document.querySelectorAll('.gallery-filter').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.gallery-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const filter = btn.dataset.filter;
            document.querySelectorAll('.gallery-item').forEach(item => {
                item.style.display = (filter === 'all' || item.dataset.category === filter) ? '' : 'none';
            });
        });
    });
}

function renderGallerySamples() {
    const sampleData = Array.from({ length: 20 }, () => Math.random() * 100);
    const labels = sampleData.map((_, i) => i + 1);

    CHART_GALLERY.forEach((item, idx) => {
        const canvas = document.getElementById(`gallery-chart-${idx}`);
        if (!canvas) return;

        const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
        let config;

        switch (item.type) {
            case 'scatter':
                config = { type: 'scatter', data: { datasets: [{ data: sampleData.map((v, i) => ({ x: i, y: v })), backgroundColor: colors[0] + 'aa', pointRadius: 3 }] } };
                break;
            case 'line':
                config = { type: 'line', data: { labels, datasets: [{ data: sampleData, borderColor: colors[0], borderWidth: 2, pointRadius: 0, tension: 0.3 }] } };
                break;
            case 'bar':
                config = { type: 'bar', data: { labels: labels.slice(0, 8), datasets: [{ data: sampleData.slice(0, 8), backgroundColor: colors.map(c => c + '88'), borderRadius: 3 }] } };
                break;
            case 'histogram':
                config = { type: 'bar', data: { labels: labels.slice(0, 10), datasets: [{ data: sampleData.slice(0, 10), backgroundColor: colors[0] + '88', barPercentage: 1, categoryPercentage: 1 }] } };
                break;
            case 'box':
                config = { type: 'bar', data: { labels: ['A', 'B'], datasets: [{ data: [[20, 60], [30, 70]], backgroundColor: [colors[0] + '88', colors[1] + '88'], borderColor: colors, borderWidth: 2, barPercentage: 0.5 }] } };
                break;
            case 'pie':
                config = { type: 'pie', data: { labels: ['A', 'B', 'C', 'D'], datasets: [{ data: [30, 25, 20, 25], backgroundColor: colors.map(c => c + 'cc') }] } };
                break;
            case 'area':
                config = { type: 'line', data: { labels, datasets: [{ data: sampleData, borderColor: colors[2], backgroundColor: colors[2] + '30', borderWidth: 2, pointRadius: 0, fill: true }] } };
                break;
            case 'stacked-bar':
                config = {
                    type: 'bar', data: {
                        labels: ['Q1', 'Q2', 'Q3', 'Q4'], datasets: [
                            { data: [30, 40, 35, 45], backgroundColor: colors[0] + '88', label: 'A' },
                            { data: [20, 15, 25, 20], backgroundColor: colors[1] + '88', label: 'B' }
                        ]
                    }, options: { scales: { x: { stacked: true }, y: { stacked: true } } }
                };
                break;
            case 'doughnut':
                config = { type: 'doughnut', data: { labels: ['X', 'Y', 'Z'], datasets: [{ data: [40, 35, 25], backgroundColor: colors.map(c => c + 'cc') }] } };
                break;
            case 'bubble':
                config = { type: 'bubble', data: { datasets: [{ data: sampleData.slice(0, 10).map((v, i) => ({ x: i * 10, y: v, r: 3 + Math.random() * 8 })), backgroundColor: colors[4] + '88' }] } };
                break;
            case 'control':
                config = {
                    type: 'line', data: {
                        labels, datasets: [
                            { data: sampleData, borderColor: colors[0], borderWidth: 2, pointRadius: 2 },
                            { data: Array(20).fill(80), borderColor: colors[1], borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
                            { data: Array(20).fill(20), borderColor: colors[1], borderWidth: 1, borderDash: [4, 4], pointRadius: 0 }
                        ]
                    }
                };
                break;
            case 'pareto':
                config = {
                    type: 'bar', data: {
                        labels: ['A', 'B', 'C', 'D', 'E'], datasets: [
                            { data: [40, 25, 15, 12, 8], backgroundColor: colors.map(c => c + '88'), borderRadius: 3, yAxisID: 'y' },
                            { type: 'line', data: [40, 65, 80, 92, 100], borderColor: colors[3], borderWidth: 2, pointRadius: 3, yAxisID: 'y1' }
                        ]
                    }, options: { scales: { y1: { position: 'right', max: 100, grid: { drawOnChartArea: false } } } }
                };
                break;
            default:
                config = { type: 'line', data: { labels, datasets: [{ data: sampleData, borderColor: colors[0] }] } };
        }

        config.options = config.options || {};
        config.options.responsive = true;
        config.options.maintainAspectRatio = false;
        config.options.plugins = { legend: { display: false } };
        config.options.scales = config.options.scales || {};
        if (config.type !== 'pie' && config.type !== 'doughnut') {
            config.options.scales.x = { ...config.options.scales.x, display: false };
            config.options.scales.y = { ...config.options.scales.y, display: false };
            if (config.options.scales.y1) config.options.scales.y1 = { ...config.options.scales.y1, display: false };
        }

        new Chart(canvas, config);
    });
}
