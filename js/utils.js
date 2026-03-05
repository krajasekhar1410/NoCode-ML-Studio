/* ============================================
   Utility Functions
   ============================================ */
const U = {
    el: (id) => document.getElementById(id),
    qs: (sel) => document.querySelector(sel),
    qsa: (sel) => document.querySelectorAll(sel),
    html: (el, h) => { if (typeof el === 'string') el = U.el(el); if (el) el.innerHTML = h; },
    show: (el) => { if (typeof el === 'string') el = U.el(el); if (el) el.style.display = ''; },
    hide: (el) => { if (typeof el === 'string') el = U.el(el); if (el) el.style.display = 'none'; },
    on: (el, evt, fn) => { if (typeof el === 'string') el = U.el(el); if (el) el.addEventListener(evt, fn); },
    round: (v, d = 4) => { const f = Math.pow(10, d); return Math.round(v * f) / f; },
    fmt: (v, d = 4) => typeof v === 'number' ? (isNaN(v) ? 'N/A' : v.toFixed(d)) : (v ?? 'N/A'),
    pct: (v) => (v * 100).toFixed(1) + '%',
    download: (content, filename, type = 'text/csv') => {
        const blob = new Blob([content], { type });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    },
    downloadJSON: (obj, filename) => U.download(JSON.stringify(obj, null, 2), filename, 'application/json'),
    toast: (msg, type = 'info') => {
        const c = U.el('toast-container');
        const icons = { success: 'fa-check-circle', error: 'fa-times-circle', info: 'fa-info-circle', warning: 'fa-exclamation-triangle' };
        const t = document.createElement('div');
        t.className = `toast ${type}`;
        t.innerHTML = `<i class="fas ${icons[type]}"></i><span>${msg}</span>`;
        c.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 4000);
    },
    modal: (title, bodyHtml) => {
        U.html('modal-title', title);
        U.html('modal-body', bodyHtml);
        U.el('modal-overlay').classList.add('active');
    },
    closeModal: () => U.el('modal-overlay').classList.remove('active'),
    colorScale: (v, min = -1, max = 1) => {
        const norm = (v - min) / (max - min);
        if (v >= 0) return `rgba(16,185,129,${Math.abs(v) * 0.6})`;
        return `rgba(239,68,68,${Math.abs(v) * 0.6})`;
    },
    uid: () => '_' + Math.random().toString(36).substr(2, 9)
};
window.U = U;
window.showToast = U.toast;
