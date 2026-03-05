/* ============================================
   Drag & Drop Engine for Visual Builder
   ============================================ */

class DragDropEngine {
    constructor() {
        this.mappings = { x: null, y: null, color: null, size: null, facet: null, label: null };
        this.draggedVar = null;
        this.init();
    }

    init() {
        // Set up drop zones
        document.querySelectorAll('.mapping-zone').forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                const varName = e.dataTransfer.getData('text/plain');
                const mapping = zone.dataset.mapping;
                if (varName && mapping) {
                    this.setMapping(mapping, varName);
                }
            });
        });
    }

    setupVariables(columns, columnTypes) {
        const list = document.getElementById('variable-list');
        if (!list) return;

        list.innerHTML = '';
        if (columns.length === 0) {
            list.innerHTML = '<div class="no-vars-msg"><p>Load data to see variables</p></div>';
            return;
        }

        columns.forEach(col => {
            const chip = document.createElement('div');
            chip.className = 'var-chip';
            chip.draggable = true;
            chip.dataset.variable = col;

            const type = columnTypes[col] || 'unknown';
            const typeClass = ['continuous', 'discrete'].includes(type) ? 'numeric' : type === 'datetime' ? 'datetime' : 'categorical';
            const typeLabel = type === 'continuous' ? 'N' : type === 'discrete' ? 'D' : type === 'datetime' ? 'T' : 'C';

            chip.innerHTML = `
                <span class="var-type ${typeClass}">${typeLabel}</span>
                <span>${col}</span>
            `;

            chip.addEventListener('dragstart', (e) => {
                this.draggedVar = col;
                e.dataTransfer.setData('text/plain', col);
                e.dataTransfer.effectAllowed = 'copy';
                chip.style.opacity = '0.5';
            });
            chip.addEventListener('dragend', () => {
                chip.style.opacity = '1';
            });

            // Double-click to auto-assign
            chip.addEventListener('dblclick', () => {
                const emptyMapping = Object.entries(this.mappings).find(([k, v]) => !v);
                if (emptyMapping) this.setMapping(emptyMapping[0], col);
            });

            list.appendChild(chip);
        });

        // Variable search
        const searchInput = document.getElementById('var-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                list.querySelectorAll('.var-chip').forEach(chip => {
                    chip.style.display = chip.dataset.variable.toLowerCase().includes(query) ? '' : 'none';
                });
            });
        }
    }

    setMapping(mapping, varName) {
        this.mappings[mapping] = varName;
        const zone = document.getElementById(`drop-${mapping}`);
        if (zone) {
            zone.classList.add('filled');
            const content = zone.querySelector('.zone-content');
            content.className = 'zone-content has-var';
            content.innerHTML = `${varName} <span class="zone-remove" onclick="window.dragDrop.clearMapping('${mapping}')">&times;</span>`;
        }
    }

    clearMapping(mapping) {
        this.mappings[mapping] = null;
        const zone = document.getElementById(`drop-${mapping}`);
        if (zone) {
            zone.classList.remove('filled');
            const content = zone.querySelector('.zone-content');
            content.className = 'zone-content';
            content.textContent = 'Drop variable here';
        }
    }

    clearAll() {
        Object.keys(this.mappings).forEach(m => this.clearMapping(m));
    }

    getMappings() {
        return { ...this.mappings };
    }
}
