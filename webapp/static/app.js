/* ═══════════════════════════════════════════════════════════════════════
   Rendezvous Search Web App — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────────────
const state = {
    nodes: [],        // [0, 1, 2, ...]
    edges: [],        // [{from: 0, to: 1}, ...]
    directed: false,
    nextNodeId: 0,
};

// ── DOM refs ───────────────────────────────────────────────────────────
const directedToggle = document.getElementById("directed-toggle");
const btnAddNode = document.getElementById("btn-add-node");
const btnRemoveNode = document.getElementById("btn-remove-node");
const nodeCountEl = document.getElementById("node-count");
const edgeFromInput = document.getElementById("edge-from");
const edgeToInput = document.getElementById("edge-to");
const edgeArrow = document.getElementById("edge-arrow");
const btnAddEdge = document.getElementById("btn-add-edge");
const edgeCountEl = document.getElementById("edge-count");
const edgeListEl = document.getElementById("edge-list");
const canvas = document.getElementById("graph-canvas");
const ctx = canvas.getContext("2d");
const validationBox = document.getElementById("validation-box");
const btnSolve = document.getElementById("btn-solve");
const loadingEl = document.getElementById("loading");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");
const presetButtons = document.getElementById("preset-buttons");

// ── Initialise ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadPresets();
    setupEventListeners();
    redraw();
});

function setupEventListeners() {
    directedToggle.addEventListener("change", () => {
        state.directed = directedToggle.checked;
        edgeArrow.textContent = state.directed ? "\u2192" : "\u2014";
        redraw();
    });

    btnAddNode.addEventListener("click", addNode);
    btnRemoveNode.addEventListener("click", removeLastNode);
    btnAddEdge.addEventListener("click", addEdge);
    btnSolve.addEventListener("click", solve);

    // Allow Enter key in edge inputs
    edgeToInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") addEdge();
    });
    edgeFromInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") edgeFromInput.nextElementSibling && edgeToInput.focus();
    });
}

// ── Presets ─────────────────────────────────────────────────────────────
async function loadPresets() {
    try {
        const resp = await fetch("/api/presets");
        const presets = await resp.json();
        presetButtons.innerHTML = "";
        for (const [key, preset] of Object.entries(presets)) {
            const btn = document.createElement("button");
            btn.className = "btn btn-small";
            btn.textContent = preset.name;
            btn.title = preset.description;
            btn.addEventListener("click", () => applyPreset(preset));
            presetButtons.appendChild(btn);
        }
    } catch (err) {
        console.error("Failed to load presets:", err);
    }
}

function applyPreset(preset) {
    state.nodes = [];
    state.edges = [];
    state.nextNodeId = 0;
    state.directed = preset.directed;
    directedToggle.checked = preset.directed;
    edgeArrow.textContent = state.directed ? "\u2192" : "\u2014";

    for (let i = 0; i < preset.nodes; i++) {
        state.nodes.push(i);
        state.nextNodeId = i + 1;
    }
    for (const [u, v] of preset.edges) {
        state.edges.push({ from: u, to: v });
    }
    updateUI();
    validate();
}

// ── Node operations ────────────────────────────────────────────────────
function addNode() {
    if (state.nodes.length >= 15) {
        showValidation("error", ["Maximum 15 nodes for web computation."]);
        return;
    }
    state.nodes.push(state.nextNodeId);
    state.nextNodeId++;
    updateUI();
    validate();
}

function removeLastNode() {
    if (state.nodes.length === 0) return;
    const removed = state.nodes.pop();
    // Remove edges involving this node
    state.edges = state.edges.filter(
        (e) => e.from !== removed && e.to !== removed
    );
    updateUI();
    validate();
}

// ── Edge operations ────────────────────────────────────────────────────
function addEdge() {
    const from = parseInt(edgeFromInput.value);
    const to = parseInt(edgeToInput.value);

    if (isNaN(from) || isNaN(to)) {
        showValidation("error", ["Please enter valid node IDs for both endpoints."]);
        return;
    }

    if (!state.nodes.includes(from)) {
        showValidation("error", [`Node ${from} does not exist.`]);
        return;
    }
    if (!state.nodes.includes(to)) {
        showValidation("error", [`Node ${to} does not exist.`]);
        return;
    }

    if (from === to) {
        showValidation("warning", ["Self-loops are allowed but unusual."]);
    }

    // Check duplicate
    const exists = state.edges.some(
        (e) => e.from === from && e.to === to
    );
    if (exists) {
        showValidation("warning", ["This edge already exists."]);
        return;
    }

    // For undirected, also check reverse
    if (!state.directed) {
        const revExists = state.edges.some(
            (e) => e.from === to && e.to === from
        );
        if (revExists) {
            showValidation("warning", ["This edge already exists (undirected)."]);
            return;
        }
    }

    state.edges.push({ from, to });
    edgeFromInput.value = "";
    edgeToInput.value = "";
    edgeFromInput.focus();
    updateUI();
    validate();
}

function removeEdge(index) {
    state.edges.splice(index, 1);
    updateUI();
    validate();
}

// ── UI Update ──────────────────────────────────────────────────────────
function updateUI() {
    nodeCountEl.textContent = `${state.nodes.length} nodes`;
    edgeCountEl.textContent = state.edges.length;

    // Edge list
    edgeListEl.innerHTML = "";
    state.edges.forEach((e, idx) => {
        const span = document.createElement("span");
        span.className = "edge-item";
        const arrow = state.directed ? "\u2192" : "\u2014";
        span.innerHTML = `${e.from}${arrow}${e.to} <span class="edge-remove" data-idx="${idx}">\u00d7</span>`;
        edgeListEl.appendChild(span);
    });

    // Edge remove handlers
    edgeListEl.querySelectorAll(".edge-remove").forEach((el) => {
        el.addEventListener("click", () => {
            removeEdge(parseInt(el.dataset.idx));
        });
    });

    redraw();
}

// ── Validation ─────────────────────────────────────────────────────────
async function validate() {
    if (state.nodes.length < 2) {
        hideValidation();
        return;
    }

    try {
        const resp = await fetch("/api/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                nodes: state.nodes,
                edges: state.edges,
                directed: state.directed,
            }),
        });
        const result = await resp.json();

        if (result.errors.length > 0) {
            showValidation("error", result.errors);
        } else if (result.warnings.length > 0) {
            showValidation("warning", result.warnings);
        } else {
            showValidation("success", [
                `Valid graph: ${result.n_nodes} nodes, ${result.n_edges} edges, diameter ${result.diameter}.`,
            ]);
        }
    } catch (err) {
        console.error("Validation error:", err);
    }
}

function showValidation(type, messages) {
    validationBox.className = `validation-box ${type}`;
    validationBox.innerHTML = messages.map((m) => `<p>${m}</p>`).join("");
}

function hideValidation() {
    validationBox.className = "validation-box hidden";
}

// ── Canvas Drawing ─────────────────────────────────────────────────────
function redraw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);

    if (state.nodes.length === 0) {
        ctx.fillStyle = "#6e7681";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Add nodes to see the graph", w / 2, h / 2);
        return;
    }

    // Layout: circular
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.35;
    const nodeRadius = Math.max(12, Math.min(20, 180 / state.nodes.length));

    const positions = {};
    state.nodes.forEach((id, i) => {
        const angle = (2 * Math.PI * i) / state.nodes.length - Math.PI / 2;
        positions[id] = {
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
        };
    });

    // Draw edges
    state.edges.forEach((e) => {
        const p1 = positions[e.from];
        const p2 = positions[e.to];
        if (!p1 || !p2) return;

        ctx.strokeStyle = "#30363d";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        // Arrow for directed
        if (state.directed) {
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const len = Math.sqrt(dx * dx + dy * dy);
            if (len === 0) return;
            const ux = dx / len;
            const uy = dy / len;
            // Arrow tip, offset by node radius
            const tipX = p2.x - ux * nodeRadius;
            const tipY = p2.y - uy * nodeRadius;
            const arrowLen = 10;
            const arrowAngle = 0.4;

            ctx.fillStyle = "#30363d";
            ctx.beginPath();
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(
                tipX - arrowLen * Math.cos(Math.atan2(uy, ux) - arrowAngle),
                tipY - arrowLen * Math.sin(Math.atan2(uy, ux) - arrowAngle)
            );
            ctx.lineTo(
                tipX - arrowLen * Math.cos(Math.atan2(uy, ux) + arrowAngle),
                tipY - arrowLen * Math.sin(Math.atan2(uy, ux) + arrowAngle)
            );
            ctx.closePath();
            ctx.fill();
        }
    });

    // Draw nodes
    state.nodes.forEach((id) => {
        const p = positions[id];
        // Circle
        ctx.fillStyle = "#161b22";
        ctx.strokeStyle = "#58a6ff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, nodeRadius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = "#e6edf3";
        ctx.font = `${Math.max(10, nodeRadius * 0.8)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(id), p.x, p.y);
    });
}

// ── Solve ──────────────────────────────────────────────────────────────
async function solve() {
    if (state.nodes.length < 2) {
        showValidation("error", ["Need at least 2 nodes."]);
        return;
    }

    // Show loading
    loadingEl.classList.remove("hidden");
    resultsEmpty.classList.add("hidden");
    resultsContent.classList.add("hidden");
    btnSolve.disabled = true;

    try {
        const resp = await fetch("/api/solve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                nodes: state.nodes,
                edges: state.edges,
                directed: state.directed,
            }),
        });
        const data = await resp.json();

        loadingEl.classList.add("hidden");
        btnSolve.disabled = false;

        if (!data.success) {
            resultsEmpty.classList.remove("hidden");
            const msgs = data.validation.errors.concat(data.validation.warnings);
            showValidation("error", msgs);
            return;
        }

        // Show warnings if any
        if (data.validation.warnings.length > 0) {
            showValidation("warning", data.validation.warnings);
        }

        // Populate stats
        document.getElementById("stat-emt").textContent = data.results.emt;
        document.getElementById("stat-support").textContent = data.results.support_size;
        document.getElementById("stat-iterations").textContent = data.results.iterations;
        document.getElementById("stat-converged").textContent =
            data.results.converged ? "Yes" : "No (max iter)";

        // Populate routes table
        const tbody = document.getElementById("routes-tbody");
        tbody.innerHTML = "";
        data.results.top_routes.forEach((r) => {
            const tr = document.createElement("tr");
            const routeStr = r.route.slice(0, 8).join(" \u2192 ");
            const suffix = r.route.length > 8 ? " \u2026" : "";
            tr.innerHTML = `
                <td style="color: var(--accent-orange)">${(r.weight * 100).toFixed(1)}%</td>
                <td>${r.start}</td>
                <td>${routeStr}${suffix}</td>
            `;
            tbody.appendChild(tr);
        });

        // Show image
        const img = document.getElementById("result-image");
        img.src = "data:image/png;base64," + data.image;

        resultsContent.classList.remove("hidden");
    } catch (err) {
        loadingEl.classList.add("hidden");
        btnSolve.disabled = false;
        resultsEmpty.classList.remove("hidden");
        showValidation("error", [`Request failed: ${err.message}`]);
    }
}

// ── Window resize ──────────────────────────────────────────────────────
window.addEventListener("resize", redraw);
