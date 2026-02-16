"""
Flask web application for the generalised rendezvous search game.

Users can:
  - Specify a number of nodes
  - Add edges (directed or undirected)
  - Use preset graph templates
  - Run the equilibrium finder and see results
"""

import json
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from engine import (
    validate_graph, RendezvousGame, EquilibriumFinder,
    build_route_pool, visualize_equilibrium,
)
import networkx as nx

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """Validate a graph and return diagnostics."""
    data = request.get_json()
    adj, directed = _parse_graph(data)
    result = validate_graph(adj, directed)
    # Remove non-serialisable graph object
    result.pop("graph", None)
    return jsonify(result)


@app.route("/api/solve", methods=["POST"])
def api_solve():
    """Run the full equilibrium search and return results + image."""
    data = request.get_json()
    adj, directed = _parse_graph(data)

    # Validate first
    validation = validate_graph(adj, directed)
    if not validation["valid"]:
        validation.pop("graph", None)
        return jsonify(dict(
            success=False,
            validation=validation,
        ))

    G = validation["graph"]
    n = G.number_of_nodes()

    # Limit computation for web use
    if n > 15:
        return jsonify(dict(
            success=False,
            validation=dict(
                valid=False,
                errors=["Graph too large for web computation (max 15 nodes). "
                        "For larger graphs, use the CLI engine directly."],
                warnings=[],
            ),
        ))

    try:
        game = RendezvousGame(G)
        rpn = 30 if n <= 8 else 20
        pool = build_route_pool(game, routes_per_node=rpn)
        finder = EquilibriumFinder(game, route_pool=pool, verbose=False)
        eq = finder.fictitious_play(max_iter=300)
        png_bytes = visualize_equilibrium(game, eq, finder)

        # Top routes for display
        top_routes = finder.top_routes(eq["weights"], top_k=8)

        # Encode image as base64
        img_b64 = base64.b64encode(png_bytes).decode("ascii")

        # Compute support size and pool size from dict-based weights
        support_size = sum(
            int((w > 1e-8).sum()) for w in eq["weights"].values()
        )
        pool_size = sum(len(v) for v in pool.values())

        return jsonify(dict(
            success=True,
            validation=dict(
                valid=True,
                warnings=validation["warnings"],
                errors=[],
                n_nodes=validation["n_nodes"],
                n_edges=validation["n_edges"],
                components=validation["components"],
                diameter=validation["diameter"],
            ),
            results=dict(
                emt=round(eq["emt"], 4),
                converged=eq["converged"],
                iterations=len(eq["history"]),
                support_size=support_size,
                pool_size=pool_size,
                top_routes=top_routes,
                horizon=game.horizon,
            ),
            image=img_b64,
        ))

    except Exception as e:
        return jsonify(dict(
            success=False,
            validation=dict(
                valid=False,
                errors=[f"Computation error: {str(e)}"],
                warnings=[],
            ),
        ))


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Return preset graph configurations."""
    presets = {
        "complete_3": {
            "name": "Complete K₃",
            "description": "Triangle — 3 nodes, all connected",
            "nodes": 3,
            "edges": [[0, 1], [0, 2], [1, 2]],
            "directed": False,
        },
        "complete_4": {
            "name": "Complete K₄",
            "description": "4-node complete graph",
            "nodes": 4,
            "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            "directed": False,
        },
        "complete_5": {
            "name": "Complete K₅",
            "description": "5-node complete graph",
            "nodes": 5,
            "edges": [[i, j] for i in range(5) for j in range(i + 1, 5)],
            "directed": False,
        },
        "path_4": {
            "name": "Path P₄",
            "description": "Linear path: 0—1—2—3",
            "nodes": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "directed": False,
        },
        "path_5": {
            "name": "Path P₅",
            "description": "Linear path: 0—1—2—3—4",
            "nodes": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "directed": False,
        },
        "cycle_4": {
            "name": "Cycle C₄",
            "description": "Square cycle",
            "nodes": 4,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "directed": False,
        },
        "cycle_5": {
            "name": "Cycle C₅",
            "description": "Pentagon cycle",
            "nodes": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            "directed": False,
        },
        "cycle_6": {
            "name": "Cycle C₆",
            "description": "Hexagon cycle",
            "nodes": 6,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            "directed": False,
        },
        "star_5": {
            "name": "Star S₅",
            "description": "Hub-and-spoke: node 0 connected to all others",
            "nodes": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
            "directed": False,
        },
        "petersen": {
            "name": "Petersen graph",
            "description": "Classic 10-node 3-regular graph",
            "nodes": 10,
            "edges": [
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # outer
                [5, 7], [7, 9], [9, 6], [6, 8], [8, 5],  # inner (pentagram)
                [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],  # spokes
            ],
            "directed": False,
        },
        "grid_2x3": {
            "name": "Grid 2×3",
            "description": "2 rows, 3 columns grid",
            "nodes": 6,
            "edges": [
                [0, 1], [1, 2],        # top row
                [3, 4], [4, 5],        # bottom row
                [0, 3], [1, 4], [2, 5],  # columns
            ],
            "directed": False,
        },
        "binary_tree": {
            "name": "Binary tree",
            "description": "Complete binary tree of depth 2 (7 nodes)",
            "nodes": 7,
            "edges": [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]],
            "directed": False,
        },
        "directed_cycle_4": {
            "name": "Directed cycle",
            "description": "One-way cycle: 0→1→2→3→0",
            "nodes": 4,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "directed": True,
        },
        "directed_star": {
            "name": "Directed star (out)",
            "description": "Hub 0 sends to all, all send back",
            "nodes": 5,
            "edges": [
                [0, 1], [0, 2], [0, 3], [0, 4],
                [1, 0], [2, 0], [3, 0], [4, 0],
            ],
            "directed": True,
        },
    }
    return jsonify(presets)


def _parse_graph(data: dict) -> tuple[dict, bool]:
    """Parse the graph from request JSON."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    directed = data.get("directed", False)

    # Build adjacency list
    adj = {}
    for n in nodes:
        node_id = int(n) if isinstance(n, (int, float, str)) else int(n["id"])
        adj[node_id] = []

    for edge in edges:
        if isinstance(edge, (list, tuple)):
            u, v = int(edge[0]), int(edge[1])
        else:
            u, v = int(edge["from"]), int(edge["to"])
        if u in adj:
            adj[u].append(v)
        if not directed and v in adj:
            adj[v].append(u)

    return adj, directed


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
