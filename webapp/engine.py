"""
══════════════════════════════════════════════════════════════════════════
    RENDEZVOUS ON ARBITRARY GRAPHS — Generalised Game Engine
══════════════════════════════════════════════════════════════════════════

Generalisation of the Kₙ rendezvous simulator to arbitrary graphs
(directed or undirected).

Two players are dropped at distinct random nodes of the graph.
Each round they independently move to an adjacent node (or stay).
They meet when they collide.  Neither knows the other's position,
only their own starting node.

A strategy maps each possible starting node to a distribution over
walks from that node.  The simulator searches over such strategies
to discover Nash equilibria via best-response dynamics.

══════════════════════════════════════════════════════════════════════════
"""

import itertools
import numpy as np
import networkx as nx
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# §1  GRAPH VALIDATION
# ─────────────────────────────────────────────────────────────────────────

def validate_graph(adj: dict[int, list[int]], directed: bool = False) -> dict:
    """
    Validate the graph and return diagnostics.

    Returns dict with keys:
        valid, warnings, errors, n_nodes, n_edges, components,
        isolated, diameter, graph
    """
    warnings = []
    errors = []
    nodes = sorted(adj.keys())
    n = len(nodes)

    if n < 2:
        errors.append("Need at least 2 nodes for a rendezvous game.")
        return dict(valid=False, warnings=warnings, errors=errors,
                    n_nodes=n, n_edges=0, components=0, isolated=[], diameter=0)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, neighbours in adj.items():
        for v in neighbours:
            G.add_edge(u, v)

    n_edges = G.number_of_edges()
    isolated = list(nx.isolates(G))

    if directed:
        components = nx.number_weakly_connected_components(G)
        is_connected = nx.is_weakly_connected(G)
        strongly_connected = nx.is_strongly_connected(G)
    else:
        components = nx.number_connected_components(G)
        is_connected = nx.is_connected(G)
        strongly_connected = is_connected

    if not is_connected:
        errors.append(
            f"Graph is disconnected ({components} components). "
            "Players starting in different components can never meet. "
            "The rendezvous game requires a connected graph."
        )

    if isolated:
        errors.append(
            f"Isolated node(s): {isolated}. These nodes have no edges, "
            "so a player starting there is trapped forever."
        )

    if directed and is_connected and not strongly_connected:
        warnings.append(
            "Graph is weakly connected but NOT strongly connected. "
            "Some node pairs may have one-way reachability only, "
            "making rendezvous impossible from certain starting positions."
        )

    try:
        if directed:
            diameter = nx.diameter(G) if strongly_connected else float('inf')
        else:
            diameter = nx.diameter(G) if is_connected else float('inf')
    except nx.NetworkXError:
        diameter = float('inf')

    if n_edges == 0:
        errors.append("Graph has no edges. Players cannot move.")

    if is_connected and not directed:
        try:
            if nx.diameter(G) == 1:
                warnings.append(
                    "Graph has diameter 1 (complete or near-complete). "
                    "See Anderson & Weber (1990) for the complete-graph case."
                )
        except nx.NetworkXError:
            pass

    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        warnings.append(
            f"Graph has {len(self_loops)} self-loop(s). "
            "Self-loops allow 'staying' as an explicit edge-move."
        )

    if not directed and is_connected and n > 2:
        bridges = list(nx.bridges(G))
        if bridges:
            warnings.append(
                f"Graph has {len(bridges)} bridge(s) — removing any one "
                "would disconnect the graph. This may create bottleneck effects."
            )

    valid = len(errors) == 0
    return dict(
        valid=valid, warnings=warnings, errors=errors,
        n_nodes=n, n_edges=n_edges, components=components,
        isolated=isolated, diameter=diameter if diameter != float('inf') else 999,
        graph=G,
    )


# ─────────────────────────────────────────────────────────────────────────
# §2  THE GENERALISED GAME ENGINE
# ─────────────────────────────────────────────────────────────────────────

class RendezvousGame:
    """
    Rendezvous search game on an arbitrary graph.
    """

    def __init__(self, G: nx.Graph, horizon: int = 0):
        self.G = G
        self.nodes = sorted(G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        self.directed = G.is_directed()

        # Adjacency: always include self-loop (player can stay)
        self.adj = {}
        for v in self.nodes:
            neighbours = sorted(set(G.neighbors(v)) | {v})
            self.adj[v] = neighbours

        if horizon > 0:
            self.horizon = horizon
        else:
            try:
                if self.directed:
                    d = nx.diameter(G) if nx.is_strongly_connected(G) else self.n
                else:
                    d = nx.diameter(G) if nx.is_connected(G) else self.n
            except nx.NetworkXError:
                d = self.n
            self.horizon = max(3 * d, 2 * self.n)

        # All ordered starting pairs (a, b) with a != b
        self.starting_pairs = [
            (a, b) for a in self.nodes for b in self.nodes if a != b
        ]

    def meeting_time(self, route_a: list[int], route_b: list[int],
                     start_a: int, start_b: int) -> int:
        """
        First time t in {0,...,T} where both players are at the same node.
        At time 0: player A at start_a, player B at start_b.
        At time t >= 1: A at route_a[(t-1) % len], B at route_b[(t-1) % len].
        """
        if start_a == start_b:
            return 0
        T = self.horizon
        for t in range(1, T + 1):
            node_a = route_a[(t - 1) % len(route_a)]
            node_b = route_b[(t - 1) % len(route_b)]
            if node_a == node_b:
                return t
        return T + 1


# ─────────────────────────────────────────────────────────────────────────
# §3  ROUTE GENERATORS
# ─────────────────────────────────────────────────────────────────────────

def random_walk_routes(game: RendezvousGame, start: int,
                       count: int, length: int,
                       rng: np.random.Generator) -> list[list[int]]:
    """Generate random walk routes from a given start node."""
    routes = set()
    for _ in range(count * 5):
        route = []
        pos = start
        for _ in range(length):
            neighbours = game.adj[pos]
            pos = neighbours[rng.integers(len(neighbours))]
            route.append(pos)
        routes.add(tuple(route))
        if len(routes) >= count:
            break
    return [list(r) for r in routes]


def bfs_route(game: RendezvousGame, start: int, length: int) -> list[int]:
    """BFS traversal route from start."""
    visited = [start]
    queue = [start]
    while queue and len(visited) < game.n:
        v = queue.pop(0)
        for u in game.adj[v]:
            if u not in visited and u != v:
                visited.append(u)
                queue.append(u)
    route = visited[1:]  # skip start
    if not route:
        route = [start]
    while len(route) < length:
        route.extend(route)
    return route[:length]


def dfs_route(game: RendezvousGame, start: int, length: int) -> list[int]:
    """DFS traversal route from start."""
    visited = []
    stack = [start]
    while stack and len(visited) < game.n:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            for u in reversed(game.adj[v]):
                if u not in visited and u != v:
                    stack.append(u)
    route = visited[1:]
    if not route:
        route = [start]
    while len(route) < length:
        route.extend(route)
    return route[:length]


def greedy_cover_routes(game: RendezvousGame, start: int,
                        count: int, length: int,
                        rng: np.random.Generator) -> list[list[int]]:
    """Move to least-recently-visited neighbour (with random tie-breaking)."""
    routes = set()
    for seed in range(count * 5):
        local_rng = np.random.default_rng(seed)
        route = []
        pos = start
        last_visit = {v: -1 for v in game.nodes}
        last_visit[start] = 0
        for t in range(1, length + 1):
            neighbours = [u for u in game.adj[pos] if u != pos]
            if not neighbours:
                neighbours = [pos]
            scored = [(last_visit[u], local_rng.random(), u) for u in neighbours]
            scored.sort()
            pos = scored[0][2]
            route.append(pos)
            last_visit[pos] = t
        routes.add(tuple(route))
        if len(routes) >= count:
            break
    return [list(r) for r in routes]


def build_route_pool(game: RendezvousGame,
                     routes_per_node: int = 30) -> dict[int, list[list[int]]]:
    """
    Build a pool of routes for each starting node.

    Returns dict mapping start_node -> list of routes (walks from that node).
    """
    rng = np.random.default_rng(0)
    route_length = min(game.horizon, 3 * game.n)

    pool = {}
    for start in game.nodes:
        seen = set()
        node_routes = []

        def _add(r):
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                node_routes.append(r)

        # Stay
        _add([start] * route_length)
        # BFS
        _add(bfs_route(game, start, route_length))
        # DFS
        _add(dfs_route(game, start, route_length))
        # Random walks
        for r in random_walk_routes(game, start, routes_per_node, route_length, rng):
            _add(r)
        # Greedy cover
        for r in greedy_cover_routes(game, start, max(3, routes_per_node // 5),
                                      route_length, rng):
            _add(r)

        pool[start] = node_routes

    return pool


# ─────────────────────────────────────────────────────────────────────────
# §4  EQUILIBRIUM FINDER
# ─────────────────────────────────────────────────────────────────────────

class EquilibriumFinder:
    """
    Searches for a symmetric Nash equilibrium on an arbitrary graph.

    A strategy is: for each starting node s, a distribution over walks
    from s. We maintain per-node weight vectors and use best-response
    dynamics (fictitious play).

    The payoff for starting pair (sa, sb) is:
        Σ_i Σ_j  w_sa[i] * w_sb[j] * meeting_time(route_i, route_j, sa, sb)

    Overall EMT averages over all starting pairs uniformly.
    """

    def __init__(self, game: RendezvousGame,
                 route_pool: Optional[dict[int, list[list[int]]]] = None,
                 verbose: bool = True):
        self.game = game
        self.verbose = verbose
        self.nodes = game.nodes
        self.n = game.n

        if route_pool is None:
            route_pool = build_route_pool(game)
        self.route_pool = route_pool  # {node: [routes]}

        # For each starting pair (sa, sb), precompute meeting times
        # M_pair[(sa, sb)] is a matrix of shape (K_sa, K_sb)
        self._build_payoff_tensors()

    def _build_payoff_tensors(self):
        """Precompute meeting time matrices for each starting pair."""
        g = self.game
        self.M_pair = {}

        for sa, sb in g.starting_pairs:
            routes_a = self.route_pool[sa]
            routes_b = self.route_pool[sb]
            Ka = len(routes_a)
            Kb = len(routes_b)
            M = np.full((Ka, Kb), g.horizon + 1, dtype=np.float64)
            for i, ra in enumerate(routes_a):
                for j, rb in enumerate(routes_b):
                    M[i, j] = g.meeting_time(ra, rb, sa, sb)
            self.M_pair[(sa, sb)] = M

    def compute_emt(self, weights: dict[int, np.ndarray]) -> float:
        """
        Compute expected meeting time for a symmetric strategy profile.

        weights: {node: weight_vector over routes from that node}
        """
        total = 0.0
        n_pairs = len(self.game.starting_pairs)
        for sa, sb in self.game.starting_pairs:
            wa = weights[sa]
            wb = weights[sb]
            M = self.M_pair[(sa, sb)]
            total += wa @ M @ wb
        return total / n_pairs

    def best_response(self, opp_weights: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """
        For each starting node, find the pure route that minimises expected
        meeting time against the opponent's strategy.
        """
        br = {}
        for s in self.nodes:
            K_s = len(self.route_pool[s])
            # Expected meeting time for each route from s, averaged
            # over opponent's possible starting nodes
            emt_per_route = np.zeros(K_s)

            # Opponent could start at any node != s
            # But we also need to consider: player A starts at s,
            # player B starts at sb (for all sb != s)
            count = 0
            for sb in self.nodes:
                if sb == s:
                    continue
                M = self.M_pair[(s, sb)]
                wb = opp_weights[sb]
                # emt_per_route[i] += M[i, :] @ wb
                emt_per_route += M @ wb
                count += 1

            # Also: player B starts at s, player A starts at sa (for all sa != s)
            # But by symmetry, this is the opponent's BR problem — we only
            # need the case where *we* start at s.
            if count > 0:
                emt_per_route /= count

            best = np.argmin(emt_per_route)
            w = np.zeros(K_s)
            w[best] = 1.0
            br[s] = w

        return br

    def fictitious_play(self, max_iter: int = 500,
                        tol: float = 1e-5) -> dict:
        """Run fictitious play to approximate equilibrium."""
        # Initialise uniform
        counts = {}
        for s in self.nodes:
            K = len(self.route_pool[s])
            counts[s] = np.ones(K) / K

        history = []
        for t in range(1, max_iter + 1):
            weights = {s: c / c.sum() for s, c in counts.items()}

            br = self.best_response(weights)
            for s in self.nodes:
                counts[s] += br[s]

            weights_new = {s: c / c.sum() for s, c in counts.items()}
            emt = self.compute_emt(weights_new)
            history.append(float(emt))

            if self.verbose and t % 100 == 0:
                total_support = sum(
                    int(np.sum(w > 1e-6)) for w in weights_new.values()
                )
                print(f"  iter {t:4d}  |  EMT = {emt:.4f}  |  total support = "
                      f"{total_support}")

            if t > 10 and abs(history[-1] - history[-2]) < tol:
                if self.verbose:
                    print(f"  Converged at iteration {t}")
                return dict(weights=weights_new, emt=float(emt),
                            history=history, converged=True)

        weights = {s: c / c.sum() for s, c in counts.items()}
        emt = self.compute_emt(weights)
        return dict(weights=weights, emt=float(emt),
                    history=history, converged=False)

    def top_routes(self, weights: dict[int, np.ndarray],
                   top_k: int = 5) -> list[dict]:
        """Return top routes across all starting nodes by weight."""
        all_routes = []
        for s in self.nodes:
            ws = weights[s]
            routes = self.route_pool[s]
            for i, w in enumerate(ws):
                if w > 1e-8:
                    all_routes.append(dict(
                        start=int(s),
                        route=[int(x) for x in routes[i]],
                        weight=float(w),
                        route_idx=i,
                    ))
        all_routes.sort(key=lambda x: -x["weight"])
        return all_routes[:top_k]

    def meeting_time_distribution(self, weights: dict[int, np.ndarray],
                                  T_max: int = None) -> dict:
        """
        Compute P(meet at time t | starting pair) for heatmap.

        Returns dict with:
            pairs: list of (sa, sb) labels
            heat: 2D array of shape (n_pairs_shown, T_max+1)
        """
        if T_max is None:
            T_max = min(self.game.horizon, 3 * self.n)

        pairs = self.game.starting_pairs
        # Subsample if too many
        if len(pairs) > 20:
            indices = np.linspace(0, len(pairs) - 1, 20, dtype=int)
            pairs_show = [pairs[i] for i in indices]
        else:
            pairs_show = pairs

        heat = np.zeros((len(pairs_show), T_max + 1))

        for hi, (sa, sb) in enumerate(pairs_show):
            wa = weights[sa]
            wb = weights[sb]
            routes_a = self.route_pool[sa]
            routes_b = self.route_pool[sb]

            for i, ra in enumerate(routes_a):
                if wa[i] < 1e-12:
                    continue
                for j, rb in enumerate(routes_b):
                    if wb[j] < 1e-12:
                        continue
                    t = self.game.meeting_time(ra, rb, sa, sb)
                    if t <= T_max:
                        heat[hi, t] += wa[i] * wb[j]

        # Normalise per pair
        row_sums = heat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        heat /= row_sums

        return dict(
            pairs=[(int(a), int(b)) for a, b in pairs_show],
            heat=heat,
        )


# ─────────────────────────────────────────────────────────────────────────
# §5  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────

def visualize_equilibrium(game: RendezvousGame,
                          eq_result: dict,
                          finder: EquilibriumFinder,
                          save_path: str = None) -> bytes:
    """
    Produce a 2x2 figure analogous to the Kn visualisations.
    Returns PNG bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import io

    G = game.G
    n = game.n
    weights = eq_result["weights"]
    history = eq_result["history"]

    palette = plt.cm.twilight_shifted(np.linspace(0.1, 0.9, 8))

    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    graph_desc = f"{n} nodes, {G.number_of_edges()} edges"
    if game.directed:
        graph_desc += " (directed)"
    fig.suptitle(
        f"Rendezvous Search \u2014 Equilibrium Landscape\n{graph_desc}",
        fontsize=20, color="#e6edf3", fontweight="bold", y=0.97,
        fontfamily="serif",
    )
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.07, right=0.95, top=0.89, bottom=0.06)

    # ════════════════════════════════════════════════════════════════════
    # (A) THE GRAPH with top routes overlaid
    # ════════════════════════════════════════════════════════════════════
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_graph.set_facecolor("#0d1117")
    ax_graph.set_title("The Arena \u2014 graph with top routes",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    if n <= 12:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    if game.directed:
        nx.draw_networkx_edges(G, pos, ax=ax_graph,
                               edge_color="#21262d", width=0.8, alpha=0.5,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.1")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax_graph,
                               edge_color="#21262d", width=0.5, alpha=0.4)

    nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                           node_color="#161b22", node_size=500,
                           edgecolors="#58a6ff", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax_graph,
                            font_color="#e6edf3", font_size=11,
                            font_family="monospace")

    top = finder.top_routes(weights, top_k=5)
    max_w = top[0]["weight"] if top else 1
    for rank, info in enumerate(top):
        route = info["route"]
        w = info["weight"]
        start = info["start"]
        color = palette[rank % len(palette)]
        alpha = 0.35 + 0.55 * (w / max_w)

        path_nodes = [start] + route[:min(len(route), n + 2)]
        pts = [pos[v] for v in path_nodes if v in pos]
        for i in range(len(pts) - 1):
            offset = 0.04 * (rank - len(top) / 2)
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            ax_graph.annotate(
                "", xy=(x1 + offset, y1 + offset),
                xytext=(x0 + offset, y0 + offset),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5 + 3 * w, alpha=alpha,
                                connectionstyle=f"arc3,rad={0.15 + 0.08 * rank}"),
            )

        route_str = "\u2192".join(str(x) for x in route[:5])
        if len(route) > 5:
            route_str += "\u2026"
        ax_graph.plot([], [], color=color, lw=3, alpha=alpha,
                      label=f"{w:.1%} from {start}: {route_str}")

    ax_graph.legend(loc="lower left", fontsize=7, facecolor="#161b22",
                    edgecolor="#30363d", labelcolor="#8b949e",
                    title="Route (weight)", title_fontsize=8)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    # ════════════════════════════════════════════════════════════════════
    # (B) STRATEGY CLOCK
    # ════════════════════════════════════════════════════════════════════
    ax_clock = fig.add_subplot(gs[0, 1], projection="polar")
    ax_clock.set_facecolor("#0d1117")
    ax_clock.set_title("Strategy Clock \u2014 where are you each turn?",
                       color="#8b949e", fontsize=13, pad=18, fontfamily="serif")

    T_show = min(game.horizon, 3 * n)
    theta_ticks = np.linspace(0, 2 * np.pi, T_show, endpoint=False)
    node_idx_map = {v: i for i, v in enumerate(game.nodes)}

    for rank, info in enumerate(top):
        route = info["route"]
        w = info["weight"]
        if w < 1e-8:
            continue
        color = palette[rank % len(palette)]
        radii = [node_idx_map.get(route[t % len(route)], 0)
                 for t in range(T_show)]
        thetas = list(theta_ticks)
        radii.append(radii[0])
        thetas.append(thetas[0] + 2 * np.pi)
        ax_clock.plot(thetas, radii, color=color,
                      lw=1.5 + 3 * w,
                      alpha=0.4 + 0.5 * (w / max_w),
                      marker="o", markersize=3)

    ax_clock.set_rmax(n - 0.5)
    ax_clock.set_rticks(list(range(n)))
    ax_clock.set_yticklabels([f"v{v}" for v in game.nodes[:n]],
                             fontsize=7, color="#8b949e")
    n_time_labels = min(T_show, 20)
    ax_clock.set_xticks(theta_ticks[:n_time_labels])
    ax_clock.set_xticklabels([f"t={t}" for t in range(n_time_labels)],
                             fontsize=7, color="#58a6ff")
    ax_clock.tick_params(colors="#30363d")
    ax_clock.spines["polar"].set_color("#30363d")
    ax_clock.grid(color="#21262d", alpha=0.5)

    # ════════════════════════════════════════════════════════════════════
    # (C) CONVERGENCE HISTORY
    # ════════════════════════════════════════════════════════════════════
    ax_conv = fig.add_subplot(gs[1, 0])
    ax_conv.set_facecolor("#0d1117")
    ax_conv.set_title("Best-Response Dynamics \u2014 convergence",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    iters = np.arange(1, len(history) + 1)
    ax_conv.fill_between(iters, history, alpha=0.15, color="#58a6ff")
    ax_conv.plot(iters, history, color="#58a6ff", lw=2)
    ax_conv.axhline(history[-1], color="#f0883e", ls="--", lw=1, alpha=0.7)
    ax_conv.text(len(history) * 0.6, history[-1] * 1.02,
                 f"EMT* = {history[-1]:.4f}", color="#f0883e", fontsize=11,
                 fontfamily="serif")
    ax_conv.set_xlabel("Iteration", color="#8b949e", fontsize=10)
    ax_conv.set_ylabel("Expected Meeting Time", color="#8b949e", fontsize=10)
    ax_conv.tick_params(colors="#8b949e")
    for spine in ax_conv.spines.values():
        spine.set_color("#30363d")
    ax_conv.grid(color="#21262d", alpha=0.4)

    # ════════════════════════════════════════════════════════════════════
    # (D) MEETING-TIME HEATMAP
    # ════════════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor("#0d1117")
    ax_heat.set_title("Meeting Probability \u2014 start pair \u00d7 time",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    dist = finder.meeting_time_distribution(weights)
    heat = dist["heat"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "midnight_fire",
        ["#0d1117", "#1a1e2e", "#2d1b69", "#6f42c1", "#f0883e", "#f9d423"],
    )
    T_max = heat.shape[1] - 1
    im = ax_heat.imshow(heat, aspect="auto", cmap=cmap, origin="lower",
                        extent=[0, T_max, 0, heat.shape[0]])
    ax_heat.set_xlabel("Time step", color="#8b949e", fontsize=10)
    ax_heat.set_ylabel("Starting pair index", color="#8b949e", fontsize=10)
    ax_heat.tick_params(colors="#8b949e")
    for spine in ax_heat.spines.values():
        spine.set_color("#30363d")
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label("P(meet at t | pair)", color="#8b949e", fontsize=9)
    cbar.ax.tick_params(colors="#8b949e")

    # ── Save ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if save_path:
        with open(save_path, "wb") as f:
            f.write(png_bytes)

    return png_bytes


# ─────────────────────────────────────────────────────────────────────────
# §6  RUNNER
# ─────────────────────────────────────────────────────────────────────────

def run_on_graph(adj: dict[int, list[int]], directed: bool = False,
                 verbose: bool = True) -> dict:
    """Full pipeline for an arbitrary graph."""
    validation = validate_graph(adj, directed)
    if not validation["valid"]:
        return dict(validation=validation, game=None, finder=None,
                    equilibrium=None, png_bytes=None)

    G = validation["graph"]
    game = RendezvousGame(G)

    if verbose:
        print(f"  Graph: {game.n} nodes, {G.number_of_edges()} edges")
        print(f"  Horizon: {game.horizon}")
        print(f"  Starting pairs: {len(game.starting_pairs)}")

    pool = build_route_pool(game)
    if verbose:
        total = sum(len(v) for v in pool.values())
        print(f"  Route pool: {total} routes total "
              f"({', '.join(f'{s}:{len(r)}' for s, r in pool.items())})")

    if verbose:
        print(f"\n  Launching best-response dynamics \u2026\n")
    finder = EquilibriumFinder(game, route_pool=pool, verbose=verbose)
    eq = finder.fictitious_play(max_iter=500)

    if verbose:
        top = finder.top_routes(eq["weights"], top_k=5)
        print(f"\n  EMT* = {eq['emt']:.4f}")
        print(f"  Converged: {eq['converged']}")
        print(f"  Top routes:")
        for info in top:
            r = info["route"][:6]
            s = "\u2192".join(str(x) for x in r)
            if len(info["route"]) > 6:
                s += "\u2026"
            print(f"    [{info['weight']:.2%}] from {info['start']}: {s}")

    png_bytes = visualize_equilibrium(game, eq, finder)

    return dict(
        validation=validation,
        game=game,
        finder=finder,
        equilibrium=eq,
        png_bytes=png_bytes,
    )
