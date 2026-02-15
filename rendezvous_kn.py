"""
══════════════════════════════════════════════════════════════════════════
    RENDEZVOUS ON Kₙ  —  A Search Game Simulator & Equilibrium Finder
══════════════════════════════════════════════════════════════════════════

Two players are dropped at distinct random nodes of the complete graph Kₙ.
Each round they independently choose a node to visit. They meet when they
collide. Neither knows the other's starting node — only their own.

Strategies are permutation-based: each player picks a random permutation
of {0,…,n−1} and cycles through it, starting from their own node.  The
simulator searches over mixtures of such "exploration orders" to discover
Nash equilibria via best-response dynamics, then paints the result.

    "Two strangers in a city of n streets,
     each walking a secret route,
     hoping the universe rhymes."

══════════════════════════════════════════════════════════════════════════
"""

import itertools
import numpy as np
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# §1  THE GAME ENGINE
# ─────────────────────────────────────────────────────────────────────────

class RendezvousGame:
    """
    Represents the rendezvous search game on the complete graph Kₙ.

    A *strategy* here is a probability distribution over deterministic
    "route plans".  A route plan of horizon T is a sequence of T nodes
    the player intends to visit on turns 1,2,…,T  (turn 0 = starting node).

    Because the graph is vertex-transitive and players only know their
    own start, we can WLOG fix Player 1 at node 0 and let Player 2
    start at node d ∈ {1,…,n−1} uniformly at random.
    """

    def __init__(self, n: int, horizon: int = 0):
        """
        Parameters
        ----------
        n : int
            Number of nodes in the complete graph.
        horizon : int
            Maximum number of turns to simulate (0 = auto = 2n).
        """
        self.n = n
        self.horizon = horizon if horizon > 0 else 2 * n

    # ── deterministic meeting time ───────────────────────────────────────

    def meeting_time(self, route_a: list[int], route_b: list[int],
                     start_a: int = 0, start_b: int = 1) -> int:
        """
        Return the first time t ∈ {0,1,…,T} at which both players occupy
        the same node, or T+1 if they never meet within the horizon.

        route_a[t] gives the node player A occupies at time t+1.
        At time 0 player A is at start_a, player B at start_b.
        """
        if start_a == start_b:
            return 0
        T = self.horizon
        for t in range(1, T + 1):
            node_a = route_a[(t - 1) % len(route_a)]
            node_b = route_b[(t - 1) % len(route_b)]
            if node_a == node_b:
                return t
        return T + 1          # failure sentinel

    # ── expected meeting time for a pair of mixed strategies ─────────────

    def expected_meeting_time(
        self,
        strat_a: "Strategy",
        strat_b: "Strategy",
        monte_carlo: int = 0,
    ) -> float:
        """
        Compute E[meeting time] averaged over:
          • the starting-gap  d ~ Uniform{1,…,n−1}
          • both players' randomised route choices

        If monte_carlo > 0, use that many samples.  Otherwise enumerate
        exactly (feasible for small strategy supports).
        """
        if monte_carlo > 0:
            return self._mc_emt(strat_a, strat_b, monte_carlo)
        return self._exact_emt(strat_a, strat_b)

    def _exact_emt(self, sa: "Strategy", sb: "Strategy") -> float:
        total = 0.0
        weight = 0.0
        n = self.n
        for ra, pa in sa.support():
            for rb, pb in sb.support():
                for d in range(1, n):
                    # shift route_b by offset d
                    rb_shifted = [(x + d) % n for x in rb]
                    t = self.meeting_time(ra, rb_shifted, 0, d)
                    total += pa * pb * t
                    weight += pa * pb
        return total / weight if weight > 0 else float('inf')

    def _mc_emt(self, sa: "Strategy", sb: "Strategy", samples: int) -> float:
        rng = np.random.default_rng()
        n = self.n
        times = []
        for _ in range(samples):
            d = rng.integers(1, n)
            ra = sa.sample(rng)
            rb_raw = sb.sample(rng)
            rb = [(x + d) % n for x in rb_raw]
            t = self.meeting_time(ra, rb, 0, d)
            times.append(t)
        return float(np.mean(times))


# ─────────────────────────────────────────────────────────────────────────
# §2  STRATEGIES
# ─────────────────────────────────────────────────────────────────────────

class Strategy:
    """
    A mixed strategy: a distribution over deterministic route plans.

    Each route plan is a tuple of node indices (relative to start=0).
    """

    def __init__(self):
        self._routes: list[tuple[int, ...]] = []
        self._probs: list[float] = []

    @staticmethod
    def pure(route: list[int]) -> "Strategy":
        s = Strategy()
        s._routes = [tuple(route)]
        s._probs = [1.0]
        return s

    @staticmethod
    def uniform_over(routes: list[list[int]]) -> "Strategy":
        s = Strategy()
        s._routes = [tuple(r) for r in routes]
        k = len(routes)
        s._probs = [1.0 / k] * k
        return s

    @staticmethod
    def from_weights(routes: list[list[int]], weights: list[float]) -> "Strategy":
        s = Strategy()
        total = sum(weights)
        s._routes = [tuple(r) for r in routes]
        s._probs = [w / total for w in weights]
        return s

    def support(self):
        """Yield (route_as_list, probability) for each route in support."""
        for r, p in zip(self._routes, self._probs):
            if p > 1e-12:
                yield list(r), p

    def sample(self, rng: np.random.Generator) -> list[int]:
        idx = rng.choice(len(self._routes), p=self._probs)
        return list(self._routes[idx])

    def size(self) -> int:
        return sum(1 for p in self._probs if p > 1e-12)

    def describe(self, top_k: int = 5) -> str:
        """Human-readable summary."""
        pairs = sorted(zip(self._probs, self._routes), reverse=True)
        lines = []
        for p, r in pairs[:top_k]:
            if p < 1e-12:
                break
            nodes = " → ".join(str(x) for x in r)
            lines.append(f"  [{p:6.2%}]  {nodes}")
        if len(pairs) > top_k:
            lines.append(f"  … and {len(pairs) - top_k} more routes")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# §3  STRATEGY GENERATORS  (the "library of wandering styles")
# ─────────────────────────────────────────────────────────────────────────

def all_permutation_routes(n: int) -> list[list[int]]:
    """
    Every permutation of {0,…,n−1} as a route (feasible for n ≤ 7).
    """
    return [list(p) for p in itertools.permutations(range(n))]


def sampled_permutation_routes(n: int, count: int,
                                rng: np.random.Generator) -> list[list[int]]:
    """Sample `count` random permutations of {0,…,n−1}."""
    routes = []
    seen = set()
    for _ in range(count * 3):
        p = tuple(rng.permutation(n))
        if p not in seen:
            seen.add(p)
            routes.append(list(p))
        if len(routes) >= count:
            break
    return routes


def canonical_strategies(n: int) -> list[list[int]]:
    """
    A curated set of "archetype" strategies for Kₙ:
      • The Patient:    stay at 0 forever
      • The Sweeper:    visit 0, 1, 2, …, n−1 in order
      • The Bouncer:    visit 0, n−1, 1, n−2, … (zigzag)
      • The Gambler:    a random permutation (one fixed seed)
    """
    patient = [0] * n
    sweeper = list(range(n))
    bouncer = []
    lo, hi = 0, n - 1
    while lo <= hi:
        bouncer.append(lo)
        if lo != hi:
            bouncer.append(hi)
        lo += 1
        hi -= 1
    rng = np.random.default_rng(42)
    gambler = list(rng.permutation(n))
    return [patient, sweeper, bouncer, list(gambler)]


# ─────────────────────────────────────────────────────────────────────────
# §4  EQUILIBRIUM SEARCH  (best-response dynamics)
# ─────────────────────────────────────────────────────────────────────────

class EquilibriumFinder:
    """
    Searches for a (symmetric) Nash equilibrium of the rendezvous game
    on Kₙ using best-response dynamics over a finite pool of pure
    strategies.

    At each iteration the current mixed strategy is updated toward the
    best-response mixture, and the process is repeated until convergence.
    """

    def __init__(self, game: RendezvousGame,
                 route_pool: Optional[list[list[int]]] = None,
                 mc_samples: int = 0,
                 verbose: bool = True):
        self.game = game
        self.verbose = verbose
        self.mc_samples = mc_samples
        n = game.n

        if route_pool is None:
            if n <= 6:
                route_pool = all_permutation_routes(n)
            else:
                rng = np.random.default_rng(0)
                route_pool = sampled_permutation_routes(n, min(200, n * 30), rng)
                route_pool += canonical_strategies(n)
        self.route_pool = route_pool
        self.K = len(route_pool)

        # payoff tensor:  M[i,j,d] = meeting time when
        #   player A uses route i, player B uses route j shifted by d
        self._build_payoff_tensor()

    def _build_payoff_tensor(self):
        g = self.game
        n, K = g.n, self.K
        pool = self.route_pool
        self.M = np.full((K, K, n - 1), g.horizon + 1, dtype=np.float64)
        for i in range(K):
            for j in range(K):
                for di, d in enumerate(range(1, n)):
                    rb_shifted = [(x + d) % n for x in pool[j]]
                    self.M[i, j, di] = g.meeting_time(pool[i], rb_shifted, 0, d)
        # average over starting gaps
        self.M_avg = self.M.mean(axis=2)  # shape (K, K)

    def best_response_weights(self, opp_weights: np.ndarray) -> np.ndarray:
        """
        Given opponent's mixed strategy (weight vector over pool),
        return a best-response weight vector (pure BR).
        """
        # expected meeting time for each of my routes
        emt = self.M_avg @ opp_weights       # shape (K,)
        br = np.argmin(emt)
        w = np.zeros(self.K)
        w[br] = 1.0
        return w

    def fictitious_play(self, max_iter: int = 300,
                        tol: float = 1e-5) -> dict:
        """
        Run fictitious play to approximate a symmetric Nash equilibrium.

        Returns a dict with keys:
          weights   – equilibrium mixture over route_pool
          emt       – expected meeting time at equilibrium
          history   – list of EMTs per iteration
          converged – whether the process converged
        """
        K = self.K
        rng = np.random.default_rng(7)
        counts = np.ones(K) / K          # cumulative counts (start uniform)
        history = []

        for t in range(1, max_iter + 1):
            weights = counts / counts.sum()
            br = self.best_response_weights(weights)
            counts += br
            weights_new = counts / counts.sum()

            # EMT at current mixture (symmetric game)
            emt = weights_new @ self.M_avg @ weights_new
            history.append(emt)

            if self.verbose and t % 25 == 0:
                print(f"  iter {t:4d}  |  EMT = {emt:.4f}  |  support = "
                      f"{np.sum(weights_new > 1e-6)}")

            if t > 10 and abs(history[-1] - history[-2]) < tol:
                if self.verbose:
                    print(f"  ✓ converged at iteration {t}")
                return dict(weights=weights_new, emt=emt,
                            history=history, converged=True)

        weights = counts / counts.sum()
        emt = weights @ self.M_avg @ weights
        return dict(weights=weights, emt=emt,
                    history=history, converged=False)

    def to_strategy(self, weights: np.ndarray) -> Strategy:
        """Convert weight vector back to a Strategy object."""
        active = [(self.route_pool[i], weights[i])
                  for i in range(self.K) if weights[i] > 1e-8]
        routes = [r for r, _ in active]
        ws = [w for _, w in active]
        return Strategy.from_weights(routes, ws)

    def payoff_matrix_summary(self, weights: np.ndarray, top_k: int = 6):
        """
        Pretty-print the payoff sub-matrix for the top-k routes by weight.
        """
        idxs = np.argsort(-weights)[:top_k]
        header = "       " + "".join(f"  R{j:<4d}" for j in idxs)
        lines = [header]
        for i in idxs:
            row = f"  R{i:<3d} "
            for j in idxs:
                row += f" {self.M_avg[i, j]:5.2f} "
            lines.append(row)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# §5  VISUALIZATION — "painting the rendezvous"
# ─────────────────────────────────────────────────────────────────────────

def visualize_equilibrium(game: RendezvousGame,
                          eq_result: dict,
                          finder: EquilibriumFinder,
                          save_path: str = "rendezvous_equilibrium.png"):
    """
    Produce a 2×2 figure:

      ┌──────────────────┬──────────────────┐
      │  (A) The Graph   │ (B) Strategy     │
      │   Kₙ with the    │  Clock — radial  │
      │   top routes      │  "sundial" of    │
      │   drawn as        │  node visits     │
      │   coloured arcs   │  over time       │
      ├──────────────────┼──────────────────┤
      │ (C) Convergence  │ (D) Meeting-Time │
      │  history of the  │  Heatmap across  │
      │  best-response   │  starting gaps   │
      │  dynamics        │  & time          │
      └──────────────────┴──────────────────┘
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.colors as mcolors
    import networkx as nx

    n = game.n
    weights = eq_result["weights"]
    history = eq_result["history"]

    # ── colour palette ───────────────────────────────────────────────────
    palette = plt.cm.twilight_shifted(np.linspace(0.1, 0.9, 8))

    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    fig.suptitle(
        f"Rendezvous on $K_{{{n}}}$  —  Equilibrium Landscape",
        fontsize=22, color="#e6edf3", fontweight="bold", y=0.97,
        fontfamily="serif",
    )
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.07, right=0.95, top=0.91, bottom=0.06)

    # ════════════════════════════════════════════════════════════════════
    # (A) THE GRAPH — Kₙ with top routes overlaid
    # ════════════════════════════════════════════════════════════════════
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_graph.set_facecolor("#0d1117")
    ax_graph.set_title("The Arena  —  $K_{" + str(n) + "}$ with top routes",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    G = nx.complete_graph(n)
    pos = nx.circular_layout(G)

    # draw faint edges
    nx.draw_networkx_edges(G, pos, ax=ax_graph,
                           edge_color="#21262d", width=0.5, alpha=0.4)
    # draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                           node_color="#161b22", node_size=500,
                           edgecolors="#58a6ff", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax_graph,
                            font_color="#e6edf3", font_size=11,
                            font_family="monospace")

    # overlay top routes as coloured arcs
    top_idxs = np.argsort(-weights)[:min(5, np.sum(weights > 1e-6))]
    for rank, idx in enumerate(top_idxs):
        route = finder.route_pool[idx]
        w = weights[idx]
        if w < 1e-8:
            continue
        color = palette[rank % len(palette)]
        alpha = 0.35 + 0.55 * (w / weights[top_idxs[0]])
        pts = [pos[route[t % len(route)]] for t in range(min(len(route), n + 1))]
        for i in range(len(pts) - 1):
            offset = 0.04 * (rank - len(top_idxs) / 2)
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            ax_graph.annotate(
                "", xy=(x1 + offset, y1 + offset),
                xytext=(x0 + offset, y0 + offset),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5 + 3 * w, alpha=alpha,
                                connectionstyle=f"arc3,rad={0.15 + 0.08 * rank}"),
            )
        # legend entry
        route_str = "→".join(str(x) for x in route[:min(5, len(route))])
        if len(route) > 5:
            route_str += "…"
        ax_graph.plot([], [], color=color, lw=3, alpha=alpha,
                      label=f"{w:.1%}  {route_str}")

    ax_graph.legend(loc="lower left", fontsize=8, facecolor="#161b22",
                    edgecolor="#30363d", labelcolor="#8b949e",
                    title="Route (weight)", title_fontsize=9)
    ax_graph.set_xlim(-1.5, 1.5)
    ax_graph.set_ylim(-1.5, 1.5)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    # ════════════════════════════════════════════════════════════════════
    # (B) STRATEGY CLOCK — a polar "sundial" of where you are each turn
    # ════════════════════════════════════════════════════════════════════
    ax_clock = fig.add_subplot(gs[0, 1], projection="polar")
    ax_clock.set_facecolor("#0d1117")
    ax_clock.set_title("Strategy Clock  —  where are you each turn?",
                       color="#8b949e", fontsize=13, pad=18, fontfamily="serif")

    T_show = min(game.horizon, 2 * n)
    theta_ticks = np.linspace(0, 2 * np.pi, T_show, endpoint=False)

    for rank, idx in enumerate(top_idxs):
        route = finder.route_pool[idx]
        w = weights[idx]
        if w < 1e-8:
            continue
        color = palette[rank % len(palette)]
        # radius = node index, angle = time step
        radii = [route[t % len(route)] for t in range(T_show)]
        thetas = list(theta_ticks)
        # close the loop
        radii.append(radii[0])
        thetas.append(thetas[0] + 2 * np.pi)
        ax_clock.plot(thetas, radii, color=color,
                      lw=1.5 + 3 * w, alpha=0.4 + 0.5 * (w / weights[top_idxs[0]]),
                      marker="o", markersize=3)

    ax_clock.set_rmax(n - 0.5)
    ax_clock.set_rticks(list(range(n)))
    ax_clock.set_yticklabels([f"v{i}" for i in range(n)],
                             fontsize=7, color="#8b949e")
    ax_clock.set_xticks(theta_ticks)
    ax_clock.set_xticklabels([f"t={t}" for t in range(T_show)],
                             fontsize=7, color="#58a6ff")
    ax_clock.tick_params(colors="#30363d")
    ax_clock.spines["polar"].set_color("#30363d")
    ax_clock.grid(color="#21262d", alpha=0.5)

    # ════════════════════════════════════════════════════════════════════
    # (C) CONVERGENCE HISTORY
    # ════════════════════════════════════════════════════════════════════
    ax_conv = fig.add_subplot(gs[1, 0])
    ax_conv.set_facecolor("#0d1117")
    ax_conv.set_title("Best-Response Dynamics  —  convergence",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    iters = np.arange(1, len(history) + 1)
    ax_conv.fill_between(iters, history, alpha=0.15, color="#58a6ff")
    ax_conv.plot(iters, history, color="#58a6ff", lw=2)
    ax_conv.axhline(history[-1], color="#f0883e", ls="--", lw=1, alpha=0.7)
    ax_conv.text(len(history) * 0.6, history[-1] + 0.05,
                 f"EMT* = {history[-1]:.4f}", color="#f0883e", fontsize=11,
                 fontfamily="serif")
    ax_conv.set_xlabel("Iteration", color="#8b949e", fontsize=10)
    ax_conv.set_ylabel("Expected Meeting Time", color="#8b949e", fontsize=10)
    ax_conv.tick_params(colors="#8b949e")
    for spine in ax_conv.spines.values():
        spine.set_color("#30363d")
    ax_conv.grid(color="#21262d", alpha=0.4)

    # ════════════════════════════════════════════════════════════════════
    # (D) MEETING-TIME HEATMAP — gap × time
    # ════════════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor("#0d1117")
    ax_heat.set_title("Meeting Probability  —  gap × time",
                       color="#8b949e", fontsize=13, pad=12, fontfamily="serif")

    # build the distribution: for each (gap d, time t) what fraction of
    # route-pairs meet exactly at time t?
    T_max = min(game.horizon, 2 * n)
    heat = np.zeros((n - 1, T_max + 1))
    total_w = 0.0
    for i in range(finder.K):
        for j in range(finder.K):
            w_ij = weights[i] * weights[j]
            if w_ij < 1e-15:
                continue
            for di, d in enumerate(range(1, n)):
                t = int(finder.M[i, j, di])
                if t <= T_max:
                    heat[di, t] += w_ij
                total_w += w_ij

    if total_w > 0:
        heat /= (total_w / (n - 1))   # normalise per gap

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "midnight_fire",
        ["#0d1117", "#1a1e2e", "#2d1b69", "#6f42c1", "#f0883e", "#f9d423"],
    )
    im = ax_heat.imshow(heat, aspect="auto", cmap=cmap, origin="lower",
                        extent=[0, T_max, 1, n])
    ax_heat.set_xlabel("Time step", color="#8b949e", fontsize=10)
    ax_heat.set_ylabel("Starting gap $d$", color="#8b949e", fontsize=10)
    ax_heat.tick_params(colors="#8b949e")
    for spine in ax_heat.spines.values():
        spine.set_color("#30363d")
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label("P(meet at time t | gap d)", color="#8b949e", fontsize=9)
    cbar.ax.tick_params(colors="#8b949e")

    # ── save ─────────────────────────────────────────────────────────────
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✦ Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# §6  THE RUNNER
# ─────────────────────────────────────────────────────────────────────────

def run(n: int = 4, verbose: bool = True):
    """
    Full pipeline: build game → search for equilibrium → visualise.
    """
    separator = "═" * 66
    print(f"\n{separator}")
    print(f"   RENDEZVOUS ON K_{n}")
    print(f"   Two strangers. {n} places. One hope: to collide.")
    print(f"{separator}\n")

    game = RendezvousGame(n)

    # ── build the route pool ─────────────────────────────────────────────
    if n <= 6:
        pool = all_permutation_routes(n)
        pool += canonical_strategies(n)
        # deduplicate
        seen = set()
        unique = []
        for r in pool:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        pool = unique
        print(f"  Route pool: {len(pool)} routes (full permutations + archetypes)")
    else:
        rng = np.random.default_rng(0)
        pool = sampled_permutation_routes(n, min(300, n * 40), rng)
        pool += canonical_strategies(n)
        seen = set()
        unique = []
        for r in pool:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        pool = unique
        print(f"  Route pool: {len(pool)} sampled routes + archetypes")

    # ── equilibrium search ───────────────────────────────────────────────
    print(f"\n  ▸ Launching best-response dynamics …\n")
    finder = EquilibriumFinder(game, route_pool=pool, verbose=verbose)
    eq = finder.fictitious_play(max_iter=400)

    # ── report ───────────────────────────────────────────────────────────
    eq_strat = finder.to_strategy(eq["weights"])
    print(f"\n{separator}")
    print(f"   EQUILIBRIUM FOUND {'(converged)' if eq['converged'] else '(max iter)'}")
    print(f"{separator}")
    print(f"\n  Expected meeting time  =  {eq['emt']:.4f} turns")
    print(f"  Support size           =  {eq_strat.size()} routes\n")
    print("  Top routes in the equilibrium mixture:\n")
    print(eq_strat.describe(top_k=8))
    print()

    # ── payoff sub-matrix ────────────────────────────────────────────────
    print("  Payoff sub-matrix (top routes):\n")
    print(finder.payoff_matrix_summary(eq["weights"], top_k=6))
    print()

    # ── benchmark against named strategies ───────────────────────────────
    print(f"\n  ▸ Benchmarking named strategies …\n")
    names = ["Patient (stay)", "Sweeper (0→n−1)", "Bouncer (zigzag)", "Gambler (rand)"]
    canonical = canonical_strategies(n)
    for name, route in zip(names, canonical):
        s = Strategy.pure(route)
        emt_vs_eq = game.expected_meeting_time(s, eq_strat, monte_carlo=5000)
        emt_vs_self = game.expected_meeting_time(s, s, monte_carlo=5000)
        route_str = "→".join(str(x) for x in route[:6])
        if len(route) > 6:
            route_str += "…"
        print(f"    {name:25s}  vs EQ: {emt_vs_eq:5.2f}   vs self: {emt_vs_self:5.2f}   [{route_str}]")
    print()

    # ── visualisation ────────────────────────────────────────────────────
    print("  ▸ Painting the equilibrium landscape …\n")
    save_path = f"rendezvous_K{n}_equilibrium.png"
    visualize_equilibrium(game, eq, finder, save_path=save_path)

    print(f"\n{separator}")
    print(f"   Done.  May the two strangers find each other.")
    print(f"{separator}\n")

    return game, finder, eq


# ─────────────────────────────────────────────────────────────────────────
# §7  MULTI-n COMPARISON
# ─────────────────────────────────────────────────────────────────────────

def compare_across_n(n_values: list[int] = None, save_path: str = "rendezvous_scaling.png"):
    """
    Run the equilibrium finder for several values of n and plot how the
    equilibrium expected meeting time scales.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if n_values is None:
        n_values = [2, 3, 4, 5, 6, 7, 8]

    results = {}
    for n in n_values:
        print(f"\n{'─' * 40}")
        print(f"  Running n = {n} …")
        print(f"{'─' * 40}")
        _, finder, eq = run(n, verbose=False)
        results[n] = eq["emt"]

    # ── plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ns = sorted(results.keys())
    emts = [results[nn] for nn in ns]

    ax.plot(ns, emts, "o-", color="#58a6ff", lw=2.5, markersize=10,
            markerfacecolor="#f0883e", markeredgecolor="#0d1117",
            markeredgewidth=2, label="Equilibrium EMT")

    # reference: n/2 (random walk lower bound intuition)
    ax.plot(ns, [nn / 2 for nn in ns], "--", color="#8b949e", lw=1.5,
            alpha=0.6, label="$n/2$ reference")

    ax.set_xlabel("n  (number of nodes in $K_n$)", color="#e6edf3", fontsize=13)
    ax.set_ylabel("Expected Meeting Time", color="#e6edf3", fontsize=13)
    ax.set_title("How Hard Is It to Find Each Other?",
                 color="#e6edf3", fontsize=17, fontweight="bold", fontfamily="serif")
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(color="#21262d", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✦ Scaling figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        ns = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
        compare_across_n(ns)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        run(n)
