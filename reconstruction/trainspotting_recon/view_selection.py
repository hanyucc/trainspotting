"""View selection on the covisibility graph of the reconstruction."""

import json
from pathlib import Path

import numpy as np

from .config import Config
from .scene import Scene


def _spectral_order(W: np.ndarray) -> np.ndarray:
    """Order nodes by the Fiedler vector of the normalized Laplacian, per connected component."""
    n = len(W)
    unvisited, components = set(range(n)), []
    while unvisited:
        stack, component = [next(iter(unvisited))], set()
        while stack:
            v = stack.pop()
            if v not in component:
                component.add(v)
                stack += [u for u in np.nonzero(W[v])[0] if u not in component]
        components.append(sorted(component))
        unvisited -= component
    components.sort(key=len, reverse=True)
    order = []
    for comp in components:
        if len(comp) <= 2:
            order += comp
            continue
        sub = W[np.ix_(comp, comp)]
        d = sub.sum(1)
        d[d == 0] = 1
        inv_sqrt = np.diag(1 / np.sqrt(d))
        _, vecs = np.linalg.eigh(np.eye(len(comp)) - inv_sqrt @ sub @ inv_sqrt)
        order += [comp[k] for k in np.argsort(vecs[:, 1])]
    return np.array(order)


def _best_cut(W: np.ndarray, nodes: list[int]):
    """Lowest-conductance sweep cut of the induced subgraph: (conductance, side A, side B)."""
    sub = W[np.ix_(nodes, nodes)]
    order = _spectral_order(sub)
    volume = sub.sum()
    best, best_k = np.inf, None
    for k in range(2, len(nodes) - 2):
        side = np.zeros(len(nodes), bool)
        side[order[k:]] = True
        vol_side = sub[side].sum()
        denominator = min(vol_side, volume - vol_side)
        if denominator <= 0:
            best, best_k = 0.0, k
            continue
        conductance = sub[np.ix_(side, ~side)].sum() / denominator
        if conductance < best:
            best, best_k = conductance, k
    if best_k is None:
        return None
    return best, [nodes[i] for i in order[:best_k]], [nodes[i] for i in order[best_k:]]


def partition(shared: np.ndarray, cfg: Config) -> list[list[int]]:
    """Recursive spectral bisection with edge weights log(1 + shared points); largest cluster first.

    A cut is accepted when its conductance is below `max_conductance`, unless both sides contain at least
    `bridge_images` views that each share at least `bridge_points` points with the other side.
    """
    W = np.log1p(shared.astype(np.float64))

    def split(nodes):
        if len(nodes) < cfg.min_cluster_size:
            return [nodes]
        cut = _best_cut(W, nodes)
        if cut is None or cut[0] >= cfg.max_conductance:
            return [nodes]
        _, a, b = cut
        block = shared[np.ix_(a, b)] >= cfg.bridge_points
        if min(int(block.any(1).sum()), int(block.any(0).sum())) >= cfg.bridge_images:
            return [nodes]
        return split(a) + split(b)

    return sorted(split(list(range(len(shared)))), key=len, reverse=True)


def select_views(scene: Scene, cfg: Config, out_dir: Path) -> list[str]:
    """Keep views with enough confirmed points, a valid depth range, in the largest covisibility cluster,
    and sharing at least one point with another kept view. Writes `view_selection.json`."""
    shared, confirmed, total = scene.covisibility()
    names = [v.name for v in scene.views]
    reasons = {name: [] for name in names}
    for i, view in enumerate(scene.views):
        if confirmed[i] < cfg.min_confirmed_points:
            reasons[view.name].append("few_confirmed_points")
        if confirmed[i] < cfg.min_confirmed_fraction * max(total[i], 1):
            reasons[view.name].append("low_confirmed_fraction")
        if scene.depth_range(view, cfg.min_valid_depth) is None:
            reasons[view.name].append("invalid_depth_range")

    survivors = [i for i, name in enumerate(names) if not reasons[name]]
    if len(survivors) >= cfg.min_cluster_size:
        clusters = partition(shared[np.ix_(survivors, survivors)], cfg)
        for cluster in clusters[1:]:
            for k in cluster:
                reasons[names[survivors[k]]].append("minority_cluster")

    while True:
        kept = [i for i, name in enumerate(names) if not reasons[name]]
        isolated = [kept[k] for k in np.flatnonzero(shared[np.ix_(kept, kept)].sum(1) == 0)] if kept else []
        if not isolated:
            break
        for i in isolated:
            reasons[names[i]].append("isolated")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "view_selection.json").write_text(json.dumps({
        name: {"kept": not reasons[name], "confirmed_points": int(confirmed[i]),
               "triangulated_points": int(total[i]), "reasons": reasons[name]}
        for i, name in enumerate(names)}, indent=1))
    return [name for name in names if not reasons[name]]
