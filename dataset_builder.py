"""exact GBC-scored Phase 2 dataset and GEN-CIM Stage A–D orchestration.

Stage A: exact GBC scores fixed-k initial groups; adaptive quality gate.
Native initializers gain one CentRA, one AdaAlg and one HEDGE group by default.
Degree, four GBC heuristics, CentRA, AdaAlg and HEDGE are protected from the quality
floor/ratio gate; all retained trajectories use the same endpoint/midpoint
weights downstream.
Each C++ expert supplies one initial group, then follows degree's trajectory,
labeling, weighting and downstream training path without a special branch.
Stage B: V_phi alone selects 1-opt neighbors (phase2_trajectory.py).
Stage C: exact GBC scores the first and last group of every trajectory.
Stage D: weighted retraining on scored endpoints and soft midpoint labels.
Optional B+ perturbation/crossover and elite buffer follow train_gim.py.
All scores are raw ordered-pair internal-node GBC in one consistent scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from gbc_types import GraphData
from phase2_trajectory import (SeedSet, Trajectory, TrajectoryDataset,
                               build_all_trajectories, init_seed_sets)
from value_net import (TrainConfig, ValueNetTrainer, ValueNetwork,
                       compute_seed_embedding_batch)
from exact_gbc_scorer import ExactGBCScorer, compile_exact_gbc
from cpp_experts import generate_cpp_expert_seed_sets


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Dataset labels, cache and conversion (GEN-CIM)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataSample:
    seed_set: SeedSet
    score: float
    weight: float
    is_exact_scored: bool
    trajectory_idx: int
    step_idx: int
    strategy: str = ""


ScoreCache = Dict[FrozenSet[int], float]


def build_dataset(graph: GraphData, trajectories: List[Trajectory],
                  scorer: ExactGBCScorer, strategy_names: Optional[List[str]] = None, *,
                  endpoint_weight: float = 1.0, midpoint_weight: float = 0.3,
                  cache: Optional[ScoreCache] = None,
                  use_proxy_for_midpoints: bool = True) -> Tuple[List[DataSample], ScoreCache]:
    """Label endpoints by exact GBC and midpoint steps by V_phi soft scores."""
    if scorer.graph.num_nodes != graph.num_nodes or scorer.k < 1:
        raise ValueError("Scorer does not match the current graph")
    if endpoint_weight <= 0 or midpoint_weight <= 0:
        raise ValueError("Sample weights must be positive")
    if strategy_names is not None and len(strategy_names) != len(trajectories):
        raise ValueError("Provide one strategy name per trajectory")
    cache = {} if cache is None else cache
    names = strategy_names or [""] * len(trajectories)
    endpoints = {}
    for trajectory in trajectories:
        for index, step in enumerate(trajectory):
            if index in (0, trajectory.length - 1) or not use_proxy_for_midpoints:
                key = frozenset(step.S.nodes)
                if key not in cache:
                    endpoints.setdefault(key, step.S)
    if endpoints:
        cache.update(zip(endpoints, scorer.score_many(list(endpoints.values()))))
    samples: List[DataSample] = []
    for trajectory_idx, (trajectory, name) in enumerate(zip(trajectories, names)):
        for step_idx, step in enumerate(trajectory):
            endpoint = step_idx in (0, trajectory.length - 1)
            scored = endpoint or not use_proxy_for_midpoints
            if scored:
                key = frozenset(step.S.nodes)
                score = cache[key]
            else:
                score = step.score
            samples.append(DataSample(step.S, score,
                                      endpoint_weight if endpoint else midpoint_weight,
                                      scored, trajectory_idx, step_idx, name))
    return samples, cache


def build_dataset_from_trajectory_dataset(graph: GraphData, traj_dataset: TrajectoryDataset,
                                          scorer: ExactGBCScorer,
                                          *, cache: Optional[ScoreCache] = None,
                                          use_proxy_for_midpoints: bool = True
                                          ) -> Tuple[List[DataSample], ScoreCache]:
    return build_dataset(graph, traj_dataset.trajectories, scorer,
                         traj_dataset.strategy_names,
                         endpoint_weight=traj_dataset.endpoint_weight,
                         midpoint_weight=traj_dataset.midpoint_weight,
                         cache=cache, use_proxy_for_midpoints=use_proxy_for_midpoints)


@dataclass
class DatasetStats:
    total_samples: int
    exact_scored: int
    proxy_labeled: int
    unique_seed_sets: int
    cache_hits: int
    cache_misses: int
    calls_saved: int
    score_min: float
    score_max: float
    score_mean: float
    elapsed_seconds: float


def build_dataset_with_stats(graph: GraphData, trajectories: List[Trajectory],
                             scorer: ExactGBCScorer, strategy_names: Optional[List[str]] = None, *,
                             endpoint_weight: float = 1.0, midpoint_weight: float = 0.3,
                             cache: Optional[ScoreCache] = None,
                             use_proxy_for_midpoints: bool = True
                             ) -> Tuple[List[DataSample], ScoreCache, DatasetStats]:
    start = time.perf_counter()
    before = scorer.calls
    samples, cache = build_dataset(graph, trajectories, scorer,
                                   strategy_names, endpoint_weight=endpoint_weight,
                                   midpoint_weight=midpoint_weight, cache=cache,
                                   use_proxy_for_midpoints=use_proxy_for_midpoints)
    scored = sum(s.is_exact_scored for s in samples)
    misses = scorer.calls - before
    values = [s.score for s in samples]
    return samples, cache, DatasetStats(
        len(samples), scored, len(samples) - scored,
        len({frozenset(s.seed_set.nodes) for s in samples}),
        max(0, scored - misses), misses, max(0, scored - misses),
        min(values, default=0.), max(values, default=0.),
        sum(values) / len(values) if values else 0., time.perf_counter() - start)


def samples_to_tensors(samples: List[DataSample], h_v: Tensor
                       ) -> Tuple[Tensor, Tensor, Tensor]:
    h_S = compute_seed_embedding_batch(h_v, [s.seed_set for s in samples])
    scores = torch.tensor([s.score for s in samples], dtype=torch.float32,
                          device=h_v.device)
    weights = torch.tensor([s.weight for s in samples], dtype=torch.float32,
                           device=h_v.device)
    return h_S, scores, weights


def filter_by_exact_scored(samples: List[DataSample]) -> List[DataSample]:
    return [s for s in samples if s.is_exact_scored]


def best_sample(samples: List[DataSample]) -> Optional[DataSample]:
    return max(samples, key=lambda s: (s.is_exact_scored, s.score)) if samples else None


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Adaptive quality gate and optional Stage B+ / elite buffer
# ═══════════════════════════════════════════════════════════════════════════════

ALWAYS_KEEP = {"degree", "gbc_diverse", "gbc_bridge", "gbc_hotspot_spread",
               "gbc_local_path"}
ALWAYS_KEEP_PREFIXES = ("centra_", "adaalg_", "hedge_")


def _always_keep(name: str) -> bool:
    """Protected initializers bypass both F_floor and the adaptive ratio gate."""
    return name in ALWAYS_KEEP or name.startswith(ALWAYS_KEEP_PREFIXES)


def quality_gate(seed_sets: Dict[str, SeedSet], scores: Sequence[float], *,
                 quality_floor: float = 1., quality_ratio: float = 0.5,
                 min_random_keep: int = 10, top_trajectories: int = 45
                 ) -> Dict[str, SeedSet]:
    """GEN-CIM gate, with native/CentRA/AdaAlg/HEDGE protected initializers.

    ``degree``, the four GBC heuristics, ``centra_*``, ``adaalg_*`` and ``hedge_*`` are
    retained regardless of ``quality_floor``/``quality_ratio``. The remaining
    initializers follow the original adaptive threshold, minimum quota and cap.
    """
    if len(seed_sets) != len(scores):
        raise ValueError("Need exactly one exact GBC score per initializer")
    protected = [(n, s, score) for (n, s), score in zip(seed_sets.items(), scores)
                 if _always_keep(n)]
    others = sorted(((n, s, score) for (n, s), score in zip(seed_sets.items(), scores)
                     if not _always_keep(n)),
                    key=lambda row: row[2], reverse=True)
    best = max((row[2] for row in others), default=0.)
    threshold = max(quality_floor, quality_ratio * best)
    kept = [row for row in others if row[2] >= threshold]
    if len(kept) < min_random_keep:
        kept = others[:min_random_keep]
    if top_trajectories > 0:
        kept = kept[:max(0, top_trajectories - len(protected))]
    return {name: S for name, S, _ in protected + kept}


def _augment(samples: List[DataSample], scorer: ExactGBCScorer, h_v: Tensor, *,
             perturb_sampling: bool, crossover: bool, seed: int,
             perturb_top_k: int = 5, perturb_r1_n: int = 10,
             perturb_r2_n: int = 5, perturb_r3_n: int = 2,
             perturb_threshold: float = 0.85, crossover_pairs: int = 3,
             crossover_n: int = 3) -> List[DataSample]:
    """GEN-CIM Stage B+: radius 1/2/3 perturbations and union crossover."""
    verified = sorted(filter_by_exact_scored(samples),
                      key=lambda s: s.score, reverse=True)[:perturb_top_k]
    rng = random.Random(seed + 9999)
    norms = h_v.norm(dim=1).cpu().tolist()
    extra = []
    proposals = []

    def add(candidate: SeedSet, threshold: float, weight: float, name: str):
        proposals.append((candidate, threshold, weight, name))

    if perturb_sampling:
        for rank, sample in enumerate(verified):
            selected = sorted(sample.seed_set.nodes, key=lambda v: (norms[v], v))
            available = sorted((v for v in range(scorer.graph.num_nodes)
                                if v not in sample.seed_set.nodes),
                               key=lambda v: (-norms[v], v))
            scale = max(.25, 1. - rank * .2)
            for radius, count, weight in ((1, perturb_r1_n, 1.),
                                          (2, perturb_r2_n, .9),
                                          (3, perturb_r3_n, .8)):
                size = min(radius, len(selected), len(available))
                if size == 0:
                    continue
                for i in range(max(1, round(count * scale))):
                    if radius == 1:
                        j = i % min(len(selected), len(available))
                        nodes = (sample.seed_set.nodes - {selected[j]}) | {available[j]}
                    else:
                        nodes = (sample.seed_set.nodes - set(rng.sample(selected, size))) | set(
                            rng.sample(available, size))
                    add(SeedSet(nodes), perturb_threshold * sample.score,
                        weight, f"perturb_r{radius}_rank{rank}")
    if crossover:
        pairs = [(a, b) for a in range(len(verified))
                 for b in range(a + 1, len(verified))][:crossover_pairs]
        for a, b in pairs:
            union = sorted(verified[a].seed_set.nodes | verified[b].seed_set.nodes)
            for _ in range(crossover_n):
                child = SeedSet(set(rng.sample(union, scorer.k)))
                add(child, perturb_threshold * max(verified[a].score, verified[b].score),
                    1., f"crossover_ep{a}xep{b}")
    if proposals:
        scores = scorer.score_many([candidate for candidate, _, _, _ in proposals])
        for (candidate, threshold, weight, name), score in zip(proposals, scores):
            if score >= threshold:
                extra.append(DataSample(candidate, score, weight,
                                        True, len(samples) + len(extra), 0, name))
    return extra


def _elite(samples: List[DataSample], top_k: int = 5, weight: float = 4.,
           rank_weights: bool = True, temperature: float = 2.) -> List[DataSample]:
    pool = filter_by_exact_scored(samples)
    if len(pool) < top_k:
        pool = samples
    elite = sorted(pool, key=lambda s: s.score, reverse=True)[:top_k]
    if not elite:
        return []
    if rank_weights:
        weights = [weight * .5 ** i for i in range(len(elite))]
    else:
        mean = sum(s.score for s in samples) / len(samples)
        raw = [math.exp((s.score - mean) / temperature) for s in elite]
        weights = [weight * x / max(raw) for x in raw]
    return [DataSample(s.seed_set, s.score, weights[i], s.is_exact_scored,
                       len(samples) + i, s.step_idx, f"elite_{s.strategy}")
            for i, s in enumerate(elite)]


@dataclass
class Phase2Result:
    samples: List[DataSample]
    cache: ScoreCache
    h_G: Tensor
    value_net: ValueNetwork
    value_trainer: ValueNetTrainer
    seed_sets: Dict[str, SeedSet]
    trajectories: TrajectoryDataset


def run_phase2(graph: GraphData, h_v: Tensor, scorer: ExactGBCScorer, k: int, *,
               node_gbc: Optional[Tensor] = None,
               H: int = 5, k_neighbors: int = 5, random_seed: int = 42,
               quality_floor: float = 1., quality_ratio: float = 0.5,
               min_random_keep: int = 10, top_trajectories: int = 45,
               perturb_sampling: bool = False, crossover: bool = False,
               elite_top_k: int = 5, elite_weight: float = 4.,
               value_epochs: int = 500,
               centra_source: Optional[Union[str, Path]] = None,
               adaalg_source: Optional[Union[str, Path]] = None,
               hedge_source: Optional[Union[str, Path]] = None,
               cpp_binary_dir: Optional[Union[str, Path]] = None,
               centra_count: int = 1,
               adaalg_count: int = 1,
               hedge_count: int = 1) -> Phase2Result:
    """Complete A–D runner with exact GBC in all former MC scoring positions.

    ``h_v`` and ``node_gbc`` come from Phase 1. Stage A starts from the native
    42 initializers, then appends CentRA, AdaAlg and HEDGE groups according to
    their counts (one per expert by default). Their C++ estimates are never
    used as labels: ExactGBCScorer
    re-scores all initializers on the same raw ordered-pair internal-node scale.
    """
    if scorer.k != k or scorer.graph.num_nodes != graph.num_nodes:
        raise ValueError("Stage A/C scorer needs the same graph and k")
    if centra_count < 0 or adaalg_count < 0 or hedge_count < 0:
        raise ValueError("CentRA/AdaAlg/HEDGE expert counts must be non-negative")

    h_G = h_v.mean(dim=0)
    print("[Phase2/Stage A] Đang tạo các nghiệm expert ban đầu...", flush=True)
    seed_sets = init_seed_sets(graph, k, h_v, random_seed,
                               node_gbc=node_gbc)
    native_count = len(seed_sets)

    cpp_seed_sets = generate_cpp_expert_seed_sets(
        graph,
        scorer.graph_path,
        k,
        centra_source=centra_source,
        adaalg_source=adaalg_source,
        hedge_source=hedge_source,
        binary_dir=cpp_binary_dir,
        centra_count=centra_count,
        adaalg_count=adaalg_count,
        hedge_count=hedge_count,
        base_seed=random_seed,
    )
    overlap = set(seed_sets) & set(cpp_seed_sets)
    if overlap:
        raise ValueError(f"C++ expert names collide with native initializers: {sorted(overlap)}")
    seed_sets.update(cpp_seed_sets)

    print(
        f"[Phase2/Stage A] Có {len(seed_sets)} expert trước lọc: "
        f"{native_count} native + {centra_count} CentRA + {adaalg_count} AdaAlg "
        f"+ {hedge_count} HEDGE. "
        f"CentRA/AdaAlg/HEDGE cùng degree + 4 GBC expert được bảo vệ khỏi F_floor.",
        flush=True,
    )
    print(f"[Phase2/Stage A] Chấm {len(seed_sets)} expert bằng exact GBC...",
          flush=True)
    scores = scorer.score_many(list(seed_sets.values()))
    best_name, best_score = max(zip(seed_sets, scores), key=lambda row: row[1])
    print(f"[Phase2/Stage A] GBC expert: min={min(scores):.6f}, "
          f"tốt nhất={best_score:.6f} ({best_name}); "
          f"lọc ordinary experts với floor={quality_floor}, ratio={quality_ratio}.",
          flush=True)
    before_gate = len(seed_sets)
    seed_sets = quality_gate(seed_sets, scores, quality_floor=quality_floor,
                             quality_ratio=quality_ratio,
                             min_random_keep=min_random_keep,
                             top_trajectories=top_trajectories)
    if not seed_sets:
        raise ValueError("Quality gate removed all seed sets")
    protected_after = sum(_always_keep(name) for name in seed_sets)
    print(
        f"[Phase2/Stage A] Sau lọc còn {len(seed_sets)}/{before_gate} expert; "
        f"{protected_after} protected expert được giữ không cần vượt F_floor.",
        flush=True,
    )

    value = ValueNetwork(embed_dim=h_v.size(1)).to(h_v.device)
    bootstrap = compute_seed_embedding_batch(h_v, list(seed_sets.values()))
    labels = torch.tensor([scorer.score(s) for s in seed_sets.values()],
                          device=h_v.device, dtype=torch.float32)
    print(f"[Phase2/Stage A] Huấn luyện V_phi lần 1 trên "
          f"{len(seed_sets)} nhãn exact, tối đa {value_epochs} epoch.", flush=True)
    trainer = ValueNetTrainer(value, TrainConfig(epochs=value_epochs,
                              patience=50, log_every=100),
                              device=h_v.device)
    trainer.fit(bootstrap, labels)
    print(f"[Phase2/Stage A] V_phi lần 1 xong sau "
          f"{len(trainer.train_losses)} epoch, "
          f"training loss tốt nhất={min(trainer.train_losses):.6f}.", flush=True)

    trajectories = build_all_trajectories(
        graph, k, h_v, value, H=H, k_neighbors=k_neighbors,
        h_G=h_G, seed_sets_override=seed_sets)

    # Keep the Stage A scores in cache so initial endpoints are reused.
    cache: ScoreCache = {frozenset(s.nodes): scorer.score(s)
                         for s in seed_sets.values()}
    print("[Phase2/Stage C] Đang gán nhãn exact GBC cho đầu/cuối quỹ đạo; "
          "bước giữa dùng nhãn mềm V_phi.", flush=True)
    samples, cache = build_dataset_from_trajectory_dataset(
        graph, trajectories, scorer, cache=cache)
    exact = [s.score for s in samples if s.is_exact_scored]
    print(f"[Phase2/Stage C] Có {len(samples)} mẫu: {len(exact)} nhãn exact, "
          f"{len(samples) - len(exact)} nhãn V_phi; "
          f"GBC exact tốt nhất={max(exact):.6f}.", flush=True)

    features, targets, weights = samples_to_tensors(samples, h_v)
    print(f"[Phase2/Stage D] Huấn luyện V_phi có trọng số trên "
          f"{len(samples)} mẫu, tối đa {value_epochs} epoch.", flush=True)
    trainer = ValueNetTrainer(value, TrainConfig(epochs=value_epochs,
                              patience=50, log_every=100),
                              device=h_v.device)
    trainer.fit(features, targets, sample_weights=weights)
    print(f"[Phase2/Stage D] V_phi xong sau {len(trainer.train_losses)} epoch; "
          f"training loss tốt nhất={min(trainer.train_losses):.6f}.", flush=True)

    if perturb_sampling or crossover:
        print("[Phase2/Stage B+] Chấm thêm nghiệm perturbation/crossover...",
              flush=True)
        samples += _augment(samples, scorer, h_v, perturb_sampling=perturb_sampling,
                            crossover=crossover, seed=random_seed)
        cache.update(scorer.cache)
    samples += _elite(samples, top_k=elite_top_k, weight=elite_weight)
    print(f"[Phase2] Hoàn thành dữ liệu: {len(samples)} mẫu, "
          f"{sum(s.is_exact_scored for s in samples)} nhãn exact, "
          f"GBC exact tốt nhất={max(s.score for s in samples if s.is_exact_scored):.6f}.",
          flush=True)
    return Phase2Result(samples, cache, h_G, value, trainer, seed_sets, trajectories)


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick smoke test with the real exact C++ evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    from tempfile import TemporaryDirectory
    from graph_utils import load_edge_list
    from phase2_trajectory import TrajectoryStep
    torch.manual_seed(42)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.txt"
        path.write_text("0 1\n1 2\n2 3\n3 4\n4 0\n", encoding="utf-8")
        graph, _ = load_edge_list(path)
        binary = compile_exact_gbc(Path(__file__).with_name("exact_gbc.cpp"),
                                   Path(tmp) / "exact_gbc")
        scorer = ExactGBCScorer(graph, path, binary, 2)
        first = SeedSet({1, 2})
        assert scorer.score(first) == scorer.score(SeedSet({2, 1}))
        assert scorer.calls == 1
        trajectory = Trajectory()
        trajectory.append(TrajectoryStep(first, 999.))
        trajectory.append(TrajectoryStep(SeedSet({2, 3}), 123.))
        trajectory.append(TrajectoryStep(SeedSet({3, 4}), 888.))
        samples, cache, stats = build_dataset_with_stats(graph, [trajectory], scorer)
        assert len(samples) == 3 and stats.exact_scored == 2
        assert samples[1].score == 123. and not samples[1].is_exact_scored
        assert samples[0].score == scorer.score(first) and samples[0].weight == 1.
        assert samples[1].weight == .3

        gate = quality_gate({
            "degree": first,
            "gbc_diverse": first,
            "gbc_bridge": first,
            "gbc_hotspot_spread": first,
            "gbc_local_path": first,
            "centra_00": first,
            "adaalg_00": first,
            "hedge_00": first,
            "good_random_0": first,
        }, [0.] * 9, quality_floor=10., min_random_keep=0)
        assert set(gate) == ALWAYS_KEEP | {"centra_00", "adaalg_00", "hedge_00"}

        calls = scorer.calls
        build_dataset(graph, [trajectory], scorer, cache=cache)
        assert scorer.calls == calls
        h_v = torch.randn(5, 8)
        # Keep the tiny smoke test fast: C++ expert integration has its own
        # subprocess path and is disabled here.
        result = run_phase2(
            graph, h_v, scorer, 2, node_gbc=torch.arange(5.),
            H=1, value_epochs=2, elite_top_k=2,
            centra_count=0, adaalg_count=0, hedge_count=0,
        )
        assert len(result.seed_sets) <= 42 and result.samples
        saved_value = Path(tmp) / "value.pt"
        result.value_trainer.save(str(saved_value))
        assert saved_value.is_file()
        assert all(s.is_exact_scored for s in result.samples
                   if s.strategy.startswith("elite_"))
        try:
            ExactGBCScorer(graph, path, binary, 6)
        except ValueError:
            pass
        else:
            raise AssertionError("A scorer with an invalid k was accepted")
    print("dataset_builder.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
