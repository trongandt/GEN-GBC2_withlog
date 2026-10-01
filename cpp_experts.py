"""Run CentRA/AdaAlg/HEDGE C++ solvers as Phase-2 initializer experts.

All supplied C++ programs emit node labels in the original edge-list ID space.
GEN-GBC uses contiguous internal IDs, so this module converts every returned
group through graph_utils' sorted external->internal mapping before creating a
SeedSet.

The C++ expert score/estimate is intentionally NOT used as a Phase-2 label.
Every returned group is re-scored by ExactGBCScorer together with the native
GEN-GBC initializers, keeping one exact GBC scale throughout the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Dict, Optional, Union

from gbc_types import GraphData
from graph_utils import load_edge_list
from phase2_trajectory import SeedSet


PathLike = Union[str, Path]


def compile_cpp_expert(source: PathLike, binary: PathLike) -> Path:
    """Compile one supplied C++ expert only when its source changed."""
    source = Path(source).resolve()
    binary = Path(binary).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"C++ expert source missing: {source}")
    if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        binary.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Phase2/CPP Experts] Đang biên dịch {source.name}...", flush=True)
        started = time.perf_counter()
        subprocess.run(
            ["g++", "-O2", "-std=c++17", "-Wall", "-Wextra",
             str(source), "-o", str(binary)],
            check=True,
        )
        print(
            f"[Phase2/CPP Experts] Biên dịch {source.name} xong sau "
            f"{time.perf_counter() - started:.1f}s.",
            flush=True,
        )
    return binary


def _node_map_for_graph(graph: GraphData, graph_path: Path) -> Dict[int, int]:
    loaded, node_map = load_edge_list(graph_path, directed=graph.directed)
    if loaded.num_nodes != graph.num_nodes or not loaded.edge_index.cpu().equal(
        graph.edge_index.cpu()
    ):
        raise ValueError("C++ expert edge list and GEN-GBC graph use different ID maps")
    return node_map


def _run_expert(
    binary: Path,
    graph_path: Path,
    graph: GraphData,
    node_map: Dict[int, int],
    k: int,
    seed: int,
    display_name: str,
) -> SeedSet:
    cmd = [
        str(binary),
        "--graph", str(graph_path),
        "--k", str(k),
        "--seed", str(seed),
    ]
    if graph.directed:
        cmd.append("--directed")

    started = time.perf_counter()
    process = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(
            f"{display_name} failed for seed={seed} with code "
            f"{process.returncode}: {detail}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{display_name} returned invalid JSON for seed={seed}: "
            f"{process.stdout[:500]}"
        ) from error

    external_nodes = payload.get("nodes")
    if not isinstance(external_nodes, list) or len(external_nodes) != k:
        raise ValueError(
            f"{display_name} seed={seed} returned {external_nodes!r}; "
            f"expected exactly k={k} node labels"
        )
    if len(set(external_nodes)) != k:
        raise ValueError(f"{display_name} seed={seed} returned duplicate nodes")

    try:
        internal_nodes = {node_map[int(v)] for v in external_nodes}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{display_name} seed={seed} returned a node outside the graph mapping"
        ) from error
    if len(internal_nodes) != k:
        raise ValueError(f"{display_name} seed={seed} collapsed after ID conversion")

    status = payload.get("status", "unknown")
    print(
        f"[Phase2/CPP Experts] {display_name} seed={seed} xong sau "
        f"{time.perf_counter() - started:.1f}s; status={status}; "
        f"nodes_internal={sorted(internal_nodes)}.",
        flush=True,
    )
    return SeedSet(internal_nodes)


def generate_cpp_expert_seed_sets(
    graph: GraphData,
    graph_path: PathLike,
    k: int,
    *,
    centra_source: Optional[PathLike] = None,
    adaalg_source: Optional[PathLike] = None,
    hedge_source: Optional[PathLike] = None,
    binary_dir: Optional[PathLike] = None,
    centra_count: int = 1,
    adaalg_count: int = 1,
    hedge_count: int = 1,
    base_seed: int = 42,
) -> Dict[str, SeedSet]:
    """Generate ``centra_*``, ``adaalg_*`` and ``hedge_*`` fixed-k initializers.

    By default each solver is called once and contributes one initial group,
    treated like the degree initializer by the shared Phase-2 pipeline.
    Explicit counts control how many runs each solver contributes.
    The experts use separate deterministic seed streams. Duplicate node sets
    remain valid expert records; every record keeps the same downstream sample
    weight as the native protected experts.
    """
    if centra_count < 0 or adaalg_count < 0 or hedge_count < 0:
        raise ValueError("CentRA/AdaAlg/HEDGE expert counts must be non-negative")
    if centra_count == 0 and adaalg_count == 0 and hedge_count == 0:
        return {}
    if not 1 <= k < graph.num_nodes:
        raise ValueError("CentRA/AdaAlg/HEDGE require 1 <= k < |V|")

    graph_path = Path(graph_path).resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph edge list missing: {graph_path}")
    node_map = _node_map_for_graph(graph, graph_path)

    binary_root = Path(binary_dir or graph_path.parent / ".gbc_cpp_experts").resolve()
    binary_root.mkdir(parents=True, exist_ok=True)
    result: Dict[str, SeedSet] = {}

    if centra_count:
        source = Path(
            centra_source or Path(__file__).with_name("centra.cpp")
        ).resolve()
        binary = compile_cpp_expert(source, binary_root / "centra_binary")
        print(
            f"[Phase2/CPP Experts] Sinh {centra_count} nghiệm CentRA độc lập...",
            flush=True,
        )
        for i in range(centra_count):
            seed = base_seed + i
            result[f"centra_{i:02d}"] = _run_expert(
                binary, graph_path, graph, node_map, k, seed, "CentRA"
            )

    if adaalg_count:
        source = Path(
            adaalg_source or Path(__file__).with_name("adaalg.cpp")
        ).resolve()
        binary = compile_cpp_expert(source, binary_root / "adaalg_binary")
        print(
            f"[Phase2/CPP Experts] Sinh {adaalg_count} nghiệm AdaAlg độc lập...",
            flush=True,
        )
        # Keep the two algorithms on disjoint deterministic seed streams.
        for i in range(adaalg_count):
            seed = base_seed + 100_000 + i
            result[f"adaalg_{i:02d}"] = _run_expert(
                binary, graph_path, graph, node_map, k, seed, "AdaAlg"
            )

    if hedge_count:
        source = Path(
            hedge_source or Path(__file__).with_name("hedge.cpp")
        ).resolve()
        binary = compile_cpp_expert(source, binary_root / "hedge_binary")
        print(
            f"[Phase2/CPP Experts] Sinh {hedge_count} nghiệm HEDGE độc lập...",
            flush=True,
        )
        # HEDGE samples its own paths and never receives another expert's group.
        for i in range(hedge_count):
            seed = base_seed + 200_000 + i
            result[f"hedge_{i:02d}"] = _run_expert(
                binary, graph_path, graph, node_map, k, seed, "HEDGE"
            )

    expected = centra_count + adaalg_count + hedge_count
    if len(result) != expected or not all(s.k == k for s in result.values()):
        raise AssertionError("C++ expert generation returned the wrong number/size of groups")
    print(
        f"[Phase2/CPP Experts] Đã tạo đủ {len(result)} nghiệm "
        f"({centra_count} CentRA + {adaalg_count} AdaAlg + {hedge_count} HEDGE).",
        flush=True,
    )
    return result
