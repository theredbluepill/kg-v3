"""Run scripts/run_ppo.py's main() with observe-only receipts, any GPU topology.

Generalised from ../pod4-2026-09-30/main-J-4rank/main_probe.py (sha256
26ba5b0c..., the wrapper of the live 4-rank runs, e.g. W&B pcy5knet). The
[nt-probe] records, the trainer instrumentation and the W&B online gate are
the same; the only behaviour change is where the NUMA binding comes from:

- main_probe.py hardcoded pod abl4mvr5w1mmn4 (GPU_NODE {0:0,1:0,2:1,3:1},
  NODE_CPUS node0 0-63,128-191 / node1 64-127,192-255).
- This file discovers it at runtime, before torch loads:
  GPU index -> PCI bus id (`nvidia-smi --query-gpu=index,pci.bus_id`, or the
  sorted /proc/driver/nvidia/gpus/<bdf> names when nvidia-smi is missing)
  -> /sys/bus/pci/devices/<bdf>/numa_node
  -> /sys/devices/system/node/node<N>/cpulist,
  intersected with the CPUs this process may use (sched_getaffinity).

Binding semantics are kept from main_probe.py: every rank gets its GPU's WHOLE
node CPU set; ranks that share a node share that set (the 4-GPU pod gave 2
ranks per node the same 128 CPUs, see its numa_bind records). There is no
even split. A rank needs about native_threads + 2 busy threads (OMP=1), so the
binding reports `oversubscribed=true` (and a warning line) only when a node
has fewer CPUs than ranks_on_node * (KG_NT_NATIVE_THREADS + 2); it never
changes the binding by itself.

Edge cases:
- numa_node = -1 (single-node host or firmware without PCI affinity), or no
  numa_node file on an existing device (kernel without CONFIG_NUMA): the
  affinity is left at the process's allowed set; the record says why.
- A GPU bus id with no /sys/bus/pci/devices entry is an error (bad mapping).
- LOCAL_RANK beyond the GPUs (or beyond CUDA_VISIBLE_DEVICES): ValueError
  naming the counts, before torch loads.
- CUDA_VISIBLE_DEVICES with numeric entries maps LOCAL_RANK to that physical
  index; UUID/MIG entries are refused (set KG_NT_NUMA=off instead).
- The mapping assumes CUDA orders devices like nvidia-smi (PCI bus order);
  launch.sh exports CUDA_DEVICE_ORDER=PCI_BUS_ID so that holds by definition.

Environment:
- KG_NT_NUMA=cpu: CPU affinity only (what the live runs use; the pod container
  returns EPERM for set_mempolicy). KG_NT_NUMA=1: affinity + MPOL_BIND.
  Unset or "off": no binding.
- KG_NT_NATIVE_THREADS (default 4): only for the oversubscription report.

`python main_probe_auto.py --topology [N_RANKS]` prints the binding every local
rank would get (JSON) and exits; bootstrap.sh and launch.sh record it.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
SYSFS_PCI = "/sys/bus/pci/devices"
SYSFS_NODE = "/sys/devices/system/node"
PROC_NVIDIA_GPUS = "/proc/driver/nvidia/gpus"


def emit(kind: str, **fields: Any) -> None:
    record = {"kind": kind, "rank": RANK, "t": time.time(), **fields}
    print("[nt-probe] " + json.dumps(record, default=str), flush=True)


# ---------------------------------------------------------------- parsing


def normalize_bus_id(raw: str) -> str:
    """nvidia-smi's '00000000:06:00.0' -> sysfs's '0000:06:00.0' (lowercase).

    sysfs prints the PCI domain with %04x, so a domain above 0xffff keeps its
    extra digits; nvidia-smi pads it to 8.
    """
    text = raw.strip().lower()
    parts = text.split(":")
    if len(parts) == 2:  # bus:dev.fn without a domain
        parts = ["0", *parts]
    if len(parts) != 3 or "." not in parts[2]:
        raise ValueError(f"not a PCI bus id: {raw!r}")
    domain = int(parts[0], 16)
    bus = int(parts[1], 16)
    dev_s, fn_s = parts[2].split(".", 1)
    return f"{domain:04x}:{bus:02x}:{int(dev_s, 16):02x}.{int(fn_s, 16):x}"


def parse_nvidia_smi_bus_ids(text: str) -> Dict[int, str]:
    """Parse `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`."""
    out: Dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("index"):
            continue  # a header, if --format=csv was used
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}")
        index = int(fields[0])
        if index in out:
            raise ValueError(f"duplicate GPU index {index} in nvidia-smi output")
        out[index] = normalize_bus_id(fields[1])
    if not out:
        raise ValueError("nvidia-smi listed no GPUs")
    if sorted(out) != list(range(len(out))):
        raise ValueError(f"GPU indices are not 0..{len(out) - 1}: {sorted(out)}")
    return out


def bus_ids_from_proc_names(names: Sequence[str]) -> Dict[int, str]:
    """/proc/driver/nvidia/gpus/<bdf> names -> {index: bdf} in PCI bus order.

    nvidia-smi numbers GPUs in PCI bus order by default, so sorting the bus ids
    reproduces its indices. Fallback only; nvidia-smi is preferred.
    """
    ids = sorted(normalize_bus_id(n) for n in names)
    if not ids:
        raise ValueError(f"no GPUs under {PROC_NVIDIA_GPUS}")
    return dict(enumerate(ids))


def parse_cpulist(text: str) -> List[int]:
    """Kernel cpulist '0-63,128-191' -> sorted CPU ids ('' -> [])."""
    cpus: Set[int] = set()
    text = text.strip()
    if not text:
        return []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            raise ValueError(f"strided cpulist not supported: {text!r}")
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"bad cpulist range {part!r} in {text!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpulist(cpus: Sequence[int]) -> str:
    """Sorted CPU ids -> compact cpulist (the inverse of parse_cpulist)."""
    ranges: List[str] = []
    ordered = sorted(set(cpus))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        ranges.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(ranges)


def parse_visible_devices(value: Optional[str]) -> Optional[List[int]]:
    """CUDA_VISIBLE_DEVICES -> physical indices, or None when unset/empty."""
    if value is None or value.strip() == "":
        return None
    entries = [e.strip() for e in value.split(",") if e.strip()]
    if not all(e.isdigit() for e in entries):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={value!r} is not a list of numeric indices; "
            "UUID/MIG entries cannot be mapped to PCI bus ids here. Use numeric "
            "indices or set KG_NT_NUMA=off."
        )
    return [int(e) for e in entries]


# ---------------------------------------------------------------- binding


@dataclass(frozen=True)
class Binding:
    local_rank: int
    gpu_index: int
    bus_id: str
    node: int
    cpus: Tuple[int, ...]
    ranks_on_node: int
    oversubscribed: bool
    reason: str

    def record(self) -> Dict[str, Any]:
        rec = asdict(self)
        rec["cpulist"] = format_cpulist(self.cpus)
        rec["cpus"] = len(self.cpus)
        return rec


def gpu_for_local_rank(
    local_rank: int, bus_ids: Dict[int, str], visible: Optional[List[int]]
) -> int:
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    if visible is None:
        if local_rank >= len(bus_ids):
            raise ValueError(
                f"LOCAL_RANK {local_rank} has no GPU: the host has {len(bus_ids)} "
                f"GPU(s) (indices 0..{len(bus_ids) - 1}); lower --nproc-per-node"
            )
        return local_rank
    if local_rank >= len(visible):
        raise ValueError(
            f"LOCAL_RANK {local_rank} has no GPU: CUDA_VISIBLE_DEVICES lists "
            f"{len(visible)} device(s) {visible}; lower --nproc-per-node"
        )
    index = visible[local_rank]
    if index not in bus_ids:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES entry {index} is not a GPU index on this host "
            f"(0..{len(bus_ids) - 1})"
        )
    return index


def resolve_binding(
    *,
    local_rank: int,
    local_world_size: int,
    bus_ids: Dict[int, str],
    visible: Optional[List[int]],
    numa_node_of: Callable[[str], int],
    node_cpus: Callable[[int], List[int]],
    allowed: Set[int],
    native_threads: int,
) -> Binding:
    """The CPU set for one local rank: its GPU's whole NUMA node (∩ allowed)."""
    if local_world_size < 1:
        raise ValueError(f"LOCAL_WORLD_SIZE must be >= 1, got {local_world_size}")
    if local_rank >= local_world_size:
        raise ValueError(
            f"LOCAL_RANK {local_rank} >= LOCAL_WORLD_SIZE {local_world_size}"
        )
    if not allowed:
        raise ValueError("this process may run on no CPU (empty affinity)")
    gpu = gpu_for_local_rank(local_rank, bus_ids, visible)
    bus = bus_ids[gpu]
    node = numa_node_of(bus)
    peers = [
        numa_node_of(bus_ids[gpu_for_local_rank(r, bus_ids, visible)])
        for r in range(local_world_size)
    ]
    ranks_on_node = sum(1 for n in peers if n == node)
    if node < 0:
        cpus = sorted(allowed)
        reason = (
            f"numa_node={node} for {bus}: single-node host or no PCI affinity; "
            "affinity left at the process's allowed set"
        )
    else:
        on_node = node_cpus(node)
        if not on_node:
            raise ValueError(f"NUMA node {node} (GPU {gpu}, {bus}) lists no CPUs")
        cpus = sorted(set(on_node) & allowed)
        if not cpus:
            raise ValueError(
                f"NUMA node {node} CPUs {format_cpulist(on_node)} do not overlap "
                f"this process's allowed CPUs {format_cpulist(sorted(allowed))}"
            )
        reason = f"GPU {gpu} {bus} -> node {node}; whole node set shared by {ranks_on_node} rank(s)"
        if len(cpus) < len(on_node):
            reason += f" (cut to the allowed set: {len(cpus)} of {len(on_node)})"
    oversubscribed = len(cpus) < ranks_on_node * (native_threads + 2)
    return Binding(
        local_rank=local_rank,
        gpu_index=gpu,
        bus_id=bus,
        node=node,
        cpus=tuple(cpus),
        ranks_on_node=ranks_on_node,
        oversubscribed=oversubscribed,
        reason=reason,
    )


# ---------------------------------------------------------------- host reads


def read_bus_ids() -> Tuple[Dict[int, str], str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return parse_nvidia_smi_bus_ids(completed.stdout), "nvidia-smi"
    except (OSError, subprocess.SubprocessError) as error:
        try:
            names = os.listdir(PROC_NVIDIA_GPUS)
        except OSError:
            raise RuntimeError(
                f"cannot map GPUs to PCI bus ids: nvidia-smi failed ({error}) and "
                f"{PROC_NVIDIA_GPUS} is unreadable; set KG_NT_NUMA=off to run unbound"
            ) from error
        return bus_ids_from_proc_names(names), f"{PROC_NVIDIA_GPUS} (nvidia-smi failed: {error})"


def make_numa_node_of(sysfs_pci: str = SYSFS_PCI) -> Callable[[str], int]:
    def numa_node_of(bus: str) -> int:
        device = Path(sysfs_pci) / bus
        if not device.is_dir():
            raise RuntimeError(
                f"PCI device {device} not found (GPU bus id mapping is wrong?); "
                "set KG_NT_NUMA=off to run unbound"
            )
        path = device / "numa_node"
        if not path.exists():
            # A kernel built without CONFIG_NUMA has no numa_node files: one node.
            return -1
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"cannot read {path} ({error}); set KG_NT_NUMA=off to run unbound"
            ) from error

    return numa_node_of


def make_node_cpus(sysfs_node: str = SYSFS_NODE) -> Callable[[int], List[int]]:
    def node_cpus(node: int) -> List[int]:
        path = Path(sysfs_node) / f"node{node}" / "cpulist"
        return parse_cpulist(path.read_text())

    return node_cpus


def host_binding(local_rank: int, local_world_size: int) -> Tuple[Binding, str]:
    bus_ids, source = read_bus_ids()
    binding = resolve_binding(
        local_rank=local_rank,
        local_world_size=local_world_size,
        bus_ids=bus_ids,
        visible=parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES")),
        numa_node_of=make_numa_node_of(),
        node_cpus=make_node_cpus(),
        allowed=set(os.sched_getaffinity(0)),
        native_threads=int(os.environ.get("KG_NT_NATIVE_THREADS", "4")),
    )
    return binding, source


def _bind_numa(*, memory: bool) -> None:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    binding, source = host_binding(LOCAL_RANK, local_world_size)
    os.sched_setaffinity(0, binding.cpus)
    fields = binding.record()
    fields.update(
        local_world_size=local_world_size,
        bus_source=source,
        cpus=len(os.sched_getaffinity(0)),
        mempolicy_mode=None,
    )
    if binding.oversubscribed:
        print(
            f"[nt-probe] WARNING rank {RANK}: {binding.ranks_on_node} ranks share "
            f"{len(binding.cpus)} CPUs on node {binding.node}",
            flush=True,
        )
    if memory and binding.node >= 0:
        if platform.machine() != "x86_64":
            raise OSError(f"set_mempolicy syscall numbers are x86_64-only, got {platform.machine()}")
        libc = ctypes.CDLL(None, use_errno=True)
        mask = ctypes.c_ulong(1 << binding.node)
        # x86_64: set_mempolicy = 238, get_mempolicy = 239; MPOL_BIND = 2.
        if libc.syscall(238, 2, ctypes.byref(mask), 64) != 0:
            raise OSError(ctypes.get_errno(), "set_mempolicy failed")
        mode = ctypes.c_int(-1)
        got = ctypes.c_ulong(0)
        if libc.syscall(239, ctypes.byref(mode), ctypes.byref(got), 64, None, 0) != 0:
            raise OSError(ctypes.get_errno(), "get_mempolicy failed")
        fields.update(mempolicy_mode=mode.value, mempolicy_nodemask=got.value)
    emit("numa_bind", **fields)


def topology_report(n_ranks: int) -> Dict[str, Any]:
    bus_ids, source = read_bus_ids()
    visible = parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    numa_node_of = make_numa_node_of()
    node_cpus = make_node_cpus()
    allowed = set(os.sched_getaffinity(0))
    native_threads = int(os.environ.get("KG_NT_NATIVE_THREADS", "4"))
    ranks = [
        resolve_binding(
            local_rank=r,
            local_world_size=n_ranks,
            bus_ids=bus_ids,
            visible=visible,
            numa_node_of=numa_node_of,
            node_cpus=node_cpus,
            allowed=allowed,
            native_threads=native_threads,
        ).record()
        for r in range(n_ranks)
    ]
    return {
        "bus_source": source,
        "gpus": {str(k): v for k, v in sorted(bus_ids.items())},
        "allowed_cpus": len(allowed),
        "native_threads": native_threads,
        "ranks": ranks,
        "oversubscribed": any(r["oversubscribed"] for r in ranks),
    }


# ---------------------------------------------------------------- probe

_STEP = {"calls": 0, "seconds": 0.0}
_ITER = {"n": 0}
_KEEP = ("time/", "perf/", "train/", "loss/", "optimizer/")


def _proc_stat() -> Tuple[int, int]:
    with open("/proc/stat") as handle:
        fields = [int(x) for x in handle.readline().split()[1:]]
    idle = fields[3] + fields[4]
    total = sum(fields[:8])
    return total - idle, total


def _gate_wandb() -> None:
    import wandb

    original_init = wandb.init

    def checked_init(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("project") != "kg-v3":
            raise RuntimeError(f"unexpected W&B project {kwargs.get('project')!r}")
        run = original_init(*args, **kwargs)
        if run.offline or run.disabled:
            raise RuntimeError("W&B run is not online; refusing to continue")
        emit("wandb", url=run.url, id=run.id, name=run.name, group=kwargs.get("group"))
        return run

    wandb.init = checked_init


def _instrument() -> None:
    from owl.kaggriculture import env as kenv
    from owl.train import ppo

    cls = kenv.KaggricultureVectorizedEnv
    original_step = cls.step
    original_init = cls.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        emit(
            "env_construct",
            n_envs=kwargs.get("n_envs"),
            native_threads=kwargs.get("native_threads"),
            seed=kwargs.get("seed"),
            seed_stride=kwargs.get("seed_stride"),
            transfer_device=str(kwargs.get("transfer_device")),
            affinity_cpus=len(os.sched_getaffinity(0)),
        )
        original_init(self, *args, **kwargs)

    def step(self: Any, actions: Any) -> Any:
        start = time.perf_counter()
        result = original_step(self, actions)
        _STEP["seconds"] += time.perf_counter() - start
        _STEP["calls"] += 1
        return result

    cls.__init__ = init
    cls.step = step

    original_iteration = ppo.PPOTrainer.train_iteration

    def train_iteration(self: Any) -> Any:
        _ITER["n"] += 1
        calls0, secs0 = _STEP["calls"], _STEP["seconds"]
        cpu0 = os.times()
        start = time.perf_counter()
        metrics = original_iteration(self)
        end = time.perf_counter()
        cpu1 = os.times()
        busy, total = _proc_stat()
        emit(
            "iteration",
            iteration=_ITER["n"],
            perf_end=end,
            wall_seconds=end - start,
            native_step_calls=_STEP["calls"] - calls0,
            native_step_seconds=_STEP["seconds"] - secs0,
            proc_cpu_seconds=(cpu1.user + cpu1.system) - (cpu0.user + cpu0.system),
            proc_cpu_total=cpu1.user + cpu1.system,
            stat_busy=busy,
            stat_total=total,
            metrics={
                k: v for k, v in metrics.items() if RANK == 0 or k.startswith(_KEEP)
            },
        )
        return metrics

    ppo.PPOTrainer.train_iteration = train_iteration


def main(argv: List[str]) -> None:
    if len(argv) >= 2 and argv[1] == "--topology":
        n = int(argv[2]) if len(argv) > 2 else len(read_bus_ids()[0])
        print(json.dumps(topology_report(n), indent=1))
        return
    if len(argv) < 2:
        raise SystemExit("usage: main_probe_auto.py scripts/run_ppo.py ARGS... | --topology [N]")
    # Bind before torch (and its thread pools) load, as main_probe.py did.
    numa = os.environ.get("KG_NT_NUMA", "off")
    if numa in ("1", "cpu"):
        _bind_numa(memory=numa == "1")
    elif numa != "off":
        raise ValueError(f"KG_NT_NUMA must be off, cpu or 1, got {numa!r}")

    import torch

    _gate_wandb()
    script = Path(argv[1]).resolve()
    sys.argv = [str(script), *argv[2:]]
    _instrument()
    spec = importlib.util.spec_from_file_location("run_ppo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ppo"] = module
    spec.loader.exec_module(module)
    emit(
        "start",
        argv=sys.argv,
        torch=torch.__version__,
        omp=os.environ.get("OMP_NUM_THREADS"),
        numa=numa,
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
    )
    module.main()


if __name__ == "__main__":
    main(sys.argv)
