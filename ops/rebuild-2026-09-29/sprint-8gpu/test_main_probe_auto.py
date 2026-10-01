"""Mac-side tests of main_probe_auto.py's topology parsing (no GPU, no torch).

Run: python3 test_main_probe_auto.py   (or: python3 -m pytest test_main_probe_auto.py)
Kit-local on purpose; not part of the repo test suite.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main_probe_auto as m  # noqa: E402

# Pod abl4mvr5w1mmn4 (../pod4-2026-09-30/env/pod/hardware.txt).
POD4_SMI = """\
0, 00000000:06:00.0
1, 00000000:77:00.0
2, 00000000:F4:00.0
3, 00000000:F5:00.0
"""
POD4_NUMA = {"0000:06:00.0": 0, "0000:77:00.0": 0, "0000:f4:00.0": 1, "0000:f5:00.0": 1}
POD4_NODES = {0: "0-63,128-191\n", 1: "64-127,192-255\n"}
# main_probe.py's hardcoded table, which the discovery must reproduce.
POD4_GPU_NODE = {0: 0, 1: 0, 2: 1, 3: 1}
POD4_NODE_CPUS = {0: [*range(0, 64), *range(128, 192)], 1: [*range(64, 128), *range(192, 256)]}

# A typical 8-GPU 2-socket box: 4 GPUs per node, with a header line (csv).
EIGHT_SMI = """\
index, pci.bus_id
0, 00000000:18:00.0
1, 00000000:2A:00.0
2, 00000000:3A:00.0
3, 00000000:5D:00.0
4, 00000000:9A:00.0
5, 00000000:AB:00.0
6, 00000000:BA:00.0
7, 00000000:DB:00.0
"""
EIGHT_NUMA = {
    "0000:18:00.0": 0, "0000:2a:00.0": 0, "0000:3a:00.0": 0, "0000:5d:00.0": 0,
    "0000:9a:00.0": 1, "0000:ab:00.0": 1, "0000:ba:00.0": 1, "0000:db:00.0": 1,
}
EIGHT_NODES = {0: "0-47,96-143", 1: "48-95,144-191"}


def binding(local_rank, world, smi, numa, nodes, allowed=None, visible=None, threads=4):
    bus_ids = m.parse_nvidia_smi_bus_ids(smi)
    node_lists = {k: m.parse_cpulist(v) for k, v in nodes.items()}
    all_cpus = set().union(*node_lists.values()) if node_lists else set(range(8))
    return m.resolve_binding(
        local_rank=local_rank,
        local_world_size=world,
        bus_ids=bus_ids,
        visible=visible,
        numa_node_of=lambda bus: numa[bus],
        node_cpus=lambda n: node_lists[n],
        allowed=set(all_cpus if allowed is None else allowed),
        native_threads=threads,
    )


class ParseTests(unittest.TestCase):
    def test_bus_id_normalization(self):
        self.assertEqual(m.normalize_bus_id("00000000:F4:00.0"), "0000:f4:00.0")
        self.assertEqual(m.normalize_bus_id(" 0000:06:00.0 "), "0000:06:00.0")
        self.assertEqual(m.normalize_bus_id("00000001:3B:00.0"), "0001:3b:00.0")
        self.assertEqual(m.normalize_bus_id("00010000:01:00.0"), "10000:01:00.0")
        self.assertEqual(m.normalize_bus_id("06:00.0"), "0000:06:00.0")
        with self.assertRaises(ValueError):
            m.normalize_bus_id("GPU-1234")

    def test_nvidia_smi_parse(self):
        self.assertEqual(
            m.parse_nvidia_smi_bus_ids(POD4_SMI),
            {0: "0000:06:00.0", 1: "0000:77:00.0", 2: "0000:f4:00.0", 3: "0000:f5:00.0"},
        )
        self.assertEqual(len(m.parse_nvidia_smi_bus_ids(EIGHT_SMI)), 8)
        for bad in ("", "0, 00000000:06:00.0\n0, 00000000:07:00.0", "1, 00000000:06:00.0",
                    "0, 00000000:06:00.0, extra"):
            with self.assertRaises(ValueError, msg=bad):
                m.parse_nvidia_smi_bus_ids(bad)

    def test_proc_fallback_orders_by_bus(self):
        names = ["0000:f5:00.0", "0000:06:00.0", "0000:F4:00.0", "0000:77:00.0"]
        self.assertEqual(
            m.bus_ids_from_proc_names(names), m.parse_nvidia_smi_bus_ids(POD4_SMI)
        )
        with self.assertRaises(ValueError):
            m.bus_ids_from_proc_names([])

    def test_cpulist(self):
        self.assertEqual(m.parse_cpulist("0-63,128-191\n"), POD4_NODE_CPUS[0])
        self.assertEqual(m.parse_cpulist("0,2,4-5"), [0, 2, 4, 5])
        self.assertEqual(m.parse_cpulist(""), [])
        self.assertEqual(m.parse_cpulist("\n"), [])
        self.assertEqual(m.parse_cpulist("7"), [7])
        for bad in ("5-3", "0-7:2", "a-b"):
            with self.assertRaises(ValueError, msg=bad):
                m.parse_cpulist(bad)
        for text in ("0-63,128-191", "0,2,4-5", "7", "64-127,192-255"):
            self.assertEqual(m.format_cpulist(m.parse_cpulist(text)), text)

    def test_visible_devices(self):
        self.assertIsNone(m.parse_visible_devices(None))
        self.assertIsNone(m.parse_visible_devices(""))
        self.assertEqual(m.parse_visible_devices("4,5,6,7"), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            m.parse_visible_devices("GPU-8f2c,GPU-11aa")


class BindingTests(unittest.TestCase):
    def test_reproduces_pod4_hardcoded_table(self):
        for rank in range(4):
            b = binding(rank, 4, POD4_SMI, POD4_NUMA, POD4_NODES)
            self.assertEqual(b.node, POD4_GPU_NODE[rank])
            self.assertEqual(list(b.cpus), POD4_NODE_CPUS[POD4_GPU_NODE[rank]])
            self.assertEqual(b.ranks_on_node, 2)
            self.assertFalse(b.oversubscribed)
            self.assertEqual(b.record()["cpus"], 128)  # as the live numa_bind records

    def test_eight_gpus_two_nodes(self):
        seen = {}
        for rank in range(8):
            b = binding(rank, 8, EIGHT_SMI, EIGHT_NUMA, EIGHT_NODES)
            self.assertEqual(b.gpu_index, rank)
            self.assertEqual(b.ranks_on_node, 4)
            self.assertEqual(len(b.cpus), 96)  # whole node, shared (no split)
            self.assertFalse(b.oversubscribed)  # 96 >= 4 * (4 + 2)
            seen.setdefault(b.node, set()).add(b.cpus)
        self.assertEqual({k: len(v) for k, v in seen.items()}, {0: 1, 1: 1})

    def test_oversubscription_flag(self):
        small = {0: "0-15", 1: "16-31"}
        b = binding(0, 8, EIGHT_SMI, EIGHT_NUMA, small)  # 16 CPUs < 4 * 6
        self.assertTrue(b.oversubscribed)
        self.assertFalse(binding(0, 8, EIGHT_SMI, EIGHT_NUMA, small, threads=2).oversubscribed)

    def test_numa_minus_one_keeps_allowed_set(self):
        numa = {bus: -1 for bus in EIGHT_NUMA}
        b = binding(3, 8, EIGHT_SMI, numa, {}, allowed=range(32))
        self.assertEqual(b.node, -1)
        self.assertEqual(list(b.cpus), list(range(32)))
        self.assertEqual(b.ranks_on_node, 8)
        self.assertTrue(b.oversubscribed)  # 32 < 8 * 6: reported, not changed
        self.assertIn("numa_node=-1", b.reason)

    def test_local_rank_beyond_gpus(self):
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK 4 has no GPU: the host has 4"):
            binding(4, 8, POD4_SMI, POD4_NUMA, POD4_NODES)
        with self.assertRaisesRegex(ValueError, "LOCAL_WORLD_SIZE"):
            binding(2, 2, POD4_SMI, POD4_NUMA, POD4_NODES)

    def test_cuda_visible_devices_maps_physical(self):
        b = binding(1, 4, EIGHT_SMI, EIGHT_NUMA, EIGHT_NODES, visible=[4, 5, 6, 7])
        self.assertEqual((b.gpu_index, b.node, b.ranks_on_node), (5, 1, 4))
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES lists 2"):
            binding(2, 3, EIGHT_SMI, EIGHT_NUMA, EIGHT_NODES, visible=[0, 1])
        with self.assertRaisesRegex(ValueError, "not a GPU index"):
            binding(0, 1, POD4_SMI, POD4_NUMA, POD4_NODES, visible=[9])

    def test_allowed_set_cuts_and_empty_overlap(self):
        b = binding(0, 4, POD4_SMI, POD4_NUMA, POD4_NODES, allowed=range(0, 32))
        self.assertEqual(list(b.cpus), list(range(32)))
        self.assertIn("cut to the allowed set", b.reason)
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            binding(2, 4, POD4_SMI, POD4_NUMA, POD4_NODES, allowed=range(0, 32))

    def test_node_without_cpus(self):
        with self.assertRaisesRegex(ValueError, "lists no CPUs"):
            binding(0, 4, POD4_SMI, POD4_NUMA, {0: "", 1: "64-127"}, allowed=range(256))


class SysfsReaderTests(unittest.TestCase):
    def test_readers_on_fake_tree(self):
        with tempfile.TemporaryDirectory() as root:
            pci = Path(root, "pci")
            node = Path(root, "node")
            for bus, n in POD4_NUMA.items():
                (pci / bus).mkdir(parents=True)
                (pci / bus / "numa_node").write_text(f"{n}\n")
            for n, text in POD4_NODES.items():
                (node / f"node{n}").mkdir(parents=True)
                (node / f"node{n}" / "cpulist").write_text(text)
            numa_of = m.make_numa_node_of(str(pci))
            cpus_of = m.make_node_cpus(str(node))
            self.assertEqual(numa_of("0000:f4:00.0"), 1)
            self.assertEqual(cpus_of(1), POD4_NODE_CPUS[1])
            with self.assertRaisesRegex(RuntimeError, "not found"):
                numa_of("0000:99:00.0")
            # Kernel without CONFIG_NUMA: the device exists, numa_node does not.
            (pci / "0000:aa:00.0").mkdir()
            self.assertEqual(numa_of("0000:aa:00.0"), -1)
            (pci / "0000:bb:00.0").mkdir()
            (pci / "0000:bb:00.0" / "numa_node").write_text("garbage\n")
            with self.assertRaisesRegex(RuntimeError, "cannot read"):
                numa_of("0000:bb:00.0")

    def test_main_rejects_unknown_numa_mode(self):
        old = os.environ.get("KG_NT_NUMA")
        os.environ["KG_NT_NUMA"] = "yes"
        try:
            with self.assertRaisesRegex(ValueError, "KG_NT_NUMA"):
                m.main(["main_probe_auto.py", "scripts/run_ppo.py"])
        finally:
            if old is None:
                del os.environ["KG_NT_NUMA"]
            else:
                os.environ["KG_NT_NUMA"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
