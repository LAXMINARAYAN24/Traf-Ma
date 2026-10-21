import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trafma.config import load_config  # noqa: E402
from trafma.world import SignalState, VehicleState, WorldSnapshot  # noqa: E402


def _sumo_ok():
    try:
        import sumolib
        sumolib.checkBinary("sumo"); sumolib.checkBinary("netconvert")
        return True
    except Exception:
        return False


needs_sumo = pytest.mark.skipif(not _sumo_ok(), reason="SUMO not installed (run: python -m trafma doctor)")


@pytest.fixture(scope="session")
def topo(tmp_path_factory):
    if not _sumo_ok():
        pytest.skip("SUMO not installed")
    from trafma.netgen import build_network, Topology
    d = tmp_path_factory.mktemp("net")
    _, tp, _, _ = build_network(load_config(), d)
    return Topology.load(tp)


@pytest.fixture
def cfg():
    return load_config()


def veh(vid, approach_edge, lane_pos, speed=0.0, waiting=0.0, topo=None, length=5.0):
    return VehicleState(vid, approach_edge, lane_pos, speed, 0.0, waiting, 0.0, 0.0, length)


def world(vehicles, topo, phase_index=0, remaining=10.0):
    state = topo.signal_phases[phase_index]["state"]
    return WorldSnapshot(10.0, {v.vid: v for v in vehicles}, SignalState(state, phase_index, remaining))
