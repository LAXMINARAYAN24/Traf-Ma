"""Ground-truth world snapshot (evaluator/sensor-layer only) and the TraCI adapter.

Agents NEVER receive a WorldSnapshot. They receive only the views produced by sensors.py.
"""
from __future__ import annotations

import os
import subprocess
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VehicleState:
    vid: str
    edge_id: str
    lane_pos: float
    speed: float
    accel: float
    waiting_s: float
    x: float
    y: float
    length: float = 5.0


@dataclass
class SignalState:
    state: str            # SUMO red/yellow/green string for the whole junction
    phase_index: int
    remaining_s: float    # time until the current phase ends


@dataclass
class WorldSnapshot:
    time: float
    vehicles: dict = field(default_factory=dict)       # vid -> VehicleState
    signal: Optional[SignalState] = None


@dataclass
class StepEvents:
    departed: list
    arrived: list
    collisions: list
    teleports: list
    emergency_stops: list


class SumoStartError(RuntimeError):
    pass


class TraciWorld:
    def __init__(self, sumocfg: Path, cfg, tls_id: str, label: str = "trafma"):
        self.sumocfg, self.cfg, self.tls_id, self.label = Path(sumocfg), cfg, tls_id, label
        self.conn = None
        self.closed_cleanly: Optional[bool] = None
        self._vlen = cfg.vehicle.length_m
        self.sumo_version = "unknown"

    # ---------------------------------------------------------------- lifecycle
    def start(self, gui: bool = False):
        import sumolib
        import traci
        binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
        cmd = [binary, "-c", self.sumocfg.name, "--seed", str(self.cfg.run.seed)]
        if gui:
            cmd += ["--start", "--quit-on-end", "--delay", str(self.cfg.run.gui_delay_ms)]
        try:
            # run inside the scenario dir so the relative log/output paths land next to the config
            cwd = os.getcwd()
            os.chdir(self.sumocfg.parent)
            try:
                traci.start(cmd, label=self.label)
            finally:
                os.chdir(cwd)
        except Exception as e:
            raise SumoStartError(f"could not start SUMO ({' '.join(cmd)}): {e}") from e
        self.conn = traci.getConnection(self.label)
        try:
            self.sumo_version = str(self.conn.getVersion()[1])
        except Exception:
            pass
        self._tc = traci.constants
        self._vars = [self._tc.VAR_SPEED, self._tc.VAR_POSITION, self._tc.VAR_LANEPOSITION,
                      self._tc.VAR_ROAD_ID, self._tc.VAR_WAITING_TIME, self._tc.VAR_ACCELERATION]

    def close(self):
        if self.conn is None:
            return
        try:
            self.conn.close()
            self.closed_cleanly = True
        except Exception:
            self.closed_cleanly = False
        finally:
            self.conn = None

    # ---------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        return self.conn.simulation.getTime()

    def step(self) -> StepEvents:
        c = self.conn
        c.simulationStep()
        dep = list(c.simulation.getDepartedIDList())
        for v in dep:
            c.vehicle.subscribe(v, self._vars)
        return StepEvents(
            departed=sorted(dep),
            arrived=sorted(c.simulation.getArrivedIDList()),
            collisions=sorted(c.simulation.getCollidingVehiclesIDList()),
            teleports=sorted(c.simulation.getStartingTeleportIDList()),
            emergency_stops=sorted(c.simulation.getEmergencyStoppingVehiclesIDList()),
        )

    def snapshot(self) -> WorldSnapshot:
        c, tc = self.conn, self._tc
        now = self.time
        vs = {}
        for vid, r in sorted(c.vehicle.getAllSubscriptionResults().items()):
            x, y = r[tc.VAR_POSITION]
            vs[vid] = VehicleState(vid, r[tc.VAR_ROAD_ID], r[tc.VAR_LANEPOSITION], r[tc.VAR_SPEED],
                                   r[tc.VAR_ACCELERATION], r[tc.VAR_WAITING_TIME], x, y, self._vlen)
        sig = SignalState(c.trafficlight.getRedYellowGreenState(self.tls_id),
                          c.trafficlight.getPhase(self.tls_id),
                          max(c.trafficlight.getNextSwitch(self.tls_id) - now, 0.0))
        return WorldSnapshot(now, vs, sig)

    # ---------------------------------------------------------------- GUI helpers
    def color_vehicle(self, vid: str, rgba: tuple):
        try:
            self.conn.vehicle.setColor(vid, rgba)
        except Exception:
            pass

    def wall(self) -> float:
        return _time.monotonic()
