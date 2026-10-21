"""P1.2 - Build the single four-arm signalised intersection with SUMO's netconvert.

Layout (arm length L, one lane per direction):
        N
        |
   W ---C--- E        edges: <A>_in  (arm end -> junction),  <A>_out (junction -> arm end)
        |
        S
Every approach has one RSU beside its incoming road. The signal uses split phasing
(one approach green at a time, then yellow, then all-red) so conflicting movements can
never be green together.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from xml.sax.saxutils import quoteattr

from . import APPROACHES
from .config import Config

TLS_ID = "C"


class NetworkBuildError(RuntimeError):
    pass


def in_edge(a: str) -> str:
    return f"{a}_in"


def out_edge(a: str) -> str:
    return f"{a}_out"


def movement_dest(origin: str, turn: str) -> str:
    """Clockwise N,E,S,W. Facing the junction from N (heading south): left=E, right=W."""
    i = APPROACHES.index(origin)
    off = {"left": 1, "straight": 2, "right": 3}[turn]
    return APPROACHES[(i + off) % 4]


def route_id(origin: str, dest: str) -> str:
    return f"r_{origin}_{dest}"


def movements():
    for o in APPROACHES:
        for t in ("straight", "left", "right"):
            yield o, t, movement_dest(o, t)


@dataclass
class ApproachGeom:
    name: str
    in_edge: str
    out_edge: str
    in_lane: str
    out_lane: str
    in_len: float
    out_len: float
    stop_xy: tuple            # lane end (stop line)
    start_xy: tuple           # lane start (arm end)
    dir_xy: tuple             # unit travel direction on the incoming lane
    rsu_xy: tuple
    link_indices: list = field(default_factory=list)   # TLS link indices belonging to this approach
    zone_polygon: list = field(default_factory=list)   # for the GUI overlay


@dataclass
class Topology:
    tls_id: str
    n_links: int
    cycle_s: float
    approaches: dict                      # name -> ApproachGeom
    arm_length_m: float
    signal_phases: list                   # [{"duration":..,"state":..,"label":..}]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=1, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "Topology":
        d = json.loads(text)
        ap = {k: ApproachGeom(**{**v, "stop_xy": tuple(v["stop_xy"]), "start_xy": tuple(v["start_xy"]),
                                 "dir_xy": tuple(v["dir_xy"]), "rsu_xy": tuple(v["rsu_xy"])})
              for k, v in d["approaches"].items()}
        return Topology(d["tls_id"], d["n_links"], d["cycle_s"], ap, d["arm_length_m"], d["signal_phases"])

    @staticmethod
    def load(path) -> "Topology":
        return Topology.from_json(Path(path).read_text(encoding="utf-8"))

    def approach_of_edge(self, edge_id: str):
        """-> (approach, 'in'|'out') or (None, None) for junction-internal / unknown edges."""
        for n, a in self.approaches.items():
            if edge_id == a.in_edge:
                return n, "in"
            if edge_id == a.out_edge:
                return n, "out"
        return None, None


@dataclass
class ScenarioFiles:
    root: Path
    net: Path
    routes: Path
    additional: Path
    sumocfg: Path
    topology: Path
    gui_settings: Path


def _binary(name: str) -> str:
    import sumolib
    try:
        return sumolib.checkBinary(name)
    except Exception as e:  # pragma: no cover
        raise NetworkBuildError(f"SUMO binary '{name}' not found; run `python -m trafma doctor`") from e


def _run(cmd: list, what: str):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise NetworkBuildError(f"{what} failed (exit {p.returncode}):\n{p.stderr or p.stdout}")
    return p


def _write_sources(cfg: Config, d: Path):
    L = cfg.network.arm_length_m
    pos = {"N": (0, L), "E": (L, 0), "S": (0, -L), "W": (-L, 0)}
    nodes = ['<nodes>', f'  <node id="{TLS_ID}" x="0" y="0" type="traffic_light" tl="{TLS_ID}"/>']
    for a, (x, y) in pos.items():
        nodes.append(f'  <node id="{a}" x="{x}" y="{y}" type="priority"/>')
    nodes.append('</nodes>')
    edges = ['<edges>']
    for a in APPROACHES:
        for eid, f, t in ((in_edge(a), a, TLS_ID), (out_edge(a), TLS_ID, a)):
            edges.append(f'  <edge id="{eid}" from="{f}" to="{t}" numLanes="1" '
                         f'speed="{cfg.network.speed_limit_mps}" priority="2"/>')
    edges.append('</edges>')
    cons = ['<connections>']
    for o, _t, dst in movements():
        cons.append(f'  <connection from="{in_edge(o)}" to="{out_edge(dst)}" fromLane="0" toLane="0"/>')
    cons.append('</connections>')
    (d / "nodes.nod.xml").write_text("\n".join(nodes), encoding="utf-8")
    (d / "edges.edg.xml").write_text("\n".join(edges), encoding="utf-8")
    (d / "connections.con.xml").write_text("\n".join(cons), encoding="utf-8")


def _signal_program(cfg: Config, links_by_approach: dict, n_links: int):
    """Split phasing: [G(A) yellow(A) all-red] for each approach in clockwise order."""
    phases = []
    for a in APPROACHES:
        idx = set(links_by_approach[a])
        g = "".join("G" if i in idx else "r" for i in range(n_links))
        y = "".join("y" if i in idx else "r" for i in range(n_links))
        r = "r" * n_links
        phases.append({"duration": cfg.signal.green_s, "state": g, "label": f"{a}_green"})
        phases.append({"duration": cfg.signal.yellow_s, "state": y, "label": f"{a}_yellow"})
        phases.append({"duration": cfg.signal.all_red_s, "state": r, "label": f"{a}_allred"})
    return phases


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_network(cfg: Config, out_dir: Path):
    """Generate net + signal program + topology.json + GUI overlay. Returns (net_path, topo_path, add_path)."""
    import sumolib
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_sources(cfg, out_dir)
    nc = _binary("netconvert")
    base = [nc, "-n", str(out_dir / "nodes.nod.xml"), "-e", str(out_dir / "edges.edg.xml"),
            "-x", str(out_dir / "connections.con.xml"), "--no-turnarounds", "true",
            "--no-warnings", "true"]
    pass1 = out_dir / "net_pass1.net.xml"
    _run(base + ["-o", str(pass1)], "netconvert (pass 1)")

    net1 = sumolib.net.readNet(str(pass1), withPrograms=True)
    tls = net1.getTLS(TLS_ID)
    conns = tls.getConnections()          # (inLane, outLane, linkNo)
    n_links = max(c[2] for c in conns) + 1
    if len(conns) != 12:
        raise NetworkBuildError(f"expected 12 signalised links, got {len(conns)}")
    links_by_approach = {a: [] for a in APPROACHES}
    for in_lane, _out_lane, link in conns:
        e = in_lane.getEdge().getID()
        links_by_approach[e.split("_")[0]].append(link)
    phases = _signal_program(cfg, links_by_approach, n_links)

    tll = ['<tlLogics>', f'  <tlLogic id="{TLS_ID}" type="static" programID="0" offset="0">']
    for ph in phases:
        tll.append(f'    <phase duration="{ph["duration"]}" state="{ph["state"]}"/>')
    tll += ['  </tlLogic>', '</tlLogics>']
    (out_dir / "tllogic.tll.xml").write_text("\n".join(tll), encoding="utf-8")

    net_path = out_dir / "net.net.xml"
    _run([nc, "-s", str(pass1), "--tllogic-files", str(out_dir / "tllogic.tll.xml"),
          "--no-warnings", "true", "-o", str(net_path)], "netconvert (pass 2, signal program)")
    pass1.unlink()

    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    progs = net.getTLS(TLS_ID).getPrograms()
    prog = list(progs.values())[0]
    if [p.state for p in prog.getPhases()] != [p["state"] for p in phases]:
        raise NetworkBuildError("signal program in net does not match the generated schedule")

    # ---- topology
    aps = {}
    off = cfg.agents.rsu_lateral_offset_m
    for a in APPROACHES:
        ie, oe = net.getEdge(in_edge(a)), net.getEdge(out_edge(a))
        il, ol = ie.getLanes()[0], oe.getLanes()[0]
        sh = il.getShape()
        (x0, y0), (x1, y1) = sh[0], sh[-1]
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        rx, ry = uy, -ux          # right-hand side of travel direction
        rsu = (x1 + rx * off, y1 + ry * off)
        zone = min(cfg.agents.rsu_zone_m, il.getLength())
        hw = 2.4
        pa = (x1 - ux * zone + rx * hw, y1 - uy * zone + ry * hw)
        pb = (x1 + rx * hw, y1 + ry * hw)
        pc = (x1 - rx * hw, y1 - ry * hw)
        pd = (x1 - ux * zone - rx * hw, y1 - uy * zone - ry * hw)
        aps[a] = ApproachGeom(a, in_edge(a), out_edge(a), il.getID(), ol.getID(), il.getLength(),
                              ol.getLength(), (x1, y1), (x0, y0), (ux, uy), rsu,
                              sorted(links_by_approach[a]), [pa, pb, pc, pd])
        if abs(il.getLength() - il.getLength()) > 0 or il.getLength() < cfg.agents.rsu_zone_m:
            raise NetworkBuildError(f"approach {a} lane ({il.getLength():.1f} m) shorter than rsu_zone_m")
    cycle = sum(p["duration"] for p in phases)
    topo = Topology(TLS_ID, n_links, cycle, aps, cfg.network.arm_length_m, phases)
    topo_path = out_dir / "topology.json"
    topo_path.write_text(topo.to_json(), encoding="utf-8")

    # ---- GUI overlay (static): RSU markers + coverage zones. Harmless in headless runs.
    add = ['<additional>']
    for a, g in aps.items():
        shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in g.zone_polygon)
        add.append(f'  <poly id="rsu_zone_{a}" type="rsu_zone" color="0,200,255,90" fill="1" layer="5" shape="{shape}"/>')
        x, y = g.rsu_xy
        s = 2.5
        body = " ".join(f"{px:.2f},{py:.2f}" for px, py in
                        ((x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)))
        add.append(f'  <poly id="rsu_{a}" type="rsu" color="0,90,255" fill="1" layer="10" shape="{body}"/>')
        add.append(f'  <poi id="rsu_{a}_label" type="rsu" color="0,90,255" layer="11" x="{x:.2f}" y="{y:.2f}"/>')
    add.append('</additional>')
    add_path = out_dir / "visual.add.xml"
    add_path.write_text("\n".join(add), encoding="utf-8")

    gs = out_dir / "gui-settings.xml"
    gs.write_text('<viewsettings>\n  <scheme name="trafma">\n    <opengl dither="0" fps="0" drawBoundaries="0"/>\n'
                  '    <background backgroundColor="240,240,240" showGrid="0"/>\n'
                  '    <vehicles vehicleQuality="2" showBlinker="1" vehicleSize.minSize="2" '
                  'vehicleName_show="1" vehicleName_size="40"/>\n    <pois poiTextParam=""/>\n'
                  '    <polys polySize_exaggeration="1" polyName_show="1" polyName_size="50"/>\n'
                  '  </scheme>\n  <delay value="50"/>\n</viewsettings>', encoding="utf-8")
    return net_path, topo_path, add_path, gs


def write_sumocfg(cfg: Config, root: Path, net: Path, routes: Path, add: Path, gs: Path) -> Path:
    p = root / "scenario.sumocfg"
    q = lambda f: quoteattr(f.name)
    p.write_text(
        "<configuration>\n  <input>\n"
        f"    <net-file value={q(net)}/>\n    <route-files value={q(routes)}/>\n"
        f"    <additional-files value={q(add)}/>\n    <gui-settings-file value={q(gs)}/>\n  </input>\n"
        f"  <time>\n    <begin value=\"0\"/>\n    <end value=\"{cfg.run.duration_s}\"/>\n"
        f"    <step-length value=\"{cfg.run.step_s}\"/>\n  </time>\n"
        "  <processing>\n    <collision.action value=\"warn\"/>\n    <collision.check-junctions value=\"true\"/>\n"
        "    <time-to-teleport value=\"-1\"/>\n    <ignore-route-errors value=\"false\"/>\n  </processing>\n"
        "  <report>\n    <no-step-log value=\"true\"/>\n    <log value=\"sumo.log\"/>\n"
        "    <error-log value=\"sumo_errors.log\"/>\n  </report>\n"
        "  <output>\n    <tripinfo-output value=\"tripinfo.xml\"/>\n  </output>\n"
        "</configuration>\n", encoding="utf-8")
    return p
