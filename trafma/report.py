"""Optional figures (needs matplotlib): layout with RSUs/zones and queue/signal time series."""
from __future__ import annotations

import json
from pathlib import Path

from . import APPROACHES
from .gate import load_manifest, _jl
from .netgen import Topology


def make_report(run_dir, snapshot_t=None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    run_dir = Path(run_dir)
    m = load_manifest(run_dir)
    topo = Topology.load(run_dir / "scenario" / "topology.json")
    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)
    paths = []

    # ---- layout + vehicles at one instant
    trace = list(_jl(run_dir / "trace.jsonl"))
    if snapshot_t is None and trace:
        snapshot_t = max(trace, key=lambda r: len(r["v"]))["t"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cols = {"N": "tab:blue", "E": "tab:orange", "S": "tab:green", "W": "tab:red"}
    for a, g in topo.approaches.items():
        (x0, y0), (x1, y1) = g.start_xy, g.stop_xy
        ax.plot([x0, x1], [y0, y1], color="0.6", lw=6, solid_capstyle="butt", zorder=1)
        ax.add_patch(Polygon(g.zone_polygon, closed=True, color="seagreen", alpha=0.25, zorder=2))
        ax.plot(*g.rsu_xy, "s", color="navy", ms=6, zorder=5)
        lx = g.rsu_xy[0] - g.dir_xy[0] * 40 + (-g.dir_xy[1]) * 22
        ly = g.rsu_xy[1] - g.dir_xy[1] * 40 + (g.dir_xy[0]) * 22
        ax.annotate(f"RSU-{a}", g.rsu_xy, xytext=(lx, ly), color="navy", weight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color="navy", lw=0.8))
    cx = sum(g.stop_xy[0] for g in topo.approaches.values()) / 4
    cy = sum(g.stop_xy[1] for g in topo.approaches.values()) / 4
    ax.plot(cx, cy, "k+", ms=14)
    if trace:
        row = next(r for r in trace if r["t"] == snapshot_t)
        # placement via the owning approach geometry (edge id -> lane axis)
        for vid, edge, pos, spd in row["v"]:
            if edge.endswith("_in"):
                g = topo.approaches[edge[0]]
                s = pos
                x = g.start_xy[0] + g.dir_xy[0] * s
                y = g.start_xy[1] + g.dir_xy[1] * s
            elif edge.endswith("_out"):
                g = topo.approaches[edge[0]]       # outgoing lane runs away from the stop line
                x = g.stop_xy[0] - g.dir_xy[0] * pos
                y = g.stop_xy[1] - g.dir_xy[1] * pos
            else:
                x, y = cx, cy
            ax.plot(x, y, "o", color="goldenrod" if spd > 0.1 else "crimson", ms=4, zorder=4)
    ax.set_aspect("equal")
    ax.set_title(f"TRAF-MA intersection: 4 RSUs, green = RSU sensing zone ({m['config']['agents']['rsu_zone_m']:.0f} m)\n"
                 f"vehicles at t={snapshot_t} s (red = halted)")
    p = out_dir / "layout.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ---- queue time series with signal shading
    truth = list(_jl(run_dir / "evaluator" / "truth.jsonl"))
    t = [r["t"] for r in truth]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, a in zip(axes, APPROACHES):
        rsu = {r["t"]: r["view"] for r in _jl(run_dir / "agents" / f"rsu_{a}.jsonl")}
        tq = [r["approaches"][a]["zone_queue_m"] for r in truth]
        oq = [rsu[x]["queue_length_m"] for x in t]
        sig = [rsu[x]["signal_state"] for x in t]
        for i, s in enumerate(sig):
            if s == "G":
                ax.axvspan(t[i] - 0.5, t[i] + 0.5, color="limegreen", alpha=0.15, lw=0)
        ax.plot(t, tq, label="evaluator truth", lw=2)
        ax.plot(t, oq, "--", label="RSU observed", lw=1.2)
        ax.set_ylabel(f"{a} queue (m)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("simulation time (s); green shading = approach has green")
    fig.suptitle(f"Zone queue per approach - run {m['run_id']}")
    p = out_dir / "queues.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths
