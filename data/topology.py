"""IEEE 34-bus radial test feeder.

Builds two coordinated artifacts from a single source of truth:
  1. A NetworkX graph (for the GNN, plotting, and spatial features).
  2. An OpenDSS deck (for physics-validated power flow).

Topology, line lengths, configurations, and spot loads follow the IEEE PES
distribution test feeder working group's 34-bus case (24.9 kV primary, with
a 4.16 kV lateral off bus 832 via the in-line XFM-1 transformer to bus 888).
Coordinates are placed on a synthetic Arizona-like footprint (Phoenix-ish lat/lon)
so the dashboard map renders sensibly without depending on external GIS data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


# --- Line configurations (R, X in ohm/mile, ampacity in A) -------------------
# Simplified from the IEEE 34-bus configurations (300, 301, 302, 303, 304).
LINE_CONFIGS: Dict[str, Dict[str, float]] = {
    "300": {"r_per_mi": 1.3368, "x_per_mi": 1.3343, "amp": 530.0},
    "301": {"r_per_mi": 1.9300, "x_per_mi": 1.4115, "amp": 230.0},
    "302": {"r_per_mi": 2.7995, "x_per_mi": 1.4855, "amp": 230.0},
    "303": {"r_per_mi": 2.7995, "x_per_mi": 1.4855, "amp": 230.0},
    "304": {"r_per_mi": 1.9217, "x_per_mi": 1.4212, "amp": 230.0},
}

# (from_bus, to_bus, length_ft, config)
LINES: List[Tuple[str, str, float, str]] = [
    ("800", "802", 2580.0, "300"),
    ("802", "806", 1730.0, "300"),
    ("806", "808", 32230.0, "300"),
    ("808", "810", 5804.0, "303"),
    ("808", "812", 37500.0, "300"),
    ("812", "814", 29730.0, "300"),
    ("814", "850", 10.0, "301"),     # downstream of Reg1 (handled in DSS)
    ("850", "816", 310.0, "301"),
    ("816", "818", 1710.0, "302"),
    ("816", "824", 10210.0, "301"),
    ("818", "820", 48150.0, "302"),
    ("820", "822", 13740.0, "302"),
    ("824", "826", 3030.0, "303"),
    ("824", "828", 840.0, "301"),
    ("828", "830", 20440.0, "301"),
    ("830", "854", 520.0, "301"),
    ("854", "852", 36830.0, "301"),
    ("852", "832", 10.0, "301"),     # downstream of Reg2 (handled in DSS)
    ("832", "858", 4900.0, "301"),
    ("832", "888", 0.0, "xfm"),       # in-line 24.9/4.16 kV transformer
    ("858", "864", 1620.0, "302"),
    ("858", "834", 5830.0, "301"),
    ("834", "860", 2020.0, "301"),
    ("834", "842", 280.0, "301"),
    ("836", "840", 860.0, "301"),
    ("836", "862", 280.0, "304"),
    ("862", "838", 4900.0, "304"),
    ("842", "844", 1350.0, "301"),
    ("844", "846", 3640.0, "301"),
    ("846", "848", 530.0, "301"),
    ("860", "836", 2680.0, "301"),
    ("888", "890", 10560.0, "300"),
    ("854", "856", 23330.0, "303"),
]

# Spot loads from the IEEE 34 case (kW, kVAr at peak / nominal). Values reflect
# the published case scaled to a single equivalent three-phase load per bus.
SPOT_LOADS_KW: Dict[str, float] = {
    "806": 30.0, "810": 16.0, "820": 135.0, "822": 21.0, "824": 5.0,
    "826": 40.0, "828": 7.0, "830": 27.0, "856": 4.0, "858": 9.0,
    "864": 2.0, "834": 16.0, "860": 23.0, "836": 27.0, "840": 27.0,
    "862": 28.0, "844": 432.0, "846": 25.0, "848": 60.0, "890": 450.0,
}

# Synthetic geographic placement on a Phoenix-area footprint.
# We use a deterministic radial spiral so the network is visually
# interpretable on a map without claiming to be a real APS feeder.
def _synth_coords() -> Dict[str, Tuple[float, float]]:
    # Approximate center near downtown Phoenix
    lat0, lon0 = 33.4484, -112.0740
    # Place buses along a deterministic walk based on graph BFS distance
    g = nx.Graph()
    for u, v, *_ in LINES:
        g.add_edge(u, v)
    # BFS from substation 800
    depths = nx.single_source_shortest_path_length(g, "800")
    coords: Dict[str, Tuple[float, float]] = {}
    # Each bus gets a position on a spiral so the map is readable.
    sorted_buses = sorted(g.nodes(), key=lambda b: (depths.get(b, 99), b))
    for i, bus in enumerate(sorted_buses):
        d = depths.get(bus, 0)
        angle = 0.55 * i  # radians
        radius = 0.004 + 0.0035 * d
        lat = lat0 + radius * math.cos(angle)
        lon = lon0 + radius * math.sin(angle) * 1.2  # mild aspect correction
        coords[bus] = (lat, lon)
    return coords


COORDS: Dict[str, Tuple[float, float]] = _synth_coords()


@dataclass
class FeederGraph:
    g: nx.Graph
    bus_index: Dict[str, int]
    index_bus: List[str]


def build_graph() -> FeederGraph:
    g = nx.Graph()
    # nodes
    for u, v, length_ft, cfg in LINES:
        for bus in (u, v):
            if bus not in g:
                lat, lon = COORDS[bus]
                g.add_node(
                    bus,
                    lat=lat,
                    lon=lon,
                    base_kv=4.16 if bus in {"888", "890"} else 24.9,
                    spot_kw=SPOT_LOADS_KW.get(bus, 0.0),
                )
        if cfg == "xfm":
            # In-line transformer modeled as zero-length link with low Z.
            g.add_edge(u, v, length_mi=0.0, r=0.01, x=0.05, amp=400.0,
                       config=cfg, kind="transformer")
        else:
            cfg_d = LINE_CONFIGS[cfg]
            length_mi = length_ft / 5280.0
            g.add_edge(
                u, v,
                length_mi=length_mi,
                r=cfg_d["r_per_mi"] * length_mi,
                x=cfg_d["x_per_mi"] * length_mi,
                amp=cfg_d["amp"],
                config=cfg,
                kind="line",
            )

    bus_list = sorted(g.nodes())
    bus_index = {b: i for i, b in enumerate(bus_list)}
    return FeederGraph(g=g, bus_index=bus_index, index_bus=bus_list)


# --- OpenDSS deck generator --------------------------------------------------
DSS_LINECODES = """
New Linecode.LC_300 nphases=3 R1=0.253 X1=0.252 C1=0 R0=0.253 X0=0.252 C0=0 Units=mi normamps=530
New Linecode.LC_301 nphases=3 R1=0.366 X1=0.267 C1=0 R0=0.366 X0=0.267 C0=0 Units=mi normamps=230
New Linecode.LC_302 nphases=3 R1=0.530 X1=0.281 C1=0 R0=0.530 X0=0.281 C0=0 Units=mi normamps=230
New Linecode.LC_303 nphases=3 R1=0.530 X1=0.281 C1=0 R0=0.530 X0=0.281 C0=0 Units=mi normamps=230
New Linecode.LC_304 nphases=3 R1=0.364 X1=0.269 C1=0 R0=0.364 X0=0.269 C0=0 Units=mi normamps=230
""".strip()


def write_opendss_deck(out_dir: Path, fg: FeederGraph, load_mults: Dict[str, float] | None = None) -> Path:
    """Write a self-contained OpenDSS deck for the feeder.

    `load_mults` lets you scale per-bus loads (defaults to 1.0). The deck uses
    constant-power loads at each bus that has a spot load, plus a slack source
    at the substation (bus 800) at 24.9 kV nominal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "ieee34.dss"

    lines = []
    lines.append("Clear")
    # Source set above 1.0 pu to mimic the substation LTC found upstream of
    # bus 800 in the published IEEE 34 case (helps the long radial reach
    # acceptable voltage even before the in-feeder regulators).
    lines.append("New Circuit.IEEE34 basekv=69.0 pu=1.03 phases=3 bus1=sourcebus")
    # Substation transformer 69 -> 24.9 kV at bus 800
    lines.append(
        "New Transformer.Sub Phases=3 Windings=2 Xhl=8 "
        "wdg=1 bus=sourcebus conn=delta kv=69 kva=2500 %r=1 "
        "wdg=2 bus=800 conn=wye kv=24.9 kva=2500 %r=1"
    )
    lines.append(DSS_LINECODES)

    # Buses where a voltage regulator is spliced in.
    # The regulator's Wdg-1 sits at the listed bus and Wdg-2 sits at "<bus>R";
    # the downstream line therefore starts from "<bus>R", not from the bus itself.
    REG_BUSES = {"814": "814R", "852": "852R"}

    # Lines
    for u, v, length_ft, cfg in LINES:
        u_dss = REG_BUSES.get(u, u)
        if cfg == "xfm":
            lines.append(
                f"New Transformer.XFM_{u}_{v} Phases=3 Windings=2 Xhl=4.08 "
                f"wdg=1 bus={u_dss} conn=wye kv=24.9 kva=500 %r=0.95 "
                f"wdg=2 bus={v} conn=wye kv=4.16 kva=500 %r=0.95"
            )
        else:
            length_mi = length_ft / 5280.0
            lines.append(
                f"New Line.L_{u}_{v} Bus1={u_dss} Bus2={v} "
                f"LineCode=LC_{cfg} Length={length_mi:.4f} Units=mi"
            )

    # Loads (constant PQ, pf 0.9)
    pf = 0.9
    tan_phi = math.tan(math.acos(pf))
    for bus, kw in SPOT_LOADS_KW.items():
        mult = (load_mults or {}).get(bus, 1.0)
        kw_eff = max(0.001, kw * mult)
        kvar_eff = kw_eff * tan_phi
        kv = 4.16 if bus in {"888", "890"} else 24.9
        lines.append(
            f"New Load.LD_{bus} Bus1={bus} Phases=3 Conn=wye Model=1 "
            f"kV={kv} kW={kw_eff:.3f} kvar={kvar_eff:.3f} Vminpu=0.85 Vmaxpu=1.20"
        )

    # Voltage regulators per the IEEE 34 case (modeled as ideal autotransformers
    # with discrete tap setting; published taps boost ~2.5% per phase). We
    # implement them as zero-impedance transformers + a regcontrol.
    lines.append(
        "New Transformer.Reg1 phases=3 windings=2 buses=(814 814R) "
        "conns=(wye wye) kvs=(24.9 24.9) kvas=(2500 2500) XHL=0.01 %loadloss=0.01"
    )
    lines.append("New RegControl.Reg1 transformer=Reg1 winding=2 vreg=125 band=2 ptratio=120")
    lines.append(
        "New Transformer.Reg2 phases=3 windings=2 buses=(852 852R) "
        "conns=(wye wye) kvs=(24.9 24.9) kvas=(2500 2500) XHL=0.01 %loadloss=0.01"
    )
    lines.append("New RegControl.Reg2 transformer=Reg2 winding=2 vreg=122 band=2 ptratio=120")

    lines.append("Set VoltageBases=[69.0 24.9 4.16]")
    lines.append("CalcVoltageBases")
    lines.append("Set ControlMode=Static")
    lines.append("Solve mode=snapshot")

    master.write_text("\n".join(lines) + "\n")
    return master


# --- Existing-asset registry -------------------------------------------------
# Maps each bus to a short, planner-readable description of nearby assets the
# action engine should reference. This is what an APS planner reaches for
# *first* before recommending a greenfield install — "do we already have a
# regulator nearby that we can re-set?"
EXISTING_ASSETS: Dict[str, Dict] = {
    "Reg1":   {"bus": "814",  "kind": "voltage_regulator",  "label": "Reg1 @ Bus 814 (32-step, ±10% boost)"},
    "Reg2":   {"bus": "852",  "kind": "voltage_regulator",  "label": "Reg2 @ Bus 852 (32-step, ±10% boost)"},
    "XFM_1":  {"bus": "832",  "kind": "transformer",        "label": "In-line 24.9/4.16 kV transformer 832→888"},
    "Sub":    {"bus": "800",  "kind": "substation",         "label": "Substation transformer 69/24.9 kV @ Bus 800"},
}


def nearby_assets(bus: str, max_hops: int = 4) -> List[Dict]:
    """Return existing assets within `max_hops` graph hops of `bus`.

    A planner sizing a battery at Bus 890 cares that there's already a 24.9/4.16 kV
    in-line transformer two hops upstream — that constrains the ride-through
    time and may mean the right answer is a dual-tap on the transformer instead
    of a battery. The list is sorted nearest-first.
    """
    g = nx.Graph()
    for u, v, *_ in LINES:
        g.add_edge(u, v)
    if bus not in g.nodes:
        return []
    try:
        dists = nx.single_source_shortest_path_length(g, bus, cutoff=max_hops)
    except Exception:
        return []
    out: List[Dict] = []
    for asset_name, info in EXISTING_ASSETS.items():
        d = dists.get(info["bus"])
        if d is None:
            continue
        out.append({**info, "name": asset_name, "hops_from_bus": int(d)})
    out.sort(key=lambda x: x["hops_from_bus"])
    return out


def edge_index_tensor(fg: FeederGraph):
    """Return (edge_index, edge_attr) for PyG.

    edge_index: shape [2, 2E] (undirected -> both directions)
    edge_attr:  shape [2E, 3] = [length_mi, r, x] normalized
    """
    import torch
    src, dst, attrs = [], [], []
    for u, v, data in fg.g.edges(data=True):
        i, j = fg.bus_index[u], fg.bus_index[v]
        for a, b in ((i, j), (j, i)):
            src.append(a)
            dst.append(b)
            attrs.append([data["length_mi"], data["r"], data["x"]])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    # normalize each column
    if edge_attr.numel() > 0:
        m = edge_attr.amax(dim=0).clamp(min=1e-6)
        edge_attr = edge_attr / m
    return edge_index, edge_attr


if __name__ == "__main__":
    fg = build_graph()
    print(f"Buses: {len(fg.g.nodes())}  Edges: {len(fg.g.edges())}")
    deck = write_opendss_deck(Path(__file__).parent / "opendss", fg)
    print(f"Wrote OpenDSS deck: {deck}")
