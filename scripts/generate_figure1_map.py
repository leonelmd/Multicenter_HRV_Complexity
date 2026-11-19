#!/usr/bin/env python3
"""World map with the three cohort sites, for Figure 1 (run: python scripts/generate_figure1_map.py).

Land polygons: Natural Earth 1:110m land (public domain), Equal Earth projection.
Pins are vector teardrops; labels are live text. Antarctica omitted.
Writes world_map_sites.svg (with labels) and world_map_pins.svg (pins only).
"""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon
import os
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
src = os.path.join(BASE, "data", "natural_earth", "ne_110m_land.geojson")
outdir = os.path.join(BASE, "figures", "Figure1", "elements")
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
def equal_earth(lon, lat):
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi)); t2 = th * th; t6 = t2 ** 3
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * t6 * t2 + 7 * A3 * t6 + 3 * A2 * t2 + A1))
    y = th * (A4 * t6 * t2 + A3 * t6 + A2 * t2 + A1)
    return x, y
LAND, INK = "#CFCFCF", "#2B2B2B"
# One neutral pin colour: sites are named by their labels, and no hue is used that could
# be read as a group (Control blue / PD red) colour.
PIN = "#3A3A3A"
SITES = [  # name, lon, lat, colour, label, label position (lon, lat), alignment
    ("Santiago", -70.67, -33.45, PIN, "Santiago, Chile\n15-min ECG\n42 C / 27 PD", (-118, -30), "right"),
    ("Bilbao", -2.93, 43.26, PIN, "Bilbao, Spain\n7.40-min PPG\n21 C / 31 PD", (-30, 22), "center"),
    ("Nagoya", 136.91, 35.18, PIN, "Nagoya, Japan\n24-h chest strap\n21 DC / 26 PD", (168, 12), "center"),
]
def pin(ax, x, y, col, h=0.15):
    # teardrop: circle of radius r centred h-r above the tip, tangent lines to the tip
    r = h * 0.38; cy = y + h - r
    a = np.linspace(np.radians(-35), np.radians(215), 60)
    pts = [(x, y)] + [(x + r * np.cos(t), cy + r * np.sin(t)) for t in a] + [(x, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=col, ec="white", lw=0.8, zorder=5))
    ax.add_patch(plt.Circle((x, cy), r * 0.42, fc="white", ec="none", zorder=6))
def draw(labels, name):
    g = json.load(open(src))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for f in g["features"]:
        geom = f["geometry"]; polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            verts, codes = [], []
            for ring in poly:
                r = np.asarray(ring)
                if r[:, 1].max() < -58: break      # Antarctica
                x, y = equal_earth(r[:, 0], r[:, 1])
                verts += list(zip(x, y)); codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
            if verts:
                ax.add_patch(PathPatch(Path(verts, codes), fc=LAND, ec="white", lw=0.3))
    for nm, lon, lat, col, lab, (llon, llat), ha in SITES:
        x, y = equal_earth(lon, lat); pin(ax, x, y, col)
        if labels:
            tx, ty = equal_earth(llon, llat)
            ax.text(tx, ty, lab, ha=ha, va="center", fontsize=8, color=INK, linespacing=1.25, zorder=7)
            ax.plot([x, tx + (0.03 if ha == "right" else 0)], [y, ty + (0 if ha == "right" else 0.21)],
                    color="#7A7A7A", lw=0.7, zorder=4)
    X, Y = equal_earth(np.array([-180, 180]), np.array([0, 0]))
    ax.set_xlim(X[0] * 1.01, X[1] * 1.01); ax.set_ylim(equal_earth(0, -58)[1], equal_earth(0, 84)[1])
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(f"{outdir}/{name}", transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
draw(True, "world_map_sites.svg"); draw(False, "world_map_pins.svg"); print("ok")
