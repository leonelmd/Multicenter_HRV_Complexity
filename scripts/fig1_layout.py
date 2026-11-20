"""Fig. 1 layout mock-up, assembled from the element SVGs (nested, editable). Units: pt."""
import re, os
EL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "Fig1_overview", "elements"); INK = "#2B2B2B"; GREY = "#8C8C8C"; MUTE = "#555555"
COL = {"santiago": "#E69F00", "bilbao": "#7F7F00", "nagoya": "#17BECF"}
def place(name, x, y, scale=None, w=None):
    s = open(os.path.join(EL, name)).read()
    s = re.sub(r"<\?xml[^>]*>|<!DOCTYPE[^>]*>|<metadata>.*?</metadata>", "", s, flags=re.S)
    m = re.search(r"<svg\b[^>]*>", s); tag = m.group(0)
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', tag).group(1).split()]
    nw = re.search(r'width="([0-9.]+)', tag); nh = re.search(r'height="([0-9.]+)', tag)
    nw, nh = (float(nw.group(1)), float(nh.group(1))) if nw else (vb[2], vb[3])
    if w: scale = w / nw
    if scale is None: scale = 1.0
    W, H = nw * scale, nh * scale
    t = re.sub(r'\s(width|height|x|y)="[^"]*"', "", tag)
    return s[:m.start()] + t.replace("<svg", f'<svg x="{x}" y="{y}" width="{W:.1f}" height="{H:.1f}" overflow="visible"', 1) + s[m.end():], W, H
def T(x, y, t, size=8.5, weight="normal", col=INK, anchor="start", style="normal"):
    return (f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" fill="{col}" text-anchor="{anchor}">{t}</text>')
def A(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GREY}" stroke-width="1.4" marker-end="url(#ah)"/>'
W, H = 520, 722
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}pt" height="{H}pt">',
     f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{GREY}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']
# ── a ──
o += [T(4, 14, "a", 12, "bold"), T(18, 14, "Three cohorts, each analysed on its own", 10, "bold")]
s, w, h = place("world_map_pins.svg", 2, 30, w=300); o.append(s)
cards = [("icon_ecg_rest.svg", "santiago", "Santiago, Chile", "Resting ECG · 15 min", "42 controls / 27 PD"),
         ("icon_ppg_fmri.svg", "bilbao", "Bilbao, Spain", "Finger PPG during fMRI · 7.4 min", "21 controls / 31 PD"),
         ("icon_chest_strap_24h.svg", "nagoya", "Nagoya, Japan", "Chest strap · 24 h (18–22 h analysed)", "21 disease controls / 26 PD")]
for k, (ic, key, n1, n2, n3) in enumerate(cards):
    y0 = 24 + k * 62
    o.append(f'<rect x="308" y="{y0}" width="208" height="56" rx="7" fill="#F7F7F7" stroke="{COL[key]}" stroke-width="1.6"/>')
    s, _, _ = place(ic, 312, y0 + 4, w=48); o.append(s)
    o += [T(366, y0 + 17, n1, 9, "bold"), T(366, y0 + 31, n2, 7.8), T(366, y0 + 45, n3, 7.8, col=MUTE)]
o += [T(8, 182, "Sub-aim 1 · acquisition modality: R wave of the ECG, finger pulse wave, chest strap", 7.5, col=MUTE, style="italic"),
      T(8, 194, "Sub-aim 2 · recording length: 7.4 min to 24 h", 7.5, col=MUTE, style="italic")]
o.append(f'<line x1="4" y1="208" x2="{W-4}" y2="208" stroke="#DDDDDD" stroke-width="1"/>')
# ── b ──
o += [T(4, 226, "b", 12, "bold"), T(18, 226, "Two ways of summarising the same RR-interval series", 10, "bold")]
o.append(f'<rect x="120" y="252" width="40" height="40" rx="6" fill="none" stroke="{GREY}" stroke-width="1" stroke-dasharray="4 3"/>')
o.append(T(140, 275, "heart", 7, col=GREY, anchor="middle"))
o.append(A(163, 272, 180, 272))
o.append(T(186, 246, "RR-interval series", 9, "bold"))
s, w, h = place("rr_input.svg", 182, 248); o.append(s)
o += [A(250, 326, 120, 352), A(300, 326, 360, 352)]
# A
o += [T(10, 366, "A · Conventional HRV", 9, "bold"), T(10, 378, "how much the intervals vary (SDNN, RMSSD)", 7.5, col=MUTE)]
s, w, h = place("A_variability.svg", 30, 384, scale=0.85); o.append(s)
# B
o += [T(214, 366, "B · Multiscale entropy", 9, "bold"), T(214, 378, "how the fluctuations are organised across time scales", 7.5, col=MUTE)]
s, w, h = place("B_coarse_graining.svg", 222, 384, scale=0.75); o.append(s)
o.append(A(352, 434, 372, 434))
s, w, h = place("B_entropy_curve.svg", 380, 384, scale=0.85); o.append(s)
o.append(f'<line x1="4" y1="494" x2="{W-4}" y2="494" stroke="#EEEEEE" stroke-width="1"/>')
# outcome
o += [T(10, 514, "Main aim · within each cohort, is complexity lower in PD?", 9, "bold")]
s, w, h = place("result_three_cohorts.svg", 80, 520); o.append(s)
o.append("</svg>")
open(os.path.join(os.path.dirname(EL), "figure1_layout.svg"), "w").write("\n".join(o)); print("ok", h)
