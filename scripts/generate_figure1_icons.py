"""Original line icons for the three acquisition modalities (Figure 1). Free to use (CC0).
Flat style, 3.5-px dark strokes, round caps; the sensor is filled with the cohort colour.
Run: python make_icons.py <outdir>"""
import numpy as np
INK, SKIN = "#2B2B2B", "#F2F2F2"
def svg(w, body): return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 120" width="{w}" height="120">{body}</svg>'
def line(d, sw=3.5, col=INK): return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
def poly(xs, ys, sw, col): return line("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys)), sw, col)
def torso(cx=60, s=1.0):
    def X(x): return cx + (x - 60) * s
    def Y(y): return 120 - (120 - y) * s
    head = f'<circle cx="{X(60)}" cy="{Y(24)}" r="{13*s}" fill="{SKIN}" stroke="{INK}" stroke-width="3.5"/>'
    body = (f'<path d="M{X(18)} {Y(118)} L{X(18)} {Y(74)} Q{X(18)} {Y(50)} {X(42)} {Y(46)} L{X(52)} {Y(44)} '
            f'Q{X(60)} {Y(52)} {X(68)} {Y(44)} L{X(78)} {Y(46)} Q{X(102)} {Y(50)} {X(102)} {Y(74)} L{X(102)} {Y(118)}" '
            f'fill="{SKIN}" stroke="{INK}" stroke-width="3.5" stroke-linejoin="round"/>')
    return head + body
def ecg_wave(x0, y0, w, h, beats=2):
    t = np.linspace(0, beats, 400); ph = t % 1
    y = (0.12*np.exp(-((ph-0.18)/0.035)**2) - 0.12*np.exp(-((ph-0.36)/0.012)**2) + 1.0*np.exp(-((ph-0.40)/0.012)**2)
         - 0.25*np.exp(-((ph-0.44)/0.013)**2) + 0.25*np.exp(-((ph-0.68)/0.05)**2))
    return x0 + t / beats * w, y0 - y * h
def ppg_wave(x0, y0, w, h, beats=3):
    t = np.linspace(0, beats, 400); ph = t % 1
    up = np.exp(-((ph-0.25)/0.07)**2); down = np.exp(-((ph-0.25)/0.16)**2)
    y = np.where(ph < 0.25, up, down) + 0.25*np.exp(-((ph-0.58)/0.08)**2)
    return x0 + t / beats * w, y0 - y * h

def icon_ecg(col):
    b = torso(cx=42, s=0.9)
    b += line("M34 68 C34 92 62 96 84 94", 2.5) + line("M54 68 C54 84 70 86 84 86", 2.5)
    b += "".join(f'<circle cx="{x}" cy="68" r="5.5" fill="{col}" stroke="{INK}" stroke-width="2.5"/>' for x in (34, 54))
    b += f'<rect x="82" y="72" width="54" height="34" rx="6" fill="white" stroke="{INK}" stroke-width="3.5"/>'
    xs, ys = ecg_wave(87, 96, 44, 16); b += poly(xs, ys, 2.5, INK)
    return svg(140, b)

def icon_ppg_mri(col):
    b = (f'<rect x="6" y="10" width="78" height="70" rx="16" fill="{SKIN}" stroke="{INK}" stroke-width="3.5"/>'
         f'<circle cx="45" cy="45" r="21" fill="white" stroke="{INK}" stroke-width="3.5"/>'
         + line("M2 74 L88 74") + line("M14 74 L14 90 M76 74 L76 90"))
    # finger pointing up, with nail, and a clip sensor across it
    b += (f'<path d="M98 118 L98 70 Q98 56 110 56 Q122 56 122 70 L122 118" fill="{SKIN}" stroke="{INK}" stroke-width="3.5" stroke-linejoin="round"/>'
          f'<path d="M104 66 Q104 61 110 61 Q116 61 116 66 L116 70 L104 70 Z" fill="white" stroke="{INK}" stroke-width="2"/>'
          f'<rect x="92" y="80" width="36" height="18" rx="6" fill="{col}" stroke="{INK}" stroke-width="3"/>'
          + line("M128 89 Q136 89 137 80", 2.5))
    xs, ys = ppg_wave(92, 40, 44, 14); b += poly(xs, ys, 2.8, col)
    return svg(140, b)

def icon_strap(col):
    b = torso(cx=52, s=1.0)
    b += (f'<path d="M10 80 Q52 90 94 80 L94 93 Q52 103 10 93 Z" fill="{col}" stroke="{INK}" stroke-width="3"/>'
          f'<rect x="39" y="82" width="26" height="16" rx="5" fill="{INK}"/>')
    xs, ys = ecg_wave(43, 93, 18, 7, beats=1); b += poly(xs, ys, 1.6, "white")
    b += (f'<circle cx="118" cy="26" r="18" fill="white" stroke="{INK}" stroke-width="3.5"/>'
          f'<text x="118" y="31" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" fill="{INK}">24 h</text>')
    return svg(140, b)

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, fn, c in [("icon_ecg_rest", icon_ecg, "#E69F00"), ("icon_ppg_fmri", icon_ppg_mri, "#7F7F00"),
                        ("icon_chest_strap_24h", icon_strap, "#17BECF")]:
        open(f"{out}/{name}.svg", "w").write(fn(c))
        open(f"{out}/{name}_neutral.svg", "w").write(fn("#BDBDBD"))
    print("ok")
