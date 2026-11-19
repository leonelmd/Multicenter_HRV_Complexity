"""Figure 1 modality icons (run: python scripts/generate_figure1_icons.py figures/Figure1/elements): filled pictograms on a 96-unit grid.
One dark ink for objects, the cohort colour for the sensor, white for detail. Original work, CC0."""
import numpy as np
INK = "#3A3A3A"
def svg(body): return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="96" height="96">{body}</svg>'
def pl(xs, ys, col, sw): 
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in zip(xs, ys))
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
def ecg(x0, y0, w, h, beats):
    t = np.linspace(0, beats, 600); p = t % 1
    y = (0.15*np.exp(-((p-.2)/.04)**2) - .15*np.exp(-((p-.42)/.012)**2) + np.exp(-((p-.46)/.013)**2)
         - .3*np.exp(-((p-.5)/.013)**2) + .3*np.exp(-((p-.72)/.05)**2))
    return x0 + t/beats*w, y0 - y*h
def ppg(x0, y0, w, h, beats):
    t = np.linspace(0, beats, 600); p = t % 1
    y = np.where(p < .3, np.exp(-((p-.3)/.08)**2), np.exp(-((p-.3)/.17)**2)) + .22*np.exp(-((p-.62)/.07)**2)
    return x0 + t/beats*w, y0 - y*h
BUST = (f'<circle cx="48" cy="25" r="15" fill="{INK}"/>'
        f'<path d="M10 94 C10 64 26 50 48 50 C70 50 86 64 86 94 Z" fill="{INK}"/>')

def icon_ecg(col):
    b = BUST
    xs, ys = ecg(29, 82, 38, 14, 2); b += pl(xs, ys, "white", 3)
    b += "".join(f'<circle cx="{x}" cy="{y}" r="5" fill="{col}" stroke="white" stroke-width="2"/>' for x, y in [(23, 70), (73, 70)])
    return svg(b)

def icon_mri(col):
    b = (f'<rect x="8" y="10" width="62" height="64" rx="14" fill="{INK}"/>'
         f'<circle cx="39" cy="42" r="19" fill="white"/>'
         f'<rect x="22" y="52" width="70" height="7" rx="3.5" fill="{INK}" stroke="white" stroke-width="2"/>'
         f'<rect x="20" y="74" width="5" height="16" rx="2" fill="{INK}"/><rect x="52" y="74" width="5" height="16" rx="2" fill="{INK}"/>')
    b += f'<circle cx="78" cy="78" r="15" fill="{col}" stroke="white" stroke-width="3"/>'
    xs, ys = ppg(69, 84, 18, 11, 2); b += pl(xs, ys, "white", 2.6)
    return svg(b)

def icon_strap(col):
    b = BUST
    b += f'<path d="M12.5 74 C30 80 66 80 83.5 74 L85 85 C66 91 30 91 11 85 Z" fill="{col}"/>'
    b += f'<rect x="37" y="73" width="22" height="15" rx="4.5" fill="white"/>'
    xs, ys = ecg(40, 84, 16, 6, 1); b += pl(xs, ys, INK, 1.8)
    # 24-h clock badge
    b += (f'<circle cx="82" cy="15" r="12" fill="white" stroke="{INK}" stroke-width="3"/>'
          f'<path d="M82 8.5 L82 15 L87 18" fill="none" stroke="{INK}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    return svg(b)

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for n, f, c in [("icon_ecg_rest", icon_ecg, "#E69F00"), ("icon_ppg_fmri", icon_mri, "#7F7F00"), ("icon_chest_strap_24h", icon_strap, "#17BECF")]:
        open(f"{out}/{n}.svg", "w").write(f(c)); open(f"{out}/{n}_neutral.svg", "w").write(f("#9E9E9E"))
    print("ok")
