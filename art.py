"""Vector art for the certificate: guilloche watermark, notary seal, corner
ornaments and crest. Everything is generated as inline SVG so the certificate
prints crisply with no external image dependencies."""
import math


def _rose(cx, cy, R, r, d, turns, pts, stroke, sw, op):
    a = []
    steps = turns * pts
    for i in range(steps + 1):
        t = 2 * math.pi * turns * i / steps
        x = cx + (R - r) * math.cos(t) + d * math.cos((R - r) / r * t)
        y = cy + (R - r) * math.sin(t) - d * math.sin((R - r) / r * t)
        a.append(f"{x:.1f},{y:.1f}")
    return (f'<polyline points="{" ".join(a)}" fill="none" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{op}"/>')


def bg_svg():
    p = ['<svg viewBox="0 0 900 900" width="100%" height="100%" '
         'xmlns="http://www.w3.org/2000/svg">']
    for R, r, d in [(420, 29, 120), (380, 23, 140), (340, 19, 150), (300, 31, 110)]:
        p.append(_rose(450, 450, R, r, d, r, 900, "#c9b98f", 0.5, 0.16))
    p.append("</svg>")
    return "".join(p)


def seal_svg(academy):
    cx = cy = 260
    p = ['<svg viewBox="0 0 520 520" width="100%" height="100%" '
         'xmlns="http://www.w3.org/2000/svg">']
    sc, Ro, rl = 48, 250, 13
    d = "M"
    for i in range(sc):
        a0 = 2 * math.pi * i / sc
        a1 = 2 * math.pi * (i + 1) / sc
        am = (a0 + a1) / 2
        x0, y0 = cx + Ro * math.cos(a0), cy + Ro * math.sin(a0)
        xm, ym = cx + (Ro + rl) * math.cos(am), cy + (Ro + rl) * math.sin(am)
        x1, y1 = cx + Ro * math.cos(a1), cy + Ro * math.sin(a1)
        if i == 0:
            d += f"{x0:.1f},{y0:.1f} "
        d += f"Q{xm:.1f},{ym:.1f} {x1:.1f},{y1:.1f} "
    d += "Z"
    p.append(f'<path d="{d}" fill="none" stroke="#a9822f" stroke-width="2.2"/>')
    for rr, sw, op in [(238, 1.4, .9), (232, .7, .7), (150, .9, .8), (144, .6, .6)]:
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{rr}" fill="none" '
                 f'stroke="#a9822f" stroke-width="{sw}" opacity="{op}"/>')
    p.append(_rose(cx, cy, 224, 14, 44, 14, 700, "#b8923f", 0.8, 0.85))
    p.append(_rose(cx, cy, 206, 12, 40, 12, 700, "#a9822f", 0.7, 0.8))
    p.append(_rose(cx, cy, 188, 16, 34, 16, 700, "#c6a34f", 0.7, 0.7))
    p.append(f'<defs><path id="at" d="M {cx-192} {cy} A 192 192 0 0 1 {cx+192} {cy}"/>'
             f'<path id="ab" d="M {cx-176} {cy} A 176 176 0 0 0 {cx+176} {cy}"/></defs>')
    top = academy.upper()[:22]
    p.append(f'<text font-family="Outfit,Arial" font-size="24" letter-spacing="6" '
             f'fill="#8a6a25"><textPath href="#at" startOffset="50%" '
             f'text-anchor="middle">{top}</textPath></text>')
    p.append('<text font-family="Outfit,Arial" font-size="18" letter-spacing="6" '
             'fill="#8a6a25"><textPath href="#ab" startOffset="50%" '
             'text-anchor="middle">CERTIFIED PROGRAM</textPath></text>')
    p.append("</svg>")
    return "".join(p)


def corner_svg():
    return ('<svg viewBox="0 0 60 60" width="100%" height="100%" '
            'xmlns="http://www.w3.org/2000/svg"><g fill="none" stroke="#b08d3f" '
            'stroke-width="1.1"><path d="M2 30 Q2 2 30 2"/><path d="M8 30 Q8 8 30 8"/>'
            '<circle cx="12" cy="12" r="2.4" fill="#b08d3f" stroke="none"/>'
            '<path d="M8 30 Q8 30 8 44 M30 8 Q44 8 44 8" stroke-width="0.8"/>'
            '<path d="M14 14 Q22 6 30 6 M14 14 Q6 22 6 30"/></g></svg>')


def crest_svg(mono):
    return ('<svg viewBox="0 0 60 60" width="100%" height="100%" '
            'xmlns="http://www.w3.org/2000/svg"><g fill="none" stroke="#b08d3f" '
            'stroke-width="1.3"><path d="M30 4 L52 12 L52 30 Q52 48 30 56 Q8 48 8 30 '
            'L8 12 Z"/><path d="M30 10 L46 16 L46 30 Q46 43 30 50 Q14 43 14 30 L14 16 Z" '
            'stroke-width="0.8"/></g><text x="30" y="37" font-family="Playfair Display,'
            'Georgia,serif" font-weight="700" font-size="19" fill="#b08d3f" '
            f'text-anchor="middle">{mono[:1]}</text></svg>')
