#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

COMP_RE = re.compile(r"^\s*-\s+(\S+)\s+(\S+).*?\+\s+(?:PLACED|FIXED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)")
DIE_RE = re.compile(r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
UNITS_RE = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)")
COMPONENTS_RE = re.compile(r"COMPONENTS\s+(\d+)")
LEF_MACRO_RE = re.compile(r"^\s*MACRO\s+(\S+)")
LEF_SIZE_RE = re.compile(r"^\s*SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;")

COLORS = {
    "attention": "#2F80ED",
    "matmul": "#F2994A",
    "shared_mesh": "#BB6BD9",
    "dma": "#EB5757",
    "control": "#27AE60",
    "memory_macro": "#D71920",
    "physical": "#7F8C8D",
    "other": "#9AA4B2",
}
LABELS = {
    "attention": "Attention",
    "matmul": "Matmul",
    "shared_mesh": "Shared mesh",
    "dma": "DMA / memory",
    "control": "Control / ABI",
    "memory_macro": "SRAM macros",
    "physical": "Physical cells",
    "other": "Other logic",
}
DEFAULT_LEFS = [
    "/home/ubuntu/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
    "/home/ubuntu/.conda-sky130/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_ef_sc_hd.lef",
    "/home/ubuntu/sram22_sky130_macros/sram22_64x32m4w32/sram22_64x32m4w32.lef",
    "/home/ubuntu/sram22_sky130_macros/sram22_64x32m4w8/sram22_64x32m4w8.lef",
]


def parse_lef_sizes(paths: Iterable[Path], dbu_per_micron: int) -> Dict[str, Tuple[int, int]]:
    sizes: Dict[str, Tuple[int, int]] = {}
    for path in paths:
        if not path.exists():
            continue
        current = None
        with path.open(errors="ignore") as fh:
            for line in fh:
                m = LEF_MACRO_RE.match(line)
                if m:
                    current = m.group(1)
                    continue
                if current:
                    sm = LEF_SIZE_RE.match(line)
                    if sm:
                        sizes[current] = (
                            max(1, int(round(float(sm.group(1)) * dbu_per_micron))),
                            max(1, int(round(float(sm.group(2)) * dbu_per_micron))),
                        )
                        current = None
    return sizes


def classify(name: str, master: str) -> str:
    low = name.lower()
    mlow = master.lower()
    if "sram" in mlow or "/mem/" in low or "mem_ext" in low:
        return "memory_macro"
    if name.startswith(("FILLER_", "TAP_", "ANTENNA_")):
        return "physical"
    if any(k in mlow for k in ("fill", "decap", "tap", "diode", "conb")):
        return "physical"
    if low.startswith(("attentionopt/", "attention/")):
        return "attention"
    if low.startswith("matmul/"):
        return "matmul"
    if low.startswith("sharedmesh/") or low.startswith("shared") or "sharedtile" in low:
        return "shared_mesh"
    if low.startswith(("dmareader/", "dmawriter/")) or "dma" in low or "tilewrite" in low or "tileread" in low:
        return "dma"
    if any(k in low for k in ("descriptor", "counter", "command", "response", "rocc", "arb", "ctrl", "status")):
        return "control"
    return "other"


def short_name(name: str, max_len: int = 48) -> str:
    if len(name) <= max_len:
        return name
    parts = name.split("/")
    if len(parts) > 1:
        tail = "/".join(parts[-3:])
        if len(tail) <= max_len - 4:
            return ".../" + tail
    return name[: max_len - 1] + "..."


def is_macro_like(group: str, master: str, size: Tuple[int, int]) -> bool:
    w, h = size
    return group == "memory_macro" or "sram" in master.lower() or w * h > 20_000_000


def parse_def(def_path: Path, grid: int, sizes: Dict[str, Tuple[int, int]], include_physical: bool) -> dict:
    dbu = 1000
    die = None
    component_target = None
    rows_seen = 0
    components_seen = 0
    in_components = False
    fallback_size = (460, 2720)
    heat = {g: Counter() for g in COLORS}
    counts = Counter()
    masters = Counter()
    bboxes = {}
    macros = []
    with def_path.open(errors="ignore") as fh:
        for line in fh:
            if line.startswith("UNITS"):
                m = UNITS_RE.search(line)
                if m:
                    dbu = int(m.group(1))
                continue
            if line.startswith("DIEAREA"):
                m = DIE_RE.search(line)
                if m:
                    die = tuple(int(m.group(i)) for i in range(1, 5))
                continue
            if line.startswith("ROW"):
                rows_seen += 1
                continue
            if line.startswith("COMPONENTS"):
                m = COMPONENTS_RE.search(line)
                component_target = int(m.group(1)) if m else None
                in_components = True
                continue
            if in_components and line.startswith("END COMPONENTS"):
                break
            if not in_components or (" PLACED " not in line and " FIXED " not in line and " COVER " not in line):
                continue
            m = COMP_RE.match(line)
            if not m:
                continue
            name, master, xs, ys, orient = m.groups()
            x, y = int(xs), int(ys)
            components_seen += 1
            group = classify(name, master)
            counts[group] += 1
            if group == "physical" and not include_physical:
                continue
            size = sizes.get(master, fallback_size)
            w, h = size
            masters[master] += 1
            if die:
                x0, y0, x1, y1 = die
                gx = min(grid - 1, max(0, int((x - x0) * grid / max(1, x1 - x0))))
                gy = min(grid - 1, max(0, int((y - y0) * grid / max(1, y1 - y0))))
                heat[group][(gx, gy)] += 1
            bb = bboxes.get(group)
            if bb is None:
                bboxes[group] = [x, y, x + w, y + h]
            else:
                bb[0], bb[1], bb[2], bb[3] = min(bb[0], x), min(bb[1], y), max(bb[2], x + w), max(bb[3], y + h)
            if is_macro_like(group, master, size):
                macros.append({"name": name, "label": short_name(name), "master": master, "group": group, "x": x, "y": y, "w": w, "h": h, "orient": orient})
    if die is None:
        raise SystemExit(f"No DIEAREA found in {def_path}")
    heat_entries = []
    max_by_group = {}
    for group, cells in heat.items():
        max_by_group[group] = max(cells.values()) if cells else 0
        for (gx, gy), c in cells.items():
            heat_entries.append([group, gx, gy, c])
    x0, y0, x1, y1 = die
    return {
        "source": str(def_path), "design": "AtikCore", "dbu": dbu,
        "die": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": x1 - x0, "h": y1 - y0},
        "grid": grid, "component_target": component_target, "components_seen": components_seen, "rows_seen": rows_seen,
        "counts": dict(counts), "masters": masters.most_common(16), "colors": COLORS, "labels": LABELS,
        "heat": heat_entries, "max_by_group": max_by_group, "bboxes": bboxes, "macros": macros,
        "include_physical": include_physical,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    source = html.escape(data["source"])
    die = data["die"]
    die_um = f"{die['w'] / data['dbu']:.0f} x {die['h'] / data['dbu']:.0f} um"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atik DEF Viewer</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0f14; --panel:#111821; --panel2:#17202b; --line:#263241; --text:#e8eef7; --muted:#9cadbf; --accent:#58c4dd; }}
* {{ box-sizing:border-box; }} html,body {{ height:100%; margin:0; background:var(--bg); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; }}
.app {{ display:grid; grid-template-columns:330px 1fr; height:100vh; min-height:620px; }}
aside {{ background:linear-gradient(180deg,#121b25 0%,#0d131b 100%); border-right:1px solid var(--line); padding:18px 16px; overflow:auto; }}
main {{ position:relative; overflow:hidden; background:radial-gradient(circle at 70% 30%,rgba(88,196,221,.10),transparent 32%),#070b10; }}
h1 {{ margin:0 0 4px; font-size:20px; letter-spacing:0; }} .sub {{ color:var(--muted); font-size:12px; line-height:1.45; overflow-wrap:anywhere; }}
.metric-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:16px 0; }} .metric {{ background:var(--panel2); border:1px solid var(--line); border-radius:7px; padding:10px; }}
.metric b {{ display:block; font-size:17px; }} .metric span {{ color:var(--muted); font-size:11px; }} .section {{ margin-top:16px; padding-top:14px; border-top:1px solid var(--line); }}
.section h2 {{ margin:0 0 9px; font-size:12px; color:#cdd8e6; text-transform:uppercase; letter-spacing:.08em; }} .layer {{ display:flex; align-items:center; gap:8px; padding:6px 0; font-size:13px; }}
.swatch {{ width:13px; height:13px; border-radius:3px; box-shadow:0 0 0 1px rgba(255,255,255,.18) inset; }} .layer input {{ accent-color:var(--accent); }} .count {{ margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }}
button,input[type=text] {{ background:#182331; color:var(--text); border:1px solid #2b3a4c; border-radius:7px; padding:8px 10px; font:inherit; }} button {{ cursor:pointer; }} button:hover {{ border-color:#47627f; }} .controls {{ display:flex; gap:8px; flex-wrap:wrap; }} .search {{ width:100%; margin-top:8px; }}
#stage {{ position:absolute; inset:0; }} #layout {{ width:100%; height:100%; display:block; image-rendering:pixelated; }}
.hud {{ position:absolute; left:16px; bottom:16px; display:flex; gap:8px; flex-wrap:wrap; max-width:calc(100% - 32px); pointer-events:none; }} .pill {{ background:rgba(13,19,27,.84); border:1px solid rgba(135,160,190,.25); border-radius:999px; padding:7px 10px; font-size:12px; color:#d7e3f2; backdrop-filter:blur(8px); }}
.tooltip {{ position:absolute; display:none; pointer-events:none; min-width:220px; max-width:390px; background:rgba(11,15,20,.94); border:1px solid #38506a; border-radius:8px; padding:10px 12px; box-shadow:0 12px 32px rgba(0,0,0,.42); font-size:12px; color:#e8eef7; }} .tooltip b {{ color:white; }} .note {{ color:var(--muted); font-size:12px; line-height:1.4; }} .kbd {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:#0b1118; border:1px solid #273545; border-radius:4px; padding:1px 5px; color:#cbd8e6; }}
@media (max-width:900px) {{ .app {{ grid-template-columns:1fr; grid-template-rows:auto 1fr; }} aside {{ max-height:44vh; }} }}
</style></head><body><div class="app"><aside>
<h1>Atik DEF Viewer</h1><div class="sub">{source}</div>
<div class="metric-grid"><div class="metric"><b>{data['design']}</b><span>design</span></div><div class="metric"><b>{die_um}</b><span>die size</span></div><div class="metric"><b>{data['components_seen']:,}</b><span>placed components parsed</span></div><div class="metric"><b>{len(data['macros']):,}</b><span>macro / large blocks</span></div></div>
<div class="controls"><button id="fitBtn">Fit</button><button id="bboxBtn">Bounds</button><button id="macroBtn">Macros</button><button id="shotBtn">PNG</button></div><input id="search" class="search" type="text" placeholder="Search macro/module name">
<div class="section"><h2>Layers</h2><div id="layers"></div><p class="note">Wheel zooms. Drag pans. Hover macros or density tiles for details. Filler/decap cells are hidden unless generated with <span class="kbd">--include-physical</span>.</p></div>
<div class="section"><h2>Top masters</h2><div id="masters" class="note"></div></div>
</aside><main><div id="stage"><canvas id="layout"></canvas></div><div class="hud"><div class="pill" id="zoomPill">zoom</div><div class="pill" id="coordPill">coord</div><div class="pill" id="statusPill">ready</div></div><div class="tooltip" id="tip"></div></main></div>
<script>
const DATA={payload};
const canvas=document.getElementById('layout'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip'),stage=document.getElementById('stage');
const zoomPill=document.getElementById('zoomPill'),coordPill=document.getElementById('coordPill'),statusPill=document.getElementById('statusPill'),layerBox=document.getElementById('layers'),mastersBox=document.getElementById('masters'),searchBox=document.getElementById('search');
const groups=Object.keys(DATA.colors); const visible=Object.fromEntries(groups.map(g=>[g,DATA.counts[g]!==undefined&&g!=='physical'])); let showBounds=true,showMacros=true,query='',scale=1,ox=0,oy=0,dragging=false,lastX=0,lastY=0;
function fmt(n){{return(n||0).toLocaleString();}} function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function rgba(hex,a){{const v=hex.replace('#','');return `rgba(${{parseInt(v.slice(0,2),16)}},${{parseInt(v.slice(2,4),16)}},${{parseInt(v.slice(4,6),16)}},${{a}})`;}}
function resize(){{const dpr=window.devicePixelRatio||1,r=stage.getBoundingClientRect();canvas.width=Math.max(1,Math.floor(r.width*dpr));canvas.height=Math.max(1,Math.floor(r.height*dpr));canvas.style.width=r.width+'px';canvas.style.height=r.height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);fit();}}
function fit(){{const r=stage.getBoundingClientRect(),pad=34,d=DATA.die;scale=Math.min((r.width-pad*2)/d.w,(r.height-pad*2)/d.h);ox=(r.width-d.w*scale)/2-d.x0*scale;oy=(r.height-d.h*scale)/2+d.y1*scale;draw();}}
function ws(x,y){{return[x*scale+ox,-y*scale+oy];}} function sw(x,y){{return[(x-ox)/scale,(oy-y)/scale];}} function rect(x,y,w,h){{const p=ws(x,y+h);return[p[0],p[1],w*scale,h*scale];}}
function die(){{const d=DATA.die,r=rect(d.x0,d.y0,d.w,d.h);ctx.fillStyle='#091018';ctx.fillRect(...r);ctx.strokeStyle='#6b829e';ctx.lineWidth=1.5;ctx.strokeRect(...r);}}
function heat(){{const d=DATA.die,cw=d.w/DATA.grid,ch=d.h/DATA.grid;for(const e of DATA.heat){{const [g,gx,gy,c]=e;if(!visible[g])continue;const max=Math.max(1,DATA.max_by_group[g]||1),a=Math.min(.78,.08+.70*Math.sqrt(c/max));ctx.fillStyle=rgba(DATA.colors[g],a);const r=rect(d.x0+gx*cw,d.y0+gy*ch,cw+1,ch+1);ctx.fillRect(r[0],r[1],Math.max(1,r[2]),Math.max(1,r[3]));}}}}
function bounds(){{if(!showBounds)return;ctx.save();ctx.setLineDash([7,5]);ctx.lineWidth=1.5;for(const [g,bb]of Object.entries(DATA.bboxes)){{if(!visible[g]||g==='physical')continue;const r=rect(bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]);ctx.strokeStyle=DATA.colors[g];ctx.strokeRect(...r);ctx.fillStyle=DATA.colors[g];ctx.font='12px ui-sans-serif,system-ui';ctx.fillText(DATA.labels[g]||g,r[0]+6,r[1]+15);}}ctx.restore();}}
function match(m){{if(!query)return true;const q=query.toLowerCase();return m.name.toLowerCase().includes(q)||m.master.toLowerCase().includes(q)||m.group.toLowerCase().includes(q);}}
function macros(){{if(!showMacros)return;ctx.lineWidth=1;for(const m of DATA.macros){{if(!visible[m.group]||!match(m))continue;const r=rect(m.x,m.y,m.w,m.h);ctx.fillStyle=rgba(DATA.colors[m.group],m.group==='memory_macro'?.70:.42);ctx.strokeStyle='#f3f7ff';ctx.fillRect(r[0],r[1],Math.max(2,r[2]),Math.max(2,r[3]));ctx.strokeRect(r[0],r[1],Math.max(2,r[2]),Math.max(2,r[3]));if(r[2]>34&&r[3]>16){{ctx.fillStyle='#fff';ctx.font='11px ui-monospace,SFMono-Regular,Menlo,monospace';ctx.fillText(m.label,r[0]+5,r[1]+13,Math.max(20,r[2]-10));}}}}}}
function draw(){{const r=stage.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);die();heat();bounds();macros();zoomPill.textContent=`zoom ${{(scale*DATA.dbu).toFixed(2)}} px/um`;statusPill.textContent=`${{DATA.design}} · grid ${{DATA.grid}}x${{DATA.grid}}`;}}
function hit(wx,wy){{for(let i=DATA.macros.length-1;i>=0;i--){{const m=DATA.macros[i];if(visible[m.group]&&match(m)&&wx>=m.x&&wx<=m.x+m.w&&wy>=m.y&&wy<=m.y+m.h)return m;}}return null;}}
function heatAt(wx,wy){{const d=DATA.die,gx=Math.floor((wx-d.x0)*DATA.grid/d.w),gy=Math.floor((wy-d.y0)*DATA.grid/d.h);if(gx<0||gy<0||gx>=DATA.grid||gy>=DATA.grid)return[];return DATA.heat.filter(e=>e[1]===gx&&e[2]===gy&&visible[e[0]]).sort((a,b)=>b[3]-a[3]).slice(0,5);}}
function move(ev){{const cr=canvas.getBoundingClientRect(),x=ev.clientX-cr.left,y=ev.clientY-cr.top;if(dragging){{ox+=x-lastX;oy+=y-lastY;lastX=x;lastY=y;draw();return;}}const p=sw(x,y),wx=p[0],wy=p[1];coordPill.textContent=`${{(wx/DATA.dbu).toFixed(1)}}um, ${{(wy/DATA.dbu).toFixed(1)}}um`;const m=hit(wx,wy),hs=heatAt(wx,wy);if(m||hs.length){{tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,window.innerWidth-430)+'px';tip.style.top=Math.min(ev.clientY+14,window.innerHeight-160)+'px';tip.innerHTML=m?`<b>${{esc(m.label)}}</b><br>${{esc(m.master)}}<br>${{DATA.labels[m.group]||m.group}}<br>${{(m.w/DATA.dbu).toFixed(1)}} x ${{(m.h/DATA.dbu).toFixed(1)}} um`:'<b>Density tile</b><br>'+hs.map(e=>`${{DATA.labels[e[0]]||e[0]}}: ${{fmt(e[3])}} cells`).join('<br>');}}else tip.style.display='none';}}
function controls(){{for(const g of groups){{if(DATA.counts[g]===undefined&&g!=='memory_macro')continue;const row=document.createElement('label');row.className='layer';row.innerHTML=`<input type="checkbox" ${{visible[g]?'checked':''}}><span class="swatch" style="background:${{DATA.colors[g]}}"></span><span>${{DATA.labels[g]||g}}</span><span class="count">${{fmt(DATA.counts[g])}}</span>`;row.querySelector('input').addEventListener('change',e=>{{visible[g]=e.target.checked;draw();}});layerBox.appendChild(row);}}mastersBox.innerHTML=DATA.masters.map(([m,c])=>`${{esc(m)}}: ${{fmt(c)}}`).join('<br>');}}
canvas.addEventListener('mousedown',ev=>{{dragging=true;const r=canvas.getBoundingClientRect();lastX=ev.clientX-r.left;lastY=ev.clientY-r.top;}});window.addEventListener('mouseup',()=>dragging=false);canvas.addEventListener('mousemove',move);canvas.addEventListener('mouseleave',()=>tip.style.display='none');canvas.addEventListener('wheel',ev=>{{ev.preventDefault();const r=canvas.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top,p=sw(x,y),factor=Math.exp(-ev.deltaY*.0014);scale*=factor;ox=x-p[0]*scale;oy=y+p[1]*scale;draw();}},{{passive:false}});
document.getElementById('fitBtn').onclick=fit;document.getElementById('bboxBtn').onclick=()=>{{showBounds=!showBounds;draw();}};document.getElementById('macroBtn').onclick=()=>{{showMacros=!showMacros;draw();}};document.getElementById('shotBtn').onclick=()=>{{const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download='atik-def-view.png';a.click();}};searchBox.oninput=e=>{{query=e.target.value.trim();draw();}};window.addEventListener('resize',resize);controls();resize();
</script></body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a self-contained interactive Atik DEF viewer.")
    ap.add_argument("def_file", type=Path)
    ap.add_argument("--out", type=Path, default=Path("atik_def_viewer.html"))
    ap.add_argument("--grid", type=int, default=220)
    ap.add_argument("--lef", type=Path, action="append", default=[])
    ap.add_argument("--include-physical", action="store_true")
    args = ap.parse_args()
    if args.grid < 32 or args.grid > 700:
        raise SystemExit("--grid should be between 32 and 700")
    if not args.def_file.exists():
        raise SystemExit(f"DEF not found: {args.def_file}")
    dbu = 1000
    with args.def_file.open(errors="ignore") as fh:
        for line in fh:
            m = UNITS_RE.search(line)
            if m:
                dbu = int(m.group(1)); break
    sizes = parse_lef_sizes([Path(p) for p in DEFAULT_LEFS] + args.lef, dbu)
    data = parse_def(args.def_file, args.grid, sizes, args.include_physical)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(data))
    print(f"wrote {args.out}")
    print(f"components parsed: {data['components_seen']:,}; macros/large blocks: {len(data['macros']):,}; grid entries: {len(data['heat']):,}")

if __name__ == "__main__":
    main()
