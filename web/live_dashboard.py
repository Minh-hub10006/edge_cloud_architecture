"""One page, one tile per Edge node; each tile embeds the synchronized live player."""
from __future__ import annotations

import json
from pathlib import Path
import re

_NODE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def render_dashboard(nodes: list[str], *, fps: float = 15, buffer_ms: int = 500, smooth_ms: int = 120,
                     poll_ms: int = 2000) -> str:
    if not nodes or len(set(nodes)) != len(nodes) or not all(_NODE.fullmatch(n) for n in nodes):
        raise ValueError("nodes must be unique ids of [A-Za-z0-9._-]")
    if not 0 < fps <= 240 or not 100 <= buffer_ms <= 1500 or not 0 < smooth_ms <= 200 or poll_ms < 500:
        raise ValueError("invalid fps, buffer_ms, smooth_ms or poll_ms")
    config = json.dumps({"nodes": nodes, "fps": fps, "bufferMs": buffer_ms, "smoothMs": smooth_ms,
                         "pollMs": poll_ms}, separators=(",", ":")).replace("<", "\\u003c")
    logic = Path(__file__).with_name("live_dashboard.js").read_text(encoding="utf-8")
    return _PAGE.replace("__LOGIC__", logic).replace("__CONFIG__", config)


# Theme and tile layout follow the project dashboard (web/frontend/style.css on plate_lisence).
_PAGE = r'''<!doctype html><html lang="vi"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Live</title><style>
:root{--bg:#f4f5f7;--card:#fff;--border:#d8dde5;--border-strong:#b8c0cc;--text:#1c2430;--muted:#5c6778;
--accent:#2c4a6e;--accent-soft:#e8eef5;--edge:#1f6b45;--danger:#9b2c2c;--radius:3px;
--ui:"Source Sans 3","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--ui);background:var(--bg);color:var(--text);font-size:15px;line-height:1.5;padding:16px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:12px}
h1{font-family:Georgia,"Times New Roman",serif;font-size:1.3rem;font-weight:600;margin-right:auto}
.meta{font-size:.8rem;color:var(--muted)}
.seg{display:inline-flex;border:1px solid var(--border-strong);border-radius:var(--radius);overflow:hidden}
.seg button{border:0;background:var(--card);color:var(--text);padding:.35rem .9rem;font:inherit;font-size:.84rem;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:.85rem}
.tile{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);display:flex;flex-direction:column;min-width:0}
.tile-head{display:flex;align-items:center;gap:.4rem;padding:.5rem .7rem;border-bottom:1px solid var(--border)}
.node{font-family:var(--mono);font-weight:500;margin-right:auto}
.badge{font-size:.68rem;padding:.1rem .35rem;border-radius:var(--radius);font-weight:600;border:1px solid transparent;white-space:nowrap}
.badge.edge{background:#e8f2ec;color:var(--edge);border-color:#c5ddd0}
.badge.cloud{background:var(--accent-soft);color:var(--accent);border-color:#c5d3e4}
.badge.live{background:#e8f2ec;color:var(--edge);border-color:#c5ddd0}
.badge.stopped,.badge.waiting{background:#f1f2f4;color:var(--muted);border-color:var(--border)}
.screen{background:#10151d;aspect-ratio:16/12.3;position:relative}
.screen iframe{position:absolute;inset:0;width:100%;height:100%;border:0}
.screen .empty{position:absolute;inset:0;display:grid;place-items:center;color:#9aa6b6;font-size:.9rem;padding:1rem;text-align:center}
.tile-foot{display:flex;align-items:center;gap:.5rem;padding:.45rem .7rem}
.tile-foot .meta{margin-right:auto;font-family:var(--mono);font-size:.72rem;overflow-wrap:anywhere}
.btn{background:var(--card);color:var(--text);border:1px solid var(--border-strong);border-radius:var(--radius);
padding:.3rem .7rem;font:inherit;font-size:.8rem;cursor:pointer;white-space:nowrap}
.btn:hover{border-color:var(--accent);color:var(--accent)}.btn:disabled{opacity:.5;cursor:default}
#cloud{font-size:.8rem}#cloud.down{color:var(--danger);font-weight:600}
.tile-head .badge.edge,.tile-head .badge.cloud{font-size:.85rem;padding:.15rem .5rem}
.grid.focus{grid-template-columns:1fr}.grid.focus .tile:not(.big){display:none}
.grid.focus .screen{aspect-ratio:auto;height:calc(100vh - 150px)}
.grid.focus .tile-head .badge.edge,.grid.focus .tile-head .badge.cloud{font-size:1.1rem}
</style></head><body>
<header><h1>Edge Live</h1>
<div class="seg" role="group" aria-label="Chế độ box">
<button type="button" data-mode="raw" aria-pressed="true">RAW</button><button type="button" data-mode="smooth" aria-pressed="false">SMOOTH</button></div>
<span class="meta" id="mode-hint"></span><span class="meta" id="cloud">Đang kết nối Cloud…</span></header>
<main class="grid" id="grid"></main>
<script>__LOGIC__</script><script>
"use strict";
const cfg=__CONFIG__,D=LiveDashboard,grid=document.getElementById("grid"),tiles={};
let mode="raw";try{if(localStorage.getItem("edgeLiveMode")==="smooth")mode="smooth";}catch(_e){}
const opts={fps:cfg.fps,bufferMs:cfg.bufferMs,smoothMs:cfg.smoothMs};
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
for(const node of cfg.nodes){
  const tile=el("section","tile"),head=el("div","tile-head"),screen=el("div","screen"),foot=el("div","tile-foot");
  const kind=el("span","badge"),state=el("span","badge waiting","Chờ luồng"),empty=el("div","empty","Chưa có luồng từ node này. Chạy live.sh edge <model> trên Edge, ô sẽ tự hiện.");
  const info=el("span","meta",D.caption(null)),restart=el("button","btn","Kết nối lại"),zoom=el("button","btn","Phóng to");
  restart.type="button";zoom.type="button";restart.disabled=true;kind.hidden=true;
  head.append(el("span","node",node),kind,state);screen.append(empty);foot.append(info,restart,zoom);tile.append(head,screen,foot);grid.append(tile);
  const t=tiles[node]={tile,screen,empty,kind,state,info,restart,zoom,frame:null,url:null};
  restart.addEventListener("click",()=>reload(t));
  zoom.addEventListener("click",()=>focus(focused===node?null:node));
}
// One Edge fills the page; the others stay connected underneath (hidden), so switching back is instant.
let focused=null;
function focus(node){
  focused=node;grid.className=node?"grid focus":"grid";
  for(const n of cfg.nodes){tiles[n].tile.className=n===node?"tile big":"tile";tiles[n].zoom.textContent=n===node?"Thu nhỏ":"Phóng to";}
}
function show(t,url){
  if(url===t.url)return;
  t.url=url;
  if(!url){t.frame?.remove();t.frame=null;t.empty.hidden=false;t.restart.disabled=true;return;}
  if(!t.frame){t.frame=el("iframe");t.frame.title="live";t.frame.allow="autoplay";t.screen.append(t.frame);}
  t.empty.hidden=true;t.restart.disabled=false;t.frame.src=url;
}
// Unload the old player first so it closes its WebRTC session, then load it again.
function reload(t){if(!t.frame||!t.url)return;const url=t.url;t.frame.src="about:blank";setTimeout(()=>{if(t.url===url)t.frame.src=url;},300);}
function setMode(next){
  mode=next;try{localStorage.setItem("edgeLiveMode",mode);}catch(_e){}
  for(const b of document.querySelectorAll(".seg button"))b.setAttribute("aria-pressed",String(b.dataset.mode===mode));
  document.getElementById("mode-hint").textContent=mode==="smooth"?`Box làm mượt ${cfg.smoothMs} ms (chỉ hiển thị)`:"Box gốc từ ByteTrack";
  poll();
}
for(const b of document.querySelectorAll(".seg button"))b.addEventListener("click",()=>setMode(b.dataset.mode));
let polling=false;
async function poll(){
  if(polling)return;polling=true;
  const cloud=document.getElementById("cloud");
  try{
    const response=await fetch("/v1/streams/status",{cache:"no-store",signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    const latest=D.latestByNode((await response.json()).streams,cfg.nodes);
    for(const node of cfg.nodes){
      const row=latest[node],t=tiles[node],s=D.tileState(row);
      t.state.className=`badge ${s.state}`;t.state.textContent=s.label;
      t.kind.hidden=!row;if(row){const p=D.placement(row.model_mode);t.kind.className=`badge ${p}`;t.kind.textContent=D.modelLabel(row.model_mode);}
      t.info.textContent=D.caption(row);
      show(t,D.tileUrl(row,mode,opts));
    }
    cloud.className="";cloud.textContent=`Cloud OK · cập nhật ${new Date().toLocaleTimeString()}`;
  }catch(e){cloud.className="down";cloud.textContent=`Mất kết nối Cloud (${e.message}); thử lại sau ${cfg.pollMs/1000}s`;}
  finally{polling=false;}
}
setMode(mode);setInterval(poll,cfg.pollMs);
</script></body></html>'''
