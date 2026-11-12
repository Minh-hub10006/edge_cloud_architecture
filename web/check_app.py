"""Small read-only page to check the results in the database (v4, 6/11). NOT the product web, which another person builds.

Video comes straight from the Edge's MediaMTX over WebRTC (WHEP), boxes and IDs only from the database, so this also
shows how to query it. Sync is exact: each video frame carries its number in the identity band (scripts/make_main_video.py,
decoded by web/live_sync.js) and the page draws the stored boxes of that frame (media_pts = number / fps). The page keeps
verified frames in its own queue for VIDEO_DELAY_MS and shows each with its boxes, so boxes are stored before their frame appears. Real camera streams have no band: the product web needs another sync.

  DB_DSN=postgresql://... NODE_HOSTS='{"edge-246":"192.168.180.246"}' python -m web.check_app   # port 18200
"""
from __future__ import annotations

import json
import os
from pathlib import Path

STREAMS_SQL = """
SELECT node_id, camera_id, stream_id, stream_epoch, count(*) AS frames, avg(box_count)::real AS boxes,
       avg(latency_ms)::real AS latency_ms, max(model_mode) AS mode, extract(epoch FROM now() - max(capture_time))::real AS age_s
FROM frames WHERE capture_time > now() - INTERVAL '10 seconds'
GROUP BY node_id, camera_id, stream_id, stream_epoch ORDER BY node_id, camera_id"""
RECENT_SQL = """
SELECT media_pts_ns / 1000000 AS pts_ms, track_id, class_name, confidence, x1, y1, x2, y2,
       (extract(epoch FROM capture_time) * 1000)::bigint AS cap_ms
FROM detections
WHERE node_id = %s AND camera_id = %s AND stream_id = %s AND stream_epoch = %s AND capture_time > now() - INTERVAL '30 seconds'
      AND capture_time > to_timestamp(%s / 1000.0)
ORDER BY capture_time"""
# 12/11: the cursor was "media_pts_ns > the newest the page has". The test video loops (-stream_loop since 11/11) and
# its frame number returns to 0: nothing was "newer" any more and the boxes vanished for good after the first loop.

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Kiểm tra kết quả v4</title>
<script src="live_sync.js"></script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f7f8;--fg:#1c2024;--muted:#60646c;--line:#d9dce0;--card:#fff;--box:#12a150}
@media (prefers-color-scheme:dark){:root{--bg:#141618;--fg:#edeef0;--muted:#a0a4ab;--line:#33373c;--card:#1d2023;--box:#3dd68c}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 12px}#none{color:var(--muted)}
#grid{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
.tile{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;max-width:100%}
.tile h2{font-size:14px;font-weight:600;margin:0 0 6px;overflow-wrap:anywhere}
.stage{position:relative;background:#000;height:78vh;max-width:100%;aspect-ratio:9/16}
.stage video,.stage canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.stage canvas{pointer-events:none}
.nums{margin:6px 0 0;color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
</style></head><body><h1>Kiểm tra kết quả trong database</h1>
<p id="none">Không có luồng nào trong 10 giây gần nhất.</p><div id="grid"></div>
<script>
const VIDEO_DELAY_MS=1200,TAG=__TAG__;
const grid=document.getElementById('grid'),none=document.getElementById('none'),tiles=new Map();
const color=getComputedStyle(document.documentElement).getPropertyValue('--box').trim();
const key=s=>[s.node_id,s.camera_id,s.stream_id,s.stream_epoch].join('|');
function tile(s){const t=document.createElement('section');t.className='tile';
 const h=document.createElement('h2');h.textContent=`${s.node_id} / ${s.camera_id} / ${s.stream_id}`;
 const st=document.createElement('div');st.className='stage';const f=document.createElement('video');f.muted=f.autoplay=f.playsInline=true;
 const c=document.createElement('canvas');c.width=720;c.height=1280;st.append(f,c);
 // the stage takes the exact shape of the video, so the canvas covers the picture and nothing is cropped
 f.addEventListener('loadedmetadata',()=>{const W=f.videoWidth,H=f.videoHeight-64;st.style.aspectRatio=W+'/'+H;
 if(W>H){st.style.height='auto';st.style.width='min(960px,100%)'}});
 const n=document.createElement('p');n.className='nums';t.append(h,st,n);grid.append(t);const v={t,c,n,s,pc:null,boxes:new Map(),hit:0,miss:0,bad:0,q:[],pool:[],newest:0,lead:null,busy:false,last:null,lastAt:0};
 if(s.video_url)play(v,f,s.video_url);return v}
async function play(v,el,url){try{const pc=v.pc=new RTCPeerConnection();pc.addTransceiver('video',{direction:'recvonly'});
 pc.ontrack=e=>{el.srcObject=e.streams[0];
 if(!el.requestVideoFrameCallback){v.err=' · cần Chrome/Edge để khóa box theo khung';return}
 const onFrame=()=>{draw(v,el);el.requestVideoFrameCallback(onFrame)};
 el.requestVideoFrameCallback(onFrame)};
 await pc.setLocalDescription(await pc.createOffer());
 await new Promise(r=>{if(pc.iceGatheringState==='complete')return r();pc.onicegatheringstatechange=()=>{if(pc.iceGatheringState==='complete')r()};setTimeout(r,1500)});
 const a=await fetch(url+'/whep',{method:'POST',headers:{'Content-Type':'application/sdp'},body:pc.localDescription.sdp});
 if(!a.ok)throw new Error('WHEP '+a.status);await pc.setRemoteDescription({type:'answer',sdp:await a.text()})}
 catch(e){v.err=' · video lỗi: '+e.message}}
async function refresh(){let list;try{list=await (await fetch('api/streams')).json()}catch(e){return}
 const seen=new Set(list.map(key));for(const [k,v] of tiles)if(!seen.has(k)){if(v.pc)v.pc.close();v.t.remove();tiles.delete(k)}
 for(const s of list){const k=key(s);if(!tiles.has(k))tiles.set(k,tile(s));const v=tiles.get(k);v.s=s;
  v.n.textContent=`${(s.frames/10).toFixed(1)} khung/s · ${s.boxes.toFixed(1)} box/khung · trễ ${s.latency_ms.toFixed(0)} ms · ${s.mode} · mới nhất ${s.age_s.toFixed(1)} s trước${v.err||''} · khung khớp ${v.hit}, chưa có box ${v.miss}, không đọc được mã ${v.bad}, box đi trước video ${v.lead??'?'} khung`}
 none.hidden=list.length>0}
async function pull(v){if(v.busy)return;v.busy=true;const s=v.s,q=new URLSearchParams({node:s.node_id,camera:s.camera_id,stream:s.stream_id,epoch:s.stream_epoch,after_cap_ms:v.cursor||-1});
 try{const r=await (await fetch('api/recent?'+q)).json();for(const f of r.frames){v.boxes.delete(f.pts_ms);v.boxes.set(f.pts_ms,f.boxes);v.newest=f.pts_ms;
 for(const b of f.boxes)if(b.cap_ms>(v.cursor||0))v.cursor=b.cap_ms}
 for(const k of v.boxes.keys()){if(v.boxes.size>600)v.boxes.delete(k);else break}}catch(e){}finally{v.busy=false}}
const band=document.createElement('canvas');band.width=128;band.height=64;const bctx=band.getContext('2d',{willReadFrequently:true});
function draw(v,el){const W=el.videoWidth,H=el.videoHeight;if(W<128||H<=64)return;
 bctx.drawImage(el,0,H-64,128,64,0,0,128,64);const id=LiveSync.decodeIdentity(bctx.getImageData(0,0,128,64).data,TAG);
 if(!id){v.bad++;return}
 // 12/11: with 4+ cameras on a slow link the video falls seconds behind the boxes; boxes are kept ~40 s (600 frames)
 // and fetched for the last 30 s, so a late video still finds them. A loop of the test video (frame number back to 0)
 // only resets the held box: the next lap's boxes replace the old ones frame by frame.
 const ms0=Math.floor(id.pts/1e6);if(v.bandMs!==undefined&&ms0<v.bandMs-1000)v.last=null;
 v.bandMs=ms0;
 const s=Math.min(1,720/W),w=Math.round(W*s),h=Math.round((H-64)*s);
 const im=v.pool.pop()||Object.assign(document.createElement('canvas'),{width:w,height:h});
 im.getContext('2d').drawImage(el,0,0,W,H-64,0,0,w,h);
 v.q.push({im,w,h,fps:id.fps,ms:Math.floor(id.pts/1e6),at:performance.now()});
 while(v.q.length>60)v.pool.push(v.q.shift().im)}
function present(v){const now=performance.now();let f=null;
 while(v.q.length&&now-v.q[0].at>=VIDEO_DELAY_MS){if(f)v.pool.push(f.im);f=v.q.shift()}
 if(!f)return;const c=v.c;if(c.width!==f.w||c.height!==f.h){c.width=f.w;c.height=f.h}
 const ctx=c.getContext('2d'),w=f.w,h=f.h;ctx.drawImage(f.im,0,0);v.pool.push(f.im);
 v.lead=Math.round((v.newest-f.ms)*f.fps/1000);
 let list=v.boxes.get(f.ms);if(list){v.hit++;v.last=list;v.lastAt=now}else{v.miss++;if(now-(v.lastAt||0)>200)return;list=v.last}
 ctx.lineWidth=Math.max(2,w/400);ctx.font=Math.round(Math.max(16,w/45))+'px system-ui';ctx.strokeStyle=ctx.fillStyle=color;
 for(const b of list){const x=b.x1*w,y=b.y1*h;ctx.strokeRect(x,y,(b.x2-b.x1)*w,(b.y2-b.y1)*h);
 ctx.fillText(`#${b.track_id??'-'} ${b.class_name}`,x+2,Math.max(16,y-4))}}
refresh();setInterval(refresh,2000);setInterval(()=>tiles.forEach(pull),200);setInterval(()=>tiles.forEach(present),33);
</script></body></html>"""


def create_app(connect=None, node_hosts=None):
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from media.copy_publisher import MAIN_KEY
    from media.frame_identity import stream_tag
    from starlette.routing import Route

    hosts = json.loads(os.getenv("NODE_HOSTS", "{}")) if node_hosts is None else node_hosts

    async def open_connection():
        if connect is not None:
            return await connect()
        import psycopg
        return await psycopg.AsyncConnection.connect(os.environ["DB_DSN"], autocommit=True)

    async def query(*steps):
        conn = await open_connection()
        try:
            async with conn.cursor() as cur:
                return [await step(cur) for step in steps]
        finally:
            await conn.close()

    async def page(request):
        return HTMLResponse(PAGE.replace("__TAG__", str(stream_tag(MAIN_KEY))))

    async def live_sync(request):
        return Response((Path(__file__).parent / "live_sync.js").read_text(encoding="utf-8"), media_type="text/javascript")

    async def streams(request):
        async def run(cur):
            await cur.execute(STREAMS_SQL)
            names = [c.name for c in cur.description]
            return [dict(zip(names, row)) for row in await cur.fetchall()]
        (rows,) = await query(run)
        for r in rows:
            host = hosts.get(r["node_id"])
            r["video_url"] = (f"http://{host}:8889/edge/{r['node_id']}/{r['camera_id']}/{r['stream_id']}/"
                              f"epoch-{r['stream_epoch']}") if host else None
        return JSONResponse(rows)

    async def recent(request):
        p = request.query_params
        try:
            stream = (p["node"], p["camera"], p["stream"], int(p["epoch"]))
            after_ns = int(p.get("after_cap_ms", "-1"))   # capture time (ms since 1970) of the newest box the page has
        except (KeyError, ValueError):
            return JSONResponse({"error": "node, camera, stream and integer epoch are required"}, status_code=400)

        async def run(cur):
            await cur.execute(RECENT_SQL, (*stream, after_ns))
            names = [c.name for c in cur.description][1:]
            frames = {}
            for row in await cur.fetchall():
                frames.setdefault(row[0], []).append(dict(zip(names, row[1:])))
            return {"frames": [{"pts_ms": t, "boxes": b} for t, b in frames.items()]}
        (result,) = await query(run)
        return JSONResponse(result)

    return Starlette(routes=[Route("/", page), Route("/live_sync.js", live_sync), Route("/api/streams", streams), Route("/api/recent", recent)])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("CHECK_PORT", "18200")), log_level="warning")
