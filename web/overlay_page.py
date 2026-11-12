"""Frame-identified WebRTC player with bounded synchronized presentation."""
from __future__ import annotations
import json
from pathlib import Path
from media.profile import StreamKey, media_path
from media.frame_identity import stream_tag


def render_overlay_page(stream_key: StreamKey, *, media_base: str, fps: float = 15,
                        websocket_path: str, allow_smoke: bool = False,
                        buffer_ms: int = 500, smooth_ms: int = 0, review: bool = False,
                        embed: bool = False) -> str:
    if not 0 < fps <= 240 or not 100 <= buffer_ms <= 1500:
        raise ValueError("fps must be 1..240 and buffer_ms 100..1500")
    if not 0 <= smooth_ms <= 200:
        raise ValueError("smooth_ms must be 0..200 (display only)")
    base = media_base.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("media_base must be an HTTP(S) URL")
    config = json.dumps({"whepUrl": f"{base}/{media_path(stream_key)}/whep",
                         "websocketPath": websocket_path, "streamKey": list(stream_key),
                         "streamTag": stream_tag(stream_key), "bufferMs": buffer_ms,
                         "smoothMs": smooth_ms, "review": review,
                         "allowSmoke": False}, separators=(",", ":")).replace("<", "\\u003c")
    engine = Path(__file__).with_name("live_sync.js").read_text(encoding="utf-8")
    page = _PAGE.replace("__ENGINE__", engine).replace("__CONFIG__", config)
    # Embedded in a /live tile: same player, video first, compact status, no title/help.
    return page.replace("<body>", '<body class="embed">', 1) if embed else page


_PAGE = r'''<!doctype html><html lang="vi"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live video</title><style>
body{background:#10151d;color:#e7edf5;font:14px system-ui;margin:20px}
main{max-width:1280px;margin:auto}canvas#display{width:100%;height:auto;background:#000;max-height:80vh;object-fit:contain}
video{position:absolute;opacity:0;width:128px;height:72px;pointer-events:none}
.status{background:#1d2734;padding:12px;line-height:1.7}h1{font-size:18px}
.embed{margin:0}.embed main{max-width:none}.embed h1,.embed p{display:none}.embed canvas#display{max-height:none}.embed #videoPts,.embed #metaPts,.embed #debug{display:none}.embed .status{font-size:11px;padding:4px 6px;line-height:1.45}
</style></head><body><main><h1>Live video + synchronized detections</h1>
<video id="video" autoplay muted playsinline></video><canvas id="display" width="1280" height="720"></canvas>
<div class="status"><div id="connection">Connecting...</div><div id="match">Waiting for verified video frame</div>
<div id="videoPts"></div><div id="metaPts"></div><div id="counts"></div><div id="buffer"></div><div id="debug"></div></div>
<p>Display smoothing adds motion lag; detections and tracking stay raw. <button onclick="downloadBoxReview()">Download box / track review</button></p>
</main><script>__ENGINE__</script><script>
"use strict";
const cfg=__CONFIG__,video=document.getElementById("video"),display=document.getElementById("display"),ctx=display.getContext("2d");
const label=id=>document.getElementById(id),timeline=new LiveSync.Timeline({delayMs:cfg.bufferMs,maxFrames:48});
const pool=new LiveSync.CanvasPool(()=>document.createElement("canvas")),health=new LiveSync.PlaybackHealth();
const smoother=new LiveSync.TrackBoxSmoother({tauMs:cfg.smoothMs}),review=new LiveSync.BoxReview();
function downloadBoxReview(){
  if(!cfg.review){alert("Open this page with &review=1 before the trial to record raw metadata.");return;}
  const blob=new Blob([JSON.stringify({...review.report(),stream_key:cfg.streamKey,smooth_ms:cfg.smoothMs},null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="yolo11s_box_review.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const marker=document.createElement("canvas");marker.width=128;marker.height=64;
const mctx=marker.getContext("2d",{willReadFrequently:true});
const captured=document.createElement("canvas"),captureCtx=captured.getContext("2d");
let pc=null,ws=null,resource=null,closed=false,videoConnected=false,metadataConnected=false,generation=0;
let frameCallback=null,lastValidAt=0,lastVideoCallbackAt=0,peerReadyAt=0,lastDrawnAt=0,videoRetry=null,wsRetry=null,invalidFrames=0,lastFrame=null,frameFps=null;
let metadataClose="",metadataLastAt=null,metadataMaxGapMs=0,metadataDisconnects=0,metadataReconnects=0,videoRetries=0,overlayExpired=false;
label("buffer").textContent=`Presentation buffer: ${cfg.bufferMs} ms (bounded)`;
function status(){const state=health.state(performance.now(),timeline.frames.length>0);
  label("connection").textContent=`Video: ${state} | Metadata: ${metadataConnected?"connected":`disconnected${metadataClose?" ("+metadataClose+")":""}`}`;}
function draw(frame,boxes){
  const image=frame.image;if(display.width!==image.width||display.height!==image.height){display.width=image.width;display.height=image.height;}
  ctx.drawImage(image,0,0);ctx.lineWidth=2;ctx.font="16px system-ui";
  for(const b of boxes){
    if(![b.left,b.top,b.right,b.bottom].every(Number.isFinite))continue;
    const x=b.left*display.width,y=b.top*display.height,w=(b.right-b.left)*display.width,h=(b.bottom-b.top)*display.height;
    ctx.strokeStyle="#42e887";ctx.strokeRect(x,y,w,h);
    const title=`${b.class_name}${b.track_id==null?"":" #"+b.track_id} ${Math.round(b.confidence*100)}%`;
    ctx.fillStyle="rgba(10,20,15,.8)";ctx.fillRect(x,Math.max(0,y-21),ctx.measureText(title).width+8,21);
    ctx.fillStyle="#fff";ctx.fillText(title,x+4,Math.max(16,y-5));
  }
}
function onVideoFrame(now,_metadata){
  if(closed)return;
  lastVideoCallbackAt=performance.now();
  try{
    if(video.videoWidth<128||video.videoHeight<=64)throw Error("Video requires v10 Edge frame identity");
    if(captured.width!==video.videoWidth||captured.height!==video.videoHeight){captured.width=video.videoWidth;captured.height=video.videoHeight;}
    captureCtx.drawImage(video,0,0);
    mctx.drawImage(captured,0,video.videoHeight-64,128,64,0,0,128,64);
    const identity=LiveSync.decodeIdentity(mctx.getImageData(0,0,128,64).data,cfg.streamTag);
    if(!identity)throw Error("Frame identity missing/corrupt; check Edge version and stream epoch");
    if(frameFps!==null&&frameFps!==identity.fps)throw Error("FPS changed inside stream epoch");
    frameFps=identity.fps;
    lastValidAt=performance.now();videoConnected=true;health.verified(lastValidAt,identity.padding);status();
    if(!identity.padding){
      const scale=Math.min(1,1280/video.videoWidth),width=Math.round(video.videoWidth*scale),height=Math.round((video.videoHeight-64)*scale);
      const budget=LiveSync.presentationBudget(width,height,identity.fps,cfg.bufferMs,pool.maxBytes);
      timeline.maxFrames=budget.maxFrames;timeline.delayMs=budget.delayMs;
      while(timeline.frames.length>=timeline.maxFrames){timeline.frames.shift().release?.();timeline.counts.videoDropped++;}
      const image=pool.acquire(width,height);
      if(!image){timeline.counts.videoDropped++;throw Error("Presentation pool full; no unverified frame displayed");}
      try{image.getContext("2d").drawImage(captured,0,0,video.videoWidth,video.videoHeight-64,0,0,image.width,image.height);}
      catch(error){pool.release(image);throw error;}
      label("buffer").textContent=`Presentation buffer: ${Math.round(timeline.delayMs)} ms | Display smoothing: ${cfg.smoothMs?cfg.smoothMs+" ms, confirmed IDs only":"OFF (raw)"}`;
      timeline.pushFrame({...identity,receivedAt:performance.now(),image,release(){pool.release(image);}});
    }
  }catch(e){invalidFrames++;label("match").textContent=e.message;}
  frameCallback=video.requestVideoFrameCallback(onVideoFrame);
}
function tick(){
  if(closed)return;
  const now=performance.now(),item=timeline.present(now);
  if(item){
    lastFrame?.release();lastFrame=item.frame;lastDrawnAt=now;health.presented(now);overlayExpired=false;
    draw(item.frame,smoother.apply(item.boxes,item.frame.pts));
    label("videoPts").textContent=`Video PTS: ${item.frame.pts} ns | encoded frame ${item.frame.seq}`;
    label("match").textContent=item.kind==="exact"?`Matched source frame ${item.event.source_frame_id}`:
      item.kind==="interpolated"?"Interpolated tracks between verified detections":`No detection for this frame (${item.reason||"deadline/drop"})`;
  }
  if(lastFrame&&now-lastDrawnAt>1000&&!overlayExpired){draw(lastFrame,[]);smoother.reset();overlayExpired=true;
    label("match").textContent=health.ending?"Stream completed (publisher padding) — overlay cleared":"Video paused/stalled — overlay expired";}
  if(lastValidAt&&now-lastValidAt>1500){videoConnected=false;status();}
  const lastFrameCallback=lastVideoCallbackAt||peerReadyAt;
  if(!health.ending&&pc?.connectionState==="connected"&&lastFrameCallback&&now-lastFrameCallback>3000){
    label("connection").textContent="Video stalled; reconnecting";retryVideo();
  }
  const c=timeline.counts;label("counts").textContent=`Matched ${c.exact} | Interpolated ${c.interpolated} | Missing ${c.unmatched} | Late ${c.late} | Overflow ${c.overflow} | Video skipped ${c.videoDropped} | Invalid ${c.invalid+invalidFrames}`;
  label("debug").textContent=`Max gaps: video ${Math.round(health.maxVideoGapMs)} ms / presentation ${Math.round(health.maxPresentationGapMs)} ms / metadata ${Math.round(metadataMaxGapMs)} ms | Missing reasons: ${JSON.stringify(timeline.missingReasons)} | Pool ${Math.round(pool.bytes/1024/1024)} MiB | Video retries ${videoRetries} | Metadata closes ${metadataDisconnects}, reconnects ${metadataReconnects}`;
  status();
  requestAnimationFrame(tick);
}
function releasePeer(){
  generation++;
  if(frameCallback!==null){video.cancelVideoFrameCallback(frameCallback);frameCallback=null;}
  if(resource){fetch(resource,{method:"DELETE"}).catch(()=>{});resource=null;}
  if(pc){pc.onconnectionstatechange=null;pc.close();pc=null;}
  video.srcObject=null;videoConnected=false;
  timeline.reset();smoother.reset();lastFrame?.release();lastFrame=null;frameFps=null;pool.clear();health.reset();overlayExpired=false;
  lastVideoCallbackAt=0;peerReadyAt=0;lastValidAt=0;lastDrawnAt=0;ctx.clearRect(0,0,display.width,display.height);status();
}
function retryVideo(){if(closed||videoRetry||health.ending)return;videoRetries++;releasePeer();videoRetry=setTimeout(()=>{videoRetry=null;startVideo();},1500);}
async function startVideo(){
  const attempt=++generation;
  const stale=()=>closed||attempt!==generation;
  try{
    if(!video.requestVideoFrameCallback)throw Error("Browser lacks requestVideoFrameCallback");
    const current=new RTCPeerConnection();pc=current;current.addTransceiver("video",{direction:"recvonly"});
    current.ontrack=e=>{if(stale())return;video.srcObject=e.streams[0]||new MediaStream([e.track]);video.play().catch(()=>{});};
    current.onconnectionstatechange=()=>{if(!stale()&&["failed","disconnected","closed"].includes(current.connectionState))retryVideo();};
    await current.setLocalDescription(await current.createOffer());
    if(stale())return;
    if(current.iceGatheringState!=="complete")await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(Error("ICE timeout")),10000);
      current.addEventListener("icegatheringstatechange",()=>{if(current.iceGatheringState==="complete"){clearTimeout(timer);resolve();}});
    });
    if(stale())return;
    const response=await fetch(cfg.whepUrl,{method:"POST",headers:{"Content-Type":"application/sdp"},body:current.localDescription.sdp,signal:AbortSignal.timeout(10000)});
    if(stale()){
      const loc=response.headers.get("Location");
      if(loc)fetch(new URL(loc,cfg.whepUrl).href,{method:"DELETE"}).catch(()=>{});
      return;
    }
    if(!response.ok)throw Error(`WHEP ${response.status}: ${await response.text()}`);
    const loc=response.headers.get("Location");if(loc)resource=new URL(loc,cfg.whepUrl).href;
    const sdp=await response.text();if(stale())return;
    await current.setRemoteDescription({type:"answer",sdp});
    if(closed||pc!==current)return;
    peerReadyAt=performance.now();
    frameCallback=video.requestVideoFrameCallback(onVideoFrame);
  }catch(e){if(stale())return;label("connection").textContent=`Video error: ${e.message}`;retryVideo();}
}
function startMetadata(){
  if(closed)return;
  clearTimeout(wsRetry);
  const socket=new WebSocket(`${location.protocol==="https:"?"wss:":"ws:"}//${location.host}${cfg.websocketPath}`);ws=socket;
  socket.onopen=()=>{if(ws!==socket)return;if(metadataDisconnects)metadataReconnects++;metadataConnected=true;metadataClose="";status();};
  socket.onclose=e=>{if(ws!==socket)return;metadataConnected=false;metadataDisconnects++;metadataClose=`code ${e.code}${e.reason?": "+e.reason:""}`;status();if(!closed)wsRetry=setTimeout(startMetadata,1500);};
  socket.onerror=()=>socket.close();
  socket.onmessage=e=>{if(ws!==socket)return;try{
    const event=JSON.parse(e.data),key=[event.node_id,event.camera_id,event.stream_id,event.stream_epoch];
    if(JSON.stringify(key)!==JSON.stringify(cfg.streamKey))throw Error("wrong stream");
    if(timeline.pushResult(event)){
      if(cfg.review)review.push(event);
      const now=performance.now();if(metadataLastAt!==null)metadataMaxGapMs=Math.max(metadataMaxGapMs,now-metadataLastAt);metadataLastAt=now;
      label("metaPts").textContent=`Metadata PTS: ${event.preview_pts} ns | source frame ${event.source_frame_id}`;
    }
  }catch(_e){timeline.counts.invalid++;}};
}
document.addEventListener("visibilitychange",()=>{if(!document.hidden){timeline.reset();smoother.reset();lastFrame?.release();lastFrame=null;ctx.clearRect(0,0,display.width,display.height);}});
window.addEventListener("beforeunload",()=>{closed=true;clearTimeout(videoRetry);clearTimeout(wsRetry);ws?.close();releasePeer();});
startMetadata();startVideo();requestAnimationFrame(tick);
</script></body></html>'''
