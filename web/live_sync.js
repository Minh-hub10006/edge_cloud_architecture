/* Frame identity and bounded presentation logic, shared with Node regressions. */
(function(root){
"use strict";
function crc16(bytes){let crc=65535;for(const b of bytes){crc^=b<<8;for(let i=0;i<8;i++)crc=((crc<<1)^((crc&32768)?4129:0))&65535;}return crc;}
function decodeIdentity(rgba,tag){
  if(rgba.length!==128*64*4)return null;
  const bytes=new Uint8Array(16);
  for(let bit=0;bit<128;bit++){
    const row=Math.floor(bit/16),col=bit%16;let sum=0;
    for(let y=2;y<6;y++)for(let x=2;x<6;x++){
      const p=((row*8+y)*128+col*8+x)*4;sum+=(rgba[p]+rgba[p+1]+rgba[p+2])/3;
    }
    const v=sum/16;if(v>80&&v<175)return null;
    bytes[Math.floor(bit/8)]|=(v>127?1:0)<<(7-bit%8);
  }
  const d=new DataView(bytes.buffer);
  if(bytes[0]!==69||bytes[1]!==86||bytes[2]!==1||bytes[3]>1||d.getUint32(4)!==tag||crc16(bytes.slice(0,14))!==d.getUint16(14))return null;
  const seq=d.getUint32(8),fps=d.getUint16(12);
  if(!fps||fps>240)return null;
  return {seq,fps,padding:bytes[3]===1,pts:Math.round(seq*1e9/fps)};
}
function iou(a,b){const w=Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left)),h=Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
  const area=w*h;return area/Math.max(1e-9,(a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-area);}
function interpolate(a,b,t){
  const used=new Set(),boxes=[];
  for(const x of a){
    let index=-1,best=.3;
    for(let i=0;i<b.length;i++){
      const y=b[i];if(used.has(i)||x.class_id!==y.class_id)continue;
      if(x.track_id!=null&&y.track_id!=null){if(x.track_id===y.track_id){index=i;break;}continue;}
      const score=iou(x,y);if(score>best){index=i;best=score;}
    }
    if(index<0)continue;
    used.add(index);const y=b[index],z={...x};
    for(const k of ["left","top","right","bottom"])z[k]=x[k]+(y[k]-x[k])*t;
    boxes.push(z);
  }
  return boxes;
}
// Display only: never change metadata, detections or tracker inputs. PTS-based
// filtering makes the response independent of browser refresh rate.
class TrackBoxSmoother{
  constructor({tauMs=0,maxGapMs=250,maxTracks=256}={}){
    this.tauMs=tauMs;this.maxGapMs=maxGapMs;this.maxTracks=maxTracks;this.reset();
  }
  reset(){this.tracks=new Map();this.lastPts=null;}
  apply(boxes,pts){
    if(!this.tauMs){this.reset();return boxes;}
    if(!Number.isFinite(pts)||this.lastPts!==null&&pts<=this.lastPts){this.reset();return boxes;}
    this.lastPts=pts;const next=new Map(),keys=new Set(),duplicate=new Set();
    for(const b of boxes){const key=JSON.stringify([b.class_id,b.track_id]);if(keys.has(key))duplicate.add(key);keys.add(key);}
    const output=boxes.map(b=>{
      if(b.track_id==null||![b.left,b.top,b.right,b.bottom].every(Number.isFinite)||b.right<=b.left||b.bottom<=b.top)return b;
      const key=JSON.stringify([b.class_id,b.track_id]);if(duplicate.has(key))return b;
      const old=this.tracks.get(key),dt=old?(pts-old.pts)/1e6:Infinity;
      let shown={...b},velocity={left:0,top:0,right:0,bottom:0};
      // A long gap or large jump is a reacquisition, not jitter to hide.
      if(old&&dt<=this.maxGapMs&&iou(old.raw,b)>.1){
        // Alpha-beta filter: predict with the track's velocity, then correct. Plain
        // EMA trails moving vehicles by ~tau; constant motion here has no steady lag.
        const alpha=1-Math.exp(-dt/this.tauMs),beta=alpha*alpha/(2-alpha);
        for(const k of ["left","top","right","bottom"]){
          const predicted=old.shown[k]+old.velocity[k]*dt,residual=b[k]-predicted;
          shown[k]=predicted+alpha*residual;velocity[k]=old.velocity[k]+beta*residual/dt;
        }
        if(shown.right<=shown.left||shown.bottom<=shown.top){shown={...b};velocity={left:0,top:0,right:0,bottom:0};}
      }
      if(next.size<this.maxTracks)next.set(key,{pts,raw:{...b},shown,velocity});
      return shown;
    });
    // Missing boxes disappear immediately; there is no last-box hold.
    this.tracks=next;return output;
  }
}
class BoxReview{
  constructor(maxEvents=900){this.maxEvents=maxEvents;this.events=[];this.overflow=0;}
  push(event){if(this.events.length>=this.maxEvents){this.overflow++;return;}this.events.push(JSON.parse(JSON.stringify(event)));}
  report(){
    const pairs=[];let tracked=0,total=0,seqGaps=0,previousSeq=null;const tracks=new Map();
    for(const e of this.events){
      const seq=e.attributes?.media_encoded_seq;if(previousSeq!==null&&seq>previousSeq+1)seqGaps+=seq-previousSeq-1;previousSeq=seq;
      for(const b of e.boxes){total++;if(b.track_id==null)continue;tracked++;
        const key=JSON.stringify([b.class_id,b.track_id]);if(!tracks.has(key))tracks.set(key,[]);
        tracks.get(key).push({pts:e.preview_pts,box:b});
      }
      for(let i=0;i<e.boxes.length;i++)for(let j=i+1;j<e.boxes.length;j++){
        const a=e.boxes[i],b=e.boxes[j],overlap=iou(a,b);
        if(overlap>=.5&&pairs.length<1000)pairs.push({source_frame_id:e.source_frame_id,iou:overlap,same_class:a.class_id===b.class_id,track_ids:[a.track_id,b.track_id]});
      }
    }
    return {schema_version:1,scope:"browser-received raw tracked metadata; not ground truth or mAP",
      events_recorded:this.events.length,events_not_recorded:this.overflow,encoded_sequence_gaps:seqGaps,
      boxes_total:total,boxes_with_track_id:tracked,distinct_class_track_ids:tracks.size,
      overlap_pairs:pairs,overlap_note:"IoU >= 0.5 is a review cue, not proof of duplicate objects",
      events:this.events};
  }
}
class Timeline{
  constructor({delayMs=500,maxFrames=24,maxResults=240,maxGapMs=250}={}){
    this.delayMs=delayMs;this.maxFrames=maxFrames;this.maxResults=maxResults;this.maxGapMs=maxGapMs;
    this.frames=[];this.results=new Map();this.lastShown=-1;this.lastReceived=-1;
    this.counts={late:0,overflow:0,videoDropped:0,invalid:0,unmatched:0,exact:0,interpolated:0};
    this.missingReasons={};
  }
  reset(){for(const f of this.frames)f.release?.();this.frames=[];this.results.clear();this.lastShown=-1;this.lastReceived=-1;}
  pushFrame(frame){
    if(frame.seq<=this.lastReceived){frame.release?.();return false;}
    this.lastReceived=frame.seq;this.frames.push(frame);
    while(this.frames.length>this.maxFrames){this.frames.shift().release?.();this.counts.videoDropped++;}
    return true;
  }
  pushResult(event){
    const seq=event.attributes?.media_encoded_seq;
    if(event.attributes?.media_identity!=="pixel-v1"||!Number.isSafeInteger(seq)||seq<0||!Number.isSafeInteger(event.preview_pts)||event.preview_pts<0||!Array.isArray(event.boxes)){
      this.counts.invalid++;return false;
    }
    if(seq<=this.lastShown)this.counts.late++;
    // A late sample may still bracket a later buffered frame, so keep it.
    this.results.set(seq,event);
    while(this.results.size>this.maxResults){this.results.delete(Math.min(...this.results.keys()));this.counts.overflow++;}
    return true;
  }
  select(frame){
    const valid=event=>!frame.fps||(event.attributes.media_fps===frame.fps&&Math.abs(event.preview_pts-Math.round(event.attributes.media_encoded_seq*1e9/frame.fps))<=1);
    const exact=this.results.get(frame.seq);
    if(exact&&valid(exact)&&Math.abs(exact.preview_pts-frame.pts)<=1)return {boxes:exact.boxes,kind:"exact",event:exact};
    let left=null,right=null;
    for(const [seq,event] of this.results){
      if(!valid(event))continue;
      if(seq<frame.seq&&(!left||event.preview_pts>left.preview_pts))left=event;
      if(seq>frame.seq&&(!right||event.preview_pts<right.preview_pts))right=event;
    }
    if(left&&right&&left.preview_pts<frame.pts&&right.preview_pts>frame.pts){
      if(right.preview_pts-left.preview_pts>this.maxGapMs*1e6)return {boxes:[],kind:"unmatched",reason:"metadata_gap"};
      const boxes=interpolate(left.boxes,right.boxes,(frame.pts-left.preview_pts)/(right.preview_pts-left.preview_pts));
      // Never guess that different confirmed track IDs are the same object.
      if(!boxes.length&&(left.boxes.length||right.boxes.length))return {boxes:[],kind:"unmatched",reason:left.boxes.length&&right.boxes.length?"track_discontinuity":"detection_transition"};
      return {boxes,kind:"interpolated"};
    }
    return {boxes:[],kind:"unmatched",reason:!left?"missing_before":"missing_after"};
  }
  present(now){
    let frame=null;
    // Capacity pressure must not starve presentation when the requested delay
    // exceeds the memory budget (or the source FPS increases).
    while(this.frames.length&&(this.frames[0].receivedAt+this.delayMs<=now||(!frame&&this.frames.length>=this.maxFrames))){
      if(frame){frame.release?.();this.counts.videoDropped++;}frame=this.frames.shift();
    }
    if(!frame)return null;
    const selected=this.select(frame);this.lastShown=frame.seq;this.counts[selected.kind]++;
    if(selected.reason)this.missingReasons[selected.reason]=(this.missingReasons[selected.reason]||0)+1;
    const previous=[...this.results.keys()].filter(seq=>seq<frame.seq).sort((a,b)=>b-a);
    for(const seq of previous.slice(1))this.results.delete(seq);
    return {frame,...selected};
  }
}
function presentationBudget(width,height,fps,requestedMs,maxBytes=64*1024*1024){
  if(![width,height,fps,requestedMs,maxBytes].every(Number.isFinite)||width<=0||height<=0||fps<=0||requestedMs<0)throw Error("Invalid presentation budget");
  // Reserve one slot for the displayed image, and two queued slots for jitter.
  const maxFrames=Math.min(48,Math.floor(maxBytes/(width*height*4))-1);
  if(maxFrames<3)throw Error("Frame too large for presentation memory budget");
  return {maxFrames,delayMs:Math.min(requestedMs,(maxFrames-2)*1000/fps)};
}
class CanvasPool{
  constructor(createCanvas,maxBytes=64*1024*1024){this.createCanvas=createCanvas;this.maxBytes=maxBytes;this.bytes=0;this.all=new Set();this.used=new Set();this.free=[];}
  dispose(image){this.bytes-=image.width*image.height*4;this.all.delete(image);image.width=0;image.height=0;}
  acquire(width,height){
    const i=this.free.findIndex(c=>c.width===width&&c.height===height);
    if(i>=0){const image=this.free.splice(i,1)[0];this.used.add(image);return image;}
    const bytes=width*height*4;
    while(this.free.length&&this.bytes+bytes>this.maxBytes)this.dispose(this.free.pop());
    if(this.bytes+bytes>this.maxBytes)return null;
    const image=this.createCanvas();image.width=width;image.height=height;
    this.bytes+=bytes;this.all.add(image);this.used.add(image);return image;
  }
  release(image){if(this.used.delete(image))this.free.push(image);}
  clear(){for(const image of this.all){image.width=0;image.height=0;}this.all.clear();this.used.clear();this.free=[];this.bytes=0;}
}
class PlaybackHealth{
  constructor(){this.reset();}
  reset(){this.lastVerifiedAt=null;this.lastPresentedAt=null;this.ending=false;this.maxVideoGapMs=0;this.maxPresentationGapMs=0;}
  verified(now,padding=false){if(this.lastVerifiedAt!==null)this.maxVideoGapMs=Math.max(this.maxVideoGapMs,now-this.lastVerifiedAt);this.lastVerifiedAt=now;if(padding)this.ending=true;}
  presented(now){if(this.lastPresentedAt!==null)this.maxPresentationGapMs=Math.max(this.maxPresentationGapMs,now-this.lastPresentedAt);this.lastPresentedAt=now;}
  state(now,buffered=false){if(this.ending)return buffered?"finishing":"ended";if(this.lastVerifiedAt===null)return "waiting";return now-this.lastVerifiedAt>1500?"stalled":"receiving";}
}
const api={crc16,decodeIdentity,interpolate,TrackBoxSmoother,BoxReview,Timeline,presentationBudget,CanvasPool,PlaybackHealth};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
else root.LiveSync=api;
})(globalThis);
