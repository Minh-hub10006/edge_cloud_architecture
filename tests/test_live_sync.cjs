"use strict";
const assert=require("node:assert/strict");
const {Timeline,interpolate,presentationBudget,CanvasPool,PlaybackHealth}=require("../web/live_sync.js");
const box=(left,id=1)=>({left,top:.2,right:left+.2,bottom:.5,class_id:2,class_name:"car",track_id:id,confidence:.9});
const event=(seq,boxes=[box(.2)])=>({source_frame_id:seq*2,preview_pts:Math.round(seq*1e9/15),attributes:{media_identity:"pixel-v1",media_encoded_seq:seq,media_fps:15},boxes});
const frame=(seq,at=0)=>({seq,pts:Math.round(seq*1e9/15),receivedAt:at,release(){this.released=true;}});
// Metadata can arrive 400ms after receipt of a frame and still match that image.
let t=new Timeline({delayMs:500});t.pushFrame(frame(748));
assert.equal(t.present(400),null);t.pushResult(event(748));
let p=t.present(500);assert.equal(p.kind,"exact");assert.equal(p.frame.seq,748);
// Browser mediaTime could start at zero or30.021s: it never enters the join.
t=new Timeline();t.pushResult(event(748));t.pushFrame({...frame(748),mediaTime:30.021});
assert.equal(t.present(500).kind,"exact");
// Missing inference between two confirmed positions moves with the object.
t=new Timeline();t.pushResult(event(10,[box(.2)]));t.pushResult(event(12,[box(.4)]));t.pushFrame(frame(11));
p=t.present(500);assert.equal(p.kind,"interpolated");assert.ok(Math.abs(p.boxes[0].left-.3)<1e-8);
// No extrapolation from stale/future data; empty detections remain empty.
t.pushFrame(frame(13,100));assert.equal(t.present(600).kind,"unmatched");
t.pushResult(event(14,[]));t.pushFrame(frame(14,200));assert.deepEqual(t.present(700).boxes,[]);
// Out of order arrivals work; expired results cannot draw over a different image.
t=new Timeline();t.pushResult(event(22));t.pushResult(event(20));t.pushFrame(frame(21));assert.equal(t.present(500).kind,"interpolated");
t.pushResult(event(19));assert.equal(t.counts.late,1);
// Queue overflow is separate from lateness; release discarded pixel buffers.
t=new Timeline({maxFrames:2,maxResults:2});const old=frame(1);t.pushFrame(old);t.pushFrame(frame(2));t.pushFrame(frame(3));
assert.equal(old.released,true);for(const seq of [20,21,22])t.pushResult(event(seq));
assert.equal(t.counts.overflow,1);assert.equal(t.counts.late,0);
// Reconnection clears both histories; identities may resume at arbitrary seq.
t.reset();assert.equal(t.frames.length,0);assert.equal(t.results.size,0);t.pushFrame(frame(900));t.pushResult(event(900));assert.equal(t.present(500).kind,"exact");
assert.equal(interpolate([box(.2,1)],[box(.3,2)],.5).length,0);
t=new Timeline();assert.equal(t.pushResult({...event(1),attributes:{}}),false);
console.log("PASS live sync: late join, bounded delay, interpolation, missing/empty results, reordering, overflow, reconnect, identity validation");

// A 1500ms requested buffer cannot fit 23 frames into a 13-frame RAM cap.
// It must present under pressure instead of discarding forever.
t=new Timeline({delayMs:1500,maxFrames:13});let shown=0;
for(let i=0;i<100;i++){t.pushFrame(frame(i,i*1000/15));if(t.present(i*1000/15))shown++;}
assert.ok(shown>70,"memory pressure starved video presentation");
t=new Timeline();
t.pushResult({...event(10),attributes:{...event(10).attributes,media_fps:30}});
t.pushResult({...event(12),attributes:{...event(12).attributes,media_fps:30}});
t.pushFrame({...frame(11),fps:15});
assert.equal(t.present(500).kind,"unmatched","wrong clock must not interpolate");

// Buffer sizing includes a displayed image; 720p30 now really holds500ms.
const budget=presentationBudget(1280,720,30,500);
assert.equal(budget.delayMs,500);assert.equal(budget.maxFrames,17);
assert.ok((budget.maxFrames+1)*1280*720*4<=64*1024*1024);
assert.ok(presentationBudget(1280,720,60,1500).delayMs<1500);
assert.throws(()=>presentationBudget(0,720,30,500));
assert.throws(()=>presentationBudget(8192,8192,30,500));
let created=0;const pool=new CanvasPool(()=>{created++;return {};});
let held=[];
for(let i=0;i<10000;i++){
  if(held.length===budget.maxFrames+1)pool.release(held.shift());
  const image=pool.acquire(1280,720);assert.ok(image);held.push(image);
  assert.ok(pool.bytes<=pool.maxBytes);
}
assert.equal(created,budget.maxFrames+1,"canvas allocation must plateau after warmup");
assert.equal(pool.acquire(1280,720),null,"never allocate beyond the budget");
for(const image of held)pool.release(image);
const resized=pool.acquire(640,360);assert.ok(resized);pool.release(resized);
pool.clear();assert.equal(pool.bytes,0);assert.equal(pool.all.size,0);

// Delayed results arriving inside the actual500ms buffer match the exact image.
t=new Timeline({delayMs:budget.delayMs,maxFrames:budget.maxFrames});
t.pushFrame({...frame(20),fps:15});assert.equal(t.present(498),null);
t.pushResult(event(20));assert.equal(t.present(500).kind,"exact");
// Beyond the buffer, no old boxes are drawn; future brackets may still use it.
t.pushFrame(frame(21,100));assert.equal(t.present(600).reason,"missing_after");
t.pushResult(event(21));assert.equal(t.counts.late,1);
t=new Timeline();t.pushResult(event(10,[box(.2,1)]));t.pushResult(event(12,[box(.2,2)]));t.pushFrame(frame(11));
p=t.present(500);assert.equal(p.reason,"track_discontinuity");assert.deepEqual(p.boxes,[]);
assert.equal(t.missingReasons.track_discontinuity,1);
t=new Timeline();t.pushResult(event(10,[box(.2)]));t.pushResult(event(12,[]));t.pushFrame(frame(11));
assert.equal(t.present(500).reason,"detection_transition");
t=new Timeline();t.pushResult(event(10));t.pushResult(event(20));t.pushFrame(frame(11));
assert.equal(t.present(500).reason,"metadata_gap");
t=new Timeline();t.pushResult(event(10,[]));t.pushResult(event(12,[]));t.pushFrame(frame(11));
p=t.present(500);assert.equal(p.kind,"interpolated");assert.deepEqual(p.boxes,[]);
// Padding is a verified terminal signal; lack of frames alone is NOT EOF.
const health=new PlaybackHealth();assert.equal(health.state(0),"waiting");
health.verified(0);health.presented(500);health.verified(422);health.presented(922);
assert.equal(health.maxVideoGapMs,422);assert.equal(health.maxPresentationGapMs,422);
assert.equal(health.state(2000),"stalled");
health.verified(2200,true);assert.equal(health.state(2400,true),"finishing");
assert.equal(health.state(5000,false),"ended");health.reset();assert.equal(health.state(5000),"waiting");
console.log("PASS jitter/buffer/pooled memory/metadata delay/track discontinuity/terminal padding regressions");
