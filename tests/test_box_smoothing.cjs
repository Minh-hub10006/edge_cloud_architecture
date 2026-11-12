"use strict";
const assert=require("node:assert/strict");
const {TrackBoxSmoother,BoxReview}=require("../web/live_sync.js");
const box=(x=0,id=7,cls=2)=>({left:.1+x,top:.1,right:.3+x,bottom:.3,track_id:id,class_id:cls,confidence:.8});
const raw=[box()];assert.equal(new TrackBoxSmoother().apply(raw,0),raw);
const f=new TrackBoxSmoother({tauMs:120});const first=box();
f.apply([first],0);const next=box(.02),copy=JSON.stringify(next);
const shown=f.apply([next],66666667)[0];assert(shown.left>first.left&&shown.left<next.left);assert.equal(JSON.stringify(next),copy);
assert.equal(f.apply([box(.03,8)],133333334)[0].left,.13); // Never blend IDs.
assert.equal(f.apply([box(.04,null)],200000001)[0].left,.14); // Unconfirmed remains raw.
f.apply([box()],266666668);f.apply([],333333335);
assert.equal(f.apply([box(.04)],400000002)[0].left,.14); // No ghost/reacquisition blend.
assert.equal(f.apply([box(.05)],900000002)[0].left,box(.05).left); // Gap reset.
assert.equal(f.apply([box(.6)],966666669)[0].left,.7); // Large jump reset.
assert.equal(f.apply([box(.01)],966666669)[0].left,.11); // Repeated timestamp reset.
const duplicates=[box(),box(.01)];assert.deepEqual(f.apply(duplicates,1033333336),duplicates);
const small=new TrackBoxSmoother({tauMs:120,maxTracks:2});small.apply([box(0,1),box(0,2),box(0,3)],0);assert.equal(small.tracks.size,2);
// A step response after equal elapsed time must agree at 15 and 30 FPS.
function step(fps){const s=new TrackBoxSmoother({tauMs:120});s.apply([box()],0);let out;for(let i=1;i<=fps;i++)out=s.apply([box(.02)],i*1e9/fps)[0];return out.left;}
assert(Math.abs(step(15)-step(30))<1e-3); // alpha-beta is near, not exactly, rate-invariant
// Constant motion: no steady-state lag (plain EMA trailed by ~tau of motion).
function lag(fps){const s=new TrackBoxSmoother({tauMs:120});let d;for(let i=0;i<=fps*2;i++){const b=box(.004*i*15/fps);d=b.left-s.apply([b],i*1e9/fps)[0].left;}return Math.abs(d);}
assert(lag(15)<1e-4&&lag(30)<1e-4);
const stationary=new TrackBoxSmoother({tauMs:120});let rawEnergy=0,shownEnergy=0;
for(let i=0;i<90;i++){const x=i%2?.01:-.01,b=box(x),out=stationary.apply([b],i*1e9/15)[0];if(i>10){rawEnergy+=x*x;shownEnergy+=(out.left-.1)**2;}}
assert(shownEnergy<rawEnergy*.3);
const review=new BoxReview(2);const event={source_frame_id:1,preview_pts:0,attributes:{media_encoded_seq:0},boxes:[box(),box(.01,8)]};
review.push(event);event.boxes[0].left=.9;review.push({...event,attributes:{media_encoded_seq:2},boxes:[box(0,null)]});review.push(event);
const report=review.report();assert.equal(report.events_recorded,2);assert.equal(report.events_not_recorded,1);assert.equal(report.encoded_sequence_gaps,1);
assert.equal(report.events[0].boxes[0].left,.1);assert.equal(report.overlap_pairs.length,1);assert.equal(report.boxes_with_track_id,2);
console.log("Track smoothing / raw review regressions PASS");
