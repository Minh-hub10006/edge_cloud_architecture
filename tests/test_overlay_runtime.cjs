"use strict";
// Execute the actual rendered page with controlled clocks and browser I/O.
// This tests integration/lifecycle; it does not replace real WebRTC hardware tests.
const assert=require("node:assert/strict"),vm=require("node:vm"),fs=require("node:fs");
const html=fs.readFileSync(0,"utf8"),scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
assert.equal(scripts.length,2);
let now=0,canvasCount=0,callback=null,nextTimer=0;const timers=new Map(),elements=new Map(),listeners={};
function canvas(){canvasCount++;return {width:0,height:0,getContext(){return {
  drawImage(){},clearRect(){},strokeRect(){},fillRect(){},fillText(){},measureText(){return {width:10};},
  getImageData(){return {data:new Uint8Array(128*64*4)};}
};}};}
const video={videoWidth:1280,videoHeight:784,srcObject:null,
  requestVideoFrameCallback(fn){callback=fn;return 1;},cancelVideoFrameCallback(){callback=null;},play(){return Promise.resolve();}};
elements.set("video",video);elements.set("display",canvas());
for(const id of ["connection","match","videoPts","metaPts","counts","buffer","debug"])elements.set(id,{textContent:""});
const sockets=[];
class Socket{constructor(){sockets.push(this);}close(){this.onclose?.({code:1000,reason:"closed"});}}
class Peer{constructor(){this.connectionState="connected";this.iceGatheringState="complete";}addTransceiver(){}close(){this.connectionState="closed";}
  async createOffer(){return {sdp:"offer"};}async setLocalDescription(x){this.localDescription=x;}async setRemoteDescription(){}
}
const context=vm.createContext({console,Uint8Array,DataView,Set,Map,Math,Number,JSON,Promise,Error,URL,
  document:{getElementById:id=>elements.get(id),createElement:()=>canvas(),addEventListener(name,fn){listeners[name]=fn;},hidden:false},
  window:{addEventListener(name,fn){listeners[name]=fn;}},location:{protocol:"http:",host:"localhost:8003"},
  performance:{now:()=>now},requestAnimationFrame(){},WebSocket:Socket,RTCPeerConnection:Peer,
  AbortSignal:{timeout(){return {}; }},fetch:async()=>({ok:true,headers:{get(){return null;}},text:async()=>"answer"}),
  setTimeout(fn,delay){const id=++nextTimer;timers.set(id,{fn,delay});return id;},clearTimeout(id){timers.delete(id);}
});
const run=code=>vm.runInContext(code,context);
for(const script of scripts)run(script);
async function flush(){await new Promise(setImmediate);for(let i=0;i<12;i++)await Promise.resolve();}
async function main(){
  await flush();assert.ok(callback,"WHEP startup installs frame callback: "+elements.get("connection").textContent);
  sockets[0].onopen();assert.match(elements.get("connection").textContent,/Metadata: connected/);
  let currentIdentity;
  context.identity=()=>currentIdentity;
  run("LiveSync.decodeIdentity=()=>identity()");
  const metadata=seq=>({node_id:"edge-01",camera_id:"cam-01",stream_id:"preview",stream_epoch:2,
    source_frame_id:seq,preview_pts:Math.round(seq*1e9/30),
    attributes:{media_identity:"pixel-v1",media_encoded_seq:seq,media_fps:30},boxes:[]});
  //600 real page callbacks/presentations must reuse canvases, not allocate600.
  for(let seq=0;seq<600;seq++){
    now=seq*1000/30;currentIdentity={seq,fps:30,pts:Math.round(seq*1e9/30),padding:false};
    sockets[0].onmessage({data:JSON.stringify(metadata(seq))});callback(now,{});run("tick()");
  }
  assert.match(elements.get("buffer").textContent,/Presentation buffer: 500 ms/);
  assert.ok(canvasCount<=22,`canvas allocations grew to${canvasCount}`);
  assert.equal(run("timeline.counts.invalid"),0);assert.equal(run("timeline.counts.unmatched"),0);
  assert.ok(run("pool.bytes<=pool.maxBytes"));assert.ok(run("timeline.counts.exact")>550);
  // Reject corrupted marker/FPS without marking the stream as completed.
  currentIdentity=null;now+=34;callback(now,{});assert.equal(run("invalidFrames"),1);
  currentIdentity={seq:600,fps:15,pts:Math.round(600*1e9/15),padding:true};now+=34;callback(now,{});
  assert.equal(run("invalidFrames"),2);assert.equal(run("health.ending"),false);
  // A peer disconnection without padding is a failure and must reconnect.
  run('pc.connectionState="disconnected";pc.onconnectionstatechange()');
  assert.equal(run("videoRetries"),1);assert.equal(run("pool.bytes"),0);
  const reconnect=[...timers.values()].find(t=>t.delay===1500);assert.ok(reconnect);reconnect.fn();await flush();
  currentIdentity={seq:700,fps:30,pts:Math.round(700*1e9/30),padding:false};now+=100;callback(now,{});
  // WS failures remain distinct from media completion and expose close code.
  sockets[0].onclose({code:1006,reason:"network test"});run("tick()");
  assert.match(elements.get("connection").textContent,/Metadata: disconnected \(code 1006: network test\)/);
  // Explicit publisher padding permits buffer drain and disables endless retries.
  currentIdentity={seq:701,fps:30,pts:Math.round(701*1e9/30),padding:true};now+=34;callback(now,{});
  assert.equal(run("health.ending"),true);assert.match(elements.get("connection").textContent,/Video: finishing/);
  now+=600;run("tick()");assert.match(elements.get("connection").textContent,/Video: ended/);
  now+=1200;run("tick()");assert.match(elements.get("match").textContent,/Stream completed/);
  run('pc.connectionState="disconnected";pc.onconnectionstatechange()');assert.equal(run("videoRetries"),1);
  now+=5000;run("tick()");assert.equal(run("videoRetries"),1);
  // Shutdown owns and releases queue + displayed image + pooled canvases.
  listeners.beforeunload();assert.equal(run("pool.bytes"),0);assert.equal(run("timeline.frames.length"),0);
  assert.equal(run("lastFrame"),null);
  console.log("PASS rendered overlay lifecycle:600 callbacks, bounded pool,500ms delay, reconnect, WS close, terminal padding, cleanup");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
