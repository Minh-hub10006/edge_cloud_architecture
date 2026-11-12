"use strict";
const fs=require("fs"),assert=require("node:assert/strict"),vm=require("vm");
// Runs the two inline scripts of the rendered /live page (dumped by test_live_dashboard.py) against a stub DOM/Cloud.
const dir=process.argv[2];
class El{constructor(tag){this.tag=tag;this.children=[];this.hidden=false;this.className="";this.textContent="";this.attrs={};this.listeners={};this.dataset={};this.disabled=false;}
  append(...c){for(const x of c){x.parent=this;this.children.push(x);}} remove(){this.removed=true;if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
  addEventListener(t,f){this.listeners[t]=f;} setAttribute(k,v){this.attrs[k]=v;} click(){this.listeners.click?.();}}
const ids={grid:new El("main"),"mode-hint":new El("span"),cloud:new El("span")};
const segButtons=[Object.assign(new El("button"),{dataset:{mode:"raw"}}),Object.assign(new El("button"),{dataset:{mode:"smooth"}})];
const store={};let response={streams:[]},fail=false,timers=[];
const sandbox={globalThis:null,console,setTimeout:(f)=>{timers.push(f);return 1;},setInterval:()=>1,
  localStorage:{getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=v;}},
  AbortSignal:{timeout:()=>undefined},
  document:{createElement:t=>new El(t),getElementById:id=>ids[id],querySelectorAll:()=>segButtons},
  fetch:async()=>{if(fail)throw Error("network down");return{ok:true,json:async()=>response};}};
sandbox.globalThis=sandbox;vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(dir+"/dash_logic.js","utf8"),sandbox);
vm.runInContext(fs.readFileSync(dir+"/dash_ui.js","utf8"),sandbox);
const tiles=ids.grid.children;const tile=i=>{const [head,screen,foot]=tiles[i].children;return{head,screen,foot,frame:screen.children.find(c=>c.tag==="iframe"),empty:screen.children[0],restart:foot.children[1],info:foot.children[0],state:head.children[2],kind:head.children[1]};};
const flush=()=>new Promise(r=>setImmediate(r));
(async()=>{
  await flush();
  assert.equal(tiles.length,3);assert.equal(tile(0).head.children[0].textContent,"edge-246");
  assert.equal(tile(0).frame,undefined);assert.equal(tile(0).empty.hidden,false);assert.equal(tile(0).restart.disabled,true);
  assert.match(ids.cloud.textContent,/Cloud OK/);
  const row=(node,epoch,age,mode="yolo_edge")=>({node_id:node,camera_id:"traffic-01",stream_id:"s",stream_epoch:epoch,age_s:age,results:5,model_mode:mode,model_id:"m",boxes:3,source_frame_id:9,result_fps:15});
  response={streams:[row("edge-252",7,0.4,"lwdetr_large_cloud"),row("edge-246",3,0.2)]};
  vm.runInContext("poll()",sandbox);await flush();
  assert.equal(tile(0).frame.src,"/v1/streams/edge-246/traffic-01/s/3/live?fps=15&buffer_ms=500&smooth_ms=0&embed=1");
  assert.equal(tile(1).kind.textContent,"LW-DETR-L · Cloud");assert.equal(tile(0).kind.textContent,"YOLOv8n · Edge");
  assert.equal(tile(0).state.textContent,"LIVE");assert.equal(tile(2).frame,undefined);assert.equal(tile(0).restart.disabled,false);
  const iframe=tile(0).frame;
  // Same stream again: no reload. New epoch: swap automatically.
  vm.runInContext("poll()",sandbox);await flush();assert.equal(tile(0).frame,iframe);
  response={streams:[row("edge-246",4,0.1),row("edge-252",7,9)]};
  vm.runInContext("poll()",sandbox);await flush();
  assert.match(tile(0).frame.src,/\/4\/live\?/);assert.equal(tile(1).state.textContent,"Đã dừng 9s");assert.match(tile(1).frame.src,/\/7\/live\?/);
  // SMOOTH: every tile reloads with smoothing; choice persists.
  segButtons[1].click();await flush();
  assert.match(tile(0).frame.src,/smooth_ms=120&embed=1$/);assert.equal(store.edgeLiveMode,"smooth");assert.equal(segButtons[1].attrs["aria-pressed"],"true");
  // Restart: blank first (old player unloads, closes WebRTC), then the same URL.
  const before=tile(0).frame.src;tile(0).restart.click();assert.equal(tile(0).frame.src,"about:blank");timers.shift()();assert.equal(tile(0).frame.src,before);
  // Zoom: one tile fills the page, its button flips; zooming it again (or another) restores/switches.
  const zoom=i=>tile(i).foot.children[2];
  zoom(1).click();assert.equal(ids.grid.className,"grid focus");assert.equal(tiles[1].className,"tile big");
  assert.equal(tiles[0].className,"tile");assert.equal(zoom(1).textContent,"Thu nhỏ");assert.equal(tile(0).frame.src,before);
  zoom(2).click();assert.equal(tiles[2].className,"tile big");assert.equal(tiles[1].className,"tile");assert.equal(zoom(1).textContent,"Phóng to");
  zoom(2).click();assert.equal(ids.grid.className,"grid");assert.equal(tiles[2].className,"tile");
  // Cloud down: banner, tiles keep their last player.
  fail=true;vm.runInContext("poll()",sandbox);await flush();
  assert.match(ids.cloud.textContent,/Mất kết nối Cloud/);assert.equal(ids.cloud.className,"down");assert.equal(tile(0).frame.src,before);
  console.log("DASHBOARD_SIM_PASS");
})().catch(e=>{console.error(e);process.exit(1);});
