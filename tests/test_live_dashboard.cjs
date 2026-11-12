"use strict";
const assert=require("node:assert/strict");
const D=require("../web/live_dashboard.js");
const row=(node,epoch,age=0.5,extra={})=>({node_id:node,camera_id:"traffic-01",stream_id:"live-1",stream_epoch:epoch,age_s:age,
  results:10,model_mode:"yolo_edge",model_id:"yolov8n",boxes:7,source_frame_id:120,result_fps:14.9,...extra});
// Newest row per node wins (server sorts most recent first); unknown nodes ignored; silent nodes stay null.
const latest=D.latestByNode([row("edge-252",9),row("edge-246",4),row("edge-252",8),row("jetson",1)],["edge-246","edge-252","edge-247"]);
assert.equal(latest["edge-252"].stream_epoch,9);assert.equal(latest["edge-246"].stream_epoch,4);
assert.equal(latest["edge-247"],null);assert.equal("jetson" in latest,false);
assert.deepEqual(D.latestByNode(undefined,["a"]),{a:null});
// LIVE only while results keep arriving; a finished run shows how long ago it stopped.
assert.equal(D.tileState(null).state,"waiting");
assert.equal(D.tileState(row("a",1,D.LIVE_AGE_S)).state,"live");
assert.deepEqual(D.tileState(row("a",1,12.4)),{state:"stopped",label:"Đã dừng 12s"});
// Tile URL: same player route, embed mode, smoothing only in SMOOTH, segments URL-encoded.
const opts={fps:15,bufferMs:500,smoothMs:120};
assert.equal(D.tileUrl(row("edge-246",4),"raw",opts),"/v1/streams/edge-246/traffic-01/live-1/4/live?fps=15&buffer_ms=500&smooth_ms=0&embed=1");
assert.match(D.tileUrl(row("edge-246",4),"smooth",opts),/smooth_ms=120&embed=1$/);
assert.equal(D.tileUrl(row("a/b?c",1),"raw",opts).startsWith("/v1/streams/a%2Fb%3Fc/"),true);
assert.equal(D.tileUrl(null,"raw",opts),null);
assert.equal(D.modelLabel("yolo11s_edge"),"YOLO11s · Edge");assert.equal(D.modelLabel("lwdetr_large_cloud"),"LW-DETR-L · Cloud");
assert.equal(D.modelLabel("new_cloud"),"new_cloud · Cloud");
assert.equal(D.placement("yolo11s_edge"),"edge");assert.equal(D.placement("lwdetr_large_cloud"),"cloud");assert.equal(D.placement(undefined),"cloud");
assert.equal(D.caption(row("a",3)),"yolov8n · 14.9 kq/s · 7 box · frame 120 · epoch 3");
assert.match(D.caption(row("a",3,0,{result_fps:null})),/— · 7 box/);
console.log("live_dashboard.js OK");
