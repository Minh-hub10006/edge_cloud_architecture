/* Multi-Edge live page: pick each node's newest stream and build its tile player URL. */
(function(root){
"use strict";
const LIVE_AGE_S=3;
// Rows come from /v1/streams/status, most recent first; keep the first row per configured node.
function latestByNode(rows,nodes){
  const out={};for(const node of nodes)out[node]=null;
  for(const row of rows||[])if(row&&Object.prototype.hasOwnProperty.call(out,row.node_id)&&out[row.node_id]===null)out[row.node_id]=row;
  return out;
}
function tileState(row,liveAgeS=LIVE_AGE_S){
  if(!row)return{state:"waiting",label:"Chờ luồng"};
  if(row.age_s<=liveAgeS)return{state:"live",label:"LIVE"};
  return{state:"stopped",label:`Đã dừng ${Math.round(row.age_s)}s`};
}
function placement(mode){return typeof mode==="string"&&mode.endsWith("_edge")?"edge":"cloud";}
const MODEL_NAMES={yolo_edge:"YOLOv8n",yolo11s_edge:"YOLO11s",rtdetr_cloud:"RT-DETRv2",lwdetr_large_cloud:"LW-DETR-L"};
function modelLabel(mode){return `${MODEL_NAMES[mode]||mode||"?"} · ${placement(mode)==="edge"?"Edge":"Cloud"}`;}
function tileUrl(row,mode,opts){
  if(!row)return null;
  const seg=v=>encodeURIComponent(String(v));
  const smooth=mode==="smooth"?opts.smoothMs:0;
  return `/v1/streams/${seg(row.node_id)}/${seg(row.camera_id)}/${seg(row.stream_id)}/${seg(row.stream_epoch)}/live`+
    `?fps=${opts.fps}&buffer_ms=${opts.bufferMs}&smooth_ms=${smooth}&embed=1`;
}
function caption(row){
  if(!row)return"Chưa nhận kết quả nào từ node này";
  const rate=row.result_fps==null?"—":`${Number(row.result_fps).toFixed(1)} kq/s`;
  return `${row.model_id} · ${rate} · ${row.boxes} box · frame ${row.source_frame_id} · epoch ${row.stream_epoch}`;
}
const api={LIVE_AGE_S,latestByNode,tileState,placement,modelLabel,tileUrl,caption};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
else root.LiveDashboard=api;
})(globalThis);
