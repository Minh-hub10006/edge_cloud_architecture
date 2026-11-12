"""Local-only synthetic fixture for the real Chromium presentation regression."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import numpy as np
from media.frame_identity import append_identity
from web.overlay_page import render_overlay_page

KEY = ("browser-test", "cam", "video", 81)
frames = []
for seq in range(701, 746):
    pixels = np.zeros((180, 320, 3), np.uint8)
    x = 30 + (seq-701)*2
    pixels[50:100, x:x+40] = (70, 130, 210)
    frames.append(append_identity(pixels, key=KEY, seq=seq, fps=15))
encoded = subprocess.run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
    "-video_size", "320x244", "-framerate", "15", "-i", "pipe:0", "-an", "-c:v", "libx264",
    "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline", "-bf", "0",
    "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
], input=b"".join(f.tobytes() for f in frames), capture_output=True, check=True, timeout=20).stdout
html = render_overlay_page(KEY, media_base="http://127.0.0.1:8889", websocket_path="/unused")
html = html.replace("startMetadata();startVideo();requestAnimationFrame(tick);", r'''
const originalCallback=onVideoFrame;
onVideoFrame=function(now,meta){
  originalCallback(now,meta);
  const frame=timeline.frames.at(-1);
  if(frame){
    const x=(30+(frame.seq-701)*2)/320;
    setTimeout(()=>timeline.pushResult({source_frame_id:frame.seq,preview_pts:frame.pts,
      attributes:{media_identity:"pixel-v1",media_encoded_seq:frame.seq,media_fps:15},
      boxes:[{left:x,top:50/180,right:x+40/320,bottom:100/180,class_id:2,class_name:"car",track_id:1,confidence:.9}]}),150);
  }
};
video.src="/clip";
video.addEventListener("loadeddata",()=>{video.play();frameCallback=video.requestVideoFrameCallback(onVideoFrame);},{once:true});
requestAnimationFrame(tick);
''').encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        content, kind = (encoded, "video/mp4") if self.path=="/clip" else (html, "text/html; charset=utf-8")
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
