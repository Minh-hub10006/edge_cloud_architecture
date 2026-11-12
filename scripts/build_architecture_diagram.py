"""Generate the editable draw.io diagram and its matching PNG preview."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ARCHITECTURE_CURRENT"
W, H = 2480, 1620
BG, INK, BORDER = "#101415", "#eef1f3", "#8c979d"
nodes, panels, notes, edges = [], [], [], []


def panel(key, x, y, w, h, title, fill):
    panels.append(dict(id=key, x=x, y=y, w=w, h=h, text=title, fill=fill))


def node(key, x, y, w, h, text, future=False, size=20):
    nodes.append(dict(id=key, x=x, y=y, w=w, h=h, text=text,
                      future=future, size=size))


def note(key, x, y, w, h, text, size=17):
    notes.append(dict(id=key, x=x, y=y, w=w, h=h, text=text, size=size))


def edge(key, source, target, points, both=False, dashed=False):
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:])), key
    edges.append(dict(id=key, source=source, target=target, points=points,
                      both=both, dashed=dashed))


panel("edge", 40, 105, 950, 1045, "EDGE — 3 planes (ví dụ: edge-246)", "#1b2839")
panel("media", 60, 165, 910, 265, "Media plane · video preview", "#19261c")
panel("data", 60, 460, 910, 360, "Data plane · inference", "#2b2021")
panel("control", 60, 850, 910, 275, "Control plane · scheduler tại Edge", "#292330")
panel("transport", 1030, 105, 240, 1045, "TRANSPORT", "#2b250f")
panel("cloud", 1310, 105, 1120, 1045, "CLOUD — dùng chung cho nhiều Edge (ví dụ: cloud-249)", "#202d1d")
panel("ingest", 1330, 165, 1080, 265, "Media ingest & decode", "#19261c")
panel("infer", 1330, 460, 1080, 360, "Inference + tracking + ResultHub", "#2b2021")
panel("feedback", 1330, 850, 1080, 275, "Cloud feedback · tổng hợp từ mọi Edge", "#292330")
panel("monitor", 40, 1190, 950, 370, "Monitoring · JSONL / Prometheus / Grafana", "#2b250f")
panel("web", 1310, 1190, 1120, 370, "Web / Browser · một trang live", "#1b2839")

note("title", 40, 25, 2400, 40, "EDGE–CLOUD VIDEO PLATFORM · KIẾN TRÚC HIỆN TẠI", 30)
note("subtitle", 40, 72, 2400, 25, "Mũi tên trắng: luồng dữ liệu / điều khiển. Khung nét đứt: phần chưa hoàn tất. Capacity hiển thị là mặc định cấu hình.", 17)

node("capture", 75, 250, 140, 105, "Capture\nFrameIdentity\nframe_id / PTS", size=18)
node("fanout", 250, 250, 140, 105, "Fan-out\ncùng frame\nsang 2 nhánh", size=18)
node("mediaq", 425, 250, 140, 105, "Media queue\nlatest · cap 8\ndrop-oldest", size=18)
node("marker", 600, 250, 140, 105, "Cadence\n+ pixel marker\nencoded_seq", size=18)
node("encoder", 775, 250, 160, 105, "FFmpeg\nH.264, bf=0\npublish RTSP", size=18)
note("marker_note", 425, 375, 350, 46, "Identity đầy đủ đi qua sidecar; marker nằm trong pixel video.", 16)

node("analyticsq", 250, 535, 140, 95, "Analytics queue\nlatest · cap 8\ndrop-oldest", size=18)
node("route", 450, 535, 150, 95, "Route [A]\nmodel_mode\ntarget_fps", size=19)
node("yolo", 670, 515, 250, 95, "YOLO tại Edge\nSNPE / NPU\n→ DetectionResult", size=20)
node("ticket", 670, 665, 250, 95, "Cloud route\n→ CloudInferenceTicket", size=20)
node("refq", 75, 685, 140, 95, "FrameRef\nsender queue\ncap 2048", size=18)
note("sidecar_note", 250, 700, 350, 65, "MediaFrameRef được gửi cho cả hai route; liên kết frame gốc với encoded_seq / PTS.", 17)

node("state", 75, 950, 240, 95, "Đo theo window\n10 features + mask\nlatency / FPS / scene", size=20)
node("policy", 365, 950, 235, 95, "Policy hiện có\nobserve / rules / cycle", size=20)
node("action", 650, 950, 285, 95, "RouteState [A]\nmodel_mode + target_fps\náp dụng window sau", size=20)
note("action_note", 650, 1060, 290, 42, "[A] được executor áp dụng tại Route phía trên.", 16)
node("drl", 75, 1060, 525, 45, "DRL trainer / learned weights: chưa triển khai", future=True, size=17)

node("t1", 1050, 250, 200, 105, "T1 · VIDEO\nRTSP / RTP\nH.264 over TCP", size=20)
node("t2", 1050, 560, 200, 150, "T2 · DATA ↔\ngRPC hoặc Kafka\nFrameRef / Ticket\nDetectionResult", size=20)
note("t2_note", 1050, 730, 200, 65, "Metadata và kết quả; Kafka không chở video.", 17)
node("t3", 1050, 950, 200, 105, "T3 · FEEDBACK\ngRPC hoặc Kafka\nCloud → Edge", size=20)

node("mtx", 1345, 250, 190, 105, "MediaMTX\nRTSP ingest", size=21)
node("decode", 1575, 250, 230, 105, "PyAV decode\nHandoff queue · cap 8\ndrop-oldest", size=20)
node("cache", 1845, 250, 525, 105, "Đọc marker → encoded_seq / PTS\nCloudFrameCache · 64 frame/stream · TTL 5s\nJoin với MediaFrameRef", size=20)
note("join_note", 1345, 375, 565, 40, "Join: stream_key / frame identity / encoded_seq / PTS.", 17)

node("receiver", 1345, 535, 210, 95, "Data receiver\nphân loại payload", size=20)
node("cloudmodel", 1600, 535, 340, 175, "Cloud inference\nSemaphore · max inflight 2\n→ tìm đúng frame trong cache\n→ D-FINE full (CUDA / GPU)\nbusy / miss / deadline → reject", size=20)
node("hub", 1995, 535, 365, 145, "ResultHub\nkiểm tra order / dedup\nByteTrack theo stream epoch\n→ DetectionResult + track_id [R]", size=20)
node("subq", 1995, 725, 365, 65, "WebSocket subscriber queue\ncap 16 · drop-oldest", size=19)
note("edge_result_note", 1510, 775, 435, 30, "DetectionResult từ Edge → ResultHub", 16)
note("reply_note", 1995, 695, 360, 22, "[R] Completion về Edge qua T2.", 16)

node("cloudtelemetry", 1345, 950, 1015, 140, "WindowFeedback cho từng Edge\ncloud_inflight + weighted cloud_arrival_pressure (gộp mọi Edge)\nCapacity phải lấy từ benchmark; thiếu capacity → unavailable\nCloud đo và báo feedback; Edge chọn model / FPS", size=21)

node("metrics", 75, 1300, 355, 100, "Edge monitor\nJSONL + Prometheus /metrics\nstate / mask / drops / latency / FPS", size=20)
node("grafana", 520, 1300, 415, 100, "Grafana Control-v1\nchưa nối đầy đủ dashboard mới", future=True, size=21)
note("monitor_note", 75, 1430, 840, 70, "Có monitor thật và policy thử nghiệm. DRL chưa train; quality constraint chưa active. Benchmark / GT quality / multi-Edge load là các bước tiếp theo.", 19)

node("video", 1345, 1300, 270, 100, "WebRTC video\nMediaMTX / WHEP", size=22)
node("metadata", 1685, 1300, 270, 100, "WebSocket metadata\nDetectionResult + track_id", size=21)
node("sync", 2015, 1275, 345, 170, "Browser synchronizer\nmarker sequence + PTS\nbuffer yêu cầu ~500 ms\nmatch → box / interpolate\nkhông match → bỏ box cũ", size=20)
note("browser_note", 1660, 1480, 700, 45, "Hai channel vào cùng live page; đồng bộ ở thời điểm render.", 19)
note("footer", 40, 1580, 2400, 25, "Queue = chờ việc có giới hạn · Cache = giữ frame để join · Semaphore = giới hạn số request đồng thời · Chưa nghiệm thu nhiều Edge vật lý.", 17)

# Explicit orthogonal routes: no automatic diagonal routing.
edge("cap_fan", "capture", "fanout", [(215, 302.5), (250, 302.5)])
edge("fan_q", "fanout", "mediaq", [(390, 302.5), (425, 302.5)])
edge("q_mark", "mediaq", "marker", [(565, 302.5), (600, 302.5)])
edge("mark_enc", "marker", "encoder", [(740, 302.5), (775, 302.5)])
edge("enc_t1", "encoder", "t1", [(935, 302.5), (1050, 302.5)])
edge("t1_mtx", "t1", "mtx", [(1250, 302.5), (1345, 302.5)])
edge("mtx_dec", "mtx", "decode", [(1535, 302.5), (1575, 302.5)])
edge("dec_cache", "decode", "cache", [(1805, 302.5), (1845, 302.5)])
edge("fan_analytics", "fanout", "analyticsq", [(320, 355), (320, 535)])
edge("analytics_route", "analyticsq", "route", [(390, 582.5), (450, 582.5)])
edge("route_yolo", "route", "yolo", [(600, 570), (635, 570), (635, 562.5), (670, 562.5)])
edge("route_ticket", "route", "ticket", [(600, 600), (635, 600), (635, 712.5), (670, 712.5)])
edge("ref_produce", "encoder", "refq", [(855, 355), (855, 445), (65, 445), (65, 665), (145, 665), (145, 685)])
edge("yolo_t2", "yolo", "t2", [(920, 562.5), (1005, 562.5), (1005, 590), (1050, 590)])
edge("ticket_t2", "ticket", "t2", [(920, 712.5), (1005, 712.5), (1005, 650), (1050, 650)])
edge("ref_t2", "refq", "t2", [(215, 755), (230, 755), (230, 800), (1015, 800), (1015, 690), (1050, 690)])
edge("t2_receiver", "t2", "receiver", [(1250, 590), (1345, 590)], both=True)
edge("ref_cache", "receiver", "cache", [(1345, 555), (1320, 555), (1320, 420), (1950, 420), (1950, 355)])
edge("cache_model", "cache", "cloudmodel", [(2110, 355), (2110, 445), (1770, 445), (1770, 535)])
edge("ticket_model", "receiver", "cloudmodel", [(1555, 582.5), (1600, 582.5)])
edge("model_hub", "cloudmodel", "hub", [(1940, 582.5), (1995, 582.5)])
edge("edge_hub", "receiver", "hub", [(1450, 630), (1450, 760), (1970, 760), (1970, 650), (1995, 650)])
edge("hub_queue", "hub", "subq", [(2177.5, 680), (2177.5, 725)])
edge("feedback_t3", "cloudtelemetry", "t3", [(1345, 1002.5), (1250, 1002.5)])
edge("t3_state", "t3", "state", [(1050, 1002.5), (1010, 1002.5), (1010, 905), (195, 905), (195, 950)])
edge("state_policy", "state", "policy", [(315, 997.5), (365, 997.5)])
edge("policy_action", "policy", "action", [(600, 997.5), (650, 997.5)])
edge("state_metrics", "state", "metrics", [(75, 997.5), (55, 997.5), (55, 1350), (75, 1350)])
edge("metrics_grafana", "metrics", "grafana", [(430, 1350), (520, 1350)], dashed=True)
edge("video_browser", "mtx", "video", [(1440, 250), (1440, 220), (2450, 220), (2450, 1535), (1480, 1535), (1480, 1400)])
edge("metadata_browser", "subq", "metadata", [(2360, 757.5), (2400, 757.5), (2400, 1245), (1820, 1245), (1820, 1300)])
edge("video_sync", "video", "sync", [(1615, 1350), (1650, 1350), (1650, 1460), (1975, 1460), (1975, 1395), (2015, 1395)])
edge("metadata_sync", "metadata", "sync", [(1955, 1350), (2015, 1350)])


def font(size, bold=False):
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")), size)


def fitted(draw, text, width, height, initial, bold=False):
    for size in range(initial, 13, -1):
        f = font(size, bold)
        lines = []
        for para in text.splitlines():
            line = ""
            for word in para.split():
                candidate = (line + " " + word).strip()
                if draw.textlength(candidate, font=f) > width and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        line_height = size + 6
        if len(lines) * line_height <= height:
            return f, lines, line_height
    raise ValueError(f"Text does not fit: {text}")


def draw_arrow(draw, points, both=False, dashed=False):
    if dashed:
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            for start in range(0, int(length), 14):
                stop = min(start + 8, length)
                p = tuple(a[i] + (b[i] - a[i]) * start / length for i in (0, 1))
                q = tuple(a[i] + (b[i] - a[i]) * stop / length for i in (0, 1))
                draw.line([p, q], fill=INK, width=2)
    else:
        draw.line(points, fill=INK, width=2, joint="curve")
    def head(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        n = math.hypot(dx, dy)
        ux, uy = dx/n, dy/n
        draw.polygon([b, (b[0]-10*ux+4*uy, b[1]-10*uy-4*ux),
                      (b[0]-10*ux-4*uy, b[1]-10*uy+4*ux)], fill=INK)
    head(points[-2], points[-1])
    if both:
        head(points[1], points[0])


def render_png():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for p in panels:
        d.rounded_rectangle((p['x'],p['y'],p['x']+p['w'],p['y']+p['h']),
                            radius=14, fill=p['fill'], outline="#667786", width=2)
        d.text((p['x']+16,p['y']+14), p['text'], font=font(23, True), fill=INK)
    for e in edges:
        draw_arrow(d, e['points'], e['both'], e['dashed'])
    for n in nodes:
        x,y,w,h = (n[k] for k in ('x','y','w','h'))
        d.rounded_rectangle((x,y,x+w,y+h), radius=12, fill="#121719", outline=BORDER, width=2)
        if n['future']:
            # Dashed upper/lower borders visibly distinguish future modules.
            for xx in range(x+14, x+w-14, 14):
                d.line((xx,y,min(xx+7,x+w-14),y), fill=INK, width=3)
                d.line((xx,y+h,min(xx+7,x+w-14),y+h), fill=INK, width=3)
        f, lines, lh = fitted(d,n['text'],w-18,h-12,n['size'])
        top = y+(h-len(lines)*lh)/2
        for i,line in enumerate(lines):
            d.text((x+w/2,top+i*lh),line,font=f,fill=INK,anchor="mt")
    for n in notes:
        f, lines, lh = fitted(d,n['text'],n['w'],n['h'],n['size'],n['id']=='title')
        for i,line in enumerate(lines):
            d.text((n['x'],n['y']+i*lh),line,font=f,fill=INK)
    img.save(OUT.with_suffix('.png'))


def render_drawio():
    file = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17', type='device')
    diagram = ET.SubElement(file,'diagram',id='architecture-current',name='Architecture hiện tại')
    model = ET.SubElement(diagram,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',
        page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),background=BG)
    root=ET.SubElement(model,'root')
    ET.SubElement(root,'mxCell',id='0'); ET.SubElement(root,'mxCell',id='1',parent='0')
    def vertex(n, style, value=None):
        c=ET.SubElement(root,'mxCell',id=n['id'],value=value if value is not None else n['text'].replace('\n','<br>'),
                        style=style,vertex='1',parent='1')
        ET.SubElement(c,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),**{'as':'geometry'})
    vertex(dict(id='background',x=0,y=0,w=W,h=H,text=''),f'fillColor={BG};strokeColor=none;')
    for p in panels:
        vertex(p,f'rounded=1;arcSize=6;html=1;fillColor={p["fill"]};strokeColor=#667786;strokeWidth=2;'
            f'fontColor={INK};fontSize=23;fontStyle=1;align=left;verticalAlign=top;spacingLeft=16;spacingTop=14;')
    lookup={n['id']:n for n in nodes}
    for e in edges:
        a,b=lookup[e['source']],lookup[e['target']]
        pts=e['points']; sx=(pts[0][0]-a['x'])/a['w']; sy=(pts[0][1]-a['y'])/a['h']
        tx=(pts[-1][0]-b['x'])/b['w']; ty=(pts[-1][1]-b['y'])/b['h']
        style=f'edgeStyle=none;rounded=0;strokeColor={INK};strokeWidth=2;endArrow=block;endFill=1;'
        style+=f'exitX={sx};exitY={sy};exitPerimeter=0;entryX={tx};entryY={ty};entryPerimeter=0;'
        if e['both']:style+='startArrow=block;startFill=1;'
        if e['dashed']:style+='dashed=1;'
        c=ET.SubElement(root,'mxCell',id=e['id'],style=style,edge='1',parent='1',source=e['source'],target=e['target'])
        geo=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        if len(pts)>2:
            arr=ET.SubElement(geo,'Array',**{'as':'points'})
            for x,y in pts[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    for n in nodes:
        vertex(n,f'rounded=1;arcSize=12;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;'
            f'fillColor=#121719;strokeColor={BORDER};strokeWidth=2;fontColor={INK};fontSize={n["size"]};spacing=8;'
            + ('dashed=1;' if n['future'] else ''))
    for n in notes:
        vertex(n,f'text;whiteSpace=wrap;html=1;align=left;verticalAlign=top;fontColor={INK};fontSize={n["size"]};'
            + ('fontStyle=1;' if n['id']=='title' else ''))
    ET.indent(file)
    ET.ElementTree(file).write(OUT.with_suffix('.drawio'),encoding='utf-8',xml_declaration=True)


if __name__ == '__main__':
    render_drawio()
    render_png()
    print(f'Generated {OUT.with_suffix(".drawio")} and .png; {len(edges)} orthogonal connectors.')
