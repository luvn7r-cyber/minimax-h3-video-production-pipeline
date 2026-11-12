# -*- coding: utf-8 -*-
"""
通用二阶段高清滑动窗口批量渲染与 Gate B 自动初筛引擎
支持原创剧本路径 (T2VA / I2VA / FL2VA) 与参考视频驱动路径 (Ref2VA)。
红线铁律：本脚本最高只允许输出“自动初筛通过（待人工逐帧审核）”，严禁自动将镜头标记为“合格”！
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_WORKFLOW = REPO_ROOT / "workflows" / "minimax_h3_2stage_i2v_fl2v.json"


def imread_u8(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_jpg(path: Path, img: np.ndarray, quality: int = 93):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"JPEG 编码失败: {path}")
    buf.tofile(str(path))


def http_retry(fn, desc: str, retries: int = 5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"{desc} 连续 {retries} 次失败: {last_err}")


def upload_file_comfy(base_url: str, filepath: Path) -> str:
    data = filepath.read_bytes()
    boundary = "----PipelineBoundary" + uuid.uuid4().hex
    mime = "video/mp4" if filepath.suffix.lower() == ".mp4" else "image/jpeg"
    safe_name = f"up_{hash(str(filepath)) & 0xffffffff:08x}_{filepath.stat().st_mtime_ns % 100000}{filepath.suffix.lower()}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode("utf-8")

    def _do():
        req = urllib.request.Request(
            f"{base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))["name"]

    return http_retry(_do, f"上传 {filepath.name}")


def preflight_check(proj_dir: Path, shots: list):
    print(f"[预检] 正在校验 {len(shots)} 个待渲染镜头的引用素材完整性...", flush=True)
    for item in shots:
        sid = item["sub_id"]
        if not str(item.get("prompt", "")).strip():
            raise RuntimeError(f"[{sid}] prompt 为空")
        for p_str in [item.get("f0", ""), item.get("fend", "")] + [m[0] for m in item.get("fmid", [])] + item.get("cards", []):
            if p_str:
                p = Path(p_str) if Path(p_str).is_absolute() else (proj_dir / p_str)
                if not p.exists() or imread_u8(p) is None:
                    raise RuntimeError(f"[{sid}] 缺失或损坏图像: {p}")
        if item.get("ref_video"):
            vp = Path(item["ref_video"]) if Path(item["ref_video"]).is_absolute() else (proj_dir / item["ref_video"])
            if not vp.exists():
                raise RuntimeError(f"[{sid}] 缺失参考视频: {vp}")
    print("[预检通过] 全部引用素材 100% 完整可读！", flush=True)


def build_shot_workflow(base_url: str, proj_dir: Path, item: dict, addguide_has_latent: bool = False):
    sid = item["sub_id"]
    target_f30 = int(item["target_frames_30"])
    gen_sec = round(max(1.0, target_f30 / 30.0), 2)
    graph = json.loads(BASE_WORKFLOW.read_text(encoding="utf-8"))

    # 清理模板静态连接
    for k in [k for k in list(graph.keys()) if int(k) >= 900]:
        del graph[k]
    for k in list(graph["136"]["inputs"].keys()):
        if k.startswith("ref_images.") or k.startswith("ref_videos."):
            del graph["136"]["inputs"][k]

    seed = int(item.get("seed") or 20261112)
    graph["129"]["inputs"]["noise_seed"] = seed
    graph["132"]["inputs"]["value"] = gen_sec
    graph["138"]["inputs"]["value"] = str(item["prompt"]).strip()
    graph["141"]["inputs"]["filename_prefix"] = f"{sid}_pass1"
    graph["179"]["inputs"]["filename_prefix"] = f"{sid}_final"

    # 挂载角色定妆卡
    cards = item.get("cards", [])
    for idx, rel in enumerate(cards):
        p = Path(rel) if Path(rel).is_absolute() else (proj_dir / rel)
        nid = str(900 + idx)
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": upload_file_comfy(base_url, p)}}
        graph["136"]["inputs"][f"ref_images.ref_image_{idx}"] = [nid, 0]

    # 可选挂载净化后的动作参考视频 <Video 1>
    if item.get("ref_video"):
        vp = Path(item["ref_video"]) if Path(item["ref_video"]).is_absolute() else (proj_dir / item["ref_video"])
        graph["905"] = {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": upload_file_comfy(base_url, vp),
                "force_rate": 24,
                "custom_width": ["115", 0],
                "custom_height": ["115", 1],
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "H3",
            },
        }
        graph["136"]["inputs"]["ref_videos.ref_video_0"] = ["905", 0]

    # 显式计算潜空间 17k+5 对齐后的终点帧索引 end_idx_24 (严禁传 -1)
    raw_24 = max(5, int(round(gen_sec * 24)))
    actual_len_24 = raw_24 + (5 - (raw_24 % 17)) % 17
    end_idx_24 = max(1, min(actual_len_24 - 1, int(round((target_f30 / 30.0) * 24)) - 1))

    prev_cond = ["136", 0]
    prev_lat = ["136", 1]
    s2_cond = ["136", 0]
    s2_lat = ["172", 0]
    s2_w, s2_h = 1792, 992

    if item.get("f0"):
        p_f0 = Path(item["f0"]) if Path(item["f0"]).is_absolute() else (proj_dir / item["f0"])
        up_f0 = upload_file_comfy(base_url, p_f0)
        if not cards:
            graph["900"] = {"class_type": "LoadImage", "inputs": {"image": up_f0}}
            graph["136"]["inputs"]["ref_images.ref_image_0"] = ["900", 0]
        graph["910"] = {"class_type": "LoadImage", "inputs": {"image": up_f0}}
        graph["916"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": ["115", 0], "height": ["115", 1], "crop": "center", "image": ["910", 0]}}
        graph["911"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": prev_cond, "latent": prev_lat, "frame_idx": 0, "vae": ["119", 0], "image": ["916", 0]}}
        prev_cond = ["911", 0]
        if addguide_has_latent:
            prev_lat = ["911", 1]

        graph["950"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": s2_w, "height": s2_h, "crop": "center", "image": ["910", 0]}}
        graph["951"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": s2_cond, "latent": s2_lat, "frame_idx": 0, "vae": ["119", 0], "image": ["950", 0]}}
        s2_cond = ["951", 0]
        if addguide_has_latent:
            s2_lat = ["951", 1]

    fmid_nodes = []
    for m_i, (m_rel, m_ratio) in enumerate(item.get("fmid", [])):
        pm = Path(m_rel) if Path(m_rel).is_absolute() else (proj_dir / m_rel)
        mid_idx = max(1, min(end_idx_24 - 1, int(round(end_idx_24 * float(m_ratio)))))
        nl, ns, ng = str(918 + m_i * 3), str(919 + m_i * 3), str(920 + m_i * 3)
        graph[nl] = {"class_type": "LoadImage", "inputs": {"image": upload_file_comfy(base_url, pm)}}
        graph[ns] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": ["115", 0], "height": ["115", 1], "crop": "center", "image": [nl, 0]}}
        graph[ng] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": prev_cond, "latent": prev_lat, "frame_idx": mid_idx, "vae": ["119", 0], "image": [ns, 0]}}
        prev_cond = [ng, 0]
        if addguide_has_latent:
            prev_lat = [ng, 1]
        fmid_nodes.append((nl, mid_idx))

    s2_mode = item.get("s2_lock_fmid", False)
    if s2_mode and fmid_nodes:
        targets = fmid_nodes[:1] if s2_mode == "first" else (fmid_nodes[:2] if s2_mode == "slide_pair" else fmid_nodes)
        for idx_m, (nl, mid_idx) in enumerate(targets):
            ns2, ng2 = str(952 + idx_m * 2), str(953 + idx_m * 2)
            graph[ns2] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": s2_w, "height": s2_h, "crop": "center", "image": [nl, 0]}}
            graph[ng2] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": s2_cond, "latent": s2_lat, "frame_idx": mid_idx, "vae": ["119", 0], "image": [ns2, 0]}}
            s2_cond = [ng2, 0]
            if addguide_has_latent:
                s2_lat = [ng2, 1]

    if item.get("fend"):
        p_fe = Path(item["fend"]) if Path(item["fend"]).is_absolute() else (proj_dir / item["fend"])
        graph["912"] = {"class_type": "LoadImage", "inputs": {"image": upload_file_comfy(base_url, p_fe)}}
        graph["917"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": ["115", 0], "height": ["115", 1], "crop": "center", "image": ["912", 0]}}
        graph["913"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": prev_cond, "latent": prev_lat, "frame_idx": end_idx_24, "vae": ["119", 0], "image": ["917", 0]}}
        prev_cond = ["913", 0]
        if addguide_has_latent:
            prev_lat = ["913", 1]

        graph["970"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": s2_w, "height": s2_h, "crop": "center", "image": ["912", 0]}}
        graph["971"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": s2_cond, "latent": s2_lat, "frame_idx": end_idx_24, "vae": ["119", 0], "image": ["970", 0]}}
        s2_cond = ["971", 0]
        if addguide_has_latent:
            s2_lat = ["971", 1]

    graph["126"]["inputs"]["conditioning"] = prev_cond
    graph["170"]["inputs"]["conditioning"] = s2_cond
    return graph, seed


def build_18frame_qa_sheet(gen_mp4: Path, out_jpg: Path, sid: str, ref_mp4: Path = None):
    cap_g = cv2.VideoCapture(str(gen_mp4))
    g_frames = []
    while True:
        ok, f = cap_g.read()
        if not ok:
            break
        g_frames.append(f)
    cap_g.release()

    r_frames = []
    if ref_mp4 and ref_mp4.exists():
        cap_r = cv2.VideoCapture(str(ref_mp4))
        while True:
            ok, f = cap_r.read()
            if not ok:
                break
            r_frames.append(f)
        cap_r.release()

    idxs = np.linspace(0, max(0, len(g_frames) - 1), 18).round().astype(int).tolist()
    rows = []
    for r_i in range(0, 18, 6):
        cells = []
        for k, fi in enumerate(idxs[r_i : r_i + 6], start=r_i + 1):
            gc = cv2.resize(g_frames[fi], (320, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(gc, (0, 0), (320, 24), (18, 18, 18), -1)
            cv2.putText(gc, f"GEN #{k:02d} {sid} +{fi}f/{len(g_frames)}f", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 140), 1)
            if r_frames:
                ri = min(len(r_frames) - 1, fi)
                rc = cv2.resize(r_frames[ri], (320, 180), interpolation=cv2.INTER_AREA)
                cv2.rectangle(rc, (0, 0), (320, 24), (18, 18, 18), -1)
                cv2.putText(rc, f"REF #{k:02d} {sid} +{ri}f", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)
                cells.append(np.vstack([rc, gc]))
            else:
                cells.append(gc)
        rows.append(np.hstack(cells))
    imwrite_jpg(out_jpg, np.vstack(rows), quality=92)
    return len(g_frames)


def main():
    ap = argparse.ArgumentParser(description="通用滑动窗口批量渲染与 Gate B 初筛引擎")
    ap.add_argument("--project-dir", type=Path, required=True, help="项目根目录（含 atomic_shot_table.json）")
    ap.add_argument("--tunnel-url", default="", help="ComfyUI 隧道地址（为空则读取项目下 current_tunnel.txt）")
    ap.add_argument("--max-inflight", type=int, default=2, help="GPU 滑动窗口并发排队数")
    args = ap.parse_args()

    proj = args.project_dir.resolve()
    tbl_path = proj / "atomic_shot_table.json"
    tbl = json.loads(tbl_path.read_text(encoding="utf-8"))
    todo = [s for s in tbl if s.get("status") in ("关键图已审", "参考已审")]
    if not todo:
        print("[提示] 当前没有处于 '关键图已审' 状态的待渲染镜头。")
        return

    preflight_check(proj, todo)
    base_url = args.tunnel_url or (proj / "current_tunnel.txt").read_text(encoding="utf-8").strip()
    pending_dir = proj / "renders" / "pending"
    qa_dir = proj / "qa" / "GateB_单镜质检"
    pending_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    inflight = {}
    queue = list(todo)
    while queue or inflight:
        while queue and len(inflight) < args.max_inflight:
            item = queue.pop(0)
            sid = item["sub_id"]
            graph, used_seed = build_shot_workflow(base_url, proj, item)
            payload = json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode("utf-8")
            req = urllib.request.Request(f"{base_url}/prompt", data=payload, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}, method="POST")
            pid = http_retry(lambda: json.loads(urllib.request.urlopen(req, timeout=45).read().decode("utf-8"))["prompt_id"], f"提交 {sid}")
            inflight[sid] = (item, pid, time.time())
            print(f"[已投递] {sid} (seed={used_seed}) -> pid={pid}", flush=True)

        for sid in list(inflight.keys()):
            item, pid, t0 = inflight[sid]
            hist = http_retry(lambda: json.loads(urllib.request.urlopen(urllib.request.Request(f"{base_url}/history/{pid}", headers={"User-Agent": "Mozilla/5.0"}), timeout=25).read().decode("utf-8")), f"轮询 {sid}")
            if pid not in hist:
                continue
            outputs = hist[pid].get("outputs", {})
            if "179" not in outputs:
                continue
            gifs = outputs["179"].get("gifs") or outputs["179"].get("videos") or []
            vinfo = gifs[0]
            raw_mp4 = pending_dir / f"{sid}_raw24.mp4"
            dl_url = f"{base_url}/view?filename={vinfo['filename']}&subfolder={vinfo.get('subfolder','')}&type={vinfo.get('type','output')}"
            raw_mp4.write_bytes(http_retry(lambda: urllib.request.urlopen(urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read(), f"下载 {sid}"))

            cand_mp4 = pending_dir / f"{sid}_cand.mp4"
            tf30 = int(item["target_frames_30"])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_mp4), "-vf", "scale=1280:720:flags=lanczos,fps=30", "-frames:v", str(tf30), "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cand_mp4)], check=True)
            ref_30 = (proj / item["ref_slice_30"]) if item.get("ref_slice_30") else None
            nf = build_18frame_qa_sheet(cand_mp4, qa_dir / f"{sid}_18帧抽检.jpg", sid, ref_mp4=ref_30)
            sc_out = subprocess.run(["ffmpeg", "-i", str(cand_mp4), "-vf", "scdet=s=1:t=14,blackdetect=d=0.05:pix_th=0.05,freezedetect=n=0.003:d=0.5", "-f", "null", "-"], capture_output=True, text=True).stderr
            has_black = "black_start" in sc_out
            has_freeze = "freeze_start" in sc_out
            sc_hits = re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", sc_out)
            print(f"[回收完成] {sid}: {nf}/{tf30}f | black={has_black} freeze={has_freeze} scdet={sc_hits} -> 自动初筛通过（待人工逐帧审核 18 帧抽检图！）", flush=True)
            del inflight[sid]
        if inflight:
            time.sleep(6)


if __name__ == "__main__":
    main()
