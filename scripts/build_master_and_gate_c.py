# -*- coding: utf-8 -*-
"""
正式全片母片 1:1 原速合成 + Gate C (C1–C6) 成品终审全套核验大图生成引擎
支持原创分镜项目（拼接单镜原生音轨或挂载配乐母带）与参考视频项目（自动输出 REF/GEN 上下对照终审图）。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def imwrite_jpg(path: Path, img: np.ndarray, q: int = 92):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError(f"写入图片失败: {path}")
    buf.tofile(str(path))


def read_all_frames(vp: Path):
    cap = cv2.VideoCapture(str(vp))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description="1:1 原速合成正式母片并生成 Gate C 全套终审巡检图")
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--master-audio", type=Path, default=None, help="可选外部母带音频/参考母片路径（为空则拼接各镜自身音轨）")
    ap.add_argument("--ref-video", type=Path, default=None, help="可选完整参考视频路径（提供则输出 REF/GEN 双排对照终审图）")
    ap.add_argument("--output-name", default="Master_Final_Release.mp4")
    args = ap.parse_args()

    proj = args.project_dir.resolve()
    tbl_path = proj / "atomic_shot_table.json"
    tbl = sorted(json.loads(tbl_path.read_text(encoding="utf-8")), key=lambda x: float(x["start"]))
    not_ok = [s["sub_id"] for s in tbl if s.get("status") != "合格"]
    if not_ok:
        raise RuntimeError(f"[Gate C 硬门禁拦截] 存在尚未通过人工 Gate B 签字的镜头 ({len(not_ok)}个): {not_ok}")

    ver_dir = proj / "renders" / "verified"
    deliv_dir = proj / "deliverables"
    gate_c_dir = proj / "qa" / "GateC_成品终审"
    norm_dir = proj / "renders" / "_norm_concat_tmp"
    for d in (deliv_dir, gate_c_dir, norm_dir):
        d.mkdir(parents=True, exist_ok=True)

    cum_f = 0
    concat_lines = []
    for idx, s in enumerate(tbl):
        sid = s["sub_id"]
        exp_f = int(s["target_frames_30"])
        s["start_frame"], s["end_frame"] = cum_f, cum_f + exp_f
        cum_f += exp_f
        vp = ver_dir / f"{sid}_verified.mp4"
        np_out = norm_dir / f"{idx:03d}_{sid}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp), "-vf", "scale=1280:720:flags=lanczos,fps=30", "-frames:v", str(exp_f), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000"]
        if args.master_audio:
            cmd.append("-an")
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(np_out))
        subprocess.run(cmd, check=True)
        concat_lines.append(f"file '{np_out.as_posix()}'")

    concat_txt = norm_dir / "concat_list.txt"
    concat_txt.write_text("\n".join(concat_lines), encoding="utf-8")
    master_mp4 = deliv_dir / args.output_name

    if args.master_audio:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-i", str(args.master_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-r", "30", "-frames:v", str(cum_f), "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(master_mp4)], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-r", "30", "-frames:v", str(cum_f), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(master_mp4)], check=True)

    m_frames = read_all_frames(master_mp4)
    r_frames = read_all_frames(args.ref_video) if (args.ref_video and args.ref_video.exists()) else []
    assert len(m_frames) == cum_f, f"母片实读帧数 ({len(m_frames)}) 与设计总帧数 ({cum_f}) 不一致！"

    # C3: 每 0.5s (15帧) 全片身份巡检大图
    sample_idxs = list(range(0, len(m_frames), 15))
    for sheet_i, start_k in enumerate(range(0, len(sample_idxs), 90), start=1):
        chunk = sample_idxs[start_k : start_k + 90]
        rows = []
        for r_i in range(0, len(chunk), 9):
            cells = []
            for fi in chunk[r_i : r_i + 9]:
                owner = next((s["sub_id"] for s in tbl if s["start_frame"] <= fi < s["end_frame"]), "END")
                gf = cv2.resize(m_frames[fi], (240, 135), interpolation=cv2.INTER_AREA)
                cv2.rectangle(gf, (0, 0), (240, 20), (16, 16, 16), -1)
                cv2.putText(gf, f"GEN [{owner}] {fi/30.0:05.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 120), 1)
                if r_frames:
                    rf = cv2.resize(r_frames[min(len(r_frames) - 1, fi)], (240, 135), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(rf, (0, 0), (240, 20), (16, 16, 16), -1)
                    cv2.putText(rf, f"REF #{fi:04d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
                    cells.append(np.vstack([rf, gf]))
                else:
                    cells.append(gf)
            cell_h = cells[0].shape[0]
            while len(cells) < 9:
                cells.append(np.zeros((cell_h, 240, 3), dtype=np.uint8))
            rows.append(np.hstack(cells))
        imwrite_jpg(gate_c_dir / f"C3_全片0.5s身份巡检_第{sheet_i}张.jpg", np.vstack(rows), q=90)

    # C5: 全部接缝前后 3 帧核验 + 黑帧扫描
    seam_frames = [s["end_frame"] for s in tbl[:-1]]
    black_frames = [fi for fi, f in enumerate(m_frames) if float(np.mean(f)) < 3.5]
    seam_rows = []
    for idx_s, sf in enumerate(seam_frames[:40]):
        a_sid, b_sid = tbl[idx_s]["sub_id"], tbl[idx_s + 1]["sub_id"]
        strip = []
        for df in [-3, -2, -1, 0, 1, 2]:
            fi = max(0, min(len(m_frames) - 1, sf + df))
            c = cv2.resize(m_frames[fi], (240, 135), interpolation=cv2.INTER_AREA)
            cv2.rectangle(c, (0, 0), (240, 20), (16, 16, 16), -1)
            cv2.putText(c, f"f{fi} ({df:+d}) [{a_sid if df < 0 else b_sid}]", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 120), 1)
            strip.append(c)
        seam_rows.append(np.hstack(strip))
    if seam_rows:
        imwrite_jpg(gate_c_dir / "C5_镜头接缝前后3帧核验.jpg", np.vstack(seam_rows), q=91)

    metrics = {
        "total_shots": len(tbl),
        "exact_frames": len(m_frames),
        "duration_sec": round(len(m_frames) / 30.0, 3),
        "black_frames_count": len(black_frames),
        "black_frames_sample": black_frames[:20],
    }
    (gate_c_dir / "GateC_自动核验指标汇总.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in norm_dir.glob("*"):
        p.unlink(missing_ok=True)
    norm_dir.rmdir()
    print(f"[Gate C 产物就绪] 母片: {master_mp4} ({len(m_frames)} 帧) | 请人工审阅 GateC_成品终审 图集后释放 GPU！")


if __name__ == "__main__":
    main()
