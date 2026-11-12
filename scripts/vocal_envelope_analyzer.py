# -*- coding: utf-8 -*-
"""
0.25s 步长音频人声能量时间窗分析器
用于从台词/歌曲母带中提取精确发声时段与呼吸停顿时段，驱动视频模型生成同步口型。
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import wave
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def analyze_audio_windows(media_path: Path, step_sec: float = 0.25, threshold_ratio: float = 0.35):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", "-af", "highpass=f=280,lowpass=f=3400", str(wav_path)],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav_path.unlink(missing_ok=True)

    win = int(round(sr * step_sec))
    rms_list = [float(np.sqrt(np.mean(pcm[i : i + win] ** 2) + 1e-12)) for i in range(0, len(pcm), win)]
    peak = float(np.percentile(rms_list, 95)) if rms_list else 1.0
    thr = peak * threshold_ratio

    segments = []
    cur_active = None
    start_t = 0.0
    for idx, r in enumerate(rms_list):
        act = r >= thr
        t = round(idx * step_sec, 2)
        if cur_active is None:
            cur_active, start_t = act, t
        elif act != cur_active:
            segments.append({"start": start_t, "end": t, "vocal_active": cur_active})
            cur_active, start_t = act, t
    if cur_active is not None:
        segments.append({"start": start_t, "end": round(len(rms_list) * step_sec, 2), "vocal_active": cur_active})
    return segments


def main():
    ap = argparse.ArgumentParser(description="提取 0.25s 人声能量时间窗用于精准口型提示词")
    ap.add_argument("input", type=Path, help="输入音频或视频文件路径")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    segs = analyze_audio_windows(args.input, step_sec=args.step)
    txt = json.dumps(segs, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(txt, encoding="utf-8")
    else:
        print(txt)


if __name__ == "__main__":
    main()
