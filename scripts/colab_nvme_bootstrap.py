# -*- coding: utf-8 -*-
"""
Colab 云端 GPU 极速唤醒与本地 NVMe 直拉部署脚本
核心原理：放弃慢速 Google Drive 挂载读取权重，直接通过机房超千兆内网 aria2c 16 线程并发拉取模型至 /content 本地 NVMe 高速固态盘。
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPAT_FILE = REPO_ROOT / "custom_nodes" / "h3_pipeline_compat" / "__init__.py"

COLAB_BOOTSTRAP_TEMPLATE = r"""import os, subprocess, threading, time, json, urllib.request
from pathlib import Path
from google.colab import runtime

print("[1/4] 正在核验 GPU 运行时...", flush=True)
_gpu = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True, capture_output=True, text=True)
if _gpu.returncode != 0 or not _gpu.stdout.strip():
    raise SystemExit('当前是 CPU 运行时，请切换至 A100 / H100 / RTX PRO 6000 GPU！')
print('GPU:', _gpu.stdout.strip(), flush=True)

LOCAL_COMFY = Path('/content/ComfyUI')
subprocess.run('apt-get update -qq && apt-get install -y -qq aria2 lz4', shell=True, check=True)

print("[2/4] 正在本地高速盘部署 ComfyUI 与节点环境...", flush=True)
if not (LOCAL_COMFY / 'main.py').exists():
    subprocess.run('git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI', shell=True, check=True)
NODE_REPOS = [
    'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git',
    'https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git',
    'https://github.com/kijai/ComfyUI-KJNodes.git',
    'https://github.com/kijai/ComfyUI-SolAttn_triton.git'
]
for r in NODE_REPOS:
    d = LOCAL_COMFY / 'custom_nodes' / r.split('/')[-1].replace('.git', '')
    if not d.exists():
        subprocess.run(f'git clone -q --depth 1 {r} "{d}"', shell=True, check=True)

subprocess.run('pip install -q -r /content/ComfyUI/requirements.txt comfy-kitchen', shell=True, check=True)
subprocess.run('for f in /content/ComfyUI/custom_nodes/*/requirements.txt; do pip install -q -r "$f"; done', shell=True)

_compat = LOCAL_COMFY / 'custom_nodes' / 'h3_pipeline_compat'
_compat.mkdir(parents=True, exist_ok=True)
(_compat / '__init__.py').write_text(COMPAT_SRC, encoding='utf-8')

import torch, safetensors
MODELS = [
    ('diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors', 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors'),
    ('text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
    ('loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'),
    ('vae/minimax_h3_video_vae_fp16.safetensors', 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors'),
    ('vae/minimax_h3_audio_vae_fp32.safetensors', 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors'),
    ('latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors', 'https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors'),
]

def valid_safetensor(p):
    if not p.exists() or p.stat().st_size < 1024:
        return False
    try:
        with safetensors.safe_open(str(p), framework="pt", device="cpu") as f:
            return len(f.keys()) > 0
    except Exception:
        return False

print("[3/4] 正在通过机房超千兆内网 16 线程直拉模型至本地 NVMe 固态盘...", flush=True)
for rel, url in MODELS:
    dst = LOCAL_COMFY / 'models' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not valid_safetensor(dst):
        if dst.exists(): dst.unlink()
        subprocess.run(f'aria2c --console-log-level=error --allow-overwrite=true --auto-file-renaming=false --file-allocation=none -x 16 -s 16 -k 1M "{url}" -d "{dst.parent}" -o "{dst.name}"', shell=True, check=True)
        assert valid_safetensor(dst), f"权重校验失败: {rel}"
    print(f"模型就绪: {rel} ({dst.stat().st_size/1e9:.2f} GB)", flush=True)

def watchdog():
    last = time.time()
    last_hist_cnt = -1
    while True:
        time.sleep(20)
        try:
            q = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5).read())
            h = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/history', timeout=5).read())
            h_cnt = len(h) if isinstance(h, dict) else 0
            if q.get('queue_running') or q.get('queue_pending') or (last_hist_cnt >= 0 and h_cnt != last_hist_cnt):
                last = time.time()
            last_hist_cnt = h_cnt
        except Exception:
            pass
        if time.time() - last > 3600:
            runtime.unassign()
            break
threading.Thread(target=watchdog, daemon=True).start()

print("[4/4] 正在启动 Cloudflare 隧道与 ComfyUI 服务...", flush=True)
subprocess.run('wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared', shell=True)
tun = subprocess.Popen(['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8188'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
threading.Thread(target=lambda: [print('\\nTUNNEL_URL:', l.strip(), '\\n', flush=True) for l in tun.stdout if '.trycloudflare.com' in l], daemon=True).start()
_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
_vram_flag = '--highvram --disable-dynamic-vram' if _vram_gb >= 35 else '--cache-none'
subprocess.run(f'python /content/ComfyUI/main.py --listen 127.0.0.1 --port 8188 {_vram_flag}', shell=True)
"""


def cdp(bridge_url: str, method: str, params=None, sid=None, timeout=30):
    body = {"method": method, "params": params or {}}
    if sid:
        body["sessionId"] = sid
    req = urllib.request.Request(bridge_url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ev(bridge_url: str, sid: str, js: str):
    res = cdp(bridge_url, "Runtime.evaluate", {"expression": js, "returnByValue": True}, sid=sid)
    return res.get("result", {}).get("result", {}).get("value")


def main():
    ap = argparse.ArgumentParser(description="极速唤醒固定 Colab 笔记本并部署本地 NVMe ComfyUI")
    ap.add_argument("--notebook-id", default=os.environ.get("COLAB_NOTEBOOK_ID", ""), required=False, help="固定复用的 Colab Notebook Drive ID")
    ap.add_argument("--tunnel-file", type=pathlib.Path, default=pathlib.Path("current_tunnel.txt"), help="保存输出隧道 URL 的文件路径")
    ap.add_argument("--bridge-url", default="http://127.0.0.1:19222", help="本地 CDP 单例桥地址")
    args = ap.parse_args()
    if not args.notebook_id:
        raise SystemExit("请通过 --notebook-id 或环境变量 COLAB_NOTEBOOK_ID 指定你的固定 Colab 笔记本 ID")

    full_code = "COMPAT_SRC = " + json.dumps(COMPAT_FILE.read_text(encoding="utf-8"), ensure_ascii=False) + "\n" + COLAB_BOOTSTRAP_TEMPLATE
    targets = cdp(args.bridge_url, "Target.getTargets").get("result", {}).get("targetInfos", [])
    colab_tid = next((t["targetId"] for t in targets if t.get("type") == "page" and args.notebook_id in t.get("url", "")), None)

    if not colab_tid:
        cr = cdp(args.bridge_url, "Target.createTarget", {"url": f"https://colab.research.google.com/drive/{args.notebook_id}", "background": True})
        colab_tid = cr["result"]["targetId"]
        sid_tmp = cdp(args.bridge_url, "Target.attachToTarget", {"targetId": colab_tid, "flatten": True})["result"]["sessionId"]
        for _ in range(30):
            time.sleep(2)
            if ev(args.bridge_url, sid_tmp, "Boolean(window.colab && window.colab.global && window.colab.global.notebook && window.colab.global.notebook.cells.length > 0)"):
                break

    sid = cdp(args.bridge_url, "Target.attachToTarget", {"targetId": colab_tid, "flatten": True})["result"]["sessionId"]
    inj_js = f"""(() => {{
        const c = window.colab.global.notebook.cells[0];
        if (c.clearOutputs) c.clearOutputs();
        c.setText({json.dumps(full_code)});
        const b = c.runButton;
        const el = b && (b.click ? b : (b.element || (b.getElement && b.getElement())));
        if (el) {{ el.click(); return 'STARTED'; }}
        return 'NO_BTN';
    }})()"""
    print("[启动注入]", ev(args.bridge_url, sid, inj_js), flush=True)

    for _ in range(220):
        time.sleep(5)
        raw = ev(args.bridge_url, sid, """(() => {
            const c = window.colab.global.notebook.cells[0];
            const outs = (c.getModel ? c.getModel().toJson().outputs : []) || [];
            return outs.map(o => Array.isArray(o.text) ? o.text.join('') : (o.text || '')).join('\\n').slice(-1500);
        })()""") or ""
        m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", raw)
        if m:
            url = m[-1]
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{url}/system_stats", headers={"User-Agent": "Mozilla/5.0"}), timeout=6) as r:
                    if r.status == 200:
                        args.tunnel_file.parent.mkdir(parents=True, exist_ok=True)
                        args.tunnel_file.write_text(url, encoding="utf-8")
                        print(f"[隧道就绪] {url}", flush=True)
                        return
            except Exception:
                pass
    raise RuntimeError("Colab ComfyUI 启动超时")


if __name__ == "__main__":
    main()
