"""
H3 Pipeline 兼容与显存/内存加速节点包：
- H3SigmaRefiner：在 [end_at_sigma, start_at_sigma] 区间内按指定间隔重采样，额外插入 extra_steps 步（加密低噪声尾段，供高分辨率二段精修）
- DT_MediaLoader / DT_MediaSplitter：按 media_state 从 input 目录加载图片/视频/音频并拆分输出
- H3SaveStage1 / H3LoadStage1：两段式工作流检查点节点（Stage 1 仅跑 2 分钟 752x416 低清动作与音轨初检并保存 conditioning + 音视频潜空间；验收通过后 Stage 2 免加载 32GB Qwen3VL 直接读取检查点进行 3D 潜空间超分与 5 步高清精修）
- 自动内存生命周期钩子：
  1. SamplerCustomAdvanced 启动前自动清空已用完的 32GB Qwen3VL 文本编码器系统内存，避免与 20GB MiniMaxH3 同时驻留内存；
  2. VAEDecodeTiled 高清分块解码前自动清空 20GB MiniMaxH3 系统内存，确保在 L4 (53GB RAM) 上留出 40GB+ 空闲内存完成 1856x1024 解码。
"""
import gc, json, math, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if hasattr(obj, "tensors") and hasattr(obj, "to"):
        try:
            return obj.detach().cpu()
        except Exception:
            return type(obj)([_to_cpu(t) for t in obj.tensors])
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


class H3SigmaRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "extra_steps": ("INT", {"default": 1, "min": 0, "max": 64}),
            "start_at_sigma": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "end_at_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "spacing": (["cosine", "linear", "karras"], {"default": "cosine"}),
        }}
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "refine"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def refine(self, sigmas, extra_steps, start_at_sigma, end_at_sigma, spacing):
        s = sigmas.detach().float().cpu()
        if extra_steps <= 0 or s.numel() < 2:
            return (sigmas,)
        hi, lo = max(start_at_sigma, end_at_sigma), min(start_at_sigma, end_at_sigma)
        idx = [i for i, v in enumerate(s.tolist()) if lo - 1e-6 <= v <= hi + 1e-6]
        if len(idx) < 2:
            return (sigmas,)
        a, b = idx[0], idx[-1]
        s_hi, s_lo = float(s[a]), float(s[b])
        n = (b - a) + extra_steps
        t = torch.linspace(0, 1, n + 1)
        if spacing == "cosine":
            w = (1 - torch.cos(t * math.pi)) / 2
            seg = s_hi + (s_lo - s_hi) * w
        elif spacing == "karras" and s_lo > 0:
            rho = 7.0
            seg = (s_hi ** (1 / rho) + t * (s_lo ** (1 / rho) - s_hi ** (1 / rho))) ** rho
        else:
            seg = s_hi + (s_lo - s_hi) * t
        out = torch.cat([s[:a], seg, s[b + 1:]])
        return (out.to(sigmas.device, sigmas.dtype),)


class H3SaveStage1:
    """保存一阶段 conditioning 与分离后的 video_latent / audio_latent 到 output 目录，供 Stage 2 超分直接复用"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "video_latent": ("LATENT",),
            "audio_latent": ("LATENT",),
            "ckpt_name": ("STRING", {"default": "stage1_latest"}),
        }}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax-H3/checkpoint"

    def save(self, conditioning, video_latent, audio_latent, ckpt_name):
        out_dir = Path(folder_paths.get_output_directory())
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c for c in ckpt_name if c.isalnum() or c in ("_", "-")) or "stage1_latest"
        ckpt_path = out_dir / f"{safe_name}_stage1.pt"
        payload = {
            "conditioning": _to_cpu(conditioning),
            "video_latent": _to_cpu(video_latent),
            "audio_latent": _to_cpu(audio_latent),
        }
        torch.save(payload, ckpt_path)
        print(f"[h3_pipeline_compat] Saved Stage 1 checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return {"ui": {"ckpt": [str(ckpt_path)]}}


class H3LoadStage1:
    """从 output 目录加载 Stage 1 保存的 conditioning、video_latent、audio_latent，跳过 32GB Qwen3VL 与一阶段采样"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name": ("STRING", {"default": "stage1_latest"}),
        }}
    RETURN_TYPES = ("CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("conditioning", "video_latent", "audio_latent")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/checkpoint"

    def load(self, ckpt_name):
        out_dir = Path(folder_paths.get_output_directory())
        safe_name = "".join(c for c in ckpt_name if c.isalnum() or c in ("_", "-")) or "stage1_latest"
        ckpt_path = out_dir / f"{safe_name}_stage1.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到 Stage 1 检查点文件: {ckpt_path}，请先运行 --stage 1")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"[h3_pipeline_compat] Loaded Stage 1 checkpoint: {ckpt_path}")
        return (payload["conditioning"], payload["video_latent"], payload["audio_latent"])


def _load_image(name):
    p = folder_paths.get_annotated_filepath(name)
    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]


class DT_MediaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"media_state": ("STRING", {"default": '{"images":[],"audios":[],"videos":[]}', "multiline": True})}}
    RETURN_TYPES = ("DT_MEDIA_BUNDLE",)
    RETURN_NAMES = ("media_bundle",)
    FUNCTION = "load"
    CATEGORY = "DT"

    def load(self, media_state):
        st = json.loads(media_state or "{}")
        return ({k: [x.get("filename") if isinstance(x, dict) else x for x in st.get(k, [])] for k in ("images", "videos", "audios")},)


class DT_MediaSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media_bundle": ("DT_MEDIA_BUNDLE",),
            "image_count": ("INT", {"default": 9, "min": 0, "max": 9}),
            "video_count": ("INT", {"default": 3, "min": 0, "max": 3}),
            "audio_count": ("INT", {"default": 3, "min": 0, "max": 3}),
            "empty_output_mode": ("STRING", {"default": "空值"}),
        }}
    RETURN_TYPES = ("IMAGE",) * 9 + ("VIDEO",) * 3 + ("AUDIO",) * 3
    FUNCTION = "split"
    CATEGORY = "DT"

    def split(self, media_bundle, image_count, video_count, audio_count, empty_output_mode):
        imgs = [_load_image(n) for n in media_bundle.get("images", [])[:image_count]]
        vids, auds = [], []
        try:
            from comfy_api.input_impl import VideoFromFile
            vids = [VideoFromFile(folder_paths.get_annotated_filepath(n)) for n in media_bundle.get("videos", [])[:video_count]]
        except Exception:
            pass
        try:
            import torchaudio
            for n in media_bundle.get("audios", [])[:audio_count]:
                w, sr = torchaudio.load(folder_paths.get_annotated_filepath(n))
                auds.append({"waveform": w[None], "sample_rate": sr})
        except Exception:
            pass
        pad = lambda xs, k: (xs + [None] * k)[:k]
        return tuple(pad(imgs, 9) + pad(vids, 3) + pad(auds, 3))


NODE_CLASS_MAPPINGS = {
    "H3SigmaRefiner": H3SigmaRefiner,
    "DT_MediaLoader": DT_MediaLoader,
    "DT_MediaSplitter": DT_MediaSplitter,
    "H3SaveStage1": H3SaveStage1,
    "H3LoadStage1": H3LoadStage1,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SigmaRefiner": "H3 Sigma Refiner (Pipeline compat)",
    "DT_MediaLoader": "dt统一素材加载 (Pipeline compat)",
    "DT_MediaSplitter": "dt素材拆分 (Pipeline compat)",
    "H3SaveStage1": "H3 Save Stage1 Checkpoint",
    "H3LoadStage1": "H3 Load Stage1 Checkpoint",
}

# 自动内存管理钩子（仅在系统内存 < 68GB 的小内存机器如 L4 上启用销毁；A100 83.5GB RAM / H100 / PRO 6000 全程常驻内存/显存，绝不销毁重读硬盘！）
try:
    import psutil
    import nodes
    import comfy.model_management as mm
    import comfy_extras.nodes_custom_sampler as ncs

    _sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[h3_pipeline_compat] System RAM: {_sys_ram_gb:.1f} GB | GPU VRAM: {_vram_gb:.1f} GB")

    def _purge_models_from_ram(keep_patcher=None, purge_diffusion=False):
        if _sys_ram_gb >= 68.0 or _vram_gb >= 38.0:
            # A100 (40GB VRAM + 83.5GB RAM) / H100 / PRO 6000：绝不销毁模型参数，保持 100% 内存/显存热驻留！
            return
        for lm in list(mm.current_loaded_models):
            try:
                if lm.model is keep_patcher:
                    continue
                m_obj = getattr(lm.model, "model", None)
                cls_name = type(m_obj).__name__ if m_obj is not None else ""
                if "VAE" in cls_name:
                    continue
                if not purge_diffusion and "MiniMaxH3" in cls_name and "Qwen" not in cls_name:
                    continue
                lm.model_unload()
                if m_obj is not None:
                    for sub in m_obj.modules():
                        sub._parameters.clear()
                        sub._buffers.clear()
                mm.current_loaded_models.remove(lm)
                print(f"[h3_pipeline_compat] Purged {cls_name} from CPU RAM")
            except Exception as e:
                print("[h3_pipeline_compat] unload err:", e)
        gc.collect()
        mm.soft_empty_cache()

    # 1. 采样开始前立刻清空 32GB Qwen3VL 文本编码器，使其绝不与 20GB MiniMaxH3 同时驻留内存
    if not getattr(ncs.SamplerCustomAdvanced, "_alex_hooked", False):
        _orig_sample = ncs.SamplerCustomAdvanced.sample
        def _patched_sample(self, noise, guider, sampler, sigmas, latent_image):
            _purge_models_from_ram(keep_patcher=getattr(guider, "model_patcher", None), purge_diffusion=False)
            return _orig_sample(self, noise, guider, sampler, sigmas, latent_image)
        ncs.SamplerCustomAdvanced.sample = _patched_sample
        ncs.SamplerCustomAdvanced._alex_hooked = True

    # 2. VAEDecode（一阶段低清解码）：仅确保 Qwen3VL 已清空，保留 MiniMaxH3 以备二阶段，并确保输出 Tensor 在 CPU 上供 VHS_VideoCombine 使用
    if not getattr(nodes.VAEDecode, "_alex_hooked", False):
        _orig_decode = nodes.VAEDecode.decode
        def _patched_decode(self, vae, samples):
            _purge_models_from_ram(keep_patcher=getattr(vae, "patcher", None), purge_diffusion=False)
            return _to_cpu(_orig_decode(self, vae, samples))
        nodes.VAEDecode.decode = _patched_decode
        nodes.VAEDecode._alex_hooked = True

    # 3. VAEDecodeTiled（二阶段 1856x1024 高清解码）：把 20GB MiniMaxH3 一并清空，留出 40GB+ 空闲内存
    if not getattr(nodes.VAEDecodeTiled, "_alex_hooked", False):
        _orig_decode_tiled = nodes.VAEDecodeTiled.decode
        def _patched_decode_tiled(self, vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
            _purge_models_from_ram(keep_patcher=getattr(vae, "patcher", None), purge_diffusion=True)
            return _to_cpu(_orig_decode_tiled(self, vae, samples, tile_size, overlap=overlap, temporal_size=temporal_size, temporal_overlap=temporal_overlap))
        nodes.VAEDecodeTiled.decode = _patched_decode_tiled
        nodes.VAEDecodeTiled._alex_hooked = True
except Exception as _e:
    print("[h3_pipeline_compat] RAM hook skipped:", _e)
