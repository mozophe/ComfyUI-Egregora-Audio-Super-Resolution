import os
import sys
import time
import tempfile
import platform
import sysconfig
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import torch

RETURN_TYPES = ("AUDIO",)
FUNCTION = "run"
CATEGORY = "Egregora/Audio"

# ---------------- I/O helpers ----------------

def _to_cs(x: np.ndarray) -> np.ndarray:
    """Return channels-first float32 [C,S]; accepts [S], [S,C], [C,S]."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        h, w = a.shape
        if w <= 8 and h > w:  # soundfile often returns [S,C]
            a = a.T
    else:
        a = a.reshape(-1)[None, :]
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m > 1.0:  # safety clamp if upstream sent > 1.0
        a = a / (m + 1e-8)
    return a.astype(np.float32)

def _save_temp_wav(cs: np.ndarray, sr: int) -> Path:
    p = Path(tempfile.gettempdir()) / f"eg_in_{int(time.time()*1000)}.wav"
    sf.write(str(p), cs.T, int(sr))
    return p


def _normalize_audio_input(AUDIO=None, audio_path: str = "", audio_url: str = "") -> Tuple[np.ndarray, int, Path]:
    """
    Accept ComfyUI AUDIO dict, or a file path/url; return ([C,S], sr, temp_wav_path).
    """
    # ComfyUI's AUDIO: {"waveform": [B,C,T], "sample_rate": sr}
    if isinstance(AUDIO, dict) and "waveform" in AUDIO and "sample_rate" in AUDIO:
        wf: torch.Tensor = AUDIO["waveform"]
        sr = int(AUDIO["sample_rate"])
        if wf.dim() == 3:
            wf = wf[0]  # [C,T]
        if wf.dim() != 2:
            raise RuntimeError(f"Unexpected AUDIO tensor shape: {tuple(wf.shape)} (want [C,T])")
        cs = wf.detach().cpu().float().numpy()
        return cs, sr, _save_temp_wav(cs, sr)

    # (arr, sr) tuple
    if isinstance(AUDIO, (list, tuple)) and len(AUDIO) == 2:
        arr, sr = AUDIO
        cs = _to_cs(np.asarray(arr))
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    # explicit file path
    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            raise RuntimeError(f"audio_path not found: {audio_path}")
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    # URL fetch
    if audio_url:
        import requests
        r = requests.get(audio_url, timeout=60); r.raise_for_status()
        p = Path(tempfile.gettempdir()) / f"eg_url_{int(time.time()*1000)}.wav"
        p.write_bytes(r.content)
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    raise RuntimeError("No AUDIO provided.")

# ---------------- CUDA/CuPy wiring (Windows) ----------------

def _wire_cuda_for_cupy_windows():
    r"""
    On Windows portable installs, make NVIDIA pip-wheel DLLs & headers discoverable:
      • Add ...\site-packages\nvidia\<package>\bin to the DLL search path
      • Point CUDA_PATH to ...\site-packages\nvidia\cuda_runtime (has include/)
    Must run BEFORE importing cupy.

    No-op when CUDA already resolves (system toolkit on PATH, or CuPy builds that
    bundle their own runtime) — the pip-wheel wiring is only a fallback.
    """
    if platform.system() != "Windows":
        return

    # sysconfig, not sys.executable.parent: a venv puts python.exe in Scripts\,
    # so the naive join yields venv\Scripts\Lib\site-packages (does not exist).
    sp = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    rt = sp / "cuda_runtime"     # contains include/ and bin/
    nvrtc = sp / "cuda_nvrtc"    # contains bin/
    embed_rt = Path(__file__).resolve().parents[2] / "python_embeded" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime"
    embed_nvrtc = Path(__file__).resolve().parents[2] / "python_embeded" / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc"

    # Let CuPy find headers at runtime (NVRTC needs CUDA runtime headers >= CUDA 12.2)
    cuda_root = None
    if rt.exists():
        cuda_root = rt
    elif embed_rt.exists():
        cuda_root = embed_rt

    if cuda_root is not None:
        os.environ.setdefault("CUDA_PATH", str(cuda_root))
        os.environ.setdefault("CUPY_CUDA_PATH", str(cuda_root))
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        if "cupy._environment" in sys.modules:
            try:
                import cupy._environment as ce  # type: ignore
                ce._cuda_path = ""
                ce._nvcc_path = ""
                ce.get_cuda_path()
                ce._setup_win32_dll_directory()
            except Exception:
                pass

    # Make DLLs loadable for this process (Python 3.8+)
    bins = [rt / "bin", nvrtc / "bin", embed_rt / "bin", embed_nvrtc / "bin"]
    for p in bins:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                os.environ["PATH"] = f"{str(p)};{os.environ.get('PATH','')}"

# ---------------- Fat Llama wrapper ----------------

def _cupy_kernel_error():
    """
    Return None if CuPy can launch a kernel, else the exception explaining why not.

    `import cupy` alone proves nothing: CuPy loads nvrtc/cudart lazily, so a CUDA-version
    mismatch imports fine and only dies on the first kernel launch — deep inside fat_llama.
    Launch one here instead.
    """
    try:
        import cupy
        cupy.asnumpy(cupy.arange(2, dtype=cupy.float32) * 2)
        return None
    except Exception as e:
        return e


def _ensure_gpu_stack():
    """
    Validate CUDA/CuPy presence early and give a friendly error if not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Fat Llama (GPU) requires an NVIDIA GPU. "
            "If you need CPU, use the separate Fat Llama — CPU/FFTW node."
        )

    if _cupy_kernel_error() is None:
        return

    # Only now fall back to pip-wheel DLL wiring. Doing it eagerly would point
    # CUDA_HOME at a stale nvidia/cuda_runtime wheel that may be a different CUDA
    # major than the system toolkit already on PATH.
    _wire_cuda_for_cupy_windows()

    err = _cupy_kernel_error()
    if err is not None:
        want = (torch.version.cuda or "12").split(".")[0]
        pkg = f"cupy-cuda{want}x"
        raise RuntimeError(
            "CuPy is unusable — it imported but could not launch a CUDA kernel. "
            f"Usually a CUDA-version mismatch: this torch is built for CUDA {torch.version.cuda}, "
            f"so CuPy must be '{pkg}'.\n"
            f"Fix: pip uninstall -y cupy-cuda12x cupy-cuda13x && pip install {pkg}\n"
            "Then restart ComfyUI. If you have no system CUDA toolkit, also install the "
            f"matching nvidia-*-cu{want} runtime wheels (nvrtc, cufft, cublas, curand, "
            "cusolver, cusparse).\n"
            f"Underlying error: {err}"
        ) from err

def _fat_llama_upscale(
    in_wav: Path,
    out_path: Path,
    target_format: str,
    max_iterations: int,
    threshold_value: float,
    target_bitrate_kbps: int,
    toggle_normalize: bool,
    toggle_autoscale: bool,
):
    """Call the public API: fat_llama.audio_fattener.feed.upscale(...)"""
    from fat_llama.audio_fattener import feed  # late import

    if not getattr(feed, "_egregora_read_audio_patch", False):
        orig_read_audio = feed.read_audio

        def _patched_read_audio(file_path, format):
            sample_rate, samples, bitrate, audio = orig_read_audio(file_path, format)
            try:
                feed._egregora_sample_width = getattr(audio, "sample_width", None)
            except Exception:
                pass
            return sample_rate, samples, bitrate, audio

        feed.read_audio = _patched_read_audio
        feed._egregora_read_audio_patch = True

    if not getattr(feed, "_egregora_write_audio_patch", False):
        orig_write_audio = feed.write_audio

        def _patched_write_audio(file_path, sample_rate, data, format):
            out = data
            try:
                m = float(np.max(np.abs(out))) if out is not None else 0.0
                if m > 1.0:
                    sw = getattr(feed, "_egregora_sample_width", None)
                    if sw:
                        scale = float(2 ** (8 * sw - 1))
                        if scale > 0:
                            out = out / scale
                    else:
                        out = out / m
            except Exception:
                pass
            return orig_write_audio(file_path, sample_rate, out, format)

        feed.write_audio = _patched_write_audio
        feed._egregora_write_audio_patch = True

    upscale = feed.upscale

    # Normalize ALWAYS on; Adaptive filter disabled for perf/stability
    upscale(
        input_file_path=str(in_wav),
        output_file_path=str(out_path),
        source_format="wav",
        target_format=target_format,
        max_iterations=int(max_iterations),
        threshold_value=float(threshold_value),
        target_bitrate_kbps=int(target_bitrate_kbps),
        toggle_normalize=bool(toggle_normalize),
        toggle_autoscale=bool(toggle_autoscale),
        toggle_adaptive_filter=False,
    )

# ---------------- ComfyUI Node ----------------

class EgregoraFatLlamaGPU:
    """
    Spectral Enhance (Fat Llama — GPU only)
    - Normalize is always ON (clamps final amplitude and prevents clipping).
    - Adaptive filter disabled for speed (still available in library if you want a "slow" node).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_format": (["wav", "flac"],),
                "max_iterations": ("INT", {"default": 300, "min": 1, "max": 5000}),
                "threshold_value": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_bitrate_kbps": ("INT", {"default": 1411, "min": 64, "max": 5000}),
                "toggle_normalize": ("BOOLEAN", {"default": True}),
                "toggle_autoscale": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "AUDIO": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    OUTPUT_NODE = False

    def run(
        self,
        target_format,
        max_iterations,
        threshold_value,
        target_bitrate_kbps,
        toggle_normalize,
        toggle_autoscale,
        AUDIO=None,
        audio_path="",
        audio_url="",
    ):
        _ensure_gpu_stack()

        # Normalize inbound audio to a temp WAV we can hand to fat_llama
        cs, in_sr, in_wav = _normalize_audio_input(AUDIO, audio_path, audio_url)

        # Choose an output temp path with chosen container
        suffix = ".wav" if target_format == "wav" else ".flac"
        out_path = Path(tempfile.gettempdir()) / f"eg_fatllama_{int(time.time()*1000)}{suffix}"

        # Run Fat Llama with always-on normalization and no adaptive filter
        _fat_llama_upscale(
            in_wav=in_wav,
            out_path=out_path,
            target_format=target_format,
            max_iterations=max_iterations,
            threshold_value=threshold_value,
            target_bitrate_kbps=target_bitrate_kbps,
            toggle_normalize=toggle_normalize,
            toggle_autoscale=toggle_autoscale,
        )

        # Read result back into Comfy
        y, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
        cs_out = _to_cs(y)
        wf = torch.from_numpy(cs_out).unsqueeze(0).contiguous()  # [1,C,T]
        return ({"waveform": wf, "sample_rate": int(sr)},)

# Register node
NODE_CLASS_MAPPINGS = {
    "EgregoraFatLlamaGPU": EgregoraFatLlamaGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraFatLlamaGPU": "🎛️ Spectral Enhance (Fat Llama — GPU)",
}
