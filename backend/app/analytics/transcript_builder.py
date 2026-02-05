from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _resolve_ffmpeg() -> Tuple[Optional[str], str]:
    """
    Returns (ffmpeg_path, notes).
    Resolution order:
      1) env VISIONINSIGHT_FFMPEG
      2) shutil.which("ffmpeg")
      3) if ffprobe exists, try sibling ffmpeg.exe in same folder
    """
    env_path = os.getenv("VISIONINSIGHT_FFMPEG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p), f"ffmpeg from env: {p}"
        return None, f"VISIONINSIGHT_FFMPEG set but not found: {env_path}"

    ffmpeg = _which("ffmpeg")
    if ffmpeg:
        return ffmpeg, f"ffmpeg from PATH: {ffmpeg}"

    ffprobe = _which("ffprobe")
    if ffprobe:
        # try sibling
        probe_dir = Path(ffprobe).parent
        cand = probe_dir / "ffmpeg.exe"
        if cand.exists():
            return str(cand), f"ffmpeg sibling of ffprobe: {cand}"
        return None, f"ffprobe found ({ffprobe}) but ffmpeg not found in PATH or sibling"

    return None, "ffmpeg not found (PATH/env)"


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return 999, "", f"subprocess error: {type(e).__name__}: {str(e)}"


def _ffmpeg_extract_wav(input_video: Path, out_wav: Path, sample_rate: int = 16000) -> Tuple[bool, str]:
    """
    Extracts mono WAV using ffmpeg.
    Returns (success, details).
    """
    ffmpeg, notes = _resolve_ffmpeg()
    if not ffmpeg:
        return False, f"{notes}"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav),
    ]

    code, out, err = _run(cmd)
    ok = (code == 0 and out_wav.exists() and out_wav.stat().st_size > 0)

    if ok:
        return True, f"ok; {notes}"

    # include useful stderr tail
    err_tail = err.strip().splitlines()[-8:]
    tail = "\n".join(err_tail).strip()
    return False, f"ffmpeg failed (code={code}); {notes}; stderr_tail:\n{tail}"


@dataclass
class TranscriptBuildResult:
    analysis_id: str
    available: bool
    backend: str
    segments_written: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "available": self.available,
            "backend": self.backend,
            "segments_written": self.segments_written,
            "notes": self.notes,
        }


class TranscriptBuilder:
    """
    Produces transcript.jsonl with segments:
      {"t_start": 12.4, "t_end": 15.1, "text": "...", "confidence": 0.92}

    Safe-by-default:
      - If ffmpeg missing or audio extraction fails -> placeholder line, pipeline continues
      - If ASR backend not installed -> placeholder line, pipeline continues

    Supported backends:
      - faster-whisper (optional): pip install faster-whisper
    """

    def build_from_video(
        self,
        analysis_id: str,
        input_video_path: str,
        transcript_jsonl_path: str,
        extracted_wav_path: Optional[str] = None,
        sample_rate: int = 16000,
        backend: str = "auto",
        model_size: str = "base",
        language: Optional[str] = None,
    ) -> TranscriptBuildResult:
        input_video = Path(input_video_path)
        out_jsonl = Path(transcript_jsonl_path)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        wav_path = Path(extracted_wav_path) if extracted_wav_path else (out_jsonl.parent / "audio.wav")

        # 1) Extract audio
        audio_ok, audio_details = _ffmpeg_extract_wav(input_video, wav_path, sample_rate=sample_rate)
        if not audio_ok:
            with out_jsonl.open("w", encoding="utf-8") as f:
                _jsonl_write(f, {
                    "t_start": 0.0,
                    "t_end": 0.0,
                    "text": "",
                    "available": False,
                    "reason": audio_details,
                })
            return TranscriptBuildResult(
                analysis_id=analysis_id,
                available=False,
                backend="none",
                segments_written=1,
                notes="audio extraction failed; see transcript.jsonl reason",
            )

        # 2) Select backend
        chosen_backend = backend
        if backend == "auto":
            chosen_backend = "faster-whisper" if self._has_faster_whisper() else "none"

        if chosen_backend != "faster-whisper":
            with out_jsonl.open("w", encoding="utf-8") as f:
                _jsonl_write(f, {
                    "t_start": 0.0,
                    "t_end": 0.0,
                    "text": "",
                    "available": False,
                    "reason": "audio extracted OK, but no ASR backend installed (pip install faster-whisper)",
                })
            return TranscriptBuildResult(
                analysis_id=analysis_id,
                available=False,
                backend="none",
                segments_written=1,
                notes="audio ok; ASR backend missing",
            )

        # 3) Run ASR
        try:
            segments = self._transcribe_faster_whisper(
                wav_path=wav_path,
                model_size=model_size,
                language=language,
            )

            with out_jsonl.open("w", encoding="utf-8") as f:
                n = 0
                for seg in segments:
                    _jsonl_write(f, seg)
                    n += 1

            return TranscriptBuildResult(
                analysis_id=analysis_id,
                available=True,
                backend="faster-whisper",
                segments_written=n,
                notes=f"ok; model={model_size}; language={language or 'auto'}",
            )
        except Exception as e:
            with out_jsonl.open("w", encoding="utf-8") as f:
                _jsonl_write(f, {
                    "t_start": 0.0,
                    "t_end": 0.0,
                    "text": "",
                    "available": False,
                    "reason": f"ASR failed: {type(e).__name__}: {str(e)}",
                })
            return TranscriptBuildResult(
                analysis_id=analysis_id,
                available=False,
                backend="faster-whisper",
                segments_written=1,
                notes="ASR failed; see transcript.jsonl reason",
            )

    def _has_faster_whisper(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _transcribe_faster_whisper(
        self,
        wav_path: Path,
        model_size: str,
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        from faster_whisper import WhisperModel  # type: ignore

        compute_type = os.getenv("TRANSCRIPT_COMPUTE_TYPE", "auto")
        beam_size = int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5"))

        model = WhisperModel(model_size, device="auto", compute_type=compute_type)

        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
        )

        out: List[Dict[str, Any]] = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue

            avg_lp = getattr(s, "avg_logprob", None)
            conf = None
            if isinstance(avg_lp, (float, int)):
                conf = max(0.0, min(1.0, (float(avg_lp) + 5.0) / 5.0))

            out.append({
                "t_start": round(float(s.start), 2),
                "t_end": round(float(s.end), 2),
                "text": text,
                "confidence": round(conf, 4) if conf is not None else None,
            })

        return out
