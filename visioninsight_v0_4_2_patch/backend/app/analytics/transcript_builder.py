from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _jsonl_write(file, payload: Dict[str, Any]) -> None:
    file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_ffmpeg() -> Tuple[Optional[str], str]:
    env_path = os.getenv("VISIONINSIGHT_FFMPEG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path), f"ffmpeg from env: {path}"
        return None, f"VISIONINSIGHT_FFMPEG set but not found: {env_path}"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg, f"ffmpeg from PATH: {ffmpeg}"

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        candidate = Path(ffprobe).parent / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate), f"ffmpeg sibling of ffprobe: {candidate}"
        return None, f"ffprobe found ({ffprobe}) but ffmpeg was not found"

    return None, "ffmpeg not found (PATH/env)"


def _run(command: List[str]) -> Tuple[int, str, str]:
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        return process.returncode, process.stdout or "", process.stderr or ""
    except Exception as error:
        return 999, "", f"subprocess error: {type(error).__name__}: {error}"


def _extract_wav(input_video: Path, output_wav: Path, sample_rate: int = 16000) -> Tuple[bool, str]:
    ffmpeg, notes = _resolve_ffmpeg()
    if not ffmpeg:
        return False, notes

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(output_wav),
    ]
    code, _stdout, stderr = _run(command)
    success = code == 0 and output_wav.exists() and output_wav.stat().st_size > 0
    if success:
        return True, f"ok; {notes}"

    stderr_tail = "\n".join(stderr.strip().splitlines()[-8:]).strip()
    return False, f"ffmpeg failed (code={code}); {notes}; stderr_tail:\n{stderr_tail}"


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
    """Extract audio and produce timestamped transcript segments."""

    def __init__(self):
        # Reusing a loaded Whisper model saves substantial time on later analyses.
        self._model_cache: Dict[Tuple[str, str, str], Any] = {}

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
        output_jsonl = Path(transcript_jsonl_path)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        wav_path = Path(extracted_wav_path) if extracted_wav_path else output_jsonl.parent / "audio.wav"

        audio_ok, audio_details = _extract_wav(input_video, wav_path, sample_rate=sample_rate)
        if not audio_ok:
            return self._write_unavailable(
                analysis_id,
                output_jsonl,
                backend="none",
                reason=audio_details,
                notes="audio extraction failed; see transcript.jsonl reason",
            )

        selected_backend = backend
        if backend == "auto":
            selected_backend = "faster-whisper" if self._has_faster_whisper() else "none"

        if selected_backend != "faster-whisper":
            return self._write_unavailable(
                analysis_id,
                output_jsonl,
                backend="none",
                reason="audio extracted OK, but no ASR backend installed (pip install faster-whisper)",
                notes="audio ok; ASR backend missing",
            )

        try:
            segments = self._transcribe_faster_whisper(
                wav_path=wav_path,
                model_size=model_size,
                language=language,
            )
            with output_jsonl.open("w", encoding="utf-8") as file:
                for segment in segments:
                    _jsonl_write(file, segment)

            return TranscriptBuildResult(
                analysis_id=analysis_id,
                available=True,
                backend="faster-whisper",
                segments_written=len(segments),
                notes=f"ok; model={model_size}; language={language or 'auto'}",
            )
        except Exception as error:
            return self._write_unavailable(
                analysis_id,
                output_jsonl,
                backend="faster-whisper",
                reason=f"ASR failed: {type(error).__name__}: {error}",
                notes="ASR failed; see transcript.jsonl reason",
            )

    @staticmethod
    def _write_unavailable(
        analysis_id: str,
        output_jsonl: Path,
        backend: str,
        reason: str,
        notes: str,
    ) -> TranscriptBuildResult:
        with output_jsonl.open("w", encoding="utf-8") as file:
            _jsonl_write(file, {
                "t_start": 0.0,
                "t_end": 0.0,
                "text": "",
                "available": False,
                "reason": reason,
            })
        return TranscriptBuildResult(
            analysis_id=analysis_id,
            available=False,
            backend=backend,
            segments_written=1,
            notes=notes,
        )

    @staticmethod
    def _has_faster_whisper() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _get_whisper_model(self, model_size: str):
        from faster_whisper import WhisperModel  # type: ignore

        device = os.getenv("TRANSCRIPT_DEVICE", "auto")
        compute_type = os.getenv("TRANSCRIPT_COMPUTE_TYPE", "auto")
        cache_key = (model_size, device, compute_type)
        model = self._model_cache.get(cache_key)
        if model is None:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._model_cache[cache_key] = model
        return model

    def _transcribe_faster_whisper(
        self,
        wav_path: Path,
        model_size: str,
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        beam_size = int(os.getenv("TRANSCRIPT_BEAM_SIZE", "5"))
        model = self._get_whisper_model(model_size)
        segments, _info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
        )

        output: List[Dict[str, Any]] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue

            average_log_probability = getattr(segment, "avg_logprob", None)
            confidence = None
            if isinstance(average_log_probability, (float, int)):
                confidence = max(0.0, min(1.0, (float(average_log_probability) + 5.0) / 5.0))

            output.append({
                "t_start": round(float(segment.start), 2),
                "t_end": round(float(segment.end), 2),
                "text": text,
                "confidence": round(confidence, 4) if confidence is not None else None,
            })
        return output
