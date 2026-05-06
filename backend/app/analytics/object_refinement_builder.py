from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _xyxy_from_bbox(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Supports common formats:
      - obj["bbox"] = [x1,y1,x2,y2]
      - obj has x1,y1,x2,y2
      - obj["xyxy"]
    """
    if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) == 4:
        x1, y1, x2, y2 = obj["bbox"]
        return float(x1), float(y1), float(x2), float(y2)

    if "xyxy" in obj and isinstance(obj["xyxy"], (list, tuple)) and len(obj["xyxy"]) == 4:
        x1, y1, x2, y2 = obj["xyxy"]
        return float(x1), float(y1), float(x2), float(y2)

    keys = ("x1", "y1", "x2", "y2")
    if all(k in obj for k in keys):
        return float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])

    return None


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class ObjectRefinement:
    track_id: int
    yolo_class: str
    refined_label: str
    semantic_group: str
    confidence: float
    candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "yolo_class": self.yolo_class,
            "refined_label": self.refined_label,
            "semantic_group": self.semantic_group,
            "confidence": round(float(self.confidence), 4),
            "candidates": self.candidates,
        }


class ObjectRefinementBuilder:
    """
    CLIP-based refinement for object tracks.

    Input:
      - input_video_path: original video
      - objects_jsonl_path: per-frame objects detections with track_id + bbox

    Output:
      - object_refinements.json: one record per track_id with refined_label
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        # Lazy init
        self._clip = None
        self._clip_preprocess = None
        self._tokenizer = None

    def build(
        self,
        input_video_path: Path,
        objects_jsonl_path: Path,
        output_refinements_json_path: Path,
        label_ontology: Optional[Dict[str, List[str]]] = None,
        samples_per_track: int = 3,
        topk: int = 5,
        min_bbox_size_px: int = 18,
    ) -> Dict[str, Any]:
        """
        Returns summary dict for logging/summary.json.
        """
        input_video_path = Path(input_video_path)
        objects_jsonl_path = Path(objects_jsonl_path)
        output_refinements_json_path = Path(output_refinements_json_path)

        if not objects_jsonl_path.exists():
            _write_json(output_refinements_json_path, {"available": False, "reason": "objects.jsonl not found"})
            return {"available": False, "reason": "objects.jsonl not found"}

        if not input_video_path.exists():
            _write_json(output_refinements_json_path, {"available": False, "reason": "input video not found"})
            return {"available": False, "reason": "input video not found"}

        ontology = label_ontology or self.default_ontology()
        all_labels, label_to_group = self._flatten_ontology(ontology)

        # Prepare CLIP
        clip_model, preprocess, tokenizer = self._get_clip()
        text_features = self._encode_texts(clip_model, tokenizer, all_labels)

        # Collect per-track samples (frame_index + bbox + yolo_class)
        track_samples = self._collect_track_samples(objects_jsonl_path, samples_per_track=samples_per_track)

        # Open video
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            _write_json(output_refinements_json_path, {"available": False, "reason": "cv2 cannot open video"})
            return {"available": False, "reason": "cv2 cannot open video"}

        refinements: List[ObjectRefinement] = []
        skipped_small = 0
        processed_tracks = 0

        for track_id, samples in sorted(track_samples.items(), key=lambda x: x[0]):
            processed_tracks += 1
            yolo_class = samples[0].get("class_name", "unknown")

            # Aggregate similarities over samples
            agg_scores = None
            used = 0

            for s in samples:
                frame_idx = _safe_int(s.get("frame", 0))
                bbox = s.get("bbox_xyxy")
                if bbox is None:
                    continue

                ok, frame = self._read_frame(cap, frame_idx)
                if not ok or frame is None:
                    continue

                crop = self._crop_xyxy(frame, bbox, min_size=min_bbox_size_px)
                if crop is None:
                    skipped_small += 1
                    continue

                img_feat = self._encode_image(clip_model, preprocess, crop)
                scores = self._cosine_sim(img_feat, text_features)  # 1D list/array

                if agg_scores is None:
                    agg_scores = scores
                else:
                    agg_scores = agg_scores + scores  # vector sum

                used += 1

            if agg_scores is None or used == 0:
                # No usable crops
                continue

            agg_scores = agg_scores / float(used)
            top = self._topk(agg_scores, all_labels, k=topk)

            refined_label = top[0]["label"]
            confidence = float(top[0]["score"])
            group = label_to_group.get(refined_label, "other")

            refinements.append(ObjectRefinement(
                track_id=int(track_id),
                yolo_class=str(yolo_class),
                refined_label=refined_label,
                semantic_group=group,
                confidence=confidence,
                candidates=top,
            ))

        cap.release()

        payload = {
            "available": True,
            "model": {"name": self.clip_model, "pretrained": self.clip_pretrained, "device": self.device},
            "tracks_total": len(track_samples),
            "tracks_refined": len(refinements),
            "skipped_small_crops": skipped_small,
            "ontology_groups": sorted(list(set(ontology.keys()))),
            "refinements": [r.to_dict() for r in refinements],
        }
        _write_json(output_refinements_json_path, payload)

        return {
            "available": True,
            "tracks_total": len(track_samples),
            "tracks_refined": len(refinements),
            "skipped_small_crops": skipped_small,
            "output": str(output_refinements_json_path),
        }

    # --------------------------
    # Ontology
    # --------------------------

    @staticmethod
    def default_ontology() -> Dict[str, List[str]]:
        """
        Minimal стартовый словарь (расширяемый).
        Сюда ты потом добавишь свои категории хоть на тысячи строк,
        но важно: управляемо и тематически.
        """
        return {
            "accessories": [
                "necklace", "chain necklace", "gold chain", "silver chain", "pendant", "choker",
                "tie", "necktie", "bow tie",
                "bracelet", "watch",
            ],
            "music_equipment": [
                "dj controller", "dj mixer", "mixing console", "turntable", "dj deck",
                "audio interface", "midi controller",
                "computer keyboard", "piano keyboard",
            ],
            "electronics": [
                "smartphone", "cell phone", "headphones", "microphone", "laptop", "camera",
            ],
            "vehicles": [
                "car", "truck", "bus", "train", "motorcycle", "bicycle",
            ],
            "sports": [
                "ball", "soccer ball", "basketball", "tennis ball",
            ],
            "other": [
                "umbrella", "traffic light", "bottle", "cup", "chair", "tv", "backpack", "handbag",
            ],
        }

    @staticmethod
    def _flatten_ontology(ontology: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
        labels: List[str] = []
        label_to_group: Dict[str, str] = {}
        for group, items in ontology.items():
            for lab in items:
                lab2 = lab.strip()
                if not lab2:
                    continue
                labels.append(lab2)
                label_to_group[lab2] = group
        # remove duplicates while preserving order
        seen = set()
        out = []
        for l in labels:
            if l in seen:
                continue
            seen.add(l)
            out.append(l)
        return out, label_to_group

    # --------------------------
    # CLIP backend (open_clip)
    # --------------------------

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_clip(self):
        if self._clip is not None:
            return self._clip, self._clip_preprocess, self._tokenizer

        try:
            import torch
            import open_clip
        except Exception as e:
            raise RuntimeError(
                "open_clip_torch is not installed. Install: pip install open_clip_torch"
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(self.clip_model)

        model.eval()
        model.to(self.device)

        self._clip = model
        self._clip_preprocess = preprocess
        self._tokenizer = tokenizer
        return model, preprocess, tokenizer

    def _encode_texts(self, model, tokenizer, labels: List[str]):
        import torch

        # Prompting helps accuracy
        prompts = [f"a photo of a {lab}" for lab in labels]
        tokens = tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _encode_image(self, model, preprocess, bgr_image):
        import torch
        from PIL import Image

        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img = preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    @staticmethod
    def _cosine_sim(image_feat, text_feat):
        import torch
        # image_feat: [1, D], text_feat: [N, D]
        sims = (image_feat @ text_feat.T).squeeze(0)
        sims = sims.float()
        # convert to probabilities-like
        probs = torch.softmax(sims * 10.0, dim=0)  # temperature
        return probs

    @staticmethod
    def _topk(score_vec, labels: List[str], k: int = 5) -> List[Dict[str, Any]]:
        import torch

        k = max(1, min(int(k), len(labels)))
        vals, idxs = torch.topk(score_vec, k=k)
        out = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            out.append({"label": labels[int(i)], "score": round(float(v), 4)})
        return out

    # --------------------------
    # Sampling + Video IO
    # --------------------------

    @staticmethod
    def _collect_track_samples(objects_jsonl_path: Path, samples_per_track: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """
        Reads objects.jsonl lines like:
          { "frame": 123, "time_sec": 4.1, "objects": [ {track_id, class_name, bbox...}, ... ] }
        Collects a few samples per track spread over time.
        """
        # gather frames per track
        by_track: Dict[int, List[Dict[str, Any]]] = {}

        for row in _iter_jsonl(objects_jsonl_path):
            frame = _safe_int(row.get("frame", 0))
            objs = row.get("objects") or row.get("detections") or []
            if not isinstance(objs, list):
                continue

            for obj in objs:
                tid = obj.get("track_id")
                if tid is None:
                    continue
                tid = _safe_int(tid, -1)
                if tid < 0:
                    continue

                bbox = _xyxy_from_bbox(obj)
                if bbox is None:
                    continue

                rec = {
                    "frame": frame,
                    "class_name": obj.get("class_name") or obj.get("class") or "unknown",
                    "bbox_xyxy": bbox,
                }
                by_track.setdefault(tid, []).append(rec)

        # pick evenly spaced samples per track
        sampled: Dict[int, List[Dict[str, Any]]] = {}
        for tid, items in by_track.items():
            if not items:
                continue
            items.sort(key=lambda x: x["frame"])
            n = len(items)
            if n <= samples_per_track:
                sampled[tid] = items
                continue

            # evenly spaced indices
            picks = []
            for j in range(samples_per_track):
                idx = int(round(j * (n - 1) / float(samples_per_track - 1)))
                picks.append(idx)
            picks = sorted(set(picks))
            sampled[tid] = [items[i] for i in picks]

        return sampled

    @staticmethod
    def _read_frame(cap: cv2.VideoCapture, frame_idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        return ok, frame

    @staticmethod
    def _crop_xyxy(frame, bbox_xyxy: Tuple[float, float, float, float], min_size: int = 18):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        x1 = int(_clamp(x1, 0, w - 1))
        y1 = int(_clamp(y1, 0, h - 1))
        x2 = int(_clamp(x2, 0, w - 1))
        y2 = int(_clamp(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1
        if bw < min_size or bh < min_size:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop
