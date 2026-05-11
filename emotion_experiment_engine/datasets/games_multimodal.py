"""Multimodal dataset adapter for game theory benchmarks.

This reuses the text game-theory parser and only adds image-path handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

from ..data_models import BenchmarkItem
from .games import GameTheoryDataset, REPO_ROOT


class GameTheoryMultimodalDataset(GameTheoryDataset):
    """Game-theory dataset variant that requires an image per scenario."""

    def _resolve_data_path(self) -> Path:
        if self.config.data_path is not None:
            candidate = Path(self.config.data_path)
            if candidate.exists():
                return candidate
        return super()._resolve_data_path()

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw_items = self._load_raw_scenarios()

        image_paths_by_id: Dict[Any, List[str]] = {}
        for idx, record in enumerate(raw_items):
            item_id = record.get("id", idx)
            paths: List[str] = []

            if isinstance(record.get("images"), list):
                paths = [str(p) for p in record["images"] if p]
            elif record.get("image_path"):
                paths = [str(record["image_path"])]
            elif record.get("image"):
                paths = [str(record["image"])]

            image_paths_by_id[item_id] = paths

        items = super()._load_and_parse_data()

        for item in items:
            paths = image_paths_by_id.get(item.id)
            if not paths:
                raise ValueError(
                    f"Missing image path(s) for item_id={item.id!r} in multimodal dataset"
                )
            metadata = dict(item.metadata or {})
            metadata["image_paths"] = paths
            metadata["image_path"] = paths[0]
            item.metadata = metadata

        return items

    @staticmethod
    def _resolve_image_paths(paths: Sequence[str]) -> List[Path]:
        resolved: List[Path] = []
        for raw in paths:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            resolved.append(path)
        return resolved

    @classmethod
    def _load_images(cls, paths: Sequence[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for path in cls._resolve_image_paths(paths):
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            images.append(Image.open(path).convert("RGB"))
        return images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        options = None
        if item.metadata and "options" in item.metadata:
            options = item.metadata["options"]

        image_paths = None
        if item.metadata and "image_paths" in item.metadata:
            image_paths = item.metadata["image_paths"]
        if not image_paths or not isinstance(image_paths, list):
            raise ValueError(f"Missing image_paths in item metadata (id={item.id!r})")

        images = self._load_images(image_paths)

        if self.prompt_wrapper:
            prompt = self.prompt_wrapper(
                context=item.context if item.context else "",
                question=item.input_text,
                answer=item.ground_truth,
                options=options,
                images=images,
            )
        else:
            prompt = (
                f"Context: {item.context}\nQuestion: {item.input_text}\nAnswer:"
                if item.context
                else f"{item.input_text}\nAnswer:"
            )

        adapted_ground_truth = (
            self.answer_wrapper(item.ground_truth)
            if self.answer_wrapper is not None
            else item.ground_truth
        )

        return {
            "item": item,
            "prompt": prompt,
            "ground_truth": adapted_ground_truth,
            "images": images,
        }

    def collate_fn(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "prompts": [item["prompt"] for item in batch_items],
            "items": [item["item"] for item in batch_items],
            "ground_truths": [item["ground_truth"] for item in batch_items],
            "images": [item["images"] for item in batch_items],
        }


__all__ = ["GameTheoryMultimodalDataset"]
