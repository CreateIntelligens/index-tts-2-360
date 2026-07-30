#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 (GPT module) with Japanese data.

This trainer expects the preprocessing pipeline to have produced manifests where each
sample record stores paths to:
  - text token ids (.npy, int32)
  - semantic codes (.npy, int32)
  - conditioning latent (.npy, float32 [32, hidden])
  - emotion vector (.npy, float32 [hidden])

The model is optimised with cross-entropy losses over text tokens and semantic codes,
with optional gradient accumulation and mixed-precision support. Checkpoints are
emitted every 1k optimiser steps (`model_step{N}.pth`), keeping only the three most
recent snapshots. TensorBoard summaries track losses and learning rate under the
chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument(
        "--train-manifest",
        dest="train_manifests",
        action="append",
        type=str,
        required=True,
        help="Training manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifests",
        action="append",
        type=str,
        required=True,
        help="Validation manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size per optimisation step.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=0, help="Validation frequency in steps (0 = once per epoch).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2, help="Weight for text CE loss.")
    parser.add_argument("--mel-loss-weight", type=float, default=0.8, help="Weight for semantic CE loss.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument(
        "--use-duration-control",
        action="store_true",
        help="Train GPT with duration embeddings derived from target semantic lengths.",
    )
    parser.add_argument(
        "--duration-dropout",
        type=float,
        default=0.3,
        help="Probability of zeroing duration embeddings when --use-duration-control is enabled.",
    )
    parser.add_argument(
        "--emotion-dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of zeroing the emotion vector during training. The vector is "
            "added to the speaker latent and (with --emotion-source target) comes from "
            "the same speaker as the prompt, so identity and emotion stay entangled and "
            "changing the emotion at inference also changes the timbre. Zeroing it on "
            "some steps leaves the conditioning latent as the only identity cue. Try 0.2."
        ),
    )
    parser.add_argument(
        "--emotion-shuffle",
        type=float,
        default=0.0,
        help=(
            "Probability of swapping in another sample's emotion vector. Decorrelates "
            "identity harder than dropout, but the vector then disagrees with the "
            "target's prosody, which is what made the first run ignore emotion entirely. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--emotion-source",
        choices=("target", "prompt"),
        default="target",
        help=(
            "Which side of a pair supplies the emotion vector. 'target' (default) keeps "
            "the vector informative so the model actually learns to use emotion "
            "conditioning. 'prompt' reproduces the earlier behaviour, where the vector "
            "came from a randomly chosen other clip and the model learned to ignore it "
            "and read emotion off the text instead."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="nccl",
        help="torch.distributed backend for multi-GPU runs (falls back to gloo without CUDA).",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        default=True,
        help=(
            "Default on: the loss only touches part of UnifiedVoice (the conditioning "
            "and emotion encoders are bypassed because those features are precomputed), "
            "so DDP would otherwise abort on unreduced parameters."
        ),
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Faster, but only safe if every parameter receives a gradient.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        dest="ddp_static_graph",
        action="store_true",
        default=True,
        help=(
            "Default on. UnifiedVoice runs the GPT stack under gradient checkpointing, "
            "whose reentrant backward marks each parameter ready twice; DDP's default "
            "reducer rejects that. A static graph is the supported way to combine the "
            "two, and this model's graph really is the same every step."
        ),
    )
    parser.add_argument(
        "--no-ddp-static-graph",
        dest="ddp_static_graph",
        action="store_false",
        help="Disable the static-graph optimisation (needs --no-gradient-checkpointing).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
        help="Force gradient checkpointing on (default: whatever the model config says).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Trade memory for speed; comfortable on 80GB cards.",
    )
    parser.add_argument(
        "--freeze-unused-modules",
        action="store_true",
        help=(
            "Set requires_grad=False on the conditioning/emotion encoders that the "
            "precomputed-feature training path never uses. Lets you drop "
            "--ddp-find-unused-parameters for extra speed."
        ),
    )
    return parser.parse_args()


@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(entries: Sequence[str], flag_name: str) -> List[ManifestSpec]:
    if not entries:
        raise ValueError(f"{flag_name} requires at least one manifest path.")
    specs: List[ManifestSpec] = []
    for raw in entries:
        value = raw.strip()
        lang: Optional[str] = None
        for separator in ("::", "@", "="):
            if separator in value:
                path_str, lang_part = value.rsplit(separator, 1)
                value = path_str.strip()
                lang = lang_part.strip().lower() or None
                break
        path = Path(value).expanduser()
        specs.append(ManifestSpec(path=path, language=lang))
    return specs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)


@dataclass
class DistContext:
    """
    Describes this process's place in the job. `world_size == 1` covers both the
    plain single-GPU run and the CPU fallback, so the training code below never
    needs to branch on "are we distributed".
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


def init_distributed(backend: str) -> DistContext:
    """
    Pick up the rank/world-size the launcher exported. Supports `torchrun`
    (RANK/WORLD_SIZE/LOCAL_RANK) and Slurm's `srun` (SLURM_PROCID/SLURM_NTASKS/
    SLURM_LOCALID). The GPU count is never hard-coded: whatever the launcher asks
    for is what we use.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(1, torch.cuda.device_count())))
        os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", "29500")
    else:
        return DistContext()

    if world_size <= 1:
        return DistContext()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistContext(rank=rank, local_rank=local_rank, world_size=world_size, enabled=True)


def unwrap_model(module: nn.Module) -> "UnifiedVoice":
    """Peel off DistributedDataParallel / TrainStep to reach the raw UnifiedVoice."""
    if isinstance(module, DistributedDataParallel):
        module = module.module
    if isinstance(module, TrainStep):
        module = module.model
    return module


@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    sample_type: str = "single"
    prompt_id: Optional[str] = None
    target_id: Optional[str] = None
    language: Optional[str] = None
    prompt_language: Optional[str] = None
    manifest_path: Optional[Path] = None


class JapaneseGPTDataset(Dataset):
    def __init__(self, manifests: Sequence[ManifestSpec], emotion_source: str = "target"):
        self.emotion_source = emotion_source
        if isinstance(manifests, ManifestSpec):
            manifests = [manifests]
        manifest_list = list(manifests)
        if not manifest_list:
            raise ValueError("No manifest paths supplied.")

        self.samples: List[Sample] = []
        self.sample_type: str = "unknown"
        self.manifest_summaries: List[Dict[str, object]] = []
        self.bad_indices: Set[int] = set()

        for spec in manifest_list:
            self._load_single_manifest(spec)

        if not self.samples:
            manifest_paths = ", ".join(str(spec.path) for spec in manifest_list)
            raise RuntimeError(f"No entries found in the provided manifests: {manifest_paths}")
        if self.sample_type != "paired":
            raise RuntimeError(
                "The GPT trainer expects prompt/target pair manifests.\n"
                "Generate paired manifests with tools/build_gpt_prompt_pairs.py and retry."
            )

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        if not value:
            raise ValueError("Empty path provided in manifest record.")
        path = Path(value)
        if path.is_absolute():
            return path
        return (base_dir / path).expanduser()

    @staticmethod
    def _normalize_language(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped.lower() if stripped else None

    def _load_single_manifest(self, spec: ManifestSpec) -> None:
        manifest_path = spec.path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        local_count = 0
        local_languages: set[str] = set()
        manifest_sample_type: Optional[str] = None
        base_dir = manifest_path.parent

        print(f"[Info] Parsing manifest {manifest_path} ...")
        processed = 0
        progress_interval = 10000

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                processed += 1
                is_paired = "prompt_condition_path" in record and "target_codes_path" in record
                if is_paired:
                    # Emotion has to come from the target by default. The prompt is a
                    # *randomly chosen* other clip of the same speaker, so its emotion is
                    # uncorrelated with the target's prosody: the vector then carries no
                    # predictive signal and the model learns to ignore emotion
                    # conditioning altogether, inferring prosody from the text instead.
                    # Speaker identity does not suffer the same way, because the prompt is
                    # the same speaker and the timbre latent stays informative.
                    if self.emotion_source == "prompt":
                        emo_path_value = record.get("prompt_emo_vec_path") or record.get(
                            "target_emo_vec_path"
                        )
                    else:
                        emo_path_value = record.get("target_emo_vec_path") or record.get(
                            "prompt_emo_vec_path"
                        )
                    if not emo_path_value:
                        raise RuntimeError(
                            f"Paired manifest entry {record.get('id')} has neither "
                            "target_emo_vec_path nor prompt_emo_vec_path."
                        )
                    target_language = self._normalize_language(
                        record.get("target_language") or record.get("language") or spec.language
                    )
                    prompt_language = self._normalize_language(record.get("prompt_language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["target_text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["target_codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["prompt_condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, emo_path_value),
                        text_len=int(record["target_text_len"]),
                        code_len=int(record["target_code_len"]),
                        condition_len=int(record.get("prompt_condition_len", 32)),
                        sample_type="paired",
                        prompt_id=record.get("prompt_id"),
                        target_id=record.get("target_id"),
                        language=target_language,
                        prompt_language=prompt_language,
                        manifest_path=manifest_path,
                    )
                else:
                    language = self._normalize_language(record.get("language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, record["emo_vec_path"]),
                        text_len=int(record["text_len"]),
                        code_len=int(record["code_len"]),
                        condition_len=int(record.get("condition_len", 32)),
                        sample_type="single",
                        manifest_path=manifest_path,
                        language=language,
                    )

                if manifest_sample_type is None:
                    manifest_sample_type = sample.sample_type
                elif manifest_sample_type != sample.sample_type:
                    raise RuntimeError(
                        f"Manifest {manifest_path} mixes sample types ({manifest_sample_type} vs {sample.sample_type})."
                    )

                self.samples.append(sample)
                local_count += 1
                if sample.language:
                    local_languages.add(sample.language)
                if sample.prompt_language:
                    local_languages.add(sample.prompt_language)

                if processed % progress_interval == 0:
                    print(
                        f"  • processed {processed:,} entries "
                        f"(kept {local_count:,}) in {manifest_path.name}"
                    )

        if local_count:
            if processed % progress_interval != 0:
                print(
                    f"  • processed {processed:,} entries "
                    f"(kept {local_count:,}) in {manifest_path.name}"
                )
            if manifest_sample_type and manifest_sample_type != "paired":
                raise RuntimeError(
                    f"Manifest {manifest_path} contains '{manifest_sample_type}' entries. "
                    "This trainer expects prompt/target pair manifests (see tools/build_gpt_prompt_pairs.py)."
                )
            if self.sample_type == "unknown":
                self.sample_type = manifest_sample_type or "unknown"
            elif manifest_sample_type and self.sample_type != manifest_sample_type:
                raise RuntimeError(
                    f"Mixed sample types encountered across manifests: {self.sample_type} vs {manifest_sample_type} (from {manifest_path})"
                )

            languages_display = sorted(local_languages)
            if not languages_display and spec.language:
                languages_display = [spec.language]
            language_text = ", ".join(languages_display) if languages_display else "unspecified"
            print(
                f"[Info] Loaded {local_count} samples ({manifest_sample_type}) from {manifest_path} "
                f"(languages: {language_text})"
            )
            self.manifest_summaries.append(
                {"path": manifest_path, "count": local_count, "languages": languages_display}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("Dataset is empty.")

        if len(self.bad_indices) >= len(self.samples):
            raise RuntimeError("All samples were marked invalid; cannot continue.")

        attempts = 0
        max_attempts = len(self.samples)
        sample_count = len(self.samples)

        while attempts < max_attempts:
            current_idx = idx % sample_count

            sample = self.samples[current_idx]
            if sample is None:
                idx += 1
                attempts += 1
                continue

            try:
                text_ids = np.load(sample.text_ids_path, allow_pickle=False)
                codes = np.load(sample.codes_path, allow_pickle=False)
                condition = np.load(sample.condition_path, allow_pickle=False)
                emo_vec = np.load(sample.emo_vec_path, allow_pickle=False)

                if text_ids.size == 0 or codes.size == 0 or condition.size == 0 or emo_vec.size == 0:
                    raise ValueError("Encountered empty feature file.")

                text_ids = text_ids.astype(np.int64, copy=False)
                codes = codes.astype(np.int64, copy=False)
                condition = condition.astype(np.float32, copy=False)
                emo_vec = emo_vec.astype(np.float32, copy=False)

                return {
                    "id": sample.id,
                    "text_ids": torch.from_numpy(text_ids),
                    "codes": torch.from_numpy(codes),
                    "condition": torch.from_numpy(condition),  # [cond_len, dim]
                    "emo_vec": torch.from_numpy(emo_vec),
                    "text_len": torch.tensor(sample.text_len, dtype=torch.long),
                    "code_len": torch.tensor(sample.code_len, dtype=torch.long),
                    "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
                    "prompt_id": sample.prompt_id if sample.prompt_id else sample.id,
                    "target_id": sample.target_id if sample.target_id else sample.id,
                    "language": sample.language,
                    "prompt_language": sample.prompt_language,
                    "manifest_path": str(sample.manifest_path) if sample.manifest_path else "",
                }

            except (FileNotFoundError, OSError, ValueError) as exc:
                if current_idx not in self.bad_indices:
                    message = (
                        f"[Warn] Skipping sample '{sample.id}' due to load failure: {exc}. "
                        "It will be removed from the dataset for this run."
                    )
                    print(message)
                    self.bad_indices.add(current_idx)

                self.samples[current_idx] = None
                if len(self.bad_indices) >= len(self.samples):
                    raise RuntimeError("All samples were marked invalid; cannot continue.")

                idx = current_idx + 1
                attempts += 1
                continue

        raise RuntimeError("Exceeded retry budget while sampling training data.")


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])
    cond_lengths = torch.stack([item["condition_len"] for item in batch])

    ids = [item["id"] for item in batch]
    prompt_ids = [item.get("prompt_id", item["id"]) for item in batch]
    target_ids = [item.get("target_id", item["id"]) for item in batch]
    languages = [item.get("language") for item in batch]
    prompt_languages = [item.get("prompt_language") for item in batch]
    manifest_paths = [item.get("manifest_path") for item in batch]

    return {
        "ids": ids,
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
        "languages": languages,
        "prompt_languages": prompt_languages,
        "manifest_paths": manifest_paths,
    }


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(
    cfg_path: Path,
    tokenizer: TextTokenizer,
    base_checkpoint: Path,
    device: torch.device,
    gradient_checkpointing: Optional[bool] = None,
) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size
    if gradient_checkpointing is not None:
        cfg.gpt.checkpointing = gradient_checkpointing

    model = UnifiedVoice(**cfg.gpt)
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    state_dict = filtered_state_dict

    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        state_dict[key] = param.detach().clone()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    return model.to(device)


class TrainStep(nn.Module):
    """
    Runs the loss computation inside `forward()`.

    This exists for DistributedDataParallel: DDP arms its gradient-reduction hooks
    when its own `forward()` is called. The loss here is built from
    `UnifiedVoice.get_logits()` rather than `UnifiedVoice.forward()`, so calling the
    model directly under DDP would silently skip gradient synchronisation and every
    rank would drift apart. Wrapping the computation keeps DDP in the loop.
    """

    def __init__(self, model: UnifiedVoice):
        super().__init__()
        self.model = model

    def forward(
        self,
        condition: torch.Tensor,
        text_ids: torch.Tensor,
        codes: torch.Tensor,
        emo_vec: torch.Tensor,
        text_lengths: torch.Tensor,
        code_lengths: torch.Tensor,
        use_duration_control: bool = False,
        duration_dropout: float = 0.3,
        emotion_dropout: float = 0.0,
        emotion_shuffle: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        device = text_ids.device
        batch_size = text_ids.size(0)
        use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

        # The emotion vector is *added* to the speaker latent (`condition + emo_vec`),
        # and with --emotion-source target it comes from the target clip — the same
        # speaker as the prompt. Speaker identity therefore arrives twice, so nothing
        # forces the model to keep the two roles apart, and at inference a changed
        # emotion vector drags the timbre with it.
        #
        # Zeroing the vector on a fraction of samples leaves `condition` as the only
        # identity cue on those steps, while the remainder still teach
        # emotion -> prosody. Shuffling instead substitutes another sample's vector,
        # which decorrelates harder but also makes the vector inconsistent with the
        # target's prosody, so it is off by default.
        if emotion_shuffle > 0.0 and batch_size > 1:
            swap = torch.rand(batch_size, device=device) < emotion_shuffle
            emo_vec = torch.where(swap.unsqueeze(1), emo_vec.roll(1, dims=0), emo_vec)
        if emotion_dropout > 0.0:
            drop = torch.rand(batch_size, device=device) < emotion_dropout
            emo_vec = torch.where(drop.unsqueeze(1), torch.zeros_like(emo_vec), emo_vec)

        text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
        text_inputs, text_targets = model.build_aligned_inputs_and_targets(
            text_inputs, model.start_text_token, model.stop_text_token
        )

        mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
        mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
        mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
            mel_inputs, model.start_mel_token, model.stop_mel_token
        )

        duration_free = model.speed_emb(torch.zeros_like(use_speed))
        if use_duration_control:
            duration_ctrl = model.get_duration_embeddings(code_lengths)
            if duration_dropout > 0.0:
                drop_mask = torch.rand(code_lengths.size(0), device=device) < duration_dropout
                if drop_mask.any():
                    duration_ctrl = torch.where(drop_mask.unsqueeze(1), duration_free, duration_ctrl)
        else:
            duration_ctrl = model.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat(
            (condition + emo_vec.unsqueeze(1), duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
            dim=1,
        )

        text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
        mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

        text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

        text_mask = (
            torch.arange(text_targets.size(1), device=device).unsqueeze(0)
            < (text_lengths + 1).unsqueeze(1)
        )
        mel_mask = (
            torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
            < (code_lengths + 1).unsqueeze(1)
        )

        text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
        mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

        text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
        mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

        with torch.no_grad():
            mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
            mel_targets_flat = mel_targets.reshape(-1)
            mel_mask_flat = mel_mask.reshape(-1)
            if mel_mask_flat.any():
                valid_logits = mel_logits_flat[mel_mask_flat]
                valid_targets = mel_targets_flat[mel_mask_flat]
                top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean()
            else:
                top1 = torch.zeros((), device=device)

        return text_loss, mel_loss, top1


def compute_losses(
    step_module: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_duration_control: bool = False,
    duration_dropout: float = 0.3,
    emotion_dropout: float = 0.0,
    emotion_shuffle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    condition = batch["condition"].to(device, non_blocking=True)
    text_ids = batch["text_ids"].to(device, non_blocking=True)
    codes = batch["codes"].to(device, non_blocking=True)
    emo_vec = batch["emo_vec"].to(device, non_blocking=True)
    text_lengths = batch["text_lengths"].to(device, non_blocking=True)
    code_lengths = batch["code_lengths"].to(device, non_blocking=True)

    text_loss, mel_loss, top1 = step_module(
        condition,
        text_ids,
        codes,
        emo_vec,
        text_lengths,
        code_lengths,
        use_duration_control,
        duration_dropout,
        emotion_dropout,
        emotion_shuffle,
    )
    return text_loss, mel_loss, {"mel_top1": float(top1.item())}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
    extra: Dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        # Always persist the bare UnifiedVoice weights so checkpoints stay
        # interchangeable between single-GPU and multi-GPU runs (and loadable by
        # the inference code, which knows nothing about DDP).
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "recent_checkpoints": recent_checkpoints,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_duration_control: bool = False,
    duration_dropout: float = 0.3,
    dist_ctx: Optional[DistContext] = None,
) -> Dict[str, float]:
    # Emotion dropout/shuffle are training-time augmentations only; validation runs
    # with the vector intact so the metric stays comparable across runs.
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            text_loss, mel_loss, metrics = compute_losses(
                model,
                batch,
                device,
                use_duration_control=use_duration_control,
                duration_dropout=duration_dropout,
            )
            bsz = batch["text_ids"].size(0)
            totals["text_loss"] += text_loss.item() * bsz
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    model.train()

    if dist_ctx is not None and dist_ctx.enabled:
        # Every rank saw a different shard, so pool the weighted sums before
        # dividing; otherwise each rank would report (and log) a different figure.
        packed = torch.tensor(
            [totals["text_loss"], totals["mel_loss"], totals["mel_top1"], float(count)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals["text_loss"], totals["mel_loss"], totals["mel_top1"] = packed[:3].tolist()
        count = int(packed[3].item())

    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    dist_ctx = init_distributed(args.ddp_backend)

    # Identical seed everywhere so all ranks start from the same weights; the data
    # order is decorrelated by DistributedSampler rather than by the seed.
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_ctx.local_rank}")
    else:
        device = torch.device("cpu")

    def log(message: str) -> None:
        if dist_ctx.is_main:
            print(message)

    if dist_ctx.enabled:
        log(
            f"[Info] Distributed run: world_size={dist_ctx.world_size} "
            f"backend={args.ddp_backend}"
        )

    output_dir = args.output_dir.resolve()
    log_root = output_dir / "logs"
    if dist_ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
    dist_ctx.barrier()
    run_name = (
        f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.environ.get("INDEXTTS_RUN_NAME") is None
        else os.environ["INDEXTTS_RUN_NAME"]
    )
    log_dir = log_root / run_name
    # Only rank 0 writes TensorBoard events, otherwise the ranks fight over files.
    writer = SummaryWriter(log_dir=str(log_dir)) if dist_ctx.is_main else None

    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(
        args.config,
        tokenizer,
        args.base_checkpoint,
        device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.freeze_unused_modules:
        # The training path consumes precomputed conditioning/emotion features, so
        # these encoders never see a gradient.
        frozen = 0
        for name in (
            "conditioning_encoder",
            "perceiver_encoder",
            "emo_conditioning_encoder",
            "emo_perceiver_encoder",
            "emovec_layer",
            "emo_layer",
        ):
            submodule = getattr(model, name, None)
            if submodule is None:
                continue
            for param in submodule.parameters():
                param.requires_grad = False
                frozen += 1
        log(f"[Info] Froze {frozen} parameter tensors in unused conditioning modules.")

    train_specs = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs = parse_manifest_specs(args.val_manifests, "--val-manifest")

    log(f"[Info] Emotion vector source: {args.emotion_source}")
    log("[Info] Loading training manifests...")
    train_dataset = JapaneseGPTDataset(train_specs, emotion_source=args.emotion_source)
    log("[Info] Loading validation manifests...")
    val_dataset = JapaneseGPTDataset(val_specs, emotion_source=args.emotion_source)

    manifest_metadata = {
        "train": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in train_dataset.manifest_summaries
        ],
        "val": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in val_dataset.manifest_summaries
        ],
    }

    def checkpoint_extra(extra_type: str) -> Dict[str, object]:
        return {"type": extra_type, "manifests": manifest_metadata}

    use_cuda = torch.cuda.is_available()

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,  # keeps every rank on the same step count
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
        drop_last=dist_ctx.enabled,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )

    step_module: nn.Module = TrainStep(model)
    if dist_ctx.enabled:
        # static_graph subsumes unused-parameter detection (DDP works it out on the
        # first iteration), and the two options are mutually exclusive.
        find_unused = args.ddp_find_unused_parameters and not args.ddp_static_graph
        step_module = DistributedDataParallel(
            step_module,
            device_ids=[dist_ctx.local_rank] if use_cuda else None,
            output_device=dist_ctx.local_rank if use_cuda else None,
            find_unused_parameters=find_unused,
            static_graph=args.ddp_static_graph,
        )
        log(
            f"[Info] DDP static_graph={args.ddp_static_graph} "
            f"find_unused_parameters={find_unused}"
        )

    log(
        f"[Info] Effective batch = {args.batch_size} x {dist_ctx.world_size} ranks "
        f"x {args.grad_accumulation} accum = "
        f"{args.batch_size * dist_ctx.world_size * args.grad_accumulation}"
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    start_epoch = 0
    recent_checkpoints: List[str] = []
    last_saved_step: int | None = None

    resume_path: str | None = None
    if args.resume:
        if args.resume == "auto":
            candidate = output_dir / "latest.pth"
            if candidate.exists():
                resume_path = str(candidate)
        else:
            resume_path = args.resume
    if resume_path:
        # Every rank restores the same checkpoint so weights and optimiser state
        # stay in lockstep from step one.
        checkpoint = torch.load(resume_path, map_location=device)
        unwrap_model(step_module).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        recent_checkpoints = checkpoint.get("recent_checkpoints", [])
        last_saved_step = checkpoint.get("step")
        log(f"[Info] Resumed from {resume_path} at epoch {start_epoch}, step {global_step}.")

    step_module.train()
    optimizer.zero_grad(set_to_none=True)

    save_every = 1000
    best_val = math.inf

    if args.val_interval > 0 and global_step > 0:
        # If we resumed exactly on a validation boundary we postpone evaluation until
        # after the next training step to avoid running validation before training.
        log("[Info] Skipping startup validation; will evaluate after next training interval.")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            # Reshuffles differently each epoch, and identically across ranks.
            train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                text_loss, mel_loss, metrics = compute_losses(
                    step_module,
                    batch,
                    device,
                    use_duration_control=args.use_duration_control,
                    duration_dropout=args.duration_dropout,
                    emotion_dropout=args.emotion_dropout,
                    emotion_shuffle=args.emotion_shuffle,
                )
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            if use_amp:
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_interval == 0 and dist_ctx.is_main:
                    writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"text_loss={text_loss.item():.4f} mel_loss={mel_loss.item():.4f} "
                        f"mel_top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                if args.val_interval > 0 and global_step > 0 and global_step % args.val_interval == 0:
                    # All ranks must take part: evaluate() ends in a collective.
                    val_metrics = evaluate(
                        step_module,
                        val_loader,
                        device,
                        use_duration_control=args.use_duration_control,
                        duration_dropout=args.duration_dropout,
                        dist_ctx=dist_ctx,
                    )
                    if dist_ctx.is_main:
                        writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                        writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                        writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                        print(
                            f"[Val] epoch={epoch + 1} step={global_step} "
                            f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                            f"mel_top1={val_metrics['mel_top1']:.4f}"
                        )
                    if val_metrics["mel_loss"] < best_val:
                        best_val = val_metrics["mel_loss"]

                if global_step % save_every == 0:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    recent_checkpoints.append(str(ckpt_path))
                    if dist_ctx.is_main:
                        save_checkpoint(
                            ckpt_path,
                            step_module,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            global_step,
                            recent_checkpoints,
                            extra=checkpoint_extra("step"),
                        )
                        torch.save(
                            {
                                "model": unwrap_model(step_module).state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict() if scaler else None,
                                "epoch": epoch,
                                "step": global_step,
                                "recent_checkpoints": recent_checkpoints,
                                "manifests": manifest_metadata,
                            },
                            output_dir / "latest.pth",
                        )
                        while len(recent_checkpoints) > 3:
                            obsolete = recent_checkpoints.pop(0)
                            try:
                                os.remove(obsolete)
                            except OSError:
                                pass
                    else:
                        while len(recent_checkpoints) > 3:
                            recent_checkpoints.pop(0)
                    # Hold the other ranks until the write lands, so a crash mid-save
                    # cannot leave some ranks running ahead of a partial checkpoint.
                    dist_ctx.barrier()
                    last_saved_step = global_step

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.val_interval == 0:
            val_metrics = evaluate(
                step_module,
                val_loader,
                device,
                use_duration_control=args.use_duration_control,
                duration_dropout=args.duration_dropout,
                dist_ctx=dist_ctx,
            )
            if dist_ctx.is_main:
                writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                print(
                    f"[Val] epoch={epoch + 1} step={global_step} "
                    f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                    f"mel_top1={val_metrics['mel_top1']:.4f}"
                )
            if val_metrics["mel_loss"] < best_val:
                best_val = val_metrics["mel_loss"]


    if global_step > 0 and last_saved_step != global_step and dist_ctx.is_main:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        recent_checkpoints.append(str(ckpt_path))
        save_checkpoint(
            ckpt_path,
            step_module,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            recent_checkpoints,
            extra=checkpoint_extra("step-final"),
        )
        torch.save(
            {
                "model": unwrap_model(step_module).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "step": global_step,
                "recent_checkpoints": recent_checkpoints,
                "manifests": manifest_metadata,
            },
            output_dir / "latest.pth",
        )
        while len(recent_checkpoints) > 3:
            obsolete = recent_checkpoints.pop(0)
            try:
                os.remove(obsolete)
            except OSError:
                pass

    dist_ctx.barrier()
    if writer is not None:
        writer.close()
    log("Training complete.")
    if dist_ctx.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
