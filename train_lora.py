"""QLoRA fine-tune of Qwen3.5-9B on the corridor bearing task.

The 9B weights are 19 GB, which leaves nothing on a 24 GB card once 7k video
tokens of activations arrive, so the language model is loaded 4-bit NF4 and
only LoRA adapters train. The vision tower stays frozen and in bf16: it sees
one frame at a time and nothing about it is wrong — what the base model cannot
do is carry rotation across frames, which happens in the language model.

Supervision is the worked trace from sft_data.py, so the model is trained to
lay the walk out (turns, then leg durations, then the vector sum) rather than
to name a number.

    torchrun --nproc_per_node=3 train_lora.py --data data/sft_train.jsonl
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


@dataclass
class VideoCfg:
    # fps must match the eval's, not be "as many frames as fit": sampling a
    # 20 s walk at a flat 128 frames is 6.4 fps, while eval_probes feeds it at
    # 2 fps, so the model would train on a denser video than it is ever scored
    # on — different frame count, different `<t seconds>` tags on the very
    # timestamps the trace reasons from. It is also what made this OOM.
    fps: float = 2.0
    max_frames: int = 128
    width: int = 448
    height: int = 256


class SFTSet(Dataset):
    def __init__(self, rows, cfg):
        self.rows, self.cfg = rows, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Batch = processed video + chat-formatted text, labels on the reply only.

    The reply's token count is measured on its own and masked in from the
    right. That is exact here because the reply always begins at a clean
    boundary — the chat template ends the prompt with `<think>\\n` — and it
    avoids running the video through the image processor a second time just
    to find out where the prompt stopped."""

    def __init__(self, processor, cfg, thinking=True, check=True):
        self.p, self.cfg, self.thinking = processor, cfg, thinking
        self.check = check

    def _reply(self, row):
        think = row.get("think") or ""
        if self.thinking:
            return f"{think}\n</think>\n\n{row['answer']}<|im_end|>"
        # With thinking off the template pre-fills an empty `<think></think>`
        # before the reply starts, so anything supervised inside that block is
        # unreachable at generation time. The worked trace still earns its
        # keep — it just has to be the visible answer.
        return (f"{think}\n\n{row['answer']}<|im_end|>" if think
                else f"{row['answer']}<|im_end|>")

    def __call__(self, rows):
        from eval_probes import load_frames

        texts, vids, metas, n_reply = [], [], [], []
        for r in rows:
            frames, meta, _dur = load_frames(
                r["video"], self.cfg.fps, self.cfg.max_frames,
                (self.cfg.width, self.cfg.height))
            vids.append(frames)
            metas.append(meta)
            prompt = self.p.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": r["question"]}]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking)
            reply = self._reply(r)
            texts.append(prompt + reply)
            n_reply.append(len(self.p.tokenizer(
                reply, add_special_tokens=False).input_ids))

        batch = self.p(text=texts, videos=vids, video_metadata=metas,
                       do_sample_frames=False, padding=True,
                       return_tensors="pt")
        ids = batch["input_ids"]
        labels = ids.clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, n in enumerate(n_reply):
            end = int(batch["attention_mask"][i].sum())
            labels[i, : end - n] = -100
            labels[i, end:] = -100

        if self.check:
            self.check = False          # once per process is enough
            i = 0
            kept = ids[i][labels[i] != -100]
            got = self.p.tokenizer.decode(kept)
            want = self._reply(rows[i])
            assert got.strip() == want.strip(), (
                f"label mask is off:\n got: {got!r}\nwant: {want!r}")

        batch["labels"] = labels
        return batch


def build_model(args):
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # the vision tower is small, frozen, and precision-sensitive
        llm_int8_skip_modules=["visual", "lm_head"])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": local_rank}, attn_implementation=args.attn)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    # prepare_model_for_kbit_training upcasts every non-quantized weight to
    # fp32 for stability. Harmless on a normal vocabulary; here it is not —
    # this tokenizer has 248k entries, so the embedding table alone is 1.0B
    # parameters and the upcast costs 2 GB of a 24 GB card. Nothing trains in
    # those weights (they carry no LoRA), so bf16 is enough.
    # Cast the norms too, not just the embedding: leaving fp32 layernorms in a
    # bf16 stack makes the hidden states disagree mid-layer. The LoRA weights
    # are created after this and stay fp32, which is where the precision
    # actually matters.
    for p in model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)

    # every linear in the text stack: the gated-delta (linear attention)
    # layers carry most of the sequence mixing, so leaving them out would
    # train the wrong half of the model for a task about time
    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj",
               "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
               "out_proj"]
    present = set()
    for name, mod in model.named_modules():
        if "visual" in name:
            continue
        leaf = name.split(".")[-1]
        if leaf in targets and isinstance(mod, torch.nn.Module):
            present.add(leaf)
    missing = sorted(set(targets) - present)
    if missing:
        print(f"[warn] LoRA targets not found in the text stack: {missing}")

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=sorted(present),
        exclude_modules=r".*visual.*")
    model = get_peft_model(model, lcfg)

    # Run the vision tower under no_grad. It is frozen and carries no LoRA, so
    # the only reason its activations were being kept was to reach parameters
    # that never update — roughly 13 GB of a 24 GB card spent on a backward
    # pass that cannot use it. prepare_model_for_kbit_training turns on input
    # grads for the text stack, which is what keeps checkpointing working, so
    # detaching here costs nothing that trains.
    visual = model.base_model.model.model.visual
    _forward = visual.forward

    def _frozen_forward(*a, **kw):
        with torch.no_grad():
            out = _forward(*a, **kw)
        return out.detach() if torch.is_tensor(out) else out

    visual.forward = _frozen_forward

    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="ckpt/lora_v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--fps", type=float, default=2.0,
                    help="must match eval_probes --fps")
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--hf-cache", default="/mnt/data0/ameen/hf_cache/hub")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    # checkpoint often: a flat learning curve should be visible in minutes,
    # not after a full epoch
    ap.add_argument("--no-thinking", dest="thinking",
                    action="store_false",
                    help="train under the thinking-off template, "
                         "which is how the evals run")
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to continue from, so training can\n                         be interrupted for a capability eval and picked up")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

    from transformers import AutoProcessor, Trainer, TrainingArguments

    rows = [json.loads(l) for l in (ROOT / args.data).read_text().splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # split by video, not by row: the same walk answering four questions must
    # not sit on both sides of the split
    vids = sorted({r["video"] for r in rows})
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac))
    val_vids = set(vids[:n_val])
    train = [r for r in rows if r["video"] not in val_vids]
    val = [r for r in rows if r["video"] in val_vids]

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    cfg = VideoCfg(args.fps, args.max_frames, args.width, args.height)
    model = build_model(args)

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"{len(train)} train / {len(val)} val rows "
              f"({len(vids) - n_val}/{n_val} videos)")

    targs = TrainingArguments(
        output_dir=str(ROOT / args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps" if val else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=8,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=SFTSet(train, cfg),
        eval_dataset=SFTSet(val, cfg) if val else None,
        data_collator=Collator(processor, cfg, thinking=args.thinking),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(ROOT / args.out / "final"))
    if int(os.environ.get("RANK", 0)) == 0:
        processor.save_pretrained(str(ROOT / args.out / "final"))
        print(f"-> {ROOT / args.out / 'final'}")


if __name__ == "__main__":
    main()
