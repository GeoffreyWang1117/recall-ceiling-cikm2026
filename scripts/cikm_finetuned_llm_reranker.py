#!/usr/bin/env python3
"""
CIKM 2026 — Fine-tuned LLM reranker (KDD rebuttal: addresses 'no fine-tuned
LLM rerankers' weakness from program-chair decision, 2026-05-15).

Protocol mirrors scripts/cikm_neural_rerankers_disjoint.py exactly:
  - 250 disjoint train users (their last training interaction = positive label)
  - 500 evaluation users (standard test split)
  - K=100 CF candidates per user
  - Pointwise binary task: "Will user like this item? Yes/No"
  - QLoRA 4-bit on Llama-3.2-3B-Instruct, r=16, alpha=32, 3 epochs
  - Eval: rank K candidates by P(Yes) - P(No), compute NDCG@10
  - Paired Wilcoxon (test fine-tuned vs CF) + Holm-Bonferroni across 3 datasets

Usage:
  python3 scripts/cikm_finetuned_llm_reranker.py \
      --datasets amazon_beauty_sampled amazon_movies_sampled amazon_electronics_sampled \
      --n_users 500 --n_train 250 --K 100 --epochs 3 --out experiments/logs/cikm_finetune_llm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cikm_controlled_recall_sweep import (  # type: ignore
    CFSVD, ndcg_at_k, safe_wilcoxon, bootstrap_ci,
)

DATA = REPO / "data" / "processed"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


# ============================================================================
# Prompt construction
# ============================================================================

def get_item_text(meta: dict, item_id: int) -> str:
    """Return short item descriptor: 'Title (Category)' truncated."""
    rec = meta.get(str(item_id)) or meta.get(item_id)
    if rec is None:
        return f"item-{item_id}"
    title = (rec.get("title") or rec.get("text") or "")[:80]
    cat = (rec.get("category") or "")[:30]
    return f"{title} ({cat})" if cat else title or f"item-{item_id}"


def make_prompt(history_titles: List[str], candidate_title: str, history_k: int = 5) -> str:
    hist = "; ".join(history_titles[-history_k:]) or "(no history)"
    return (
        "You are a recommendation reranker. Decide whether the user is likely to "
        "enjoy the candidate item based on their interaction history.\n"
        f"User history (recent items): {hist}\n"
        f"Candidate item: {candidate_title}\n"
        "Answer with a single token: Yes or No."
    )


# ============================================================================
# Training data construction
# ============================================================================

def build_train_examples(train_df, user_history, eval_users, cf, meta,
                         K: int = 100, n_train: int = 250, n_neg: int = 5,
                         seed: int = 42):
    """Return list of {'prompt': str, 'label': 'Yes'|'No'} ~= 250*(1+n_neg) examples."""
    rng = np.random.default_rng(seed)
    eval_set = set(eval_users)
    last_train = train_df.sort_values("user_id").groupby("user_id").tail(1)
    held_out = dict(zip(last_train["user_id"].astype(int).values,
                        last_train["item_id"].astype(int).values))
    pool = [u for u, h in user_history.items()
            if u not in eval_set and len(h) >= 4 and u in held_out]
    if len(pool) < 50:
        return []

    train_users = list(rng.choice(pool, size=min(n_train, len(pool)), replace=False))
    examples = []
    for uid in train_users:
        ho_item = held_out[uid]
        hist = [h for h in user_history.get(uid, []) if h != ho_item]
        cands, _ = cf.topk(uid, K, hist)
        if len(cands) == 0:
            continue
        hist_titles = [get_item_text(meta, h) for h in hist[-5:]]
        # Positive
        examples.append({
            "prompt": make_prompt(hist_titles, get_item_text(meta, ho_item)),
            "label": "Yes",
        })
        # Negatives from CF top-K minus the positive
        neg_pool = [c for c in cands if c != ho_item]
        if not neg_pool:
            continue
        for cid in rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False):
            examples.append({
                "prompt": make_prompt(hist_titles, get_item_text(meta, int(cid))),
                "label": "No",
            })
    return examples


# ============================================================================
# Training (QLoRA pointwise)
# ============================================================================

def train_lora(examples, model_id=MODEL_ID, out_dir=None, epochs=3, lr=2e-4,
               batch_size=8, max_len=512, seed=42, log_prefix=""):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Format as instruction-style chat
    def format_ex(ex):
        msgs = [{"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["label"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list(examples).map(format_ex)

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=max_len,
                  padding="max_length")
        labels = []
        for ids, am in zip(out["input_ids"], out["attention_mask"]):
            lbl = [tok_id if m == 1 else -100 for tok_id, m in zip(ids, am)]
            labels.append(lbl)
        out["labels"] = labels
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    collator = None  # already padded to max_len

    args = TrainingArguments(
        output_dir=str(out_dir) if out_dir else "/tmp/lora_out",
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=20, save_strategy="no",
        bf16=True, optim="paged_adamw_8bit", seed=seed,
        report_to="none", dataloader_num_workers=2,
    )
    trainer_kwargs = {"model": model, "args": args, "train_dataset": ds}
    if collator is not None:
        trainer_kwargs["data_collator"] = collator
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    if out_dir:
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
    return model, tok


# ============================================================================
# Eval (rank K=100 candidates per user by P(Yes) - P(No))
# ============================================================================

@torch.no_grad()
def score_candidates(model, tok, prompts, batch_size=16, max_len=512):
    """Return numpy array of P(Yes) - P(No) logits per prompt."""
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    model.eval()
    scores = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(prompts), batch_size), desc="    score", leave=False):
        batch = prompts[i:i + batch_size]
        chats = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ) for p in batch]
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        out = model(**enc)
        logits = out.logits  # [B, T, V]
        # Last non-pad token logits per sequence
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0)), last_idx, :]
        diff = (last_logits[:, yes_id] - last_logits[:, no_id]).float().cpu().numpy()
        scores.extend(diff.tolist())
    return np.array(scores)


def eval_on_dataset(model, tok, eval_users, user_cands, user_history, test_gt,
                    meta, K=100, batch_size=16):
    """Return list of NDCG@10 per user (re-ranked by LLM)."""
    ndcgs = []
    for uid in tqdm(eval_users, desc="  eval users"):
        cands = user_cands[uid]
        if len(cands) == 0:
            ndcgs.append(0.0)
            continue
        hist_titles = [get_item_text(meta, h) for h in user_history.get(uid, [])[-5:]]
        prompts = [make_prompt(hist_titles, get_item_text(meta, int(c))) for c in cands]
        s = score_candidates(model, tok, prompts, batch_size=batch_size)
        order = np.argsort(-s)
        ranked = [cands[i] for i in order]
        ndcgs.append(ndcg_at_k(ranked, test_gt.get(uid, []), k=10))
    return ndcgs


# ============================================================================
# Per-dataset run
# ============================================================================

def run_dataset(name: str, args, out_log: dict):
    print(f"\n{'='*72}\nDATASET: {name.upper()}\n{'='*72}")
    ds_path = DATA / name
    train_df = pd.read_parquet(ds_path / "train.parquet").astype({"user_id": int, "item_id": int})
    test_df = pd.read_parquet(ds_path / "test.parquet").astype({"user_id": int, "item_id": int})
    with open(ds_path / "item_metadata.json") as f:
        meta = json.load(f)
    rng = np.random.default_rng(args.seed)

    cf = CFSVD(n_factors=128); cf.fit(train_df)
    user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_gt = test_df.groupby("user_id")["item_id"].apply(list).to_dict()
    eligible = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]
    n_eval = min(args.n_users, len(eligible))
    eval_users = list(rng.choice(eligible, size=n_eval, replace=False))

    # Pre-compute CF top-K + CF NDCG baseline
    user_cands = {}
    cf_ndcgs = []
    for uid in eval_users:
        cands, _ = cf.topk(uid, args.K, user_history.get(uid, []))
        user_cands[uid] = list(map(int, cands))
        cf_ndcgs.append(ndcg_at_k(user_cands[uid], test_gt.get(uid, []), k=10))
    cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_ndcgs, seed=0)
    print(f"  CF NDCG@10 = {cf_mean:.4f}  CI=[{cf_lo:.4f},{cf_hi:.4f}]  n={len(eval_users)}")

    # Training examples (disjoint pool)
    print("  Building disjoint training pool...")
    train_ex = build_train_examples(
        train_df, user_history, eval_users, cf, meta,
        K=args.K, n_train=args.n_train, n_neg=args.n_neg, seed=args.seed,
    )
    print(f"  Training examples: {len(train_ex)}  "
          f"(pos rate = {sum(1 for e in train_ex if e['label']=='Yes')/max(len(train_ex),1):.3f})")
    if len(train_ex) < 100:
        print("  SKIP: too few training examples")
        out_log[name] = {"error": "insufficient_train"}
        return

    # Train LoRA
    out_dir = REPO / "experiments" / "checkpoints" / "finetune_llm_reranker" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Training LoRA -> {out_dir}")
    model, tok = train_lora(
        train_ex, out_dir=out_dir, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch, max_len=args.max_len, seed=args.seed, log_prefix=name,
    )

    # Eval on test users
    print("  Evaluating on test users...")
    ft_ndcgs = eval_on_dataset(
        model, tok, eval_users, user_cands, user_history, test_gt, meta,
        K=args.K, batch_size=args.eval_batch,
    )
    ft_mean, ft_lo, ft_hi = bootstrap_ci(ft_ndcgs, seed=2)
    p = safe_wilcoxon(ft_ndcgs, cf_ndcgs)
    delta = (ft_mean - cf_mean) / max(cf_mean, 1e-9) * 100
    print(f"  FT-LLM NDCG@10 = {ft_mean:.4f}  CI=[{ft_lo:.4f},{ft_hi:.4f}]  "
          f"delta_vs_CF={delta:+.1f}%  paired_Wilcoxon_p={p:.4f}")

    out_log[name] = {
        "n_eval": len(eval_users),
        "n_train": len(train_ex),
        "K": args.K,
        "model": MODEL_ID,
        "lora_r": 16, "lora_alpha": 32, "epochs": args.epochs, "lr": args.lr,
        "cf_ndcg": cf_mean, "cf_ci": [cf_lo, cf_hi],
        "ft_ndcg": ft_mean, "ft_ci": [ft_lo, ft_hi],
        "delta_pct_vs_cf": delta,
        "paired_wilcoxon_p": p,
        "ft_scores_len": len(ft_ndcgs),
    }
    # free GPU before next dataset
    del model
    torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon_beauty_sampled", "amazon_movies_sampled",
                             "amazon_electronics_sampled"])
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--n_train", type=int, default=250)
    ap.add_argument("--n_neg", type=int, default=5)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/logs/cikm_finetune_llm.json")
    args = ap.parse_args()

    log = {"args": vars(args), "results": {}}
    for ds in args.datasets:
        try:
            run_dataset(ds, args, log["results"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            log["results"][ds] = {"error": str(e)}

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
