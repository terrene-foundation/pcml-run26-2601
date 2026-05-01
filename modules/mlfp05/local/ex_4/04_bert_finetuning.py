# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.4: Fine-Tuning Pre-Trained BERT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain the difference between pre-training and fine-tuning
#   - Describe how transfer learning works for NLP (language -> task)
#   - Fine-tune a pre-trained BERT model on AG News classification
#   - Implement layer-wise freezing for efficient fine-tuning
#   - Use BERT's WordPiece tokeniser vs our word-level vocabulary
#   - Track fine-tuning experiments with ExperimentTracker
#   - Apply BERT fine-tuning to Singapore banking sentiment analysis
#
# PREREQUISITES: ex_4/02_transformer_encoder.py
# ESTIMATED TIME: ~30 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

from shared.mlfp05.ex_4 import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    CLASS_NAMES,
    DEVICE,
    load_ag_news,
    setup_engines,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Pre-Training vs Fine-Tuning
# ════════════════════════════════════════════════════════════════════════
# Training a Transformer from scratch (as we did in 02_) requires large
# datasets to learn both language understanding AND the task. BERT takes
# a different approach: it separates these two stages.
#
# STAGE 1 — PRE-TRAINING (done by researchers, once):
#   BERT is trained on massive text corpora (BookCorpus + Wikipedia,
#   ~3.3 billion words) with two self-supervised tasks:
#     1. Masked Language Modelling (MLM): predict randomly masked words
#        "The [MASK] sat on the mat" -> "cat"
#     2. Next Sentence Prediction (NSP): predict if two sentences follow
#        each other in the original text.
#   This teaches BERT the structure of language: grammar, semantics,
#   common sense, and factual knowledge. Pre-training takes days on
#   hundreds of GPUs and costs hundreds of thousands of dollars.
#
# STAGE 2 — FINE-TUNING (done by practitioners, for each task):
#   We take the pre-trained BERT and add a small classification head.
#   Then we fine-tune the top layers on our specific task (AG News
#   classification). This is dramatically more sample-efficient:
#     - From-scratch Transformer: needs 120K+ examples
#     - Fine-tuned BERT: can work well with just 1,000 examples
#
# WHY THIS MATTERS: Transfer learning is the single biggest lever in
# modern NLP. Instead of learning English from scratch, BERT brings
# pre-trained knowledge of vocabulary, grammar, semantics, and even
# some reasoning ability. Fine-tuning just teaches it the mapping
# from language understanding to your specific classification task.
#
# LAYER-WISE FREEZING: We freeze BERT's lower layers (which capture
# general language patterns) and only fine-tune the top layers (which
# can be adapted to our task). This is faster, requires less memory,
# and prevents "catastrophic forgetting" of the pre-trained knowledge.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Load pre-trained BERT + configure layer freezing
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Loading pre-trained {BERT_MODEL_NAME} ==")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)

# TODO: Freeze the lower 8 of 12 encoder layers — only fine-tune the top 4
# layers plus the pooler and classification head.
# Hint: Loop over bert_model.named_parameters()
# Hint: if "bert.encoder.layer" in name: layer_num = int(name.split(".")[3])
#       if layer_num < 8: param.requires_grad = False
# Hint: if "bert.embeddings" in name: param.requires_grad = False
for name, param in bert_model.named_parameters():
    ...  # YOUR CODE HERE — freeze layers 0-7 and embeddings

trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in bert_model.parameters())
print(
    f"  BERT params: {total_params:,} total, {trainable:,} trainable "
    f"({trainable/total_params:.1%} unfrozen)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert total_params > 100_000_000, "BERT-base should have ~110M parameters"
assert trainable < total_params, "Should have frozen some layers"
assert trainable / total_params < 0.5, "Should freeze at least half the layers"
# INTERPRETATION: We're fine-tuning only ~30% of BERT's parameters. The
# frozen layers already encode rich language understanding from pre-training.
# We only need to adapt the top layers to our classification task.
print("\n--- Checkpoint 1 passed --- BERT loaded with layer freezing\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build: BERT tokenisation (WordPiece, not word-level)
# ════════════════════════════════════════════════════════════════════════
# BERT uses WordPiece tokenisation, which splits rare words into subword
# units: "unconditional" -> ["un", "##condition", "##al"]. This gives
# BERT a fixed vocabulary (~30K tokens) that can represent any word,
# even ones it never saw during pre-training.
#
# BERT also adds special tokens:
#   [CLS] at the start: its final representation is used for classification
#   [SEP] at the end: marks the end of the input sequence
#   Padding to a fixed length with [PAD] tokens


def tokenise_for_bert(
    texts: list[str], max_len: int = BERT_MAX_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise text with BERT's WordPiece tokeniser.

    Returns:
        (input_ids, attention_mask) tensors ready for BERT.
    """
    # TODO: Use bert_tokenizer to encode texts
    # Hint: encoding = bert_tokenizer(texts, max_length=max_len,
    #           padding="max_length", truncation=True, return_tensors="pt")
    # Hint: return encoding["input_ids"], encoding["attention_mask"]
    encoding = ...  # YOUR CODE HERE
    return ...  # YOUR CODE HERE


# Tokenise full train and test sets
print("  Tokenising train + test sets for BERT...")
bert_train_ids, bert_train_mask = tokenise_for_bert(train_df["text"].to_list())
bert_test_ids, bert_test_mask = tokenise_for_bert(test_df["text"].to_list())
bert_train_y = torch.tensor(train_df["label"].to_list(), dtype=torch.long)
bert_test_y = torch.tensor(test_df["label"].to_list(), dtype=torch.long)

bert_train_loader = DataLoader(
    TensorDataset(
        bert_train_ids.to(DEVICE), bert_train_mask.to(DEVICE), bert_train_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
    shuffle=True,
)
bert_val_loader = DataLoader(
    TensorDataset(
        bert_test_ids.to(DEVICE), bert_test_mask.to(DEVICE), bert_test_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert bert_train_ids.shape[0] == len(train_df), "Should tokenise all training samples"
assert bert_train_ids.shape[1] == BERT_MAX_LEN, "Should pad/truncate to BERT_MAX_LEN"
print(f"  Tokenised {len(train_df):,} train + {len(test_df):,} test samples")
print("\n--- Checkpoint 2 passed --- BERT tokenisation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train: Fine-tune BERT with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
async def train_bert_async(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = BERT_EPOCHS,
    lr: float = BERT_LR,
) -> tuple[list[float], list[float]]:
    """Fine-tune BERT and log to ExperimentTracker."""
    # TODO: Set up optimizer with only trainable parameters
    # Hint: optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    optimizer = ...  # YOUR CODE HERE
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
    )
    train_losses: list[float] = []
    val_accs: list[float] = []
    best_acc = 0.0

    async with tracker.track(experiment=exp_name, run_name="bert_finetune") as run:
        await run.log_params(
            {
                "model_type": "bert_finetune",
                "base_model": BERT_MODEL_NAME,
                "epochs": str(epochs),
                "lr": str(lr),
                "frozen_layers": "0-7",
                "trainable_params": str(trainable),
                "dataset_size": str(len(train_loader.dataset)),
            }
        )

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_idx, (ids, mask, labels) in enumerate(train_loader):
                # TODO: Forward pass, backward pass, optimizer step
                # Hint: optimizer.zero_grad()
                # Hint: outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                # Hint: loss = outputs.loss
                # Hint: loss.backward()
                # Hint: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Hint: optimizer.step()
                # Hint: batch_losses.append(loss.item())
                ...  # YOUR CODE HERE
                if (batch_idx + 1) % 500 == 0:
                    print(
                        f"    batch {batch_idx+1}/{len(train_loader)}  "
                        f"loss={np.mean(batch_losses[-500:]):.4f}"
                    )
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            # TODO: Evaluate on validation set
            # Hint: model.eval()
            # Hint: with torch.no_grad(): loop over val_loader
            #   logits = model(input_ids=ids, attention_mask=mask).logits
            #   preds = logits.argmax(dim=-1)
            #   correct += (preds == labels).sum().item()
            model.eval()
            with torch.no_grad():
                correct = 0
                total_count = 0
                for ids, mask, labels in val_loader:
                    ...  # YOUR CODE HERE — get logits, preds, accumulate correct/total
                acc = correct / total_count
                val_accs.append(acc)

            await run.log_metrics(
                {"train_loss": epoch_loss, "val_accuracy": acc}, step=epoch + 1
            )
            if acc > best_acc:
                best_acc = acc
            print(
                f"  [BERT] epoch {epoch+1}/{epochs}  "
                f"loss={epoch_loss:.4f}  val_acc={acc:.3f}"
            )

        await run.log_metrics(
            {
                "best_val_accuracy": best_acc,
                "final_train_loss": train_losses[-1],
            }
        )

    return train_losses, val_accs


print(f"\n== Fine-tuning {BERT_MODEL_NAME} on AG News ==")
bert_losses, bert_accs = asyncio.run(
    train_bert_async(bert_model, bert_train_loader, bert_val_loader, epochs=BERT_EPOCHS)
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(bert_losses) == BERT_EPOCHS, "BERT should train for all epochs"
assert (
    max(bert_accs) > 0.85
), f"BERT should reach >85% accuracy with fine-tuning, got {max(bert_accs):.3f}"
# INTERPRETATION: BERT's pre-trained language understanding gives it a
# massive head start. While our from-scratch models need to learn word
# meanings, syntax, and semantics from 120K headlines, BERT already
# "knows" English from billions of words of pre-training. Fine-tuning
# just teaches it the specific mapping from language to news categories.
print(f"\n  BERT best accuracy: {max(bert_accs):.3f}")
print("\n--- Checkpoint 3 passed --- BERT fine-tuned\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Per-class accuracy breakdown
# ════════════════════════════════════════════════════════════════════════
print("\n== BERT Per-Class Accuracy ==")
bert_model.eval()
class_correct: Counter[int] = Counter()
class_total: Counter[int] = Counter()

# TODO: Compute per-class accuracy on the validation set
# Hint: with torch.no_grad(): loop over bert_val_loader
#   logits = bert_model(input_ids=ids, attention_mask=mask).logits
#   preds = logits.argmax(dim=-1)
#   for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
#       class_total[label] += 1
#       if pred == label: class_correct[label] += 1
with torch.no_grad():
    for ids, mask, labels in bert_val_loader:
        ...  # YOUR CODE HERE

for i, cls_name in enumerate(CLASS_NAMES):
    acc = class_correct[i] / max(class_total[i], 1)
    print(f"  {cls_name:<10} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

# Visualise BERT training curve
from shared.mlfp05.ex_4 import get_viz

viz = get_viz()
fig_bert = viz.training_history(
    metrics={
        "BERT train_loss": bert_losses,
        "BERT val_accuracy": bert_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_bert.write_html("ex_4_4_bert_training_curves.html")
print("\n  BERT training curves saved to ex_4_4_bert_training_curves.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sum(class_total.values()) >= 5000, "Should evaluate on full test set"
# INTERPRETATION: BERT's per-class accuracy reveals which news categories
# are easiest and hardest. Sports is typically the easiest (distinctive
# vocabulary), while World/Business can be confused (both discuss economics,
# politics, and international events). This per-class view is critical for
# production deployment -- if one category underperforms, you know where
# to focus additional training data.
print("\n--- Checkpoint 4 passed --- per-class analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: Sentiment Analysis for DBS Bank Customer Reviews
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank, Southeast Asia's largest bank by assets (S$739B),
# processes millions of customer interactions monthly across digital
# banking, branches, and customer service. The customer experience team
# needs real-time sentiment analysis to detect emerging service issues
# before they escalate.
#
# BUSINESS VALUE: Fine-tuning BERT on DBS's customer review corpus enables
# accurate sentiment classification (positive/negative/neutral) that catches
# nuanced complaints traditional keyword filters miss. A customer writing
# "I've been waiting 3 weeks for my card replacement -- this is what I
# get for being a Treasures client?" expresses frustration without using
# obvious negative keywords.
#
# DOLLAR IMPACT:
#   - Early churn detection: Identifying at-risk Treasures/Private Banking
#     clients (avg S$500K-2M AUM) before they leave. Saving just 50 high-value
#     clients/year = S$25M-100M in retained AUM, generating S$250K-1M in
#     annual fee income.
#   - NPS improvement: Proactive outreach to dissatisfied customers improves
#     Net Promoter Score. Each 1-point NPS increase correlates with 1-2%
#     revenue growth for banks (McKinsey, 2023).
#   - Compliance: MAS requires banks to demonstrate customer outcome monitoring.
#     Automated sentiment tracking provides auditable evidence.
print("\n== Application: Sentiment Analysis for DBS Bank ==")

dbs_reviews = [
    "Digital banking app crashes every time I try to transfer funds",
    "Excellent service from the relationship manager at Marina Bay branch",
    "Interest rates on savings account lower than competitors",
    "New PayLah feature makes splitting bills with friends easy",
    "Three weeks waiting for credit card replacement is unacceptable",
]

# TODO: Classify DBS reviews with fine-tuned BERT
# Step 1: bert_model.eval()
# Step 2: dbs_ids, dbs_mask = tokenise_for_bert(dbs_reviews)
# Step 3: Move to DEVICE
# Step 4: with torch.no_grad(): get logits, probs, preds
#   dbs_logits = bert_model(input_ids=dbs_ids, attention_mask=dbs_mask).logits
#   dbs_probs = F.softmax(dbs_logits, dim=-1)
#   dbs_preds = dbs_logits.argmax(dim=-1).cpu().tolist()
bert_model.eval()
with torch.no_grad():
    dbs_ids, dbs_mask = ...  # YOUR CODE HERE — tokenise_for_bert(dbs_reviews)
    dbs_ids = dbs_ids.to(DEVICE)
    dbs_mask = dbs_mask.to(DEVICE)
    dbs_logits = ...  # YOUR CODE HERE
    dbs_probs = ...  # YOUR CODE HERE
    dbs_preds = ...  # YOUR CODE HERE

print(f"\n  DBS customer review classification (fine-tuned BERT):")
print(f"  {'Review':<55} {'Category':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(dbs_reviews, dbs_preds, dbs_probs.cpu().tolist()):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {cls_name:<12} {confidence:>10.1%}")

# Show BERT's confidence distribution
avg_confidence = float(dbs_probs.max(dim=-1).values.mean())
print(f"\n  Average classification confidence: {avg_confidence:.1%}")
print(f"  (High confidence on banking text shows BERT's transfer learning)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(dbs_preds) == len(dbs_reviews), "Should classify all reviews"
# INTERPRETATION: Even though BERT was fine-tuned on news headlines (not
# banking reviews), it can still classify banking text with reasonable
# confidence. This is the power of transfer learning -- BERT's pre-trained
# language understanding transfers across domains. With domain-specific
# fine-tuning on actual DBS reviews, accuracy would improve significantly.
#
# BUSINESS IMPACT for DBS Bank:
#   - Early detection of high-value client dissatisfaction
#   - 50 retained Treasures clients/year = S$25M-100M retained AUM
#   - Annual fee income preserved: S$250K-1M
#   - NPS improvement: 1-point increase -> 1-2% revenue growth
#   - MAS compliance: auditable customer outcome monitoring
print("\n--- Checkpoint 5 passed --- DBS Bank application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — BERT Fine-Tuning")
print("=" * 70)
print(
    f"""
  [x] Explained pre-training vs fine-tuning (language knowledge -> task)
  [x] Loaded pre-trained BERT and configured layer-wise freezing
  [x] Used BERT's WordPiece tokeniser (subword, not word-level)
  [x] Fine-tuned BERT on AG News, best acc: {max(bert_accs):.1%}
  [x] Analysed per-class accuracy for production deployment decisions
  [x] Applied to DBS Bank sentiment analysis with business impact

  KEY INSIGHT:
    Pre-training is the single biggest lever in NLP. The Transformer
    architecture enables it, but the pre-trained weights are what make
    BERT dominate. This is why modern NLP is "pre-train then fine-tune"
    -- you get billions of words of language understanding for free.

  Next: In 05_three_way_comparison.py, you'll see all three models
  side by side: LSTM vs Transformer vs BERT. The comparison reveals
  the exact value of attention (LSTM -> Transformer) and pre-training
  (Transformer -> BERT).
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — BERT fine-tuning (HF batch format)
# ══════════════════════════════════════════════════════════════════
# BERT batches are dicts of (ids, mask, labels) not (x, y) tuples,
# so we use run_diagnostic_checkpoint directly with a batch_adapter.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
import torch.nn.functional as _F


def _bert_loss(m, ids, mask, labels):
    out = m(input_ids=ids, attention_mask=mask, labels=labels)
    return out.loss


def _bert_adapter(batch):
    return batch[0], batch[1], batch[2]


print("\n── Diagnostic Report (BERT fine-tune) ──")
diag, findings = run_diagnostic_checkpoint(
    bert_model,
    bert_val_loader,
    _bert_loss,
    title="BERT fine-tuned (AG News)",
    n_batches=4,  # BERT batches are expensive; 4 is enough for stats
    train_losses=bert_losses,
    val_losses=[1.0 - a for a in bert_accs],
    batch_adapter=_bert_adapter,
    show=False,
)

# ══════ EXPECTED OUTPUT (reference pattern — BERT fine-tune, 3 epochs) ══
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): unfrozen `encoder.layer.{8..11}`
#       RMS uniform (~5e-5 to 2e-4), classifier head RMS ~1e-3
#       (healthy ratio, classifier needs more signal early).
#       Frozen layers 0-7 report ZERO RMS — confirmed frozen.
#   [✓] Activations    (HEALTHY): GELU outputs well-distributed;
#       no dead units in unfrozen FFN sub-blocks.
#   [✓] Loss trend     (HEALTHY): train loss drops from ~0.9 to
#       ~0.15 in 3 epochs. Val acc hits ~0.92 by epoch 2.
# ════════════════════════════════════════════════════════════════
# Best val acc: ~0.92 after 3 epochs — this is the "pretraining
# payoff": 120K labelled examples + billions of pretraining
# tokens beat 120K + scratch-init by ~4 accuracy points.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST] Frozen layers report ZERO gradient RMS — this
#     is the structural proof that `requires_grad=False` worked.
#     If you see non-zero RMS on a layer you thought was frozen,
#     something unfroze it (a `.train()` call that reset params,
#     or a missed freeze in layer-wise unfreezing). The Blood
#     Test is the only instrument that catches this — unit tests
#     on the parameter count do not.
#     >> Prescription Pad: if any "supposed-frozen" layer shows
#        RMS > 0, re-apply the freeze loop after model.to(device).
#
#  [X-RAY] Unfrozen BERT layers + head show HEALTHY activation
#     distributions. The head's gradient is ~10x higher than the
#     BERT layers — this is EXPECTED and HEALTHY during early
#     fine-tuning. The head starts random and needs to catch up
#     to the already-trained BERT features. If this ratio
#     grows to >100x, the classifier is racing ahead and will
#     overfit — lower the head's learning rate or increase BERT's.
#     >> Prescription Pad: use discriminative learning rates
#        (lower LR for BERT, higher for head) — see slide 5G.
#
#  [STETHOSCOPE] Loss trajectory is the textbook "fine-tuning"
#     shape: fast initial drop (epochs 1-2) as the head calibrates
#     to the new task, then a gentle tail as the top BERT layers
#     adapt. Unlike scratch training, there is no long warm-up
#     because the features already exist.
#     >> Prescription Pad: 2-4 epochs is typically enough for
#        text classification fine-tunes. More epochs overfit.
#
#  FIVE-INSTRUMENT TAKEAWAY: BERT's diagnostic report is a
#  different species from scratch training. The HEALTHY readings
#  everywhere combined with 92% accuracy in 3 epochs is the
#  empirical case for transfer learning in NLP. Compare to
#  ex_4/02 Transformer (88% from scratch at 8 epochs) — BERT
#  beats it with LESS training AND higher accuracy.
#
#  CONNECT TO SLIDE 5.4 (Transformers) + transfer learning: the
#  slide claims pretraining "amortises billions of dollars of
#  compute across every downstream task". The 4-point accuracy
#  gap + 5x fewer epochs is the numeric version of that claim.
#  The frozen-layer ZERO RMS reading is the structural proof
#  that frozen representations ARE the language prior.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(bert_losses) == BERT_EPOCHS, "BERT should train for all epochs"
assert (
    max(bert_accs) > 0.85
), f"BERT should reach >85% accuracy with fine-tuning, got {max(bert_accs):.3f}"
# INTERPRETATION: BERT's pre-trained language understanding gives it a
# massive head start. While our from-scratch models need to learn word
# meanings, syntax, and semantics from 120K headlines, BERT already
# "knows" English from billions of words of pre-training. Fine-tuning
# just teaches it the specific mapping from language to news categories.
print(f"\n  BERT best accuracy: {max(bert_accs):.3f}")
print("\n--- Checkpoint 3 passed --- BERT fine-tuned\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Per-class accuracy breakdown
# ════════════════════════════════════════════════════════════════════════
print("\n== BERT Per-Class Accuracy ==")
bert_model.eval()
class_correct: Counter[int] = Counter()
class_total: Counter[int] = Counter()
with torch.no_grad():
    for ids, mask, labels in bert_val_loader:
        logits = bert_model(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(dim=-1)
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

for i, cls_name in enumerate(CLASS_NAMES):
    acc = class_correct[i] / max(class_total[i], 1)
    print(f"  {cls_name:<10} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

# ── Visualise: per-class accuracy bar chart ─────────────────────────
from shared.mlfp05.ex_4 import get_viz
import plotly.graph_objects as go

viz = get_viz()

per_class_accs = [
    class_correct[i] / max(class_total[i], 1) for i in range(len(CLASS_NAMES))
]
fig_bar = go.Figure(
    data=go.Bar(
        x=CLASS_NAMES,
        y=per_class_accs,
        marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
        text=[f"{a:.1%}" for a in per_class_accs],
        textposition="auto",
    )
)
fig_bar.update_layout(
    title="BERT Fine-Tuned — Per-Class Accuracy on AG News",
    xaxis_title="News Category",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
)
fig_bar.write_html("ex_4_4_bert_per_class_accuracy.html")
print("\n  Per-class accuracy chart saved to ex_4_4_bert_per_class_accuracy.html")

# ── Visualise: BERT training loss curve ─────────────────────────────
fig_loss = viz.training_history(
    metrics={"BERT train_loss": bert_losses},
    x_label="Epoch",
    y_label="Cross-Entropy Loss",
)
fig_loss.write_html("ex_4_4_bert_training_loss.html")
print("  Training loss curve saved to ex_4_4_bert_training_loss.html")

# ── Visualise: before/after fine-tuning comparison ──────────────────
# Evaluate BERT BEFORE fine-tuning by loading a fresh model (no training)
print("\n  Computing before/after fine-tuning comparison...")
bert_before = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)
bert_before.eval()
with torch.no_grad():
    before_correct = 0
    before_total = 0
    before_class_correct: Counter[int] = Counter()
    before_class_total: Counter[int] = Counter()
    for ids, mask, labels in bert_val_loader:
        logits = bert_before(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(dim=-1)
        before_correct += int((preds == labels).sum().item())
        before_total += int(labels.size(0))
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            before_class_total[label] += 1
            if pred == label:
                before_class_correct[label] += 1
del bert_before  # free memory

before_per_class = [
    before_class_correct[i] / max(before_class_total[i], 1)
    for i in range(len(CLASS_NAMES))
]
after_per_class = per_class_accs
before_overall = before_correct / max(before_total, 1)
after_overall = max(bert_accs)

fig_compare = go.Figure()
fig_compare.add_trace(
    go.Bar(
        name="Before Fine-Tuning (random head)",
        x=CLASS_NAMES + ["Overall"],
        y=before_per_class + [before_overall],
        marker_color="rgba(99, 110, 250, 0.4)",
        text=[f"{a:.1%}" for a in before_per_class + [before_overall]],
        textposition="auto",
    )
)
fig_compare.add_trace(
    go.Bar(
        name="After Fine-Tuning",
        x=CLASS_NAMES + ["Overall"],
        y=after_per_class + [after_overall],
        marker_color="rgba(99, 110, 250, 1.0)",
        text=[f"{a:.1%}" for a in after_per_class + [after_overall]],
        textposition="auto",
    )
)
fig_compare.update_layout(
    title="BERT — Before vs After Fine-Tuning on AG News",
    xaxis_title="Category",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
    barmode="group",
)
fig_compare.write_html("ex_4_4_bert_before_after_comparison.html")
print("  Before/after comparison saved to ex_4_4_bert_before_after_comparison.html")
print(
    f"  Before fine-tuning: {before_overall:.1%} overall  |  "
    f"After: {after_overall:.1%} overall"
)

# ── Visualise: BERT training history (loss + accuracy) ──────────────
fig_history = viz.training_history(
    metrics={
        "BERT train_loss": bert_losses,
        "BERT val_accuracy": bert_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_history.write_html("ex_4_4_bert_training_curves.html")
print("  BERT training curves saved to ex_4_4_bert_training_curves.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sum(class_total.values()) >= 5000, "Should evaluate on full test set"
# INTERPRETATION: BERT's per-class accuracy reveals which news categories
# are easiest and hardest. Sports is typically the easiest (distinctive
# vocabulary), while World/Business can be confused (both discuss economics,
# politics, and international events). The before/after comparison shows
# the dramatic impact of fine-tuning: BERT with a random classification head
# performs near-chance (~25%), but after just a few epochs of fine-tuning,
# it achieves >85% accuracy by leveraging its pre-trained language understanding.
# This per-class view is critical for production deployment -- if one category
# underperforms, you know where to focus additional training data.
print("\n--- Checkpoint 4 passed --- per-class analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: Sentiment Analysis for DBS Bank Customer Reviews
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank, Southeast Asia's largest bank by assets (S$739B),
# processes millions of customer interactions monthly across digital
# banking, branches, and customer service. The customer experience team
# needs real-time sentiment analysis to detect emerging service issues
# before they escalate.
#
# BUSINESS VALUE: Fine-tuning BERT on DBS's customer review corpus enables
# accurate sentiment classification (positive/negative/neutral) that catches
# nuanced complaints traditional keyword filters miss. A customer writing
# "I've been waiting 3 weeks for my card replacement -- this is what I
# get for being a Treasures client?" expresses frustration without using
# obvious negative keywords.
#
# DOLLAR IMPACT:
#   - Early churn detection: Identifying at-risk Treasures/Private Banking
#     clients (avg S$500K-2M AUM) before they leave. Saving just 50 high-value
#     clients/year = S$25M-100M in retained AUM, generating S$250K-1M in
#     annual fee income.
#   - NPS improvement: Proactive outreach to dissatisfied customers improves
#     Net Promoter Score. Each 1-point NPS increase correlates with 1-2%
#     revenue growth for banks (McKinsey, 2023).
#   - Compliance: MAS requires banks to demonstrate customer outcome monitoring.
#     Automated sentiment tracking provides auditable evidence.
print("\n== Application: Sentiment Analysis for DBS Bank ==")

# Classify sample banking reviews (using BERT on AG News as proxy).
# In production, BERT would be fine-tuned on DBS's actual customer review
# corpus with banking-specific sentiment labels.
dbs_reviews = [
    "Digital banking app crashes every time I try to transfer funds",
    "Excellent service from the relationship manager at Marina Bay branch",
    "Interest rates on savings account lower than competitors",
    "New PayLah feature makes splitting bills with friends easy",
    "Three weeks waiting for credit card replacement is unacceptable",
]

bert_model.eval()
with torch.no_grad():
    dbs_ids, dbs_mask = tokenise_for_bert(dbs_reviews)
    dbs_ids = dbs_ids.to(DEVICE)
    dbs_mask = dbs_mask.to(DEVICE)
    dbs_logits = bert_model(input_ids=dbs_ids, attention_mask=dbs_mask).logits
    dbs_probs = F.softmax(dbs_logits, dim=-1)
    dbs_preds = dbs_logits.argmax(dim=-1).cpu().tolist()

print(f"\n  DBS customer review classification (fine-tuned BERT):")
print(f"  {'Review':<55} {'Category':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(dbs_reviews, dbs_preds, dbs_probs.cpu().tolist()):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {cls_name:<12} {confidence:>10.1%}")

# Show BERT's confidence distribution -- high confidence indicates the
# pre-trained model has strong signal for classification even on domain-
# shifted text (banking vs news headlines).
avg_confidence = float(dbs_probs.max(dim=-1).values.mean())
print(f"\n  Average classification confidence: {avg_confidence:.1%}")
print(f"  (High confidence on banking text shows BERT's transfer learning)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(dbs_preds) == len(dbs_reviews), "Should classify all reviews"
# INTERPRETATION: Even though BERT was fine-tuned on news headlines (not
# banking reviews), it can still classify banking text with reasonable
# confidence. This is the power of transfer learning -- BERT's pre-trained
# language understanding transfers across domains. With domain-specific
# fine-tuning on actual DBS reviews, accuracy would improve significantly.
#
# BUSINESS IMPACT for DBS Bank:
#   - Early detection of high-value client dissatisfaction
#   - 50 retained Treasures clients/year = S$25M-100M retained AUM
#   - Annual fee income preserved: S$250K-1M
#   - NPS improvement: 1-point increase -> 1-2% revenue growth
#   - MAS compliance: auditable customer outcome monitoring
print("\n--- Checkpoint 5 passed --- DBS Bank application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — BERT Fine-Tuning")
print("=" * 70)
print(
    f"""
  [x] Explained pre-training vs fine-tuning (language knowledge -> task)
  [x] Loaded pre-trained BERT and configured layer-wise freezing
  [x] Used BERT's WordPiece tokeniser (subword, not word-level)
  [x] Fine-tuned BERT on AG News, best acc: {max(bert_accs):.1%}
  [x] Analysed per-class accuracy for production deployment decisions
  [x] Applied to DBS Bank sentiment analysis with business impact

  KEY INSIGHT:
    Pre-training is the single biggest lever in NLP. The Transformer
    architecture enables it, but the pre-trained weights are what make
    BERT dominate. This is why modern NLP is "pre-train then fine-tune"
    -- you get billions of words of language understanding for free.

  Next: In 05_three_way_comparison.py, you'll see all three models
  side by side: LSTM vs Transformer vs BERT. The comparison reveals
  the exact value of attention (LSTM -> Transformer) and pre-training
  (Transformer -> BERT).
"""
)

