"""
TraceZero - Full Upgraded Model Training v4
Model 1 : RoBERTa fine-tuned on AI4Privacy dataset
Model 2 : DistilBERT Risk Classifier (High / Medium / Low)
Model 3 : Sentence-BERT for PII deduplication / similarity
Extras  : GPU support, per-entity F1, confusion matrix, saved reports
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NER_MODEL_CHECKPOINT  = "Jean-Baptiste/roberta-large-ner-english"
RISK_MODEL_CHECKPOINT = "distilbert-base-uncased"
SBERT_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"

NER_OUTPUT_DIR   = "./model"
RISK_OUTPUT_DIR  = "./risk_model"
SBERT_OUTPUT_DIR = "./sbert_model"
REPORTS_DIR      = "./reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-EMAIL",
    "B-PHONE",
    "B-ADDR", "I-ADDR",
    "B-LOC",  "I-LOC",
    "B-DOB",
    "B-SSN"
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# AI4Privacy has many label types — map them to our schema
AI4PRIVACY_MAP = {
    # Names
    "B-NAME_STUDENT":        "B-PER",
    "I-NAME_STUDENT":        "I-PER",
    "B-NAME_EMPLOYEE":       "B-PER",
    "I-NAME_EMPLOYEE":       "I-PER",
    "B-NAME_GIVEN":          "B-PER",
    "I-NAME_GIVEN":          "I-PER",
    "B-NAME_FAMILY":         "B-PER",
    "I-NAME_FAMILY":         "I-PER",
    # Email
    "B-EMAIL":               "B-EMAIL",
    "I-EMAIL":               "B-EMAIL",
    # Phone
    "B-PHONE_NUM":           "B-PHONE",
    "I-PHONE_NUM":           "B-PHONE",
    "B-TEL_NUM":             "B-PHONE",
    "I-TEL_NUM":             "B-PHONE",
    # Address
    "B-STREET_ADDRESS":      "B-ADDR",
    "I-STREET_ADDRESS":      "I-ADDR",
    "B-ADDRESS":             "B-ADDR",
    "I-ADDRESS":             "I-ADDR",
    # Location
    "B-CITY":                "B-LOC",
    "I-CITY":                "I-LOC",
    "B-STATE":               "B-LOC",
    "I-STATE":               "I-LOC",
    "B-COUNTRY":             "B-LOC",
    "I-COUNTRY":             "I-LOC",
    "B-LOCATION":            "B-LOC",
    "I-LOCATION":            "I-LOC",
    # DOB / Date
    "B-DATE_TIME":           "B-DOB",
    "I-DATE_TIME":           "I-DOB",
    "B-DOB":                 "B-DOB",
    "I-DOB":                 "I-DOB",
    # SSN / ID
    "B-ID_NUM":              "B-SSN",
    "I-ID_NUM":              "B-SSN",
    "B-SSN":                 "B-SSN",
    "I-SSN":                 "B-SSN",
    "B-PASSPORT_NUM":        "B-SSN",
    "I-PASSPORT_NUM":        "B-SSN",
    # Default
    "O":                     "O",
}

RISK_LABELS   = ["Low", "Medium", "High"]
RISK_LABEL2ID = {l: i for i, l in enumerate(RISK_LABELS)}
RISK_ID2LABEL = {i: l for l, i in RISK_LABEL2ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_ai4privacy_dataset(max_samples: int = 5000):
    """Load AI4Privacy from HuggingFace with fixed label parsing."""
    print("\n📥 Loading AI4Privacy dataset from HuggingFace...")
    try:
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

        print(f"   Dataset columns: {ds.column_names}")
        print(f"   Sample row keys: {list(ds[0].keys())}")

        records = []
        skipped = 0

        for example in ds:
            # AI4Privacy uses 'tokens' and 'ner_tags' or 'labels'
            tokens = (example.get("tokens") or
                      example.get("words") or
                      example.get("token") or [])

            raw_labels = (example.get("ner_tags") or
                          example.get("labels") or
                          example.get("tags") or [])

            if not tokens or not raw_labels:
                skipped += 1
                continue

            if len(tokens) != len(raw_labels):
                skipped += 1
                continue

            # Convert int labels to string labels using dataset features
            if raw_labels and isinstance(raw_labels[0], int):
                try:
                    feat = ds.features.get("ner_tags") or ds.features.get("labels")
                    if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
                        label_names = feat.feature.names
                        str_labels = [label_names[l] for l in raw_labels]
                    else:
                        # fallback: map ints directly
                        str_labels = [str(l) for l in raw_labels]
                except Exception:
                    str_labels = [str(l) for l in raw_labels]
            else:
                str_labels = [str(l) for l in raw_labels]

            # Map to our label schema
            mapped = [AI4PRIVACY_MAP.get(l, "O") for l in str_labels]
            records.append({
                "tokens":   list(tokens),
                "ner_tags": [LABEL2ID.get(l, 0) for l in mapped],
            })

        print(f"   ✅ Loaded {len(records)} examples (skipped {skipped})")

        if len(records) == 0:
            raise ValueError("No valid records parsed from AI4Privacy")

        return Dataset.from_list(records)

    except Exception as e:
        print(f"   ⚠️  AI4Privacy failed: {e}")
        print("   📂 Falling back to local pii_data.csv...")
        return load_local_csv("dataset/pii_data.csv")


def load_local_csv(path: str):
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        tokens = str(row["text"]).split()
        labels = str(row["labels"]).split()
        labels = (labels + ["O"] * len(tokens))[: len(tokens)]
        records.append({
            "tokens":   tokens,
            "ner_tags": [LABEL2ID.get(l, 0) for l in labels],
        })
    print(f"   ✅ Loaded {len(records)} examples from local CSV")
    return Dataset.from_list(records)


def assign_risk(text: str) -> int:
    t = text.lower()
    if any(k in t for k in ["ssn", "social security", "dob", "date of birth",
                              "born", "passport", "id number"]):
        return RISK_LABEL2ID["High"]
    if any(k in t for k in ["@", "email", "phone", "mobile", "call",
                              "contact", "address", "street"]):
        return RISK_LABEL2ID["Medium"]
    return RISK_LABEL2ID["Low"]


def load_risk_dataset(path: str):
    df = pd.read_csv(path)
    return Dataset.from_list([
        {"text": str(r["text"]), "label": assign_risk(str(r["text"]))}
        for _, r in df.iterrows()
    ])

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   📊 Confusion matrix → {path}")


def save_json_report(data: dict, filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   📄 Report → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — RoBERTa NER
# ─────────────────────────────────────────────────────────────────────────────
def train_ner_model():
    print("\n" + "="*60)
    print("  MODEL 1: RoBERTa PII NER")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(
        NER_MODEL_CHECKPOINT, add_prefix_space=True)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids  = tokenized.word_ids(batch_index=i)
            prev_word = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_word:
                    label_ids.append(labels[wid])
                else:
                    label_ids.append(-100)
                prev_word = wid
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    seqeval = evaluate.load("seqeval")
    all_true_preds  = []
    all_true_labels = []

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_preds, true_labels = [], []
        for pred_row, label_row in zip(predictions, labels):
            tp, tl = [], []
            for p_id, l_id in zip(pred_row, label_row):
                if l_id != -100:
                    tp.append(ID2LABEL[p_id])
                    tl.append(ID2LABEL[l_id])
            true_preds.append(tp)
            true_labels.append(tl)
            all_true_preds.extend(tp)
            all_true_labels.extend(tl)
        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    # Load & tokenize
    dataset   = load_ai4privacy_dataset(max_samples=5000)
    split     = dataset.train_test_split(test_size=0.15, seed=42)
    tokenized = split.map(tokenize_and_align, batched=True,
                          remove_columns=split["train"].column_names)  # ← fix col mismatch

    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=NER_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16 if DEVICE == "cuda" else 8,
        per_device_eval_batch_size=16  if DEVICE == "cuda" else 8,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        warmup_steps=100,
        fp16=DEVICE == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=False,   # ← fix col mismatch
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("🚀 Training RoBERTa NER model...")
    trainer.train()
    trainer.save_model(NER_OUTPUT_DIR)
    tokenizer.save_pretrained(NER_OUTPUT_DIR)

    # Save reports
    print("\n📊 Saving NER evaluation reports...")
    eval_results = trainer.evaluate()
    save_json_report(eval_results, "ner_eval_results.json")

    if all_true_labels:
        unique_labels = sorted(set(all_true_labels + all_true_preds))
        report = classification_report(
            all_true_labels, all_true_preds,
            labels=unique_labels, output_dict=True, zero_division=0
        )
        save_json_report(report, "ner_per_entity_report.json")
        print("\n📈 Per-Entity F1 Scores:")
        for entity, metrics in report.items():
            if isinstance(metrics, dict) and entity not in ("accuracy", "macro avg", "weighted avg"):
                f1 = metrics.get("f1-score", 0)
                p  = metrics.get("precision", 0)
                r  = metrics.get("recall", 0)
                print(f"   {entity:20s} F1={f1:.3f}  P={p:.3f}  R={r:.3f}")

    print(f"\n✅ NER model saved → {NER_OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — DistilBERT Risk Classifier
# ─────────────────────────────────────────────────────────────────────────────
def train_risk_classifier():
    print("\n" + "="*60)
    print("  MODEL 2: DistilBERT Risk Classifier")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL_CHECKPOINT)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=128)

    accuracy_metric = evaluate.load("accuracy")
    all_preds_risk  = []
    all_labels_risk = []

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        all_preds_risk.extend(predictions.tolist())
        all_labels_risk.extend(labels.tolist())
        return accuracy_metric.compute(predictions=predictions, references=labels)

    dataset   = load_risk_dataset("dataset/pii_data.csv")
    split     = dataset.train_test_split(test_size=0.15, seed=42)
    tokenized = split.map(tokenize, batched=True,
                          remove_columns=["text"])   # keep only 'label' + tokenizer cols

    model = AutoModelForSequenceClassification.from_pretrained(
        RISK_MODEL_CHECKPOINT,
        num_labels=len(RISK_LABELS),
        id2label=RISK_ID2LABEL,
        label2id=RISK_LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=RISK_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16 if DEVICE == "cuda" else 8,
        per_device_eval_batch_size=16  if DEVICE == "cuda" else 8,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        fp16=DEVICE == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("🚀 Training Risk Classifier...")
    trainer.train()
    trainer.save_model(RISK_OUTPUT_DIR)
    tokenizer.save_pretrained(RISK_OUTPUT_DIR)

    print("\n📊 Saving Risk Classifier reports...")
    eval_results = trainer.evaluate()
    save_json_report(eval_results, "risk_eval_results.json")

    if all_preds_risk and all_labels_risk:
        save_confusion_matrix(
            all_labels_risk, all_preds_risk,
            labels=RISK_LABELS,
            title="Risk Classifier — Confusion Matrix",
            filename="risk_confusion_matrix.png"
        )
        report = classification_report(
            all_labels_risk, all_preds_risk,
            target_names=RISK_LABELS, output_dict=True, zero_division=0
        )
        save_json_report(report, "risk_classification_report.json")
        print("\n📈 Risk Classification Report:")
        for label in RISK_LABELS:
            m = report.get(label, {})
            print(f"   {label:8s}  F1={m.get('f1-score',0):.3f}  "
                  f"P={m.get('precision',0):.3f}  R={m.get('recall',0):.3f}")

    print(f"\n✅ Risk model saved → {RISK_OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Sentence-BERT (Deduplication)
# ─────────────────────────────────────────────────────────────────────────────
def save_sbert_model():
    print("\n" + "="*60)
    print("  MODEL 3: Sentence-BERT (Deduplication / Similarity)")
    print("="*60)
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        print("📥 Downloading Sentence-BERT...")
        sbert = SentenceTransformer(SBERT_MODEL_NAME)
        sbert.save(SBERT_OUTPUT_DIR)
        print(f"✅ Sentence-BERT saved → {SBERT_OUTPUT_DIR}")

        # Demo
        test_sentences = [
            "John Smith lives at 123 Main St New York",
            "John Smith residing at 123 Main Street, NY",
            "Sarah Johnson DOB 1990-01-01 SSN 123-45-6789",
        ]
        embeddings = sbert.encode(test_sentences)
        sim        = cosine_similarity(embeddings)
        print(f"\n   Similarity Demo:")
        print(f"   Same person (sent 1 vs 2): {sim[0][1]:.3f}  ← should be HIGH")
        print(f"   Diff person (sent 1 vs 3): {sim[0][2]:.3f}  ← should be LOW")

        save_json_report({
            "model": SBERT_MODEL_NAME,
            "demo_same_person_similarity": float(sim[0][1]),
            "demo_diff_person_similarity": float(sim[0][2]),
        }, "sbert_demo_report.json")

    except ImportError:
        print("⚠️  Run: pip install sentence-transformers")
    except Exception as e:
        print(f"⚠️  SBERT error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🛡️  TraceZero — Full Model Training Pipeline")
    print(f"   GPU: {'✅ ' + torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '❌ CPU only'}")

    train_ner_model()
    train_risk_classifier()
    save_sbert_model()

    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED AND SAVED!")
    print("="*60)
    print(f"   NER model   → {NER_OUTPUT_DIR}/")
    print(f"   Risk model  → {RISK_OUTPUT_DIR}/")
    print(f"   SBERT model → {SBERT_OUTPUT_DIR}/")
    print(f"   Reports     → {REPORTS_DIR}/")