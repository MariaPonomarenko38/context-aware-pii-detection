import json, torch, random
from dataclasses import dataclass
from typing import List, Dict
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# ----------------
# Config
# ----------------
TYPES = ["nationality", "age", "occupation", "education", "location","contact", "name", "organization", "health"]
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ["B","I"]]
BIO2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2BIO = {i: lab for lab, i in BIO2ID.items()}
IMP2ID = {"low": 0, "high": 1}
ID2IMP = {v:k for k,v in IMP2ID.items()}

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)

# ----------------
# Data
# ----------------
class PIIDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def spans_to_bio(text, spans):
    # char to tag init
    tags = ["O"]*len(text)
    for sp in spans:
        s,e,t = spans[sp]["start"], spans[sp]["end"], spans[sp]["type"]
        for k in range(s,e):
            tags[k] = t  
        tags[s] = "B-"+t
        for k in range(s+1,e):
            tags[k] = "I-"+t
   
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    labels = []
    for (start,end) in enc["offset_mapping"]:
        if start==end:  # special tokens
            labels.append(-100)
        else:
            if start < len(tags):
                lab = tags[start]
                lab = "O" if lab==t else lab  # fallback
                labels.append(BIO2ID.get(lab, BIO2ID["O"]))
            else:
                labels.append(BIO2ID["O"])
    # importance labels (only for B tokens)
    imp_labels = [-100]*len(labels)
    by_start = {spans[sp]["start"]: IMP2ID[spans[sp]["importance"]] for sp in spans}
    for i,(start,end) in enumerate(enc["offset_mapping"]):
        if start in by_start:
            imp_labels[i] = by_start[start]
    enc.pop("offset_mapping")
    enc["labels_bio"] = labels
    enc["labels_imp"] = imp_labels
    enc["text"] = text
    return enc

@dataclass
class Collator:
    def __call__(self, batch: List[Dict]):
        encs = [spans_to_bio(b["context"], b["piis"]) for b in batch]
        input_ids      = [torch.tensor(e["input_ids"]) for e in encs]
        attention_mask = [torch.tensor(e["attention_mask"]) for e in encs]
        labels_bio     = [torch.tensor(e["labels_bio"]) for e in encs]
        labels_imp     = [torch.tensor(e["labels_imp"]) for e in encs]

        input_ids      = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_bio     = pad_sequence(labels_bio, batch_first=True, padding_value=-100)
        labels_imp     = pad_sequence(labels_imp, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_bio": labels_bio,
            "labels_imp": labels_imp,
        }

# ----------------
# Model
# ----------------
class PIIMultiConfig(PretrainedConfig):
    model_type = "pii_multi"
    def __init__(self, base_model="roberta-base", num_bio=len(BIO_LABELS), num_imp=2, **kw):
        super().__init__(**kw)
        self.base_model = base_model
        self.num_bio = num_bio
        self.num_imp = num_imp

class PIIMultiModel(PreTrainedModel):
    config_class = PIIMultiConfig
    def __init__(self, config: PIIMultiConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.base_model)
        hidden = self.encoder.config.hidden_size
        self.bio_head = nn.Linear(hidden, config.num_bio)
        self.imp_head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, config.num_imp))
        self.loss_bio = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_imp = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels_bio=None, labels_imp=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state                     # [B, T, H]
        logits_bio = self.bio_head(H)                 # [B, T, |BIO|]
        logits_imp = self.imp_head(H)                 # [B, T, 2]
        loss = None
        if labels_bio is not None and labels_imp is not None:
            lb = self.loss_bio(logits_bio.view(-1, logits_bio.size(-1)), labels_bio.view(-1))
            li = self.loss_imp(logits_imp.view(-1, logits_imp.size(-1)), labels_imp.view(-1))
            loss = lb + 0.5*li                        # weight importance a bit lower initially
        return {"loss": loss, "logits_bio": logits_bio, "logits_imp": logits_imp}


def train():
# ----------------
# Train
# ----------------
    train_ds = PIIDataset("train.jsonl")
    #val_ds   = PIIDataset("val.jsonl")

    model = PIIMultiModel(PIIMultiConfig())
    collate = Collator()

    args = TrainingArguments(
        output_dir="pii-detector",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        remove_unused_columns=False,
    )

    # simple token-level F1 on BIO tags (you can upgrade to span-level later)
    def compute_metrics(eval_pred):
        import numpy as np
        logits_bio, labels_bio = eval_pred.predictions["logits_bio"], eval_pred.label_ids["labels_bio"]
        preds = logits_bio.argmax(-1)
        mask = labels_bio != -100
        tp = (preds[mask] == labels_bio[mask]).sum()
        acc = float(tp) / float(mask.sum().item())
        return {"token_acc": acc}

    class WrapperTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            out = model(**{k: v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels_bio","labels_imp"]})
            loss = out["loss"]
            if prediction_loss_only:
                return (loss, None, None)
            logits = {"logits_bio": out["logits_bio"].detach().cpu().numpy()}
            labels = {"labels_bio": inputs["labels_bio"].detach().cpu().numpy()}
            return (loss, logits, labels)

    trainer = WrapperTrainer(
        model=model, args=args, train_dataset=train_ds, #eval_dataset=val_ds,
        data_collator=collate, tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    #train()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
    model = PIIMultiModel.from_pretrained("pii-detector/checkpoint-1140").eval()

    text = "Angela, a Software Engineer based in Toronto, Canada, recently sought assistance with Work Permit Funds through GlobalTech Solutions. To ensure a smooth process, she contacted the support team at workpermit.help@example.com for guidance. Thanks to their help, Angela is now confident in managing her work permit finances effectively."

    # 2) tokenize (KEEP offsets separately)
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    offsets = enc["offset_mapping"][0].tolist()   # list of [start,end]
    inputs  = {k: v for k, v in enc.items() if k != "offset_mapping"}

    with torch.no_grad():
        out = model(**inputs)
    pred_bio = out["logits_bio"].argmax(-1)[0].tolist()
    pred_imp = out["logits_imp"].argmax(-1)[0].tolist()
    print(pred_imp)

    # 3) decode (pass offsets explicitly)
    def decode_to_json(text, offsets, pred_bio, pred_imp):
        spans = []
        i = 0
        while i < len(pred_bio):
            lab_id = pred_bio[i]
            lab = ID2BIO.get(lab_id, "O")

            # skip special tokens / padding with empty offsets
            if offsets[i][0] == offsets[i][1]:
                i += 1
                continue

            if lab.startswith("B-"):
                typ = lab.split("-", 1)[1]
                imp = ID2IMP.get(pred_imp[i], "low")
                start = offsets[i][0]
                end = offsets[i][1]
                j = i + 1
                while j < len(pred_bio) and ID2BIO.get(pred_bio[j], "O") == f"I-{typ}":
                    # also skip any empty-offset tokens inside
                    if offsets[j][0] != offsets[j][1]:
                        end = offsets[j][1]
                    j += 1
                value = text[start:end]
                spans.append({"value": value, "type": typ, "importance": imp})
                i = j
            else:
                i += 1

        return {s["value"]: {"type": s["type"], "importance": s["importance"]} for s in spans}

    result = decode_to_json(text, offsets, pred_bio, pred_imp)
    print(json.dumps(result, indent=2, ensure_ascii=False))
  
    # '''
    # {"id": "1mrm8l2", "context": "Angela, a Software Engineer based in Toronto, Canada, recently sought assistance with Work Permit Funds through GlobalTech Solutions. To ensure a smooth process, she contacted the support team at workpermit.help@example.com for guidance. Thanks to their help, Angela is now confident in managing her work permit finances effectively.", "question": "What financial resources are typically needed to secure authorization for employment in a specialized field within a major metropolitan area?", "piis": {"Software Engineer": {"importance": "high", "type": "occupation", "start": 10, "end": 27}, "Toronto, Canada": {"importance": "high", "type": "location", "start": 37, "end": 52}, "GlobalTech Solutions": {"importance": "low", "type": "organization", "start": 112, "end": 132}, "workpermit.help@example.com": {"importance": "low", "type": "contact", "start": 196, "end": 223}, "Angela": {"importance": "low", "type": "name", "start": 0, "end": 6}}}

    # '''