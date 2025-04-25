import os, re, math, json, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from safetensors.torch import save_file
from bitsandbytes.optim import Adam8bit
from collections import Counter, OrderedDict
from tqdm import tqdm

# ========== 模型配置 ==========

default_config = {
    "hidden_size": 512,
    "num_layers": 3,
    "num_heads": 8,
    "num_experts": 4,
    "expert_loss": 0.01,
    "rope_dim": 64,
    "rope_base": 10000,
    "forward_dim": 1280,
    "window_size": 1024,
    "vocab_size": 16000,
    "max_seq_length": 1024,
    "batch_size": 8,
    "split_valid": 0.1,
    "weight_decay": 0.1,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "betas_range": (0.9, 0.999),
    "global_tokens": {
        "<|padding|>": 0,
        "<|unknown|>": 1},
    "special_tokens": {
        "<|system|>": 2,
        "<|user|>": 3,
        "<|think|>": 4,
        "<|assistant|>": 5,
        "<|function|>": 6,
        "<|end|>": 7,
        "\\n": 8}
}

# ========== 前饋專家 ==========

class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config["num_experts"]
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.experts = nn.ModuleList([GEGLU(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config["hidden_size"], self.num_experts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([exp(x_flat) for exp in self.experts], dim=1)
        weighted = (expert_outputs * gate_probs.unsqueeze(-1)).sum(dim=1)
        utilization = gate_probs.mean(dim=0)
        util_loss = ((utilization - 1.0/self.num_experts)**2).sum()
        out = weighted.view(B, T, D)
        return self.dropout(out), util_loss

# ========== 自注意力 ==========

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = config["hidden_size"] // config["num_heads"]
        assert config["rope_dim"] <= hd
        self.num_heads = config["num_heads"]
        self.head_dim = hd
        self.qkv = nn.Linear(config["hidden_size"], 3*self.num_heads*self.head_dim)
        self.out_proj = nn.Linear(self.num_heads*self.head_dim, config["hidden_size"])
        self.attn_dropout = nn.Dropout(config["dropout_rate"])
        if config["rope_dim"] > 0:
            self.rotary = RotaryEmbedding(config)

    def forward(self, x, mask=None, input_ids=None, pos_offset=0):
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        if hasattr(self, 'rotary'):
            cos, sin = self.rotary(q, offset=pos_offset)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask, -torch.finfo(q.dtype).max)
        w = torch.softmax(attn, dim=-1)
        w = self.attn_dropout(w)
        out = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)

# ========== 位置編碼 ==========

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        inv_freq = 1.0 / (config["rope_base"] ** (torch.arange(0, config["rope_dim"], 2).float() / config["rope_dim"]))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        B, H, T, _ = x.shape
        pos = torch.arange(offset, offset + T, device=x.device).float()
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin

def apply_rotary(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ========== 殘差模塊 ==========

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["hidden_size"])
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config["hidden_size"])
        self.ffn = MoEFeedForward(config)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x, mask=None, input_ids=None, pos_offset=0):
        res = x
        x = res + self.dropout(self.attn(self.ln1(x), mask, input_ids, pos_offset))
        out, util = self.ffn(self.ln2(x))
        return x + self.dropout(out), util

# ========== 激活函數 ==========

class GEGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config["hidden_size"], config["forward_dim"] * 2)
        self.out = nn.Linear(config["forward_dim"], config["hidden_size"])

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.out(x1 * F.gelu(x2))

# ========== 聊天模型 ==========

class ChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_layers"])])
        self.ln_f = nn.LayerNorm(config["hidden_size"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.mask_cache = {}

    def get_mask(self, T, device):
        if T not in self.mask_cache:
            i = torch.arange(T, device=device).unsqueeze(1)
            j = torch.arange(T, device=device).unsqueeze(0)
            base = (j > i) | ((i - j) >= self.config["window_size"])
            self.mask_cache[T] = base.unsqueeze(0).unsqueeze(1)
        return self.mask_cache[T]

    def forward(self, input_ids, attention_mask=None, labels=None, pos_offset=0):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        mask = self.get_mask(T, x.device)
        if attention_mask is not None:
            pad = (attention_mask == 0).view(B, 1, 1, T)
            mask = mask | pad
        util_loss = 0
        for blk in self.blocks:
            x, u = blk(x, mask, input_ids, pos_offset)
            util_loss += u
        util_loss /= len(self.blocks)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]),
                labels.view(-1),
                ignore_index=self.config["global_tokens"]["<|padding|>"])
            loss = loss + self.config["expert_loss"] * util_loss
        return {"loss": loss, "logits": logits}

# ========== 分詞器 ==========

class ChatTokenizer:
    def __init__(self, config):
        self.config = config
        self.split_tokens = OrderedDict()
        for t, idx in config["global_tokens"].items():
            self.split_tokens[t] = idx
        for t, idx in config["special_tokens"].items():
            self.split_tokens[t] = idx
        toks = sorted(self.split_tokens.keys(), key=lambda x: len(x), reverse=True)
        self.pattern = re.compile(rf"({'|'.join(list(map(re.escape, toks)))})|([a-zA-Z]+)|( )|([0-9])|(_)|([^\s])", re.UNICODE)

    def tokenize(self, text):
        return [m.group() for m in self.pattern.finditer(text)]

    def convert_tokens_to_ids(self, tokens, update=True):
        unk = self.split_tokens["<|unknown|>"]
        ids = []
        for t in tokens:
            if update and t not in self.split_tokens:
                self.split_tokens[t] = len(self.split_tokens)
            ids.append(self.split_tokens.get(t, unk))
        return ids

    def __call__(self, text, max_len=None, trunc=True, update=False):
        toks = self.tokenize(text)
        ids = self.convert_tokens_to_ids(toks, update)
        if trunc and max_len:
            ids = ids[:max_len]
        if max_len:
            pad_id = self.split_tokens["<|padding|>"]
            ids = ids + [pad_id] * (max_len - len(ids))
        mask = [1 if i != self.split_tokens["<|padding|>"] else 0 for i in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def build_split_tokens(self, stages, min_freq=1):
        texts = []
        for s in stages:
            with open(s["file_path"], encoding="utf-8") as f:
                texts.extend(f.read().splitlines())
        freq = Counter()
        for t in tqdm(texts, desc="Tokenize"):
            for tok in self.tokenize(t):
                if tok not in self.config["special_tokens"]:
                    freq[tok] += 1
        new = [t for t, c in freq.most_common() if c >= min_freq]
        avail = self.config["vocab_size"] - len(self.split_tokens)
        for t in new[:avail]:
            self.split_tokens[t] = len(self.split_tokens)

    def get_split_tokens(self):
        return self.split_tokens

    def decode(self, ids):
        inv = {idx: t for t, idx in self.split_tokens.items()}
        return ''.join(inv.get(i, "<|unknown|>") for i in ids)

# ========== 數據集 ==========

class ChatDataset(Dataset):
    def __init__(self, tokenizer, path, config):
        with open(path, encoding="utf-8") as f:
            self.data = [l for l in f.read().splitlines() if l.strip()]
        self.tokenizer = tokenizer
        self.max_len = config["max_seq_length"] + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], self.max_len, update=False)
        ids = enc["input_ids"]
        return {"input_ids": ids[:-1], "attention_mask": enc["attention_mask"][:-1], "labels": ids[1:]}

# ========== 訓練週期 ==========

def run_epoch(model, data_loader, device, pad_id, epoch, optimizer=None):
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    if optimizer is not None:
        scaler = torch.amp.GradScaler()
        optimizer.zero_grad(set_to_none=True)
        mode = "Train"
        lr = optimizer.param_groups[0]["lr"]
    else:
        mode = "Valid"
        lr = 0.0
    pbar = tqdm(data_loader, desc=f"[{mode} {epoch+1:02d}]", dynamic_ncols=True)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if optimizer is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = outputs["loss"]
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
        total_loss += loss.item()
        mask = batch["labels"] != pad_id
        correct = ((outputs["logits"].argmax(dim=-1) == batch["labels"]) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()
        acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0.0
        pbar.set_postfix({"loss": f"{loss.item():.6f}", "acc": f"{acc:.6f}", "lr": f"{lr:.6f}"})
    return total_loss / len(data_loader), total_correct / total_tokens if total_tokens > 0 else 0.0

# ========== 階段訓練 ==========

def stage_train(stages, config):
    print(f"\n========== Tokenizer ==========\n")
    tokenizer = ChatTokenizer(config)
    tokenizer.build_split_tokens(stages)
    pad_id = tokenizer.get_split_tokens()["<|padding|>"]

    model = ChatModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam8bit(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=config["betas_range"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=1)
    num_workers = min(8, os.cpu_count() or 1)

    for stage in stages:
        print(f"\n========== {stage['stage_name']} ==========\n")
        dataset = ChatDataset(tokenizer, stage["file_path"], config)
        indices = torch.randperm(len(dataset)).tolist()
        split_idx = int(len(dataset) * (1 - config["split_valid"]))
        train_loader = DataLoader(Subset(dataset, indices[:split_idx]), batch_size=config["batch_size"],
            num_workers=num_workers, persistent_workers=True, shuffle=True, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, indices[split_idx:]), batch_size=config["batch_size"],
            num_workers=num_workers, persistent_workers=True, shuffle=False, pin_memory=True)

        for epoch in range(stage["epochs"]):
            model.train()
            train_loss, train_acc = run_epoch(model, train_loader, device, pad_id, epoch, optimizer)
            model.eval()
            val_loss, val_acc = run_epoch(model, val_loader, device, pad_id, epoch)
            scheduler.step(val_loss)

            save_path = os.path.join("./model", f"{stage['stage_name']}_epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer.get_split_tokens(), f, indent=4, ensure_ascii=False)
            with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            state = model.state_dict()
            save_file(state, os.path.join(save_path, "model.safetensors"))

# ========== 初始訓練 ==========

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    stages = [
        {"stage_name": "dialogues", "file_path": "./data/daily_dialogues.txt", "epochs": 20}]
    stage_train(stages, default_config)
