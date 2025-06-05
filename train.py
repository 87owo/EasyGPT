import os, re, math, json, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from safetensors.torch import save_file
from bitsandbytes.optim import Adam8bit
from collections import Counter, OrderedDict
from tqdm import tqdm

# ================================================

default_config = {
    "hidden_size": 512,
    "ffn_hidden_size": 1536,
    "block_count": 12,
    "num_heads": 8,
    "num_kv_heads": 8,
    "rope_dim": 64,
    "rope_base": 10000,
    "vocab_size": 32000,
    "max_seq_length": 1024,
    "batch_size": 4,
    "split_valid": 0.01,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "betas_range": (0.9, 0.999),
    "layer_norm_eps": 1e-6,
    "global_tokens": {
        "<|padding|>": 0,
        "<|unknown|>": 1
    },
    "special_tokens": {
        "<|system|>": 2,
        "<|user|>": 3,
        "<|think|>": 4,
        "<|assistant|>": 5,
        "<|function|>": 6,
        "<|end|>": 7,
        "\\n": 8,
        "EasyGPT": 9,
        "87owo": 10,
    }
}

# ================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, offset=0, device=None):
        pos = torch.arange(offset, offset + seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        return cos, sin

# ================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

# ================================================

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.rope_dim = config["rope_dim"]
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.head_dim = self.hidden_size // self.num_heads
        self.rope = RotaryEmbedding(config["rope_dim"], base=config["rope_base"])

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, mask=None, pos_offset=0):
        B, T, C = x.shape
        device = x.device

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        rope_dim = min(self.rope_dim, self.head_dim)
        if rope_dim > 0:
            cos, sin = self.rope(T, pos_offset, device)
            cos = cos.squeeze(0).unsqueeze(0)
            sin = sin.squeeze(0).unsqueeze(0)
            q1, q2 = q[..., :rope_dim], q[..., rope_dim:]
            k1, k2 = k[..., :rope_dim], k[..., rope_dim:]
            q1 = q1 * cos + rotate_half(q1) * sin
            k1 = k1 * cos + rotate_half(k1) * sin
            q = torch.cat([q1, q2], dim=-1)
            k = torch.cat([k1, k2], dim=-1)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

# ================================================

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.ffn_hidden_size = config["ffn_hidden_size"]
        self.in_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.gelu(x1) * x2
        x = self.up_proj(x)
        return self.dropout(x)

# ================================================

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.attn = SelfAttention(config)
        self.ffn_norm = RMSNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x, mask=None, pos_offset=0):
        residual = x
        x = self.attn_norm(x)
        x = residual + self.dropout(self.attn(x, mask=mask, pos_offset=pos_offset))

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))
        return x

# ================================================

class ChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["block_count"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def get_mask(self, T, device):
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        mask = (j > i).unsqueeze(0).unsqueeze(1)
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None, pos_offset=0):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)
        mask = self.get_mask(T, device)

        if attention_mask is not None:
            pad_mask = (attention_mask == 0).view(B, 1, 1, T)
            mask = mask | pad_mask

        for blk in self.blocks:
            x = blk(x, mask=mask, pos_offset=pos_offset)

        x = self.norm(x)
        logits = self.head(x)
        loss = None

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]),
                labels.view(-1), ignore_index=self.config["global_tokens"]["<|padding|>"])
        return {"loss": loss, "logits": logits}

# ================================================

class ChatTokenizer:
    def __init__(self, config):
        self.config = config
        self.split_tokens = OrderedDict()
        for t, idx in config["global_tokens"].items():
            self.split_tokens[t] = idx
        for t, idx in config["special_tokens"].items():
            self.split_tokens[t] = idx

        toks = sorted(self.split_tokens.keys(), key=lambda x: len(x), reverse=True)
        self.pattern = re.compile(rf"({'|'.join(map(re.escape, toks))})|([a-zA-Z]+)|( )|([0-9])|(_)|([^\s])", re.UNICODE)

    def tokenize(self, text):
        return [m.group() for m in self.pattern.finditer(text)]

    def convert_tokens_to_ids(self, tokens, update=True):
        unk = self.split_tokens["<|unknown|>"]
        ids = []
        for t in tokens:
            if update and t not in self.split_tokens:
                if len(self.split_tokens) < self.config["vocab_size"]:
                    self.split_tokens[t] = len(self.split_tokens)
                else:
                    ids.append(unk)
                    continue
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
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def build_split_tokens(self, stages, min_freq=1):
        freq = Counter()
        for i, stage in enumerate(stages):
            path = stage["file_path"]
            with open(path, encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                for line in tqdm(f, desc=f"[Tokenize {i+1:02d}]", total=total_lines):
                    line = line.strip()
                    if not line:
                        continue
                    for tok in self.tokenize(line):
                        if tok not in self.config["special_tokens"] and tok not in self.config["global_tokens"]:
                            freq[tok] += 1

        new_tokens = [t for t, c in freq.most_common() if c >= min_freq]
        avail = self.config["vocab_size"] - len(self.split_tokens)
        for t in new_tokens[:avail]:
            self.split_tokens[t] = len(self.split_tokens)

    def get_split_tokens(self):
        return self.split_tokens

    def decode(self, ids):
        inv = {idx: t for t, idx in self.split_tokens.items()}
        return ''.join(inv.get(i, "<|unknown|>") for i in ids)

# ================================================

class ChatDataset(Dataset):
    def __init__(self, tokenizer, path, config):
        self.tokenizer = tokenizer
        self.max_len = config["max_seq_length"] + 1
        self.path = path
        self.offsets = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8", errors="replace").strip()
        enc = self.tokenizer(line, self.max_len, update=False)
        ids = enc["input_ids"]
        return {"input_ids": ids[:-1], "attention_mask": enc["attention_mask"][:-1], "labels": ids[1:]}

# ================================================

class CustomLRScheduler:
    def __init__(self, optimizer, base_lr=1e-4, gamma=0.8):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.gamma = gamma

    def step(self, epoch):
        new_lr = self.base_lr * (self.gamma ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

# ================================================

def run_epoch(model, data_loader, device, pad_id, epoch, optimizer=None, scaler=None):
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    mode = "Train" if optimizer is not None else "Valid"
    if optimizer is not None:
        lr = optimizer.param_groups[0]["lr"]
    else:
        lr = 0.0

    pbar = tqdm(data_loader, desc=f"[{mode} {epoch+1:02d}]", dynamic_ncols=True)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if optimizer is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = outputs["loss"].mean()
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

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc

# ================================================

def stage_train(stages, config):
    print(f"\n========== Tokenizer ==========\n")
    tokenizer = ChatTokenizer(config)
    tokenizer.build_split_tokens(stages)
    pad_id = tokenizer.get_split_tokens()["<|padding|>"]

    model = ChatModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam8bit(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=config["betas_range"])
    scheduler = CustomLRScheduler(optimizer, base_lr=config["learning_rate"], gamma=0.8)

    num_workers = min(8, os.cpu_count() or 1)
    scaler = torch.amp.GradScaler()
    global_epoch = 0

    for stage in stages:
        print(f"\n========== {stage['stage_name']} ==========\n")
        dataset = ChatDataset(tokenizer, stage["file_path"], config)

        indices = torch.randperm(len(dataset)).tolist()
        split_idx = int(len(dataset) * (1 - config["split_valid"]))
        train_dataset = Subset(dataset, indices[:split_idx])
        val_dataset = Subset(dataset, indices[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
            num_workers=num_workers, persistent_workers=(num_workers > 0), shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
            num_workers=num_workers, persistent_workers=(num_workers > 0), shuffle=False, pin_memory=True)

        for _ in range(stage["epochs"]):
            scheduler.step(global_epoch)
            model.train()
            train_loss, train_acc = run_epoch(model, train_loader, device, pad_id, global_epoch, optimizer=optimizer, scaler=scaler)
            model.eval()
            val_loss, val_acc = run_epoch(model, val_loader, device, pad_id, global_epoch, optimizer=None, scaler=None)

            save_path = os.path.join("./model", f"{stage['stage_name']}_epoch_{global_epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer.get_split_tokens(), f, indent=4, ensure_ascii=False)
            with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            state = model.state_dict()
            save_file(state, os.path.join(save_path, "model.safetensors"))

            global_epoch += 1

# ================================================

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    stages = [
        {"stage_name": "Pre-training", "file_path": "./data/wiki_dataset_en_filter.txt", "epochs": 10},
        {"stage_name": "Fine-tuning", "file_path": "./data/daily_dataset_en_filter.txt", "epochs": 10},]
    stage_train(stages, default_config)
