import os, re, math, json, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from safetensors.torch import save_file
from bitsandbytes.optim import Adam8bit
from collections import Counter
from functools import partial
from tqdm import tqdm

# ========== 模型配置 ==========

config = {
    "hidden_size": 256,
    "num_layers": 2,
    "num_heads": 4,
    "num_experts": 3,
    "expert_loss": 0.01,
    "rope_dim": 64,
    "rope_base": 10000,
    "forward_dim": 640,
    "window_size": 512,
    "split_size": 32000,
    "length_size": 1024,
    "split_length": 5,
    "batch_size": 8,
    "epochs_size": 30,
    "split_valid": 0.1,
    "weight_decay": 0.1,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "betas_range": [0.9, 0.999],
    "global_tokens": {
        "<|padding|>": 0,
        "<|unknown|>": 1},
    "special_tokens": {
        "<|system|>": 2,
        "<|user|>": 3,
        "<|think|>": 4,
        "<|assistant|>": 5,
        "<|function|>": 6,
        "<|end|>": 7}
}

# ========== 前饋專家 ==========

class MoEFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([GEGLU() for _ in range(config["num_experts"])])
        self.gate = nn.Linear(config["hidden_size"], config["num_experts"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        B, seq, d = x.shape
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, expert_indices = torch.max(gate_probs, dim=-1)
        x_flat = x.view(-1, d)
        expert_indices_flat = expert_indices.view(-1)
        expert_weights_flat = expert_weights.view(-1)
        expert_output_dim = self.experts[0](x_flat[:1]).shape[-1]
        output_flat = torch.zeros(x_flat.size(0), expert_output_dim, device=x.device, dtype=x.dtype)
        scores_sum = torch.zeros(config["num_experts"], device=x.device)
        counts = torch.zeros(config["num_experts"], device=x.device)
        for expert_idx in range(config["num_experts"]):
            token_mask = (expert_indices_flat == expert_idx)
            token_indices = torch.where(token_mask)[0]
            if token_indices.numel() == 0:
                continue
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            output_flat[token_indices] = expert_output * expert_weights_flat[token_indices].unsqueeze(1)
            scores_sum[expert_idx] += expert_weights_flat[token_indices].sum()
            counts[expert_idx] += token_indices.numel()
        expert_usage = scores_sum / (counts + 1e-8)
        utilization_loss = ((expert_usage - 1.0 / config["num_experts"]) ** 2).sum()
        output = output_flat.view(B, seq, -1)
        return self.dropout(output), utilization_loss

# ========== 注意力塊 ==========

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = config["hidden_size"] // config["num_heads"]
        self.qkv_proj = nn.Linear(config["hidden_size"], 3 * config["num_heads"] * self.head_dim)
        self.out_proj = nn.Linear(config["num_heads"] * self.head_dim, config["hidden_size"])
        self.attn_dropout = nn.Dropout(config["dropout_rate"])
        self.register_buffer("global_tokens_indices", torch.tensor(
            [v for v in config["special_tokens"].values()], dtype=torch.long))
        if config["rope_dim"] > 0:
            self.rotary_emb = RotaryEmbedding()
        nn.init.xavier_normal_(self.qkv_proj.weight)

    def forward(self, x, mask=None, input_ids=None, position_offset=0):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, config["num_heads"], self.head_dim).transpose(1, 2)
        k = k.view(B, T, config["num_heads"], self.head_dim).transpose(1, 2)
        v = v.view(B, T, config["num_heads"], self.head_dim).transpose(1, 2)
        if config["rope_dim"] > 0:
            cos, sin = self.rotary_emb(q, offset=position_offset)
            q = self.apply_rotary(q, cos, sin)
            k = self.apply_rotary(k, cos, sin)
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril().view(1, 1, T, T)
        if input_ids is not None:
            global_mask = torch.isin(input_ids, self.global_tokens_indices)
            causal_mask = causal_mask | global_mask.view(B, 1, T, 1)
        scale = math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) / scale
        attn_logits = attn_logits.masked_fill(~causal_mask, -torch.finfo(q.dtype).max)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, -torch.finfo(q.dtype).max)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(attn_output)

    def apply_rotary(self, x, cos, sin):
        if config["rope_dim"] == 0:
            return x
        x_rot = x[..., :config["rope_dim"]]
        x_pass = x[..., config["rope_dim"]:]
        x1, x2 = x_rot.chunk(2, dim=-1)
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        x_rot = torch.cat((x1_new, x2_new), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)

# ========== 位置編碼 ==========

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq = 1.0 / (config["rope_base"] ** (torch.arange(0, config["rope_dim"], 2).float() / config["rope_dim"]))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        B, num_heads, T, _ = x.shape
        positions = torch.arange(offset, offset + T, dtype=torch.float, device=x.device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]
        return cos, sin

# ========== 激活函數 ==========

class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(config["hidden_size"], config["forward_dim"] * 2)
        self.out = nn.Linear(config["forward_dim"], config["hidden_size"])

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.out(x1 * F.gelu(x2))

# ========== 變換模塊 ==========

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"])
        self.self_attn = SelfAttention()
        self.norm2 = nn.LayerNorm(config["hidden_size"])
        self.ffn = MoEFeedForward()
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x, mask=None, input_ids=None, position_offset=0):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask, input_ids, position_offset))
        ffn_out, moe_loss = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out), moe_loss

# ========== 聊天模塊 ==========

class ChatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config["split_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config["num_layers"])])
        self.norm = nn.LayerNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["split_size"])

    def forward(self, input_ids, attention_mask=None, labels=None, position_offset=0):
        x = self.embed(input_ids)
        B, T, _ = x.size()
        causal_mask = self.get_adaptive_causal_mask(T, x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
            mask = causal_mask | pad_mask
        else:
            mask = causal_mask
        moe_losses = 0
        for block in self.blocks:
            x, loss = block(x, mask, input_ids, position_offset)
            moe_losses += loss
        moe_losses /= len(self.blocks)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=config["global_tokens"]["<|padding|>"])
            loss += config["expert_loss"] * moe_losses
        return {"loss": loss, "logits": logits}

    def get_adaptive_causal_mask(self, T, device):
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        mask = (j > i) | ((i - j) >= config["window_size"])
        return mask

# ========== 分詞模塊 ==========

class ChatTokenizer:
    def __init__(self):
        self.split_tokens = {token: i for i, token in enumerate({**config["global_tokens"], **config["special_tokens"]})}
        escaped_tokens = map(re.escape, config["special_tokens"].keys())
        self.pattern = re.compile(f'({"|".join(escaped_tokens)})' r'|(\\n)|([a-zA-Z]+)|( )|([0-9])|(_)|([^\s])', re.UNICODE)

    def tokenize(self, text):
        tokens = [m.group() for m in self.pattern.finditer(text)]
        if not config["split_length"]:
            return tokens
        candidates = {t for t in tokens if t.isalpha() and len(t) <= config["split_length"]}
        segmenter = CustomSegmenter(candidates)
        output = []
        for t in tokens:
            if t.isalpha() and len(t) > config["split_length"]:
                seg = segmenter.segment(t)
                output.extend(seg if seg is not None else [t])
            else:
                output.append(t)
        return output

    def convert_tokens_to_ids(self, tokens, update_split_tokens=True):
        unk = self.split_tokens.get("<|unknown|>")
        if update_split_tokens:
            ids = []
            for t in tokens:
                if t not in self.split_tokens:
                    self.split_tokens[t] = len(self.split_tokens)
                ids.append(self.split_tokens[t])
            return ids
        else:
            return [self.split_tokens.get(t, unk) for t in tokens]

    def __call__(self, text, length_size=None, truncation=True, padding="max_length", update_split_tokens=False):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens, update_split_tokens)
        if truncation and length_size is not None:
            ids = ids[:length_size]
        if padding == "length_size" and length_size is not None:
            pad_id = self.split_tokens.get("<|padding|>")
            ids += [pad_id] * (length_size - len(ids))
        mask = [1 if i != self.split_tokens.get("<|padding|>") else 0 for i in ids]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def build_split_tokens(self, file_path, min_freq=1):
        with open(file_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        total_text = 0
        freq = Counter()
        pbar = tqdm(texts, desc="[Token 01]", dynamic_ncols=True)
        for text in pbar:
            tokens = self.tokenize(text)
            freq.update(t for t in tokens if t not in config["special_tokens"])
            total_text += len(text)
            pbar.set_postfix({"total": total_text, "token": len(freq)})
        filtered_tokens = {token: count for token, count in freq.items() if count >= min_freq}
        tokens_sorted = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        max_new_tokens = config["split_size"] - len(self.split_tokens) if config["split_size"] else None
        if max_new_tokens is not None:
            tokens_sorted = tokens_sorted[:max_new_tokens]
        pbar = tqdm(tokens_sorted, desc="[Token 02]", dynamic_ncols=True)
        for i, (token, _) in enumerate(pbar):
            if token not in self.split_tokens:
                self.split_tokens[token] = len(self.split_tokens)
                pbar.set_postfix({"text": token, "token": len(self.split_tokens)})

    def get_split_tokens(self):
        return self.split_tokens

    def decode(self, token_ids):
        inv_tokens = {v: k for k, v in self.split_tokens.items()}
        tokens = [inv_tokens.get(token_id, "<|unknown|>") for token_id in token_ids]
        return "".join(tokens)

# ========== 細分模塊 ==========

class CustomSegmenter:
    def __init__(self, candidates):
        self.candidates = candidates
        self.trie = self._build_trie(candidates)

    def _build_trie(self, candidates):
        trie = {}
        for word in candidates:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['$'] = True
        return trie

    def segment(self, token):
        segments = []
        i = 0
        n = len(token)
        while i < n:
            node = self.trie
            found = None
            j = i
            while j < n and j - i < config["split_length"] and token[j] in node:
                node = node[token[j]]
                if '$' in node:
                    found = token[i:j+1]
                j += 1
            if found is None:
                segments.append(token[i])
                i += 1
            else:
                segments.append(found)
                i += len(found)
        return segments if len(segments) < len(token) else None

# ========== 數據處理 ==========

class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        with open(file_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.data = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.data[idx], config["length_size"] + 1, padding="length_size", update_split_tokens=False)
        input_ids = encoding["input_ids"].squeeze()
        return {"input_ids": input_ids[:-1], "attention_mask": encoding["attention_mask"].squeeze()[:-1], "labels": input_ids[1:]}

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

# ========== 訓練階段 ==========

def stage_train(stage_name, file_path):
    tokenizer = ChatTokenizer()
    tokenizer.build_split_tokens(file_path)
    dataset = ChatDataset(tokenizer, file_path)
    model = ChatModel()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    indices = torch.randperm(len(dataset)).tolist()
    split_idx = int(len(dataset) * (1 - config["split_valid"]))
    num_workers = min(8, os.cpu_count() or 1)
    pad_id = tokenizer.get_split_tokens()["<|padding|>"]
    train_loader = DataLoader(Subset(dataset, indices[:split_idx]), batch_size=config["batch_size"], num_workers=num_workers, persistent_workers=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, indices[split_idx:]), batch_size=config["batch_size"], num_workers=num_workers, persistent_workers=True, shuffle=False, pin_memory=True)
    optimizer = Adam8bit(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=config["betas_range"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=1)
    for epoch in range(config["epochs_size"]):
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, device, pad_id, epoch, optimizer)
        model.eval()
        val_loss, val_acc = run_epoch(model, val_loader, device, pad_id, epoch)
        scheduler.step(val_loss)
        save_path = os.path.join("./model", f"{stage_name}_epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.get_split_tokens(), f, indent=4, ensure_ascii=False)
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        save_file(model.state_dict(), os.path.join(save_path, "model.safetensors"))

# ========== 初始訓練 ==========

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_train("pretrain", "./data/dialogues.txt")
