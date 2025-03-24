from safetensors.torch import load_file
from train import *

# ========== 預測對話 ==========

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7, repetition_penalty=1.2, presence_penalty=-1.0):
    encoded = tokenizer(f"<|user|>{prompt}<|assistant|>")
    generated = encoded["input_ids"].unsqueeze(0).to(device)
    print("\nAssistant: ", end="", flush=True)
    unknown_id = tokenizer.split_tokens.get("<|unknown|>")
    end_id = tokenizer.split_tokens.get("<|end|>")
    newline_id = tokenizer.split_tokens.get("\\n")
    with torch.no_grad():
        while generated.size(1) < max_length:
            if generated.size(1) > config["window_size"]:
                current_input = generated[:, -config["window_size"]:]
                pos_offset = generated.size(1) - config["window_size"]
            else:
                current_input = generated
                pos_offset = 0
            outputs = model(current_input, position_offset=pos_offset)
            logits = outputs["logits"][0, -1, :]
            gen_tokens = set(generated.squeeze().tolist())
            for token in gen_tokens:
                logits[token] = logits[token] * repetition_penalty if logits[token] < 0 else logits[token] / repetition_penalty
            vocab_size = logits.shape[0]
            mask = torch.tensor([token not in gen_tokens for token in range(vocab_size)], device=logits.device)
            logits[mask] += presence_penalty
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            if token_id == unknown_id:
                probs[unknown_id] = 0.0
                if probs.sum() > 0:
                    probs /= probs.sum()
                    token_id = torch.multinomial(probs, num_samples=1).item()
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            token_str = tokenizer.decode([token_id])
            if token_id == end_id:
                break
            elif token_id == newline_id:
                print("\n", end="")
            else:
                print(token_str, end="", flush=True)

# ========== 初始預測 ==========

if __name__ == "__main__":
    print("EasyGPT Beta V1.3 Torch Inference (Dev)")
    model_dir = "./model/pretrain_epoch_30"
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        token_dict = json.load(f)
    tokenizer = ChatTokenizer()
    tokenizer.split_tokens = token_dict
    device = torch.device("cpu") #("cuda" if torch.cuda.is_available() else "cpu")
    model = ChatModel().to(device)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()
    while True:
        print("="*50)
        prompt = input("User: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            break
        response = generate_response(model, tokenizer, prompt)
        print("\n", end="")
