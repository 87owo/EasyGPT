from safetensors.torch import load_file
from train import *

# ========== 預測對話 ==========

def generate_response(model, tokenizer, prompt, max_length=1024, temperature=0.7, repetition_penalty=1.2, presence_penalty=-1.0):
    encoding = tokenizer(f"<|user|>{prompt}<|assistant|>")
    generated = encoding["input_ids"].unsqueeze(0).to(device)
    print("\nAssistant: ", end="", flush=True)
    with torch.no_grad():
        while generated.size(1) < max_length:
            if generated.size(1) > config["window_size"]:
                current_input = generated[:, -config["window_size"]:]
                pos_offset = generated.size(1) - config["window_size"]
            else:
                current_input = generated
                pos_offset = 0
            outputs = model(current_input, position_offset=pos_offset)
            logits = outputs["logits"]
            next_token_logits = logits[0, -1, :]
            for token_id in set(generated.squeeze().tolist()):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty
            generated_set = set(generated.squeeze().tolist())
            vocab_size = next_token_logits.shape[0]
            mask = torch.tensor([token_id not in generated_set for token_id in range(vocab_size)], device=next_token_logits.device)
            next_token_logits[mask] += presence_penalty
            next_token_logits = next_token_logits / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = next_token.item()
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            token_str = tokenizer.decode([token_id])
            if token_id == tokenizer.split_tokens.get("<|end|>"):
                print("\n", end="", flush=True)
                break
            elif token_id == tokenizer.split_tokens.get("\\n"):
                print("\n", end="", flush=True)
            else:
                print(token_str, end="", flush=True)
    print(f"\nTokens: {generated.size(1)}")

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
