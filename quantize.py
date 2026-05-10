import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset 
from tqdm import tqdm
import pandas as pd

# --- CONFIGURATION ---
MODEL_ID = "facebook/opt-125m" # Change to "facebook/opt-1.3b" for more obvious results
DATASET_ID = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model, tokenizer, dataset, device = "cuda"):
    model.eval()
    nlls = []
    total_time = 0
    total_tokens = 0
    test_data = dataset["test"].select(range(10))
    for batch in tqdm(test_data, desc = "Evaluating"):
        inputs = tokenizer(batch["text"], return_tensors = "pt").to(device)
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 2: continue

        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids, labels = input_ids)
            end_time = time.time()

            neg_log_likelihood =  outputs.loss
            nlls.append(neg_log_likelihood)

            total_time += (end_time - start_time)
            total_tokens += input_ids.shape[1]

    ppl = torch.exp(torch.stack(nlls).mean())
    tokens_per_sec = total_tokens / total_time

    return ppl.item(), tokens_per_sec

print(f"Loading dataset and tokenizer for {MODEL_ID}...")
dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

results = []

# 1. Baseline: FP16
print("\n>>> Testing FP16 (Baseline)...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
ppl, tps = evaluate_model(model_fp16, tokenizer, dataset)
results.append({"Format": "FP16", "Perplexity": ppl, "Throughput (tok/s)": tps})
del model_fp16 # Free VRAM

# 2. INT8 (LLM.int8())
print("\n>>> Testing INT8 (Symmetric Quantization)...")
int8_config = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=int8_config, device_map="auto")
ppl, tps = evaluate_model(model_8bit, tokenizer, dataset)
results.append({"Format": "INT8", "Perplexity": ppl, "Throughput (tok/s)": tps})
del model_8bit

# 3. 4-bit (NF4)
print("\n>>> Testing 4-bit (NF4)...")
# Note: NF4 is a non-uniform quantization format better suited for Gaussian weights
nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model_4bit = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=nf4_config, device_map="auto")
ppl, tps = evaluate_model(model_4bit, tokenizer, dataset)
results.append({"Format": "4-bit (NF4)", "Perplexity": ppl, "Throughput (tok/s)": tps})
del model_4bit

# --- FINAL SUMMARY ---
df = pd.DataFrame(results)
print("\n" + "="*30)
print("FINAL COMPARISON RESULTS")
print("="*30)
print(df.to_string(index=False))