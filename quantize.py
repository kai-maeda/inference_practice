import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset 
from tqdm import tqdm


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

dataset = load_dataset("wikitext" , "wikitext-2-raw-v1")
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)

