from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import json
import torch

def load_model():
    # Load model and tokenizer with quantization for Llama 2
    model_name = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"
    cache_dir = "/tmp/transformers_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',  # Adjusts based on available hardware
        trust_remote_code=True
    )

    return model, tokenizer

def generate_with_function(model, tokenizer, input_text):
    # Define function metadata
    search_bing_metadata = {
        "function": "search_bing",
        "description": "Search the web for content on Bing.",
        "arguments": [
            {"name": "query", "type": "string", "description": "The search query string"}
        ]
    }

    functionList = json.dumps(search_bing_metadata, indent=4, separators=(',', ': '))

    B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"

    prompt = f"{B_FUNC}{functionList.strip()}{E_FUNC}{B_INST} {input_text.strip()} {E_INST}\n\n"
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

    streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)

    return streamer.text.strip()

def generate_without_function(model, tokenizer, input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
model, tokenizer = load_model()
output = generate_with_function(model, tokenizer, "What is the capital of France?")
print(output)
