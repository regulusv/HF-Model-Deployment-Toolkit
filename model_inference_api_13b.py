from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import json


def load_model():
    model_name = "Trelis/Llama-2-13b-chat-hf-function-calling-v2"
    auth_token = "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"  # 替换为你的实际令牌
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=auth_token)
    return model, tokenizer


def generate_with_function_call(input_json, model, tokenizer):
    # parse input json
    input_data = json.loads(input_json)
    messages = input_data.get('messages', [])
    function_calls = input_data.get('functions', [])

    # handle function calls
    function_responses = []
    for call in function_calls:
        # TODO: handle function calls
        function_response = {
            "name": call['name'],
            "response": f"Executed function {call['name']} with arguments {call.get('parameters', {})}"
        }
        function_responses.append(function_response)

    # generate input prompt
    user_messages = " ".join([msg['content'] for msg in messages if msg['role'] == 'user'])
    prompt = f"{user_messages}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    # generate response
    output_tokens = model.generate(**inputs, max_new_tokens=500)
    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # generate output json
    output_json = json.dumps({
        "response_text": response_text,
        "function_responses": function_responses
    })

    return output_json


# test
# model, tokenizer = load_model()
# input_json = '{"messages": [{"role": "user", "content": "What is the weather like in Boston?"}],
# "functions": [{"name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}]}'
# print(generate_with_function_call(input_json, model, tokenizer))

def generate_without_function(model, tokenizer, input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    output = model.generate(inputs, max_length=512)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return response_text.strip()
