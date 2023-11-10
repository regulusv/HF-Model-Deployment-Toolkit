from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify

# Hugging Face
hf_token = "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"

app = Flask(__name__)

# load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


@app.route("/predict", methods=["POST"])
def predict():
    # get post from request
    input_text = request.json.get("input_text", None)
    if input_text is None:
        return jsonify({"error": "No input text provided"}), 400

    # from input text to inference, generate response text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512)

    # decode response text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # return response text
    return jsonify({"response_text": response_text})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
