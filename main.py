from flask import Flask, request, jsonify
from gpt_model_inference_api import load_model, generate_with_function, generate_without_function

app = Flask(__name__)
model, tokenizer = load_model()

@app.route("/predict_with_function", methods=["POST"])
def predict_with_function():
    input_text = request.json.get("input_text", None)
    if input_text is None:
        return jsonify({"error": "No input text provided"}), 400

    response_text = generate_with_function(model, tokenizer, input_text)
    return jsonify({"response_text": response_text})

@app.route("/predict_without_function", methods=["POST"])
def predict_without_function():
    input_text = request.json.get("input_text", None)
    if input_text is None:
        return jsonify({"error": "No input text provided"}), 400

    response_text = generate_without_function(model, tokenizer, input_text)
    return jsonify({"response_text": response_text})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
