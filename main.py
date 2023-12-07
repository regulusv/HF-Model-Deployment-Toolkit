from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

if len(sys.argv) > 1:
    model_version = sys.argv[1]
    if model_version == "7b":
        from model_inference_api_7b import load_model, generate_with_function, generate_without_function
    elif model_version == "13b":
        from model_inference_api_13b import load_model, generate_with_function_call, generate_without_function
    else:
        raise ValueError("Invalid model version. Please specify '7b' or '13b'.")
else:
    raise ValueError("No model version specified. Please specify '7b' or '13b'.")

model, tokenizer = load_model()


@app.route("/predict_with_function", methods=["POST"])
def predict_with_function():
    input_text = None
    if request.json is not None:
        input_text = request.json.get("input_text", None)
    if input_text is None:
        return jsonify({"error": "No input text provided"}), 400

    response_text = generate_with_function(model, tokenizer, input_text)
    return jsonify({"response_text": response_text})


@app.route("/predict_with_function_call", methods=["POST"])
def predict_with_function_call():
    input_json = request.data.decode("utf-8")
    output_json = generate_with_function_call(input_json)
    return jsonify(output_json)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
