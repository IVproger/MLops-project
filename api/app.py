import os
import mlflow.pyfunc
from flask import Flask, request, make_response

model = mlflow.pyfunc.load_model("./model")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    msg = """
    This is the ML service to predict flight cancellation\n\n

    This API has two main endpoints:\n
    1. /info: to get info about the deployed model.\n
    2. /predict: to send predict requests to our deployed model.\n
    """

    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response


@app.route("/info", methods=["GET"])
def info():
    response = make_response(str(model.metadata), 200)
    response.content_type = "text/plain"
    return response


@app.route("/predict", methods=["POST"])
def predict():
    data: dict = request.json
    print("Request:", data, flush=True)

    # Get the prediction
    prediction = model.predict(data)
    print("Prediction:", prediction, flush=True)

    # Prepare the response
    content = {"On-Time": float(prediction[0][0]), "Cancelled": float(prediction[0][1])}
    response = make_response(content, 200)
    response.headers["content-type"] = "application/json"
    return response


if __name__ == "__main__":
    # Run a local server to accept requests to the API
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
