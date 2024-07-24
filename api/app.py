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

    data.update(
        {
            "FlightDate": 0,
            "Diverted": False,
            "Year": 0,
            "DOT_ID_Marketing_Airline": 0,
            "Flight_Number_Marketing_Airline": 0,
            "DOT_ID_Operating_Airline": 0,
            "Flight_Number_Operating_Airline": 0,
            "OriginAirportID": 0,
            "OriginAirportSeqID": 0,
            "OriginCityMarketID": 0,
            "OriginCityName": 0,
            "OriginState": 0,
            "OriginStateName": 0,
            "DestAirportID": 0,
            "DestAirportSeqID": 0,
            "DestCityMarketID": 0,
            "DestCityName": 0,
            "DestState": 0,
            "DestStateName": 0,
            "DepTimeBlk": 0,
            "ArrTimeBlk": 0,
            "DistanceGroup": 0,
            "DivAirportLandings": 0,
        }
    )

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
