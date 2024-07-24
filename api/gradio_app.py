import os

import gradio as gr
from src.utils import init_hydra
from src.data import preprocess_data
import requests
import pandas as pd

cfg = init_hydra("main")
PREDICT_URL = os.environ.get("PREDICT_URL", "http://localhost:8083/predict")


# You need to define a parameter for each column in your raw dataset
def predict(
    CRSArrTime,
    CRSDepTime,
    CRSElapsedTime,
    Quarter,
    Month,
    DayofMonth,
    DayOfWeek,
    Origin,
    OriginStateFips,
    OriginWac,
    Dest,
    DestStateFips,
    DestWac,
    Distance,
    Airline,
    Operating_Airline,
    IATA_Code_Marketing_Airline,
    IATA_Code_Operating_Airline,
    Marketing_Airline_Network,
    Operated_or_Branded_Code_Share_Partners,
):
    features = {
        "CRSArrTime": CRSArrTime,
        "CRSDepTime": CRSDepTime,
        "CRSElapsedTime": CRSElapsedTime,
        "Quarter": Quarter,
        "Month": Month,
        "DayofMonth": DayofMonth,
        "DayOfWeek": DayOfWeek,
        "Origin": Origin,
        "OriginStateFips": OriginStateFips,
        "OriginWac": OriginWac,
        "Dest": Dest,
        "DestStateFips": DestStateFips,
        "DestWac": DestWac,
        "Distance": Distance,
        "Airline": Airline,
        "Operating_Airline": Operating_Airline,
        "IATA_Code_Marketing_Airline": IATA_Code_Marketing_Airline,
        "IATA_Code_Operating_Airline": IATA_Code_Operating_Airline,
        "Marketing_Airline_Network": Marketing_Airline_Network,
        "Operated_or_Branded_Code_Share_Partners": Operated_or_Branded_Code_Share_Partners,
    }
    print(features)

    # Build a dataframe of one row
    raw_df = pd.DataFrame([features])

    # Transform input data
    X, _ = preprocess_data(
        cfg,
        raw_df,
        require_target=False,
    )

    # Convert it into JSON
    payload = X.iloc[0, :].to_json()
    print(payload)

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=PREDICT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    print(response.text)

    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()


# Only one interface is enough
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(
            label="CRSArrTime",
            info="Scheduled arrival time in HHMM, e.g. '1430' for 14:30.",
            value=1130,
        ),
        gr.Number(
            label="CRSDepTime",
            info="Scheduled departure time in HHMM.",
        ),
        gr.Number(
            label="CRSElapsedTime",
            info="Scheduled elapsed time of the flight in minutes.",
        ),
        gr.Number(
            label="Quarter",
            info="Quarter of the year as an integer.",
        ),
        gr.Number(
            label="Month",
            info="Month of the flight as an integer.",
        ),
        gr.Number(
            label="DayofMonth",
            info="Day of the month as an integer.",
        ),
        gr.Number(
            label="DayOfWeek",
            info="Day of the week as an integer (1=Monday, 7=Sunday).",
        ),
        gr.Text(
            label="Origin",
            info="Origin airport code.",
            max_lines=1,
        ),
        gr.Number(
            label="OriginStateFips",
            info="FIPS code of the origin state.",
        ),
        gr.Number(
            label="OriginWac",
            info="World Area Code for the origin.",
        ),
        gr.Text(
            label="Dest",
            info="Destination airport code.",
            max_lines=1,
        ),
        gr.Number(
            label="DestStateFips",
            info="FIPS code of the destination state.",
        ),
        gr.Number(
            label="DestWac",
            info="World Area Code for the destination.",
        ),
        gr.Number(
            label="Distance",
            info="Distance of the flight in miles.",
        ),
        gr.Text(
            label="Airline",
            info="Enter the airline name.",
            max_lines=1,
            value="Commutair Aka Champlain Enterprises, Inc.",
        ),
        gr.Text(
            label="Operating_Airline",
            info="Unique Carrier Code.",
            max_lines=1,
        ),
        gr.Text(
            label="IATA_Code_Marketing_Airline",
            info="Code assigned by IATA and commonly used to identify a carrier.",
            max_lines=1,
        ),
        gr.Text(
            label="IATA_Code_Operating_Airline",
            info="Code assigned by IATA and commonly used to identify a carrier.",
            max_lines=1,
        ),
        gr.Text(
            label="Marketing_Airline_Network",
            info="Unique Marketing Carrier Code.",
            max_lines=1,
        ),
        gr.Text(
            label="Operated_or_Branded_Code_Share_Partners",
            info="Reporting Carrier Operated or Branded Code Share Partners.",
            max_lines=1,
        ),
    ],
    outputs=gr.Text(label="Prediction Result"),
)


if __name__ == "__main__":
    # Launch the web UI locally on port 5155
    demo.launch(server_name="0.0.0.0", server_port=5155)
