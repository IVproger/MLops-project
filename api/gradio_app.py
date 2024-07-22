import os

import gradio as gr
from src.utils import init_hydra
from src.data import preprocess_data
import json
import requests
import pandas as pd

cfg = init_hydra("main")
INVOCATIONS_URL = os.environ.get("INVOCATIONS_URL", "http://localhost:8082/invocations")


# You need to define a parameter for each column in your raw dataset
def predict(
    Airline,
    ArrTimeBlk,
    CRSArrTime,
    CRSDepTime,
    CRSElapsedTime,
    DOT_ID_Marketing_Airline,
    DOT_ID_Operating_Airline,
    DayOfWeek,
    DayofMonth,
    DepTimeBlk,
    Dest,
    DestAirportID,
    DestAirportSeqID,
    DestCityMarketID,
    DestStateFips,
    DestWac,
    Distance,
    DistanceGroup,
    DivAirportLandings,
    Flight_Number_Marketing_Airline,
    Flight_Number_Operating_Airline,
    Month,
    Origin,
    OriginAirportID,
    OriginAirportSeqID,
    OriginCityMarketID,
    OriginStateFips,
    OriginWac,
    Quarter,
):
    features = {
        "Airline": Airline,
        "ArrTimeBlk": ArrTimeBlk,
        "CRSArrTime": CRSArrTime,
        "CRSDepTime": CRSDepTime,
        "CRSElapsedTime": CRSElapsedTime,
        "DOT_ID_Marketing_Airline": DOT_ID_Marketing_Airline,
        "DOT_ID_Operating_Airline": DOT_ID_Operating_Airline,
        "DayOfWeek": DayOfWeek,
        "DayofMonth": DayofMonth,
        "DepTimeBlk": DepTimeBlk,
        "Dest": Dest,
        "DestAirportID": DestAirportID,
        "DestAirportSeqID": DestAirportSeqID,
        "DestCityMarketID": DestCityMarketID,
        "DestStateFips": DestStateFips,
        "DestWac": DestWac,
        "Distance": Distance,
        "DistanceGroup": DistanceGroup,
        "DivAirportLandings": DivAirportLandings,
        "Flight_Number_Marketing_Airline": Flight_Number_Marketing_Airline,
        "Flight_Number_Operating_Airline": Flight_Number_Operating_Airline,
        "Month": Month,
        "Origin": Origin,
        "OriginAirportID": OriginAirportID,
        "OriginAirportSeqID": OriginAirportSeqID,
        "OriginCityMarketID": OriginCityMarketID,
        "OriginStateFips": OriginStateFips,
        "OriginWac": OriginWac,
        "Quarter": Quarter,
    }
    print(features)

    # Build a dataframe of one row
    raw_df = pd.DataFrame([features])

    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X, _ = preprocess_data(
        cfg,
        raw_df,
        require_target=False,
    )

    # Convert it into JSON
    payload = json.dumps({"inputs": X.iloc[0, :].to_dict()})

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=INVOCATIONS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()


# Only one interface is enough
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Text(
            label="Airline",
            info="Enter the airline name.",
            value="Commutair Aka Champlain Enterprises, Inc.",
            max_lines=1,
        ),
        gr.Text(label="ArrTimeBlk", info="Arrival time block, e.g., '0800-0859'."),
        gr.Number(
            value=1130,
            label="CRSArrTime",
            info="Scheduled arrival time in HHMM, e.g. '1430' for 14:30.",
            int_only=True,
        ),
        gr.Number(
            label="CRSDepTime", info="Scheduled departure time in HHMM.", int_only=True
        ),
        gr.Number(
            label="CRSElapsedTime",
            info="Scheduled elapsed time of the flight in minutes.",
            int_only=True,
        ),
        gr.Number(
            label="DOT_ID_Marketing_Airline",
            info="DOT ID for the marketing airline.",
            int_only=True,
        ),
        gr.Number(
            label="DOT_ID_Operating_Airline",
            info="DOT ID for the operating airline.",
            int_only=True,
        ),
        gr.Number(
            label="DayOfWeek",
            info="Day of the week as an integer (1=Monday, 7=Sunday).",
            int_only=True,
        ),
        gr.Number(
            label="DayofMonth", info="Day of the month as an integer.", int_only=True
        ),
        gr.Text(label="DepTimeBlk", info="Departure time block, e.g., '0800-0859'."),
        gr.Text(label="Dest", info="Destination airport code."),
        gr.Number(
            label="DestAirportID", info="Destination airport's ID.", int_only=True
        ),
        gr.Number(
            label="DestAirportSeqID",
            info="Destination airport's sequence ID.",
            int_only=True,
        ),
        gr.Number(
            label="DestCityMarketID", info="Destination city market ID.", int_only=True
        ),
        gr.Number(
            label="DestStateFips",
            info="FIPS code of the destination state.",
            int_only=True,
        ),
        gr.Number(
            label="DestWac", info="World Area Code for the destination.", int_only=True
        ),
        gr.Number(
            label="Distance", info="Distance of the flight in miles.", int_only=True
        ),
        gr.Number(
            label="DistanceGroup",
            info="Grouping of the flight distance into categories.",
            int_only=True,
        ),
        gr.Number(
            label="DivAirportLandings",
            info="Number of diversions with airport landings.",
            int_only=True,
        ),
        gr.Number(
            label="Flight_Number_Marketing_Airline",
            info="Flight number assigned by the marketing airline.",
            int_only=True,
        ),
        gr.Number(
            label="Flight_Number_Operating_Airline",
            info="Flight number assigned by the operating airline.",
            int_only=True,
        ),
        gr.Number(
            label="Month", info="Month of the flight as an integer.", int_only=True
        ),
        gr.Text(label="Origin", info="Origin airport code."),
        gr.Number(label="OriginAirportID", info="Origin airport's ID.", int_only=True),
        gr.Number(
            label="OriginAirportSeqID",
            info="Origin airport's sequence ID.",
            int_only=True,
        ),
        gr.Number(
            label="OriginCityMarketID", info="Origin city market ID.", int_only=True
        ),
        gr.Number(
            label="OriginStateFips",
            info="FIPS code of the origin state.",
            int_only=True,
        ),
        gr.Number(
            label="OriginWac", info="World Area Code for the origin.", int_only=True
        ),
        gr.Number(
            label="Quarter", info="Quarter of the year as an integer.", int_only=True
        ),
    ],
    outputs=gr.Text(label="Prediction Result"),
)


if __name__ == "__main__":
    # Launch the web UI locally on port 5155
    demo.launch(server_name="0.0.0.0", server_port=5155)
