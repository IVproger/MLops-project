import json
import os

import gradio as gr
from src.utils import init_hydra
from src.data import preprocess_data
import requests
import pandas as pd

cfg = init_hydra("main")
PREDICT_URL = os.environ.get("PREDICT_URL", "http://localhost:8083/predict")

# fmt: off
EXAMPLES = [  # This is the list of examples that will be shown in the UI
    [1155, 1025, 90.0, 1, 2, 14, 1, "OAK", 6, 91, "SAN", 6, 91, 446.0, "Southwest Airlines Co.", "WN", "WN", "WN", "WN", "WN"],
    [905, 620, 105.0, 2, 4, 6, 3, "PHX", 4, 81, "DEN", 8, 82, 602.0, "Southwest Airlines Co.", "WN", "WN", "WN", "WN", "WN"],
    [1401, 1245, 136.0, 2, 4, 17, 7, "BWI", 24, 35, "ORD", 17, 41, 621.0, "SkyWest Airlines Inc.", "OO", "AA", "OO", "AA", "AA_CODESHARE"],
    [929, 743, 106.0, 2, 4, 1, 5, "CMH", 39, 44, "LGA", 36, 22, 479.0, "Republic Airlines", "YX", "DL", "YX", "DL", "DL_CODESHARE"],
    [1547, 1423, 204.0, 2, 4, 6, 3, "RFD", 17, 41, "AZA", 4, 81, 1373.0, "Allegiant Air", "G4", "G4", "G4", "G4", "G4"],
    [1445, 1355, 170.0, 2, 4, 3, 7, "IND", 18, 42, "DEN", 8, 82, 977.0, "Southwest Airlines Co.", "WN", "WN", "WN", "WN", "WN"],
    [2125, 2040, 45.0, 2, 4, 17, 7, "JMS", 38, 66, "DVL", 38, 66, 83.0, "SkyWest Airlines Inc.", "OO", "UA", "OO", "UA", "UA_CODESHARE"],
    [2230, 2145, 165.0, 2, 4, 4, 1, "HOU", 48, 74, "PHX", 4, 81, 1020.0, "Southwest Airlines Co.", "WN", "WN", "WN", "WN", "WN"],
    [1530, 1329, 121.0, 2, 4, 3, 7, "LGA", 36, 22, "CLT", 37, 36, 544.0, "American Airlines Inc.", "AA", "AA", "AA", "AA", "AA"],
    [1842, 1550, 172.0, 2, 4, 18, 1, "DCA", 51, 38, "MIA", 12, 33, 919.0, "American Airlines Inc.", "AA", "AA", "AA", "AA", "AA"],
]
# fmt: on


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

    # Build a dataframe of one row
    raw_df = pd.DataFrame([features])

    # Transform input data
    X, _ = preprocess_data(
        cfg,
        raw_df,
        require_target=False,
    )

    # Convert it into JSON
    payload = X.iloc[0, :].to_dict()
    for key, value in payload.items():
        if not (
            key.endswith("cos")
            or key.endswith("sin")
            or key.startswith("CRSElapsedTime")
            or key == "Distance"
        ):
            payload[key] = int(value)
    payload = json.dumps(payload)
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
            label="CRSDepTime",
            info="Scheduled departure time in HHMM, e.g. '1130' for 11:30.",
            precision=0,
            minimum=0,
            maximum=2400,
        ),
        gr.Number(
            label="CRSArrTime",
            info="Scheduled arrival time in HHMM, e.g. '1430' for 14:30.",
            precision=0,
            minimum=0,
            maximum=2400,
        ),
        gr.Number(
            label="CRSElapsedTime",
            info="Scheduled elapsed time of the flight in minutes.",
            precision=0,
            minimum=1,
        ),
        gr.Number(
            label="Quarter",
            info="Quarter of the year as an integer.",
            precision=0,
            minimum=1,
            maximum=4,
        ),
        gr.Number(
            label="Month",
            info="Month of the flight as an integer.",
            precision=0,
            minimum=1,
            maximum=12,
        ),
        gr.Number(
            label="DayofMonth",
            info="Day of the month as an integer.",
            precision=0,
            minimum=1,
            maximum=31,
        ),
        gr.Number(
            label="DayOfWeek",
            info="Day of the week as an integer (1=Monday, 7=Sunday).",
            precision=0,
            minimum=1,
            maximum=7,
        ),
        gr.Text(
            label="Origin",
            info="Origin airport code.",
            max_lines=1,
        ),
        gr.Number(
            label="OriginStateFips",
            info="FIPS code of the origin state.",
            precision=0,
            minimum=0,
        ),
        gr.Number(
            label="OriginWac",
            info="World Area Code for the origin.",
            precision=0,
            minimum=0,
        ),
        gr.Text(
            label="Dest",
            info="Destination airport code.",
            max_lines=1,
        ),
        gr.Number(
            label="DestStateFips",
            info="FIPS code of the destination state.",
            precision=0,
            minimum=0,
        ),
        gr.Number(
            label="DestWac",
            info="World Area Code for the destination.",
            precision=0,
            minimum=0,
        ),
        gr.Number(
            label="Distance",
            info="Distance of the flight in miles.",
            precision=0,
            minimum=0,
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
    outputs=gr.Label(label="Prediction Result"),
    examples=EXAMPLES,
    examples_per_page=50,
    live=True,  # Run model prediction immediately after changing the input
)


if __name__ == "__main__":
    # Launch the web UI locally on port 5155
    demo.launch(server_name="0.0.0.0", server_port=5155)
