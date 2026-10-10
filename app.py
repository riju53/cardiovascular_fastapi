
from pathlib import Path
from typing import Literal, Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = Path(__file__).parent / "cardiovascular_model.pkl"
ml_model = joblib.load(MODEL_PATH)

app = FastAPI(title="Cardiovascular Classification API")


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=18, lt=100)]
    sex: Literal["Female", "Male"]
    height_cm: Annotated[int, Field(..., gt=140, lt=210)]
    weight_kg: Annotated[int, Field(..., gt=30, lt=200)]
    systolic_bp: Annotated[int, Field(..., gt=70, lt=210)]
    cholesterol: Annotated[int, Field(..., gt=50, lt=600)]
    glucose: Annotated[int, Field(..., gt=40, lt=600)]
    smoking: Literal["Yes", "No"]
    family_history: Literal["No", "Yes"]
    physical_activity: Literal["High", "Low", "Medium"]
    diabetes: Literal["No", "Yes"]
    hypertension_history: Literal["No", "Yes"]


@app.get("/")
def home():
    return {"message": "Cardiovascular Classification API is running"}


@app.post("/predict")
def predict_cardiovascular(data: UserInput):
    input_data = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "height_cm": data.height_cm,
        "weight_kg": data.weight_kg,
        "systolic_bp": data.systolic_bp,
        "cholesterol": data.cholesterol,
        "glucose": data.glucose,
        "smoking": data.smoking,
        "family_history": data.family_history,
        "physical_activity": data.physical_activity,
        "diabetes": data.diabetes,
        "hypertension_history": data.hypertension_history
    }])

    prediction = ml_model.predict(input_data)[0]

    # Convert NumPy scalar predictions to native Python values
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return JSONResponse(
        status_code=200,
        content={"predicted_category": prediction}
    )
