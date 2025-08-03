from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("logistic_model.pkl")

# Create the FastAPI app
app = FastAPI()

# Define expected input features (simplified version)
class PatientData(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    race: str
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    insulin: str
    change: str
    diabetesMed: str

@app.post("/predict")
def predict_readmission(data: PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {"prediction": prediction}
