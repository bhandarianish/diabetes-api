from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load("logistic_model.pkl")
    print(" Model loaded successfully.")
except FileNotFoundError:
    print(" Model file not found")
except Exception as e:
    print(f" Error loading model: {e}")
    
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
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)




