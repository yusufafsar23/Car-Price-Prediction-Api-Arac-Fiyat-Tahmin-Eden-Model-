from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd



model = joblib.load("car_prediction_model.pkl")
app=FastAPI()
class CarFeatures(BaseModel):
    Brand: str
    Year:int
    Fuel_Type:str
    Transmission: str
    Mileage:int
    Condition: str
    

@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API çalışıyor!"}
    
@app.post("/predict")    
def predict(data:CarFeatures):
    input_df = pd.DataFrame([{
            "Brand": data.Brand,
            "Year": data.Year,
            "Fuel Type": data.Fuel_Type,
            "Transmission": data.Transmission,
            "Mileage": data.Mileage,
            "Condition": data.Condition,
    }])

    
    prediction = model.predict(input_df)
    return {"Tahmin Edilen Fiyat": float(prediction[0])}
