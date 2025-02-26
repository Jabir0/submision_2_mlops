from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI()

# URL dari TensorFlow Serving (gunakan versi model yang benar)
TF_SERVING_URL = "http://localhost:8501/v1/models/hearts_model/versions/1740451872:predict"

@app.get("/")
def home():
    return {"message": "Heart Disease Detection API is running!"}

@app.post("/predict")
def predict(data: dict):
    if "features" not in data:
        raise HTTPException(status_code=400, detail="Request must contain 'features' key.")

    try:
        # Format data sesuai dengan API TensorFlow Serving
        input_data = {"instances": [data["features"]]}
        
        # Kirim request ke TensorFlow Serving
        response = requests.post(TF_SERVING_URL, json=input_data)

        # Cek apakah request berhasil
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        return {"prediction": result["predictions"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Heroku akan memberikan PORT sendiri
    uvicorn.run(app, host="0.0.0.0", port=port)

