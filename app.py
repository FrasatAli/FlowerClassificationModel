from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
import shutil
import tempfile
from predict.predict import FlowerPredictor

app = FastAPI(title="Flower Classification API")

try:
    predictor = FlowerPredictor()
except Exception as e:
    logging.critical(f"API startup failed: {e}")
    raise


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(400, detail="Only JPG/PNG allowed")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        result = predictor.predict(temp_path)
        os.unlink(temp_path)

        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)