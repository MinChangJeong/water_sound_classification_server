from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import shutil
import os

app = FastAPI()

class Prediction(BaseModel):
    result: int

@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    try:
        base_dir = "/app/uploads"  # 예시로 '/app/uploads' 경로를 사용합니다. 실제 경로를 사용해야 합니다.

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        file_path = os.path.join(base_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 파일을 Spectrogram으로 변환
        waveform, sample_rate = librosa.load(file_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        resized_spectrogram = np.resize(spectrogram, (128, 128, 1))  # 모델에 맞게 크기 조정

        # 모델 불러오기
        loaded_model = load_model('water_saver.keras')

        # 예측
        predictions = loaded_model.predict(np.expand_dims(resized_spectrogram, axis=0))

        # 결과 리턴
        print(predictions)
        if predictions >= 0.8:
            return {"result": 1}
        else:
            return {"result": 0}
        
    except Exception as e:
        return {"error": str(e)}
