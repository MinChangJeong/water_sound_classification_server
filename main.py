from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

class Prediction(BaseModel):
    result: int

@app.post("/classify", response_model=Prediction)
async def classify_audio(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 파일을 Spectrogram으로 변환
    waveform, sample_rate = librosa.load(contents, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    resized_spectrogram = np.resize(spectrogram, (128, 128, 1))  # 모델에 맞게 크기 조정
    
    # 모델 불러오기
    loaded_model = load_model('water_saver.keras')
    
    # 예측
    predictions = loaded_model.predict(np.expand_dims(resized_spectrogram, axis=0))
    
    # 결과 리턴
    if predictions >= 0.89:
        return Prediction(result=1)
    else:
        return Prediction(result=0)
