import librosa
import numpy as np
import tensorflow as tf

def analyze_voice(audio_path, model_path='voice_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, f"Model loading failed: {e}"

    try:
        # Use FFmpeg backend explicitly
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio file.")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        arr = np.expand_dims(mfcc_scaled, axis=0)
    except Exception as e:
        return None, f"Audio loading failed: {e}"

    try:
        prediction = model.predict(arr)[0][0]
        score = round(prediction * 100, 2)
        if prediction > 0.5:
            message = f"🧠 Cognitive Impairment Likely — Voice Score: {score}%"
        else:
            message = f"✅ Normal Cognitive Function — Voice Score: {score}%"
        return score, message
    except Exception as e:
        return None, f"Prediction failed: {e}"
