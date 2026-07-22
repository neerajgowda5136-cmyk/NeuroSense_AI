import tensorflow as tf
import numpy as np
from PIL import Image

def analyze_drawing(image_path, model_path='model.keras'):
    # --- Load trained model ---
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, f"Model loading failed: {e}"

    # --- Preprocess image ---
    try:
        # Convert to RGB because model expects 3 channels
        img = Image.open(image_path).convert('RGB').resize((128, 128))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
    except Exception as e:
        return None, f"Image loading failed: {e}"

    # --- Predict cognitive state ---
    try:
        prediction = model.predict(arr)[0][0]
        score = round(prediction * 100, 2)
        if prediction > 0.5:
            message = f"🧠 Cognitive Impairment Likely — Drawing Score: {score}%"
        else:
            message = f"✅ Normal Cognitive Function — Drawing Score: {score}%"
        return score, message
    except Exception as e:
        return None, f"Prediction failed: {e}"
