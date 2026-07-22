import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Load voice data ---
def load_voice_data(base_path='data/voice'):
    X, y = [], []
    for label in ['normal', 'impaired']:
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            if file.endswith('.mp3') or file.endswith('.wav'):
                path = os.path.join(folder, file)
                try:
                    y_audio, sr = librosa.load(path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    X.append(mfcc_scaled)
                    y.append(0 if label == 'normal' else 1)
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")
    return np.array(X), np.array(y)

# --- Build model ---
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Train and save ---
def train_voice_model():
    X, y = load_voice_data()
    print(f"Loaded {len(X)} samples.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(X.shape[1])
    history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val))

    model.save('voice_model.keras')
    print("✅ Voice model trained and saved as 'voice_model.keras'.")

if __name__ == "__main__":
    train_voice_model()
