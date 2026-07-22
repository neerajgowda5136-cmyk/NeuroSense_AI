# train_drawing_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess
import os

# --- Step 1: Auto-generate data ---
print("🚀 Generating synthetic data...")
subprocess.run(["python", "generate_data.py"])
print("✅ Data generation finished.\n")

# --- Step 2: Prepare data directories ---
train_dir = 'data/train'
val_dir = 'data/val'

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("⚠️ Training or validation data missing. Please check data folders.")
    exit()

# --- Step 3: Data preprocessing ---
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=16, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=16, class_mode='binary'
)

# --- Step 4: Build CNN model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Step 5: Train model ---
print("🧠 Training model...")
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# --- Step 6: Evaluate and save ---
loss, acc = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {acc:.2f}")

model.save('model.keras')
with open('drawing_accuracy.txt', 'w') as f:
    f.write(str(acc))
print("✅ Model saved as model.pkl and accuracy recorded.")
