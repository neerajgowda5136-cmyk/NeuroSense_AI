import os
import random
from PIL import Image, ImageDraw
from gtts import gTTS

# --- CLOCK DRAWING GENERATION ---
def generate_clock_image(path, impaired=False):
    img = Image.new('RGB', (128, 128), 'white')
    draw = ImageDraw.Draw(img)
    # Draw clock circle
    draw.ellipse((10, 10, 118, 118), outline='black', width=3)

    # Draw numbers (simplified)
    for i in range(12):
        angle = i * 30
        x = 64 + 45 * random.uniform(0.9, 1.1) * (random.choice([-1, 1]) if impaired else 1)
        y = 64 + 45 * random.uniform(0.9, 1.1) * (random.choice([-1, 1]) if impaired else 1)
        draw.text((x, y), str(i + 1), fill='black')

    # Draw clock hands
    hour_angle = random.randint(0, 360)
    minute_angle = random.randint(0, 360)
    draw.line((64, 64, 64 + 40 * random.uniform(0.8, 1.2) * (random.choice([-1, 1]) if impaired else 1),
               64 + 40 * random.uniform(0.8, 1.2) * (random.choice([-1, 1]) if impaired else 1)),
              fill='black', width=3)
    draw.line((64, 64, 64 + 55 * random.uniform(0.8, 1.2) * (random.choice([-1, 1]) if impaired else 1),
               64 + 55 * random.uniform(0.8, 1.2) * (random.choice([-1, 1]) if impaired else 1)),
              fill='black', width=2)

    img.save(path)

# --- VOICE SAMPLE GENERATION ---
def generate_voice_sample(path, impaired=False):
    if impaired:
        text = "I... am... feeling... a bit... confused... today."
    else:
        text = "Today is a sunny day. I am feeling good."
    tts = gTTS(text=text, lang='en')
    tts.save(path)

# --- DATASET CREATION ---
def generate_dataset():
    for category in ['train', 'val']:
        for label in ['normal', 'impaired']:
            folder_img = f'data/{category}/{label}'
            folder_voice = f'data/voice/{label}'
            os.makedirs(folder_img, exist_ok=True)
            os.makedirs(folder_voice, exist_ok=True)

            # Generate 20 clock drawings
            for i in range(20):
                generate_clock_image(f'{folder_img}/{label}_{i}.png', impaired=(label == 'impaired'))

            # Generate 20 voice samples
            for i in range(20):
                generate_voice_sample(f'{folder_voice}/{label}_{i}.mp3', impaired=(label == 'impaired'))

if __name__ == "__main__":
    generate_dataset()
    print("✅ Generated 20 normal + 20 impaired clock drawings and voice samples for train/val.")
