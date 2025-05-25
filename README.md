# 📸🧠 StoryCrafter: AI-Powered Visual Storytelling

Turn a series of images into a narrated short story using state-of-the-art AI models!  
This project combines image captioning, object detection, LLM-based storytelling, and TTS voice narration.

---

## 🎯 What It Does

1. 🖼️ **Capture or Load Images**  
2. 🧠 **Generate Smart Captions** using [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)  
3. 🔍 **Detect Objects** with [YOLOv8](https://github.com/ultralytics/ultralytics)  
4. 📖 **Write a Genre-Specific Story** with [LLaMA 2](https://ai.meta.com/llama/)  
5. 🔊 **Narrate the Story** using a cloned voice with [F5-TTS](https://github.com/TensorSpeech/FastSpeech2)  
6. 🎬 **Display it with Subtitles and Audio** — like a slideshow movie!

---

## 🧰 Requirements

- Python 3.9+
- GPU (recommended)
- Pre-trained models:
  - `Salesforce/blip2-opt-2.7b`
  - `YOLOv8` variant (`yolo11x.pt`)
  - `llama-2-7b-chat.Q5_K_M.gguf`
  - `F5-TTS` with cloned voice checkpoint

---

## 🗂 Folder Structure

├── images/ # Input image frames (if not using webcam)
├── models/ # All models stored here
│ ├── blip2_model/
│ ├── blip2_processor/
│ ├── llama-2-7b-chat.Q5_K_M.gguf
│ └── yolo11x.pt
├── audio/
│ └── reference_audio.wav # 20s voice sample
├── tests/
│ └── outputX.wav # Output narrated files
├── main.py # Run this file!
└── README.md

---

## 🚀 Getting Started

```bash
# Clone the repo and navigate into the directory
git clone https://github.com/yourusername/storycrafter.git
cd storycrafter

# Install dependencies
pip install -r requirements.txt

# Run and take new photos using your webcam
python main.py --photo

# Or run with existing images from the /images folder
python main.py
```

# 📌 At the end, press any key to view the narrated story frame-by-frame.

## 🎮 Choose Your Genre
You'll be prompted to select the story's theme:

What genre do you wish the story to be in?
Examples:

🧙 Fantasy

👽 Sci-Fi

🕵️ Mystery

🎭 Drama

🎉 Comedy

🦸 Superhero

...or make up your own!

This genre is passed as context to the LLM.

## 🧪 Voice Cloning (Optional)
Record a 20-second voice sample and save it as:
audio/reference_audio.wav

Edit the main.py to include a reference text that matches your voice (helps TTS align better).

Install and configure F5-TTS or your preferred voice model.

Make sure the CLI tool is available:
f5-tts_infer-cli --text "Your story here" --ref audio/reference_audio.wav

## 📽️ Output
Once the story and narration are generated, the program displays a slideshow:

Each image appears

Subtitles are shown

Audio is played frame-by-frame

You can save the audio (outputX.wav) from /tests/.

## 🧠 Powered By
BLIP-2 – Vision-Language model for captions

YOLOv8 – Object detection

LLaMA 2 – Story generation LLM

F5-TTS – Voice synthesis

## 💡 Future Ideas
Add background music and transitions

Support for longer stories (multi-paragraph)

Web-based drag-and-drop UI

Real-time mobile integration

## 📄 License
MIT License – feel free to modify and build on this project.
