📸🧠 Visual Storyteller AI 🎬🗣️
Turn a series of images into a visual story narrated in a custom voice!
This Python-based project uses BLIP2, YOLOv8, LLaMA, and F5-TTS to generate descriptive captions, create storylines, and voice-narrate them — all with just a few images.

🚀 Features
🧠 BLIP2: Extract intelligent captions from each image

👁️ YOLOv8: Detect objects in the scene with real-time object detection

📝 LLaMA 2 Chat: Generate engaging, genre-specific stories from image sequences

🔊 F5-TTS: Clone your voice from a 20-second reference and narrate the story

📷 Capture or Load: Use webcam or load images from a folder

🎭 Genre Selection: Pick a story genre (e.g., horror, fantasy, sci-fi) for personalized narration

🖼️ Visual Display: Watch your story unfold with subtitles and voiceover

🧰 Requirements
Python 3.8+

GPU strongly recommended (BLIP2 and LLaMA are heavy)

Dependencies (install via pip):

pip install torch torchvision torchaudio transformers opencv-python ultralytics sounddevice llama-cpp-python
Install F5-TTS and export your reference voice

Download:

LLaMA2 GGUF model (e.g., llama-2-7b-chat.Q5_K_M.gguf)

BLIP2 and YOLO model weights

Reference audio and text

📂 Folder Structure
css
Copiar
Editar
.
├── models/
│   ├── blip2_model/
│   ├── blip2_processor/
│   ├── llama-2-7b-chat.Q5_K_M.gguf
│   └── yolo11x.pt
├── audio/
│   └── reference_audio.wav
├── images/
│   └── (your images go here)
├── tests/
│   └── output1.wav ...
├── your_script.py
🧪 Run It!
🔴 Capture Photos with Webcam
bash
Copiar
Editar
python your_script.py --photo
🟢 Use Existing Images
bash
Copiar
Editar
python your_script.py
Then choose a genre and enjoy the storytelling magic!

👤 Voice Cloning Setup
Train or download a speaker model from F5-TTS.

Place the .pt file in F5-TTS/ckpts/my_speak/

Record 20s of reference audio (audio/reference_audio.wav)

Set the reference text used when training the speaker encoder.

🧠 How it Works (Simplified)
📷 Capture or load images

🧠 BLIP2 generates a caption

🔍 YOLOv8 identifies objects

🧩 LLaMA combines caption + object data to write a plot continuation

🗣️ F5-TTS reads it aloud in a cloned voice

🖼️ Story shown with subtitles and narration

🥧 Example Use Case
Take 5 photos of a toy story using your webcam.

Choose "fantasy" as your genre.

Watch and listen to your custom story unfold!

📜 License
MIT License. Use responsibly.

❤️ Inspired by
🤖 Salesforce BLIP2

🔍 Ultralytics YOLO

💬 Meta LLaMA

🔊 F5-TTS
