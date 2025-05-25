import torch
import cv2
import os
import time
import argparse
import subprocess
import re
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
from collections import Counter
from llama_cpp import Llama

# Path to images
path_to_images = "images/"

# Device
if torch.cuda.is_available():
    print("Using GPU!")
    DEVICE = "cuda"
else:
    print("Using CPU...")
    DEVICE = "cpu"

# Encoders
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name, cache_dir="models/blip2_processor", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir="models/blip2_model", torch_dtype=torch.float16).to(DEVICE).eval()

# Object detection
yolo_model = YOLO("models/yolo11x.pt").to(DEVICE).eval() # Use n-nano, s-small, m-medium, l-large, x-xtra large

class Visualizer():
    def __init__(self, model, processor, yolo_model, sys_prompt, llm):
        self.model = model
        self.processor = processor
        self.yolo_model = yolo_model
        self.sys_prompt = sys_prompt
        self.llm = llm

    def generate_caption(self, image):
        with torch.no_grad():
            input = self.processor(images=image, return_tensors="pt").to(DEVICE)
            output = model.generate(**input)
        print(f"\nCaption:{self.processor.decode(output[0], skip_special_tokens=True)}")
        return self.processor.decode(output[0], skip_special_tokens=True)

    def get_objects(self, image) -> dict:
        with torch.no_grad():
            results = self.yolo_model(image)[0]
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        filtered_ids = [cid for cid, conf in zip(class_ids, confidences) if conf > 0.5]
        class_counts = Counter(filtered_ids)
        named_counts = {self.yolo_model.names[cid]: count for cid, count in class_counts.items()}

        return named_counts
    
    def generate_better_caption(self, caption, context, max_tokens = 50):
        prompt = f"[INST] <<SYS>>\n{self.sys_prompt}\n<</SYS>>\n\n The current history: {context} \n Current caption:{caption} [/INST]"
        output = self.llm(prompt, max_tokens=max_tokens, echo=False) 
        return output["choices"][0]["text"]


# Path to LLM 
MODEL_PATH = "models/llama-2-7b-chat.Q5_K_M.gguf" 

# Load model
n_gpu_layers_val = 35
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers_val,
    n_ctx=4096,
    n_threads=6,
    verbose=False 
)

class StoryTeller():
    def __init__(self, sys_message: str, model):
        self.sys_message = sys_message
        self.model = model
        self.conversation_history = [""]
    
    def get_message(self):
        return ''.join(self.conversation_history)

    def generate_story(self, message: str, max_tokens=300):
        prompt = f"[INST] <<SYS>>\n{self.sys_message}\n<</SYS>>\n\n" \
             f"[NO CONVERSATIONAL ACKNOWLEDGMENTS ALLOWED]\n" \
             f"The current history: {self.get_message()}\n" \
             f"Current description: {message} [/INST]"
        
        output = self.model(prompt, max_tokens=max_tokens, echo=False) 
        response = output["choices"][0]["text"]
        if any(phrase in re.findall(r"\b\w+\b", response.lower()) for phrase in ["here", "sure", "great"]):
            response = '\n'.join(line for line in response.split("\n") if not any(phrase in re.findall(r"\b\w+\b", line.lower()) for phrase in ["here", "sure", "great"]))

        self.conversation_history.append(response)
    
    def get_story(self):
        return self.conversation_history


# Narrator
narrator_model_path = "F5-TTS/ckpts/my_speak/model_2500.pt"   # This was the one that seemed like the best model 
ref_audio = "audio/reference_audio.wav"
ref_text = "Would you please be so kind to tell me why the quick brown fox jumped over the lazy dog?"
test_path = "tests/"

class Narrator():
    def __init__(self, model, test_path, ref_text, ref_audio):
        self.model = model
        self.path = test_path
        self.ref_text = ref_text
        self.ref_audio = ref_audio
    
    def narrate(self, text, i):
        command = f'f5-tts_infer-cli --ckpt_file "{self.model}" --ref_audio "{self.ref_audio}" --ref_text "{self.ref_text}" --gen_text "{text}" -w output{i}.wav'
        subprocess.run(command, shell=True)
        return

    def generate_all_audios(self, texts):
        for i, text in enumerate(texts):
            self.narrate(text, i+1)
        return

def take_photos():
    '''
    Take the photos for the story to be generated
    '''
    cap = cv2.VideoCapture(0)

    # Count the photos
    count = 0
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read frame, ending stream")
            break

        # Show the video feed
        cv2.imshow("Take photos!", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Ending stream")
            break

        elif key == ord('k'):
            count += 1
            print(f"Taking photo {count}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(rgb_frame)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    return frame_list 

def load_images_from_folder(folder_path):
    image_list = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    sorted_filenames = sorted(os.listdir(folder_path))
    for filename in sorted_filenames:
        if filename.lower().endswith(supported_extensions):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_list.append(rgb_image)
            else:
                print(f"Failed to read: {filename}")

    return image_list


def show_story(texts, frames):
    for i, (text, frame) in enumerate(zip(texts, frames)):
        window_name = f"Image {i+1}"
        cv2.namedWindow(window_name)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        draw_multiline_text(  # Put subtitles
            rgb_frame,
            text,
            start_pos=(50, 100),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            scale=1,
            color=(255, 255, 255),
            thickness=4,
            max_width=500
        )

        cv2.imshow(window_name, rgb_frame)
        cv2.moveWindow(window_name, 0, 0) # I was having problems in it not showing on the main monitor

        waveform, sample_rate = torchaudio.load(f'tests/output{i+1}.wav')
        sd.play(waveform.numpy().T, samplerate=sample_rate)
        while True:
            key = cv2.waitKey(100)  
            if key == ord('q'): # Go to next Frame
                sd.stop() 
                cv2.destroyAllWindows()
                break  
            elif key == ord('k'): # Exit
                sd.stop()
                cv2.destroyAllWindows()
                return  

            if not sd.get_stream().active: # Audio finished
                cv2.destroyAllWindows()
                break  
    
    cv2.destroyAllWindows()
    return 


def draw_multiline_text(img, text, start_pos, font, scale, color, thickness, max_width):
    """
    Draw multiline text on an image, wrapping lines if they exceed max_width.

    Parameters:
    - img: the image array (BGR or RGB)
    - text: the full text string
    - start_pos: (x, y) tuple for the top-left starting point
    - font: cv2 font, e.g., cv2.FONT_HERSHEY_SIMPLEX
    - scale: font scale
    - color: text color (B, G, R)
    - thickness: line thickness
    - max_width: maximum width (in pixels) before wrapping
    """
    x, y = start_pos
    line_height = cv2.getTextSize("Test", font, scale, thickness)[0][1] + 10  # Add some padding

    words = text.split(' ')
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]

        if text_size[0] > max_width:
            # Draw current line and start a new one
            cv2.putText(img, current_line.strip(), (x, y), font, scale, color, thickness, cv2.LINE_AA)
            y += line_height
            current_line = word + " "
        else:
            current_line = test_line

    # Draw the last line
    if current_line:
        cv2.putText(img, current_line.strip(), (x, y), font, scale, color, thickness, cv2.LINE_AA)



def main():
    parser = argparse.ArgumentParser(description="Generate Stories")
    parser.add_argument('--photo', action='store_true', help="Save captured frames to disk")
    args = parser.parse_args()
    if args.photo:
        frames = take_photos()
    else:
        frames = load_images_from_folder(path_to_images)

    if frames == []:
        print("No images detected. Use --photo to take photos or put images in images/")
        return
    
    genre = str(input("What genre do you wish the story to be in? "))

    system_prompt1 = f"""
    You are a creative storyteller in the {genre} genre. Your ONLY task is to generate exactly 1–2 sentences that naturally continue the current story, 
    while integrating relevant details from the current image description. 

    **DO NOT**:
    - Include any introductions, confirmations, conclusions, or conversational fillers.
    - Say things like "Great!", "Here it is:", "Sure," or "Okay, here you go."
    - Acknowledge the request in any way before or after the sentences.

    **FAILURE CONDITIONS**:
    - If you produce any text other than the 1–2 sentence continuation, it is considered an ERROR.
    - If you acknowledge your action in any way, it is considered an ERROR.
    - If you do not follow this structure exactly, it is considered an ERROR.

    **OUTPUT REQUIREMENT**:
    - Only the 1–2 sentence continuation. NOTHING else.
    - It must appear as if it is part of the story, with no extra commentary or formatting.
    """

    system_prompt2 = f"""
    You are a Visual Encoder. Your ONLY task is to provide a single, context-based description of an image, 
    informed by the previous story and the current image's description. 

    **DO NOT**:
    - Add any narrative or storytelling.
    - Include introductions, conclusions, or conversational fillers.
    - Write more than one sentence—exactly one concise, visual summary is required.

    **OUTPUT REQUIREMENT**:
    - Only the single, descriptive sentence. NOTHING else.
    - It must appear as a clear, visual summary, with no extra commentary or formatting."""

    storyteller = StoryTeller(system_prompt1, llm)
    visual = Visualizer(model, processor, yolo_model, system_prompt2, llm)
    narrator = Narrator(narrator_model_path, test_path, ref_text, ref_audio)
    end = ""
    for i, img in enumerate(frames):
        caption = visual.generate_caption(img)
        objects = visual.get_objects(img)
        context = storyteller.get_message()
        better_caption = visual.generate_better_caption(caption, context)
        if objects:
            objects_str = ', '.join([f"{obj}: {objects[obj]}" for obj in objects.keys()])
        else:
            objects_str = "No objects detected"

        if i == len(frames) - 1:
            end = "\nStory ends after this turn"

        prompt = f"Plot: {better_caption}\nObjects: [{objects_str}]  {end} "
        print(f"\n {prompt}")
        storyteller.generate_story(prompt)
        print(f"\n{storyteller.get_message()}")   

    list_texts = storyteller.get_story()[1:] # Remove the empty
    narrator.generate_all_audios(list_texts)  

    # Blank so the user can specify when to watch the story
    blank = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(blank, "Press Any key to continue!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Press any key", blank)
    print("Press any key to see story!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_story(list_texts, frames)
    
    return 

if __name__ == "__main__":
    main()
