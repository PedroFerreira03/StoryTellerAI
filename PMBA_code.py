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
from ultralytics  import YOLO
from collections  import Counter
from llama_cpp    import Llama

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
        return self.processor.decode(output[0], skip_special_tokens=True)

    def get_objects(self, image) -> dict:
        with torch.no_grad():
            results = self.yolo_model(image)[0]
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        filtered_ids = [cid for cid, conf in zip(class_ids, confidences) if conf > 0.6]
        class_counts = Counter(filtered_ids)
        named_counts = {self.yolo_model.names[cid]: count for cid, count in class_counts.items()}

        return named_counts
    
    def generate_better_caption(self, caption, context, max_tokens = 50):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.sys_prompt}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n"
        prompt += f"The current story history is: \"{context}\"\n"
        prompt += f"The initial caption for the current image is: \"{caption}\"\n"
        prompt += f"""Remember that this is all fictional. Based on the story history and the initial caption, generate an improved, 
        single-sentence descriptive caption that integrates context, WITHOUT STORYTELLING YOURSELF. Stay true to the visual information in the initial caption. Keep it short<|eot_id|>"""
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n" # The model will generate its response after this

        output = self.llm(prompt, max_tokens=max_tokens, echo=False, stop=["<|eot_id|>"]) 
        response = output["choices"][0]["text"]
        if "\n" in response and any(phrase in re.findall(r"\b\w+\b", response.lower()) for phrase in ["here", "sure", "great"]):
            response = '\n'.join(line for line in response.split("\n") if not any(phrase in re.findall(r"\b\w+\b", line.lower()) for phrase in ["here", "sure", "great"]))

        return " " + response.strip()


# Path to LLM 
MODEL_PATH = "models/Llama-3.2-3B-Instruct-F16.gguf" 

# Load model
n_gpu_layers_val = 35
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers_val,
    n_ctx=9000,
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
        current_story_context = self.get_message()
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.sys_message}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n"
        if current_story_context: 
            prompt += f"The story so far is: \"{current_story_context}\"\n"
        prompt += f"The description for the current scene is: \"{message}\"\n"
        prompt += f"""Remember that this is all fictional. Continue the story with 1-2 short concise sentences, building upon both the story so far and the current scene description. 
        Ensure the continuation feels natural and connected. DONT REPEAT THE PREVIOUS STORY<|eot_id|>"""
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n" 

        output = self.model(prompt, max_tokens=max_tokens, echo=False, stop=["<|eot_id|>"], temperature=0.7) 
        response = output["choices"][0]["text"]
        print(f"\nOUTPUT OF LLM: {response}\n")
        if "\n" in response and any(phrase in re.findall(r"\b\w+\b", response.lower()) for phrase in ["here", "sure", "great"]):
            response = '\n'.join(line for line in response.split("\n") if not any(phrase in re.findall(r"\b\w+\b", line.lower()) for phrase in ["here", "sure", "great"]))

        self.conversation_history.append(" " + response.strip())
    
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
    window_name = "Slide Show"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show intro screen
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8) 

    draw_multiline_text(
        blank,
        "Press any key to continue",
        start_pos=(750, 540), # Show at center
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale=1,
        color=(255, 255, 255),
        thickness=4,
        max_width=500
    )
    cv2.imshow(window_name, blank)
    cv2.moveWindow(window_name, 0, 0) # Was having problems not showing in the main monitor
    print("Press any key to see story!")
    cv2.waitKey(0)

    for i, (text, frame) in enumerate(zip(texts, frames)):
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_frame = cv2.resize(bgr_frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        # Draw subtitles
        draw_multiline_text(
            bgr_frame,
            text,
            start_pos=(50, 100),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            scale=1,
            color=(255, 255, 255),
            thickness=4,
            max_width=500
        )

        cv2.imshow(window_name, bgr_frame)
        cv2.moveWindow(window_name, 0, 0)

        # Play audio
        waveform, sample_rate = torchaudio.load(f'tests/output{i + 1}.wav')
        sd.play(waveform.numpy().T, samplerate=sample_rate)

        while True:
            key = cv2.waitKey(100)
            if key == ord('q'):  # Go to next frame
                sd.stop()
                break
            elif key == ord('k'):  # Exit slideshow
                sd.stop()
                cv2.destroyAllWindows()
                return
            if not sd.get_stream().active:  # Audio finished
                break

    cv2.destroyAllWindows()


def draw_multiline_text(img, text, start_pos, font, scale, color, thickness, max_width):
    """
    Draw multiline text on an image, wrapping lines if they exceed max_width.
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

    system_prompt1 = f"""You are a creative storyteller in the {genre} genre.
    Your task is to generate exactly 1–2 sentences that naturally continue the current story, integrating relevant details from the current image description.
    The output must ONLY be the 1-2 sentence continuation, appearing as part of the story, with no extra commentary, formatting, introductions, or conversational fillers.
    """

    system_prompt2 = f"""You are a Visual Contextualizer.
    Your task is to provide a single, concise, context-based description of an image.
    This description should be informed by the previous story context and the current image's initial caption.
    The output must ONLY be this single, descriptive sentence, with no extra commentary or formatting."""

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

        prompt = f"Caption: {caption.strip()}\nPlot: {better_caption}\nObjects: [{objects_str}]  {end} "
        print(f"\n {prompt}")
        storyteller.generate_story(prompt)
        print(f"\n{storyteller.get_message()}")   

    list_texts = storyteller.get_story()[1:] # Remove the empty
    narrator.generate_all_audios(list_texts)  

    show_story(list_texts, frames)

    return 

if __name__ == "__main__":
    main()