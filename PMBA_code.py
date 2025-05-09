import torch
import cv2
import os
import time
import argparse
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from llama_cpp import Llama

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

# Alterações a fazer
# 1 - Narrador

# Path to images
path_to_images = "images/"

# Device
if torch.cuda.is_available():
    print("Using GPU!")
    device = "cuda"
else:
    print("Using CPU...")
    device = "cpu"

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
        if any(phrase in response.lower() for phrase in ["here", "sure", "great"]): # If it has unncecessary information
            response = " " + '\n'.join(line for line in response.split("\n") if not any(phrase in line.lower() for phrase in ["here", "sure", "great"]))

        self.conversation_history.append(response)

# Encoders
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name, cache_dir="models/blip2_processor", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir="models/blip2_model", torch_dtype=torch.float32).to(device).eval()

class Visualizer():
    def __init__(self, model, processor, yolo_model, sys_prompt, llm):
        self.model = model
        self.processor = processor
        self.yolo_model = yolo_model
        self.sys_prompt = sys_prompt
        self.llm = llm

    def generate_caption(self, image):
        with torch.no_grad():
            input = self.processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**input)
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

        
# Object detection
yolo_model = YOLO("models/yolo11x.pt").to(device).eval() # Use n-nano, s-small, m-medium, l-large, x-xtra large

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
            print(f"📸 Taking photo {count}")
            frame_list.append(frame)

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
                image_list.append(img)
            else:
                print(f"⚠️ Failed to read: {filename}")
    return image_list

def main():
    parser = argparse.ArgumentParser(description="Generate Stories")
    parser.add_argument('--photo', action='store_true', help="Save captured frames to disk")
    args = parser.parse_args()
    if args.photo:
        frames = take_photos()
    else:
        frames = load_images_from_folder(path_to_images)

    if frames == []:
        return
    
    genre = str(input("What genre do you wish the story to be in? "))

    system_prompt1 = f"""
    You are a creative storyteller in the {genre} genre. Your ONLY task is to generate exactly 1–2 sentences that naturally continue the current story, 
    while integrating relevant details from the current image description. 

    🛑 **DO NOT**:
    - Include any introductions, confirmations, conclusions, or conversational fillers.
    - Say things like "Great!", "Here it is:", "Sure," or "Okay, here you go."
    - Acknowledge the request in any way before or after the sentences.

    ⚠️ **FAILURE CONDITIONS**:
    - If you produce any text other than the 1–2 sentence continuation, it is considered an ERROR.
    - If you acknowledge your action in any way, it is considered an ERROR.
    - If you do not follow this structure exactly, it is considered an ERROR.

    🔎 **OUTPUT REQUIREMENT**:
    - Only the 1–2 sentence continuation. NOTHING else.
    - It must appear as if it is part of the story, with no extra commentary or formatting.
    """

    system_prompt2 = f"""
    You are a Visual Encoder. Your ONLY task is to provide a single, context-based description of an image, 
    informed by the previous story and the current image's description. 

    🛑 **DO NOT**:
    - Add any narrative or storytelling.
    - Include introductions, conclusions, or conversational fillers.
    - Write more than one sentence—exactly one concise, visual summary is required.

    🔎 **OUTPUT REQUIREMENT**:
    - Only the single, descriptive sentence. NOTHING else.
    - It must appear as a clear, visual summary, with no extra commentary or formatting."""

    storyteller = StoryTeller(system_prompt1, llm)
    visual = Visualizer(model, processor, yolo_model, system_prompt2, llm)
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
    

if __name__ == "__main__":
    main()
