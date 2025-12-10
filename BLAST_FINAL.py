

import os
from dotenv import load_dotenv
import cv2
import time
import torch
import speech_recognition as sr
import pyttsx3
import threading
import queue

import google.generativeai as genai 
from PIL import Image
from ultralytics import YOLO


try:
    import pythoncom
except ImportError:
    pythoncom = None



load_dotenv(os.path.join(os.path.dirname(__file__), "secrets", "gemini.env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_gagan", "")


MIC_ID = 0


genai.configure(api_key=GEMINI_API_KEY)
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except:
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

print("🚀 BLAST Integrated System Starting...")


# THREADED CAMERA 

class CameraStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.status = status
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.status, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()


# 3. SPEECH ENGINE (Threaded)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            if pythoncom: pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            print(f"🔊 Blas: '{text}'")
            engine.say(text)
            engine.runAndWait()
            del engine
            if pythoncom: pythoncom.CoUninitialize()
        except Exception as e:
            print(f"Speech Error: {e}")
        finally:
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)


# 4. LOAD MODELS

# YOLO
try:
    print("🔄 Loading YOLOv8...")
    model = YOLO("yolov8x.pt") 
    if torch.cuda.is_available(): 
        model.to('cuda')
        print("✓ YOLO on CUDA")
    else:
        print("✓ YOLO on CPU")
except Exception as e:
    print(f"✗ YOLO Error: {e}")
    exit()

# MiDaS
try:
    print("🔄 Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    if torch.cuda.is_available(): midas.to('cuda')
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    print("✓ MiDaS Loaded")
except Exception: 
    midas = None
    print("⚠️ MiDaS not loaded (Depth disabled)")

# Global Vars
memory = {}
last_listen_time = time.time()
LISTEN_COOLDOWN = 4 


# 5. CORE FUNCTIONS (Logic)


def get_detailed_zone(x, y, frame_width, frame_height):
    """
    Returns precise location: Top-Left, Center, Bottom-Right, etc.
    """
    # Vertical Zone
    if y < frame_height / 3: vert = "top"
    elif y > 2 * frame_height / 3: vert = "bottom"
    else: vert = "center"
    
    # Horizontal Zone
    if x < frame_width / 3: horiz = "left"
    elif x > 2 * frame_width / 3: horiz = "right"
    else: horiz = "center"
    
    
    if vert == "center" and horiz == "center":
        return "in the center"
    elif vert == "center":
        return f"on the {horiz}"
    elif horiz == "center":
        return f"at the {vert}"
    else:
        return f"at the {vert}-{horiz}"

def calculate_distance_meters(depth_map, box):
    """
    Calculates distance in METERS using a calibration coefficient.
    """
    x1, y1, x2, y2 = map(int, box)
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    h, w = depth_map.shape
    
    x_safe = min(w-1, max(0, x_center))
    y_safe = min(h-1, max(0, y_center))
    
    depth_val = depth_map[y_safe, x_safe]
    
    if depth_val <= 0: return 0.0
    
    
    return (800 / depth_val) 

def update_memory(label, zone, distance):
    memory[label] = {
        "zone": zone, 
        "timestamp": time.strftime("%H:%M:%S"), 
        "distance": distance
    }

def listen_command():
    r = sr.Recognizer()
    try:
        with sr.Microphone(device_index=MIC_ID) as source:
            print(f"\n🎤 Listening (Mic {MIC_ID})...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            r.energy_threshold = 300 
            
            speak("Listening.")
            
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            print("Processing audio...")
            text = r.recognize_google(audio)
            print(f"✓ You said: '{text}'")
            return text
    except sr.WaitTimeoutError:
        speak("I didn't hear anything.")
        return None
    except sr.UnknownValueError:
        speak("I didn't understand.")
        return None
    except Exception as e:
        print(f"Mic Error: {e}")
        return None

# GEMINI TASKS 

def describe_scene(frame):
    def task():
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            resp = gemini_model.generate_content(["Describe this scene briefly for a blind person.", pil_img])
            speak(resp.text)
        except Exception as e: speak("Connection error.")
    threading.Thread(target=task).start()

def read_text(frame):
    def task():
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            resp = gemini_model.generate_content(["Read any text in this image naturally.", pil_img])
            speak(resp.text)
        except Exception as e: speak("Reading failed.")
    threading.Thread(target=task).start()


# 6. MAIN LOOP

def main():
    global last_listen_time
    
    
    cam = CameraStream(0)
    time.sleep(1.0) 

    print("\n✅ BLAST Integrated Running")
    print("⌨️ Press 's' to speak")
    print("⌨️ Press 'q' to quit")

    while True:
        # Get frame from threaded camera
        ret, frame = cam.read()
        if not ret or frame is None: continue
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # 1. YOLO Inference
        results = model(frame, verbose=False)[0]
        
        # 2. MiDaS Depth
        depth_map = None
        if midas:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_img = transform(img).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                pred = midas(t_img)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            depth_map = pred.cpu().numpy()

        # 3. Process Objects
        for r in results.boxes:
            box = r.xyxy[0]
            label = model.names[int(r.cls[0])]
            x1, y1, x2, y2 = map(int, box)
            
            # Distance & Zone
            dist_m = 0
            if depth_map is not None:
                dist_m = calculate_distance_meters(depth_map, box)
            
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            zone = get_detailed_zone(center_x, center_y, frame.shape[1], frame.shape[0])
            
            update_memory(label, zone, dist_m)
            
            # VISUALIZATION 
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label Background
            label_text = f"{label}: {dist_m:.1f}m" if dist_m > 0 else label
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("BLAST Integrated", frame)

        # 4. Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if time.time() - last_listen_time > LISTEN_COOLDOWN:
                command = listen_command()
                if command:
                    cl = command.lower()
                    if "describe" in cl: describe_scene(frame)
                    elif "read" in cl: read_text(frame)
                    
                    # Logic for "Where is" / "How far"
                    elif "where" in cl or "find" in cl or "far" in cl or "distance" in cl:
                        found = False
                        for obj_name in memory.keys():
                            if obj_name in cl:
                                info = memory[obj_name]
                                d = info["distance"]
                                z = info["zone"]
                                speak(f"The {obj_name} is {z}, about {d:.1f} meters away.")
                                found = True
                                break
                        if not found: speak("I don't see that object right now.")
                    
                last_listen_time = time.time()
                
        elif key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    print("Terminated.")

if __name__ == "__main__":
    main()