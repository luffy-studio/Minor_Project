# # diagnose_mapping.py
# # Usage: edit DATASET_PATH and MODEL_PATH below, then run:
# #   python diagnose_mapping.py

# import os, sys, traceback
# import numpy as np
# import cv2
# from collections import defaultdict
# import math
# import time

# # Edit these to your paths
# DATASET_PATH = r"C:\Users\sarva\Desktop\Sign-Language-To-Text-and-Speech-Conversion-3543543master\Sign-Language-To-Text-and-Speech-Conversion-master\AtoZ_3.1 copy"
# MODEL_PATH = r"./cnn8grps_rad1_model.h5"   # change if different

# # Optional: limit per-folder samples (set None to use all)
# SAMPLES_PER_FOLDER = 30

# # Preprocessing variants to try
# PREPROCESS_VARIANTS = [
#     "bgr_norm",    # BGR floats 0..1 (cv2 default)
#     "rgb_norm",    # RGB floats 0..1
#     "rgb_imagenet",# RGB normalized by ImageNet mean/std
#     "gray_norm",   # grayscale 0..1 (single channel)
#     "bgr_255",     # BGR ints 0..255
# ]

# def load_keras_model(path):
#     try:
#         from tensorflow.keras.models import load_model
#     except Exception as e:
#         print("ERROR: tensorflow.keras not available:", e)
#         raise
#     print("Loading model from", path)
#     model = load_model(path)
#     print("Model loaded.")
#     return model

# def list_class_folders(dataset_path):
#     folders = []
#     for name in sorted(os.listdir(dataset_path)):
#         full = os.path.join(dataset_path, name)
#         if os.path.isdir(full):
#             folders.append((name, full))
#     return folders

# def read_image(path, target_size=(400,400)):
#     img = cv2.imread(path)
#     if img is None:
#         return None
#     img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
#     return img

# def preprocess_variant(img, variant):
#     # img is BGR uint8 resized to (400,400,3)
#     if variant == "bgr_norm":
#         arr = img.astype("float32") / 255.0            # B,G,R channels last
#     elif variant == "rgb_norm":
#         arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
#     elif variant == "rgb_imagenet":
#         tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
#         mean = np.array([0.485, 0.456, 0.406], dtype="float32")
#         std  = np.array([0.229, 0.224, 0.225], dtype="float32")
#         arr = (tmp - mean) / std
#     elif variant == "gray_norm":
#         g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
#         arr = np.expand_dims(g, axis=-1)   # (H,W,1)
#     elif variant == "bgr_255":
#         arr = img.astype("float32")
#     else:
#         raise ValueError("Unknown variant: "+variant)
#     return arr

# def sample_files(folder, limit=None):
#     exts = (".jpg",".jpeg",".png",".bmp")
#     files = [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
#     if limit is None or limit >= len(files):
#         return files
#     # pick uniformly spaced samples
#     step = max(1, len(files)//limit)
#     sel = files[::step][:limit]
#     return sel

# def main():
#     print("DATASET_PATH:", DATASET_PATH)
#     print("MODEL_PATH:", MODEL_PATH)
#     if not os.path.exists(DATASET_PATH):
#         print("Dataset path not found.")
#         return
#     if not os.path.exists(MODEL_PATH):
#         print("Model file not found at", MODEL_PATH)
#         return

#     try:
#         model = load_keras_model(MODEL_PATH)
#     except Exception as e:
#         print("Failed to load model:", e)
#         traceback.print_exc()
#         return

#     # show model input/output shape info
#     try:
#         inp_shape = model.input_shape
#         out_shape = model.output_shape
#         print("Model input_shape:", inp_shape)
#         print("Model output_shape:", out_shape)
#     except Exception:
#         print("Could not read model shapes.")

#     folders = list_class_folders(DATASET_PATH)
#     if not folders:
#         print("No class subfolders found in dataset path.")
#         return

#     print("Found class folders (count={}):".format(len(folders)))
#     for i,(name,full) in enumerate(folders):
#         print(i, name, full)

#     # run predictions per folder and per preprocess variant
#     results = {}  # results[variant][folder_name] = list of (top_idx, prob)
#     for variant in PREPROCESS_VARIANTS:
#         print("\n=== Running variant:", variant, "===\n")
#         results[variant] = {}
#         for cname, cfull in folders:
#             files = sample_files(cfull, SAMPLES_PER_FOLDER)
#             if not files:
#                 print("No images in folder", cname)
#                 continue
#             top_indices = []
#             top_probs = []
#             for fpath in files:
#                 img = read_image(fpath)
#                 if img is None:
#                     continue
#                 arr = preprocess_variant(img, variant)
#                 # adapt batch shape depending on model input
#                 batch = np.expand_dims(arr, axis=0)
#                 # if model expects channels-first, try transpose (Heuristic)
#                 # We'll attempt direct predict, if shape mismatch -> try channels_first
#                 preds = None
#                 try:
#                     preds = model.predict(batch, verbose=0)
#                 except Exception as e:
#                     # try channels-first if possible
#                     try:
#                         b2 = np.transpose(batch, (0,3,1,2))
#                         preds = model.predict(b2, verbose=0)
#                     except Exception as e2:
#                         # give up for this image
#                         print("Model predict failed for image", fpath, "variant", variant)
#                         # print(e); print(e2)
#                         preds = None
#                 if preds is None:
#                     continue
#                 prob = np.array(preds[0]).ravel()
#                 idx = int(np.argmax(prob))
#                 p = float(np.max(prob))
#                 top_indices.append(idx)
#                 top_probs.append(p)
#             if not top_indices:
#                 print("No predictions for folder", cname, "variant", variant)
#                 continue
#             # aggregate: mode of top_indices, and average prob
#             from collections import Counter
#             counter = Counter(top_indices)
#             mode_idx, mode_count = counter.most_common(1)[0]
#             avg_prob = sum(top_probs)/len(top_probs)
#             results[variant][cname] = {"mode_idx": mode_idx, "mode_count": mode_count, "n": len(top_indices), "avg_prob": avg_prob}
#             print("Folder:", cname, "-> mode_index:", mode_idx, "count:", mode_count, "n:", len(top_indices), "avg_prob:", round(avg_prob,3))

#     # Summarize best variant per folder
#     print("\n\n=== Summary per folder: best-preprocess suggestion based on highest avg_prob ===\n")
#     for cname, _ in folders:
#         best = None
#         for variant in PREPROCESS_VARIANTS:
#             info = results.get(variant, {}).get(cname)
#             if info is None:
#                 continue
#             score = info["avg_prob"]
#             if best is None or score > best[0]:
#                 best = (score, variant, info)
#         if best is None:
#             print(cname, "-> no data")
#             continue
#         score, variant, info = best
#         suggested_label = info["mode_idx"]
#         print(f"{cname} : best_variant={variant} avg_prob={score:.3f} predicted_model_index={suggested_label}")

#     # Build mapping suggestion: folder_name -> predicted model index
#     mapping = {}
#     for variant in PREPROCESS_VARIANTS:
#         for cname in results.get(variant, {}):
#             info = results[variant][cname]
#             mapping.setdefault(cname, []).append((variant, info["avg_prob"], info["mode_idx"], info["n"]))
#     print("\n=== Mapping candidates per folder (variant, avg_prob, mode_idx, n) ===")
#     for cname, lst in mapping.items():
#         print(cname)
#         for t in lst:
#             print("   ", t)

#     print("\nDONE. Interpret results above.")
#     print("If many folders map to the same model index -> label-order mismatch (or model collapsed).")
#     print("If avg_prob values are very small (eg <0.2) across variants -> model may be poorly trained or expects different preprocessing/shape.")
#     print("If one variant consistently gives higher avg_prob -> use that preprocessing in runtime code (e.g. rgb_norm).")
#     print("\nNext steps:")
#     print("1) If mapping shows folder 'A' -> model_index 4, 'B' -> 2, etc: reorder your LABELS in runtime to match.")
#     print("   Example: if folder order is A..Z but model outputs index meaning different letter, set LABELS = ['Q','W',...].")
#     print("2) If model expects channels-first, note that in preprocess you must transpose input to (1, C, H, W).")
#     print("3) If multiple folders map to same index -> check training logs, class imbalance, or try evaluating model on holdout test set.")

# if __name__ == "__main__":
#     main()
# signlang_tts_complete_fixed.py
"""
Robust Sign Language -> Text + TTS app (single-file).
Features:
 - Automatic preprocessing selection (tries variants and picks one that gives highest confidence)
 - Stable-frame append logic
 - TTS chain: win32 (SAPI) -> pyttsx3 -> gTTS+playsound
 - Safe handling of OpenCV lacking GUI builds
 - Detailed debug prints + saves debug images (debug_input_*.jpg)
Notes:
 - Set LABELS to match your model's training label order (default A..Z).
 - Optional libs: pywin32, pyttsx3, gTTS, playsound, enchant, cvzone
"""
import os
import sys
import time
import math
import tempfile
import traceback
import threading
import platform
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

# OpenCV
try:
    import cv2
except Exception as e:
    print("Fatal: cv2 import failed:", e)
    raise

# cvzone HandDetector (optional)
try:
    from cvzone.HandTrackingModule import HandDetector
except Exception as e:
    print("⚠️ cvzone.HandTrackingModule import failed:", e)
    HandDetector = None

# TensorFlow Keras model
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    print("⚠️ TensorFlow/Keras import failed:", e)
    load_model = None

# enchant suggestions (optional)
try:
    import enchant
    ddd = enchant.Dict("en-US")
    ENCHANT_AVAILABLE = True
except Exception as e:
    ENCHANT_AVAILABLE = False
    ddd = None

# ------------------ TTS backends detection ------------------
WIN32_AVAILABLE = False
PYTTSX3_AVAILABLE = False
GTTS_AVAILABLE = False
PLAYSOUND_AVAILABLE = False
wincl = None

if platform.system() == "Windows":
    try:
        import win32com.client as _wincl
        wincl = _wincl
        WIN32_AVAILABLE = True
    except Exception:
        WIN32_AVAILABLE = False

try:
    import pyttsx3
    try:
        eng_probe = pyttsx3.init()
        try:
            eng_probe.stop()
        except Exception:
            pass
        PYTTSX3_AVAILABLE = True
    except Exception:
        PYTTSX3_AVAILABLE = False
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except Exception:
    PLAYSOUND_AVAILABLE = False

def choose_tts_backend():
    if WIN32_AVAILABLE:
        return "win32"
    if PYTTSX3_AVAILABLE:
        return "pyttsx3"
    if GTTS_AVAILABLE and PLAYSOUND_AVAILABLE:
        return "gtts"
    return None

# ------------------ Config ------------------
MODEL_PATH = "./cnn8grps_rad1_model.h5"
# If your model used a custom mapping, change LABELS accordingly:
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEBUG_SAVE_INPUT = True
STABLE_FRAMES = 8
OFFSET = 29
# During first few predictions, the system will try these preprocess variants (in order)
PREPROCESS_VARIANTS = [
    "bgr_norm",         # BGR floats 0-1 channels-last (cv2 default)
    "rgb_norm",         # RGB floats 0-1 channels-last
    "rgb_imagenet",     # RGB normalized with imagenet mean/std
    "gray_norm",        # grayscale 0-1 single channel
    "bgr_255",          # BGR ints 0-255
]

# ------------------ Safe HandDetector init ------------------
if HandDetector is not None:
    try:
        hd = HandDetector(maxHands=1)
        hd2 = HandDetector(maxHands=1)
    except Exception as e:
        print("⚠️ HandDetector init error:", e)
        hd = None
        hd2 = None
else:
    hd = None
    hd2 = None

# ------------------ Application ------------------
class Application:
    def __init__(self):
        print("\n" + "="*60)
        print(" SIGN LANGUAGE -> TEXT + TTS (complete fixed)")
        print("="*60 + "\n")

        # camera
        self.vs = self.find_camera()
        if self.vs is None:
            messagebox.showerror("Camera error", "No camera found. Connect camera and restart.")
            sys.exit(1)

        # model
        self.model = None
        if load_model is not None and os.path.exists(MODEL_PATH):
            try:
                print("Loading model:", MODEL_PATH)
                self.model = load_model(MODEL_PATH)
                print("✓ Model loaded, output shape:", getattr(self.model, "output_shape", None))
            except Exception as e:
                print("⚠️ Model load failed:", e)
                traceback.print_exc()
                self.model = None
        else:
            print("⚠️ Model missing or tensorflow not available. Predictions disabled until model is loaded.")

        # TTS
        self.tts_backend = choose_tts_backend()
        print("TTS backend:", self.tts_backend)
        self.speaking = False
        self.speak_engine = None  # for pyttsx3 lazy init

        # state
        self.str = ""
        self.current_symbol = "Ready"
        self.word = ""
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.pts = None
        self.STABLE_FRAMES = STABLE_FRAMES
        self.stable_counter = 0
        self.prev_stable_label = None
        self.last_appended_label = None
        self.count = -1
        self.ten_prev_char = [" "]*10

        # autodetect preprocessing (None until chosen)
        self.selected_preprocess = None
        self.preprocess_attempts = 0
        self.preprocess_max_attempts = 3  # try selection for first few frames

        # white canvas
        self.white_path = "./white.jpg"
        if not os.path.exists(self.white_path):
            white = np.ones((400,400,3), np.uint8)*255
            cv2.imwrite(self.white_path, white)
            print("✓ white.jpg created")

        # GUI
        self.setup_gui()
        print("System ready. Show your hand to camera.")
        self.video_loop()

    def find_camera(self):
        print("Searching camera indices 0..4")
        for idx in range(5):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_ANY)
            except Exception:
                cap = cv2.VideoCapture(idx)
            if cap is None:
                continue
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✓ Camera opened at index", idx)
            return cap
        print("❌ No camera found")
        return None

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1300x820")
        self.root.configure(bg='#2c3e50')
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.bind('<Key>', self._on_keypress)

        # title
        title_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, borderwidth=3)
        title_frame.place(x=50, y=10, width=1200, height=60)
        tk.Label(title_frame, text="Sign Language Recognition System", font=("Arial", 26, "bold"),
                 bg='#34495e', fg='#ecf0f1').pack(pady=10)

        # camera panel
        camera_frame = tk.Frame(self.root, bg='#34495e', relief=tk.SUNKEN, borderwidth=3)
        camera_frame.place(x=50, y=90, width=580, height=550)
        tk.Label(camera_frame, text="📹 CAMERA FEED", font=("Arial", 14, "bold"),
                 bg='#34495e', fg='#3498db').pack(pady=5)
        self.panel = tk.Label(camera_frame, bg='black')
        self.panel.pack(padx=10, pady=5)

        # skeleton panel
        skeleton_frame = tk.Frame(self.root, bg='#34495e', relief=tk.SUNKEN, borderwidth=3)
        skeleton_frame.place(x=670, y=90, width=580, height=550)
        tk.Label(skeleton_frame, text="✋ HAND SKELETON", font=("Arial", 14, "bold"),
                 bg='#34495e', fg='#e74c3c').pack(pady=5)
        self.panel2 = tk.Label(skeleton_frame, bg='white')
        self.panel2.pack(padx=10, pady=5)

        # detected char / sentence
        char_frame = tk.Frame(self.root, bg='#ecf0f1', relief=tk.RIDGE, borderwidth=3)
        char_frame.place(x=50, y=660, width=580, height=70)
        tk.Label(char_frame, text="Detected:", font=("Arial", 18, "bold"),
                 bg='#ecf0f1', fg='#2c3e50').pack(side=tk.LEFT, padx=20)
        self.panel3 = tk.Label(char_frame, text="Ready", font=("Arial", 32, "bold"),
                               fg='#e74c3c', bg='#ecf0f1')
        self.panel3.pack(side=tk.LEFT, padx=20)

        sent_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, borderwidth=3)
        sent_frame.place(x=670, y=660, width=580, height=70)
        tk.Label(sent_frame, text="💬 SENTENCE:", font=("Arial", 12, "bold"),
                 bg='white', fg='#2c3e50', anchor='w').pack(fill=tk.X, padx=10, pady=(5,0))
        self.panel5 = tk.Label(sent_frame, text="", font=("Arial", 16, "bold"),
                               bg='white', fg='#27ae60', anchor='w', justify='left')
        self.panel5.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,5))

        if ENCHANT_AVAILABLE:
            tk.Label(self.root, text="💡 Word Suggestions:", font=("Arial", 12, "bold"),
                     fg='#f39c12', bg='#2c3e50').place(x=50, y=745)
            btn_w, btn_h = 135, 35
            self.b1 = tk.Button(self.root, text="", font=("Arial", 10), command=self.action1)
            self.b1.place(x=50, y=770, width=btn_w, height=btn_h)
            self.b2 = tk.Button(self.root, text="", font=("Arial", 10), command=self.action2)
            self.b2.place(x=195, y=770, width=btn_w, height=btn_h)
            self.b3 = tk.Button(self.root, text="", font=("Arial", 10), command=self.action3)
            self.b3.place(x=340, y=770, width=btn_w, height=btn_h)
            self.b4 = tk.Button(self.root, text="", font=("Arial", 10), command=self.action4)
            self.b4.place(x=485, y=770, width=btn_w, height=btn_h)

        # controls
        self.speak_btn = tk.Button(self.root, text="🔊 SPEAK", font=("Arial", 16, "bold"),
                                   bg='#27ae60', fg='white', command=self.speak_fun)
        self.speak_btn.place(x=1020, y=770, width=120, height=35)
        self.clear_btn = tk.Button(self.root, text="🗑️ CLEAR", font=("Arial", 16, "bold"),
                                   bg='#e74c3c', fg='white', command=self.clear_fun)
        self.clear_btn.place(x=1160, y=770, width=120, height=35)

        self.tts_label = tk.Label(self.root, text=f"TTS: {self.tts_backend or 'NONE'}", font=("Arial", 10),
                                  fg='white', bg='#2c3e50')
        self.tts_label.place(x=1020, y=740)

    def _on_keypress(self, event):
        if hasattr(event, 'char') and event.char.lower() == 'q':
            self.destructor()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                # try reopen
                print("Camera read failed; attempting reopen")
                try: self.vs.release()
                except Exception: pass
                time.sleep(0.1)
                self.vs = self.find_camera()
                self.root.after(50, self.video_loop)
                return

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # detect hands
            hands = []
            if hd is not None:
                try:
                    hands, display_frame = hd.findHands(display_frame, draw=True, flipType=False)
                except Exception as e:
                    print("hd.findHands failed:", e)
                    hands = []

            # show camera feed
            try:
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (550,480))
                im = Image.fromarray(resized)
                imgtk = ImageTk.PhotoImage(im)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
            except Exception as e:
                print("Error updating camera panel:", e)

            # white canvas
            white = cv2.imread(self.white_path)
            if white is None:
                white = np.ones((400,400,3), np.uint8) * 255

            if hands:
                hand = hands[0]
                x,y,w,h = hand.get('bbox', (0,0,0,0))
                y1 = max(0, y - OFFSET); y2 = min(frame.shape[0], y + h + OFFSET)
                x1 = max(0, x - OFFSET); x2 = min(frame.shape[1], x + w + OFFSET)
                image_crop = frame[y1:y2, x1:x2]

                try:
                    full_lm = hand.get('lmList', None)
                    if full_lm and len(full_lm) >= 21:
                        pts_in_crop = []
                        for lm in full_lm:
                            cx = int(lm[0]) - x1
                            cy = int(lm[1]) - y1
                            pts_in_crop.append([cx, cy])
                        self.pts = pts_in_crop
                        os_off = ((400 - w)//2) - 15
                        os1_off = ((400 - h)//2) - 15
                        self.draw_hand_skeleton(white, os_off, os1_off)

                        if self.model is not None:
                            # if preprocessing not chosen yet, run detect_preprocess attempt
                            if self.selected_preprocess is None and self.preprocess_attempts < self.preprocess_max_attempts:
                                self.detect_preprocess_and_predict(white)
                                self.preprocess_attempts += 1
                            else:
                                self.predict(white)
                    else:
                        # fallback with hd2 on crop
                        if hd2 is not None:
                            try:
                                handz, _ = hd2.findHands(image_crop, draw=False, flipType=False)
                                if handz:
                                    self.pts = handz[0].get('lmList', None)
                                    os_off = ((400 - w)//2) - 15
                                    os1_off = ((400 - h)//2) - 15
                                    self.draw_hand_skeleton(white, os_off, os1_off)
                                    if self.model is not None:
                                        if self.selected_preprocess is None and self.preprocess_attempts < self.preprocess_max_attempts:
                                            self.detect_preprocess_and_predict(white)
                                            self.preprocess_attempts += 1
                                        else:
                                            self.predict(white)
                            except Exception as e:
                                print("hd2.findHands failed:", e)
                except Exception as e:
                    print("Error mapping landmarks:", e)
                    traceback.print_exc()

            # update skeleton panel
            try:
                white_resized = cv2.resize(white, (450,450))
                sk_rgb = cv2.cvtColor(white_resized, cv2.COLOR_BGR2RGB)
                im2 = Image.fromarray(sk_rgb)
                imgtk2 = ImageTk.PhotoImage(im2)
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2)
            except Exception as e:
                print("Error updating skeleton panel:", e)

            # update labels
            self.panel3.config(text=self.current_symbol)
            self.panel5.config(text=self.str or "")

            if ENCHANT_AVAILABLE:
                try:
                    self.b1.config(text=self.word1)
                    self.b2.config(text=self.word2)
                    self.b3.config(text=self.word3)
                    self.b4.config(text=self.word4)
                except Exception:
                    pass

        except Exception as e:
            print("Video loop uncaught error:", e)
            traceback.print_exc()
        finally:
            self.root.after(10, self.video_loop)

    def draw_hand_skeleton(self, white, os, os1):
        if self.pts is None or len(self.pts) < 21:
            return
        pts = self.pts
        lt = 5; pr = 6
        for t in range(0,4):
            cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), lt)
        for t in range(5,8):
            cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (255,0,0), lt)
        for t in range(9,12):
            cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,0,255), lt)
        for t in range(13,16):
            cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (255,0,255), lt)
        for t in range(17,20):
            cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,165,255), lt)
        cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (100,100,100), lt)
        cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (100,100,100), lt)
        cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (100,100,100), lt)
        cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (100,100,100), lt)
        cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (100,100,100), lt)
        for i in range(21):
            cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), pr, (255,255,0), -1)
            cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), pr, (0,0,0), 1)

    def distance(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # ----- Automatic preprocess selection helper -----
    def detect_preprocess_and_predict(self, test_image):
        """
        Try several preprocessing variants and pick the one that gives the highest max probability.
        Sets self.selected_preprocess and returns predict result via standard flow.
        """
        if self.model is None:
            return

        base = cv2.resize(test_image, (400,400))
        candidates = []
        saves = []
        for var in PREPROCESS_VARIANTS:
            try:
                if var == "bgr_norm":
                    arr = base.astype('float32')/255.0  # HWC BGR
                elif var == "rgb_norm":
                    arr = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0
                elif var == "rgb_imagenet":
                    tmp = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0
                    mean = np.array([0.485,0.456,0.406])
                    std = np.array([0.229,0.224,0.225])
                    arr = (tmp - mean) / std
                elif var == "gray_norm":
                    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype('float32')/255.0
                    arr = np.expand_dims(g, axis=-1)
                elif var == "bgr_255":
                    arr = base.astype('float32')  # 0-255
                else:
                    continue
                batch = np.expand_dims(arr, 0)
                preds = None
                try:
                    preds = self.model.predict(batch, verbose=0)
                except Exception as e:
                    print(f"detect_preprocess: model.predict failed for {var} ->", e)
                    preds = None
                if preds is None:
                    continue
                prob = np.array(preds[0]).ravel()
                maxp = float(np.max(prob))
                idx = int(np.argmax(prob))
                candidates.append((var, maxp, idx, prob))
                # save debug image representation
                fn = f"debug_input_{var}.jpg"
                try:
                    # convert arr to savable uint8 RGB for human inspection
                    sav = arr.copy()
                    if sav.ndim == 3 and sav.shape[-1] == 1:
                        sav2 = (np.clip(sav.squeeze(), 0, 1) * 255).astype('uint8')
                        cv2.imwrite(fn, sav2)
                    else:
                        if sav.dtype.kind == 'f':
                            sav2 = np.clip(sav, 0, 1) * 255
                            if sav2.shape[-1] == 3:
                                # we saved as RGB for rgb variants; convert to BGR for cv2
                                try:
                                    bgr = cv2.cvtColor(sav2.astype('uint8'), cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(fn, bgr)
                                except Exception:
                                    cv2.imwrite(fn, sav2.astype('uint8'))
                            else:
                                cv2.imwrite(fn, sav2.astype('uint8'))
                        else:
                            cv2.imwrite(fn, sav.astype('uint8'))
                    saves.append(fn)
                except Exception:
                    pass
            except Exception as e:
                print("detect_preprocess variant error:", e)
        if not candidates:
            print("detect_preprocess: no successful variant predictions")
            return

        # choose variant with highest max prob
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        var_best, maxp, idx_best, prob_best = best
        mapped = LABELS[idx_best] if idx_best < len(LABELS) else str(idx_best)
        self.selected_preprocess = var_best
        print("=== Preprocess auto-selection ===")
        print("Candidates (var, max_prob, top_idx):")
        for c in candidates:
            print(c[0], f"{c[1]:.4f}", c[2], "->", (LABELS[c[2]] if c[2] < len(LABELS) else c[2]))
        print("Selected preprocess:", var_best, "with prob", maxp, "mapped to", mapped)
        # after selection, run normal predict flow using selected variant
        self.predict(test_image)

    def predict(self, test_image):
        """
        Single predict using selected_preprocess if available,
        otherwise default to rgb_norm.
        Puts debug outputs and stable-append logic.
        """
        if self.pts is None or len(self.pts) < 21:
            return
        if self.model is None:
            return

        base = cv2.resize(test_image, (400,400))
        prep = self.selected_preprocess or "rgb_norm"

        try:
            if prep == "bgr_norm":
                arr = base.astype('float32')/255.0
            elif prep == "rgb_norm":
                arr = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0
            elif prep == "rgb_imagenet":
                tmp = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0
                mean = np.array([0.485,0.456,0.406])
                std = np.array([0.229,0.224,0.225])
                arr = (tmp - mean) / std
            elif prep == "gray_norm":
                g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype('float32')/255.0
                arr = np.expand_dims(g, axis=-1)
            elif prep == "bgr_255":
                arr = base.astype('float32')
            else:
                # fallback
                arr = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0
        except Exception as e:
            print("Predict preprocess error:", e)
            arr = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype('float32')/255.0

        batch = np.expand_dims(arr, 0)
        if DEBUG_SAVE_INPUT:
            try:
                dbgname = f"debug_input_selected_{prep}.jpg"
                if arr.ndim==3 and arr.shape[-1]==3:
                    # arr is RGB possibly normalized floats -> convert to BGR uint8
                    try:
                        tmp = (np.clip(arr, 0, 1) * 255).astype('uint8')
                        cv2.imwrite(dbgname, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
                    except Exception:
                        pass
                elif arr.ndim==3 and arr.shape[-1]==1:
                    try:
                        tmp = (np.clip(arr.squeeze(), 0, 1) * 255).astype('uint8')
                        cv2.imwrite(dbgname, tmp)
                    except Exception:
                        pass
            except Exception:
                pass

        # predict
        try:
            preds = self.model.predict(batch, verbose=0)
        except Exception as e:
            print("Model predict raised:", e)
            return
        try:
            prob = np.array(preds[0], dtype='float32').ravel()
        except Exception:
            prob = np.array(preds).ravel()

        # debug
        print("DEBUG predict (prep={}): prob shape {}".format(prep, prob.shape))
        topk = min(5, prob.size)
        idx_sorted = np.argsort(prob)[::-1][:topk]
        print("DEBUG top indices:", idx_sorted)
        print("DEBUG top probs:", prob[idx_sorted])

        ch1 = int(idx_sorted[0])
        mapped = LABELS[ch1] if ch1 < len(LABELS) else str(ch1)
        final_label = mapped

        # stable-append
        if final_label == self.prev_stable_label:
            self.stable_counter += 1
        else:
            self.stable_counter = 1
            self.prev_stable_label = final_label

        if self.stable_counter >= self.STABLE_FRAMES:
            if len(final_label) == 1 and final_label.isalpha():
                if final_label != self.last_appended_label:
                    if self.str.strip() == "":
                        self.str = final_label
                    else:
                        self.str = (self.str + final_label).strip()
                    self.last_appended_label = final_label
                    print("Appended:", final_label, "->", self.str)
            self.stable_counter = 0

        # update GUI state
        self.prev_char = ch1
        self.current_symbol = str(mapped)
        self.count += 1
        self.ten_prev_char[self.count % 10] = mapped

        # suggestions
        if ENCHANT_AVAILABLE and len(self.str.strip()) != 0:
            st = self.str.rfind(" "); word = self.str[st+1:]
            self.word = word
            if len(word.strip()) != 0:
                try:
                    suggestions = ddd.suggest(word)
                    self.word1 = suggestions[0] if len(suggestions)>0 else " "
                    self.word2 = suggestions[1] if len(suggestions)>1 else " "
                    self.word3 = suggestions[2] if len(suggestions)>2 else " "
                    self.word4 = suggestions[3] if len(suggestions)>3 else " "
                except Exception:
                    self.word1 = self.word2 = self.word3 = self.word4 = " "

    # suggestion actions
    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        if idx_word != -1:
            self.str = self.str[:idx_word] + self.word1.upper()
    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        if idx_word != -1:
            self.str = self.str[:idx_word] + self.word2.upper()
    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        if idx_word != -1:
            self.str = self.str[:idx_word] + self.word3.upper()
    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        if idx_word != -1:
            self.str = self.str[:idx_word] + self.word4.upper()

    # TTS thread
    def _speak_thread(self, text):
        try:
            # win32 SAPI
            if self.tts_backend == "win32" and wincl is not None:
                try:
                    s = wincl.Dispatch("SAPI.SpVoice")
                    s.Speak(text)
                    return
                except Exception:
                    print("win32 SAPI speak failed, falling back:", traceback.format_exc())

            # pyttsx3
            if self.tts_backend == "pyttsx3":
                try:
                    # lazy init
                    if self.speak_engine is None:
                        self.speak_engine = pyttsx3.init()
                        self.speak_engine.setProperty("rate", 140)
                    self.speak_engine.say(text)
                    self.speak_engine.runAndWait()
                    return
                except Exception:
                    print("pyttsx3 failed:", traceback.format_exc())

            # gTTS fallback
            if GTTS_AVAILABLE and PLAYSOUND_AVAILABLE:
                try:
                    tts = gTTS(text=text, lang='en')
                    fd, fname = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    tts.save(fname)
                    playsound(fname)
                    try: os.remove(fname)
                    except Exception: pass
                    return
                except Exception:
                    print("gTTS/playsound failed:", traceback.format_exc())

            print("No TTS backend worked.")
        finally:
            self.speaking = False
            try:
                self.root.after(0, lambda: self.speak_btn.config(state=tk.NORMAL, text="🔊 SPEAK"))
            except Exception:
                pass

    def speak_fun(self):
        text = (self.str or "").strip()
        if not text:
            # ask to speak example
            try:
                if messagebox.askyesno("Nothing to speak", "Sentence empty. Speak a test sentence 'Yes please change the code accordingly'?"):
                    text = "Yes please change the code accordingly"
                else:
                    return
            except Exception:
                return

        if self.speaking:
            print("Already speaking")
            return

        backend = choose_tts_backend()
        if backend is None:
            messagebox.showwarning("TTS unavailable", "No TTS backend available. Install pywin32 or pyttsx3 or (gTTS+playsound).")
            return

        self.speaking = True
        try:
            self.speak_btn.config(state=tk.DISABLED, text="🔈 Speaking...")
        except Exception:
            pass
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()

    def clear_fun(self):
        self.str = ""
        self.current_symbol = "Ready"
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.last_appended_label = None
        self.prev_stable_label = None
        self.stable_counter = 0
        print("✓ Text cleared")

    def destructor(self):
        print("Closing application...")
        print("Last detected:", self.ten_prev_char)
        try: self.root.quit()
        except Exception: pass
        try: self.root.destroy()
        except Exception: pass
        if self.vs:
            try: self.vs.release()
            except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception:
            # many wheels built without GUI support -> ignore
            pass
        # stop pyttsx3 engine if present
        try:
            if self.speak_engine is not None:
                try: self.speak_engine.stop()
                except Exception: pass
        except Exception:
            pass

# ------------------ Run ------------------
if __name__ == "__main__":
    try:
        app = Application()
        app.root.mainloop()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception:
        print("Fatal error:")
        traceback.print_exc()
