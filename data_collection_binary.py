import cv2
import numpy as np
import os
import traceback
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = r'C:\Users\devansh raval\PycharmProjects\pythonProject\cnn9.h5'  # update if needed
FALLBACK_MODEL = os.path.join(os.getcwd(), 'cnn9.h5')
TEST_GRAY_DIR = r'D:\test_data_2.0\Gray_imgs'
TEST_GRAY_WITH_DRAW_DIR = r'D:\test_data_2.0\Gray_imgs_with_drawing'
CAPTURE_DEVICE = 0
IMG_SIZE = 400
OFFSET = 30
MAX_SAVE_PER_TOGGLE = 50
# ----------------------------------------

# create base dirs if missing
for d in (TEST_GRAY_DIR, TEST_GRAY_WITH_DRAW_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        print(f"Warning: couldn't create {d}: {e}")

# load model (try primary then fallback), continue without model if not found
model = None
for candidate in (MODEL_PATH, FALLBACK_MODEL):
    if os.path.isfile(candidate):
        try:
            model = load_model(candidate)
            print(f"Model loaded from: {candidate}")
            break
        except Exception as e:
            print(f"Failed loading model from {candidate}: {e}")
if model is None:
    print("Model not loaded. Continuing without model (model=None).")

# initialize camera and hand detector
cap = cv2.VideoCapture(CAPTURE_DEVICE)
hd = HandDetector(maxHands=1, detectionCon=0.6)  # reuse this

# letter / filename state
p_dir = "A"   # folder (capital)
c_dir = "a"   # prefix (lowercase)
def ensure_letter_dirs(letter):
    g = os.path.join(TEST_GRAY_DIR, letter)
    gd = os.path.join(TEST_GRAY_WITH_DRAW_DIR, letter)
    os.makedirs(g, exist_ok=True)
    os.makedirs(gd, exist_ok=True)
    return g, gd

ensure_letter_dirs(p_dir)
try:
    count = len(os.listdir(os.path.join(TEST_GRAY_DIR, p_dir)))
except Exception:
    count = 0

flag_record = False
suv = 0
step = 1

# preallocated canvases
BLANK_WHITE = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
BLANK_MIDGRAY = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 148

def safe_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

# SIMPLE fixed preprocess: force (1,400,400,1)
def preprocess_for_model(gray_img):
    img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # channel
    img = np.expand_dims(img, axis=0)   # batch
    return img

def safe_put_text(img, text, org=(10,30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

print("Starting. Keys: 'a' toggle record, 'n' next letter, ESC quit.")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break
        frame = cv2.flip(frame, 1)

        # --- handle different return types from findHands() ---
        find_result = hd.findHands(frame, draw=False, flipType=True)
        # cvzone versions: sometimes return (img, hands) or just hands list
        if isinstance(find_result, tuple) or isinstance(find_result, list):
            # if it's length 2 and second element is list -> (img, hands)
            if len(find_result) == 2 and isinstance(find_result[1], list):
                # find_result[0] may be image-with-draw; we want raw frame for processing
                hands = find_result[1]
            else:
                # sometimes it directly returns list of hands
                # but if it returned img only (rare), ensure hands becomes []
                if all(isinstance(x, dict) for x in find_result):
                    hands = find_result
                else:
                    hands = []
        else:
            # unknown type -> fallback to empty
            hands = []

        img_final_binary = None
        img_final_gray_no_draw = None
        img_final_gray_with_draw = None

        if hands:
            # ensure first hand is a dict-like mapping with keys
            hand = hands[0]
            # Some versions of cvzone return dict, some return lists with keys as attributes.
            # Safely extract bbox and lmList with .get fallback
            bbox = None
            lmList = None
            try:
                # primary expected
                bbox = hand.get('bbox') if hasattr(hand, 'get') else None
            except Exception:
                bbox = None
            # fallback: sometimes hand is a list like [x,y,w,h,...] or has attributes
            if bbox is None:
                # try attribute access
                if isinstance(hand, dict) and 'bbox' in hand:
                    bbox = hand['bbox']
                elif hasattr(hand, 'bbox'):
                    try:
                        bbox = hand.bbox
                    except Exception:
                        bbox = None
                else:
                    # try if hand itself is a list/tuple of numbers
                    if isinstance(hand, (list, tuple)) and len(hand) >= 4 and all(isinstance(v, (int, float)) for v in hand[:4]):
                        bbox = (int(hand[0]), int(hand[1]), int(hand[2]), int(hand[3]))
            # landmarks
            try:
                lmList = hand.get('lmList') if hasattr(hand, 'get') else None
            except Exception:
                lmList = None
            if lmList is None:
                if isinstance(hand, dict) and 'lmList' in hand:
                    lmList = hand['lmList']
                elif hasattr(hand, 'lmList'):
                    try:
                        lmList = hand.lmList
                    except Exception:
                        lmList = None

            if bbox is None:
                # give up this hand safely
                hands = []
            else:
                # bbox should be iterable of 4 numbers
                try:
                    x, y, w, h = map(int, bbox)
                except Exception:
                    # if bbox is nested like [x,y,w,h]
                    try:
                        x = int(bbox[0]); y = int(bbox[1]); w = int(bbox[2]); h = int(bbox[3])
                    except Exception:
                        x = y = w = h = 0

                crop = safe_crop(frame, x - OFFSET, y - OFFSET, x + w + OFFSET, y + h + OFFSET)
                if crop is not None and crop.size != 0:
                    # grayscale without drawing
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (3, 3), 1)

                    # binary
                    blur2 = cv2.GaussianBlur(gray, (5, 5), 2)
                    th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                    _, test_image_binary = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    # center into 400x400 canvases
                    hh, ww = blur.shape[:2]
                    top = (IMG_SIZE - hh) // 2
                    left = (IMG_SIZE - ww) // 2
                    img_final_gray_no_draw = BLANK_MIDGRAY.copy()
                    img_final_gray_no_draw[top:top + hh, left:left + ww] = blur

                    hh2, ww2 = test_image_binary.shape[:2]
                    top2 = (IMG_SIZE - hh2) // 2
                    left2 = (IMG_SIZE - ww2) // 2
                    img_final_binary = BLANK_WHITE.copy()
                    img_final_binary[top2:top2 + hh2, left2:left2 + ww2] = test_image_binary

                    # draw skeleton using landmarks if available
                    white_rgb = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
                    pts = lmList
                    if pts:
                        osx, osy = left, top
                        try:
                            for t in range(0, 4):
                                cv2.line(white_rgb, (pts[t][0] + osx, pts[t][1] + osy), (pts[t+1][0] + osx, pts[t+1][1] + osy), (0,255,0), 3)
                            for t in range(5,8):
                                cv2.line(white_rgb, (pts[t][0] + osx, pts[t][1] + osy), (pts[t+1][0] + osx, pts[t+1][1] + osy), (0,255,0), 3)
                            for t in range(9,12):
                                cv2.line(white_rgb, (pts[t][0] + osx, pts[t][1] + osy), (pts[t+1][0] + osx, pts[t+1][1] + osy), (0,255,0), 3)
                            for t in range(13,16):
                                cv2.line(white_rgb, (pts[t][0] + osx, pts[t][1] + osy), (pts[t+1][0] + osx, pts[t+1][1] + osy), (0,255,0), 3)
                            for t in range(17,20):
                                cv2.line(white_rgb, (pts[t][0] + osx, pts[t][1] + osy), (pts[t+1][0] + osx, pts[t+1][1] + osy), (0,255,0), 3)

                            cv2.line(white_rgb, (pts[5][0] + osx, pts[5][1] + osy), (pts[9][0] + osx, pts[9][1] + osy), (0,255,0), 3)
                            cv2.line(white_rgb, (pts[9][0] + osx, pts[9][1] + osy), (pts[13][0] + osx, pts[13][1] + osy), (0,255,0), 3)
                            cv2.line(white_rgb, (pts[13][0] + osx, pts[13][1] + osy), (pts[17][0] + osx, pts[17][1] + osy), (0,255,0), 3)
                            cv2.line(white_rgb, (pts[0][0] + osx, pts[0][1] + osy), (pts[5][0] + osx, pts[5][1] + osy), (0,255,0), 3)
                            cv2.line(white_rgb, (pts[0][0] + osx, pts[0][1] + osy), (pts[17][0] + osx, pts[17][1] + osy), (0,255,0), 3)

                            for i in range(len(pts)):
                                cv2.circle(white_rgb, (pts[i][0] + osx, pts[i][1] + osy), 2, (0,0,255), 1)
                        except Exception:
                            pass

                        white_gray = cv2.cvtColor(white_rgb, cv2.COLOR_BGR2GRAY)
                        img_final_gray_with_draw = BLANK_MIDGRAY.copy()
                        hwg, wwg = white_gray.shape[:2]
                        if top + hwg <= IMG_SIZE and left + wwg <= IMG_SIZE:
                            img_final_gray_with_draw[top:top + hwg, left:left + wwg] = white_gray
                        else:
                            # center fallback
                            cy = (IMG_SIZE - hwg) // 2
                            cx = (IMG_SIZE - wwg) // 2
                            img_final_gray_with_draw[cy:cy + hwg, cx:cx + wwg] = white_gray

                    # show previews
                    if img_final_binary is not None:
                        cv2.imshow("binary", img_final_binary)
                    if img_final_gray_no_draw is not None:
                        cv2.imshow("gray_no_draw", img_final_gray_no_draw)
                    if img_final_gray_with_draw is not None:
                        cv2.imshow("gray_with_draw", img_final_gray_with_draw)

                    # model prediction (fixed preprocess)
                    if model is not None and img_final_binary is not None:
                        inp = preprocess_for_model(img_final_binary)
                        try:
                            probs = model.predict(inp)[0]
                            ch1 = int(np.argmax(probs))
                            predicted_char = chr(ch1 + 65) if 0 <= ch1 < 26 else str(ch1)
                            safe_put_text(frame, f"Pred: {predicted_char}", (x - OFFSET, y - OFFSET - 10))
                        except Exception as e:
                            print("Prediction error:", e)

        # show main frame
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('n'):
            # next letter
            p_dir = chr(ord(p_dir) + 1)
            c_dir = chr(ord(c_dir) + 1)
            if ord(p_dir) == ord('Z') + 1:
                p_dir = "A"
                c_dir = "a"
            ensure_letter_dirs(p_dir)
            try:
                count = len(os.listdir(os.path.join(TEST_GRAY_DIR, p_dir)))
            except Exception:
                count = 0
            flag_record = False
        if key == ord('a'):
            flag_record = not flag_record
            if flag_record:
                suv = 0
                print("Recording ON")
            else:
                print("Recording OFF")

        # recording logic (every 2 steps, max 50)
        if flag_record:
            if suv >= MAX_SAVE_PER_TOGGLE:
                flag_record = False
                print("Auto stop after saves.")
            else:
                if step % 2 == 0:
                    target_dir_gray = os.path.join(TEST_GRAY_DIR, p_dir)
                    target_dir_draw = os.path.join(TEST_GRAY_WITH_DRAW_DIR, p_dir)
                    os.makedirs(target_dir_gray, exist_ok=True)
                    os.makedirs(target_dir_draw, exist_ok=True)

                    if img_final_gray_no_draw is not None:
                        fname = os.path.join(target_dir_gray, f"{c_dir}{count}.jpg")
                        cv2.imwrite(fname, img_final_gray_no_draw)
                    if img_final_gray_with_draw is not None:
                        fname2 = os.path.join(target_dir_draw, f"{c_dir}{count}.jpg")
                        cv2.imwrite(fname2, img_final_gray_with_draw)

                    count += 1
                    suv += 1
                step += 1

    except Exception:
        print("EXC:", traceback.format_exc())
        # do not crash; continue loop

cap.release()
cv2.destroyAllWindows()
