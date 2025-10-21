import cv2
import os
import numpy as np
import pickle
import random
import sqlite3
from glob import glob
from threading import Thread
import pyttsx3
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split

def calibrate_skin():
    print("\n>>> Skin Calibration <<<")
    print("1. Place your hand inside the blue rectangle.")
    print("2. Ensure good, consistent lighting.")
    print("3. Press 'c' to capture skin sample.")
    print("4. Press 'q' to quit.\n")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Cannot open camera. Try a different camera index.")
        return

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("❌ Failed to read frame from camera.")
            break

        h, w = frame.shape[:2]
        roi_size = 200
        left = max(0, w - roi_size - 50)
        right = min(w, left + roi_size)
        top = max(0, (h - roi_size) // 2)
        bottom = min(h, top + roi_size)

        roi = frame[top:bottom, left:right]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Put hand here", (left - 80, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Skin Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if roi.size == 0:
                print("❌ ROI is empty. Try again.")
                continue
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            with open("hist", "wb") as f:
                pickle.dump(roi_hist, f)
            print("✅ Skin histogram saved as 'hist'")
            break
        elif key == ord('q'):
            print("⚠️ Calibration canceled.")
            break

    cam.release()
    cv2.destroyAllWindows()

def flip_images():
    gest_folder = "gestures"
    if not os.path.exists(gest_folder):
        print("❌ gestures folder not found!")
        return
    for g_id in os.listdir(gest_folder):
        g_path = os.path.join(gest_folder, g_id)
        if not os.path.isdir(g_path):
            continue
        print(f"Flipping images in {g_id}...")
        for i in range(1, 1201):
            path = os.path.join(g_path, f"{i}.jpg")
            new_path = os.path.join(g_path, f"{i + 1200}.jpg")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, 0)
            if img is None:
                continue
            flipped = cv2.flip(img, 1)
            cv2.imwrite(new_path, flipped)
    print("✅ Flipping complete.")

def get_hand_hist():
    if not os.path.exists("hist"):
        print("❌ Hand histogram 'hist' not found! Run skin calibration first.")
        return None
    with open("hist", "rb") as f:
        return pickle.load(f)

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        conn.execute("CREATE TABLE gesture (g_id INTEGER PRIMARY KEY, g_name TEXT NOT NULL)")
        conn.commit()
        print("✅ Database created.")

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    try:
        conn.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input(f"⚠️ g_id {g_id} exists. Update? (y/n): ")
        if choice.lower() == 'y':
            conn.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
        else:
            print("Skipped.")
            conn.close()
            return
    conn.commit()
    conn.close()

def store_images(g_id):
    total_pics = 1200
    hist = get_hand_hist()
    if hist is None:
        return
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera not accessible.")
        return
    x, y, w, h = 300, 100, 300, 300
    create_folder(f"gestures/{g_id}")
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, img = cam.read()
        if not ret or img is None:
            break
        img = cv2.flip(img, 1)
        h_img, w_img = img.shape[:2]
        x = min(x, w_img - w - 1)
        y = min(y, h_img - h - 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh_gray = thresh.copy()
        contours, _ = cv2.findContours(thresh_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh_gray[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)
                save_img = cv2.resize(save_img, (50, 50))
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite(f"gestures/{g_id}/{pic_no}.jpg", save_img)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("thresh", thresh_gray)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            if not flag_start_capturing:
                frames = 0
        if flag_start_capturing:
            frames += 1
        if pic_no >= total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()

def collect_data():
    if not os.path.exists("hist"):
        print("⚠️ 'hist' not found. Starting calibration...")
        calibrate_skin()
        if not os.path.exists("hist"):
            print("❌ Calibration failed. Cannot collect data.")
            return
    init_create_folder_database()
    g_id = input("Enter gesture ID (e.g., 0, 1, 2): ").strip()
    g_name = input("Enter gesture name: ").strip()
    if not g_id.isdigit():
        print("❌ Invalid ID. Must be an integer.")
        return
    store_in_db(g_id, g_name)
    store_images(g_id)

def prepare_data():
    if not os.path.exists("gestures") or not os.listdir("gestures"):
        print("❌ No gesture data found. Collect data first.")
        return

    print("🔄 Preparing training data...")
    gestures = sorted([d for d in os.listdir("gestures") if os.path.isdir(f"gestures/{d}")], key=int)
    images = []
    labels = []

    for g_id in gestures:
        g_path = f"gestures/{g_id}"
        for img_name in os.listdir(g_path):
            if not img_name.endswith(".jpg"):
                continue
            img_path = os.path.join(g_path, img_name)
            img = cv2.imread(img_path, 0)
            if img is not None:
                img = cv2.resize(img, (50, 50))
                images.append(img)
                labels.append(int(g_id))

    if len(images) == 0:
        print("❌ No valid images found.")
        return

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)

    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    with open("train_images", "wb") as f:
        pickle.dump(x_train, f)
    with open("train_labels", "wb") as f:
        pickle.dump(y_train, f)
    with open("val_images", "wb") as f:
        pickle.dump(x_val, f)
    with open("val_labels", "wb") as f:
        pickle.dump(y_val, f)

    print(f"✅ Saved {len(x_train)} training and {len(x_val)} validation samples.")

def train_model():
    if not (os.path.exists("train_images") and os.path.exists("train_labels")):
        print("❌ Training data not prepared. Run 'Prepare training data' first.")
        return

    def get_num_of_classes():
        return len(glob('gestures/*'))

    num_of_classes = get_num_of_classes()
    if num_of_classes < 2:
        print("❌ Need at least 2 gesture classes to train.")
        return

    image_x, image_y = 50, 50

    def cnn_model():
        model = Sequential()
        model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
        model.add(Conv2D(64, (5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_of_classes, activation='softmax'))
        sgd = optimizers.SGD(learning_rate=1e-2)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        filepath = "cnn_model_keras2.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        return model, [checkpoint]

    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    train_labels = np_utils.to_categorical(train_labels, num_of_classes)
    val_labels = np_utils.to_categorical(val_labels, num_of_classes)

    model, callbacks = cnn_model()
    model.summary()
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels),
              epochs=15, batch_size=500, callbacks=callbacks)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"✅ CNN Accuracy: {scores[1]*100:.2f}%")
    K.clear_session()

def recognize_gesture():
    if not os.path.exists("cnn_model_keras2.h5"):
        print("❌ Model not found! Train first.")
        return

    hist = get_hand_hist()
    if hist is None:
        return

    model = load_model("cnn_model_keras2.h5")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    image_x, image_y = 50, 50

    def keras_process_image(img):
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        return np.reshape(img, (1, image_x, image_y, 1))

    def keras_predict(model, image):
        processed = keras_process_image(image)
        pred_probab = model.predict(processed, verbose=0)[0]
        pred_class = int(np.argmax(pred_probab))
        return float(np.max(pred_probab)), pred_class

    def get_pred_text_from_db(pred_class):
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.execute("SELECT g_name FROM gesture WHERE g_id = ?", (pred_class,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "Unknown"

    def get_operator(pred_text):
        op_map = {'1': '+', '2': '-', '3': '*', '4': '/', '5': '%',
                  '6': '**', '7': '>>', '8': '<<', '9': '&', '0': '|'}
        return op_map.get(pred_text, "")

    def get_img_contour_thresh(img):
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh_gray = thresh.copy()
        x, y, w, h = 300, 100, 300, 300
        h_img, w_img = thresh_gray.shape
        x = min(x, w_img - w - 1)
        y = min(y, h_img - h - 1)
        crop = thresh_gray[y:y+h, x:x+w]
        contours, _ = cv2.findContours(crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        return img, contours, thresh_gray, (x, y, w, h)

    def say_text(text):
        if not getattr(engine, '_inLoop', False):
            engine.say(text)
            engine.runAndWait()

    def calculator_mode(cam):
        flag = {"first": False, "operator": False, "second": False, "clear": False}
        count_same_frames = 0
        first, second, operator, pred_text, calc_text = "", "", "", "", ""
        info = "Enter first number"
        Thread(target=say_text, args=(info,)).start()

        while True:
            ret, img = cam.read()
            if not ret or img is None: break
            img = cv2.resize(img, (640, 480))
            img, contours, thresh, (x, y, w, h) = get_img_contour_thresh(img)
            old_pred_text = pred_text
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if save_img.size == 0:
                        continue
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)
                    pred_probab, pred_class = keras_predict(model, save_img)
                    pred_text = get_pred_text_from_db(pred_class) if pred_probab > 0.7 else ""

                    if old_pred_text == pred_text:
                        count_same_frames += 1
                    else:
                        count_same_frames = 0

                    if pred_text == "C" and count_same_frames > 5:
                        first, second, operator, pred_text, calc_text = '', '', '', '', ''
                        flag = {k: False for k in flag}
                        info = "Enter first number"
                        Thread(target=say_text, args=(info,)).start()
                        count_same_frames = 0

                    elif pred_text == "Best of Luck " and count_same_frames > 15:
                        if flag["clear"]:
                            first, second, operator, pred_text, calc_text = '', '', '', '', ''
                            flag = {k: False for k in flag}
                            info = "Enter first number"
                            Thread(target=say_text, args=(info,)).start()
                        elif second != '':
                            try:
                                calc_text += " = " + str(eval(calc_text))
                            except:
                                calc_text = "Invalid"
                            Thread(target=say_text, args=(calc_text.replace('**','^'),)).start()
                            flag["clear"] = True
                        elif first != '':
                            flag["first"] = True
                            info = "Enter operator"
                            Thread(target=say_text, args=(info,)).start()
                        count_same_frames = 0

                    elif pred_text != "Best of Luck " and pred_text.isnumeric():
                        if not flag["first"] and count_same_frames > 15:
                            first += pred_text
                            calc_text += pred_text
                            Thread(target=say_text, args=(pred_text,)).start()
                            count_same_frames = 0
                        elif not flag["operator"] and count_same_frames > 15:
                            op = get_operator(pred_text)
                            if op:
                                calc_text += op
                                flag["operator"] = True
                                info = "Enter second number"
                                Thread(target=say_text, args=(info,)).start()
                                count_same_frames = 0
                        elif not flag["second"] and count_same_frames > 15:
                            second += pred_text
                            calc_text += pred_text
                            Thread(target=say_text, args=(pred_text,)).start()
                            count_same_frames = 0

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(blackboard, "Predicted: " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
            cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Calculator", res)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return 0
            elif key == ord('t'): return 1

    def text_mode(cam):
        word = ""
        pred_text = ""
        count_same_frame = 0
        while True:
            ret, img = cam.read()
            if not ret or img is None: break
            img = cv2.resize(img, (640, 480))
            img, contours, thresh, (x, y, w, h) = get_img_contour_thresh(img)
            old_text = pred_text
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if save_img.size == 0:
                        continue
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)
                    pred_probab, pred_class = keras_predict(model, save_img)
                    pred_text = get_pred_text_from_db(pred_class) if pred_probab > 0.7 else ""
                    if old_text == pred_text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0
                    if count_same_frame > 20 and len(pred_text) == 1:
                        word += pred_text
                        Thread(target=say_text, args=(pred_text,)).start()
                        count_same_frame = 0
                elif cv2.contourArea(contour) < 1000:
                    if word:
                        Thread(target=say_text, args=(word,)).start()
                        word = ""
            else:
                if word:
                    Thread(target=say_text, args=(word,)).start()
                    word = ""

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(blackboard, "Predicted: " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Text Mode", res)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return 0
            elif key == ord('c'): return 2

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera not accessible for recognition.")
        return

    mode = 1
    while True:
        if mode == 1:
            mode = text_mode(cam)
        elif mode == 2:
            mode = calculator_mode(cam)
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n=== Hand Gesture Recognition System ===")
        print("0. Calibrate skin color (REQUIRED first step)")
        print("1. Collect new gesture data")
        print("2. Augment data (flip images)")
        print("3. Prepare training data")
        print("4. Train model")
        print("5. Run real-time recognition")
        print("6. Exit")
        choice = input("Choose an option: ").strip()

        if choice == '0':
            calibrate_skin()
        elif choice == '1':
            collect_data()
        elif choice == '2':
            flip_images()
        elif choice == '3':
            prepare_data()
        elif choice == '4':
            train_model()
        elif choice == '5':
            recognize_gesture()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
