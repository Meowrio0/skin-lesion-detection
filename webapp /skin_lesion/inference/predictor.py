import tensorflow as tf
import numpy as np
import cv2
import os

# 模型路徑
SEG_MODEL = os.path.join(os.path.dirname(__file__), 'models/unet_skin_segmentation.tflite')
CLASS_MODEL = os.path.join(os.path.dirname(__file__), 'models/student_model_distilled.tflite')

# 類別標籤
class_labels = ['basal cell carcinoma', 'melanoma', 'nevus', 'squamous cell carcinoma']
SEG_IMG_SIZE = 256
CLASS_IMG_SIZE = 224

# 初始化 TFLite interpreter（只載入一次）
seg_interpreter = tf.lite.Interpreter(model_path=SEG_MODEL)
seg_interpreter.allocate_tensors()
seg_input = seg_interpreter.get_input_details()
seg_output = seg_interpreter.get_output_details()

class_interpreter = tf.lite.Interpreter(model_path=CLASS_MODEL)
class_interpreter.allocate_tensors()
class_input = class_interpreter.get_input_details()
class_output = class_interpreter.get_output_details()

# ✅ 支援直接處理 memory image (numpy array)，不需從硬碟讀檔
def run_prediction(rgb_img):
    # 確保輸入為 RGB 格式的 numpy array
    resized = cv2.resize(rgb_img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 語意分割推論
    seg_interpreter.set_tensor(seg_input[0]['index'], input_tensor)
    seg_interpreter.invoke()
    mask = seg_interpreter.get_tensor(seg_output[0]['index'])[0, :, :, 0]
    binary_mask = (mask > 0.5).astype(np.uint8)

    # 擷取病灶區域
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = resized[y:y+h, x:x+w]
    else:
        cropped = resized

    # 分類推論
    lesion_img = cv2.resize(cropped, (CLASS_IMG_SIZE, CLASS_IMG_SIZE)).astype(np.float32) / 255.0
    lesion_tensor = np.expand_dims(lesion_img, axis=0)

    class_interpreter.set_tensor(class_input[0]['index'], lesion_tensor)
    class_interpreter.invoke()
    pred = class_interpreter.get_tensor(class_output[0]['index'])[0]

    class_idx = np.argmax(pred)
    confidence = float(pred[class_idx])

    return class_labels[class_idx], confidence