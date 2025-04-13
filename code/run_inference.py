# run_inference.py

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 設定模型與圖片路徑
SEG_MODEL = "models/unet_skin_segmentation.tflite"
CLASS_MODEL = "models/student_model_distilled.tflite"
IMG_PATH = "images/test1.jpg"  # 你可以換成任意圖片檔名

# 圖片大小
SEG_IMG_SIZE = 256
CLASS_IMG_SIZE = 224

# 類別標籤
class_labels = ['basal cell carcinoma', 'melanoma', 'nevus', 'squamous cell carcinoma']

# 載入 segmentation 模型
seg_interpreter = tf.lite.Interpreter(model_path=SEG_MODEL)
seg_interpreter.allocate_tensors()
seg_input = seg_interpreter.get_input_details()
seg_output = seg_interpreter.get_output_details()

# 載入 classification 模型
class_interpreter = tf.lite.Interpreter(model_path=CLASS_MODEL)
class_interpreter.allocate_tensors()
class_input = class_interpreter.get_input_details()
class_output = class_interpreter.get_output_details()

# 載入圖片
img = cv2.imread(IMG_PATH)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb_img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
input_tensor = resized.astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# segmentation 推論
seg_interpreter.set_tensor(seg_input[0]['index'], input_tensor)
seg_interpreter.invoke()
mask = seg_interpreter.get_tensor(seg_output[0]['index'])[0, :, :, 0]
binary_mask = (mask > 0.5).astype(np.uint8)

# 找出病灶區域
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = resized[y:y+h, x:x+w]
else:
    cropped = resized

# classification 推論
lesion_img = cv2.resize(cropped, (CLASS_IMG_SIZE, CLASS_IMG_SIZE)).astype(np.float32) / 255.0
lesion_tensor = np.expand_dims(lesion_img, axis=0)

class_interpreter.set_tensor(class_input[0]['index'], lesion_tensor)
class_interpreter.invoke()
pred = class_interpreter.get_tensor(class_output[0]['index'])[0]
class_idx = np.argmax(pred)
confidence = pred[class_idx]
predicted_class = class_labels[class_idx]

# 顯示結果
print(f"\n🖼️ 處理圖片：{os.path.basename(IMG_PATH)}")
print(f"➡️ 預測結果：{predicted_class}（信心值：{confidence:.2f}）")