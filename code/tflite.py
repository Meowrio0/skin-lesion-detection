import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.colab import drive

# 掛載 Google Drive
drive.mount('/content/drive')

# ---------- 建立儲存目錄 ----------
TFLITE_DIR = '/content/drive/MyDrive/TFLite'
os.makedirs(TFLITE_DIR, exist_ok=True)

# ---------- 載入 segmentation 模型（.keras） ----------
def combined_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_loss_val = 1.0 - (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    bce_loss_val = tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(y_true, y_pred))
    return 0.7 * dice_loss_val + 0.3 * bce_loss_val

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3]) - intersection
    return tf.keras.backend.mean((intersection + 1.0) / (union + 1.0))

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1,2,3])
    return tf.keras.backend.mean((2. * intersection + 1.0) / (
        tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3]) + 1.0))

print("\n🔄 載入 Segmentation 模型中...")
seg_model_path = '/content/drive/MyDrive/skin_segmentation_results/unet_skin_segmentation_best.keras'
seg_model = load_model(seg_model_path, custom_objects={
    'combined_loss': combined_loss,
    'iou': iou,
    'dice_coefficient': dice_coefficient
})
print("✅ Segmentation 模型載入成功！")

# ---------- 轉換 segmentation 為 TFLite ----------
print("🔁 轉換 Segmentation 模型為 TFLite...")
seg_converter = tf.lite.TFLiteConverter.from_keras_model(seg_model)
seg_converter.optimizations = [tf.lite.Optimize.DEFAULT]
seg_tflite = seg_converter.convert()

seg_output_path = os.path.join(TFLITE_DIR, 'unet_skin_segmentation.tflite')
with open(seg_output_path, 'wb') as f:
    f.write(seg_tflite)
print(f"✅ Segmentation TFLite 儲存至：{seg_output_path}")

# ---------- 載入 classification 模型（.h5） ----------
print("\n🔄 載入 Classification 模型中...")
class_model_path = '/content/drive/MyDrive/skin_lesion_classification_result/student_model_distilled.h5'
class_model = load_model(class_model_path)
print("✅ Classification 模型載入成功！")

# ---------- 轉換 classification 為 TFLite ----------
print("🔁 轉換 Classification 模型為 TFLite...")
class_converter = tf.lite.TFLiteConverter.from_keras_model(class_model)
class_converter.optimizations = [tf.lite.Optimize.DEFAULT]
class_tflite = class_converter.convert()

class_output_path = os.path.join(TFLITE_DIR, 'student_model_distilled.tflite')
with open(class_output_path, 'wb') as f:
    f.write(class_tflite)
print(f"✅ Classification TFLite 儲存至：{class_output_path}")