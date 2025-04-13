import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from google.colab import drive

# 掛載Google Drive
drive.mount('/content/drive')

# 設定隨機種子，確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 影像尺寸
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# 設定數據增強函數 - 增加穩健性
def data_augmentation(image, mask):
    # 創建一個隨機數生成器以確保圖像和遮罩使用相同的變換
    rng = np.random.RandomState(np.random.randint(0, 2**31 - 1))

    # 50%機率水平翻轉
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # 20%機率垂直翻轉
    if rng.random() < 0.2:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # 30%機率進行小角度旋轉 (-15 to 15度)
    if rng.random() < 0.3:
        angle = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_WIDTH/2, IMG_HEIGHT/2), angle, 1)
        image = cv2.warpAffine(image, M, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.warpAffine(mask.reshape(IMG_HEIGHT, IMG_WIDTH), M, (IMG_WIDTH, IMG_HEIGHT))
        mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
    
    # 20%機率調整亮度
    if rng.random() < 0.2:
        brightness = rng.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
    
    return image, mask

# 改進的數據加載函數 - 包含數據增強
def load_images(image_folder, mask_folder, augment=True, num_augmentations=1):
    image_filenames = sorted(os.listdir(image_folder))
    mask_filenames = sorted(os.listdir(mask_folder))
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        # 讀取並調整大小
        img = cv2.imread(os.path.join(image_folder, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0  # 歸一化
        
        # 讀取遮罩
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = (mask > 0).astype(np.float32)  # 二值化
        mask = mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
        
        # 添加原始圖像和遮罩
        images.append(img)
        masks.append(mask)
        
        # 進行數據增強
        if augment:
            for _ in range(num_augmentations):
                aug_img, aug_mask = data_augmentation(img.copy(), mask.copy())
                images.append(aug_img)
                masks.append(aug_mask)
    
    return np.array(images), np.array(masks)

# 複合損失函數 - 結合Dice損失和BCE
def combined_loss(y_true, y_pred):
    # Dice損失
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss_val = 1.0 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    # 二元交叉熵損失
    bce_loss_val = K.mean(K.binary_crossentropy(y_true, y_pred))
    
    # 結合損失，加權更重視Dice損失
    return 0.7 * dice_loss_val + 0.3 * bce_loss_val

# IoU指標（與原先相同）
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean((intersection + 1.0) / (union + 1.0))

# Dice係數（與原先相同）
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    return K.mean((2. * intersection + 1.0) / (K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) + 1.0))

# 優化的U-Net模型
def create_optimized_unet_model():
    # 添加L2正則化
    reg = l2(1e-5)
    
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # 契約路徑 - 輕量級但有效的濾波器配置
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(input_layer)
    c1 = SpatialDropout2D(0.05)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(p1)
    c2 = SpatialDropout2D(0.05)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(p2)
    c3 = SpatialDropout2D(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # 瓶頸層 - 稍微增加深度
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(p3)
    c4 = SpatialDropout2D(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c4)
    
    # 擴張路徑 - 對應編碼層
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(u5)
    c5 = SpatialDropout2D(0.1)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(u6)
    c6 = SpatialDropout2D(0.05)(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(u7)
    c7 = SpatialDropout2D(0.05)(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(c7)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs=input_layer, outputs=outputs)
    return model

# 設定數據集路徑
base_path = '/content/drive/MyDrive/PH2 Resized Archive（皮膚病變資料集）'
image_folder = os.path.join(base_path, 'trainx')
mask_folder = os.path.join(base_path, 'trainy')

# 加載數據 - 現在包含數據增強
X, Y = load_images(image_folder, mask_folder, augment=True, num_augmentations=2)
print(f"載入數據完成，總共 {len(X)} 張圖像")

# 訓練/測試數據分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 創建並編譯模型 - 使用改進的損失函數和優化器設置
model = create_optimized_unet_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4, decay=1e-6),  # 添加學習率衰減
    loss=combined_loss,
    metrics=[iou, dice_coefficient]
)

# 顯示模型摘要
model.summary()

# 設定保存結果的目錄
save_dir = '/content/drive/MyDrive/skin_segmentation_results'
os.makedirs(save_dir, exist_ok=True)

# 定義回調函數
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

# 儲存最佳模型
checkpoint = ModelCheckpoint(
    os.path.join(save_dir, "unet_skin_segmentation_best.keras"),
    monitor="val_dice_coefficient",
    verbose=1,
    save_best_only=True,
    mode="max"
)

# 調整提前停止策略 - 增加patience
early_stopping = EarlyStopping(
    monitor="val_dice_coefficient",  # 改為監控Dice係數而非損失
    patience=12,                      # 增加patience以允許更長時間的訓練
    verbose=1,
    restore_best_weights=True,
    mode="max"                        # 因為我們要最大化Dice係數
)

# 降低學習率 - 調整策略
reduce_lr = ReduceLROnPlateau(
    monitor="val_dice_coefficient",    # 同樣監控Dice係數
    factor=0.7,                      # 不要一次降低太多
    patience=5,                       # 更早開始調整學習率
    verbose=1,
    min_lr=1e-7,
    mode="max"                         # 最大化Dice係數
)

# 記錄訓練過程
csv_logger = CSVLogger(os.path.join(save_dir, 'training_log.csv'))

# 添加TensorBoard視覺化
tensorboard = TensorBoard(
    log_dir=os.path.join(save_dir, 'logs'),
    histogram_freq=1,
    write_images=True
)

callbacks = [checkpoint, early_stopping, reduce_lr, csv_logger, tensorboard]

# 訓練模型 - 調整批次大小和驗證分割
history = model.fit(
    x_train,
    y_train,
    batch_size=16,                  # 適中的批次大小
    epochs=100,                     # 設置更大的epoch數，讓early stopping決定何時停止
    validation_split=0.2,           # 保持合理的驗證集大小
    callbacks=callbacks,
    shuffle=True                    # 確保每個epoch都打亂訓練數據
)

# 保存最終模型
model.save(os.path.join(save_dir, "unet_skin_segmentation_final.keras"))

# 繪製訓練曲線
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['iou'], label='IoU')
plt.plot(history.history['val_iou'], label='Validation IoU')
plt.title('IoU Curve')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coefficient'], label='Dice')
plt.plot(history.history['val_dice_coefficient'], label='Validation Dice')
plt.title('Dice Coefficient Curve')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_curves.png'))
plt.show()

# 評估模型在測試集上的表現
test_loss, test_iou, test_dice = model.evaluate(x_test, y_test)
print(f"測試集上的損失: {test_loss:.4f}")
print(f"測試集上的IoU: {test_iou:.4f}")
print(f"測試集上的Dice系數: {test_dice:.4f}")

# 顯示預測結果
plt.figure(figsize=(20, 10))
for i in range(8):  # 顯示8個樣本
    if i < len(x_test):
        plt.subplot(2, 4, i+1)

        # 獲取預測
        test_sample = x_test[i:i+1]
        pred_mask = model.predict(test_sample)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # 在原始圖像上疊加遮罩
        overlay = x_test[i].copy()
        # 為預測遮罩添加紅色半透明疊加
        mask_display = np.zeros_like(overlay)
        mask_display[:,:,0] = pred_mask.reshape(IMG_HEIGHT, IMG_WIDTH) * 255  # 紅色通道

        # 混合原始圖像和遮罩
        alpha = 0.5
        cv2_overlay = cv2.addWeighted(overlay, 1, mask_display, alpha, 0)

        plt.imshow(cv2_overlay)
        plt.title(f"樣本 {i+1}")
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prediction_overview.png'))
plt.show()

# 添加對比展示 - 預測結果與真實標籤的對比
plt.figure(figsize=(20, 15))
for i in range(5):  # 顯示5個樣本的對比
    if i < len(x_test):
        # 原始圖片
        plt.subplot(5, 3, i*3+1)
        plt.imshow(x_test[i])
        plt.title(f"原始圖像 {i+1}")
        plt.axis('off')
        
        # 真實標籤
        plt.subplot(5, 3, i*3+2)
        plt.imshow(y_test[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title(f"真實標籤 {i+1}")
        plt.axis('off')
        
        # 預測結果
        plt.subplot(5, 3, i*3+3)
        pred_mask = model.predict(x_test[i:i+1])
        plt.imshow(pred_mask.reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title(f"預測結果 {i+1}")
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comparison_results.png'))
plt.show()

# 儲存測試指標
with open(os.path.join(save_dir, 'test_metrics.txt'), 'w') as f:
    f.write(f"測試集上的損失: {test_loss:.4f}\n")
    f.write(f"測試集上的IoU: {test_iou:.4f}\n")
    f.write(f"測試集上的Dice系數: {test_dice:.4f}\n")

# 計算和顯示混淆矩陣相關指標
def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(bool).flatten()
    y_pred = y_pred.astype(bool).flatten()
    
    # 計算TP, FP, TN, FN
    TP = np.sum(np.logical_and(y_pred, y_true))
    FP = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    TN = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
    FN = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
    
    # 計算精確度、召回率、F1分數
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

# 計算整個測試集的指標
all_preds = model.predict(x_test)
all_preds = (all_preds > 0.5).astype(np.uint8)
metrics = calculate_metrics(y_test, all_preds)

print(f"精確度 (Precision): {metrics['precision']:.4f}")
print(f"召回率 (Recall): {metrics['recall']:.4f}")
print(f"F1分數: {metrics['f1_score']:.4f}")
print(f"準確度 (Accuracy): {metrics['accuracy']:.4f}")

# 將這些指標也寫入文件
with open(os.path.join(save_dir, 'test_metrics.txt'), 'a') as f:
    f.write(f"精確度 (Precision): {metrics['precision']:.4f}\n")
    f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
    f.write(f"F1分數: {metrics['f1_score']:.4f}\n")
    f.write(f"準確度 (Accuracy): {metrics['accuracy']:.4f}\n")

print("訓練和評估完成！")
print(f"所有結果已保存至: {save_dir}")