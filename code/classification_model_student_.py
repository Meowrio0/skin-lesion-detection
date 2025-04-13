# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Set random seeds for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Disable meta optimizer (optional for Colab, but kept for consistency)
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

# ========== Environment Settings ==========
BASE_DIR = '/content/drive/MyDrive'
OUTPUT_DIR = os.path.join(BASE_DIR, 'skin_lesion_classification_result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_DATASET_DIR = '/content/drive/MyDrive/split_dataset'
TRAIN_SPLIT_DIR = os.path.join(SPLIT_DATASET_DIR, "train")
TEST_SPLIT_DIR = os.path.join(SPLIT_DATASET_DIR, "test")

IMG_SIZE = 224  # For DenseNet121 pretrained weights
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5  # Using the teacher model's learning rate

# ========== Load Model with Custom Objects ==========
def load_model_with_custom_objects(model_path):
    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(**kwargs)

    custom_objects = {
        'CustomInputLayer': CustomInputLayer,
        'InputLayer': CustomInputLayer,
        'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,
        'DTypePolicy': tf.keras.mixed_precision.Policy
    }

    try:
        print(f"[Debug] Attempting to load model from {model_path}")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("[Debug] Model loaded successfully, recompiling...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        return model
    except Exception as e:
        print(f"Detailed error loading model: {e}")
        return None

# ========== Memory Optimization ==========
def limit_memory_usage():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[Info] GPU memory growth enabled")
        else:
            print("[Info] No GPU detected, using CPU")
    except Exception as e:
        print(f"[Warning] Failed to set GPU memory growth: {e}")

# Disable oneDNN opts (optional for Colab)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========== Create Data Generators from Split Folders with Oversampling ==========
def create_data_generators_from_split():
    import glob

    train_list = []
    train_classes = os.listdir(TRAIN_SPLIT_DIR)
    for cls in train_classes:
        class_dir = os.path.join(TRAIN_SPLIT_DIR, cls)
        if os.path.isdir(class_dir):
            files = glob.glob(os.path.join(class_dir, "*.jpg"))
            for f in files:
                train_list.append({"filename": f, "diagnosis": cls})
    train_df = pd.DataFrame(train_list)

    test_list = []
    test_classes = os.listdir(TEST_SPLIT_DIR)
    for cls in test_classes:
        class_dir = os.path.join(TEST_SPLIT_DIR, cls)
        if os.path.isdir(class_dir):
            files = glob.glob(os.path.join(class_dir, "*.jpg"))
            for f in files:
                test_list.append({"filename": f, "diagnosis": cls})
    test_df = pd.DataFrame(test_list)

    X = train_df.index.values.reshape(-1, 1)
    y = train_df['diagnosis']
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    train_df_over = train_df.loc[X_res.flatten()].reset_index(drop=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df_over,
        x_col='filename',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

# ========== Build Student Model (MobileNetV2) ==========
def build_student_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.7),  # 增加 dropout
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)),  # 加強正則化
        layers.BatchNormalization(),
        layers.Dropout(0.7),  # 增加 dropout
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ========== Plot Training History ==========
def plot_history(history, model_name):
    if "loss" in history:
        train_loss = history["loss"]
        train_acc = history["accuracy"]
        val_loss = history["val_loss"]
        val_acc = history["val_accuracy"]
    else:
        train_loss = history["train_loss"]
        train_acc = history["train_acc"]
        val_loss = history["val_loss"]
        val_acc = history["val_acc"]

    epochs_range = range(1, len(train_loss)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Accuracy Curve')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{model_name}_training_history.png')
    plt.savefig(save_path)
    plt.show()
    print(f"[Info] Training history saved at: {save_path}")

# ========== Plot Confusion Matrix ==========
def plot_confusion_matrix(model, generator, model_name):
    steps = math.ceil(generator.n / generator.batch_size)
    preds = model.predict(generator, steps=steps)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes

    class_indices = generator.class_indices
    class_names = [None] * len(class_indices)
    for key, value in class_indices.items():
        class_names[value] = key

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    save_path = os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()
    print(f"[Info] Confusion matrix saved at: {save_path}")

# ========== Distillation Training ==========
def train_distillation(teacher_model, student_model, train_generator, val_generator, epochs, temperature=7, alpha=0.5):
    teacher_model.trainable = False
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_loss_metric = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    categorical_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    kl_div_loss_fn = tf.keras.losses.KLDivergence()

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # EarlyStopping 參數設定
    best_val_loss = np.inf
    patience = 8  # 若連續 8 個 epoch 驗證損失沒有改善則停止
    wait = 0
    best_student_weights = None

    @tf.function
    def train_step(images, labels):
        teacher_preds = teacher_model(images, training=False)
        teacher_preds_temp = tf.nn.softmax(teacher_preds / temperature)
        with tf.GradientTape() as tape:
            student_preds = student_model(images, training=True)
            student_preds_temp = tf.nn.softmax(student_preds / temperature)
            hard_loss = categorical_loss_fn(labels, student_preds)
            soft_loss = kl_div_loss_fn(teacher_preds_temp, student_preds_temp)
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
        gradients = tape.gradient(loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(labels, student_preds)
        return loss

    @tf.function
    def val_step(images, labels):
        student_preds = student_model(images, training=False)
        loss = categorical_loss_fn(labels, student_preds)
        val_loss_metric.update_state(loss)
        val_acc_metric.update_state(labels, student_preds)
        return loss

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()
        val_loss_metric.reset_state()
        val_acc_metric.reset_state()

        # 固定步數來確保一個 epoch 完整遍歷資料集
        steps_per_epoch = train_generator.n // train_generator.batch_size
        for i in range(steps_per_epoch):
            images, labels = next(train_generator)
            _ = train_step(images, labels)

        val_steps = val_generator.n // val_generator.batch_size
        for i in range(val_steps):
            images, labels = next(val_generator)
            _ = val_step(images, labels)

        train_loss = train_loss_metric.result().numpy()
        train_acc = train_acc_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()
        val_acc = val_acc_metric.result().numpy()
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # 只使用 EarlyStopping，不降低學習率
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_student_weights = student_model.get_weights()
            print(f"[Info] Best model updated with validation loss: {val_loss:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"[Info] Early stopping triggered at epoch {epoch+1}")
                break

    # 載入最佳權重
    if best_student_weights is not None:
        student_model.set_weights(best_student_weights)
        print("[Info] Best student weights loaded.")

    # 儲存最終模型
    student_model.save(os.path.join(OUTPUT_DIR, 'student_model_distilled.h5'))
    history_dict = {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history
    }
    return student_model, history_dict
# ========== Model Evaluation ==========
def evaluate_model(model, generator, model_name):
    loss, acc = model.evaluate(generator, verbose=1)
    print(f"{model_name} - Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return loss, acc

# ========== Main Process ==========
def main():
    limit_memory_usage()
    train_generator, val_generator = create_data_generators_from_split()
    num_classes = len(train_generator.class_indices)

    print(f"[Info] Number of classes: {num_classes}")
    print(f"[Info] Training samples: {train_generator.n}")
    print(f"[Info] Validation samples: {val_generator.n}")

    # Load pre-trained teacher model
    print("[Info] Loading pre-trained DenseNet201 teacher model...")
    teacher_model_path = '/content/drive/MyDrive/skin_lesion_classification_result/best_DenseNet201.h5'
    teacher_model = load_model_with_custom_objects(teacher_model_path)

    if teacher_model is None:
        print("[Error] Failed to load teacher model. Exiting.")
        return

    teacher_model.summary()
    print(f"[Info] Loaded teacher model from: {teacher_model_path}")

    # Evaluate teacher model
    evaluate_model(teacher_model, val_generator, "DenseNet201")
    plot_confusion_matrix(teacher_model, val_generator, "DenseNet201")

    # Build and train student model
    print("[Info] Building MobileNetV2 student model...")
    student_model = build_student_model((IMG_SIZE, IMG_SIZE, 3), num_classes)
    student_model.summary()

    print("[Info] Training distillation...")
    distilled_student, distillation_history = train_distillation(
        teacher_model, student_model, train_generator, val_generator, epochs=100, temperature=7, alpha=0.5
    )
    print("[Info] Distillation training complete!")

    plot_history(distillation_history, "Student_MobileV2_Distilled")
    evaluate_model(distilled_student, val_generator, "Student_MobileV2 (Distilled)")
    plot_confusion_matrix(distilled_student, val_generator, "Student_MobileV2_Distilled")

    print(f"[Info] All results saved at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
