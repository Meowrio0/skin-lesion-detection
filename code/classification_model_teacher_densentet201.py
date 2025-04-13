import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random  # Added for reproducibility
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from google.colab import drive
import zipfile

# Mount Google Drive for saving results
drive.mount('/content/drive')

# Set random seeds for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ========== Environment Settings ==========
# Change these paths according to your Google Drive structure
BASE_DIR = '/content/drive/MyDrive'
OUTPUT_DIR = os.path.join(BASE_DIR, 'skin_lesion_classification_result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_DATASET_DIR = '/content/drive/MyDrive/split_dataset'
TRAIN_SPLIT_DIR = os.path.join(SPLIT_DATASET_DIR, "train")
TEST_SPLIT_DIR = os.path.join(SPLIT_DATASET_DIR, "test")

IMG_SIZE = 224  # For DenseNet121 pretrained weights
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4  # Using the teacher model's learning rate

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

# ========== Create Data Generators from Split Folders with Oversampling ==========
def create_data_generators_from_split():
    """
    Reads image paths and labels from the split folders (TRAIN_SPLIT_DIR and TEST_SPLIT_DIR),
    applies oversampling on the training set, and creates ImageDataGenerators.
    """
    import glob

    # Build training DataFrame
    train_list = []
    train_classes = os.listdir(TRAIN_SPLIT_DIR)
    for cls in train_classes:
        class_dir = os.path.join(TRAIN_SPLIT_DIR, cls)
        if os.path.isdir(class_dir):
            # Assuming images are in .jpg format
            files = glob.glob(os.path.join(class_dir, "*.jpg"))
            for f in files:
                train_list.append({"filename": f, "diagnosis": cls})
    train_df = pd.DataFrame(train_list)

    # Build testing DataFrame
    test_list = []
    test_classes = os.listdir(TEST_SPLIT_DIR)
    for cls in test_classes:
        class_dir = os.path.join(TEST_SPLIT_DIR, cls)
        if os.path.isdir(class_dir):
            files = glob.glob(os.path.join(class_dir, "*.jpg"))
            for f in files:
                test_list.append({"filename": f, "diagnosis": cls})
    test_df = pd.DataFrame(test_list)

    # Print dataset statistics
    print(f"[Info] Number of training images: {len(train_df)}")
    print(f"[Info] Number of testing images: {len(test_df)}")
    print(f"[Info] Training class distribution:")
    print(train_df['diagnosis'].value_counts())

    # Apply oversampling on the training set to balance classes
    X = train_df.index.values.reshape(-1, 1)
    y = train_df['diagnosis']
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    train_df_over = train_df.loc[X_res.flatten()].reset_index(drop=True)

    print(f"[Info] After oversampling - Training class distribution:")
    print(train_df_over['diagnosis'].value_counts())

    # Data augmentation for training set with advanced augmentation parameters
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,           # Advanced augmentation: shear transformation
        zoom_range=0.2,
        brightness_range=[0.8, 1.2], # Advanced augmentation: brightness adjustment
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    # Testing set only rescales images
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

# ========== Build Teacher Model (Using DenseNet121) ==========
def build_teacher_model(input_shape, num_classes):
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)

    # First freeze the entire model
    base_model.trainable = False

    # Then unfreeze the last 50 layers for fine-tuning
    # DenseNet121 has many layers and benefits from more fine-tuning
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.9),  # Slightly reduced dropout
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),  # Added regularization and increased units
        layers.BatchNormalization(),
        layers.Dropout(0.8),  # Kept the same
        layers.Dense(num_classes, activation='softmax')
    ])

    # Use a lower learning rate with weight decay
    optimizer = optimizers.Adam(
        learning_rate=LEARNING_RATE,  # Lower learning rate for stable training
        weight_decay=1e-5    # Added weight decay for better generalization
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]  # Added top-5 accuracy metric
    )

    return model

# ========== Training Function ==========
def train_model_func(model, train_generator, val_generator, model_name, epochs):
    checkpoint_path = os.path.join(OUTPUT_DIR, f'best_{model_name}.h5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    return history, checkpoint_path

# ========== Plot Training History ==========
def plot_history(history, model_name):
    train_loss = history.history["loss"]
    train_acc = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]

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

    # Save history to CSV for later analysis
    history_df = pd.DataFrame({
        'epoch': epochs_range,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    })
    history_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_history.csv'), index=False)

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

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add text annotations
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

    # Calculate per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Print per-class metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    print("\nPer-class metrics:")
    print(metrics_df)

    # Save metrics to file
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_class_metrics.csv'), index=False)

# ========== Model Evaluation ==========
def evaluate_model(model, generator, model_name):
    loss, acc = model.evaluate(generator, verbose=1)
    print(f"{model_name} - Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, f'{model_name}_evaluation.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Validation Loss: {loss:.4f}\n")
        f.write(f"Validation Accuracy: {acc:.4f}\n")

    return loss, acc

# ========== Main Process ==========
def main():
    limit_memory_usage()

    # Create data generators
    print("[Info] Creating data generators...")
    train_generator, val_generator = create_data_generators_from_split()
    num_classes = len(train_generator.class_indices)

    print(f"[Info] Number of classes: {num_classes}")
    print(f"[Info] Training samples: {train_generator.n}")
    print(f"[Info] Validation samples: {val_generator.n}")

    # Save class indices for later use
    class_indices = train_generator.class_indices
    with open(os.path.join(OUTPUT_DIR, 'class_indices.txt'), 'w') as f:
        for cls, idx in class_indices.items():
            f.write(f"{cls}: {idx}\n")

    print("[Info] Building DenseNet201 teacher model...")
    teacher_model = build_teacher_model((IMG_SIZE, IMG_SIZE, 3), num_classes)
    teacher_model.summary()

    # Save model architecture visualization
    try:
        tf.keras.utils.plot_model(
            teacher_model,
            to_file=os.path.join(OUTPUT_DIR, 'teacher_model_architecture.png'),
            show_shapes=True,
            show_layer_names=True
        )
    except Exception as e:
        print(f"[Warning] Could not save model visualization: {e}")

    print("[Info] Training teacher model...")
    teacher_history, teacher_model_path = train_model_func(teacher_model, train_generator, val_generator, "DenseNet201", epochs=EPOCHS)
    print(f"[Info] Best teacher model saved at: {teacher_model_path}")

    # Save the final model
    teacher_model.save(os.path.join(OUTPUT_DIR, 'densenet201_teacher_final.h5'))

    # Plot training history
    plot_history(teacher_history, "DenseNet201")

    # Load best model for evaluation
    best_teacher_model = tf.keras.models.load_model(teacher_model_path)

    # Evaluate the model
    print("[Info] Evaluating teacher model...")
    evaluate_model(best_teacher_model, val_generator, "DenseNet201")

    # Plot confusion matrix
    print("[Info] Generating confusion matrix...")
    plot_confusion_matrix(best_teacher_model, val_generator, "DenseNet201")

    print(f"[Info] All results saved at: {OUTPUT_DIR}")
    print("[Info] Process completed successfully!")

if __name__ == "__main__":
    main()
