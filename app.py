import kagglehub
import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_images(folder_paths, folder_types, target_size=(128, 128), sample_count=5):
    images = []
    labels = []
    all_filenames = []

    for folder_path, folder_type in zip(folder_paths, folder_types):
        label = 1 if folder_type.lower() == 'real' else 0
        print(f"Loading images from: {folder_path}")

        sample_images = []  
        sample_titles = []  

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png')):
                img_path = os.path.join(folder_path, filename)

                try:
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img)
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    img_array = img / 255.0  
                    img_array = np.expand_dims(img_array, axis=-1)

                    if len(sample_images) < sample_count:
                        sample_images.append(img_array)
                        sample_titles.append(f"{folder_type.capitalize()}: {filename}")

                    images.append(img_array)
                    labels.append(label)
                    all_filenames.append(img_path)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"Total images loaded: {len(images)}")
    return images, labels

# Download dataset
path = kagglehub.dataset_download("divyanshrai/handwritten-signatures")

# Define dataset paths
forged_data = [
    "./Dataset_Signature_Final/Dataset/dataset1/forge/",
    "./Dataset_Signature_Final/Dataset/dataset2/forge/",
    "./Dataset_Signature_Final/Dataset/dataset3/forge/",
    "./Dataset_Signature_Final/Dataset/dataset4/forge/"
]

real_data = [
    "./Dataset_Signature_Final/Dataset/dataset1/real/",
    "./Dataset_Signature_Final/Dataset/dataset2/real/",
    "./Dataset_Signature_Final/Dataset/dataset3/real/",
    "./Dataset_Signature_Final/Dataset/dataset4/real/"
]

# Load images
forged_imgs, forged_label = load_images(forged_data, ["forged"] * len(forged_data))
real_imgs, real_label = load_images(real_data, ["real"] * len(real_data))

# Concatenate data
all_images = np.concatenate((forged_imgs, real_imgs), axis=0)
all_labels = np.concatenate((forged_label, real_label), axis=0)

# Shuffle dataset
all_images, all_labels = shuffle(all_images, all_labels, random_state=42)

# Convert labels to categorical format
all_labels = to_categorical(all_labels, num_classes=2)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.3, random_state=42)

# Print dataset shapes
print(f"Train images shape: {x_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}, Test labels shape: {y_test.shape}")

# Define CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes (real, forged)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create and compile model
model = create_model()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    batch_size=32
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot Training Results
def plot_train(hist):
    history = hist.history

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    if 'accuracy' in history:  
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy Over Epochs')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid()

    plt.tight_layout()
    plt.show()

# Plot training results
plot_train(history)

# Save the trained model
model.save("signature_authentication_model.h5")
print("Model saved successfully!")
