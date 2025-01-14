import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics.pairwise import cosine_similarity

# Suppress TensorFlow debugging info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Set UTF-8 encoding for the script
sys.stdout.reconfigure(encoding="utf-8")

# Data augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)


# Load the database images and their corresponding card names
def load_card_database(database_path, model, target_size):
    card_features = []
    card_names = []
    batch_images = []
    batch_names = []

    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(database_path, filename))
            if img is not None:
                processed_img = preprocess_image(img, target_size)
                batch_images.append(processed_img)
                batch_names.append(os.path.splitext(filename)[0])

                # Batch processing
                if len(batch_images) >= batch_size:
                    batch_images = np.vstack(batch_images)
                    print(f"Batch images shape: {batch_images.shape}")
                    features = model.predict(batch_images, verbose=0)
                    card_features.extend(features)
                    card_names.extend(batch_names)
                    batch_images = []
                    batch_names = []

    # Process any remaining images
    if batch_images:
        batch_images = np.vstack(batch_images)
        print(f"Remaining batch images shape: {batch_images.shape}")
        features = model.predict(batch_images, verbose=0)
        card_features.extend(features)
        card_names.extend(batch_names)

    return np.array(card_features), card_names


# Preprocess images for the model
def preprocess_image(image, target_size):
    image = cv2.resize(
        image, (target_size[1], target_size[0])
    )  # Correct the target size order
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


# Find the closest match using cosine similarity
def find_closest_match(
    input_image, card_features, card_names, model, target_size=(300, 450)
):
    input_image_processed = preprocess_image(input_image, target_size)
    print(f"Input image processed shape: {input_image_processed.shape}")
    input_features = model.predict(input_image_processed, verbose=0).flatten()

    similarities = cosine_similarity([input_features], card_features)
    closest_match_index = np.argmax(similarities)
    return card_names[closest_match_index]


# Parameters
batch_size = 8  # Adjust the batch size based on your GPU/CPU memory capacity

# Load the pre-trained model (EfficientNetB4 for this example)
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(300, 450, 3)
)

# Optionally, fine-tune the model on your dataset
# base_model.trainable = True  # Uncomment this line to enable fine-tuning

# Create a new model with data augmentation and the base model
input_layer = tf.keras.layers.Input(shape=(300, 450, 3))
augmented_layer = data_augmentation(input_layer)
output_layer = base_model(augmented_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model (necessary only if fine-tuning)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy')

# Load the card database
database_path = "data/cards"
card_features, card_names = load_card_database(database_path, model, (300, 450))

while True:
    input_file = input("Enter the path to the input image: ")
    # Load the input image
    input_image_path = f"data/card_tests/{input_file}"
    input_image = cv2.imread(input_image_path)

    # Measure the time taken to find the closest match
    start_time = time.time()
    closest_match = find_closest_match(input_image, card_features, card_names, model)
    end_time = time.time()

    # Print the closest match and the time taken
    print(f"The closest match is: {closest_match}")
    print(f"Time taken to find the closest match: {end_time - start_time} seconds")
