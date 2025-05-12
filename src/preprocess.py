import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re

def load_images_from_folder(folder):
    print(f"Scanning folder: {folder}")
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to load image '{img_path}'. Skipping due to invalid file.")
                    continue
                print(f"Loaded {filename} with shape {img.shape}")
                yield img, filename
            except cv2.error as e:
                print(f"Error: Failed to load '{img_path}' due to memory issue. Attempting to resize and retry...")
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
                    if img is not None:
                        print(f"Resized and loaded {filename} with shape {img.shape}")
                        yield img, filename
                    else:
                        print(f"Warning: Failed to load even with reduced resolution '{img_path}'. Skipping.")
                except cv2.error as e2:
                    print(f"Error: Final attempt failed for '{img_path}'. Reason: {e2}")
                    continue

def resize_image(image, size=(64, 64)):
    resized_image = cv2.resize(image, size)
    return resized_image

def normalize_image(image):
    normalized_image = np.array(image) / 255.0
    return normalized_image

def augment_images(image, num_augmented=2):
    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    image = image.reshape((1,) + image.shape)
    augmented_images = []
    count = 0
    for batch in datagen.flow(image, batch_size=1):
        augmented_images.append(batch[0])
        count += 1
        if count >= num_augmented:
            break
    return augmented_images

def preprocess_data(raw_data_folder, processed_data_folder):
    try:
        if not os.path.exists(processed_data_folder):
            print(f"Creating directory: {processed_data_folder}")
            os.makedirs(processed_data_folder)
    except PermissionError:
        print(f"Error: No permission to create directory '{processed_data_folder}'. Try running as administrator or checking folder permissions.")
        exit()
    except OSError as e:
        print(f"Error: Failed to create directory '{processed_data_folder}'. Reason: {e}")
        exit()

    count = 0
    image_generator = load_images_from_folder(raw_data_folder)
    processed_images = []

    for img, filename in image_generator:
        try:
            # Process one image at a time
            resized_img = resize_image(img)
            normalized_img = normalize_image(resized_img)
            
            # Add original image
            processed_images.append((normalized_img, filename, False))
            
            # Add augmented images
            augmented_imgs = augment_images(normalized_img)
            for aug_img in augmented_imgs:
                processed_images.append((aug_img, filename, True))
            
            # Save immediately to free memory
            for i, (image, orig_filename, is_augmented) in enumerate(processed_images):
                sign_name = re.match(r'([^_]+)_', orig_filename)
                sign_name = sign_name.group(1) if sign_name else 'unknown'
                suffix = f"aug_{count}" if is_augmented else f"orig_{count}"
                output_filename = f"{sign_name}_processed_{suffix}.png"
                output_path = os.path.join(processed_data_folder, output_filename)
                
                try:
                    cv2.imwrite(output_path, image * 255)
                    print(f"Saved: {output_path}")
                    count += 1
                except PermissionError:
                    print(f"Error: No permission to write image '{output_path}'. Check folder permissions or run as administrator.")
                    return count
                except OSError as e:
                    print(f"Error: Failed to write image '{output_path}'. Reason: {e}")
                    return count
            
            # Clear processed images to free memory
            processed_images = []
            
        except Exception as e:
            print(f"Error processing image '{filename}'. Skipping. Reason: {e}")
            continue

    if count == 0:
        print(f"No images were processed. Check raw data folder and permissions.")
    return count

# Example usage
raw_data_folder = r"E:\new yousef\hand-gesture-recognition\data\raw"
processed_data_folder = r"E:\new yousef\hand-gesture-recognition\data\processed"
num_saved = preprocess_data(raw_data_folder, processed_data_folder)
print(f"Total images saved: {num_saved}")