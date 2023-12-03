import os
import glob
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def load_image_into_numpy_array(path):
    """
    Load an image from file and convert it to a numpy array.
    """
    image = Image.open(path).convert('RGB')
    return np.array(image)

def draw_detections(image, boxes, classes, scores, threshold=0.5, class_names=None):
    """
    Draw bounding boxes and labels on an image.
    """
    height, width, _ = image.shape
    for i in range(boxes.shape[0]):
        if scores[i] > threshold:
            box = boxes[i]
            class_id = int(classes[i])
            score = scores[i]

            # Scale box to image coordinates
            top_left = (int(box[1] * width), int(box[0] * height))
            bottom_right = (int(box[3] * width), int(box[2] * height))

            # Draw the box
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # Draw label
            if class_names and class_id in class_names:
                label = f"{class_names[class_id]}: {score:.2f}"
                label_position = (top_left[0], top_left[1] + 20)  # Position just below the top
                cv2.putText(image, label, label_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Path to the directory containing the exported model
model_dir = 'exported_models/ssd_mobilenet/saved_model'

# Load the model
print("Loading model...")
model = tf.saved_model.load(model_dir)

# Path to the folder containing test images
test_image_directory = '../assets/test_dataset'
test_image_paths = glob.glob(os.path.join(test_image_directory, '*.png'))  # Adjust as needed

# Class names mapping
class_names = {1: 'Bottle'}  # Update with your class names

# Loop through the test images and perform inference
for image_path in test_image_paths:
    image_np = load_image_into_numpy_array(image_path)
    image_np = image_np.astype(np.uint8)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

    # Run inference
    detections = model(input_tensor)

    # Extract elements from the model's output
    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()[0]

    # Draw detections on the image
    image_with_detections = draw_detections(image_np, detection_boxes, detection_classes, detection_scores, class_names=class_names)

    # Display the image (using OpenCV)
    cv2.imshow('Detection', image_with_detections)
    cv2.waitKey(0)  # Wait for a key press to continue

cv2.destroyAllWindows()

