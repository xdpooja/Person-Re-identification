import cv2
import numpy as np
from sklearn.cluster import KMeans
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the video file
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Create a directory to store cropped images
output_dir = 'output_cropped_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process the video frames
frame_count = 0
cropped_images = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 represents people in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Crop and save the detected people
    for i in indexes:
        if i[0] < len(boxes):  # Ensure the index is within valid range
            x, y, w, h = boxes[i[0]]
            person_crop = frame[y:y + h, x:x + w]
            person_filename = os.path.join(output_dir, f'person_{frame_count}_{i}.jpg')
            cv2.imwrite(person_filename, person_crop)
            cropped_images.append((person_filename, (x, y, w, h)))  # Store image and its position/size

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Load VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Extract features from cropped images
image_features = []
target_image_size = (224, 224)  # Set the desired image size for VGG16

for image_path, (x, y, w, h) in cropped_images:
    image = cv2.imread(image_path)
    if image is not None:
        # Resize and preprocess the image for VGG16
        image = cv2.resize(image, target_image_size)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Extract features using VGG16
        features = model.predict(image)
        features = features.flatten()  # Flatten the features
        features = np.concatenate([features, [x, y, w, h]])  # Add position/size information
        image_features.append(features)

if image_features:
    image_features = np.vstack(image_features)  # Stack extracted features vertically

    # Apply KMeans clustering on extracted features
    num_clusters = 8
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(image_features)

    # Create directories for clustered images
    cluster_dirs = []  # Store cluster directories
    for cluster_idx in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_idx}')
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_dirs.append(cluster_dir)

    # Move clustered images to respective directories based on KMeans labels
    for (image_path, _), label in zip(cropped_images, kmeans.labels_):
        image_filename = os.path.basename(image_path)
        cluster_dir = cluster_dirs[label]
        new_path = os.path.join(cluster_dir, image_filename)
        os.rename(image_path, new_path)

    print(f"Number of clusters created: {num_clusters}")
else:
    print("No valid images found for clustering.")
