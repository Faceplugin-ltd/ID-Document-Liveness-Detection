import cv2
import os

from ultralytics import YOLO

model = YOLO('weight/best.pt')
# Define the target class name (case-insensitive)

def process_image(path):
    frame = cv2.imread(path)
    results = model(frame)

    # Process predictions to extract relevant information
    predicted_class_list = results[0].boxes.cls.tolist()  # Assuming single image output
    if predicted_class_list:
        predicted_class = predicted_class_list[0]
        if predicted_class == 3.0:
            print(">>>>>>>>>>>>>>> Live")
        else:
            print(">>>>>>>>>>>>>>> Fake")
    else:
        print(">>>>>>>>>>>>>>> NO ID Card Detected! Upload another image!")

input_img_dir = "test_image"
for root, dirs, files in os.walk(input_img_dir):
    for file in files:
        print(file)
        img_path = os.path.join(root, file)
        process_image(img_path)
