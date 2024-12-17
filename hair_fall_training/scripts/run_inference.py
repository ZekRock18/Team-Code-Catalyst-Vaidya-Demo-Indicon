from openvino.runtime import Core
import cv2
import numpy as np
import os

# Load OpenVINO model
ie = Core()
model_path = "../model/eye_disease.xml"
compiled_model = ie.compile_model(model=model_path, device_name="CPU")

# Model input/output details
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = image[np.newaxis, :, :, :]  # Add batch dimension
    return image.astype(np.float32) / 255.0

# Predict function
def predict(image_path):
    input_image = preprocess_image(image_path)
    result = compiled_model([input_image])[output_layer]
    return np.argmax(result), result

# Test the model
test_image = "../dataset/diabetic_retinopathy/sample_image.jpg"  # Replace with your image path
class_idx, probabilities = predict(test_image)

# Map class index to labels
class_labels = ['Cataract', 'Normal', 'Glaucoma', 'Diabetic Retinopathy']
predicted_label = class_labels[class_idx]

print(f"Predicted Class: {predicted_label}, Probabilities: {probabilities}")
