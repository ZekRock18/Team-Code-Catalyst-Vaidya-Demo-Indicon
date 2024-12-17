import os
import torch
from torchvision import transforms, models
from PIL import Image

# Set paths
DATA_DIR = "D:/HackathonProjectsTest/hair_fall_classification/img"  # Update to your test image directory
MODEL_PATH = "./models/hair_classification.pth"  # Update to your trained model path

# Define classes for hair classification
CLASSES = ["Stage1", "Stage2", "Stage3"]  # Ensure these match your training dataset classes

def load_model(model_path):
    """
    Load the pre-trained model and optimize for CPU usage.
    """
    try:
        # Load ResNet-18 architecture
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))  # Adjust output layer

        # Load trained weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # Optimize the model for CPU
        model.eval()  # Set the model to evaluation mode
        return model.to(memory_format=torch.channels_last)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image_path):
    """
    Preprocess an input image for the model.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        raise


def predict_hair_stage(model, image_tensor):
    """
    Predict the hair classification stage.
    """
    try:
        with torch.no_grad():
            # Convert image tensor to channels-last format
            image_tensor = image_tensor.to(memory_format=torch.channels_last)

            # Perform inference
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = probabilities.max(1)
            return predicted_class.item(), probabilities[0].tolist()
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


def main():
    # Load the model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Iterate through images in the data directory
    for image_name in os.listdir(DATA_DIR):
        image_path = os.path.join(DATA_DIR, image_name)

        # Check if the file is an image
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Skipping non-image file: {image_name}")
            continue

        try:
            image_tensor = preprocess_image(image_path)

            # Predict the hair classification stage
            predicted_class, probabilities = predict_hair_stage(model, image_tensor)
            print(f"Image: {image_name}")
            print(f"Predicted Stage: {CLASSES[predicted_class]}")
            print(f"Probabilities: {probabilities}")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")


if __name__ == "__main__":
    main()
