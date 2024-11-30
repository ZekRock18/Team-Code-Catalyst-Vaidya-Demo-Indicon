import os
import torch
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define constants
MODEL_PATH = "./models/eye_disease.pth"
CLASSES = ["Cataract", "Retinopathy", "Glaucoma", "Normal"]  # Modify based on your model
PREVENTIVE_CARE = {
    "Cataract": [
        "Wear sunglasses to protect your eyes from UV rays.",
        "Maintain a healthy diet rich in antioxidants.",
        "Schedule regular eye check-ups."
    ],
    "Retinopathy": [
        "Control blood sugar levels if diabetic.",
        "Avoid smoking and excessive alcohol consumption.",
        "Maintain a healthy weight and monitor blood pressure."
    ],
    "Glaucoma": [
        "Use prescribed eye drops regularly.",
        "Get regular eye pressure check-ups.",
        "Exercise regularly, but avoid activities that increase eye pressure."
    ],
    "Normal": ["No issues detected. Keep maintaining a healthy lifestyle!"]
}


# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))  # Adjust for your classes
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model


# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Predict the disease
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = probabilities.max(1)
        return predicted_class.item(), probabilities[0].tolist()


# Main app function
def main():
    st.title("Eye Disease Prediction App")
    st.write("Upload an eye image to predict potential diseases and get preventive care tips.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        model = load_model(MODEL_PATH)
        image_tensor = preprocess_image(image)
        predicted_class, probabilities = predict(model, image_tensor)

        # Display results
        st.write("### Prediction Results")
        st.write(f"**Predicted Disease:** {CLASSES[predicted_class]}")
        st.write("### Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASSES[i]}: {prob * 100:.2f}%")

        # Bar chart for probabilities
        st.write("### Visualization")
        fig, ax = plt.subplots()
        sns.barplot(x=CLASSES, y=[p * 100 for p in probabilities], ax=ax)
        ax.set_title("Disease Probabilities")
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

        # Preventive care
        st.write("### Preventive Care Tips")
        st.write("\n".join(PREVENTIVE_CARE[CLASSES[predicted_class]]))


if __name__ == "__main__":
    main()
