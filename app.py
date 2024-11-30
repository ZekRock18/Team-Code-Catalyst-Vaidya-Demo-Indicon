import streamlit as st
# Set page config at the top of your script
st.set_page_config(page_title="Disease Prediction", page_icon="🌟", layout="wide")

import os
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants for models and classes
MODEL_PATH_SKIN = "D:/HackathonProjectsTest/Indicon_demo/skin_disease_prediciton/models/skin_disease.pth"
MODEL_PATH_EYE = "D:/HackathonProjectsTest/Indicon_demo/eye_disease_predicition/models/eye_disease.pth"
CLASSES_SKIN = [
    "Acne", "Actinic_Keratosis", "Bullous", "DrugEruption", "Eczema",
    "Infestations_Bites", "Lichen", "Lupus", "Moles", "Seborrh_Keratoses",
    "Sun_Sunlight_Damage", "Vasculitis", "Vitiligo", "Warts"
]
CLASSES_EYE = ["Cataract", "Retinopathy", "Glaucoma", "Normal"]

PREVENTIVE_CARE_SKIN = {
    "Acne": "Keep your skin clean, use non-comedogenic products, avoid touching your face.",
    "Actinic_Keratosis": "Wear sunscreen daily, avoid excessive sun exposure, and get regular skin checks.",
    "Bullous": "Avoid direct trauma to the skin, use gentle skincare products.",
    "DrugEruption": "Consult your doctor for medication adjustments, avoid further exposure to the drug.",
    "Eczema": "Moisturize regularly, avoid triggers, use prescribed creams.",
    "Infestations_Bites": "Use anti-itch creams, keep the affected area clean and dry.",
    "Lichen": "Use corticosteroid creams, avoid scratching, and manage stress.",
    "Lupus": "Use sunscreen, avoid excessive sun exposure, and manage stress.",
    "Moles": "Get regular checkups for new or changing moles.",
    "Seborrh_Keratoses": "Generally, no treatment is needed, but removal can be considered if they cause discomfort.",
    "Sun_Sunlight_Damage": "Wear sunscreen, avoid tanning, and moisturize your skin.",
    "Vasculitis": "Consult with your healthcare provider for proper treatment and medications.",
    "Vitiligo": "Use sunscreen to protect depigmented areas, consider cosmetic camouflage products.",
    "Warts": "Use over-the-counter treatments, avoid touching or picking at warts."
}

PREVENTIVE_CARE_EYE = {
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

# Helper functions
@st.cache_data
def load_model(model_path, output_classes):
    """Load the trained model and adjust for the specific number of classes."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(output_classes))  # Adjust for your classes
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image):
    """Preprocess input image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    """Predict the class of the disease."""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = probabilities.max(1)
        return predicted_class.item(), probabilities[0].tolist()

def display_preventive_care(disease, model_type="skin"):
    """Display preventive care based on the predicted disease."""
    if model_type == "skin":
        preventive_care = PREVENTIVE_CARE_SKIN
    else:
        preventive_care = PREVENTIVE_CARE_EYE

    tips = preventive_care.get(disease, ["No specific preventive care available."])
    
    st.write("### Preventive Care Tips:")
    
    # This ensures all tips are printed on one line
    if isinstance(tips, list):
        st.write(", ".join(tips))  # Join list items with commas
    else:
        st.write(tips)
def plot_probabilities(probabilities, classes):
    """Plot the prediction probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=classes, y=[p * 100 for p in probabilities], ax=ax)
    ax.set_title("Prediction Probabilities")
    ax.set_ylabel("Probability (%)")
    
    # Rotate the x-axis labels to vertical
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    
    st.pyplot(fig)

# Streamlit app
def main():
    # Sidebar for navigation
    st.sidebar.title("Select Prediction Model")
    page = st.sidebar.radio("Choose the disease prediction model:", ("Home", "Skin Disease", "Eye Disease"))

    if page == "Home":
        st.title("Welcome to Disease Prediction App 🌟")
        st.write("Choose a model from the sidebar to predict the disease and get preventive care tips!")
    
    elif page == "Skin Disease":
        st.title("Skin Disease Prediction 🧴")
        st.write("Upload an image of your skin for disease prediction.")

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image.", use_column_width=True)

            # Load the model and make prediction
            model = load_model(MODEL_PATH_SKIN, CLASSES_SKIN)
            image_tensor = preprocess_image(img)
            predicted_class, probabilities = predict(model, image_tensor)

            disease = CLASSES_SKIN[predicted_class]
            st.subheader(f"Predicted Disease: {disease}")
            st.write(f"Prediction Probability: {probabilities[predicted_class] * 100:.2f}%")
            plot_probabilities(probabilities, CLASSES_SKIN)
            display_preventive_care(disease, model_type="skin")

    elif page == "Eye Disease":
        st.title("Eye Disease Prediction 👁️")
        st.write("Upload an image of your eye for disease prediction.")

        uploaded_image = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image.", use_column_width=True)

            # Load the model and make prediction
            model = load_model(MODEL_PATH_EYE, CLASSES_EYE)
            image_tensor = preprocess_image(img)
            predicted_class, probabilities = predict(model, image_tensor)

            disease = CLASSES_EYE[predicted_class]
            st.subheader(f"Predicted Disease: {disease}")
            st.write(f"Prediction Probability: {probabilities[predicted_class] * 100:.2f}%")
            plot_probabilities(probabilities, CLASSES_EYE)
            display_preventive_care(disease, model_type="eye")

if __name__ == "__main__":
    main()
