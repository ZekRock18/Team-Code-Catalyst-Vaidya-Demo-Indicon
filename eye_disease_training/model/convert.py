import torch
from torchvision import models

# Recreate the model architecture
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust for 4 classes

# Load the state dictionary
model.load_state_dict(torch.load("D:\HackathonProjectsTest\INDICON\eye_disease_classification\model\eye_disease.pth"))

# Set the model to evaluation mode
model.eval()

# Now the model is ready for conversion
from openvino.tools.pytorch import mo_pytorch

# Convert to OpenVINO IR
ir_model = mo_pytorch.convert_model(model, input_shape=(1, 3, 224, 224))
ir_model.save_model("eye_disease_openvino")
