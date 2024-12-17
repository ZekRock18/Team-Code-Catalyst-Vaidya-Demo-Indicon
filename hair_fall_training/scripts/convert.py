import torch
import torch.onnx


model = torch.load('D:\HackathonProjectsTest\INDICON\eye_disease_classification\model\eye_disease.pth')




# Example input tensor with the appropriate size that matches the model's input
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size based on your model's input shape

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)