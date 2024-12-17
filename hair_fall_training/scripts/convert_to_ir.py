from openvino.tools.mo import convert_model

# Input PyTorch model path
input_model = "../model/eye_disease.pth"

# Convert to IR format
ir_model = convert_model(
    model_path=input_model,
    input_shape=[1, 3, 224, 224],
    output_dir="../model",
    data_type="FP16"
)

print("Model converted to OpenVINO IR format!")
