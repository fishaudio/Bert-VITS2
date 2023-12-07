from onnx_modules import export_onnx
import os

if __name__ == "__main__":
    export_path = "MyModel"
    model_path = "S:\\VSGIT\\bert-vits2\\G_178000.pth"
    config_path = "S:\\VSGIT\\bert-vits2\\config.json"
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    export_onnx(export_path, model_path, config_path)
