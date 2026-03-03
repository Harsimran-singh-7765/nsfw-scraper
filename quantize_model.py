import torch
import torch.nn as nn
import timm
import os

# Config
MODEL_PATH = "models/efficientnet_v2s_8922.pth"
OUTPUT_PATH = "models/efficientnet_v2s_8922_int8.pth"
MODEL_NAME = "tf_efficientnetv2_s"
NUM_CLASSES = 5

def quantize():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found!")
        return

    print(f"📦 Loading model: {MODEL_PATH}...")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("⚙️  Quantizing model (INT8 dynamic)...")
    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    print(f"💾 Saving quantized model to: {OUTPUT_PATH}...")
    torch.save(model_q.state_dict(), OUTPUT_PATH)

    fp32_size = os.path.getsize(MODEL_PATH) / 1e6
    int8_size = os.path.getsize(OUTPUT_PATH) / 1e6
    
    print(f"\n✅ Success!")
    print(f"  FP32 Size: {fp32_size:.2f} MB")
    print(f"  INT8 Size: {int8_size:.2f} MB")
    print(f"  Reduction: {100*(1-int8_size/fp32_size):.1f}%")

if __name__ == "__main__":
    quantize()
