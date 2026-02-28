import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path

AVAILABLE_MODELS = {
    "efficientnet_v2s": {
        "id": "efficientnet_v2s",
        "name": "EfficientNet-V2-S",
        "arch": "tf_efficientnetv2_s",
        "accuracy": "96.4%",
        "filename": "efficientnet_v2s.pth",
        "description": "Fast & highly accurate (Epoch 10) on 5-class noisy dataset."
    },
    "resnet50_v1": {
        "id": "resnet50_v1",
        "name": "ResNet-50 (Legacy)",
        "arch": "resnet50",
        "accuracy": "85.6%",
        "filename": "resnet50_v1.pth",
        "description": "Original model baseline trained on raw uncleaned data."
    }
}

class NSFWClassifier:
    def __init__(self, model_id="efficientnet_v2s"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        self.active_model_id = None
        self.model = None
        
        # Load the default model
        self.load_model(model_id)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_id):
        if model_id not in AVAILABLE_MODELS:
            print(f"❌ ERROR: Model ID '{model_id}' not found in registry. Falling back to efficientnet_v2s.")
            model_id = "efficientnet_v2s"
            
        if self.active_model_id == model_id:
            return True # Already loaded
            
        model_info = AVAILABLE_MODELS[model_id]
        print(f"🚀 Initializing NSFW Classifier (Model: {model_info['name']})...")
        
        try:
            # Create new model instance
            new_model = timm.create_model(model_info['arch'], pretrained=False, num_classes=len(self.classes))
            
            model_path = os.path.join(os.path.dirname(__file__), 'models', model_info['filename'])
            if not os.path.exists(model_path):
                # fallback check in current dir
                model_path = os.path.join('models', model_info['filename'])
                
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                new_model.load_state_dict(state_dict)
                self.model = new_model.to(self.device).eval()
                self.active_model_id = model_id
                print(f"✅ SUCCESS: Loaded '{model_info['filename']}'")
                return True
            else:
                print(f"❌ ERROR: Weights file '{model_path}' not found!")
                return False
        except Exception as e:
            print(f"❌ ERROR loading model {model_id}: {e}")
            return False

    def get_available_models(self):
        # Return list of models with current active status flag
        registry = []
        for mid, info in AVAILABLE_MODELS.items():
            model_data = info.copy()
            model_data['active'] = (mid == self.active_model_id)
            registry.append(model_data)
        return registry

    @torch.no_grad()
    def predict(self, image_path):
        if self.model is None:
            return {"error": "No model is currently loaded"}
            
        try:
            image = Image.open(image_path).convert('RGB')
            # Check if image is valid
            if image.size[0] < 10 or image.size[1] < 10:
                return {"error": "Image too small or corrupted"}
                
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            conf, idx = torch.max(probabilities, 0)
            
            result = {
                "class": self.classes[idx],
                "confidence": float(conf),
                "probabilities": {cl: float(prob) for cl, prob in zip(self.classes, probabilities)},
                "model_used": self.active_model_id
            }
            # Log prediction for debugging
            print(f"🔮 Prediction for {os.path.basename(image_path)} [{self.active_model_id}]: {result['class']} ({result['confidence']:.2%})")
            return result
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}

# Singleton instance
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = NSFWClassifier()
    return classifier
