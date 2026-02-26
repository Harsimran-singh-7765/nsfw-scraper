import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path

class NSFWClassifier:
    def __init__(self, model_path='best_model.pth', model_name='resnet50'):
        # Check architecture 
        print(f"🚀 Initializing NSFW Classifier (Model: {model_name})...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        
        # Initialize model architecture
        self.model = timm.create_model(model_name, pretrained=False, num_classes=len(self.classes))
        
        # Possible paths
        paths_to_check = [
            model_path,
            os.path.join('scripts', model_path),
            os.path.join(os.path.dirname(__file__), 'scripts', model_path)
        ]
        
        loaded = False
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    state_dict = torch.load(p, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"✅ SUCCESS: Loaded model weights from '{p}'")
                    print(f"📊 Model classes: {self.classes}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"❌ ERROR: Failed to load weights from {p}: {e}")
        
        if not loaded:
            print(f"⚠️ WARNING: Model weights file '{model_path}' NOT FOUND in any common locations!")
            print(f"   Make sure you uploaded 'best_model.pth' to the root or 'scripts/' folder.")
            print(f"   Currently using RANDOM WEIGHTS (Results will be incorrect!)")
            
        self.model = self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path):
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
                "probabilities": {cl: float(prob) for cl, prob in zip(self.classes, probabilities)}
            }
            # Log prediction for debugging
            print(f"🔮 Prediction for {os.path.basename(image_path)}: {result['class']} ({result['confidence']:.2%})")
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
