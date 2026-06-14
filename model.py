import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Attention Mechanism)
    Squeezes spatial dimension using Global Average Pooling,
    obtains channel-wise dependencies, and scales the input.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Compute channel-wise attention weights
        w = self.fc(x).view(b, c, 1, 1)
        # Re-weight the features map
        return x * w

class SkinDiseaseCNN(nn.Module):
    """
    Enhanced Skin Disease Classification Network.
    Uses MobileNetV2 as a Transfer Learning backbone,
    appends a Squeeze-and-Excitation (SE) Block for attention,
    and classifies into N skin disease categories.
    """
    def __init__(self, num_classes=7):
        super(SkinDiseaseCNN, self).__init__()
        # Load pre-trained MobileNetV2
        # Weights are initialized from ImageNet to leverage transfer learning
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Extract the features part
        self.features = self.backbone.features
        
        # Attention Mechanism: MobileNetV2 features output channel size is 1280
        self.attention = SEBlock(channels=1280, reduction=16)
        
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final Classifier
        # MobileNetV2 dropout is 0.2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # 1. Feature extraction using backbone (transfer learning)
        x = self.features(x)
        
        # 2. Apply attention mechanism (SE Block)
        x = self.attention(x)
        
        # 3. Pooling and flattening
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # 4. Final classification
        x = self.classifier(x)
        return x

def get_model(num_classes=7):
    """
    Instantiates and returns the SkinDiseaseCNN model.
    """
    return SkinDiseaseCNN(num_classes=num_classes)
