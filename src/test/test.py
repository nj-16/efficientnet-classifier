import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import urllib.request

# Step 1: Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Step 2: Load your dataset
dataset = datasets.ImageFolder('images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 3: Load pretrained EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Step 4: Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = []
with urllib.request.urlopen(url) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f]

# Step 5: Get mapping for your class labels
class_names = dataset.classes  # ['cats', 'dogs']

# Step 6: Run through the data
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        pred_labels = [imagenet_classes[pred] for pred in predictions]
        true_label_names = [class_names[label] for label in labels]

        print(f"True Labels (Index): {labels.tolist()}")
        print(f"True Labels (Name): {true_label_names}")
        print(f"Predicted Indices: {predictions.tolist()}")
        print(f"Predicted Labels: {pred_labels}")
