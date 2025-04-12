# improved_v3_run_final.py
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import pickle
import os

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Datasets ---
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Model ---
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 10)

# Freeze all layers, unfreeze only the last N blocks + classifier
unfreeze_blocks = 9
for param in model.parameters():
    param.requires_grad = False
for param in model._blocks[-unfreeze_blocks:].parameters():
    param.requires_grad = True
for param in model._fc.parameters():
    param.requires_grad = True

model = model.to(device)

# --- Loss, Optimizer, Scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# --- Training & Evaluation ---
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0
    loop = tqdm(loader)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Training Loop with Early Stopping ---
v3_accs = []
EPOCHS = 20
early_stop_patience = 3
best_acc = 0
best_epoch = 0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    acc = evaluate(model, test_loader)
    v3_accs.append(acc)
    scheduler.step()
    print(f"Train Loss: {train_loss:.4f} | Test Accuracy: {acc:.2f}%")

    # Track best
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model_v3.pth")
        print("🔥 Best model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    # Early stopping
    if epochs_no_improve >= early_stop_patience:
        print(f"\n⏹️ Early stopping at epoch {epoch}. Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        break

# --- Save accuracy history ---
with open("v3_accs.pkl", "wb") as f:
    pickle.dump(v3_accs, f)

print(f"\n✅ Final Best Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")