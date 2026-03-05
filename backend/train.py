import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

def train_model():
    # 1. Setup Hardware Acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Your exact dataset path from filelist.txt
    data_dir = r"E:\CV Project Files\Github Clone\ClinicalVisionDFU\USE CASE - 02\DFU\Patches"
    
    # 3. Medical Data Augmentation (The secret to 95%+ accuracy)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30), # Simulates different camera angles
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulates different room lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Load the Dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    print(f"Detected classes: {full_dataset.class_to_idx}")
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 5. Initialize Pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 Classes (Normal / Abnormal)
    model = model.to(device)

    # 6. Optimizer & Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 7. Training Loop
    num_epochs = 15
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # Save the highest accuracy model directly to your backend folder
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_path = os.path.join(os.path.dirname(__file__), "model", "dfu_model.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"🏆 Best model saved to {save_path}!")

    print(f'\nFinished! Highest Validation Accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    train_model()