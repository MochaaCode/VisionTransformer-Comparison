# main.py - Vision Transformer Comparison (ViT vs Swin Transformer)
# Jalankan di Google Colab dengan GPU enabled

import torch
import torchvision
import timm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchinfo import summary
from tqdm import tqdm
import time
import pandas as pd
import os
from google.colab import drive

# Mount Google Drive (simpan hasil di sini)
drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/VisionTransformer-Comparison"
os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)

print("✅ Setup complete! Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 1. DATASET PREPARATION (CIFAR-10)
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("✅ Dataset CIFAR-10 loaded successfully!")
print(f"Training set: {len(trainset)} images, Test set: {len(testset)} images")

# 2. MODEL IMPLEMENTATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT Base
model_vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
for param in model_vit.parameters():
    param.requires_grad = False
for param in model_vit.head.parameters():
    param.requires_grad = True
model_vit = model_vit.to(device)

# Swin Base
model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=10)
for param in model_swin.parameters():
    param.requires_grad = False
for param in model_swin.head.parameters():
    param.requires_grad = True
model_swin = model_swin.to(device)

print("✅ Models initialized successfully!")

# 3. TRAINING FUNCTION
def train_model(model, trainloader, testloader, epochs=5, lr=1e-3, model_name="model"):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"\n🚀 Starting training for {model_name}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        history['train_loss'].append(running_loss/len(trainloader))
        history['val_loss'].append(val_loss/len(testloader))
        history['val_acc'].append(acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss/len(testloader):.4f}, Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{BASE_DIR}/models/{model_name}_best.pth")
            print(f"🎉 New best accuracy: {best_acc:.2f}% - Model saved!")
        
        scheduler.step()
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.legend()
    
    plt.savefig(f"{BASE_DIR}/results/training_curves_{model_name}.png")
    plt.close()
    
    return history, best_acc

# 4. TRAINING EXECUTION
print("\n" + "="*50)
print("Starting ViT Training")
print("="*50)
history_vit, best_acc_vit = train_model(model_vit, trainloader, testloader, epochs=5, lr=1e-3, model_name="vit")

print("\n" + "="*50)
print("Starting Swin Training")
print("="*50)
history_swin, best_acc_swin = train_model(model_swin, trainloader, testloader, epochs=5, lr=1e-3, model_name="swin")

# 5. EVALUATION FUNCTION
def evaluate_model(model, testloader, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Warm-up
    print(f"\n🔥 Warm-up for {model_name} inference...")
    for _ in range(10):
        with torch.no_grad():
            inputs, _ = next(iter(testloader))
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Actual inference time measurement
    print(f"⏱️  Measuring inference time for {model_name}...")
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc=f"Inference {model_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{BASE_DIR}/results/confusion_matrix_{model_name}.png")
    plt.close()
    
    # Calculate inference metrics
    avg_time_per_image = total_time / len(testset) * 1000  # ms
    throughput = len(testset) / total_time  # images/sec
    
    print(f"\n✅ {model_name} Evaluation Complete!")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Avg Inference Time: {avg_time_per_image:.2f} ms/image")
    print(f"   Throughput: {throughput:.2f} images/sec")
    
    return {
        'accuracy': accuracy,
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'avg_time_per_image': avg_time_per_image,
        'throughput': throughput,
        'confusion_matrix': cm
    }

# 6. EVALUATION EXECUTION
print("\n" + "="*50)
print("Evaluating ViT Model")
print("="*50)
metrics_vit = evaluate_model(model_vit, testloader, "ViT")

print("\n" + "="*50)
print("Evaluating Swin Model")
print("="*50)
metrics_swin = evaluate_model(model_swin, testloader, "Swin")

# 7. FINAL COMPARISON
print("\n" + "="*60)
print("📊 FINAL COMPARISON RESULTS")
print("="*60)

comparison_data = {
    'Metric': ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score', 'Avg Inference Time (ms)', 'Throughput (img/s)'],
    'ViT': [
        f"{metrics_vit['accuracy']*100:.2f}",
        f"{metrics_vit['precision']:.3f}",
        f"{metrics_vit['recall']:.3f}",
        f"{metrics_vit['f1']:.3f}",
        f"{metrics_vit['avg_time_per_image']:.2f}",
        f"{metrics_vit['throughput']:.2f}"
    ],
    'Swin Transformer': [
        f"{metrics_swin['accuracy']*100:.2f}",
        f"{metrics_swin['precision']:.3f}",
        f"{metrics_swin['recall']:.3f}",
        f"{metrics_swin['f1']:.3f}",
        f"{metrics_swin['avg_time_per_image']:.2f}",
        f"{metrics_swin['throughput']:.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison to CSV
comparison_df.to_csv(f"{BASE_DIR}/results/metrics_comparison.csv", index=False)

# 8. PARAMETER COUNT
print("\n" + "="*50)
print("📏 MODEL SIZE COMPARISON")
print("="*50)

def count_parameters(model, model_name):
    model_info = summary(model, input_size=(1, 3, 224, 224), verbose=0)
    total_params = model_info.total_params
    trainable_params = model_info.trainable_params
    non_trainable_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter
    
    print(f"{model_name} Parameters:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {non_trainable_params:,}")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb
    }

params_vit = count_parameters(model_vit, "ViT")
params_swin = count_parameters(model_swin, "Swin Transformer")

# Save parameter info
param_data = {
    'Model': ['ViT', 'Swin Transformer'],
    'Total Parameters': [params_vit['total_params'], params_swin['total_params']],
    'Trainable Parameters': [params_vit['trainable_params'], params_swin['trainable_params']],
    'Model Size (MB)': [params_vit['model_size_mb'], params_swin['model_size_mb']]
}
pd.DataFrame(param_data).to_csv(f"{BASE_DIR}/results/parameters_comparison.csv", index=False)

print("\n" + "="*60)
print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"💡 Best Accuracy: ViT = {best_acc_vit:.2f}%, Swin = {best_acc_swin:.2f}%")
print(f"📁 Results saved to: {BASE_DIR}/results/")
print(f"💾 Models saved to: {BASE_DIR}/models/")
print("✅ All deliverables ready for GitHub repository!")