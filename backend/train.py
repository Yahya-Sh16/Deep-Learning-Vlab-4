import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import asyncio

from model import CIFAR10_CNN

# Use a subset so the live simulation doesn't take forever
SUBSET_TRAIN_SIZE = 10000   # Increased from 1000 to 10000 (20% of training data)
SUBSET_TEST_SIZE = 2000     # Increased from 200 to 2000 (20% of test data)

# Cache the dataset in memory to avoid slow disk read during simulation run
_trainset = None
_testset = None

def get_datasets():
    global _trainset, _testset
    if _trainset is None:
        # Very basic transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Subsetting
        _trainset = torch.utils.data.Subset(full_trainset, range(SUBSET_TRAIN_SIZE))
        _testset = torch.utils.data.Subset(full_testset, range(SUBSET_TEST_SIZE))
        
    return _trainset, _testset

# Define CIFAR-10 classes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

async def run_simulation(config):
    # Retrieve hyperparameters
    model_capacity = config.get("model_capacity", "Medium")
    dropout_rate = config.get("dropout", 0.0)
    solver = config.get("solver", "adam").lower()
    alpha = config.get("alpha", 0.0001)           # Weight Decay
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 30)  # Increased from 15 to 30 for better convergence
    lr = config.get("learning_rate", 0.001)

    yield json.dumps({'type': 'log', 'message': f'Initializing {model_capacity} Capacity CNN on CIFAR-10'})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yield json.dumps({'type': 'log', 'message': f'Using device: {device}'})
    
    trainset, testset = get_datasets()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    model = CIFAR10_CNN(capacity=model_capacity, dropout_rate=dropout_rate).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if solver == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=alpha)
    else:
        # Default Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
        
    yield json.dumps({'type': 'log', 'message': f'Starting training for {epochs} epochs...'})

    loss_curve = []
    acc_curve = []
    val_loss_curve = []
    val_acc_curve = []
    
    start_time = time.time()
    
    # Allow async interruption (important for FastAPI generator)
    await asyncio.sleep(0)
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Yield control back momentarily and provide batch-level logs
            if (i + 1) % 10 == 0:
                yield json.dumps({'type': 'log', 'message': f'  [Epoch {epoch}] Training batch {i+1}...'})
                await asyncio.sleep(0.01)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        loss_curve.append(train_loss)
        acc_curve.append(train_acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)
        
        # Format strings safely
        epoch_str = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%"
        
        yield json.dumps({'type': 'log', 'message': epoch_str})
        
        epoch_data = {
            'type': 'epoch',
            'epoch': epoch,
            'total_epochs': epochs,
            'loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'accuracy': train_acc,
            'val_accuracy': val_acc,
            'loss_curve': loss_curve,
            'val_loss_curve': val_loss_curve,
            'accuracy_curve': acc_curve,
            'val_accuracy_curve': val_acc_curve
        }
        yield json.dumps(epoch_data)
        
        # Crucial for returning control to the event loop so SSE flushes to client
        await asyncio.sleep(0.01)

    training_time = time.time() - start_time
    yield json.dumps({'type': 'log', 'message': f'Training completed in {training_time:.1f}s'})

    # Evaluation for Confusion Matrix and Sandbox Predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    # Store some sandbox samples specifically (using first batch of testset)
    sandbox_samples = []
    sample_count = 10
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Bring back to CPU for numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Save predictions for frontend sandbox (just get first batch)
            if len(sandbox_samples) < sample_count:
                cpu_inputs = inputs.cpu().numpy()
                for i in range(len(cpu_inputs)):
                    if len(sandbox_samples) >= sample_count:
                        break
                    
                    # Convert to flat array 3072 for frontend canvas
                    # CPU input shape is (3, 32, 32). Flatten gives R..R, G..G, B..B as expected.
                    flat_img = cpu_inputs[i].flatten().tolist()
                    sandbox_samples.append({
                        "image": flat_img,
                        "actual": CLASSES[labels[i].item()],
                        "predicted": CLASSES[predicted[i].item()]
                    })
                    
    cm = confusion_matrix(all_labels, all_preds)
    
    result_data = {
        'type': 'result',
        'confusion_matrix': cm.tolist(),
        'class_labels': CLASSES,
        'accuracy': val_acc_curve[-1],
        'training_time': round(training_time, 1),
        'predictions': sandbox_samples
    }
    
    yield json.dumps(result_data)
