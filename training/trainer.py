import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from evaluation.metrics import compute_metrics

def train_model(model, train_loader, val_loader, optimizer, pos_weight, epochs=20, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    history = {
        'train_loss': [], 'val_loss': [],
        'precision': [], 'recall': [], 'f1': [], 'auprc': [],
        'grad_variance': [],
        'final_precisions': None, 'final_recalls': None
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        var_list = []
        for p in model.parameters():
            if p.grad is not None:
                # Approximate variance of gradients in the last batch
                # Avoid NaN if only 1 element
                if p.grad.numel() > 1:
                    var_list.append(torch.var(p.grad.view(-1)).item())
        epoch_grad_variance = float(np.mean(var_list)) if var_list else 0.0
        history['grad_variance'].append(epoch_grad_variance)
        
        val_loss, val_targets, val_probs = evaluate_model(model, val_loader, criterion, pos_weight, device)
        history['val_loss'].append(val_loss)
        
        precision, recall, f1, auprc, precisions, recalls = compute_metrics(val_targets, val_probs)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        history['auprc'].append(auprc)
        
        if epoch == epochs - 1:
            history['final_precisions'] = precisions
            history['final_recalls'] = recalls
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | AUPRC: {auprc:.4f}")
        
    return history

def evaluate_model(model, val_loader, criterion, pos_weight, device):
    model.eval()
    val_loss = 0.0
    all_targets, all_probs = [], []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            
    val_loss /= len(val_loader.dataset)
    return val_loss, np.array(all_targets), np.array(all_probs)
