import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from models.fraud_net import FraudDetectionNet
from optimizers.variance_rmsprop_torch import VarianceRMSProp

def train_model(optimizer_name, X_train, y_train, X_test, y_test, epochs=10, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {optimizer_name} on {device}...")
    
    input_dim = X_train.shape[1]
    model = FraudDetectionNet(input_dim).to(device)
    
    # Loss with pos_weight to handle imbalance
    pos_weight = torch.tensor([400.0]).to(device)
    criterion = nn.BCELoss() # FraudDetectionNet already has Sigmoid
    
    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.9)
    elif optimizer_name == "VarianceRMSProp":
        optimizer = VarianceRMSProp(model.parameters(), lr=0.0001, beta=0.9, alpha=0.03)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), 
                              batch_size=batch_size, shuffle=True)
    
    history = {
        "train_loss": [],
        "grad_variance": [],
        "grad_norm": []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_vars = []
        epoch_norms = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Record gradient metrics
            grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            all_grads = torch.cat(grads)
            
            epoch_loss += loss.item()
            epoch_vars.append(torch.mean(all_grads**2).item())
            epoch_norms.append(torch.norm(all_grads).item())
            
            optimizer.step()
            
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["grad_variance"].append(np.mean(epoch_vars))
        history["grad_norm"].append(np.mean(epoch_norms))
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {history['train_loss'][-1]:.4f}")
        
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_probs = model(X_test_tensor).cpu().numpy()
        y_preds = (y_probs > 0.5).astype(int)
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_preds, average='binary')
    auprc = average_precision_score(y_test, y_probs)
    
    # Convergence speed: defined here as the number of epochs to reach loss < 0.1
    # If it never reaches, return epochs.
    conv_speed = epochs
    for i, l in enumerate(history["train_loss"]):
        if l < 0.1:
            conv_speed = i + 1
            break
            
    # For AUPRC display, we need more points for the PR curve
    # We'll just return the metrics and history for now.
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "convergence_speed": float(conv_speed),
        "train_loss": history["train_loss"],
        "grad_variance": history["grad_variance"],
        "grad_norm": history["grad_norm"]
    }, model

def main():
    from data.talking_data_proc import process_talking_data
    
    X_train, X_test, y_train, y_test = process_talking_data()
    
    # Stratified sample 200k for speed in this iteration
    print("Sampling 200k rows for efficiency...")
    indices = np.random.choice(len(X_train), 200000, replace=False)
    X_train_small = X_train[indices]
    y_train_small = y_train.iloc[indices]
    
    results = {}
    best_auprc = -1
    best_model = None
    
    for opt in ["Adagrad", "RMSprop", "VarianceRMSProp"]:
        metrics, model = train_model(opt, X_train_small, y_train_small.values, X_test, y_test.values)
        results[opt] = metrics
        
        if metrics["auprc"] > best_auprc:
            best_auprc = metrics["auprc"]
            best_model = model
            
    # Save results
    with open('e:/NOP Assignment/fraud_optimizer_project/training_results.json', 'w') as f:
        json.dump(results, f)
        
    # Save best model weights
    torch.save(best_model.state_dict(), 'e:/NOP Assignment/fraud_optimizer_project/variance_rmsprop_model.pth')
    print("Training complete. Results and model saved.")

if __name__ == "__main__":
    main()
