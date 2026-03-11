import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(csv_path, batch_size=256, test_size=0.2, random_state=42):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.sample(50000, random_state=42)
    
    # Check if 'Class' exists
    if 'Class' not in df.columns:
        raise ValueError("Dataset does not contain 'Class' column")
        
    X = df.drop(['Class'], axis=1).values
    y = df['Class'].values
    
    # Handle missing values if any
    if np.isnan(X).any():
        print("Handling missing values...")
        X = np.nan_to_num(X)
        
    # Standardize features
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset (stratified due to heavy imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Calculate class weights for BCEWithLogitsLoss based on training set
    fraud_weight = len(y_train) / (2 * sum(y_train))
    normal_weight = len(y_train) / (2 * (len(y_train) - sum(y_train)))
    pos_weight = fraud_weight / normal_weight
    
    print(f"Class distribution - Train 0: {len(y_train) - sum(y_train)}, 1: {sum(y_train)}")
    print(f"Computed positive class weight: {pos_weight:.4f}")
    
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader, pos_weight, X.shape[1]
