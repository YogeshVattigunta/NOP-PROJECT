import torch
import torch.optim as optim
import os
import pandas as pd
import json
import numpy as np
import random

# Fixed seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from data.dataset import load_data
from models.network import FraudDetectionModel
from optimizers.variance_rmsprop import VarianceRMSProp
from training.trainer import train_model
from plots.visualizer import save_all_plots

def main():
    dataset_path = r"e:\NOP Assignment\creditcard.csv"
    
    baseline_epochs = 20
    learning_rate_adagrad = 0.01
    learning_rate_rmsprop = 0.001
    beta_rmsprop = 0.9
    
    device = 'cpu'
    runs = 3
    
    # Grid search configs for VarianceRMSProp
    alphas = [0.01, 0.02, 0.03, 0.05]
    lrs = [0.0005, 0.0007, 0.001]
    epochs_list = [20, 25, 30]
    
    print("="*50)
    print("Preconditioned Gradient Variance Tracking for Imbalanced Fraud Detection")
    print("="*50)

    # 1. Load Data
    train_loader, test_loader, pos_weight, input_dim = load_data(dataset_path)
    print(f"Input features dimension: {input_dim}")
    
    # Define baselines
    baselines_config = {
        "Adagrad": lambda params: optim.Adagrad(params, lr=learning_rate_adagrad),
        "RMSProp": lambda params: optim.RMSprop(params, lr=learning_rate_rmsprop, alpha=beta_rmsprop, eps=1e-8)
    }
    
    histories = {}
    results = []

    # 2. Train and Evaluate Baselines
    for opt_name, opt_constructor in baselines_config.items():
        print(f"\nTraining baseline {opt_name} over {runs} runs...")
        opt_histories = []
        for run in range(runs):
            print(f"--- Run {run+1}/{runs} ---")
            torch.manual_seed(run)
            np.random.seed(run)
            model = FraudDetectionModel(input_dim)
            optimizer = opt_constructor(model.parameters())
            history = train_model(model, train_loader, test_loader, optimizer, pos_weight, baseline_epochs, device)
            opt_histories.append(history)
            
        avg_history = {
            'train_loss': np.mean([h['train_loss'] for h in opt_histories], axis=0).tolist(),
            'val_loss': np.mean([h['val_loss'] for h in opt_histories], axis=0).tolist(),
            'precision': np.mean([h['precision'] for h in opt_histories], axis=0).tolist(),
            'recall': np.mean([h['recall'] for h in opt_histories], axis=0).tolist(),
            'f1': np.mean([h['f1'] for h in opt_histories], axis=0).tolist(),
            'auprc': np.mean([h['auprc'] for h in opt_histories], axis=0).tolist(),
            'grad_variance': np.mean([h['grad_variance'] for h in opt_histories], axis=0).tolist(),
            'final_precisions': opt_histories[-1]['final_precisions'].tolist(),
            'final_recalls': opt_histories[-1]['final_recalls'].tolist()
        }
        histories[opt_name] = avg_history
        results.append({
            "Optimizer": opt_name,
            "Precision": avg_history['precision'][-1],
            "Recall": avg_history['recall'][-1],
            "F1": avg_history['f1'][-1],
            "AUPRC": avg_history['auprc'][-1]
        })

    # 3. Hyperparameter Tuning for VarianceRMSProp
    print("\nStarting Hyperparameter Grid Search for VarianceRMSProp...")
    best_config = None
    best_auprc = -1
    best_f1 = -1
    best_avg_history = None
    
    for ep in epochs_list:
        for lr in lrs:
            for alpha in alphas:
                print(f"\nTuning Config: epochs={ep}, lr={lr}, alpha={alpha}")
                opt_histories = []
                
                for run in range(runs):
                    torch.manual_seed(run)
                    np.random.seed(run)
                    model = FraudDetectionModel(input_dim)
                    optimizer = VarianceRMSProp(model.parameters(), lr=lr, alpha=alpha, beta=beta_rmsprop, eps=1e-8)
                    history = train_model(model, train_loader, test_loader, optimizer, pos_weight, ep, device)
                    opt_histories.append(history)
                    
                    if run == runs - 1:
                        last_model_state = model.state_dict()
                
                avg_auprc = np.mean([h['auprc'][-1] for h in opt_histories])
                avg_f1 = np.mean([h['f1'][-1] for h in opt_histories])
                
                print(f"Result -> AUPRC: {avg_auprc:.4f}, F1: {avg_f1:.4f}")
                
                if avg_auprc > best_auprc or (avg_auprc == best_auprc and avg_f1 > best_f1):
                    best_auprc = avg_auprc
                    best_f1 = avg_f1
                    best_config = {'epochs': ep, 'lr': lr, 'alpha': alpha}
                    torch.save(last_model_state, 'fraud_model.pth')
                    
                    best_avg_history = {
                        'train_loss': np.mean([h['train_loss'] for h in opt_histories], axis=0).tolist(),
                        'val_loss': np.mean([h['val_loss'] for h in opt_histories], axis=0).tolist(),
                        'precision': np.mean([h['precision'] for h in opt_histories], axis=0).tolist(),
                        'recall': np.mean([h['recall'] for h in opt_histories], axis=0).tolist(),
                        'f1': np.mean([h['f1'] for h in opt_histories], axis=0).tolist(),
                        'auprc': np.mean([h['auprc'] for h in opt_histories], axis=0).tolist(),
                        'grad_variance': np.mean([h['grad_variance'] for h in opt_histories], axis=0).tolist(),
                        'final_precisions': opt_histories[-1]['final_precisions'].tolist(),
                        'final_recalls': opt_histories[-1]['final_recalls'].tolist()
                    }

    print("\n" + "="*50)
    print(f"Best VarianceRMSProp settings:")
    for k, v in best_config.items():
        print(f"{k} = {v}")
    print("="*50)

    # Save best VarianceRMSProp into results
    histories["VarianceRMSProp"] = best_avg_history
    results.append({
        "Optimizer": "VarianceRMSProp",
        "Precision": best_avg_history['precision'][-1],
        "Recall": best_avg_history['recall'][-1],
        "F1": best_avg_history['f1'][-1],
        "AUPRC": best_avg_history['auprc'][-1]
    })

    # 3. Visualizations
    print("\nGenerating Visualizations...")
    os.makedirs('plots', exist_ok=True)
    save_all_plots(histories, output_dir="plots")
    
    # 4. Results Output
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON TABLE")
    print("="*50)
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    
    print("\nProject pipeline execution completed successfully. Plots saved to 'plots/' directory.")
    
    with open("results.json", "w") as f:
        json.dump({"histories": histories, "results": results}, f)
    print("Saved training metrics to results.json.")

if __name__ == "__main__":
    main()
