import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_loss(histories, save_path="plots/training_loss.png"):
    plt.figure(figsize=(10, 6))
    for opt_name, history in histories.items():
        plt.plot(history['train_loss'], label=f"{opt_name} Train")
        plt.plot(history['val_loss'], '--', label=f"{opt_name} Val")
    plt.title("Training and Validation Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(histories, save_path="plots/pr_curve.png"):
    plt.figure(figsize=(10, 6))
    for opt_name, history in histories.items():
        precisions = history['final_precisions']
        recalls = history['final_recalls']
        auprc = history['auprc'][-1]
        plt.plot(recalls, precisions, label=f"{opt_name} (AUPRC={auprc:.4f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_convergence(histories, save_path="plots/convergence.png"):
    plt.figure(figsize=(10, 6))
    for opt_name, history in histories.items():
        plt.plot(history['f1'], label=opt_name)
    plt.title("Convergence Comparison (F1-Score over Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_f1_bar_chart(histories, save_path="plots/f1_comparison.png"):
    plt.figure(figsize=(8, 6))
    opt_names = list(histories.keys())
    f1_scores = [histories[opt]['f1'][-1] for opt in opt_names]
    
    bars = plt.bar(opt_names, f1_scores, color=['blue', 'orange', 'green'])
    plt.title("Final F1-Score Comparison")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_all_plots(histories, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plot_training_loss(histories, os.path.join(output_dir, "training_loss.png"))
    plot_pr_curve(histories, os.path.join(output_dir, "pr_curve.png"))
    plot_convergence(histories, os.path.join(output_dir, "convergence.png"))
    plot_f1_bar_chart(histories, os.path.join(output_dir, "f1_comparison.png"))
