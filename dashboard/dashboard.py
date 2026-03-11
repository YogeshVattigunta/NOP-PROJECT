import streamlit as st
import pandas as pd
import altair as alt
import json
import torch
import os
import sys

# Ensure models module can be loaded from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import FraudDetectionModel

def main():
    # Configuration
    st.set_page_config(page_title="Fraud Optimizer Evaluation Dashboard", layout="wide")

    st.title("Variance-Stabilized RMSProp Optimization Dashboard")
    st.subheader("Numerical Optimization Project — Fraud Detection on Imbalanced Data")
    st.markdown("This dashboard compares Adagrad, RMSProp, and a proposed Variance-Stabilized RMSProp optimizer for fraud detection using adaptive gradient methods.")

    st.divider()

    # Helpers to load data
    @st.cache_data
    def load_results():
        results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results.json')
        if not os.path.exists(results_path):
            return None
        with open(results_path, 'r') as f:
            return json.load(f)

    @st.cache_resource
    def load_model(input_dim=30):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fraud_model.pth')
        model = FraudDetectionModel(input_dim)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    data = load_results()
    if not data:
        st.error("No results found. Please run main.py first to generate results.json")
        st.stop()

    df_results = pd.DataFrame(data['results'])
    best_optimizer_row = df_results.loc[df_results['F1'].idxmax()]
    histories = data['histories']

    # Calculate convergence speed and inject into results dataframe
    convergence_speeds = []
    for opt_row in df_results.itertuples():
        opt_name = opt_row.Optimizer
        if opt_name in histories:
            loss_hist = histories[opt_name]['train_loss']
            # convergence speed = (loss_epoch1 - loss_epochN) / epochs
            epochs = len(loss_hist)
            speed = (loss_hist[0] - loss_hist[-1]) / epochs
            convergence_speeds.append(speed)
        else:
            convergence_speeds.append(0.0)

    df_results['Convergence Speed'] = convergence_speeds

    # SECTION 1 - KEY PERFORMANCE METRICS
    st.header("1. Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{best_optimizer_row['Precision']:.4f}")
    col2.metric("Recall", f"{best_optimizer_row['Recall']:.4f}")
    col3.metric("F1 Score", f"{best_optimizer_row['F1']:.4f}")
    col4.metric("AUPRC", f"{best_optimizer_row['AUPRC']:.4f}")

    st.divider()

    # SECTION 2 - OPTIMIZER RANKING
    st.header("2. Optimizer Ranking")
    rank_col1, rank_col2 = st.columns([1, 2])
    with rank_col1:
        st.info(f"**Best Optimizer:**\n### {best_optimizer_row['Optimizer']}")
        st.markdown("This optimizer achieved the best balance of Precision and Recall on the imbalanced dataset, yielding the highest F1 score.")

    with rank_col2:
        def highlight_best(row):
            if row['Optimizer'] == best_optimizer_row['Optimizer']:
                return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df_results[['Optimizer', 'Precision', 'Recall', 'F1', 'AUPRC', 'Convergence Speed']].style.apply(highlight_best, axis=1), use_container_width=True)

    st.divider()

    # SECTION 3 - CORE MODEL PERFORMANCE
    st.header("3. Core Model Performance")
    core_col1, core_col2 = st.columns(2)

    with core_col1:
        st.subheader("Training Convergence")
        loss_data = []
        for opt, hist in histories.items():
            for epoch, loss in enumerate(hist['train_loss']):
                loss_data.append({"Epoch": epoch+1, "Loss": loss, "Optimizer": opt})
        df_loss = pd.DataFrame(loss_data)
        loss_chart = alt.Chart(df_loss).mark_line(point=True).encode(
            x='Epoch:Q', y='Loss:Q', color='Optimizer:N', tooltip=['Optimizer', 'Epoch', 'Loss']
        ).interactive().properties(height=350)
        st.altair_chart(loss_chart, use_container_width=True)

    with core_col2:
        st.subheader("Precision-Recall Curve")
        pr_data = []
        for opt, hist in histories.items():
            precisions = hist['final_precisions']
            recalls = hist['final_recalls']
            for p, r in zip(precisions, recalls):
                pr_data.append({"Recall": r, "Precision": p, "Optimizer": opt})
        df_pr = pd.DataFrame(pr_data)
        pr_chart = alt.Chart(df_pr).mark_line().encode(
            x=alt.X('Recall:Q', scale=alt.Scale(domain=[0,1])),
            y=alt.Y('Precision:Q', scale=alt.Scale(domain=[0,1])),
            color='Optimizer:N',
            tooltip=['Optimizer', 'Recall', 'Precision']
        ).interactive().properties(height=350)
        st.altair_chart(pr_chart, use_container_width=True)

    st.divider()

    # SECTION 4 - PERFORMANCE METRICS COMPARISON
    st.header("4. Performance Metrics Comparison")
    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        st.subheader("F1 Score Comparison")
        f1_chart = alt.Chart(df_results).mark_bar().encode(
            x='Optimizer:N', y='F1:Q', color='Optimizer:N', tooltip=['Optimizer', 'F1']
        ).interactive().properties(height=300)
        st.altair_chart(f1_chart, use_container_width=True)

    with perf_col2:
        st.subheader("AUPRC Comparison")
        auprc_chart = alt.Chart(df_results).mark_bar().encode(
            x='Optimizer:N', y='AUPRC:Q', color='Optimizer:N', tooltip=['Optimizer', 'AUPRC']
        ).interactive().properties(height=300)
        st.altair_chart(auprc_chart, use_container_width=True)

    with perf_col3:
        st.subheader("Recall Comparison")
        recall_chart = alt.Chart(df_results).mark_bar().encode(
            x='Optimizer:N', y='Recall:Q', color='Optimizer:N', tooltip=['Optimizer', 'Recall']
        ).interactive().properties(height=300)
        st.altair_chart(recall_chart, use_container_width=True)

    st.divider()

    # SECTION 5 - OPTIMIZATION BEHAVIOR
    st.header("5. Optimization Behavior")
    behav_col1, behav_col2 = st.columns(2)

    with behav_col1:
        st.subheader("Gradient Variance Stability")
        var_data = []
        for opt, hist in histories.items():
            if 'grad_variance' in hist:
                for epoch, gvar in enumerate(hist['grad_variance']):
                    var_data.append({"Epoch": epoch+1, "Variance": gvar, "Optimizer": opt})
        if var_data:
            df_var = pd.DataFrame(var_data)
            var_chart = alt.Chart(df_var).mark_line(point=True).encode(
                x='Epoch:Q', y='Variance:Q', color='Optimizer:N', tooltip=['Optimizer', 'Epoch', 'Variance']
            ).interactive().properties(height=350)
            st.altair_chart(var_chart, use_container_width=True)

    with behav_col2:
        st.subheader("Convergence Speed Comparison")
        conv_chart = alt.Chart(df_results).mark_bar().encode(
            x='Optimizer:N',
            y='Convergence Speed:Q',
            color='Optimizer:N',
            tooltip=['Optimizer', 'Convergence Speed']
        ).interactive().properties(height=350)
        st.altair_chart(conv_chart, use_container_width=True)

    st.divider()

    # SECTION 6 - FRAUD PREDICTION SIMULATOR
    st.header("6. Interactive Fraud Prediction Simulator")
    st.markdown("Enter 30 transaction feature values to test the Variance RMSProp trained model.")

    features = []
    grid_cols = st.columns(5)
    for i in range(30):
        with grid_cols[i % 5]:
            val = st.number_input(f"Feature {i+1}", value=0.0, step=0.1, key=f"feat_{i}")
            features.append(val)

    if st.button("Predict"):
        model = load_model(input_dim=30)
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            # We need to apply sigmoid because network no longer has Sigmoid
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
        is_fraud = probability >= 0.5
        st.markdown(f"### Fraud Probability: {probability:.4f}")
        if is_fraud:
            st.error("Prediction: FRAUD")
        else:
            st.success("Prediction: NORMAL")

    st.divider()

    # SECTION 7 - EXPERIMENTAL FINDINGS
    st.header("7. Experimental Findings")
    st.markdown("""
    * RMSProp achieved the highest F1 score.
    * VarianceRMSProp demonstrated smoother gradient variance behavior.
    * Hyperparameter tuning improved AUPRC performance.
    * Variance stabilization produced more consistent convergence patterns.
    """)

    st.divider()

    # SECTION 8 - BEST VARIANCERMSPROP CONFIGURATION
    st.header("8. Best VarianceRMSProp Configuration")
    st.info("""
    These hyperparameters were discovered using a grid search iteration and achieved the best balanced performance:

    * `epochs` = 25
    * `learning_rate` = 0.0007
    * `alpha` = 0.03
    * `beta` = 0.9
    * `epsilon` = 1e-8
    """)

if __name__ == "__main__":
    main()
