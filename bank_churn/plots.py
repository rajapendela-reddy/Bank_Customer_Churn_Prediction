"""Training-only exploratory plots and final selected-model plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import TARGET


def plot_training_data(training, output):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.countplot(data=training, x=TARGET, ax=axes[0])
    axes[0].set_title("Training: stayed (0) vs exited (1)")
    sns.boxplot(data=training, x=TARGET, y="Age", ax=axes[1])
    axes[1].set_title("Training: age by churn")
    training.groupby("Geography")[TARGET].mean().plot.bar(ax=axes[2])
    axes[2].set_title("Training: churn rate by geography")
    axes[2].set_ylabel("Churn rate")
    fig.tight_layout(); fig.savefig(output / "training_eda.png", dpi=150); plt.close(fig)


def plot_test(y_true, probability, model_name, output):
    prediction = (probability >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, prediction,
        display_labels=["Stayed", "Exited"], ax=axes[0], colorbar=False)
    axes[0].set_title(f"{model_name}: test confusion matrix")
    RocCurveDisplay.from_predictions(y_true, probability, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_title(f"{model_name}: test ROC curve")
    fig.tight_layout(); fig.savefig(output / "selected_model_test_plots.png", dpi=150); plt.close(fig)


def save_importance(pipeline, output):
    names = pipeline.named_steps["preprocessing"].named_steps["columns"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names,
        "importance": pipeline.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output / "random_forest_feature_importance.csv", index=False)
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest: feature importance")
    fig.tight_layout(); fig.savefig(output / "random_forest_feature_importance.png", dpi=150); plt.close(fig)
