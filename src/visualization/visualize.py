import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y: pd.Series, save_path: str = None) -> None:
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Legit, 1: Fraud)')
    plt.ylabel('Count')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()