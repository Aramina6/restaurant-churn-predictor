import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15, save_path='results/feature_importance.png'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features')
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
