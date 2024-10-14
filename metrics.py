from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch

def compute_metrics(outputs, targets, average = None):

    # convert logits to class distributions
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()

    accuracy = (targets == predicted).mean()
    f1 = f1_score(targets, predicted, average=average)
    precision = precision_score(targets, predicted, average=average)
    recall = recall_score(targets, predicted, average=average)

    conf_matrix = confusion_matrix(targets, predicted)

    if outputs.shape[1] == 2: # Check for binary classification
        roc_auc = roc_auc_score(targets, outputs[:, 1])
    elif outputs.shape[1] > 2:
        roc_auc = roc_auc_score(targets, outputs.cpu().numpy(), multi_class='ovr')

    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc
    }

def plot_roc_curve(outputs, targets):
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    probabilities = outputs[:, 1]

    fpr, tpr, threshold = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color = 'r', linewidth = 2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.grid(True)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    plt.savefig('roc_curve.png')
    plt.savefig('roc_curve.svg', format='svg')
    plt.show()
    

