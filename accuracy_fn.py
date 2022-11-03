import torch

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions."""

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
