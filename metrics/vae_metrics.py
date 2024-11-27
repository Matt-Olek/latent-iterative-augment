import torch
from sklearn.metrics import f1_score
from utils import to_default_device


def validate_vae(model, test_data):
    """Validate VAE model performance"""
    model.eval()
    with torch.no_grad():
        x, y = test_data
        x = to_default_device(x)
        y = to_default_device(y)
        x_hat, mu, log_var, _ = model(x)
        y_pred = model.classifier(mu).argmax(dim=1)
        accuracy = (y_pred == y.argmax(dim=1)).float().mean().item()
        f1 = f1_score(
            y.argmax(dim=1).cpu().numpy(), y_pred.cpu().numpy(), average="macro"
        )
        return accuracy, f1
