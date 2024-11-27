import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import tqdm
from utils import to_default_device, get_model_path
from visualization import plot_latent_space_viz, plot_latent_space_neighbors


class VAETrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("LEARNING_RATE", 1e-4),
            weight_decay=1e-6,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=100
        )
        self.loss_function = nn.CrossEntropyLoss()

    def contrastive_loss(self, z, labels, margin=1.0):
        pairwise_distances = torch.cdist(z, z, p=2)
        labels = labels.argmax(dim=1)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask

        positive_loss = positive_mask * pairwise_distances.pow(2)
        negative_loss = negative_mask * nn.functional.relu(
            margin - pairwise_distances
        ).pow(2)

        return (positive_loss + negative_loss).mean()

    def calculate_kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0
        total_class_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total = 0

        for batch, target in data_loader:
            self.optimizer.zero_grad()
            x = to_default_device(batch)
            target = to_default_device(target)

            x_hat, mu, log_var, y_pred = self.model(x)

            # Calculate losses
            kl_div = self.calculate_kl_divergence(mu, log_var)
            contrastive_loss = self.contrastive_loss(mu, target)
            class_loss = self.loss_function(y_pred, target.argmax(dim=1))
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")

            # Combine losses
            loss = contrastive_loss + recon_loss + class_loss + kl_div

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_class_loss += class_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            correct += torch.sum(
                torch.argmax(y_pred, dim=1) == torch.argmax(target, dim=1)
            ).item()
            total += len(target)

        # Calculate averages
        num_batches = len(data_loader)
        metrics = {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_div": total_kl_div / num_batches,
            "class_loss": total_class_loss / num_batches,
            "contrastive_loss": total_contrastive_loss / num_batches,
            "accuracy": correct / total,
        }

        return metrics

    def train(self, train_loader, test_dataset, nb_classes, logs, name="vae"):
        if self.config["WANDB"]:
            wandb.init(
                project=self.config["WANDB_PROJECT"],
                config=self.config,
                tags=["train", self.config["DATASET"], name],
                name=f'{self.config["DATASET"]} {name}',
            )
            wandb.watch(self.model)

        best_metrics = {"acc": 0, "f1": 0}
        early_stop_counter = 0

        for epoch in tqdm.tqdm(range(self.config["VAE_NUM_EPOCHS"])):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            from metrics import validate_vae

            acc, f1 = validate_vae(self.model, test_dataset)

            # Log metrics
            if self.config["WANDB"]:
                wandb.log(
                    {
                        **train_metrics,
                        "test_accuracy": acc,
                        "test_f1": f1,
                        "lr": float(self.scheduler.get_last_lr()[0]),
                    }
                )

            # Visualization
            if epoch % 100 == 0 and self.config["AUGMENT_PLOT"]:
                plot_latent_space_viz(
                    self.model,
                    train_loader,
                    test_dataset,
                    num_classes=nb_classes,
                    type="3d",
                    id=epoch,
                )
                plot_latent_space_neighbors(
                    self.model,
                    train_loader,
                    num_neighbors=5,
                    alpha=self.config["ALPHA"],
                    num_classes=nb_classes,
                )

            # Early stopping
            if acc > best_metrics["acc"]:
                best_metrics["acc"] = acc
                best_metrics["f1"] = f1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config["EARLY_STOP_PATIENCE"]:
                    print("Early stopping triggered at epoch:", epoch)
                    break

        # Save model if configured
        if self.config["SAVE_VAE"]:
            model_path = get_model_path(self.config, name)
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        # Update logs
        logs[f"{name}_best_acc"] = best_metrics["acc"]
        logs[f"{name}_best_f1"] = best_metrics["f1"]

        if self.config["WANDB"]:
            wandb.finish()

        return self.model, logs
