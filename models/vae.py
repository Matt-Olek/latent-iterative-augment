import torch
import torch.nn as nn
from utils import to_default_device


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=1000,
        hidden_dim_classifier=100,
        latent_dim=720,
        learning_rate=1e-4,
        knn=5,
    ):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder(input_dim, hidden_dim)
        self.decoder = self.build_decoder(latent_dim, hidden_dim, input_dim)

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        self.classifier = self.build_classifier(
            latent_dim, hidden_dim_classifier, num_classes
        )

    def build_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def build_decoder(self, latent_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def build_classifier(self, latent_dim, hidden_dim, num_classes):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_layer(h), self.log_var_layer(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, var):
        eps = torch.randn_like(var)
        return mean + var * eps

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)
        y_pred = self.classifier(z)
        return x_hat, mean, log_var, y_pred

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            return self.decoder(z).cpu().numpy()

    def augment(self, x, num_samples):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
            mu, log_var = torch.chunk(encoded, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            z = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.latent_dim)
            return self.decoder(z).cpu().numpy()
