import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(ConvEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),  # (B, 1, 28, 28) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B, 32, 14, 14) -> (B, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (B, 64, 7, 7) -> (B, 128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # (B, 128, 4, 4) -> (B, 128*4*4)
            nn.Linear(128 * 4 * 4, feature_dim)  # (B, 128*4*4) -> (B, 512)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ConvDecoder(nn.Module):
    def __init__(self, feature_dim):
        super(ConvDecoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 128 * 4 * 4),  # (B, 512) -> (B, 128*4*4)
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # (B, 128*4*4) -> (B, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # (B, 128, 4, 4) -> (B, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (B, 64, 7, 7) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # (B, 32, 14, 14) -> (B, 1, 28, 28)
            nn.Sigmoid()  # Use sigmoid if input images are normalized to [0, 1]
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device, default='AE'):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        if default == 'AE':
            for v in range(view):
                self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
                self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        else:
            for v in range(view):
                self.encoders.append(ConvEncoder(input_size[v], feature_dim).to(device))
                self.decoders.append(ConvDecoder(feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # nn.ReLU(),
            # nn.Linear(feature_dim, high_feature_dim)
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.clustering_module = nn.Sequential(
            nn.Linear(feature_dim, class_num)
            # nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        qns = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = F.normalize(self.feature_contrastive_module(z), dim=1)
            z_noise = z + torch.rand_like(z) * 0.001
            q = self.clustering_module(z)
            q_noise = self.clustering_module(z_noise)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            qns.append(q_noise)
            xrs.append(xr)
        return hs, qs, qns, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            z = F.normalize(z, dim=1)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        probs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.clustering_module(z)
            prob = F.softmax(q, 1)
            pred = torch.argmax(prob, dim=1)
            probs.append(prob)
            preds.append(pred)
        return probs, preds

    def forward_single(self, x, v):
        z = self.encoders[v](x)
        q = self.clustering_module(z)
        return q
