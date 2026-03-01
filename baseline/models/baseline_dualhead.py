import torch
import torch.nn as nn
import torchvision.models as tvm


class SeverityAggregator(nn.Module):
    def __init__(self, w=(0.0, 0.33, 0.66, 1.0),
                 alpha=0.6, beta=0.3, gamma=0.1,
                 eta=0.9, T=0.25, eps=1e-6, W_spatial=None):
        super().__init__()
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32).view(1, 4, 1, 1))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.T = T
        self.eps = eps

        if W_spatial is None:
            xs = torch.linspace(-1.0, 1.0, 8)
            ys = torch.linspace(-1.0, 1.0, 8)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            sigma = 0.5
            W = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            W = W / (W.sum() + eps)
            W = W.view(1, 1, 8, 8)
        else:
            W = W_spatial
        self.register_buffer("W", W.float())

    def forward(self, G_hat):  # [B,4,8,8] after softmax
        B = G_hat.shape[0]

        # s_ij = sum_c w_c * g_ijc
        s = (self.w * G_hat).sum(dim=1, keepdim=True)  # [B,1,8,8]

        # opacity-aware mean
        S_op = s.mean(dim=(2, 3))  # [B,1]

        # spatial importance
        S_sp = (s * self.W).sum(dim=(2, 3))  # [B,1]

        # dominance: softmax pooling over tiles
        flat = (s / self.T).view(B, 1, -1)
        a = torch.softmax(flat, dim=-1).view(B, 1, 8, 8)  # [B,1,8,8]
        S_dom_raw = (a * s).sum(dim=(2, 3))  # [B,1]

        # transparent ratio per tile: g1 / (g1+g2+g3)
        g1 = G_hat[:, 1:2, :, :]
        g23 = G_hat[:, 2:4, :, :].sum(dim=1, keepdim=True)
        r_tr = g1 / (g1 + g23 + self.eps)  # [B,1,8,8]
        r_tr_tilde = (a * r_tr).sum(dim=(2, 3))  # [B,1]

        S_dom = (1.0 - self.eta * r_tr_tilde) * S_dom_raw  # [B,1]

        S_agg = self.alpha * S_op + self.beta * S_sp + self.gamma * S_dom
        S_agg = torch.clamp(S_agg, 0.0, 1.0)
        return S_agg


class BaselineDualHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pool8 = nn.AdaptiveAvgPool2d((8, 8))

        self.tile_head = nn.Conv2d(512, 4, kernel_size=1, bias=True)
        self.glob_fc = nn.Linear(512, 1)

        self.aggregator = SeverityAggregator()

    def forward(self, x):
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)     # [B,512,H,W]
        f8 = self.pool8(f)     # [B,512,8,8]

        logits = self.tile_head(f8)              # [B,4,8,8]
        G_hat = torch.softmax(logits, dim=1)     # [B,4,8,8]

        v = f8.mean(dim=(2, 3))                  # [B,512]
        S_hat = torch.sigmoid(self.glob_fc(v))   # [B,1]

        S_agg = self.aggregator(G_hat)           # [B,1]

        return {"logits": logits, "G_hat": G_hat, "S_hat": S_hat, "S_agg": S_agg}
