from collections import OrderedDict
import torch
import torch.nn as nn
from models.attention import SimpleSelfAttention, Marginals, Marginals


def unet_block(in_channels, features, name):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class UNetEncoder(nn.Module):
    def __init__(self, channels=1, num_classes=1, init_features=32, spatial_dim=240,
                 learnable_attn=False, learnable_marginals=False, attn_embed_factor=8, **kwargs):
        super(UNetEncoder, self).__init__()

        self.features = init_features
        self.spatial_dim = spatial_dim
        self.in_channels = channels
        self.num_classes = num_classes
        self.learnable_attn = learnable_attn
        self.learnable_marginals = learnable_marginals
        self.embed_channels = attn_embed_factor * init_features

        self.encoder1 = unet_block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder2 = unet_block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder3 = unet_block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder4 = unet_block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bottleneck = unet_block(self.features * 8, self.features * 16, name="bottleneck")

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.features * 16, self.num_classes)

        if self.learnable_attn:
            self.tau_k = SimpleSelfAttention(input_channels=self.features * 4, embed_channels=self.embed_channels)
            self.tau_l = SimpleSelfAttention(input_channels=self.features * 8, embed_channels=self.embed_channels)
            if self.learnable_marginals:
                self.marginals = Marginals(margin_dim=256)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.encoder3(x)
        if self.learnable_attn:
            x, tau_k = self.tau_k(x)
        x = self.pool3(x)
        x = self.encoder4(x)
        if self.learnable_attn:
            x, tau_l = self.tau_l(x)
        x = self.pool4(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        if self.learnable_attn:
            if self.learnable_marginals:
                marginal_pairs = self.marginals(tau_k, tau_l)
                return x, (tau_k, tau_l), marginal_pairs
            return x, (tau_k, tau_l), None
        return x
        # return x, None, None      # clashes with captum


def get_unet_encoder_classifier(**kwargs):
    model = UNetEncoder(**kwargs)
    if kwargs["weights"] is not None:
        print("Loading pretrained model from: '{}'".format(kwargs["weights"]))
        weights = torch.load(kwargs["weights"])
        model.load_state_dict(weights, strict=False)
    if kwargs["freeze_backbone"]: # TODO
        # Enable training only on the self attention / final layers
        for layer in model.modules():
            for p in layer.parameters():
                p.requires_grad = False
        for layer in model.modules():
            if hasattr(layer, "name") or any([s in layer._get_name() for s in ["Linear"]]):
                for p in layer.parameters():
                    p.requires_grad = True
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}\nTotal trainable parameters: {}".format(total_params, trainable_params))
    return model