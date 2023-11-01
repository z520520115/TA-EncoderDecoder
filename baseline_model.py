import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module):
    def __init__(self, num_features=256, num_classes=2, dropout_rate=0.3):
        super(DCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(num_features * 56 * 56, num_classes)

    def forward(self, x):

        x = x.squeeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class LSTM(nn.Module):
    def __init__(self, input_size=3 * 224 * 224, hidden_size=512, num_layers=2, num_classes=2, dropout=0.4, device='cuda'):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):

        self.lstm.flatten_parameters()
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        # Apply Softmax
        out = self.softmax(out)

        return out


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=2, channels=3, dim=512, depth=6, heads=8, mlp_dim=2048):
        super().__init__()

        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch_size, _, height, width = img.shape
        img = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        img = img.contiguous().view(batch_size, -1, self.patch_dim)
        x = self.to_patch_embedding(img.view(-1, self.patch_dim))
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x.view(batch_size, -1, self.dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = self.to_logits(x[:, 0])
        return x