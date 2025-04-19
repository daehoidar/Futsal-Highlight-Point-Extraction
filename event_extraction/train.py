import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset

class EventDataset(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips  # list of [T, H, W, C]
        self.labels = labels

    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        clip = self.clips[idx].transpose(3,0,1,2)  # C,T,H,W
        return torch.tensor(clip, dtype=torch.float32), torch.tensor(self.labels[idx])

class EventModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=128):
        super().__init__()
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        feats = self.cnn(x)  # [B*T,512]
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        return self.classifier(out)

def train(args):
    # load dataset, split, DataLoader, optim, loss
    # placeholder: user should implement data loading from labels
    dataset = EventDataset([...], [...])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = EventModel(num_classes=3).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(args.device), y.to(args.device)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-new', action='store_true')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-path', type=str, default='event_model.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(args)