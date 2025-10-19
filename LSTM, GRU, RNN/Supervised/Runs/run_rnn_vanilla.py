import argparse, time, math, os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from RNN_Vanilla import RNN_Vanilla

# We reuse CIFAR-10 to keep parity with your CNN runners.
# Each 32x32x3 image is reshaped into a sequence of T=32 steps with D=96 features (3*32).

def make_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # 90/10 split from train
    n = len(trainset)
    n_val = n // 10
    n_train = n - n_val
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    def collate(batch):
        xs, ys = [], []
        for x, y in batch:
            # x: (3,32,32) -> (32, 96)
            seq = x.permute(1,2,0).reshape(32, -1)
            xs.append(seq)
            ys.append(y)
        x = torch.stack(xs, 0)
        y = torch.tensor(ys)
        return x, y

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader, test_loader


def train_eval(model, loaders, device, epochs=50, lr=1e-3, weight_decay=1e-4, patience=7):
    train_loader, val_loader, test_loader = loaders
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_loss = loss_sum / total
        train_acc = correct / total

        # val
        model.eval()
        with torch.no_grad():
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = loss_sum / total
        val_acc = correct / total
        print(f"Epoch {ep:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    with torch.no_grad():
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_loss = loss_sum / total
    test_acc = correct / total
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = make_loaders(batch_size=args.batch)

    model = RNN_Vanilla(input_dim=96, hidden_size=args.hidden, num_layers=args.layers, num_classes=10)
    train_eval(model, (train_loader, val_loader, test_loader), device, epochs=args.epochs, lr=args.lr, weight_decay=args.wd, patience=args.patience)

if __name__ == '__main__':
    main()
