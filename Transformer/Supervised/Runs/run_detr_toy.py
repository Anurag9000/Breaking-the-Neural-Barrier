import argparse
import random
from pathlib import Path
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_detr_toy import DETRToy

# --- Toy rectangles detection dataset (single class + background) ---
class Rectangles(Dataset):
    def __init__(self, n: int = 2000, size: int = 128, max_objects: int = 5):
        super().__init__()
        self.n, self.size, self.max_objects = n, size, max_objects
        self.num_classes = 2  # {background, rectangle}
        random.seed(0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        S = self.size
        img = torch.zeros(3, S, S)
        nobj = random.randint(1, self.max_objects)
        boxes = []
        labels = []
        for _ in range(nobj):
            x0 = random.randint(0, S-20)
            y0 = random.randint(0, S-20)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            x1 = min(S-1, x0 + w)
            y1 = min(S-1, y0 + h)
            img[:, y0:y1, x0:x1] = torch.rand(3,1,1)
            cx = (x0 + x1) / 2 / S
            cy = (y0 + y1) / 2 / S
            bw = (x1 - x0) / S
            bh = (y1 - y0) / S
            boxes.append([cx, cy, bw, bh])
            labels.append(1)
        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes):
    # extremely simplified L2 + class cost + greedy matching for toy demo
    # pred: (Q, C), (Q, 4); target: (N,) (N,4)
    with torch.no_grad():
        Q = pred_boxes.size(0); N = tgt_boxes.size(0)
        if N == 0:
            return []
        cost = torch.cdist(pred_boxes, tgt_boxes)  # (Q,N)
        cls_cost = -pred_logits[:, 1].unsqueeze(1).repeat(1, N)  # prefer rectangle class
        total = cost + 0.5*cls_cost
        match = []
        chosen_q = set(); chosen_n = set()
        for _ in range(min(Q, N)):
            q, n = torch.nonzero(total == total.min())[0].tolist()
            if q in chosen_q or n in chosen_n:
                total[q, n] = total.max() + 1
                continue
            match.append((q, n)); chosen_q.add(q); chosen_n.add(n)
            total[q, :] = total.max() + 1
            total[:, n] = total.max() + 1
        return match


def loss_fn(pred_logits, pred_boxes, tgt_labels, tgt_boxes):
    # classification CE + L1 bbox + IoU-like (1 - IoU of boxes as cxcywh)
    Q = pred_boxes.size(0); device = pred_boxes.device
    if tgt_boxes.numel() == 0:
        # all should be background
        ce = nn.functional.cross_entropy(pred_logits, torch.zeros(Q, dtype=torch.long, device=device))
        return ce
    match = hungarian_match(pred_logits.detach(), pred_boxes.detach(), tgt_labels, tgt_boxes)
    if not match:
        return pred_boxes.sum()*0 + pred_logits.sum()*0
    q_idx = torch.tensor([q for q,_ in match], dtype=torch.long, device=device)
    n_idx = torch.tensor([n for _,n in match], dtype=torch.long, device=device)
    ce = nn.functional.cross_entropy(pred_logits[q_idx], tgt_labels[n_idx])
    l1 = nn.functional.l1_loss(pred_boxes[q_idx], tgt_boxes[n_idx])
    # IoU (approx) for cxcywh boxes
    def box_area_wh(b):
        return b[...,2]*b[...,3]
    iou = box_area_wh(torch.min(pred_boxes[q_idx], tgt_boxes[n_idx])) / (box_area_wh(pred_boxes[q_idx]) + box_area_wh(tgt_boxes[n_idx]) - box_area_wh(torch.min(pred_boxes[q_idx], tgt_boxes[n_idx]) ) + 1e-6)
    iou_loss = (1 - iou).mean()
    return ce + l1 + iou_loss


def collate(batch):
    imgs, boxes, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, boxes, labels


def evaluate(model, loader, device):
    model.eval(); total=0; correct=0
    with torch.no_grad():
        for imgs, boxes, labels in loader:
            imgs = imgs.to(device)
            logits, pred_b = model(imgs)
            # crude metric: count matched rectangles classified as class-1
            for i in range(len(labels)):
                q = logits[i].softmax(-1)[:,1]
                top = q.topk(k=min(q.numel(), len(labels[i]))).indices
                correct += top.numel()
                total += len(labels[i])
    return correct / max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    train_ds = Rectangles(1500)
    val_ds = Rectangles(300)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = DETRToy(num_classes=2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad, patience = 0.0, 0, 5

    # Init Logger

    logger = ContinuousLogger(Path('results_run_detr_toy'), 'run_detr_toy', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for imgs, boxes, labels in train_loader:
            imgs = imgs.to(args.device)
            logits, pred_b = model(imgs)
            loss = 0.0
            for i in range(len(labels)):
                tgt_b = boxes[i].to(args.device)
                tgt_y = labels[i].to(args.device)
                loss = loss + loss_fn(logits[i], pred_b[i], tgt_y, tgt_b)
            loss = loss / len(labels)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        score = evaluate(model, val_loader, args.device)
        # Log

        msg = f"Epoch {epoch}: toy mAP-like={score:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if score > best + 1e-6:
            best = score; bad = 0
            torch.save({'model': model.state_dict()}, 'DETRToy_best.pth')
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.'); break
    print('Done. Best val score:', best)

if __name__ == '__main__':
    main()