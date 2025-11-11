from __future__ import annotations
import argparse, torch
import torch.nn as nn
from torch.optim import AdamW
from drsas_net.config import add_common_args, load_config
from drsas_net.data.dataset import make_dataloader
from drsas_net.models.model import build_model
from drsas_net.utils.metrics import accuracy
from drsas_net.utils.logger import SimpleLogger
from drsas_net.utils.seed import set_seed

def train_one_epoch(model, loader, criterion, opt, device, logger, log_interval=50):
    model.train()
    total = 0; correct = 0; running = 0.0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        acc = accuracy(logits, y)
        total += y.numel(); correct += int(acc * y.numel()); running += loss.item()
        if step % log_interval == 0:
            logger.log(step, loss=running/log_interval, acc=correct/total)
            running = 0.0
    return correct/total

def main(argv=None):
    p = argparse.ArgumentParser("DRSAS-Net Trainer")
    add_common_args(p)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_cfg = cfg.get("data", {})
    train_loader = make_dataloader(
        root=data_cfg.get("root","./data"),
        split=data_cfg.get("train_split","train"),
        batch_size=data_cfg.get("batch_size",16),
        num_workers=data_cfg.get("num_workers",4),
        in_channels=cfg.get("model",{}).get("in_channels",1),
        num_classes=cfg.get("model",{}).get("num_classes",10),
    )

    # Model/optim
    mcfg = cfg.get("model", {})
    model = build_model(name=mcfg.get("name","DRSASNet"), in_channels=mcfg.get("in_channels",1), num_classes=mcfg.get("num_classes",10)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=cfg.get("train", {}).get("lr",1e-3), weight_decay=cfg.get("train", {}).get("weight_decay",1e-4))
    logger = SimpleLogger()

    epochs = cfg.get("train", {}).get("epochs",20)
    log_interval = cfg.get("train", {}).get("log_interval",50)
    best = 0.0

    for epoch in range(1, epochs+1):
        acc = train_one_epoch(model, train_loader, criterion, opt, device, logger, log_interval=log_interval)
        print(f"Epoch {epoch} done. train_acc={acc:.4f}")
        if acc > best:
            best = acc
            ckpt_dir = cfg.get("train", {}).get("ckpt_dir","./checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(ckpt_dir, "model.pth"))
            print(f"Saved checkpoint to {ckpt_dir}/model.pth (best_acc={best:.4f})")

if __name__ == "__main__":
    main()
