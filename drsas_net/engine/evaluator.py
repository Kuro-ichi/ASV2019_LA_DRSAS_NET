from __future__ import annotations
import argparse, os, torch
from drsas_net.config import add_common_args, load_config
from drsas_net.data.dataset import make_dataloader
from drsas_net.models.model import build_model
from drsas_net.utils.metrics import accuracy

def main(argv=None):
    p = argparse.ArgumentParser("DRSAS-Net Evaluator")
    add_common_args(p)
    p.add_argument("--checkpoint", type=str, required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)
    device = cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")

    # data
    data_cfg = cfg.get("data", {})
    val_loader = make_dataloader(
        root=data_cfg.get("root","./data"),
        split=data_cfg.get("val_split","val"),
        batch_size=cfg.get("eval", {}).get("batch_size",32),
        num_workers=data_cfg.get("num_workers",4),
        in_channels=cfg.get("model",{}).get("in_channels",1),
        num_classes=cfg.get("model",{}).get("num_classes",10),
    )

    # model
    mcfg = cfg.get("model", {})
    model = build_model(name=mcfg.get("name","DRSASNet"), in_channels=mcfg.get("in_channels",1), num_classes=mcfg.get("num_classes",10)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total = 0; correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc = accuracy(logits, y)
            total += y.numel(); correct += int(acc * y.numel())

    print(f"Eval accuracy: {correct/total:.4f} over {total} samples")

if __name__ == "__main__":
    main()
