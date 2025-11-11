#!/usr/bin/env python
import argparse, torch
from pathlib import Path
from drsas_net.models.model import build_model

def main():
    ap = argparse.ArgumentParser("DRSAS-Net Predict")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input", type=str, required=True, help="Folder of inputs (placeholder demo)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    p = Path(args.input)
    files = sorted(p.glob("*"))
    print(f"Predicting on {len(files)} files (placeholder). Replace with real preprocessing.")

if __name__ == "__main__":
    main()
