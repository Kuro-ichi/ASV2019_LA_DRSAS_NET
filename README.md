# DRSAS-Net (Python package scaffold)

This is a modular, production-friendly scaffold derived from the original notebook `drsas-net.ipynb`.

## Structure
```
drsas_net/
  __init__.py
  config.py
  data/
    __init__.py
    dataset.py
  models/
    __init__.py
    model.py
  engine/
    __init__.py
    trainer.py
    evaluator.py
  utils/
    __init__.py
    metrics.py
    logger.py
    seed.py
scripts/
  train.py
  eval.py
  predict.py
configs/
  defaults.yaml
tests/
  test_smoke.py
README.md
pyproject.toml
requirements.txt
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Training
python scripts/train.py --config configs/defaults.yaml

# Evaluation
python scripts/eval.py --checkpoint checkpoints/model.pth --data ./data

# Prediction
python scripts/predict.py --checkpoint checkpoints/model.pth --input ./samples
```

## Notes
- Replace the placeholder model in `models/model.py` with the architecture for DRSAS-Net.
- Plug your dataset logic into `data/dataset.py`.
- Extend metrics in `utils/metrics.py`.
- Hyperparameters live in `configs/defaults.yaml` and are merged into argparse at runtime.
