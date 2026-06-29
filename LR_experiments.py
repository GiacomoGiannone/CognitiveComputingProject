# LR_experiments.py
# Script per la ricerca degli iperparametri (Learning Rate) sul modello BERT di Quality Scoring.
# Testa tre valori di learning rate e produce una tabella riassuntiva delle metriche di validazione.

import os
import sys
from datetime import datetime
import warnings

# Disattiva i warning non critici per mantenere l'output pulito
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configura matplotlib in modalità headless (Agg) prima di importare qualsiasi modulo che lo usi
import matplotlib
matplotlib.use('Agg')

# Aggiungi il progetto al path per garantire gli import relativi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.score_agent import ScoreAgent


# ── Configurazione ──────────────────────────────────────────
TRAIN_PATH = "data/bert_dataset/train.jsonl"
VAL_PATH = "data/bert_dataset/val.jsonl"
LEARNING_RATES = [1e-5, 2e-5, 5e-5]
EPOCHS = 3
BATCH_SIZE = 4
EXPERIMENTS_DIR = "models/experiments"


def run_experiments():
    """Esegue gli esperimenti di hyperparameter search sul learning rate."""
    
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER SEARCH - Learning Rate Experiments")
    print("=" * 70)
    print(f"  Train dataset:    {TRAIN_PATH}")
    print(f"  Val dataset:      {VAL_PATH}")
    print(f"  Learning rates:   {LEARNING_RATES}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Output directory: {EXPERIMENTS_DIR}")
    print("=" * 70)

    # Verifica esistenza dei dataset
    if not os.path.exists(TRAIN_PATH):
        print(f"\n ERRORE: File di training non trovato: {TRAIN_PATH}")
        sys.exit(1)
    if not os.path.exists(VAL_PATH):
        print(f"\n ERRORE: File di validazione non trovato: {VAL_PATH}")
        sys.exit(1)

    # Raccolta risultati
    results = []

    for i, lr in enumerate(LEARNING_RATES, 1):
        lr_label = f"{lr:.0e}"  # es. "1e-05", "2e-05", "5e-05"
        experiment_dir = os.path.join(EXPERIMENTS_DIR, f"lr_{lr_label}")

        print(f"\n{'─' * 70}")
        print(f"  EXPERIMENT {i}/{len(LEARNING_RATES)} — Learning Rate: {lr}")
        print(f"  Output: {experiment_dir}")
        print(f"{'─' * 70}")

        # Inizializza un'istanza pulita di ScoreAgent per ogni esperimento
        agent = ScoreAgent()

        # Addestra il modello con il learning rate corrente
        trainer = agent.train(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            output_dir=experiment_dir,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=lr,
            plot_loss=True
        )

        # Estrai le metriche di validazione consolidate
        eval_metrics = trainer.evaluate()
        mse = eval_metrics.get("eval_mse", float("nan"))
        mae = eval_metrics.get("eval_mae", float("nan"))
        acc = eval_metrics.get("eval_acc_within_0.1", float("nan"))

        results.append({
            "lr": lr,
            "lr_label": lr_label,
            "mse": mse,
            "mae": mae,
            "acc": acc,
        })

        print(f"\n  Experiment {i} complete:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    Acc:  {acc:.2%}")

    # ── Tabella riassuntiva ──────────────────────────────────
    header = f"{'Learning Rate':>15} | {'MSE':>10} | {'MAE':>10} | {'Acc (±0.1)':>12}"
    separator = "-" * len(header)

    summary_lines = []
    summary_lines.append("")
    summary_lines.append("=" * 60)
    summary_lines.append("  EXPERIMENT SUMMARY — Learning Rate Search")
    summary_lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"  Epochs: {EPOCHS}  |  Batch Size: {BATCH_SIZE}")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append(header)
    summary_lines.append(separator)

    best_idx = -1
    best_mse = float("inf")

    for idx, r in enumerate(results):
        line = f"{r['lr_label']:>15} | {r['mse']:>10.6f} | {r['mae']:>10.6f} | {r['acc']:>11.2%}"
        summary_lines.append(line)
        if r["mse"] < best_mse:
            best_mse = r["mse"]
            best_idx = idx

    summary_lines.append(separator)
    if best_idx >= 0:
        best = results[best_idx]
        summary_lines.append(f"\n  BEST: lr={best['lr_label']}  (MSE={best['mse']:.6f}, MAE={best['mae']:.6f}, Acc={best['acc']:.2%})")
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    # Stampa a terminale
    print(summary_text)

    # Scrivi il file di log
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    summary_path = os.path.join(EXPERIMENTS_DIR, "experiment_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiments()
