import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from data import VOCAB_SIZE, create_dataloaders
from model import MiniTransformerClassifier
from utils import count_parameters, ensure_dir, format_seconds, plot_training_curves, save_json, set_seed


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "acc": total_correct / total_examples,
    }


def train_one_run(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        verify_labels=args.verify_labels,
    )

    model = MiniTransformerClassifier(
        vocab_size=VOCAB_SIZE,
        max_len=20,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state_dict = None

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        train_loss = running_loss / total_examples
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_time = time.time() - start_time

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_metrics = evaluate(model, test_loader, device)
    param_count = count_parameters(model)

    model_path = os.path.join(args.output_dir, f"{args.run_name}_best.pt")
    torch.save(model.state_dict(), model_path)

    curve_path = os.path.join(args.output_dir, f"{args.run_name}_curve.png")
    plot_training_curves(history, curve_path)

    results = {
        "run_name": args.run_name,
        "use_positional_encoding": args.use_positional_encoding,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "train_time_seconds": total_train_time,
        "train_time_readable": format_seconds(total_train_time),
        "param_count": param_count,
        "model_path": model_path,
        "curve_path": curve_path,
    }

    results_path = os.path.join(args.output_dir, f"{args.run_name}_results.json")
    save_json(results, results_path)

    print("\nFinal results")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="validation.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="model_A")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verify_labels", action="store_true")
    parser.add_argument("--use_positional_encoding", action="store_true")
    parser.add_argument("--no_positional_encoding", action="store_true")

    args = parser.parse_args()

    if args.no_positional_encoding:
        args.use_positional_encoding = False
    elif not args.use_positional_encoding:
        # default to True unless explicitly disabled
        args.use_positional_encoding = True

    train_one_run(args)
