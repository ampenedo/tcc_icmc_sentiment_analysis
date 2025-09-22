#!/usr/bin/env python3
"""
Split mínimo com chardet + opção de manter ou não proporções.
Os arquivos de saída são nomeados automaticamente com a estratégia e % de treino/val.
"""

import argparse
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split

def detect_encoding(path, sample_size=1_000_000):
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    res = chardet.detect(raw) or {}
    enc = res.get("encoding") or "utf-8"
    return enc

def map_rating_to_sentiment(r):
    try:
        r = int(round(float(r)))
    except Exception:
        return None
    if r in (1, 2): return "neg"
    if r == 3: return "neu"
    if r in (4, 5): return "pos"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--sep", default=",", help="Separador do CSV (padrão: ',').")
    ap.add_argument("--test-size", type=float, default=0.2, help="Proporção para validação (padrão: 0.2).")
    ap.add_argument("--random-state", type=int, default=42, help="Semente (padrão: 42).")
    ap.add_argument("--no-stratify", action="store_true",
                    help="Se passado, divide SEM manter proporções de classes.")
    args = ap.parse_args()

    # Detectar encoding
    enc = detect_encoding(args.input_csv)
    print(f"[info] encoding detectado: {enc}")

    # Ler CSV
    df = pd.read_csv(args.input_csv, sep=args.sep, encoding=enc, low_memory=False)
    print(f"[info] linhas totais: {len(df):,}")

    # Derivar sentimento
    if "overall_rating" not in df.columns:
        raise ValueError("Coluna 'overall_rating' não encontrada.")
    df["sentiment"] = df["overall_rating"].apply(map_rating_to_sentiment)

    dist = df["sentiment"].value_counts(dropna=False)
    print("\n[info] distribuição completa:")
    print(dist)

    df_strat = df[df["sentiment"].notna()].copy()

    stratify = None if args.no_stratify else df_strat["sentiment"]

    # Split
    train_df, valid_df = train_test_split(
        df_strat,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify
    )

    print(f"\n[info] treino: {len(train_df):,} | validação: {len(valid_df):,}")
    print("\n[info] proporções treino:")
    print((train_df["sentiment"].value_counts() / len(train_df)).round(4))
    print("\n[info] proporções validação:")
    print((valid_df["sentiment"].value_counts() / len(valid_df)).round(4))

    # Decide nomes de saída
    tag = "stratify" if not args.no_stratify else "nstratify"
    pct_train = int(round((1 - args.test_size) * 100))
    train_name = f"data/treino_{tag}_{pct_train}.csv"
    val_name   = f"data/validacao_{tag}_{100 - pct_train}.csv"

    train_df.to_csv(train_name, index=False, encoding="utf-8")
    valid_df.to_csv(val_name, index=False, encoding="utf-8")
    print(f"\n[ok] salvo: {train_name} e {val_name}")

if __name__ == "__main__":
    main()
