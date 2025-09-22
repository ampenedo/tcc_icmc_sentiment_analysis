import pandas as pd
import argparse

def main(input_csv, output_cleaned, output_duplicates, sep, encoding):
    # Le forçando tipos das colunas de deduplicação
    # - submission_date: datetime (parse_dates)
    # - reviewer_id e product_id: string (evita tipos mistos)
    df = pd.read_csv(
        input_csv,
        sep=sep,
        encoding=encoding,
        dtype={
            "reviewer_id": "string",
            "product_id": "string",
        },
        parse_dates=["submission_date"],
        infer_datetime_format=True,   # ajuda a acelerar o parse
        dayfirst=False,               # ajuste data for no formato dia/mês
        low_memory=False              # evita DtypeWarning por leitura em chunks
    )

    # (Opcional) Normalizar espaços em IDs, se houver
    for col in ("reviewer_id", "product_id"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Define as colunas de deduplicação
    dedup_columns = ["submission_date", "reviewer_id", "product_id"]

    # Por segurança, garante que as colunas existem
    missing = [c for c in dedup_columns if c not in df.columns]
    if missing:
        raise ValueError(f"As colunas abaixo não foram encontradas no CSV: {missing}")

    # Identifica duplicatas (mantém a primeira ocorrência)
    dup_mask = df.duplicated(subset=dedup_columns, keep="first")
    duplicates = df[dup_mask].copy()

    # Remove duplicatas
    cleaned = df[~dup_mask].copy()

    # Salva os resultados
    cleaned.to_csv(output_cleaned, index=False)
    duplicates.to_csv(output_duplicates, index=False)

    print(f"Arquivo limpo salvo em: {output_cleaned}")
    print(f"Arquivo de duplicatas salvo em: {output_duplicates}")
    print(
        f"Linhas originais: {len(df)}, Linhas finais: {len(cleaned)}, "
        f"Duplicatas removidas: {len(duplicates)}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicatas baseadas em (submission_date, reviewer_id, product_id)."
    )
    parser.add_argument("input_csv", help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output-cleaned", default="cleaned.csv", help="Arquivo CSV limpo (default: cleaned.csv)")
    parser.add_argument("--output-duplicates", default="duplicates.csv", help="Arquivo CSV com duplicatas (default: duplicates.csv)")
    parser.add_argument("--sep", default=",", help="Separador do CSV (default: ,)")
    parser.add_argument("--encoding", default=None, help="Encoding do CSV (ex.: utf-8, latin-1). Se não informado, pandas tenta inferir.")
    args = parser.parse_args()

    main(args.input_csv, args.output_cleaned, args.output_duplicates, args.sep, args.encoding)
