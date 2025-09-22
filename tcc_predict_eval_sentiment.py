#!/usr/bin/env python3
"""
Prediction + Evaluation script for B2W sentiment (Portuguese).

- Carrega um pipeline.joblib (TF-IDF, W2V ou BERT wrapper)
- Lê metadata.json ao lado do pipeline para:
  * descobrir se existem meta-colunas injetadas como tokens
  * descobrir as colunas de texto usadas no treino
- Gera predições e, se houver ground truth, produz:
  * matriz de confusão (raw e normalizada) PNG
  * classification report (CSV)
  * métricas agregadas (accuracy, F1 macro/weighted/micro, balanced accuracy)
  * top features por classe (apenas se TF-IDF + modelo linear com coef_)
  * relatório Markdown
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    f1_score, balanced_accuracy_score
)
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe


# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------
def exp_tag_from_pipeline(pipeline_path: str):
    """Extrai tag do cenário do nome da pasta e tenta ler metadata.json."""
    p = Path(pipeline_path).resolve()
    parent = p.parent
    parent_name = parent.name.lower()

    m = re.search(r'(?:^|[_-])(n?stratify)[_-](\d{1,3})(?=$)', parent_name)
    tag = f"{m.group(1)}_{m.group(2)}" if m else None

    model = None
    train_csv_name = None
    meta = None
    meta_path = parent / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not tag:
                tag = meta.get("train_tag")
            model = meta.get("model")
            src = meta.get("source_train_csv")
            if src:
                train_csv_name = Path(src).name
        except Exception:
            pass

    if not tag:
        tag = "unknown"
    return tag, model, train_csv_name, meta


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
    )
    return df


def robust_read_csv(path, sep, encoding):
    if sep:
        return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding=encoding, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=",", encoding=encoding, low_memory=False)


# ---------------------------------------------------------
# Desserializar pipelines
# ---------------------------------------------------------
class TextConcatenator(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols=("review_title","review_text")):
        self.text_cols = tuple(text_cols)
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        missing = [c for c in self.text_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas de texto faltando: {missing}. Disponíveis: {list(df.columns)}")
        return df[list(self.text_cols)].fillna("").astype(str).agg(" \n ".join, axis=1)


class SimpleCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._ws = re.compile(r"\s+")
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for t in X:
            t = "" if t is None else str(t).lower()
            t = self._ws.sub(" ", t).strip()
            out.append(t)
        return out


class W2VVectorizer:
    """
    Mesmo transformador usado no treino (média de embeddings).
    Em pipelines salvos, `self.kv` já vem serializado.
    """
    def __init__(self, w2v_model=None, size=300, window=5, min_count=2, sg=1,
                 epochs=15, workers=-1, tfidf_weight=False, tfidf_params=None):
        self.w2v_model = w2v_model
        self.size = size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers if workers != -1 else None
        self.tfidf_weight = bool(tfidf_weight)
        self.tfidf_params = tfidf_params or {}
        self.kv = None
        self.idf_ = {}
        self.dim_ = int(size)

    def _tokenize(self, texts):
        return [t.split() for t in texts]

    def fit(self, X, y=None):
        # Mantido por compatibilidade; normalmente não é chamado na inferência.
        from gensim.models import Word2Vec, KeyedVectors
        texts = list(X)
        sents = self._tokenize(texts)

        if self.w2v_model:
            p = str(self.w2v_model)
            try:
                self.kv = KeyedVectors.load(p, mmap='r')
            except Exception:
                try:
                    self.kv = KeyedVectors.load_word2vec_format(p, binary=p.lower().endswith(".bin"))
                except Exception as e:
                    raise RuntimeError(f"Falha ao carregar embeddings em {p}: {e}")
            self.dim_ = self.kv.vector_size
        else:
            w2v = Word2Vec(
                sentences=sents,
                vector_size=self.size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                workers=(self.workers or 1),
                epochs=self.epochs
            )
            self.kv = w2v.wv
            self.dim_ = self.kv.vector_size

        if self.tfidf_weight:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(
                lowercase=False, strip_accents=None, analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                **self.tfidf_params
            )
            tfidf.fit(texts)
            inv_vocab = {i: t for t,i in tfidf.vocabulary_.items()}
            idf_diag = tfidf.idf_
            self.idf_ = {inv_vocab[i]: float(idf_diag[i]) for i in range(len(inv_vocab))}
        return self

    def transform(self, X):
        import numpy as np
        if self.kv is None:
            raise RuntimeError("W2VVectorizer não está inicializado (kv=None) no pipeline carregado.")
        texts = list(X)
        sents = self._tokenize(texts)
        dim = self.dim_
        out = np.zeros((len(sents), dim), dtype=np.float32)
        for i, toks in enumerate(sents):
            vecs = []
            weights = []
            for tok in toks:
                if tok in self.kv:
                    vecs.append(self.kv[tok])
                    weights.append(self.idf_.get(tok, 1.0) if self.tfidf_weight else 1.0)
            if vecs:
                V = np.vstack(vecs)
                w = np.array(weights, dtype=np.float32)
                w = w / (w.sum() + 1e-9)
                out[i] = (V * w[:, None]).sum(axis=0)
        return out


# ---------------------------------------------------------
# Hugging Face — apenas para inferência
# ---------------------------------------------------------
def _lazy_import_hf():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    return torch, AutoTokenizer, AutoModelForSequenceClassification


class HFTextClassifier:
    """Wrapper salvo dentro do pipeline para inferência com modelos HF locais."""
    def __init__(self, model_dir: str, max_length: int = 256, infer_batch_size: int = 32):
        self.model_dir = str(model_dir)
        self.max_length = int(max_length)
        self.infer_batch_size = int(infer_batch_size)
        self._tok = None
        self._model = None
        self._device = None
        self._id2label = None

    def _resolve_model_dir(self, pipe_dir: Path) -> str:
        """
        Resolve o diretório do modelo:
          1) como veio;
          2) relativo ao CWD;
          3) relativo à pasta do pipeline;
          4) fallback: pasta irmã 'hf_model'.
        Retorna caminho absoluto em formato POSIX.
        """
        candidates = []
        raw = Path(self.model_dir)
        candidates.append(raw)
        candidates.append(Path.cwd() / raw)
        candidates.append(pipe_dir / raw)
        candidates.append(pipe_dir / "hf_model")
        for c in candidates:
            if c.is_dir():
                return c.resolve().as_posix()
        raise FileNotFoundError(f"Diretorio do modelo HF não encontrado. Testados: {[str(c) for c in candidates]}")

    def _lazy_load(self, pipe_dir: Path):
        if self._model is None:
            torch, AutoTokenizer, AutoModel = _lazy_import_hf()
            model_path = self._resolve_model_dir(pipe_dir)
            self._tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._id2label = getattr(self._model.config, "id2label", {0:"neg",1:"neu",2:"pos"})

    def fit(self, X, y=None):
        return self

    def predict(self, X, pipe_dir: Path = None):
        import numpy as np
        torch, *_ = _lazy_import_hf()
        if pipe_dir is None:
            pipe_dir = Path.cwd()
        self._lazy_load(pipe_dir)
        texts = list(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.infer_batch_size):
                batch = texts[i:i+self.infer_batch_size]
                enc = self._tok(
                    batch, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt"
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                preds.extend([self._id2label.get(i, "neu") for i in ids])
        return np.array(preds)


# ---------------------------------------------------------
# Meta-colunas -> tokens (para pipelines que esperam "__meta_tokens")
# ---------------------------------------------------------
def _norm_yesno(val: str) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    if s in {"1","y","yes","true","sim"}:
        return "sim"
    if s in {"0","n","no","false","nao","não"}:
        return "nao"
    if s in {"", "nan", "none", "null"}:
        return "unk"
    return re.sub(r"\s+", "_", s)

def make_meta_tokens(df: pd.DataFrame, cols):
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    pieces = []
    for c in cols:
        if c not in df.columns:
            tokens = pd.Series(["unk"] * len(df), index=df.index).map(lambda v: f"{c}=unk")
            pieces.append(tokens)
            continue
        col = df[c].astype(str)
        uniq = set(col.str.lower().str.strip().unique())
        looks_bool = uniq <= {"0","1","y","n","yes","no","true","false","sim","nao","não","", "nan"}
        if looks_bool:
            tokens = col.map(_norm_yesno).map(lambda v: f"{c}={v}")
        else:
            tokens = col.fillna("").astype(str).str.strip().str.lower()
            tokens = tokens.replace({"": "unk", "nan": "unk"})
            tokens = tokens.str.replace(r"\s+", "_", regex=True).map(lambda v: f"{c}={v}")
        pieces.append(tokens)
    meta = pd.concat(pieces, axis=1).agg(" ".join, axis=1)
    return "[META] " + meta


# ---------------------------------------------------------
# Plot de matriz de confusão (com arredondamento por linha = 100%)
# ---------------------------------------------------------
def plot_confusion(cm, labels, title, out_path, dpi=300, cmap=None):
    """
    Plota matriz de confusão. Se 'cm' vier normalizada (valores entre 0 e 1),
    as anotações são percentuais inteiros por linha somando 100% exatos
    (método do maior resto). Caso contrário, mostra contagens inteiras.
    """
    cm = np.asarray(cm)
    eps = 1e-9
    is_norm = (cm.dtype.kind in "fc") and (np.nanmin(cm) >= -eps) and (np.nanmax(cm) <= 1.0 + eps)

    if cmap is None:
        cmap = "GnBu" if is_norm else "PuBu"

    norm = Normalize(vmin=0, vmax=1) if is_norm else None

    # arredonda por linha pra garantir que soma seja 100
    def row_round_to_100(row_fractions):
        p = np.clip(np.asarray(row_fractions, dtype=float) * 100.0, 0.0, None)
        base = np.floor(p).astype(int)
        rema = p - base
        deficit = 100 - int(base.sum())
        if deficit > 0:
            idx = np.argsort(rema)[::-1][:deficit]
            base[idx] += 1
        elif deficit < 0:
            idx = np.argsort(rema)[:(-deficit)]
            base[idx] -= 1
        return base

    if is_norm:
        rounded_pct = np.vstack([row_round_to_100(cm[i, :]) for i in range(cm.shape[0])])
        annot = np.empty_like(rounded_pct, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{int(rounded_pct[i, j])}%"
    else:
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{int(cm[i, j])}"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    im = ax.imshow(cm, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    ax.set_title(title, fontsize=16, color="black")
    ax.set_xlabel("Valores Preditos", color="black")
    ax.set_ylabel("Valores Verdadeiros", color="black")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", color="black", fontsize=12)
    ax.set_yticklabels(labels, color="black", fontsize=12)
    ax.tick_params(axis="both", which="both", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    ax.set_xticks(np.arange(-.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#e6e6e6", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporção" if is_norm else "Contagem", color="black")
    cbar.outline.set_edgecolor("black")
    cbar.ax.yaxis.set_tick_params(color="black")
    for t in cbar.ax.get_yticklabels():
        t.set_color("black")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, annot[i, j],
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=6, foreground="black")]
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def top_features_linear_svc(pipeline, k=20):
    """Retorna top features (apenas TF-IDF + clf linear com coef_)."""
    try:
        vec = (pipeline.named_steps.get("vec") or pipeline.named_steps.get("tfidf"))
        clf = pipeline.named_steps.get("clf")
    except Exception:
        return None
    if vec is None or clf is None:
        return None
    try:
        feature_names = np.array(vec.get_feature_names_out())
    except Exception:
        return None
    if not hasattr(clf, "coef_"):
        return None
    classes = clf.classes_
    tops = {}
    for idx, cls in enumerate(classes):
        coefs = clf.coef_[idx]
        top_pos_idx = np.argsort(coefs)[-k:][::-1]
        tops[str(cls)] = feature_names[top_pos_idx].tolist()
    return tops

from sklearn.base import BaseEstimator, TransformerMixin

class StopwordRemover(BaseEstimator, TransformerMixin):
    """Remove tokens que estão no conjunto de stopwords.
    OBS: no predict, o próprio objeto desserializado já carrega o set salvo no treino.
    """
    def __init__(self, stopwords=None):
        # No predict, esse valor virá do objeto serializado; ainda assim, deixamos um default seguro.
        self.stopwords = set(stopwords) if stopwords is not None else set()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            toks = ("" if text is None else str(text)).split()
            kept = [t for t in toks if t not in self.stopwords]
            out.append(" ".join(kept))
        return out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report_dir", default="reports")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--text_cols", default=None)
    ap.add_argument("--target_col", default="sentiment")
    args = ap.parse_args()

    # 1) Carrega pipeline e metadados
    pipe_path = Path(args.pipeline).resolve()
    pipe = joblib.load(pipe_path)
    exp_tag, model_name, train_csv_name, meta = exp_tag_from_pipeline(args.pipeline)

    # Se o classificador for HFTextClassifier, ele precisa saber a pasta do pipeline
    pipe_dir = pipe_path.parent
    clf = None
    try:
        clf = pipe.named_steps["clf"]
    except Exception:
        pass
    if isinstance(clf, HFTextClassifier):
        # força o resolve dentro do próprio objeto (quando ele lazy-loadar)
        clf._lazy_load(pipe_dir)  # faz o resolve + carrega tokenizer/modelo

    # 2) Lê CSV e normaliza header
    df = robust_read_csv(args.input, sep=args.sep, encoding=args.encoding)
    df = normalize_columns(df)

    # 3) Reconstrói __meta_tokens se o modelo foi treinado com meta_as_tokens=1
    meta_cols = []
    expects_meta = False
    if isinstance(meta, dict):
        meta_info = meta.get("meta") or {}
        expects_meta = bool(meta_info.get("as_tokens", False))
        meta_cols = meta_info.get("columns") or []
    if expects_meta and meta_cols:
        df["__meta_tokens"] = make_meta_tokens(df, meta_cols)

    # 4) Descobre colunas de texto do pipeline (ou usa --text_cols)
    if args.text_cols:
        text_cols = [c.strip().lower() for c in args.text_cols.split(",") if c.strip()]
    else:
        try:
            concat = pipe.named_steps["concat"]
        except Exception:
            concat = None
        text_cols = getattr(concat, "text_cols", None) or ("review_title", "review_text")

    # 5) Garante que as colunas esperadas existem
    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        avail = list(df.columns)
        raise RuntimeError(
            "As seguintes colunas esperadas pelo pipeline não estão no CSV de entrada: "
            f"{missing}\nDisponíveis: {avail}\n"
            "Dica: confira se normalizou o header (minúsculas, '_' no lugar de espaços) "
            "e se usou o mesmo CSV schema do treino."
        )

    # 6) Predição — use sempre o pipeline (ele aplica concat + clean antes do clf)
    preds = pipe.predict(df)

    # (opcional) sanity check
    if len(preds) != len(df):
        raise RuntimeError(
            f"Tamanho das predições ({len(preds)}) difere do número de linhas do CSV ({len(df)}). "
            "Isso geralmente ocorre quando o classificador é chamado diretamente com o DataFrame, "
            "em vez de passar pelo pipeline (concat + clean)."
        )


    out_df = df.copy()
    out_df["predicted_sentiment"] = preds

    # 7) Diretório de relatórios
    report_dir = Path(args.report_dir)
    if args.report_dir == "reports":  # default: salva ao lado do pipeline
        report_dir = pipe_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 8) Salva predições
    out_path = Path(args.output)
    if out_path.name in ("preds.csv", "preds"):
        out_path = report_dir / f"preds_{exp_tag}.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Predictions written to: {out_path}")

    # 9) Avaliação (se houver ground truth)
    target = args.target_col.strip().lower()
    if target in df.columns and df[target].notna().any():
        label_map_pt = {
            "negative": "neg", "negativo": "neg", "neg": "neg",
            "neutral":  "neu", "neutro":  "neu", "neu": "neu",
            "positive": "pos", "positivo": "pos", "pos": "pos",
        }
        y_true = df[target].astype(str).str.lower().str.strip().map(lambda s: label_map_pt.get(s, s))
        y_pred = out_df["predicted_sentiment"].astype(str).str.lower().str.strip().map(lambda s: label_map_pt.get(s, s))

        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            (report_dir / "README.txt").write_text(
                "Sem amostras válidas para avaliação após filtrar rótulos nulos.\n",
                encoding="utf-8"
            )
            print("[INFO] Sem amostras válidas para avaliação após filtrar NaNs.")
            return

        all_labels = sorted(set(y_true.dropna()) | set(y_pred.dropna()))
        preferred = [c for c in ("neg", "neu", "pos") if c in all_labels]
        labels = preferred if preferred else all_labels
        if len(labels) == 0:
            (report_dir / "README.txt").write_text(
                "Nenhuma classe disponível para avaliação.\n",
                encoding="utf-8"
            )
            print("[INFO] Nenhum label disponível para avaliação.")
            return

        cm_path  = report_dir / f"confusion_matrix_{exp_tag}.png"
        cmn_path = report_dir / f"confusion_matrix_normalized_{exp_tag}.png"
        cr_csv   = report_dir / f"classification_report_{exp_tag}.csv"
        acc_path = report_dir / f"accuracy_{exp_tag}.txt"
        tf_path  = report_dir / f"top_features_{exp_tag}.txt"
        md_path  = report_dir / f"evaluation_report_{exp_tag}.md"

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        plot_confusion(cm, labels, "Matriz de Confusão", cm_path)
        plot_confusion(cm_norm, labels, "Matriz de Confusão (Normalizada)", cmn_path)

        cls_report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(cls_report).transpose()
        cr_df.to_csv(cr_csv, index=True, encoding="utf-8")

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=labels)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels)
        f1_micro    = f1_score(y_true, y_pred, average="micro",    labels=labels)

        def _safe_f1(cls):
            try:
                return float(cls_report[cls]["f1-score"])
            except Exception:
                return float("nan")
        f1_neg = _safe_f1("neg")
        f1_neu = _safe_f1("neu")
        f1_pos = _safe_f1("pos")

        acc_path.write_text(f"{acc:.6f}", encoding="utf-8")
        (report_dir / f"f1_{exp_tag}.txt").write_text(
            "f1_macro={:.6f}\n"
            "f1_weighted={:.6f}\n"
            "f1_micro={:.6f}\n"
            "balanced_accuracy={:.6f}\n"
            "f1_neg={:.6f}\n"
            "f1_neu={:.6f}\n"
            "f1_pos={:.6f}\n".format(
                f1_macro, f1_weighted, f1_micro, bal_acc, f1_neg, f1_neu, f1_pos
            ),
            encoding="utf-8"
        )

        tops = top_features_linear_svc(pipe, k=20)
        if tops is not None:
            with open(tf_path, "w", encoding="utf-8") as f:
                for cls, feats in tops.items():
                    f.write(f"[{cls}]\n")
                    for w in feats:
                        f.write(f"  {w}\n")
                    f.write("\n")

        md = []
        md.append("# B2W Sentiment — Relatório de Avaliação")
        md.append("")
        md.append(f"- **Cenário**: `{exp_tag}`")
        if model_name: md.append(f"- **Modelo**: {model_name}")
        if train_csv_name: md.append(f"- **Dataset de treino**: {train_csv_name}")
        md.append(f"- **Amostras**: {len(out_df)}")
        md.append(f"- **F1 macro**: {f1_macro:.4f}")
        md.append(f"- **F1 weighted**: {f1_weighted:.4f}")
        md.append(f"- **Acurácia**: {acc:.4f}")
        md.append(f"- **Balanced accuracy**: {bal_acc:.4f}")
        md.append(f"- **F1 por classe** — neg: {f1_neg:.4f} | neu: {f1_neu:.4f} | pos: {f1_pos:.4f}")
        md.append("")
        md.append("## Matrizes de Confusão")
        md.append(f"![Matriz de Confusão]({cm_path.name})")
        md.append(f"![Matriz de Confusão (Normalizada)]({cmn_path.name})")
        md.append("")
        md.append("## Relatório de Classificação (CSV)")
        md.append(f"- `{cr_csv.name}`")
        if tops is not None:
            md.append("")
            md.append("## Top Features por classe (LinearSVC)")
            for cls, feats in tops.items():
                md.append(f"- **{cls}**: " + ", ".join(feats))
        md_path.write_text("\n".join(md), encoding="utf-8")

        print(f"[OK] Artefatos da avaliação do modelo em: {report_dir}")
    else:
        (report_dir / "README.txt").write_text(
            "Coluna de rótulos de referência não encontrada ou vazia; somente predições foram geradas.\n",
            encoding="utf-8"
        )
        print("[INFO] Coluna de rótulos de referência não encontrada. Ignorando avaliação.")


if __name__ == "__main__":
    main()
