#!/usr/bin/env python3
"""
Treinador unificado B2W Sentiment (PT) com TF-IDF/W2V + (NB/LogReg/SVM/RF) ou BERTimbau.

O que este script faz
---------------------
1) Lê um CSV de treino, normaliza nomes de colunas e escolhe automaticamente
   as colunas de texto (ou usa as passadas em --text_cols).
2) Concatena e limpa os textos com SimpleCleaner para o pré-processamento.
3) Remove stopwords em PT-BR (opcional) nos modelos clássicos.
4) Treina um dos algoritmos clássicos 
    - Multinomial Naive Baiyes
    - Complement Naive Baiyes
    - Bernoulli Naive Baiyes
    - Regressao Logistica
    - Support Vector Machine
    - Random Forest (obs: não testado nesta versão)
    - Fine Tuning com BERT (BERTimbau)
5) Normaliza sentimeno para "neg"/"neu"/"pos" segundo critério a partir do overall_rating:
    1-2 --> Review Negativo
    3 --> Review Neutro
    4-5 --> Review Positivo.
6) Salva os artefatos em: artifacts/<model_label>.
   - Algoritmos Clássicos: pipeline.joblib + metadata.json
   - BERT: hf_model/ + pipeline.joblib com HFTextClassifier + metadata.json
"""
# Imports gerais
from __future__ import annotations
import argparse, json, re, sys, shlex
from pathlib import Path
from typing import Optional, Tuple, List, Set
import pandas as pd
import joblib
# =========================
#  Imports para SciKit-Learn
# =========================
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin

# =========================
#  Pré-processamento do dataset
# =========================
try:
    from tcc_components import TextConcatenator, SimpleCleaner  # type: ignore
except Exception:
    class TextConcatenator(BaseEstimator, TransformerMixin):
        """Concatena colunas de texto em uma única string por linha.
        Ex.: (review_title, review_text) -> "<title> \n <text>"
        """
        def __init__(self, text_cols=("review_title", "review_text")):
            self.text_cols = tuple(text_cols)
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            df = pd.DataFrame(X).copy()
            return df[list(self.text_cols)].fillna("").astype(str).agg(" \n ".join, axis=1)
    class SimpleCleaner(BaseEstimator, TransformerMixin):
        """Limpeza simples: converte texto para minúsculas, limpa espaços e pontas do texto."""
        def __init__(self):
            import re as _re
            self._ws = _re.compile(r"\s+")
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return [self._ws.sub(" ", ("" if t is None else str(t)).lower()).strip() for t in X]

# ---------- Remoção de Stopwords ----------
#
DEFAULT_PT_STOPWORDS: Set[str] = {
    # Stopwords comuns (se não passar arquivo de parametro com stopwords, assume estas)
    "a","à","agora","ainda","além","algo","algumas","alguns","ali",
    "ao","aos","apenas","apoio","após","aquela","aquelas","aquele","aqueles",
    "aqui","as","às","Assim","até","através","cada","cerca","com","como",
    "da","daquela","daquelas","daquele","daqueles","das","de","dela","dele","deles",
    "demais","depois","desde","dessa","dessas","desse","desses","desta","destas",
    "deste","destes","deve","devem","devendo","dever","disse","disso","disto","dito",
    "diz","dizem","do","dos","e","é","ela","elas","ele","eles","em","enquanto","entre","era",
    "essa","essas","esse","esses","esta","está","estão","estas","estava","estavam","isso","isto",
    "já","la","lá","lhe","lhes","lo","logo","longe","mais","mas","máximo",
    "me","mesma","mesmas","mesmo","mesmos","meu","meus","minha","minhas",
    "na","nas","nem","nenhum","nos","nós","os","ou","outra","outras","outro","outros","para",
    "pela","pelas","pelo","pelos","pequena","pequeno","pequenos","poderia","podiam",
    "por","porque","porquê","pouca","pouco","poucos","primeiro","primeiros","própria","próprias",
    "próprio","próprios","quais","qual","qualquer","quando","quanto","que","são","se","sem",
    "sempre","ser","será","só","sob","sobre","tal","também","te","tem","têm","ter","toda",
    "todas","todo","todos","um","uma","umas","uns","você","vocês"
}

class StopwordRemover(BaseEstimator, TransformerMixin):
    """Classe para remover tokens que pertencem a um conjunto de stopwords."""
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.stopwords = set(stopwords) if stopwords else set()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = []
        for text in X:
            toks = ("" if text is None else str(text)).split()
            kept = [t for t in toks if t not in self.stopwords]
            out.append(" ".join(kept))
        return out

# =========================
#  Vetorizador Word2Vec (gensim)
# =========================
class W2VVectorizer:
    """
    Converte documentos em vetores densos (np.array),
    usando média dos embeddings.
    - Se w2v_model for passado, carrega KeyedVectors/word2vec binário.
    - Senão, treina Word2Vec no próprio dataset de treino (tokenização simples por espaço).
    """
    def __init__(self, w2v_model=None, size=300, window=5, min_count=2, sg=1, epochs=15, workers=-1, tfidf_weight=False, tfidf_params=None):
        self.w2v_model = w2v_model
        self.size = size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers if workers != -1 else None
        self.tfidf_weight = bool(tfidf_weight)
        self.tfidf_params = tfidf_params or {}
        self.kv = None           # gensim KeyedVectors
        self.idf_ = {}           # dict palavra -> idf (se tfidf_weight)
        self.dim_ = int(size)

    def _tokenize(self, texts):
        # Assume que SimpleCleaner/StopwordRemover já fez o pré-processamento
        return [t.split() for t in texts]

    def fit(self, X, y=None):
        from gensim.models import Word2Vec, KeyedVectors
        texts = list(X)
        sents = self._tokenize(texts)

        # 1) Embeddings
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

        # 2) IDF (opcional)
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
            raise RuntimeError("W2VVectorizer não inicializado (kv=None). Objeto salvo deve conter os vetores.")
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
                    if self.tfidf_weight:
                        weights.append(self.idf_.get(tok, 1.0))
                    else:
                        weights.append(1.0)
            if vecs:
                V = np.vstack(vecs)
                w = (np.array(weights, dtype=np.float32) / (sum(weights) + 1e-9))[:, None]
                out[i] = (V * w).sum(axis=0)
        return out

# =========================
#  Imports Lazy para Transformers/Torch
# =========================
def _lazy_import_hf():
    """Importa Torch/Transformers só se necessário (contornar erro se não instalados)."""
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
            TrainerCallback,
        )
        return (
            torch,
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
            TrainerCallback,
        )
    except Exception:
        print("[ERRO] Para --algo bert : pip install transformers torch", file=sys.stderr)
        raise

# Aliases para os principais features usados na análise
DEFAULT_TEXT_ALIASES = [
    ("review_title", "review_text"),
    ("title", "review_text"),
    ("review_title", "text"),
    ("title", "text"),
    ("review_text",),
    ("text",),
]

def sanitize_label(s: str) -> str:
    """Normaliza rótulos para definir nomes de pasta/arquivo de saída (minúsculas e hífens)."""
    s = s.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9._-]+", "-", s).strip("-_.") or "model"

# Funções auxiliares para criação/organização dos metadados e tokens
def _drop_none(d: dict) -> dict:
    """Remove chaves com valores None do JSON ("limpeza" do arquivo)."""
    return {k: v for k, v in d.items() if v is not None}

def _norm_yesno(val: str) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    if s in {"1","y","yes","true","sim"}:
        return "sim"
    if s in {"0","n","no","false","nao","não"}:
        return "nao"
    if s in {"", "nan", "none", "null"}:
        return "unk"
    return re.sub(r"\s+", "_", s)

def make_meta_tokens(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """String única por linha com tokens do tipo 'col=valor', antecedidos por [META]."""
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

def parse_train_tag(train_csv_path: str) -> Tuple[str, str, Optional[int]]:
    """Cria tag a partir do nome dos arquivos de treino/validação
    para montar posteriormente nomes de artefatos, reports etc.."""
    name = Path(train_csv_path).name.lower()
    m = re.search(r'(?:^|[_-])(n?stratify)[_-](\d{1,3})(?=\.|[_-]|$)', name)
    if m:
        strategy = m.group(1)
        pct = int(m.group(2))
        if 0 < pct <= 100:
            return f"{strategy}_{pct}", strategy, pct
    return "unknown", "unknown", None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Regex para padronizar os nomes das colunas: apenas minúsculas, replace de espaço por _"""
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"\s+", " ", regex=True).str.strip().str.lower().str.replace(" ", "_")
    )
    return df

def read_csv_robust(path: str, sep: Optional[str], encoding: str) -> pd.DataFrame:
    """Leitura de CSV: tenta inferir separador se não indicado."""
    if sep is not None:
        return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding=encoding, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=",", encoding=encoding, low_memory=False)

def derive_sentiment_from_rating(sr: pd.Series) -> pd.Series:
    """Mapeia nota 1..5 em rótulos ('neg','neu','pos')."""
    def _map(v):
        try:
            r = int(v)
        except Exception:
            return None
        if r in (1, 2):
            return "neg"
        if r == 3:
            return "neu"
        if r in (4, 5):
            return "pos"
        return None
    return sr.map(_map)

def pick_text_cols(df: pd.DataFrame, user_cols: Optional[List[str]]) -> tuple[str, ...]:
    """Escolhe colunas de texto passadas como parametro ou tenta encontrá-las se não fornecido."""
    if user_cols:
        cols = [c.strip().lower() for c in user_cols if c.strip()]
        return tuple(cols)
    for cand in DEFAULT_TEXT_ALIASES:
        if all(c in df.columns for c in cand):
            return cand
    raise ValueError(
        "Não encontradas colunas de texto. Procurado: "
        + ", ".join([" + ".join(c) for c in DEFAULT_TEXT_ALIASES])
        + f". Colunas: {list(df.columns)}"
    )

def compute_balanced_sample_weight(y: pd.Series) -> pd.Series:
    """Retorna peso por amostra para balancear classes:
      w_c = n_samples / (n_classes * count_c)."""
    vc = y.value_counts()
    n = len(y)
    k = len(vc)
    w_per_class = {c: n / (k * cnt) for c, cnt in vc.items()}
    return y.map(w_per_class)

def build_classifier(algo: str, args):
    """Instancia o classificador usando argumentos passados na linha de comando."""
    algo = algo.lower()
    if algo == "mnb":
        return MultinomialNB(alpha=args.alpha)
    if algo == "cnb":
        return ComplementNB(alpha=args.alpha)
    if algo == "bnb":
        return BernoulliNB(alpha=args.alpha, binarize=args.binarize)
    if algo == "logreg":
        class_weight = None if args.balance == "none" else "balanced"
        return LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            class_weight=class_weight,
            max_iter=args.max_iter,
            multi_class="ovr",
        )
    if algo == "svm":
        class_weight = None if args.balance == "none" else "balanced"
        return LinearSVC(C=args.svm_C, loss=args.svm_loss, class_weight=class_weight)
    if algo == "rf":
        class_weight = None if args.balance == "none" else "balanced"
        rf_max_features = args.rf_max_features
        try:
            rf_max_features = float(rf_max_features)
        except Exception:
            pass
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=rf_max_features,
            bootstrap=bool(args.rf_bootstrap),
            n_jobs=args.rf_n_jobs,
            random_state=args.rf_random_state,
            class_weight=class_weight,
        )
    raise ValueError(f"Algoritmo não suportado (clássico): {algo}")

# =========================
#  Hugging Face (BERTimbau): predição leve + treinamento
# =========================
class HFTextClassifier:
    """Wrapper simples para usar modelos Hugging Face dentro de um pipeline sklearn."""
    def __init__(self, model_dir: str, max_length: int = 256, infer_batch_size: int = 32):
        self.model_dir = str(model_dir)
        self.max_length = int(max_length)
        self.infer_batch_size = int(infer_batch_size)
        self._tok = None
        self._model = None
        self._device = None
        self._id2label = None

    def _lazy_load(self):
        if self._model is None:
            torch, AutoTokenizer, AutoModelForSequenceClassification, *_ = _lazy_import_hf()
            self._tok = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()
            # Usa GPU se disponível, senão processa usando CPU
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._id2label = (
                self._model.config.id2label if hasattr(self._model, "config") else {0: "neg", 1: "neu", 2: "pos"}
            )
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        import numpy as np
        torch, *_ = _lazy_import_hf()
        self._lazy_load()
        texts = list(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.infer_batch_size):
                batch = texts[i : i + self.infer_batch_size]
                enc = self._tok(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                preds.extend([self._id2label.get(i, "neu") for i in ids])
        return np.array(preds)
# Treinamento com BERT
def train_with_bert(
    exp_dir: Path,
    texts: List[str],
    labels_str: List[str],
    text_cols: tuple,
    args,
):
    """Fine-tuning BERTimbau, grava artefatos HF + pipeline sklearn."""
    (
        torch,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        *rest,
    ) = _lazy_import_hf()

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_warning()
    model_name = args.hf_model_name
    max_len = args.hf_max_length

    # Mapeamento de rótulo
    label2id = {"neg": 0, "neu": 1, "pos": 2}
    id2label = {0: "neg", 1: "neu", 2: "pos"}
    y_ids = [label2id[s] for s in labels_str]

    # Pesos de classe (opcional)
    class_weights = None
    if args.balance == "balanced":
        from collections import Counter
        cnt = Counter(y_ids)
        total = sum(cnt.values())
        k = len(cnt)
        w = [total / (k * cnt[i]) for i in range(k)]
        class_weights = torch.tensor(w, dtype=torch.float)

    # Tokenizer + dataset
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    class TextDS(torch.utils.data.Dataset):
        def __init__(self, texts, y, tok, max_len):
            self.texts = texts
            self.y = y
            self.tok = tok
            self.max_len = max_len
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            enc = self.tok(self.texts[idx], truncation=True, padding=False, max_length=self.max_len)
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.y[idx], dtype=torch.long)
            return item

    ds_train = TextDS(texts, y_ids, tok, max_len)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=(8 if args.hf_fp16 else None))

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )
    # Usa GPU se disponível, senão CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if bool(args.hf_fp16) and device != "cuda":
        print("[aviso] --hf_fp16 ignorado: não há CUDA disponível.", file=sys.stderr)

    if getattr(args, "hf_freeze_encoder", 0) == 1:
        for name, p in model.named_parameters():
            if name.startswith("bert."):
                p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] device: {device} | trainable params: {trainable_params:,} | "
          f"batch={args.hf_batch_size} x accum={args.hf_grad_accum} "
          f"(efetivo={args.hf_batch_size * args.hf_grad_accum})")
    if getattr(args, "hf_freeze_encoder", 0) == 1:
        print("[info] encoder congelado; treinando apenas a cabeça de classificação.")

    # Treinamento
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            self.class_weights = class_weights
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    # Define diretorios de saída
    out_hf  = exp_dir / "hf_model"   # modelo final (inferência)
    out_ckpt = exp_dir / "hf_ckpt"   # checkpoints de treino
    out_hf.mkdir(parents=True, exist_ok=True)
    out_ckpt.mkdir(parents=True, exist_ok=True)

    # Argumentos do treino
    targs = TrainingArguments(
        output_dir=str(out_ckpt),
        num_train_epochs=args.hf_epochs,
        per_device_train_batch_size=args.hf_batch_size,
        gradient_accumulation_steps=args.hf_grad_accum,
        learning_rate=args.hf_lr,
        weight_decay=args.hf_weight_decay,
        warmup_ratio=args.hf_warmup_ratio,
        logging_steps=max(50, args.hf_save_steps // 5),
        fp16=bool(args.hf_fp16),
        seed=args.hf_seed,
        report_to="none",
        remove_unused_columns=False,
        save_steps=args.hf_save_steps,
        save_total_limit=args.hf_save_total_limit
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=ds_train,
        data_collator=collator,
        tokenizer=tok,
    )

    print("[info] Treinando (fine-tuning) BERTimbau...")
    resume_arg = None
    if args.hf_resume == "auto":
        resume_arg = True
    elif args.hf_resume:
        resume_arg = args.hf_resume  # caminho específico

    if resume_arg is not None:
        trainer.train(resume_from_checkpoint=resume_arg)
    else:
        trainer.train()

    print("[info] Salvando modelo HF...")
    trainer.save_model(str(out_hf))
    tok.save_pretrained(str(out_hf))

    # Pipeline sklearn para inferencia
    print("[info] Gerando pipeline.joblib com HFTextClassifier...")
    pipe = Pipeline(
        [
            ("concat", TextConcatenator(text_cols=text_cols)),
            ("clean", SimpleCleaner()),
            # Por padrão, não remover stopwords para BERT
            ("clf", HFTextClassifier(model_dir=str(out_hf), max_length=max_len, 
                                     infer_batch_size=args.hf_infer_bs)),
        ]
    )
    joblib.dump(pipe, exp_dir / "pipeline.joblib")

    return {
        "label2id": label2id,
        "id2label": id2label,
        "hf_model_dir": str(out_hf),
        "device": device,
        "trainable_params": int(trainable_params),
    }

# =========================
# ============ MAIN 
# =========================
def _expand_args_file(argv: list[str]) -> list[str]:
    """Expande --args_file <caminho> lendo parâmetros (uma flag por linha)."""
    if "--args_file" in argv:
        i = argv.index("--args_file")
        if i == len(argv) - 1:
            raise SystemExit("--args_file requer caminho do arquivo")
        path = argv[i + 1]
        txt = Path(path).read_text(encoding="utf-8") # assumindo utf-8
        tokens = shlex.split(txt, comments=True, posix=True)
        return argv[:i] + argv[i + 2:] + tokens
    return argv

def _load_stopwords(args) -> Set[str]:
    """Carrega stopwords: de arquivo customizado ou do conjunto padrão sugerido no código."""
    if args.stopwords_file:
        p = Path(args.stopwords_file)
        if not p.exists():
            raise FileNotFoundError(f"--stopwords_file não encontrado: {p}")
        words = [w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]
        return set(words)
    return set(DEFAULT_PT_STOPWORDS)

def main():
    argv = _expand_args_file(sys.argv[1:])
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--args_file", default=None, help="Arquivo de texto com argumentos (uma flag por linha).")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--sep", default=None, help="Separador CSV (',' ou ';'). Se omitido, infere.")
    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--text_cols", default=None, help="Colunas de texto (separado por virgula)")
    ap.add_argument("--meta_cols", default="", help="Colunas extras para virar tokens (ex.: recommend_to_a_friend)") # Não ajudou, considerar remover
    ap.add_argument("--meta_as_tokens", type=int, choices=[0,1], default=1,
                    help="1=injetar colunas extras como tokens no texto")
    ap.add_argument("--target_col", default="sentiment")
    ap.add_argument("--model_label", required=True, help="SUbdiretorio de 'artifacts/' onde os artefatos são salvos.")
    ap.add_argument("--run_suffix", default="", help="Sufixo (opcional) do diretorio (ex.: v2).")

    # Representação (clássicos)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--tfidf_max_features", type=int, default=None,
                    help="Se definido, limita o vocabulário do TF-IDF para no máximo N features.")
    ap.add_argument("--vectorizer_mode", choices=["tfidf", "w2v"], default="tfidf",
                    help="tfidf (padrão) ou w2v (Word2Vector).")

    # Word2Vec (se usar --vectorizer_mode w2v)
    ap.add_argument("--w2v_model", default=None,
                    help="Caminho para embeddings pré-treinados (KeyedVectors .kv/.kv.zip ou word2vec .bin). Se não passado, usa o próprio dataset.")
    ap.add_argument("--w2v_size", type=int, default=300, help="Dimensão do vetor (treino do zero).")
    ap.add_argument("--w2v_window", type=int, default=5, help="Janela de contexto.")
    ap.add_argument("--w2v_min_count", type=int, default=2, help="Min count de tokens.")
    ap.add_argument("--w2v_sg", type=int, choices=[0,1], default=1, help="0=CBOW, 1=Skip-gram.")
    ap.add_argument("--w2v_epochs", type=int, default=15, help="Epochs de treino do W2V.")
    ap.add_argument("--w2v_workers", type=int, default=-1, help="Núcleo(s). -1 usa todos.")
    ap.add_argument("--w2v_tfidf_weight", type=int, choices=[0,1], default=0,
                    help="1 = pondera média por IDF (TF-IDF) das palavras.")

    # Stopwords
    ap.add_argument("--remove_stopwords", type=int, choices=[0,1], default=1,
                    help="1 = remover stopwords PT-BR (apenas modelos clássicos).")
    ap.add_argument("--stopwords_file", default=None,
                    help="Arquivo .txt com stopwords (uma por linha).")

    # Classificadores
    ap.add_argument("--algo", default="mnb", choices=["mnb", "cnb", "bnb", "logreg", "svm", "bert", "rf"])
    ap.add_argument("--balance", default="balanced", choices=["balanced", "none"],
                    help="Para logreg/svm usa class_weight; NB usa sample_weight; BERT usa loss ponderada.")

    # Redução de dimensionalidade (RF)
    ap.add_argument("--feat_reduce", choices=["none", "chi2", "svd"], default="none",
                    help="Redução de features para RF: 'chi2' (seleção) ou 'svd' (LSA).")
    ap.add_argument("--chi2_k", type=int, default=20000, help="Top-K features para SelectKBest(chi2).")
    ap.add_argument("--svd_components", type=int, default=300, help="Componentes para TruncatedSVD (LSA).")

    # Específico para Naive Bayes
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--binarize", type=float, default=0.0, help="Threshold do BernoulliNB")

    # Específico para Regressão Logística
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--penalty", default="l2", choices=["l2", "l1"])
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "saga"])
    ap.add_argument("--max_iter", type=int, default=1000)

    # Específico para SVM
    ap.add_argument("--svm_C", type=float, default=1.0)
    ap.add_argument("--svm_loss", default="squared_hinge", choices=["hinge", "squared_hinge"])

    # Específico para Hugging Face (BERT)
    ap.add_argument("--hf_model_name", default="models/bertimbau-base",
                    help="Caminho local do modelo/tokenizer (ou repo_id se online).")
    ap.add_argument("--hf_epochs", type=float, default=3.0)
    ap.add_argument("--hf_batch_size", type=int, default=16)
    ap.add_argument("--hf_grad_accum", type=int, default=1)
    ap.add_argument("--hf_lr", type=float, default=2e-5)
    ap.add_argument("--hf_weight_decay", type=float, default=0.01)
    ap.add_argument("--hf_warmup_ratio", type=float, default=0.1)
    ap.add_argument("--hf_max_length", type=int, default=256)
    ap.add_argument("--hf_infer_bs", type=int, default=32)
    ap.add_argument("--hf_seed", type=int, default=42)
    ap.add_argument("--hf_fp16", type=int, choices=[0, 1], default=0)
    ap.add_argument("--hf_freeze_encoder", type=int, choices=[0,1], default=0,
                    help="1 = congela o encoder e treina só a cabeça")
    ap.add_argument("--hf_save_steps", type=int, default=1000, help="Salva checkpoints a cada N steps.")
    ap.add_argument("--hf_save_total_limit", type=int, default=2, help="Máximo de checkpoints para manter.")
    ap.add_argument("--hf_resume", default="",
                    help='"" (padrão) = não retoma; "auto" = retoma do último checkpoint; ou passe um caminho.')

    # Específico para Random Forest
    ap.add_argument("--rf_n_estimators", type=int, default=300)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--rf_min_samples_split", type=int, default=2)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=1)
    ap.add_argument("--rf_max_features", default="sqrt",
                    help='sqrt|log2 ou fração (0,1]. "sqrt" por padrão. (Evite "auto", legado)')
    ap.add_argument("--rf_bootstrap", type=int, choices=[0,1], default=1)
    ap.add_argument("--rf_n_jobs", type=int, default=-1)
    ap.add_argument("--rf_random_state", type=int, default=42)

    args = ap.parse_args()

    # Diretório de saída
    tag_str, split_strategy, pct_train = parse_train_tag(args.train_csv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    label = sanitize_label(args.model_label)
    suffix = f"_{sanitize_label(args.run_suffix)}" if args.run_suffix else ""
    exp_dir = out_root / f"{label}{suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] diretório de artefatos: {exp_dir}")

    # Dados
    print("Lendo treino...")
    df_raw = read_csv_robust(args.train_csv, sep=args.sep, encoding=args.encoding)
    df = normalize_columns(df_raw)

    # Colunas de texto
    user_cols = [c.strip() for c in args.text_cols.split(",")] if args.text_cols else None
    if user_cols:
        user_cols = [c for c in user_cols if c]
    text_cols = pick_text_cols(df, user_cols)

    # Meta -> tokens (opcional)
    meta_cols = [c.strip().lower() for c in (args.meta_cols or "").split(",") if c.strip()]
    if args.meta_as_tokens and meta_cols:
        df["__meta_tokens"] = make_meta_tokens(df, meta_cols)
        text_cols = tuple(list(text_cols) + ["__meta_tokens"])

    print(f"[info] usando colunas de texto: {list(text_cols)}")
    if meta_cols:
        print(f"[info] meta cols (como tokens): {meta_cols}")

    # Alvo
    target = args.target_col.strip().lower()
    if target not in df.columns or df[target].isna().all():
        if "overall_rating" not in df.columns:
            raise ValueError(
                f"Target '{target}' não encontrado e 'overall_rating' ausente para derivação. Colunas: {list(df.columns)}"
            )
        df[target] = derive_sentiment_from_rating(df["overall_rating"])

    # Rótulos
    y_raw = df[target].astype(str).str.lower().str.strip()
    label_map_pt = {
        "negative": "neg", "negativo": "neg", "neg": "neg",
        "neutral": "neu", "neutro": "neu", "neu": "neu",
        "positive": "pos", "positivo": "pos", "pos": "pos",
    }
    y = y_raw.map(lambda s: label_map_pt.get(s, s))
    valid_mask = y.isin({"neg", "neu", "pos"})
    if valid_mask.sum() < len(y):
        print(f"[aviso] removendo {(~valid_mask).sum()} linhas com rótulos inválidos.")
    y = y[valid_mask]
    X_df = df.loc[valid_mask, list(text_cols)].copy()

    # Define Stopwords
    stopwords_used: Set[str] = set()
    if args.remove_stopwords:
        stopwords_used = _load_stopwords(args)
        print(f"[info] stopwords: ON ({len(stopwords_used)} termos{' - arquivo custom' if args.stopwords_file else ' - builtin PT-BR'})")
    else:
        print("[info] stopwords: OFF")

    # Caminho 1: BERT
    if args.algo.lower() == "bert":
        concat = TextConcatenator(text_cols=text_cols)
        clean = SimpleCleaner()
        texts_concat = concat.transform(X_df)
        texts_clean = clean.transform(texts_concat)

        if args.remove_stopwords:
            print("[info] Remoção de stopwords NÃO aplicada para BERT.")

        hf_info = train_with_bert(
            exp_dir=exp_dir,
            texts=list(texts_clean),
            labels_str=list(y),
            text_cols=text_cols,
            args=args,
        )

        bert_params = _drop_none({
            "balance": args.balance,
            "hf_model_name": args.hf_model_name,
            "epochs": args.hf_epochs,
            "batch_size": args.hf_batch_size,
            "grad_accum": args.hf_grad_accum,
            "effective_batch_size": args.hf_batch_size * args.hf_grad_accum,
            "lr": args.hf_lr,
            "weight_decay": args.hf_weight_decay,
            "warmup_ratio": args.hf_warmup_ratio,
            "max_length": args.hf_max_length,
            "infer_batch_size": args.hf_infer_bs,
            "seed": args.hf_seed,
            "fp16": bool(args.hf_fp16),
            "freeze_encoder": bool(getattr(args, "hf_freeze_encoder", 0)),
            "save_steps": getattr(args, "hf_save_steps", None),
            "save_total_limit": getattr(args, "hf_save_total_limit", None),
            "resume": getattr(args, "hf_resume", None),
            "device": hf_info.get("device"),
            "trainable_params": hf_info.get("trainable_params"),
        })

        meta = {
            "artifact_dir": str(exp_dir.resolve()),
            "artifacts": {
                "pipeline": "pipeline.joblib",
                "metadata": "metadata.json",
                "hf_model_dir": "hf_model",
            },
            "source_train_csv": str(Path(args.train_csv).resolve()),
            "train_tag": tag_str,
            "train_split_strategy": split_strategy,
            "train_pct": pct_train,
            "n_samples": int(len(y)),
            "text_columns": list(text_cols),
            "meta": {"as_tokens": bool(args.meta_as_tokens), "columns": meta_cols},
            "target_column": target,
            "model_label": label,
            "classes": ["neg", "neu", "pos"],
            "class_distribution": y.value_counts().to_dict(),
            "vectorizer": {"type": "hf"},
            "model": "bert",
            "model_params": bert_params,
            "label2id": hf_info["label2id"],
            "id2label": hf_info["id2label"],
            "stopwords": {"enabled": False, "count": 0, "source": None},
        }
        (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK! Artefatos salvos em: {exp_dir}")
        print(" - pipeline: pipeline.joblib")
        print(" - metadata: metadata.json")
        print(" - hf_model/: modelo Hugging Face")
        return

    # Caminho 2: Clássicos (representação + algoritmo de classificação)
    if args.vectorizer_mode == "tfidf":
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=False,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.tfidf_max_features,
        )
        vec_meta = {
            "type": "tfidf",
            "ngram_range": (1, args.ngram_max),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "strip_accents": "unicode",
            "max_features": args.tfidf_max_features,
        }
    else:  # Word2Vec
        tfidf_params = dict(min_df=args.min_df, max_df=args.max_df)
        vectorizer = W2VVectorizer(
            w2v_model=args.w2v_model,
            size=args.w2v_size,
            window=args.w2v_window,
            min_count=args.w2v_min_count,
            sg=args.w2v_sg,
            epochs=args.w2v_epochs,
            workers=args.w2v_workers,
            tfidf_weight=bool(args.w2v_tfidf_weight),
            tfidf_params=tfidf_params
        )
        vec_meta = {
            "type": "w2v",
            "source": ("pretrained" if args.w2v_model else "trained_on_corpus"),
            "size": args.w2v_size,
            "window": args.w2v_window,
            "min_count": args.w2v_min_count,
            "sg": args.w2v_sg,
            "epochs": args.w2v_epochs,
            "tfidf_weight": bool(args.w2v_tfidf_weight),
        }

    clf = build_classifier(args.algo, args)

    steps = [
        ("concat", TextConcatenator(text_cols=text_cols)),
        ("clean",  SimpleCleaner()),
    ]
    if args.remove_stopwords and len(stopwords_used) > 0:
        steps.append(("stopw", StopwordRemover(stopwords=stopwords_used)))
    steps.append(("vec", vectorizer))

    # Redução (opcional) - só faz sentido com TF-IDF; com W2V já é denso/compacto.
    if args.vectorizer_mode == "tfidf" and args.algo == "rf" and args.feat_reduce != "none":
        if args.feat_reduce == "chi2":
            steps.append(("reduce", SelectKBest(score_func=chi2, k=args.chi2_k)))
        elif args.feat_reduce == "svd":
            steps.append(("svd", TruncatedSVD(n_components=args.svd_components, random_state=42)))
            steps.append(("norm", Normalizer(copy=False)))
    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    print(f"Treinando o modelo ({args.algo})...")
    if args.algo in {"mnb", "cnb", "bnb"} and args.balance == "balanced":
        sw = compute_balanced_sample_weight(y)
        pipe.fit(X_df, y, clf__sample_weight=sw.values)
    else:
        pipe.fit(X_df, y)

    # Salva artefatos
    pipe_path = exp_dir / "pipeline.joblib"
    meta_path = exp_dir / "metadata.json"
    joblib.dump(pipe, pipe_path)

    # Salva Metadados (algoritmos clássicos)
    vectorizer_meta = _drop_none(vec_meta)
    classic_params = _drop_none({
        "balance": args.balance,
        # Naive Bayes
        "alpha": args.alpha if args.algo in {"mnb","cnb","bnb"} else None,
        "bnb_binarize": args.binarize if args.algo == "bnb" else None,
        # Regressão Logística
        "C": args.C if args.algo == "logreg" else None,
        "penalty": args.penalty if args.algo == "logreg" else None,
        "solver": args.solver if args.algo == "logreg" else None,
        "max_iter": args.max_iter if args.algo == "logreg" else None,
        "multi_class": "ovr" if args.algo == "logreg" else None,
        # SVM
        "svm_C": args.svm_C if args.algo == "svm" else None,
        "svm_loss": args.svm_loss if args.algo == "svm" else None,
        # Random Forest
        "rf_n_estimators": args.rf_n_estimators if args.algo == "rf" else None,
        "rf_max_depth": args.rf_max_depth if args.algo == "rf" else None,
        "rf_min_samples_split": args.rf_min_samples_split if args.algo == "rf" else None,
        "rf_min_samples_leaf": args.rf_min_samples_leaf if args.algo == "rf" else None,
        "rf_max_features": args.rf_max_features if args.algo == "rf" else None,
        "rf_bootstrap": bool(args.rf_bootstrap) if args.algo == "rf" else None,
        "feat_reduce": args.feat_reduce if args.algo == "rf" else None,
        "chi2_k": args.chi2_k if (args.algo == "rf" and args.feat_reduce == "chi2") else None,
        "svd_components": args.svd_components if (args.algo == "rf" and args.feat_reduce == "svd") else None,
    })

    meta = {
        "artifact_dir": str(exp_dir.resolve()),
        "artifacts": {"pipeline": "pipeline.joblib", "metadata": "metadata.json"},
        "source_train_csv": str(Path(args.train_csv).resolve()),
        "train_tag": tag_str,
        "train_split_strategy": split_strategy,
        "train_pct": pct_train,
        "n_samples": int(len(y)),
        "text_columns": list(text_cols),
        "meta": {"as_tokens": bool(args.meta_as_tokens), "columns": meta_cols},
        "target_column": target,
        "model_label": label,
        "classes": sorted(list(set(y))),
        "class_distribution": y.value_counts().to_dict(),
        "vectorizer": vectorizer_meta,
        "model": args.algo,
        "model_params": classic_params,
        "stopwords": {
            "enabled": bool(args.remove_stopwords),
            "count": int(len(stopwords_used)),
            "source": ("custom_file" if args.stopwords_file else ("builtin_pt" if args.remove_stopwords else None)),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK! Artefatos salvos em: {exp_dir}")
    print(f" - pipeline: {pipe_path.name}")
    print(f" - metadata: {meta_path.name}")

if __name__ == "__main__":
    main()