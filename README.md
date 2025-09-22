# TCC ICMC — Sentiment Analysis (B2W Reviews, PT-BR)

Este repositório reúne os **scripts exatamente como usados** na monografia do MBA para análise de sentimentos **em Português**. O objetivo é **reprodutibilidade** e **transparência** — sem alterar o código.

> **Importante:** Os scripts originais não foram modificados. Este README e os demais documentos servem para orientar a execução/uso no GitHub.

## Dataset de referência (B2W Reviews)

O trabalho original foi baseado no dataset público **B2W Reviews** (PT-BR).  
Repositório oficial do dataset: **https://github.com/thiagorainmaker77/b2w-reviews01**

> Este repositório **não inclui** os arquivos do dataset. Se você quiser reproduzir os resultados, baixe o dataset no link acima e aponte os scripts para seus CSVs locais.

## Estrutura sugerida do repositório

```
tcc_icmc_sentiment_analysis/
├─ tcc_components.py
├─ tcc_split_dataset.py
├─ tcc_train_sentiment_analysis_unified.py
├─ tcc_predict_eval_sentiment.py
├─ requirements.txt
├─ .gitignore
├─ LICENSE-CODE          # MIT (código)
├─ LICENSE-DATA          # CC BY-NC 4.0 (dados/relatórios)
├─ DISCLAIMER.md
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ CITATION.cff
├─ MODEL_CARD.md
└─ docs/
   ├─ examples/
   │  ├─ @args_train_example.txt
   │  └─ README-examples.md
   └─ results/           # artefatos/relatórios gerados em execução (opcional no .gitignore)
```

## Scripts principais e uso básico

### 1) `tcc_split_dataset.py` — Split de treino/validação
Divide o CSV bruto em treino/validação, com opção de **estratificação por classe**.

Parâmetros relevantes:
- `--sep`             : separador do CSV (`,` ou `;`)
- `--test-size`       : proporção para validação (padrão: `0.2`)
- `--random-state`    : semente (padrão: `42`)
- `--no-stratify`     : se presente, **desativa** a estratificação

Exemplo:
```bash
python tcc_split_dataset.py   --sep ","   --test-size 0.2   --random-state 42   caminho/para/raw_b2w_reviews.csv
```
Saídas típicas (exemplo):
```
data/treino_stratify_80.csv
data/validacao_stratify_20.csv
```

### 2) `tcc_train_sentiment_analysis_unified.py` — Treino unificado (clássicos e BERTimbau)
Treina modelos clássicos (**Naive Bayes, Logistic Regression, SVM, Random Forest**) com **TF-IDF** ou **Word2Vec**, ou realiza fine-tuning de **BERTimbau** (`--algo bert`). Salva artefatos organizados em `artifacts/<model_label>/` (incluindo `pipeline.joblib` e `metadata.json`).

Parâmetros (seleção):
- Entrada/saída: `--train_csv`, `--out_dir` (default: `artifacts`), `--model_label`, `--run_suffix`
- Dados: `--sep`, `--encoding` (default: `utf-8-sig`), `--text_cols` (lista de colunas de texto, separadas por vírgula), `--meta_cols` (para transformar colunas categóricas em tokens), `--target_col` (default: `sentiment`)
- Vetorização clássica: `--min_df` (default: 2), `--max_df` (default: 0.9), `--ngram_max` (default: 1)
- Word2Vec (opcional): `--w2v_size`, `--w2v_window`, `--w2v_min_count`, `--w2v_sg`, `--w2v_epochs`, `--w2v_workers`
- Algoritmo: `--algo` com escolhas **`mnb`, `cnb`, `bnb`, `logreg`, `svm`, `rf`, `bert`**
- Balanceamento: `--balance` com escolhas **`balanced`** ou **`none`**
- Hiperparâmetros clássicos:
  - NB: `--alpha`, `--binarize`
  - LogReg: `--C`, `--penalty`, `--solver`, `--max_iter`
  - SVM: `--svm_C`, `--svm_loss`
  - RF: `--rf_n_estimators`, `--rf_max_depth`, `--rf_min_samples_split`, `--rf_min_samples_leaf`, `--rf_bootstrap`, `--rf_n_jobs`, `--rf_random_state`
- BERTimbau (*HuggingFace*): `--hf_model_name` (ex.: caminho local `"models/bertimbau-base"`), `--hf_epochs`, `--hf_batch_size`, `--hf_grad_accum`, `--hf_lr`, `--hf_weight_decay`, `--hf_warmup_ratio`, `--hf_max_length`, `--hf_infer_bs`, `--hf_seed`, `--hf_fp16`, `--hf_save_steps`, `--hf_save_total_limit`

Dicas:
- Você pode usar um **arquivo de argumentos** com `--args_file` (veja `docs/examples/@args_train_example.txt`).  
- Para BERT, instale `torch` + `transformers` (ver sessão *Ambiente*).

Exemplos:
```bash
# Regressão Logística + TF-IDF
python tcc_train_sentiment_analysis_unified.py   --train_csv data/treino_stratify_80.csv   --model_label logreg_tfidf_b2w   --algo logreg   --min_df 2   --max_df 0.9   --ngram_max 2   --balance balanced

# SVM + TF-IDF
python tcc_train_sentiment_analysis_unified.py   --train_csv data/treino_stratify_80.csv   --model_label svm_tfidf_b2w   --algo svm   --svm_C 1.0   --svm_loss squared_hinge   --balance balanced

# BERTimbau (fine-tuning)
python tcc_train_sentiment_analysis_unified.py   --train_csv data/treino_stratify_80.csv   --model_label bertimbau_ft_b2w   --algo bert   --hf_model_name models/bertimbau-base   --hf_epochs 3   --hf_batch_size 16   --hf_max_length 256
```

### 3) `tcc_predict_eval_sentiment.py` — Predição e avaliação
Carrega um `pipeline.joblib` de `artifacts/<model_label>/` e gera predições. Se houver *ground truth*, salva **matriz de confusão** (normalizada e bruta), **classification report** (CSV) e um **relatório Markdown** com as métricas.

Parâmetros:
- `--pipeline`      : caminho para `pipeline.joblib` do modelo treinado
- `--input`         : CSV de entrada para predição (com ou sem rótulos)
- `--output`        : CSV de saída com predições
- `--report_dir`    : diretório para relatórios (default: `reports`)
- `--sep`, `--encoding`, `--text_cols`, `--target_col` (default: `sentiment`)

Exemplo:
```bash
python tcc_predict_eval_sentiment.py   --pipeline artifacts/logreg_tfidf_b2w/pipeline.joblib   --input data/validacao_stratify_20.csv   --output reports/preds.csv   --report_dir reports   --target_col sentiment
```

## Ambiente e instalação

Recomendado **Python 3.10+**.

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -U pip wheel
pip install -r requirements.txt
```

> Para **BERT**: instale `torch` + `transformers`. Se for usar GPU, siga a página oficial do PyTorch para a versão de CUDA apropriada.

## Reprodutibilidade e boas práticas

- **Seeds/fatores estocásticos:** use `--random-state` no split e registre versões das libs (`requirements.txt`).  
- **Artefatos versionados:** cada *run* grava `metadata.json` e o `pipeline.joblib` sob `artifacts/<model_label>/`.  
- **Métricas principais:** em cenários desbalanceados, valorize **F1-score macro** além de accuracy.

## Licença

- **Código (scripts Python)** → [MIT License](LICENSE-CODE)  
- **Documentos, relatórios e datasets associados** → [CC BY-NC 4.0](LICENSE-DATA)  

Resumindo:  
- Código é livre (inclusive comercial).  
- Dados/textos só podem ser reutilizados para fins **não comerciais**, com atribuição.

Veja detalhes em **DISCLAIMER.md**.
