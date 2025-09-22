# Model Card — TCC ICMC Sentiment Analysis (PT-BR)

Este documento resume objetivos, dados, modelos, métricas e considerações éticas dos scripts deste repositório.

## Objetivo

Implementar e avaliar pipelines de **análise de sentimentos em português**, comparando algoritmos clássicos (Naive Bayes, Regressão Logística, SVM, Random Forest) e **BERTimbau**.  
Caráter **acadêmico e experimental**, parte de uma monografia de MBA.

## Dados

- **Dataset de referência**: **B2W Reviews (PT-BR)** — repositório oficial: https://github.com/thiagorainmaker77/b2w-reviews01  
- Estrutura mínima: colunas de texto (e.g., `review_title`, `review_text`) e coluna `overall_rating` (1–5).  
- Mapeamento de classes (prática comum no trabalho): 1–2 → `neg`, 3 → `neu`, 4–5 → `pos`.  
- **Aviso**: o dataset **não está incluído** aqui. Baixe do repositório oficial e siga LGPD/GDPR e termos de uso.

## Modelos

### Clássicos
- Vetorização: **TF-IDF** ou **Word2Vec**
- Classificadores: **MultinomialNB**, **ComplementNB**, **BernoulliNB**, **Logistic Regression**, **SVM**, **Random Forest**

### Transformers
- **BERTimbau (base)** com *fine-tuning* via HuggingFace.

## Métricas

Os scripts produzem:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Matriz de confusão** (normalizada e bruta)
- Relatórios CSV/Markdown por classe e agregados

**Recomendação**: enfatizar **F1-score macro** quando houver **desbalanceamento** entre classes.

## Limitações

- **Classe neutra** tende a ser mais ambígua (zona cinzenta entre polaridades).  
- **Dependência de dados**: modelos clássicos funcionam bem com menos dados; BERT exige mais dados/recursos.  
- **Generalização**: treinado em B2W pode não transferir para outros domínios sem *fine-tuning* adicional.

## Considerações éticas

- **Uso responsável**: não interprete como julgamento definitivo de opiniões humanas.  
- **Privacidade**: evite dados pessoais/sensíveis.  
- **Viés**: os modelos podem herdar vieses do dataset; avalie criticamente em novos contextos.

## Licenciamento

- **Código (scripts Python)**: [MIT License](LICENSE-CODE)  
- **Dados, relatórios e documentos textuais**: [CC BY-NC 4.0](LICENSE-DATA)  

---
*Mantido por Adriano (2025).*
