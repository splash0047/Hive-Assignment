# Recommended Libraries, Extensions, and Plugins

## Core Python dependencies

Recommended minimal set:

```toml
pandas
pyarrow
numpy
scikit-learn
sentence-transformers
faiss-cpu
pydantic
pydantic-settings
typer
rich
python-dotenv
pyyaml
matplotlib
pytest
ruff
```

Optional:
```toml
streamlit        # manual labeling UI
mlflow           # experiment/evaluation tracking
kagglehub        # convenient public Kaggle dataset retrieval
rapidfuzz        # duplicate/near-duplicate checks
```

Use one LLM SDK only:
- provider-native SDK; or
- a thin provider abstraction if switching is necessary.

Do not add both LangChain and LlamaIndex. Prefer neither for the core project.

## Why this stack

### scikit-learn
Use for:
- TF-IDF baseline;
- Logistic Regression;
- calibration;
- classification metrics.

### sentence-transformers
Use for:
- semantic intent features;
- historical query embeddings.

### FAISS
Use for:
- exact dense similarity retrieval over the brand corpus.

For this assignment-sized corpus, `IndexFlatIP` is simpler than approximate indexes.

### MLflow - optional
Useful if you want:
- experiment records;
- GenAI evaluation traces;
- prompt/model comparison.

But it adds setup. The project should still work without it.

## Antigravity integrations/plugins to enable if available

Priority:
1. Git/GitHub integration
2. Python language tooling
3. terminal execution
4. test runner
5. notebook/Jupyter support for exploratory analysis
6. Mermaid/Markdown preview
7. environment variable support
8. Docker only if needed
9. optional browser/documentation search

## Useful editor extensions if working locally

- Python
- Pylance / equivalent language server
- Ruff
- Jupyter
- GitLens or equivalent
- YAML
- Markdown/Mermaid preview
- SQLite viewer only if using SQLite/MLflow locally

## Avoid unnecessary complexity

Skip unless evidence says you need them:
- Kubernetes
- Redis
- Celery
- PostgreSQL
- hosted vector databases
- microservices
- agent graph frameworks
- frontend frameworks
- GPU training infrastructure

The interview includes live explanation/modification. Every dependency should earn its place.
