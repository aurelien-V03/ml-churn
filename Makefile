# Raccourcis des services du projet.
#
# `make` seul liste les cibles disponibles.

.DEFAULT_GOAL := help
.PHONY: help db api mlflow

help: ## Liste les cibles disponibles
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-8s %s\n", $$1, $$2}'

db: ## Demarre PostgreSQL et pgAdmin (en arriere-plan)
	docker compose up -d

api: ## Lance le serveur FastAPI sur http://127.0.0.1:8000
	uv run uvicorn ml_churn.api.main:app --reload

mlflow: ## Ouvre l'interface MLflow sur http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
