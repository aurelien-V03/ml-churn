## Structure du projet

```
churn-notebook.ipynb        Déroulé complet : exploration, ingestion, entraînement
docker-compose.yml          PostgreSQL + pgAdmin
docs/                       Données sources (CSV) et énoncé du cas d'usage
mlflow.db, mlartifacts/     Suivi des expérimentations (généré)

artifacts/                  Modèles entraînés, versionnés dans git
├── classification/         Une tâche par dossier
│   ├── baseline/           Régression logistique
│   └── final/              XGBoost
└── regression/
    ├── baseline/
    └── final/
        └── AAAA-MM-JJ/     Un dossier par jour d'entraînement
            ├── *.joblib    Le modèle sérialisé
            ├── *.json      Versions des bibliothèques, seuil, features, métriques
            └── *_train.csv Extraits gold consommés (train / validation / test)

src/
├── ml_churn/               Package installé (`uv run python -m ml_churn...`)
│   ├── api/                Service FastAPI : /health, /ready, /predict
│   ├── drift/              Dérive des données : KS et PSI
│   │   └── data/           Schémas Pydantic des requêtes et réponses
│   ├── ui/                 Page de test : index.html, style.css, app.js, clients.js
│   ├── ingestion/          Médaillon : CSV → bronze → silver → gold
│   │   ├── db.py           Connexion PostgreSQL
│   │   ├── logs.py         Format de log commun aux trois couches
│   │   ├── models/         Modèles SQLAlchemy, un fichier par couche
│   │   └── scripts/        Un script par couche, plus `ingest_all`
│   └── training/
│       ├── common/         Gold, découpage, métriques, figures, SHAP, sauvegarde
│       │   └── tracking/   Suivi MLflow, commun à tous les modèles
│       ├── classification/ Prédiction du churn (cible `churn`)
│       │   ├── baseline/   Régression logistique, seuil fixe à 0.5
│       │   └── final/      XGBoost + recherche d'hyperparamètres
│       └── regression/     Valeur vie client (cible `valeur_vie_client_eur`)
│           ├── baseline/   Régression linéaire, paramètres par défaut
│           └── final/      XGBoost
└── visualization/          Hors package, importé via `sys.path`
    ├── scripts/            Un script par type de graphique, plus `plot_all`
    └── graphs/             PNG générés, un dossier par colonne (généré)
```

Trois responsabilités séparées :

| Dossier | Rôle |
| --- | --- |
| `ingestion/` | Charger et transformer les données, du CSV brut à la table exploitable |
| `visualization/` | Comprendre les données : distributions, valeurs extrêmes, relations au churn |
| `training/` | Entraîner et évaluer les modèles, en suivant les runs dans MLflow |
| `api/` | Servir les modèles entraînés par HTTP, sans jamais réentraîner |

Les scripts sont autonomes, exécutables en ligne de commande comme importables
depuis le notebook. Le code partagé entre
plusieurs scripts vit dans un `common/` — jamais dupliqué d'un modèle à
l'autre.

## Commandes utiles

- docker compose `docker compose up -d`
- serveur FastApi `uv run uvicorn ml_churn.api.main:app --reload`
- mlflow `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

## 3. API de prédiction

Le service entraîne ses modèles au démarrage, à partir de la couche gold : il
ne relit rien depuis `artifacts/`. Il dépend donc de PostgreSQL pour démarrer.
Le seuil et les hyperparamètres retenus par la recherche sont figés en tête de
`api/registry.py`.

Page de test sur <http://127.0.0.1:8000/> et documentation interactive sur
<http://127.0.0.1:8000/docs>.

| Endpoint | Rôle | Réponse |
| --- | --- | --- |
| `GET /health` | Le service répond | `{"status": "ok"}` |
| `GET /ready` | Les deux familles de modèles sont entraînées | `{"ready": true, "models": {"churn": ["xgboost"], "clv": ["xgboost"]}}`, ou 503 |
| `GET /ui-config` | Clé d'API remise à la page de test | `{"api_key": "…"}` |
| `POST /drift` | Dérive entre l'entraînement et les lignes transmises | `{"reference", "current", "columns": [{"column", "ks", "p_value", "psi", "verdict"}]}` |
| `POST /predict-churn` | Probabilité de churn d'un client | `{"model", "probability", "threshold", "churn"}` |
| `POST /predict-clv` | Valeur vie client estimée | `{"model", "lifetime_value_eur"}` |

`/health` et `/ready` sont distincts comme les sondes Kubernetes : le service
peut répondre avant d'être capable de prédire, le temps de charger ses modèles.

Les deux endpoints de prédiction partagent le même corps, `model` et `data`. Les clés de `data`
sont les colonnes de la couche gold ; celles que le modèle n'utilise pas sont
ignorées, une ligne gold complète peut donc être envoyée telle quelle.

```bash
curl -X POST http://127.0.0.1:8000/predict-churn \
  -H 'content-type: application/json' \
  -d '{"model": "xgboost", "data": {"anciennete_mois": 12, "nb_integrations": 0, ...}}'
```

```json
{"model": "xgboost", "probability": 0.77, "threshold": 0.4, "churn": true}
```

Même requête sur `/predict-clv` :

```json
{"model": "xgboost", "lifetime_value_eur": 264317.94}
```