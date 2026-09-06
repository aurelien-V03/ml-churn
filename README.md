## Notes personnelles

### Données

- catalogue_plans : offre de service
- churn_saas_complet : dataset complet (5035 lignes)
- churn_saas_echantillon : echantillon (50 lignes) ≈ 1%

### Acteurs

- Editeur SaaS
- Equipe customer
- Clients

### Objectifs

Estimer la probabilité de churn d'un client

livrable :
- modele de classification (cible principal) : le client resilie ou ne resilie pas à l'échéance
- modele de regression (cible secondaire) : estimation de la valeur vie client

### Etapes

- Pipeline données (architecture en médaillon)
    - bronze
    stockage telle quelle
    - silver
    - gold