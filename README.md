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

Estimer la probabilité de churn d'un client (résiliation à l'échéance)

livrable :
- modele de classification (cible principal) : le client resilie ou ne resilie pas à l'échéance
- modele de regression (cible secondaire) : estimation de la valeur vie client


### Etapes

# 1. Cadrage métier

### Définition des KPI métier :

Churn
- Taux de churn : clients ayant résiliés / clients total (définir un niveau auquel il ne faut pas passer en dessous)
- Nombre de client sauvés : nombre de client qui souhaités résilier leur abonnement mais ont changés d'avis

-> l'objectif est de garder un maximum de client

Valeur vie client
- Ecart estimation CLV - CLV a un instant T 

-> l'objectif est d'avoir un indicateur permettant de juger de la marge de profit restante, l'objectif est donc de rapprocher ces valeurs pour maximiser le chiffre d'affaire

### FP vs FN (problème de classification)

Faux positifs = le modèle prédit une resiliation alors que le client ne prévoit pas de résilier (fausse alerte)
Faux négatifs = le modèle prédit une non-résiliation alors que le client prévoit de résilier (résiliation manquée)

Les faux positifs sont moins grave que les faux négatifs car ils impliquent principalement du temps d'investigation de la part du CSM et un dérangement du client, alors que les faux négatifs impliquent une perte de chiffre d'affaire immédiate.

### Hypothèse

La resiliation est à ces raisons :
- satisfaction client
    - support de qualité (peu de demande et traitement rapide)
    - services offert sont utilisés (correspond à la demande de l'utilisateur)
- Bon payeur
    - grosse entreprise
    - pas de retard de paiement    



# 2. Choix du type de modèle

On connait le label (ce que l'on veut prédire = churn) donc il s'agit d'un problème de machine learning
avec apprentissage supervisé

# 3. Visualisation des données sous forme graphique

Scripts de génération : `src/visualization/scripts`
Graphiques générés : `src/visualization/graphs`

| Colonne | Distribution |
| --- | --- |
| `date_souscription` | asymétrique |
| `jour_souscription` | normale |
| `anciennete_mois` | asymétrique |
| `sieges_souscrits` | asymétrique |
| `utilisateurs_actifs` | asymétrique |
| `taux_adoption_pct` | asymétrique |
| `connexions_30j` | asymétrique |
| `heures_usage_30j` | asymétrique |
| `nb_integrations` | asymétrique |
| `derniere_connexion_jours` | asymétrique |
| `tickets_support_90j` | asymétrique |
| `delai_reponse_support_h` | asymétrique |
| `csat` | asymétrique |
| `retards_paiement_12m` | asymétrique |
| `sante_compte_fin_periode` | asymétrique |

Observation par rapport aux hypothèses :

- satisfaction client
    - Plus le délai de réponse du support et long plus le taux de churn augmente
    - Plus le nombre de tickets supports au cours des 90j augmente plus le taux de churn augmente
- Bon payeur
    - Plus l'entreprise est grosse plus le churn rate tend à baisser
    - Plus le client a des retards de payement plus le taux de churn augmente

# 4. Ingestion

### 🥉 Bronze

Les CSV de `docs/` sont copiés tels quels dans le schéma `bronze` : **toutes les
colonnes métier sont en `TEXT`**, aucune conversion ni nettoyage (dates
hétérogènes, virgules décimales, casse et espaces restent bruts — c'est le
travail de la couche silver). Chaque ligne porte en plus `_source_file`,
`_source_line` et `_ingested_at`.

| CSV | Table |
| --- | --- |
| `docs/catalogue_plans.csv` | `bronze.catalogue_bronze` |
| `docs/churn_saas_complet.csv` | `bronze.churn_saas_complet_bronze` |
| `docs/churn_saas_echantillon.csv` | `bronze.churn_saas_echantillon_bronze` |

On utilise uniquement le fichire churn_saas_complet comme source de donnée, churn_saas_echantillon n'est pas inclu car cela provoquerait des doublons.
Pas besoin d'anonymisation car les données concernent des entreprises.

### 🥈 Silver

Chaque action est une fonction indépendante référencée dans `TRANSFORMATIONS`
(`src/ml_churn/ingestion/scripts/ingest_silver.py`) et appliquée dans cet ordre.
Le catalogue est chargé en amont, ses plans servant à l'imputation du revenu.

| # | Action | Effet |
| --- | --- | --- |
| 1 | `dedupliquer_clients` | Un client_id ne doit apparaitre qu'une fois. |
| 2 | `standardiser_date_souscription` | Ramene les trois formats de date rencontres au seul format AAAA-MM-JJ. |
| 3 | `standardiser_jour_souscription` | lundi -> L, mardi -> M, mercredi -> ME, jeudi -> J, vendredi -> V, samedi -> S, dimanche -> D. |
| 4 | `standardiser_secteur` | Tech -> TE, Finance -> FI, Commerce -> CO, Sante -> SA, Industrie -> IN, Public -> PB, Education -> EN. |
| 5 | `standardiser_pays` | Nom du pays -> code ISO 3166-1 alpha-2 (France -> FR, Allemagne -> DE...). |
| 6 | `standardiser_taille_entreprise` | Harmonise la casse : tpe / " TPE " -> TPE, idem PME, ETI, GE. |
| 7 | `standardiser_plan` | Pro -> PRO, Business -> BUS, Starter -> STR, Enterprise -> ENT. |
| 8 | `standardiser_couleur_theme_interface` | clair -> C, vert -> VE, bleu -> B, violet -> V, sombre -> S. |
| 9 | `standardiser_groupe_experimentation` | A -> A, B -> B, control -> C. |
| 10 | `deriver_polarite_csm` | Ajoute `polarite_csm` a partir du commentaire, sans toucher au texte. |
| 11 | `appliquer_regles_metier` | Supprime les lignes violant une regle metier, colonne par colonne. |
| 12 | `typer_colonnes` | Convertit les colonnes texte vers les types du modele silver. |
| 13 | `imputer_revenu_par_catalogue` | Reconstitue le revenu manquant : sieges souscrits x prix du plan. |
| 14 | `imputer_valeurs_manquantes` | Comble les valeurs manquantes : médiane pour les numériques, mode pour les catégorielles. |

Les sections ci-dessous détaillent chacune de ces étapes.

#### Colonnes écartées

Le filtre se déclare dans le modèle : une colonne absente de `ChurnSaasSilver`
n'est pas lue depuis bronze.

| Colonne | Raison |
| --- | --- |
| `fonctionnalites_total` | Dénormalisation du plan, identique à `catalogue_silver.fonctionnalites_incluses` sur toutes les lignes. Le catalogue est la source de vérité. |

Toutes les autres colonnes du CSV sont reprises.

### 1. Déduplication

`client_id` ne doit apparaître qu'une fois.

### 2. Standardisation

Casse et espaces parasites sont harmonisés avant toute correspondance : le CSV
contient `TPE`, `" TPE "` et `tpe` pour la même valeur. Une valeur vide reste à
`NULL` ; une valeur hors table de correspondance déclenche un `WARNING` et est
mise à `NULL`.

#### Clients (`churn_saas_silver`)

| Colonne | Transformation |
| --- | --- |
| `date_souscription` | `AAAA-MM-JJ`, `JJ/MM/AAAA` et `JJ mois AAAA` → **`AAAA-MM-JJ`** |
| `jour_souscription` | lundi→`L`, mardi→`M`, mercredi→`ME`, jeudi→`J`, vendredi→`V`, samedi→`S`, dimanche→`D` |
| `secteur` | Tech→`TE`, Finance→`FI`, Commerce→`CO`, Santé→`SA`, Industrie→`IN`, Public→`PB`, Éducation→`EN` |
| `pays` | nom → code ISO 3166-1 alpha-2 (`FR`, `ES`, `CA`, `DE`, `CH`, `BE`) |
| `taille_entreprise` | harmonisation de casse → `TPE`, `PME`, `ETI`, `GE` |
| `plan` | Pro→`PRO`, Business→`BUS`, Starter→`STR`, Enterprise→`ENT` |
| `couleur_theme_interface` | clair→`C`, vert→`VE`, bleu→`B`, violet→`V`, sombre→`S` |
| `groupe_experimentation` | A→`A`, B→`B`, control→`C` |

#### Catalogue des plans (`catalogue_silver`)

| Colonne | Transformation |
| --- | --- |
| `plan` | Pro→`PRO`, Business→`BUS`, Starter→`STR`, Enterprise→`ENT` |
| `support_dedie` | `Oui` / `Non` → booléen |

Le plan reçoit le même code que dans `churn_saas_silver` : c'est ce qui permet
de joindre les deux tables (`join silver.catalogue_silver using (plan)`).

### 3. Polarité du commentaire CSM

`commentaire_csm` provient d'une liste fermée de 13 formulations. Une analyse de
polarité par lexique en dérive la colonne `polarite_csm`, sans toucher au texte
d'origine : termes négatifs (*insatisfait*, *risque*, *baisse*…), termes positifs
(*satisfait*, *engagé*, *ambassadeur*) et modificateurs inverseurs (*peu*,
*faible*, comme dans « compte **peu actif** »). Le signe du score décide.

| Polarité | Termes du lexique |
| --- | --- |
| `ALERTE` | `insatisfait`, `mecontentement`, `risque`, `baisse`, `friction`, `limite`, `relance`, `sollicite`, `multiples tickets`, `depart` — ainsi que les modificateurs `peu` et `faible` |
| `POSITIF` | `satisfait`, `engage`, `ambassadeur`, `actif` — sauf précédés d'un modificateur (« peu actif ») |
| `NEUTRE` | aucun terme du lexique reconnu |
| `ABSENT` | pas de commentaire |

Un TF-IDF a été écarté : sur des phrases aussi courtes, il regroupe sur le
vocabulaire partagé et classe « Client très satisfait » avec « Client
insatisfait ».

> À confirmer auprès de l'équipe customer : si ces commentaires sont saisis à la
> résiliation plutôt qu'en amont, la colonne est une fuite de données, au même
> titre que `sante_compte_fin_periode`.

### 4. Règles métier

Toute ligne violant une règle est **supprimée**. Conventions : une valeur absente
n'est pas une violation (seules les valeurs présentes et hors bornes sont
rejetées) ; exception pour `client_id`, où l'absence est une violation puisque
c'est la clé.

| Colonne | Règle |
| --- | --- |
| `client_id` | format `CLI-<chiffres>` |
| `anciennete_mois` | entre 1 et 36 |
| `utilisateurs_actifs` | entre 0 et `sieges_souscrits` de la ligne |
| `taux_adoption_pct` | entre 0 et 100 |
| `csat` | entre 1 et 5 |
| `sante_compte_fin_periode` | entre 0 et 100 |
| `churn` | vaut 0 ou 1 |
    
### 5.Gestion des outliers

L'observation visuelle des graphiques ne denote aucun outliers significatif

### 6. Typage

Les colonnes texte sont converties vers les types du modèle : `date` pour la
souscription, `integer` pour 13 colonnes, `numeric` pour les taux, durées et
montants, `varchar` pour les codes. La lecture numérique tolère les unités du
CSV (`20.0%`, `3.1 h`, `280.62 €`, `40,0`) : sans cela, des valeurs valides
seraient perdues.

### 7.Imputation

Appliquée après le typage. Médiane pour les colonnes numériques, mode pour les
catégorielles. La valeur retenue est calculée sur l'ensemble des lignes et
affichée dans le log de l'ingestion.

| Colonne | Méthode |
| --- | --- |
| `delai_reponse_support_h` | médiane |
| `csat` | médiane |
| `heures_usage_30j` | médiane |
| `taux_adoption_pct` | médiane |
| `retards_paiement_12m` | médiane |
| `secteur` | mode |
| `nb_integrations` | médiane |
| `pays` | mode |
| `revenu_mensuel_recurrent_eur` | `sieges_souscrits` × `prix_mensuel_par_siege_eur` |
| `polarite_csm` | modalité explicite |

## 🥇 Gold

`silver` → `gold`, pour les deux tables : `catalogue_gold` et `churn_saas_gold`.
La structure de silver est reprise, complétée par des colonnes calculées
destinées à la modélisation. La couche est entièrement rechargée à chaque
exécution.

#### Colonnes écartées

Même mécanisme qu'en silver : une colonne absente de `ChurnSaasGold` n'est pas
lue depuis silver.

| Colonne | Raison |
| --- | --- |
| `commentaire_csm` | Texte libre sans usage en modélisation ; `polarite_csm` en conserve le signal. |

`fonctionnalites_total`, déjà écartée en silver, n'apparaît donc pas non plus
ici.

Comme en silver, chaque action est une fonction indépendante, listée dans les
`transformations` de la table concernée
(`src/ml_churn/ingestion/scripts/ingest_gold.py`) :

| # | Action | Effet |
| --- | --- | --- |
| 1 | `ajouter_colonnes_derivees` | Calcule les 4 colonnes ci-dessous |
| 2 | `encoder_categorielles` | Encode les colonnes catégorielles en one-hot |

### 1. Colonnes dérivées

| Colonne | Calcul | Intérêt |
| --- | --- | --- |
| `inactivite_relative` | `derniere_connexion_jours / (anciennete_mois × 30)` | 15 jours sans connexion ne pèsent pas pareil à 1 mois et à 3 ans d'ancienneté |
| `inactif_30j` | `derniere_connexion_jours >= 30` | Isole les comptes dormants, dont le risque de résiliation est nettement plus élevé |
| `taux_fonctionnalites` | `fonctionnalites_utilisees / catalogue.fonctionnalites_incluses` | Normalise par le plan : 3 fonctionnalités sur 8 (Starter) ou sur 40 (Enterprise) ne décrivent pas la même adoption |
| `niveau_anciennete` | tranches d'`anciennete_mois` | Segmentation du cycle de vie, encodée en one-hot |

Les seuils sont des constantes en tête du script (`SEUIL_INACTIVITE_JOURS`,
`NIVEAUX_ANCIENNETE`).

Découpage retenu pour `niveau_anciennete` — les bornes suivent le cycle
contractuel (fin d'onboarding, premier renouvellement annuel) :

| Niveau | Ancienneté |
| --- | --- |
| `RECENT` | 1 à 3 mois |
| `ETABLI` | 4 à 12 mois |
| `ANCIEN` | 13 mois et plus |

### 2. Encodage one-hot

Dix colonnes catégorielles donnent **47 colonnes binaires** (`0` / `1`), selon
la convention `[nom_colonne]_valeur` : `pays_fr`, `plan_str`,
`polarite_csm_alerte`, `niveau_anciennete_recent`…

| Colonne encodée | Origine |
| --- | --- |
| `jour_souscription`, `secteur`, `pays`, `taille_entreprise`, `plan`, `couleur_theme_interface`, `code_datacenter`, `groupe_experimentation` | standardisées en silver |
| `polarite_csm` | dérivée du commentaire CSM en silver |
| `niveau_anciennete` | dérivée en gold |

`polarite_csm` donne `polarite_csm_alerte`, `_neutre`, `_positif` et `_absent` :
l'absence de commentaire est une modalité à part entière, distincte d'un
commentaire neutre.

Deux règles de nommage : minuscules, et tout caractère non alphanumérique
remplacé par `_` (`eu-w1` → `code_datacenter_eu_w1`), pour obtenir des
identifiants SQL utilisables sans guillemets. Les colonnes d'origine sont
conservées à côté de leur encodage.

Les modalités sont déclarées dans `MODALITES_ONE_HOT`
(`src/ml_churn/ingestion/models/gold.py`) et non déduites des données : le
schéma de la table reste ainsi stable quel que soit le contenu du lot chargé, et
toute modalité inattendue déclenche un `WARNING` au lieu de casser l'insertion.

# 5. Entrainement des modèles

## 1. Choix des modèles

Modèles disponibles :
- regression linéaire : regression
- regression logistique : classification
- Random Forest : classification
- XG Boost : classification
- Gradient boosting : classification

### Baseline models

- regression linéaire : regression
- regression logistique : classification

## 2. Choic des métriques

Il s'agit d'un dataset déséquilibré (28%)
