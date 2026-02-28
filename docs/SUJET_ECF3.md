# ECF - MACHINE LEARNING SUPERVISÉ
## Prédiction de Churn Client - TeleCom+
### Scikit-learn + Spark MLlib

---

## CONTEXTE

Vous êtes Data Analyst chez TeleCom+. Le département Customer Success vous demande de construire un modèle prédictif pour identifier les clients à risque de départ. Chaque client perdu = 500€ de perte mensuelle.

Dataset : 7043 clients avec 21 variables. Taux de churn : 26.5%

Objectif : Comparer deux approches ML (Scikit-learn vs Spark MLlib) et recommander la meilleure pour production.

**Bonus défi :** Implémenter la même solution en Spark pour traiter des données à l'échelle Big Data.

---

## DONNÉES

Fichier fourni : `03_DONNEES.csv`

**Contient :**
- 7043 clients
- 21 variables
- Pas de valeurs manquantes
- Prêt à utiliser

**Variables explicatives (20) :**
- Démographiques : gender, SeniorCitizen, Partner, Dependents, tenure
- Contrat : Contract, PhoneService, MultipleLines
- Services Internet : InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Facturation : InternetCharges, MonthlyCharges, TotalCharges

**Target (1) :**
- Churn : Yes/No (26.5% churn)

---

## TRAVAIL À RÉALISER

### 1. EXPLORATION ET PRÉPARATION

- Charger les données et afficher info/describe
- Analyser la distribution de Churn
- Identifier et traiter les valeurs manquantes
- Encoder les variables catégorielles (One-hot encoding)
- Normaliser les variables numériques (StandardScaler)

### 2. MODÉLISATION

- Split train/test (70/30, stratify=y, seed=42)
- Entraîner 3 modèles :
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Calculer accuracy, precision, recall, f1-score pour chaque
- Sélectionner le meilleur modèle

### 3. ÉVALUATION ET INTERPRÉTATION

- Afficher matrice de confusion du meilleur modèle
- Extraire et visualiser les 10 features les plus importantes
- Cross-validation (5-fold)
- GridSearchCV pour optimisation hyperparamètres

### 4. RECOMMANDATIONS MÉTIER

- Identifier les 20 clients avec probabilité churn > 80%
- Analyser leurs profils (Contract, InternetService, tenure, MonthlyCharges)
- Proposer 3 actions de rétention
- Estimer ROI (coût action vs économie)

### 5. BONUS - SPARK MLLIB 

Reproduire la même solution avec Spark MLlib :

**5a. Setup Spark**
- Initialiser SparkSession
- Charger CSV dans DataFrame Spark
- Afficher schema et statistics

**5b. Préparation Spark**
- StringIndexer pour variables catégorielles
- VectorAssembler pour features
- StandardScaler pour normalisation
- Pipeline Spark

**5c. Modélisation Spark**
- LogisticRegression (Spark MLlib)
- RandomForestClassifier (Spark MLlib)
- Entraîner sur données distribuées
- Évaluer (accuracy, f1-score)

**5d. Comparaison Scikit-learn vs Spark**
- Tableau comparatif : Performance, Temps exécution, Scalabilité
- Insights : Quand utiliser Spark vs Scikit-learn ?
- Recommandations pour production

---

## LIVRABLES

**Minimum (obligatoire) :**
1. **Notebook Jupyter** : Sections 1-4 (Scikit-learn)
   - Code exécutable, commenté, avec visualisations
2. **Rapport** (2-3 pages) : Résumé, méthodo, résultats, recommandations
3. **Fichiers de résultats** : predictions_test.csv, model_metrics.json, feature_importance.csv

**Bonus (optionnel - défi Spark) :**
4. **Notebook Spark** : Section 5 (Spark MLlib)
   - Code PySpark complet
   - Comparaison avec Scikit-learn
   - Analyse de scalabilité

