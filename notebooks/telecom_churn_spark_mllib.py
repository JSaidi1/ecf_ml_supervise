"""
- Script telecom_churn_spark_mllib.py
- Ce script sera executé dans le node spark qui tourne sur docker
- Pour l'executer :
    1) Mettre en place la config. docker nécéssaire :
        > docker compose up -d --build
    2) Executer la commande :
        > docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /notebooks/telecom_churn_spark_mllib.py
"""
#====================================================================================================
#                           ECF - MACHINE LEARNING SUPERVISÉ - Spark MLlib
#                            Sujet : Prédiction de Churn Client - TeleCom+ 
#
# - Dataset: `03_DONNEES.csv` (7 043 clients)  
# - Target: `Churn` (Yes/No)
#
# - Livrables :
#   Minimum (obligatoire) :
#       1. Ce script (telecom_churn_spark_mllib.py)
#           - Code exécutable, commenté, avec visualisations
#       2. Metrics sur spark 
#       3. Rapport (2-3 pages) : Résumé, méthodo, résultats, recommandations
#====================================================================================================
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time
import platform
import json
import warnings

from utils.utils_logs import log_message, log_df, log_table

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
)
from pyspark.sql.functions import col, when

#-------------------------------------------------------------------------------
# Config. des Chemins des données
#-------------------------------------------------------------------------------
# ───────────── INPUTS ─────────────
IN_DIR = os.path.join("/", "data")
IN_CSV = os.path.join(IN_DIR, "03_DONNEES.csv")
IN_METRICS_SKLEARN_JSON = os.path.join("/", "output", "sklearn", "sklean_metrics.json")

# ───────────── OUTPUTS ────────────
OUT_DIR = os.path.join("/", "output")
OUT_SPARK_DIR = os.path.join(OUT_DIR, "spark")
# Assurer OUT_SKLEARN_DIR
Path(OUT_SPARK_DIR).mkdir(parents=True, exist_ok=True)
OUT_METRICS_JSON = os.path.join(OUT_SPARK_DIR, "spark_metrics.json")

# ───────────── OTHERS ─────────────
ROOT_DIR = Path(__file__).resolve().parent  # ...\ecf_ml_supervise\notebooks
CURRENT_SCRIPT_NAME = os.path.basename(os.path.abspath(__file__))
LOG_DIR = ROOT_DIR.parent / "logs"
LOG_FILE_NAME = "telecom_churn_spark_mllib.log"

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def run(cmd, **kwargs):
    # --- Execute cmd
    log_message(msg_log=f"  Exec. cmd '{cmd}' ...", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge everything
        text=True,
        **kwargs
    )

    if r.returncode != 0:
        output = r.stdout or ""

        # --- Keep only lines that look like real errors
        error_lines = [
            line for line in output.splitlines()
            if "ERROR" in line
            or "Traceback" in line
            or "Exception" in line
            or "Py4JJavaError" in line
        ]

        # --- Raise SystemExit(r.returncode)
        raise RuntimeError(f"Commande échouée : {cmd} - returncode : {r.returncode} - erreur : {error_lines}")
    log_message(msg_log=f"  Succes Exec. cmd '{cmd}'", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def show_startup_message():
    """Show startup message"""
    log_message(msg_log="=" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME, file_log_clear=True)
    log_message(msg_log=" " * 5 + f"ECF - MACHINE LEARNING SUPERVISÉ - Spark MLlib - script: {CURRENT_SCRIPT_NAME}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(msg_log=" " * 22 + f"Sujet : Prédiction de Churn Client - TeleCom+", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(msg_log=" " * 26 + f"Script : {CURRENT_SCRIPT_NAME}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(msg_log="=" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

def create_spark_session(app_name:str, master_cluster_url:str, log_level:str):
    """Cree et configure la session Spark."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master_cluster_url) \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.configurationFile=file:/opt/spark/conf/log4j2.properties") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.configurationFile=file:/opt/spark/conf/log4j2.properties") \
        .getOrCreate()

    ## Reduire les logs
    spark.sparkContext.setLogLevel(log_level)

    ## Affichage versions
    log_message(f"Spark version  : {spark.version}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(f"Spark UI       : {spark.sparkContext.uiWebUrl}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(f"Python version : {platform.python_version()}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(f"Python path    : { sys.executable}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(f"Java version   : {spark._jvm.java.lang.System.getProperty('java.version')}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
    log_message(f"Java home      : { spark._jvm.java.lang.System.getProperty('java.home')}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

    log_message("Creation de la session Spark avec succès", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

    return spark

def close_spark_session(spark:SparkSession):
    spark.stop()
    spark.sparkContext._gateway.shutdown()
    log_message(msg_log="La session spark a ete arretee avec succes", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():
    spark = None

    try:

        # -----------------------------------------------------------------------------------
        # Clear de la console
        # -----------------------------------------------------------------------------------
        clear_console()
       
        # -----------------------------------------------------------------------------------
        # Affichage du message de démarrage
        # -----------------------------------------------------------------------------------
        show_startup_message()

        # -----------------------------------------------------------------------------------
        # Optionel : pour les infos. sur l'éxecution de ce script
        # -----------------------------------------------------------------------------------
        start_time_script = time.time()
        
        # -----------------------------------------------------------------------------------
        # 5a. Setup Spark
        # -----------------------------------------------------------------------------------
        log_message("*** 5a. Setup Spark ***", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        # --- Initialiser SparkSession
        log_message("Initialiser SparkSession :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        spark = create_spark_session(
            app_name="ECF - MACHINE LEARNING SUPERVISÉ - Spark MLlib", 
            master_cluster_url="spark://spark-master:7077",
            log_level="ERROR"
            ) 
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        # --- Charger CSV dans DataFrame Spark
        log_message(msg_log="Charger CSV dans DataFrame Spark :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(IN_CSV)
        log_message("Chargement avec succès du CSV dans la DataFrame Spark df", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Afficher schema et statistics
        log_message(msg_log="Afficher schema et statistics :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        initial_count = df.count()
        log_message(msg_log=f"    - Lignes en entree : {initial_count:,}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message(msg_log=f"    - Nombre de colonnes : {len(df.columns)}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message(msg_log="    - Schema :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message(msg_log=f"\n{df._jdf.schema().treeString()}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_df(msg_log="    - Appercu des données (df) :", df=df, limit_lines=5, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        describe_df = df.describe()
        log_message(msg_log=f"    - Statistiques :\n{describe_df._jdf.showString(20, 20, False)}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # -----------------------------------------------------------------------------------
        # 5b. Préparation Spark
        # -----------------------------------------------------------------------------------
        log_message(msg_log="─" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("*** 5b. Préparation Spark ***", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Création de la cible de la classification
        target = "Churn"
        df = df.withColumn("Churn_bool", when(col(target)=="Yes", 1.0).otherwise(0.0))
        log_df(msg_log="Création de la cible 'Churn_bool' à partir de Churn (0-> No, 1->Yes) :", df=df, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Definition des colonnes catégorielles et numériques
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
        numeric_cols = ['SeniorCitizen', 'tenure', 'InternetCharges', 'MonthlyCharges', 'TotalCharges']

        log_message(msg_log=f"Colonnes catégorielles : {categorical_cols}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message(msg_log=f"Colonnes numériques : {numeric_cols}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- StringIndexer pour variables catégorielles
        log_message(msg_log="StringIndexer pour variables catégorielles :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        indexers = [
            StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            for c in categorical_cols
        ]
        log_message("Creation avec succès du StringIndexer pour les variables catégorielles", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- VectorAssembler pour features
        log_message(msg_log="VectorAssembler pour features :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        # feature_cols = [c for c in df.columns if c not in [target, "Churn_bool", "customerID"]]
        # log_message(msg_log=f"feature_cols : {feature_cols}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        assembler_inputs = [c + "_idx" for c in categorical_cols] + numeric_cols

        assembler = VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="features_raw",
            handleInvalid="keep"
        )
        log_message("Creation avec succès du VectorAssembler pour features", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- StandardScaler pour normalisation
        log_message(msg_log="StandardScaler pour normalisation :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        log_message("Creation avec succès du StandardScaler pour normalisation", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Pipeline Spark
        preprocessing = Pipeline(stages=indexers + [assembler, scaler])
        log_message("Creation avec succès de la pipeline du preprocessing : preprocessing = Pipeline(stages=indexers + [assembler, scaler])", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # -----------------------------------------------------------------------------------
        # 5c. Modélisation Spark
        # -----------------------------------------------------------------------------------      
        log_message(msg_log="─" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("*** 5c. Modélisation Spark ***", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)  

        # --- RandomForestClassifier (Spark MLlib)
        log_message(msg_log="RandomForestClassifier (Spark MLlib) :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="Churn_bool",
            numTrees=100,
            seed=42
        )

        # --- Pipeline R.F
        pipeline_rf = Pipeline(stages=preprocessing.getStages() + [rf])
        log_message("Creation avec succès de la pipeline pour RandomForestClassifier : pipeline_rf = Pipeline(stages=preprocessing.getStages() + [rf]))", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- LogisticRegression (Spark MLlib)
        log_message(msg_log="LogisticRegression (Spark MLlib) :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        lr = LogisticRegression(
            featuresCol="features", 
            labelCol="Churn_bool",
            maxIter=100, 
            regParam=0.05, 
            family="binomial"
        )
        
        # --- Pipeline L.R
        pipeline_lr = Pipeline(stages=preprocessing.getStages() + [lr])
        log_message("Creation avec succès de la pipeline pour LogisticRegression : pipeline_lr = Pipeline(stages=preprocessing.getStages() + [lr])", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Split train/test 70/30
        train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
        log_message("Split train/test 70/30", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Entraîner sur données distribuées
        
        # Entraînement LogisticRegression
        start = time.time()
        model_lr = pipeline_lr.fit(train_df)
        time_lr = time.time() - start
        log_message(f"Temps entraînement LogisticRegression : {time_lr:.2f} s", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # Entraînement RandomForestClassifier
        start = time.time()
        model_rf = pipeline_rf.fit(train_df)
        time_rf = time.time() - start
        log_message(f"Temps entraînement RandomForestClassifier : {time_rf:.2f} s", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Évaluer (accuracy, f1-score)
        log_message("Évaluer (accuracy, f1-score) :", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        # LogisticRegression
        pred_lr = model_lr.transform(test_df)

        evaluator_acc_lr = MulticlassClassificationEvaluator(
            labelCol="Churn_bool", predictionCol="prediction", metricName="accuracy"
        )
        evaluator_f1_lr = MulticlassClassificationEvaluator(
            labelCol="Churn_bool", predictionCol="prediction", metricName="f1"
        )

        acc_lr = evaluator_acc_lr.evaluate(pred_lr)
        f1_lr  = evaluator_f1_lr.evaluate(pred_lr)

        log_message(f"LogisticRegression -> Accuracy : {acc_lr:.4f} - F1-score : {f1_lr:.4f}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # RandomForestClassifier
        pred_rf = model_rf.transform(test_df)

        evaluator_acc_rf = MulticlassClassificationEvaluator(
            labelCol="Churn_bool", predictionCol="prediction", metricName="accuracy"
        )
        evaluator_f1_rf = MulticlassClassificationEvaluator(
            labelCol="Churn_bool", predictionCol="prediction", metricName="f1"
        )

        acc_rf = evaluator_acc_rf.evaluate(pred_rf)
        f1_rf  = evaluator_f1_rf.evaluate(pred_rf)

        log_message(f"RandomForestClassifier -> Accuracy : {acc_rf:.4f} - F1-score : {f1_rf:.4f}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        # --- Enregistrement des metrics
        spark_results = {
            "Logistic Regression": {
                "time":  format(time_lr, ".3f"),
                "accuracy":   format(acc_lr, ".3f"),
                "f1-score": format(f1_lr, ".3f")
            },
            "Random Forest": {
                "time":  format(time_rf, ".3f"),
                "accuracy":   format(acc_rf, ".3f"),
                "f1-score": format(f1_rf, ".3f")
            }
        }

        with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(spark_results, f, indent=2, ensure_ascii=False)
        
        log_message(msg_log="", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        log_message(msg_log=f"Enregistrement des metrics spark : '{OUT_METRICS_JSON}'", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        
        # -----------------------------------------------------------------------------------
        # 5d. Comparaison Scikit-learn vs Spark
        # -----------------------------------------------------------------------------------      
        log_message(msg_log="─" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("*** 5d. Comparaison Scikit-learn vs Spark ***", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        log_message("", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)  

        # --- Chargement metrics sklearn
        try:
            with open(IN_METRICS_SKLEARN_JSON, "r", encoding="utf-8") as f:
                metrics_sklearn = json.load(f)

            sklearn_lr_t = metrics_sklearn["Logistic Regression"]["time"]
            sklearn_lr_acc = metrics_sklearn["Logistic Regression"]["accuracy"]
            sklearn_lr_f1 = metrics_sklearn["Logistic Regression"]["f1-score"]

            sklearn_rf_t = metrics_sklearn["Random Forest"]["time"]
            sklearn_rf_acc = metrics_sklearn["Random Forest"]["accuracy"]
            sklearn_rf_f1 = metrics_sklearn["Random Forest"]["f1-score"]

            # --- Création du tableau pour la comparaison des metrics pour le Logistic Regression
            headers = ["", "Duration (sec)", "Accuracy", "F1-score"]
            data_lr = [{"": "Logistic Regression (sklearn)", "Duration (sec)": sklearn_lr_t, "Accuracy": sklearn_lr_acc, "F1-score": sklearn_lr_f1},
                    {"": f"Logistic Regression (spark)", "Duration (sec)": format(time_lr, ".3f"), "Accuracy": format(acc_lr, ".3f"), "F1-score": format(f1_lr, ".3f")}
                    ]
            data_rf = [{"": "Random Forest (sklearn)", "Duration (sec)": sklearn_rf_t, "Accuracy": sklearn_rf_acc, "F1-score": sklearn_rf_f1},
                    {"": f"Random Forest (spark)", "Duration (sec)": format(time_rf, ".3f"), "Accuracy": format(acc_rf, ".3f"), "F1-score": format(f1_rf, ".3f")}
                    ]

            log_table(data=data_lr, headers=headers, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_table(data=data_rf, headers=headers, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

            log_message(msg_log=f"", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

            # --- Comparaisons
            log_message(msg_log=f"Logistic Regression : ", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> Duration : l'entrainement est plus rapide sur Scikit-learn que sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> Accuracy : Il est quasiment la même sur Scikit-learn et sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> F1-score : Il est plus petit sur Scikit-learn que sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            
            log_message(msg_log=f"", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            
            log_message(msg_log=f"Random Forest : ", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> Duration : l'entrainement est plus rapide sur Scikit-learn que sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> Accuracy : Il est quasiment la même sur Scikit-learn et sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
            log_message(msg_log=f"=> F1-score : Il est plus petit sur Scikit-learn que sur Spark", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        except Exception as e:
            log_message(msg_log=f"Echec lors de la lecture des metriques sklearn : {e}", level="error", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        log_message(msg_log="---", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

        log_message(level="info", msg_log="Fin du traitement, le script a été exécuté avec succès", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)

    except Exception as e:
        log_message(level="error", msg_log=f"{e}", file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)
        raise
        
    finally:
        try:
            if spark:
                close_spark_session(spark)
        except:
            pass

        log_message(msg_log="=" * 95, file_log=True, file_log_dir=LOG_DIR, file_log_name=LOG_FILE_NAME)


if __name__ == "__main__":
    main()