"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
import pathlib
from dotenv import load_dotenv
import argparse
import pandas as pd
from src.data.import_data import split_train_test
from src.pipeline.build_features import create_pipeline
from src.models.train_evaluate import evaluate_model
from loguru import logger

logger.debug("That's it, beautiful and simple logging!")
logger.add("file_{time}.log")

# ENVIRONMENT CONFIGURATION ---------------------------

load_dotenv('./configuration/.env')

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

n_trees = args.n_trees
jeton_api = os.environ.get("JETON_API", "")
data_path = os.environ.get("URL_RAW", "FAIL")
# data_path = os.environ["URL_RAW"]
logger.info(data_path)
data_train_path = os.environ.get("train_path", "data/derived/train.parquet")
data_test_path = os.environ.get("test_path", "data/derived/test.parquet")
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

if jeton_api.startswith("$"):
    logger.success("API token has been configured properly")
else:
    logger.info("API token has not been configured")

# IMPORT ET EXPLORATION DONNEES --------------------------------

p = pathlib.Path("data/derived/")
p.mkdir(parents=True, exist_ok=True)

TrainingData = pd.read_csv(data_path)
# TrainingData = pd.read_parquet(data_path)

# SPLIT TRAIN/TEST --------------------------------

X_train, X_test, y_train, y_test = split_train_test(TrainingData, train_path=data_train_path, test_path=data_test_path, test_size=0.1)


# PIPELINE ----------------------------

# Create the pipeline
pipe = create_pipeline(
    n_trees, max_depth=MAX_DEPTH, max_features=MAX_FEATURES
)

# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)

# Evaluate the model
score, matrix = evaluate_model(pipe, X_test, y_test)
logger.info(f"{score:.1%} de bonnes réponses sur les données de test pour validation")
logger.info(20 * "-")
logger.info("matrice de confusion")
logger.info(matrix)
