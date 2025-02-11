"""This module corresponds to the Titanic modelization script"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv
import argparse
from functions import split_column
from dotenv import load_dotenv

# ENVIRONMENT CONFIGURATION
parser = argparse.ArgumentParser(description="Entraînement modèle Titanic")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres pour la RF"
)
args = parser.parse_args() # transforme arguments de la ligne de commande en objet python
print(args.n_trees)

load_dotenv()

jeton_api = os.environ.get("JETON_API", "")

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


# Définition des variables globales
N_TREES = args.n_trees
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

## Un peu d'exploration et de feature engineering

TrainingData = pd.read_csv("./data.csv")

TrainingData.head()
TrainingData["Ticket"] = split_column(TrainingData, "Ticket", "/")
TrainingData["Name"] = split_column(TrainingData, "Name", ",")
TrainingData.isnull().sum()

### Statut socioéconomique
fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")

### Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=N_TREES)),
    ]
)

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On split notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour
# apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train]).to_csv("train.csv")
pd.concat([X_test, y_test]).to_csv("test.csv")


# Random Forest

# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset
# d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
