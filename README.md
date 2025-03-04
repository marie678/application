# Probabilité de survie sur le Titanic

Pour pouvoir utiliser ce projet, il
est recommandé de créer un fichier `config.yaml`
ayant la structure suivante:

```yaml
jeton_api: ####
data_path: https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw/data.csv
```

Pour installer les dépendances

```bash
pip install -r requirements.txt
```
# Titre du projet
## Utiliser ce projet

- Spécifier les variables d'environnement suivantes :
```
JETON_API=changme
```
- Exécuter le projet :
```python
python titanic.py --n_trees=50
```
créer environnement virtuel et installer les packages grâce qu fichier requirements.txt

Pour utiliser l'API, il faut un fichier .env contenant le jeton API.
