<<<<<<< HEAD
from sklearn.metrics import confusion_matrix
from loguru import logger

@logger.catch
=======
'''fonctions d’évaluation du modèle'''
from sklearn.metrics import confusion_matrix

>>>>>>> 0236329662a831306959f1e272f713b8952dfd59
def evaluate_model(pipe, X_test, y_test):
    """
    Evaluate the model by calculating the score and confusion matrix.

    Args:
        pipe (sklearn.pipeline.Pipeline): The trained pipeline object.
        X_test (pandas.DataFrame): The test data.
        y_test (pandas.Series): The true labels for the test data.

    Returns:
        tuple: A tuple containing the score and confusion matrix.
    """
    score = pipe.score(X_test, y_test)
    matrix = confusion_matrix(y_test, pipe.predict(X_test))
    return score, matrix
