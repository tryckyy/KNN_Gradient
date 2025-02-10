import pandas as pd
import numpy as np
import KNN
import Gradient

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv("Python_pandas_numpy/data/titanic/train.csv")
    print("Succès : Lecture des données")



    # Prétraitement des données

    # 1. Remplacer les colonnes Sex et Embarked par des codes de catégories
    data["Sex"] = data["Sex"].astype("category").cat.codes
    data["Embarked"] = data["Embarked"].astype("category").cat.codes

    # 2. Remplir les âges manquants avec la valeur moyenne

    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # 3. Supprimer les colonnes Ticket, Name, Cabin et PassengerId

    print("Colonnes dans le DataFrame :", data.columns.tolist())
    columns_to_drop = ["Ticket", "Name", "Cabin", "PassengerId"]
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns, axis=1, inplace=True)
    print("Colonnes dans le DataFrame :", data.columns.tolist())

    # 4. Afficher la matrice de corrélation et vérifier quelles caractéristiques influencent le plus la variable Survived

    print( data.corr() ) #les caractéristiques qui influencent le plus la variable Survived sont Sex, PClass et Fare

    # Prenez les 80 % premières lignes avec toutes les colonnes sauf Survived et convertissez-les en tableau numpy, nommé X_train
    num_rows80 = int(0.8 * len(data))
    num_rows20 = int(0.2 * len(data))
    if 'Survived' in data.columns:


        X_train = data.iloc[:num_rows80].drop(columns=['Survived']).to_numpy()

    # Prenez les 80 % premières lignes de la colonne Survived et convertissez-les en tableau numpy, nommé y_train

        y_train = data["Survived"].iloc[:num_rows80].to_numpy()
    # Prenez les 20 % dernières lignes avec toutes les colonnes sauf Survived et convertissez-les en tableau numpy, nommé X_test

    X_test = data.iloc[-num_rows20:, :].drop(columns=["Survived"]).to_numpy()

    # Prenez les 20 % dernières lignes de la colonne Survived et convertissez-les en tableau numpy, nommé y_test

    y_test = data["Survived"].iloc[-num_rows20:].to_numpy()

    # Implémentez la fonction KNN_algo et obtenez y_prediction

    y_prediction = KNN.KNN_algo(X_train, y_train, X_test, 2)

    # Évaluez la précision de KNN_algo (la distance entre y_prediction et y_test)
    knn_accuracy = KNN.evaluate_accuracy(y_prediction, y_test)

    # Implementation Gradient algorithm
    c = Gradient.RL_GD(X_train, y_train, 1000, 0.02, affichage = False)
    logreg_predictions = Gradient.prediction(c, X_test)
    logreg_accuracy = KNN.evaluate_accuracy(logreg_predictions, y_test)
    logreg_probs = 1 / (1 + np.exp(-X_test @ c))
    logreg_log_loss = Gradient.log_loss(c, X_test, y_test)

    print(f"KNN Accuracy: {knn_accuracy:.2f}%")
    print(f"Logistic Regression Accuracy: {logreg_accuracy:.2f}%")
    print(f"Logistic Regression Log Loss: {logreg_log_loss:.2f}")


