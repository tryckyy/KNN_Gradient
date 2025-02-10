# TP Python

## Etapes premilinaires

Télécharger ce dossier
Activate l'environnement utilisé dans le cours précédent
Installer les packages suivants (s'ils ne sont pas installées) :
```bash
conda install numpy
conda install pandas 
```
Tester le code avec
```
 python .\main.py 
```

# Exercices:
## Familiarisez-vous avec votre jeu de données : [Détails des donnée](https://www.kaggle.com/competitions/titanic/data)
### A l'aide du pandas, répondre les questions suivantes:


1. Quelle est la taille du jeu de données (combien de lignes et de colonnes) ?
891 lignes et 8 colonnes

2. Affichez le nom de toutes les colonnes
['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

3. Affichez les dix premières lignes
print(data.head(10))

4. Vérifiez le tarif le plus élevé
512.3292

5. Combien de passagers ont plus de 60 ans ?
22

6. Combien de personnes ont entre 30 et 40 ans ?
180

8. Quel est le pourcentage de personnes ayant survécu ?
38.38%

8. Quel est le pourcentage de femmes et d'hommes ayant survécu ?
female    74.203822%
male      18.890815%

9. Quel est le taux de survie pour chaque classe (Pclass) ?
1    62.962963%
2    47.282609%
3    24.236253%

10. Afficher la matrice de corrélation et vérifier quelles caractéristiques influencent le plus la variable Survived
print(data.corr() )
Les caractéristiques qui influencent le plus la variable Survived sont Sex, PClass et Fare

## Prétraitement des données
1. Remplacer les colonnes Sex et Embarked par des codes de catégories
2. Remplir les âges manquants avec la valeur moyenne
3. Supprimer les colonnes Tickets, Name, Cabin et PassengerId
4. Prenez les 80 % premières lignes avec toutes les colonnes sauf Survived et convertissez-les en tableau numpy, nommé X_train
5. Prenez les 80 % premières lignes de la colonne Survived et convertissez-les en tableau numpy, nommé y_train
6. Prenez les 20 % dernières lignes avec toutes les colonnes sauf Survived et convertissez-les en tableau numpy, nommé X_test
7. Prenez les 20 % dernières lignes de la colonne Survived et convertissez-les en tableau numpy, nommé y_test


## Numpy pour KNN
### A l'aide du numpy, coder l'algorithme suivant: 
L’algorithme des k plus proches voisins s'écrit en abrégé k-NN ou KNN , de l'anglais k-nearest
neighbors, appartient à la famille des algorithmes d’apprentissage automatique sans paramètres.
L’algorithme des k plus proches voisins est un algorithme d’apprentissage supervisé, il est
nécessaire d’avoir des données labellisées. À partir d’un ensemble E de données labellisées (données d'entraînement), il sera
possible de classer (déterminer le label) d’une nouvelle ensemble de donnée T (donnée n’appartenant pas à E, données de teste).

### Principe de algorithme
On suppose que l'ensemble E contiennent n données labellisées et T, une autre ensemble des données
disjointe de E qui ne possède pas de label. Soit d une fonction qui renvoie la distance
(qui reste à choisir) entre la donnée u de T et une donnée quelconque appartenant à E. Soit un entier k
inférieur ou égal à n.
Le principe de l’algorithme de k-plus proches voisins est le suivant :

Pour chaque donnée u dans T:
* On calcule les distances (par exemple Euclidean) entre la donnée u et chaque donnée appartenant à E.
* On retient les k données du jeu de données E les plus proches de u.
* On attribue à u la classe qui est la plus fréquente parmi les k données les plus proches.

# A faire

1. Coder l'algorithme KNN à l'aide de Numpy

    Remplir la fonction KNN_algo dans le fichier KNN.py. La fonction va retourner les prédictions pour T

    **Remarque** : Essayer de coder de façon avec moins de boucle (for), rappelle que numpy nous permet de faire les calculs terme à terme.

2. Coder l'évaluation : La précision de prédictions 

3. Testez votre code avec k=2 et quelle est la précision de l'algorithm KNN ?

La précision de l'algorithm KNN est de 69.66%

4. Modifier la valeur de k (et retester. Quelle est votre observation sur effect de k ?

Pour k =  0 precision =  64.60674157303372%
Pour k =  1 precision =  66.29213483146067%
Pour k =  2 precision =  69.66292134831461%
Pour k =  3 precision =  73.03370786516854%
Pour k =  4 precision =  73.03370786516854%
Pour k =  5 precision =  74.15730337078652%
Pour k =  6 precision =  74.15730337078652%
Pour k =  7 precision =  77.52808988764045%
Pour k =  8 precision =  72.47191011235955%
Pour k =  9 precision =  73.03370786516854%

La précision varie en fonction de la valeur de k, elle augmente jusqu'à k=7 puis rediminue. 
La précision ne dépasse pas 77.53% et ne descends pas en dessous de 64.61% pour k entre 0 et 9. 
