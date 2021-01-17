import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from helpers import plotConfusionMatrix, plotClassifiersAccuracy

random_state = 4
db_path = "./db/breast-cancer-wisconsin.data"
missing_values = "?"
df = pd.read_csv(db_path, na_values=missing_values)

# # Replace missing values
df.fillna(5, inplace=True)

# Remove useless columns
df.drop(["sample"], axis=1, inplace=True)

labels = df[["class"]].to_numpy().ravel()
df.drop(["class"], axis=1, inplace=True)

train_inputs, test_inputs, train_classes, test_classes = train_test_split(df, labels, test_size=0.3, random_state=random_state)

classifiers = [
    ("Drzewa\nDecyzyjne", DecisionTreeClassifier(random_state=random_state, max_depth=5)),
    ("Naive\nBayes", GaussianNB()),
    ("k-NN 1", KNeighborsClassifier(n_neighbors=1)),
    ("k-NN 3", KNeighborsClassifier(n_neighbors=3)),
    ("k-NN 5", KNeighborsClassifier(n_neighbors=5)),
    ("Neural\nNetwork", MLPClassifier(solver='lbfgs', alpha=1e-5,
                                      hidden_layer_sizes=(3,), random_state=random_state)),
    ("Random\nForest", RandomForestClassifier(random_state=random_state)),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
]

classifiers_accuracy = []
for title, classifier in classifiers:
    test_results = classifier.fit(train_inputs, train_classes).predict(test_inputs)

    if (title.startswith("Drzewa decyzyjne")):
        plt.figure(dpi=800)
        tree.plot_tree(classifier)
        plt.show()

    accuracy = accuracy_score(test_classes, test_results)
    classifiers_accuracy.append((title, accuracy))

    # Confusion matrix
    _confusion_matrix = confusion_matrix(test_classes, test_results)
    confusion_matrix_normalized = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    plotConfusionMatrix(confusion_matrix_normalized, title=title)

    print(f'{title}, accuracy: {round(accuracy * 100, 2)}%')

plotClassifiersAccuracy(classifiers_accuracy)
