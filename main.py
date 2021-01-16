import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

db_path = "./db/breast-cancer-wisconsin.data"

missing_values = "?"
df = pd.read_csv(db_path, na_values=missing_values)

# TODO why -99999
# Replace missing values
df.replace(missing_values, 0, inplace=True)

# Remove useless columns
df.drop(["sample"], axis=1, inplace=True)


headers = ["clump-thickness",
           "uniformity-of-cell-size",
           "uniformity-of-cell-shape",
           "marginal-adhesion",
           "single-epithelial-cell-size",
           "bare-nuclei",
           "bland-chromatin",
           "normal-nuclei",
           "mitoses",
           "class"]  # class, label or attribute we are trying to predict

total_values_missing = 0

# TODO
# Count missing values differently
for header in headers:
    missing_values_sum = df[header].isnull().sum()
    total_values_missing += missing_values_sum

    # Fill missing values using median
    median = df[header].median()
    df[header].fillna(median, inplace=True)

print("Missing attribute values:", total_values_missing)


random_state = 42

target = df[["class"]].to_numpy().ravel()
data = df.drop(["class"], axis=1)

train_inputs, test_inputs, train_classes, test_classes = train_test_split(data, target, test_size=0.3, random_state=random_state)

classifiers = [
    ("Decision Tree Classifier", DecisionTreeClassifier(random_state=random_state, max_depth=5)),
    ("Naive Bayes", GaussianNB()),
    ("k-NN 1", KNeighborsClassifier(n_neighbors=1)),
    ("k-NN 3", KNeighborsClassifier(n_neighbors=3)),
    ("k-NN 5", KNeighborsClassifier(n_neighbors=5)),
    ("k-NN 11", KNeighborsClassifier(n_neighbors=11)),
    ("Neural Network", MLPClassifier(solver='lbfgs', alpha=1e-5,
                                     hidden_layer_sizes=(3,), random_state=random_state)),
    ("Random Forest", RandomForestClassifier(random_state=random_state))
]

for name, c in classifiers:
    test_results = c.fit(train_inputs, train_classes).predict(test_inputs)
    accuracy = accuracy_score(test_classes, test_results)
    c = confusion_matrix(test_classes, test_results)

    print("===============================")
    print(name)
    print("accuracy:", accuracy)
