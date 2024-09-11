import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import joblib

# load data
df = pd.read_csv("car.data")
# print(df.head())

# Transform data with label encoder
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(df["buying"]))
maint = le.fit_transform(list(df["maint"]))
door = le.fit_transform(list(df["door"]))
persons = le.fit_transform(list(df["persons"]))
lug_boot = le.fit_transform(list(df["lug_boot"]))
safety = le.fit_transform(list(df["safety"]))
cls = le.fit_transform(list(df["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Create Train/Test Split
X_test, X_train, y_test, y_train = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Training model
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = sklearn.metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plot predictions
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

# Print best accuracy and k value for this model
print( "The best accuracy was %.4f" % mean_acc.max(), "with k=", mean_acc.argmax()+1)

best_knn_model = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(X_train, y_train)

# Save the best model and k value
model_filename = 'best_knn_model.pkl'
joblib.dump(best_knn_model, model_filename)

print(f"Best model saved as '{model_filename}'")