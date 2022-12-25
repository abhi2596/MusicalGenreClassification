from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from dataset import create_dataset
from sklearn.preprocessing import MinMaxScaler


def knn_train(x_train, y_train, no_of_neighbors):
    knn = KNeighborsClassifier(no_of_neighbors)
    knn.fit(x_train, y_train)
    return knn


def knn_predict(x_test, knn):
    y_pred = knn.predict(x_test)
    return y_pred


def main():
    x, y = create_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    model = knn_train(x_train, y_train, 3)
    x_test = scaler.transform(x_test)
    y_pred = knn_predict(x_test, model)
    print("Accuracy of the model is:",accuracy_score(y_test, y_pred))
    print("Classifcation Report",classification_report(y_test,y_pred))


if __name__ == "__main__":
    main()
