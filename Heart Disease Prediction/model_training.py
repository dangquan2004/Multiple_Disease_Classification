import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


# Load csv data into pandas data frame
heart_data = pd.read_csv("heart_disease_data.csv")

# Splitting features and Target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data['target']


# Create models:
def create_model(num_model):
    model_list = []
    for i in range(num_model):
        model = LogisticRegression(max_iter=2000)
        model_list.append(model)
    return model_list


def split_data(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=2)
    return (X_train, Y_train), (X_test, Y_test)


def fit_data(model, train_data):
    model.fit(train_data[0], train_data[1])
    return model


def evaluate_model(model, evaluate_data):
    X_data, Y_data = evaluate_data
    X_train_predicton = model.predict(X_data)
    data_accuracy = accuracy_score(X_train_predicton, Y_data)
    return data_accuracy


def pick_model(test_acc_list, models):
    best_accuracy = 0
    best_model = None
    best_model_index = None
    for i, accuracy in enumerate(test_acc_list):
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = models[i]
            best_model_index = i
    return best_model, best_model_index


def train_and_pick():
    models = create_model(10)
    train_accuracy_list = []
    test_accuracy_list = []
    test_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(test_size)):
        train_data, test_data = split_data(X, Y, test_size[i])
        model = fit_data(models[i], train_data)
        train_data_accuracy = evaluate_model(model, train_data)
        test_data_accuracy = evaluate_model(model, test_data)
        test_accuracy_list.append(test_data_accuracy)
        train_accuracy_list.append(train_data_accuracy)
        models.append(model)
    best_model, best_model_index = pick_model(test_accuracy_list, models)
    return best_model


def make_prediction(model, input_data):
    input_data_as_arr = np.asarray(input_data)
    input_data_reshape = input_data_as_arr.reshape(1, -1)
    prediction = model.predict(input_data_reshape)

    if prediction[0] == 0:
        print("The person does not have Heart Disease!")
    else:
        print("The person have Heart Disease")


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


best_model = train_and_pick()
save_model(best_model, "../saved_models/heart_disease_model.sav")
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
make_prediction(best_model, input_data)