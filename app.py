from flask import Flask, request, render_template, flash, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import time

# Load and preprocess data
data = pd.read_csv('breast-cancer.csv')
if "id" in data.columns:
    data.drop("id", axis=1, inplace=True)
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

features = data.columns.drop("diagnosis").tolist()
X = data[features]
y = data["diagnosis"]
scaler = MinMaxScaler()

# Timing for scaling
scaling_start_time = time.time()
X_scaled = scaler.fit_transform(X)
scaling_end_time = time.time()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "password"

# Convert dataset to list for modal
data_list = data[features].values.tolist()

# Train and evaluate function
def train_and_evaluate(X_train, X_test, y_train, y_test, model, param_grid=None, validation_method="split"):
    if param_grid:
        grid_search_start_time = time.time()
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        grid_search_end_time = time.time()
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        grid_search_start_time = grid_search_end_time = 0

    if validation_method == "crossval":
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        kf = KFold(n_splits=5)
        evaluation_start_time = time.time()
        scores = cross_validate(best_model, X_train, y_train, cv=kf, scoring=scoring)
        evaluation_end_time = time.time()
        accuracy = scores['test_accuracy'].mean()
        precision = scores['test_precision'].mean()
        recall = scores['test_recall'].mean()
        f1 = scores['test_f1'].mean()
        tn = fp = fn = tp = None
        best_model.fit(X_train, y_train)
    else:
        evaluation_start_time = time.time()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        evaluation_end_time = time.time()

    return best_model, accuracy, precision, recall, f1, tn, fp, fn, tp, grid_search_start_time, grid_search_end_time, evaluation_start_time, evaluation_end_time

def feature_selection(X, y, method, num_features, model=None):
    feature_selection_start_time = time.time()
    if method == "sfs":
        sfs = SequentialFeatureSelector(model, n_features_to_select=num_features)
        sfs.fit(X, y)
        indices = sfs.get_support(indices=True)
    elif method == "chi2":
        chi2_selector = SelectKBest(chi2, k=num_features)
        chi2_selector.fit(X, y)
        indices = chi2_selector.get_support(indices=True)
    elif method == "embedded":
        rf = RandomForestClassifier(random_state=0)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = importances.argsort()[::-1][:num_features]
    feature_selection_end_time = time.time()
    return indices, feature_selection_start_time, feature_selection_end_time

# Flask route for index page
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    accuracy = precision = recall = f1 = tn = fp = fn = tp = waktu = None

    def convert_time(seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"

    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in features]
            input_data = pd.DataFrame([input_data], columns=features)
            input_data = scaler.transform(input_data)

            algorithm = request.form["algorithm"]
            feature_selection_method = request.form["feature_selection"]
            gridsearch = request.form["gridsearch"] == "yes"
            num_features = int(request.form.get("num_features", 30))
            validation_method = request.form["validation"]
            start_time = time.time()

            if algorithm == "SVM":
                model = SVC(kernel='rbf', C=1.0, gamma='scale')
                param_grid = {
                    "kernel": ["linear", "poly", "rbf"],
                    "gamma": [1, 0.1, 0.01, 0.001],
                    "C": [0.1, 1, 10, 100, 1000],
                } if gridsearch else None
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=0)
                param_grid = {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [2, 4, 6, 8, 10],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [1, 5, 10],
                } if gridsearch else None

            if validation_method == "split":
                splitting_start_time = time.time()
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
                splitting_end_time = time.time()
                if feature_selection_method != "none" and num_features < 30:
                    indices, feature_selection_start_time, feature_selection_end_time = feature_selection(X_train, y_train, feature_selection_method, num_features, model)
                    X_train = X_train[:, indices]
                    X_test = X_test[:, indices]
                    input_data = input_data[:, indices]
                else:
                    feature_selection_start_time = feature_selection_end_time = 0
                best_model, accuracy, precision, recall, f1, tn, fp, fn, tp, grid_search_start_time, grid_search_end_time, evaluation_start_time, evaluation_end_time = train_and_evaluate(
                    X_train, X_test, y_train, y_test, model, param_grid, validation_method
                )
            else:
                splitting_start_time = splitting_end_time = 0
                if feature_selection_method != "none" and num_features < 30:
                    indices, feature_selection_start_time, feature_selection_end_time = feature_selection(X_scaled, y, feature_selection_method, num_features, model)
                    X_selected = X_scaled[:, indices]
                    input_data = input_data[:, indices]
                else:
                    X_selected = X_scaled
                    feature_selection_start_time = feature_selection_end_time = 0
                best_model, accuracy, precision, recall, f1, tn, fp, fn, tp, grid_search_start_time, grid_search_end_time, evaluation_start_time, evaluation_end_time = train_and_evaluate(
                    X_selected, None, y, None, model, param_grid, validation_method
                )

            prediction = best_model.predict(input_data)[0]
            result = "Malignant" if prediction == 1 else "Benign"
            end_time = time.time()

            total_execution_time = end_time - start_time
            print("\nWaktu eksekusi per bagian:")
            print("Scaling time:", convert_time(scaling_end_time - scaling_start_time))
            print("Splitting time:", convert_time(splitting_end_time - splitting_start_time))
            print("Feature selection time:", convert_time(feature_selection_end_time - feature_selection_start_time))
            print("GridSearchCV time:", convert_time(grid_search_end_time - grid_search_start_time))
            print("Evaluation time:", convert_time(evaluation_end_time - evaluation_start_time))
            print("Total execution time:", convert_time(total_execution_time))
            waktu = convert_time(total_execution_time)
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")

    return render_template("index.html", features=features, result=result, accuracy=accuracy,
                           precision=precision, recall=recall, f1=f1, tn=tn, fp=fp, fn=fn, tp=tp, data=data_list, waktu=waktu)

if __name__ == "__main__":
    app.run(debug=True)
