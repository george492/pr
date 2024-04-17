from idlelib import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import tkinter as tk
from scipy.stats import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import tkinter.messagebox as messagebox

# Load the dataset
data1 = pd.read_csv("water_potability - water_potability.csv.csv")
# Data Preprocessing
# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data1), columns=data1.columns)
# Removing Duplicate Rows
data.drop_duplicates(inplace=True)
# Handling Class Imbalance (Up-sampling Minority Class)
majority_class = data[data['Potability'] == 0]
minority_class = data[data['Potability'] == 1]
minority_upsampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)
data = pd.concat([majority_class, minority_upsampled])
# Feature Engineering - Interaction Terms
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interaction = poly.fit_transform(data.drop('Potability', axis=1))
# Feature Selection
y = data['Potability']
# Splitting into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_interaction, y, test_size=0.2, random_state=42)
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model Training (Decision Tree)
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_scaled, y_train)
dt_accuracy = dt_classifier.score(X_test_scaled, y_test)
# Model Training (SVM)
svm_classifier = SVC()
svm_classifier.fit(X_train_scaled, y_train)
svm_accuracy = svm_classifier.score(X_test_scaled, y_test)
# Model Training (Logistic Regression with L2 Regularization)
lr_classifier = LogisticRegression(penalty='l2', C=1.0)  # L2 regularization with default regularization strength
lr_classifier.fit(X_train_scaled, y_train)
lr_accuracy = lr_classifier.score(X_test_scaled, y_test)
# Model Training (Random Forest)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
rf_accuracy = rf_classifier.score(X_test_scaled, y_test)
# Evaluating Models
# Decision Tree
dt_predictions = dt_classifier.predict(X_test_scaled)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
print("Confusion Matrix for Decision Tree:")
print(dt_confusion_matrix)
print("Decision Tree Accuracy:", dt_accuracy)
# SVM
svm_predictions = svm_classifier.predict(X_test_scaled)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
print("\nConfusion Matrix for SVM:")
print(svm_confusion_matrix)
print("SVM Accuracy:", svm_accuracy)
# Logistic Regression
lr_predictions = lr_classifier.predict(X_test_scaled)
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)
print("\nConfusion Matrix for Logistic Regression:")
print(lr_confusion_matrix)
print("Logistic Regression Accuracy:", lr_accuracy)
# Random Forest
rf_predictions = rf_classifier.predict(X_test_scaled)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
print("\nConfusion Matrix for Random Forest:")
print(rf_confusion_matrix)
print("Random Forest Accuracy:", rf_accuracy)
import tkinter as tk
from tkinter import messagebox

class PreprocessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Water Potability Test")
        self.root.geometry("1920x1080")

        self.label = tk.Label(root, text="Welcome to water potability test!", font=("Arial", 14))
        self.label.pack(pady=10)

        self.options = ["Svm", "Decison tree", "Logestic regression", "random forest"]
        self.selected_options = []

        for option in self.options:
            var = tk.IntVar()
            checkbox = tk.Radiobutton(root, text=option, variable=var, command=self.update_selection)
            checkbox.pack(anchor=tk.W)
            setattr(self, f"var_{option}", var)

        self.variable_names = [
            "pH", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic carbon",
            "Trihalomethanes", "Turbidity"
        ]

        self.entries = {}

        for name in self.variable_names:
            label = tk.Label(root, text=name + ":", bg="lightblue")
            label.pack(anchor=tk.W, padx=10, pady=5)

            entry = tk.Entry(root)
            entry.pack(anchor=tk.W, padx=10)
            self.entries[name] = entry

        self.submit_button = tk.Button(root, text="Submit", command=self.submit_data)
        self.submit_button.pack(pady=10)

    def submit_data(self):
        data = {name: entry.get() for name, entry in self.entries.items()}
        for key, value in data.items():
            if value == '':
                messagebox.showerror("Error", "Please fill in all fields.")
                self.clear_entries()
                return
            try:
                float_value = float(value)
                if float_value < 0:
                    messagebox.showerror("Error", "Please enter positive values.")
                    self.clear_entries()
                    return
            except ValueError:
                messagebox.showerror("Error", "Please fill in all values numerical.")
                self.clear_entries()
                return
        print("Entered data:", data)
        self.selected_options = [option for option in self.options if getattr(self, f"var_{option}").get() == 1]
        if self.selected_options:
            print("You voted for:", ", ".join(self.selected_options))
        else:
            print("Please select at least one option.")
    def update_selection(self):
        self.selected_options = [option for option in self.options if getattr(self, f"var_{option}").get() == 1]

    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, 'end')

if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingApp(root)
    root.mainloop()
