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
# print("Confusion Matrix for Decision Tree:")
# print(dt_confusion_matrix)
# print("Decision Tree Accuracy:", dt_accuracy)
# SVM
svm_predictions = svm_classifier.predict(X_test_scaled)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
# print("\nConfusion Matrix for SVM:")
# print(svm_confusion_matrix)
# print("SVM Accuracy:", svm_accuracy)
# Logistic Regression
lr_predictions = lr_classifier.predict(X_test_scaled)
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)
# print("\nConfusion Matrix for Logistic Regression:")
# print(lr_confusion_matrix)
# print("Logistic Regression Accuracy:", lr_accuracy)
# Random Forest
rf_predictions = rf_classifier.predict(X_test_scaled)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
# print("\nConfusion Matrix for Random Forest:")
# print(rf_confusion_matrix)
# print("Random Forest Accuracy:", rf_accuracy)
class FirstWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Water Potability System")
        # Load and display an image
        # try:
        #     self.image = tk.PhotoImage(file="CleanWater.png")
        #     self.image_label = tk.Label(root, image=self.image)
        #     self.image_label.pack()
        # except tk.TclError:
        #     messagebox.showerror("Error", "Image file not found!")

        # Display message
        self.message_label = tk.Label(root, text="Safe and Clean water", font=("Helvetica", 14))
        self.message_label.pack()

        # Button to open the second window
        self.open_models_button = tk.Button(root, text="Models", command=self.open_second_window)
        self.open_models_button.pack()

        self.open_models_button = tk.Button(root, text="predict", command=self.open_third_window)
        self.open_models_button.pack()

        self.open_models_button.bind("<Enter>", lambda e: self.open_models_button.config(bg="green"))
        self.open_models_button.bind("<Leave>", lambda e: self.open_models_button.config(bg="SystemButtonFace"))

    def open_second_window(self):
        self.root.withdraw()  # Hide the first window
        second_window = tk.Toplevel()  # Create a new top-level window
        SecondWindow(second_window)

    def open_third_window(self):
        self.root.withdraw()  # Hide the first window
        third_window = tk.Toplevel()  # Create a new top-level window
        ThirdWindow(third_window)


class SecondWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x700")
        self.root.title("Water Potability Models")

        # Create three checkboxes
        self.checkbox1 = tk.Checkbutton(self.root, text="DT", command=self.dt_info)
        self.checkbox2 = tk.Checkbutton(self.root, text="LR", command=self.lr_info)
        self.checkbox3 = tk.Checkbutton(self.root, text="SVM", command=self.svm_info)
        self.checkbox4 = tk.Checkbutton(self.root, text="RF", command=self.rf_info)
        # Add hover effects to buttons
        self.checkbox1.bind("<Enter>", lambda e: self.checkbox1.config(bg="lightblue"))
        self.checkbox1.bind("<Leave>", lambda e: self.checkbox1.config(bg="SystemButtonFace"))

        self.checkbox2.bind("<Enter>", lambda e: self.checkbox2.config(bg="purple"))
        self.checkbox2.bind("<Leave>", lambda e: self.checkbox2.config(bg="SystemButtonFace"))

        self.checkbox3.bind("<Enter>", lambda e: self.checkbox3.config(bg="lightcoral"))
        self.checkbox3.bind("<Leave>", lambda e: self.checkbox3.config(bg="SystemButtonFace"))

        self.checkbox4.bind("<Enter>", lambda e: self.checkbox4.config(bg="lightcoral"))
        self.checkbox4.bind("<Leave>", lambda e: self.checkbox4.config(bg="SystemButtonFace"))

        # Button to close the second window
        self.close_button = tk.Button(self.root, text="Close", command=self.root.destroy)
        self.open_models_button1 = tk.Button(self.root, text="predict", command=self.open_third_window)
        self.open_models_button1.pack()
        self.open_models_button2 = tk.Button(self.root, text="main menu", command=self.open_first_window)
        self.open_models_button2.pack()
        self.close_button.bind("<Enter>", lambda e: self.close_button.config(bg="pink"))
        self.close_button.bind("<Leave>", lambda e: self.close_button.config(bg="SystemButtonFace"))
        # Arrange widgets using grid or pack as needed
        self.checkbox1.pack()
        self.checkbox2.pack()
        self.checkbox3.pack()
        self.checkbox4.pack()
        self.close_button.pack()

        self.root.mainloop()

    def dt_info(self):
        # Calculate and display accuracy and confusion matrix for Decision Tree
        messagebox.showinfo("Decision Tree \n",
                            f"Accuracy: {dt_accuracy:.2f}\nConfusion Matrix:\n{dt_confusion_matrix}")

    def open_third_window(self):
        self.root.withdraw()  # Hide the current window
        third_window = tk.Toplevel()  # Create a new top-level window
        ThirdWindow(third_window)

    def open_first_window(self):
        self.root.withdraw()  # Hide the current window
        first_window = tk.Toplevel()  # Create a new top-level window
        FirstWindow(first_window)

    def lr_info(self):
        messagebox.showinfo("Logistic Regression\n",
                            f"Accuracy: {lr_accuracy:.2f}\nConfusion Matrix:\n{lr_confusion_matrix}")

    def svm_info(self):
        messagebox.showinfo("SVM Confusion Matrix",
                            f"Accuracy: {svm_accuracy:.2f}\nConfusion Matrix:\n{svm_confusion_matrix}")

    def rf_info(self):
        messagebox.showinfo("SVM Confusion Matrix",
                            f"Accuracy: {rf_accuracy:.2f}\nConfusion Matrix:\n{rf_confusion_matrix}")


class ThirdWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x700")
        self.root.title("Water Potability Models")

        self.selected_option = tk.StringVar()  # Shared variable for radiobuttons

        self.options = ["Svm", "Decision tree", "Logistic regression", "Random forest"]

        for option in self.options:
            radiobutton = tk.Radiobutton(self.root, text=option, variable=self.selected_option, value=option)
            radiobutton.pack(anchor=tk.W)

        self.variable_names = [
            "pH", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic carbon",
            "Trihalomethanes", "Turbidity"
        ]

        self.entries = {}

        for name in self.variable_names:
            label = tk.Label(self.root, text=name + ":", bg="lightblue")
            label.pack(anchor=tk.W, padx=10, pady=5)

            entry = tk.Entry(self.root)
            entry.pack(anchor=tk.W, padx=10)
            self.entries[name] = entry

        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit_data)
        self.submit_button.pack(pady=10)
        self.open_models_button2 = tk.Button(self.root, text="main menu", command=self.open_first_window)
        self.open_models_button2.pack()
        self.open_models_button3 = tk.Button(self.root, text="Models", command=self.open_second_window)
        self.open_models_button3.pack()

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
        if self.selected_option.get():
            prediction = ""
            if self.selected_option.get() == "Svm":
                prediction = self.predict_svm(data)
            elif self.selected_option.get() == "Decision tree":
                prediction = self.predict_decision_tree(data)
            elif self.selected_option.get() == "Logistic regression":
                prediction = self.predict_logistic_regression(data)
            elif self.selected_option.get() == "Random forest":
                prediction = self.predict_random_forest(data)
            messagebox.showinfo("Prediction", f"Prediction: {prediction}")
        else:
            messagebox.showerror("Error", "Please select one option.")
    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, 'end')

    def open_first_window(self):
        self.root.withdraw()  # Hide the current window
        first_window = tk.Toplevel()  # Create a new top-level window
        FirstWindow(first_window)

    def open_second_window(self):
        self.root.withdraw()  # Hide the current window
        second_window = tk.Toplevel()  # Create a new top-level window
        SecondWindow(second_window)

    def predict_svm(self, input_data):
        # Apply the same preprocessing steps as during training
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)  # Apply PolynomialFeatures
        input_scaled = scaler.transform(input_interaction)  # Apply StandardScaler
        # Predict using the SVM classifier
        prediction = svm_classifier.predict(input_scaled)
        return prediction

    def predict_decision_tree(self, input_data):
        # Apply the same preprocessing steps as during training
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)  # Apply PolynomialFeatures
        input_scaled = scaler.transform(input_interaction)  # Apply StandardScaler
        # Predict using the Decision Tree classifier
        prediction = dt_classifier.predict(input_scaled)
        return prediction

    def predict_random_forest(self, input_data):
        # Apply the same preprocessing steps as during training
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)  # Apply PolynomialFeatures
        input_scaled = scaler.transform(input_interaction)  # Apply StandardScaler
        # Predict using the Random Forest classifier
        prediction = rf_classifier.predict(input_scaled)
        return prediction

    def predict_logistic_regression(self, input_data):
        # Apply the same preprocessing steps as during training
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)  # Apply PolynomialFeatures
        input_scaled = scaler.transform(input_interaction)  # Apply StandardScaler
        # Predict using the Logistic Regression classifier
        prediction = lr_classifier.predict(input_scaled)
        return prediction
if __name__ == "__main__":
    root = tk.Tk()
    app = FirstWindow(root)
    root.mainloop()