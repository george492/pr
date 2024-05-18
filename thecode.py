import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
import random
import tkinter.messagebox as messagebox
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# preprocessing
data1 = pd.read_csv("water_potability - water_potability.csv.csv")

imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data1), columns=data1.columns)

data.drop_duplicates(inplace=True)

majority_class = data[data['Potability'] == 0]
minority_class = data[data['Potability'] == 1]
minority_upsampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)
data = pd.concat([majority_class, minority_upsampled])

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interaction = poly.fit_transform(data.drop('Potability', axis=1))

y = data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X_interaction, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
interaction_poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_interaction = interaction_poly.fit_transform(data.drop('Potability', axis=1))

#gridsearch for DT

'''
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_param_grid = {
    'max_depth': [20,40,50],
    'criterion': ['entropy','gini'],
    'min_samples_split': [2,4,8],
    'min_samples_leaf': [1,2,3],
}

dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_param_grid, cv=20, scoring='accuracy')
dt_grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters for Decision Tree
best_dt_params = dt_grid_search.best_params_
'''

# DT Model with best hyperparameters
dt_classifier = DecisionTreeClassifier(max_depth=40, criterion='entropy', min_samples_split=2, min_samples_leaf=1, random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

dt_y_pred = dt_classifier.predict(X_test_scaled)

dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_conf_matrix = confusion_matrix(y_test, dt_y_pred)

'''
# Grid search model Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [1000,1100,1200,1300,1400],
    'max_depth': [5,10,15,20],
    'min_samples_split': [2,4,8],
    'min_samples_leaf': [2,4,8],
    'class_weight': ['balanced',None], 
    'criterion': ['gini','entropy'], 
    'bootstrap': [False,True]  
}

rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, cv=10, scoring='accuracy')
rf_grid_search.fit(X_train_scaled, y_train)

best_rf_params = rf_grid_search.best_params_
'''
#  Random Forest model with the Best hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=1400, max_depth=20, min_samples_split=2, min_samples_leaf=2, class_weight='balanced', criterion='gini', bootstrap=False, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

rf_y_pred = rf_classifier.predict(X_test_scaled)  # Use X_test_scaled instead of X_train_scaled

rf_accuracy = accuracy_score(y_test, rf_y_pred)  # Use y_test instead of y_train
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)  # Use y_test instead of y_train

'''
# grid search for logestic regression

lr_param_grid = {
    'C': [10,20,100],
    'tol': [0.001,0.0001,0.0001] ,
    'max_iter': [100,200,1000],
    'solver': ['saga'],
    'penalty': ['l1','l2'],
    'class_weight': ['balanced',None],
}

lr_classifier = LogisticRegression()
lr_grid_search = GridSearchCV(estimator=lr_classifier, param_grid=lr_param_grid, cv=20, scoring='accuracy')
lr_grid_search.fit(X_train_scaled, y_train)

best_lr_params = lr_grid_search.best_params_
'''
# best hyperparameters for logestic regression
lr_classifier = LogisticRegression(C=20, penalty='l2', solver='saga', max_iter=1000, class_weight='balanced', tol=0.0001)
lr_classifier.fit(X_train_scaled, y_train)

lr_y_pred = lr_classifier.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_conf_matrix = confusion_matrix(y_test, lr_y_pred)

'''
# grid search for SVM
param_grid = {
'C': [10, 20, 100],
'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
'gamma': ['scale', 'auto', 0.1, 1.0]
}
svm_classifier = SVC()
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
'''
# svm  Model with best hyperparameters
svm_classifier = SVC(C=20,kernel='rbf')
svm_classifier.fit(X_train_scaled, y_train)

svm_y_pred = svm_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, svm_y_pred)
conf_matrix = confusion_matrix(y_test, svm_y_pred)

'''
# KNN model grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
}

knn_classifier = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
'''
#best hyperparameters for knn
knn_classifier = KNeighborsClassifier(n_neighbors=9, weights='distance')

knn_classifier.fit(X_train_scaled, y_train)

knn_y_pred = knn_classifier.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_conf_matrix = confusion_matrix(y_test, knn_y_pred)

#gui

class FirstWindow:
    def __init__(self, eroot):
        self.root = eroot
        self.root.geometry("1000x800")
        self.root.title("Water Potability System")
        self.root.configure(bg="#00203F")

        self.message_label = tk.Label(eroot, bg="#ADEFD1", text="Safe and Clean water", font=("Helvetica", 14))
        self.message_label.pack(pady=(20, 0))

        self.open_models_button = tk.Button(eroot, bg="#ADEFD1", text="Models", command=self.open_second_window)
        self.open_models_button.bind("<Enter>", lambda e: self.open_models_button.config(bg="lightblue"))
        self.open_models_button.bind("<Leave>", lambda e: self.open_models_button.config(bg="#ADEFD1"))
        self.open_models_button.pack(pady=10)

        self.open_modelsPrediction_button = tk.Button(eroot, bg="#ADEFD1", text="Prediction(Sample Entry)",
                                                      command=self.open_third_window)
        self.open_modelsPrediction_button.pack(pady=5)
        self.open_modelsPrediction_button.bind("<Enter>", lambda e: self.open_modelsPrediction_button.config(
            bg="lightblue"))
        self.open_modelsPrediction_button.bind("<Leave>", lambda e: self.open_modelsPrediction_button.config(
            bg="#ADEFD1"))

        self.open_eda_button = tk.Button(eroot, bg="#ADEFD1", text="Exploratory Data Analysis",
                                         command=self.open_fourth_window)
        self.open_eda_button.pack(pady=5)
        self.open_eda_button.bind("<Enter>", lambda e: self.open_eda_button.config(bg="lightblue"))
        self.open_eda_button.bind("<Leave>", lambda e: self.open_eda_button.config(bg="#ADEFD1"))

        self.close_button = tk.Button(self.root, bg="#ADEFD1", text="Close", command=self.root.destroy)
        self.close_button.bind("<Enter>", lambda e: self.close_button.config(bg="lightblue"))
        self.close_button.bind("<Leave>", lambda e: self.close_button.config(bg="#ADEFD1"))
        self.close_button.pack(side="bottom", anchor="center", pady=(0, 30))

    def open_second_window(self):
        self.root.withdraw()
        second_window = tk.Toplevel()
        SecondWindow(second_window)

    def open_third_window(self):
        self.root.withdraw()
        third_window = tk.Toplevel()
        ThirdWindow(third_window)

    def open_fourth_window(self):
        self.root.withdraw()
        fourth_window = tk.Toplevel()

        FourthWindow(fourth_window, data1)

class SecondWindow:
    def __init__(self, eroot):
        self.root = eroot
        self.root.geometry("1000x800")
        self.root.title("Water Potability Models")
        self.root.configure(bg="#00203F")

        self.checkbox1 = tk.Checkbutton(self.root, text="DT", command=self.dt_info)
        self.checkbox1.config(width=10, height=2)
        self.checkbox2 = tk.Checkbutton(self.root, text="LR", command=self.lr_info)
        self.checkbox2.config(width=10, height=2)
        self.checkbox3 = tk.Checkbutton(self.root, text="SVM", command=self.svm_info)
        self.checkbox3.config(width=10, height=2)
        self.checkbox4 = tk.Checkbutton(self.root, text="RF", command=self.rf_info)
        self.checkbox4.config(width=10, height=2)
        self.checkbox5 = tk.Checkbutton(self.root, text="KNN", command=self.Knn_info)
        self.checkbox5.config(width=10, height=2)

        self.checkbox1.bind("<Enter>", lambda e: self.checkbox1.config(bg="lightblue"))
        self.checkbox1.bind("<Leave>", lambda e: self.checkbox1.config(bg="#ADEFD1"))

        self.checkbox2.bind("<Enter>", lambda e: self.checkbox2.config(bg="lightblue"))
        self.checkbox2.bind("<Leave>", lambda e: self.checkbox2.config(bg="#ADEFD1"))

        self.checkbox3.bind("<Enter>", lambda e: self.checkbox3.config(bg="lightblue"))
        self.checkbox3.bind("<Leave>", lambda e: self.checkbox3.config(bg="#ADEFD1"))

        self.checkbox4.bind("<Enter>", lambda e: self.checkbox4.config(bg="lightblue"))
        self.checkbox4.bind("<Leave>", lambda e: self.checkbox4.config(bg="#ADEFD1"))

        self.open_models_button1 = tk.Button(self.root, bg="#ADEFD1", text="Prediction(Sample Entry)",
                                             command=self.open_third_window)
        self.open_models_button1.bind("<Enter>", lambda e: self.open_models_button1.config(bg="lightblue"))
        self.open_models_button1.bind("<Leave>", lambda e: self.open_models_button1.config(bg="#ADEFD1"))

        self.open_models_button2 = tk.Button(self.root, bg="#ADEFD1", text="Main Menu", command=self.open_first_window)
        self.open_models_button2.bind("<Enter>", lambda e: self.open_models_button2.config(bg="lightblue"))
        self.open_models_button2.bind("<Leave>", lambda e: self.open_models_button2.config(bg="#ADEFD1"))
        self.open_models_button2.pack(side="bottom", anchor="center", pady=(0, 30))
        self.open_models_button1.pack(pady=(30, 0))

        self.checkbox1.pack(pady=10)
        self.checkbox2.pack(pady=10)
        self.checkbox3.pack(pady=10)
        self.checkbox4.pack(pady=10)
        self.checkbox5.pack(pady=10)

        self.root.mainloop()

    @staticmethod
    def dt_info():
        messagebox.showinfo("Decision Tree \n",
                            f"Accuracy: {dt_accuracy:.2f}\nConfusion Matrix:\n{dt_conf_matrix}")

    def open_third_window(self):
        self.root.withdraw()
        third_window = tk.Toplevel()
        ThirdWindow(third_window)

    def open_first_window(self):
        self.root.withdraw()
        first_window = tk.Toplevel()
        FirstWindow(first_window)

    @staticmethod
    def lr_info():
        messagebox.showinfo("Logistic Regression\n",
                            f"Accuracy: {lr_accuracy:.2f}\nConfusion Matrix:\n{lr_conf_matrix}")

    @staticmethod
    def svm_info():
        messagebox.showinfo("SVM Confusion \n",
                            f"Accuracy: {accuracy:.2f}\nConfusion Matrix:\n{conf_matrix}")

    @staticmethod
    def rf_info():
        messagebox.showinfo("Random forest \n",
                            f"Accuracy: {rf_accuracy:.2f}\nConfusion Matrix:\n{rf_conf_matrix}")
    @staticmethod
    def Knn_info():
        messagebox.showinfo("KNN \n",
                            f"Accuracy: {knn_accuracy:.2f}\nConfusion Matrix:\n{knn_conf_matrix}")

class ThirdWindow:
    def __init__(self, eroot):
        self.root = eroot
        self.root.geometry("1200x1000")
        self.root.title("Water Potability Models")
        self.root.configure(bg="#00203F")

        self.selected_option = tk.StringVar()
        self.options = ["Svm", "Decision tree", "Logistic regression", "Random forest", "Knn"]
        for option in self.options:
            radiobutton = tk.Radiobutton(self.root, bg="#ADEFD1", text=option,
                                         variable=self.selected_option, value=option)
            radiobutton.config(width=15, height=2)
            radiobutton.pack(pady=10)

        tk.Label(self.root, text="", bg="#00203F").pack()

        self.variable_names = [
            "pH", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic carbon",
            "Trihalomethanes", "Turbidity"
        ]
        self.entry_ranges = [(0, 14), (47, 323), (320, 6123), (0, 14), (129, 482), (181, 753), (2, 29), (0, 125),
                             (1, 7)]

        self.entries = {}
        for name, (min_val, max_val) in zip(self.variable_names, self.entry_ranges):
            label = tk.Label(self.root, text=name, bg="#ADEFD1")
            label.config(width=15, height=1)
            label.pack()

            entry = tk.Entry(self.root)
            entry.config(width=15)
            entry.pack()

            # Add entry to the dictionary with its associated range
            self.entries[name] = {'entry': entry, 'range': (min_val, max_val)}

        self.submit_button = tk.Button(self.root, bg="#ADEFD1", text="Submit", command=self.submit_data)
        self.submit_button.pack(pady=10)

        self.random_fill_button = tk.Button(self.root, bg="#ADEFD1", text="Random Fill", command=self.random_fill)
        self.random_fill_button.pack(pady=10)

        self.open_models_button3 = tk.Button(self.root, bg="#ADEFD1", text="Models", command=self.open_second_window)
        self.open_models_button3.pack(pady=10)

        self.open_models_button2 = tk.Button(self.root, bg="#ADEFD1", text="Main Menu", command=self.open_first_window)
        self.open_models_button2.pack(side="bottom", anchor="center", pady=(0, 30))

    def random_fill(self):
        for name, entry_info in self.entries.items():
            entry = entry_info['entry']
            min_val, max_val = entry_info['range']
            value = str(random.randint(min_val, max_val))
            entry.delete(0, 'end')
            entry.insert(0, value)

    def submit_data(self):
        mydata = {name: entry_info['entry'].get() for name, entry_info in self.entries.items()}
        for name, entry_info in self.entries.items():
            entry = entry_info['entry']
            min_val, max_val = entry_info['range']
            value = entry.get()
            if value == '':
                messagebox.showerror("Error", "Please fill in all fields.")
                return
            try:
                float_value = float(value)
                if not (min_val <= float_value <= max_val):
                    messagebox.showerror("Error", f"Please enter a value between {min_val} and {max_val} for {name}.")
                    return
            except ValueError:
                messagebox.showerror("Error", f"Please fill in a numerical value for {name}.")
                return

        if self.selected_option.get():
            prediction = ""
            if self.selected_option.get() == "Svm":
                prediction = self.predict_svm(mydata)
            elif self.selected_option.get() == "Decision tree":
                prediction = self.predict_decision_tree(mydata)
            elif self.selected_option.get() == "Logistic regression":
                prediction = self.predict_logistic_regression(mydata)
            elif self.selected_option.get() == "Random forest":
                prediction = self.predict_random_forest(mydata)
            elif self.selected_option.get() == "Knn":
                prediction = self.predict_Knn(mydata)
            messagebox.showinfo("Prediction", f"Prediction: {int(prediction[0])}")
        else:
            messagebox.showerror("Error", "Please select one option.")

    def open_first_window(self):
        self.root.withdraw()
        first_window = tk.Toplevel()
        FirstWindow(first_window)

    def open_second_window(self):
        self.root.withdraw()
        second_window = tk.Toplevel()
        SecondWindow(second_window)

    def predict_svm(self, input_data):
        input_svm = [float(input_data[key]) for key in self.variable_names]
        array_svm = np.array(input_svm).reshape(1, -1)
        input_interaction_svm = poly.transform(array_svm)
        input_scaled_svm = scaler.transform(input_interaction_svm)
        prediction = svm_classifier.predict(input_scaled_svm)
        return prediction

    def predict_decision_tree(self, input_data):
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)
        input_scaled = scaler.transform(input_interaction)
        prediction = dt_classifier.predict(input_scaled)
        return prediction
    def predict_Knn(self, input_data):
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)
        input_scaled = scaler.transform(input_interaction)
        prediction = knn_classifier.predict(input_scaled)
        return prediction
    def predict_random_forest(self, input_data):
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)
        input_scaled = scaler.transform(input_interaction)
        prediction = rf_classifier.predict(input_scaled)
        return prediction

    def predict_logistic_regression(self, input_data):
        input_values = [float(input_data[key]) for key in self.variable_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_interaction = poly.transform(input_array)
        input_scaled = scaler.transform(input_interaction)
        prediction = lr_classifier.predict(input_scaled)
        return prediction


class FourthWindow:
    def __init__(self, eroot, data):
        self.root = eroot
        self.root.geometry("1200x1000")
        self.root.title("Exploratory Data Analysis and Data Preprocessing")
        self.root.configure(bg="#00203F")

        self.eda_button = tk.Button(self.root, bg="#ADEFD1", text="EDA Information", command=self.display_eda_info)
        self.eda_button.pack(pady=10)

        self.preprocess_button = tk.Button(self.root, bg="#ADEFD1", text="Data Preprocessing Summary",
                                           command=self.display_preprocess_info)
        self.preprocess_button.pack(pady=10)

        self.return_button = tk.Button(self.root, bg="#ADEFD1", text="Return to Main Menu",
                                       command=self.open_first_window)
        self.return_button.pack(side="bottom", anchor="center", pady=(0, 30))

        self.data = data

    def display_eda_info(self):
        # Create a new window to display the EDA information
        eda_window = tk.Toplevel(self.root)
        eda_window.geometry("800x600")
        eda_window.title("Exploratory Data Analysis (EDA) Information")
        eda_window.configure(bg="#00203F")

        # Display the information
        info_text = """
        Exploratory Data Analysis (EDA) Summary:
        - Basic statistics of the dataset:
        """
        info_text += str(self.data.head()) + "\n"

        info_text += "Shape of The DataSet\n"
        info_text += str(self.data.shape) + "\n"

        info_text += "Data Types of the Features\n"
        info_text += str(self.data.dtypes) + "\n"

        info_text += "Sum of Null Values for each Feature\n"
        info_text += str(self.data.isnull().sum()) + "\n"

        info_text += "\nSum of Duplicated Values in the Dataset: " + str(self.data.duplicated().sum()) + "\n"

        # Display the potability percentage
        potable_percentage = self.data.Potability[self.data.Potability == 1].count() / self.data.Potability.count() * 100
        info_text += f"{potable_percentage:.2f} % of samples are potable (1)\n"

        # Create a figure and subplots for count plot and heatmap
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

        # Count plot
        sns.countplot(x=self.data['Potability'], ax=axes[0])
        axes[0].set_title('Potability Count')
        axes[0].set_xlabel('Potability')
        axes[0].set_ylabel('Count')

        # Heatmap
        sns.heatmap(self.data.corr(), annot=True, cmap="inferno", ax=axes[1])
        axes[1].set_title('Correlation Heatmap')

        # Adjust layout
        plt.tight_layout()

        # Convert the figure to a Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=eda_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Add a Text widget to display the information
        info_label = tk.Label(eda_window, text=info_text, justify=tk.LEFT)
        info_label.pack(expand=True, fill=tk.BOTH)

    def display_preprocess_info(self):
        preprocess_info = """
        Data Preprocessing Summary:
        - Handling missing values: Replaced null values with mean for 'ph', 'Sulfate', and 'Trihalomethanes'.
        - Feature Scaling: Applied StandardScaler to normalize features.
        - Polynomial Features: Created interaction terms up to degree 2 using PolynomialFeatures.
        - Class Balancing: Addressed class imbalance by up-sampling the minority class.
        - Outlier Handling: Checked for and handled outliers in the dataset.
        - Feature Selection: Considered relevant features for modeling.
        - Data Splitting: Split the data into training and testing subsets.
        """

        # Create a new window to display the preprocessing information
        preprocess_window = tk.Toplevel(self.root)
        preprocess_window.geometry("600x400")
        preprocess_window.title("Data Preprocessing Summary")
        preprocess_window.configure(bg="#00203F")

        # Add a Text widget to display the preprocessing information
        preprocess_label = tk.Label(preprocess_window, text=preprocess_info, justify=tk.LEFT)
        preprocess_label.pack(expand=True, fill=tk.BOTH)

    def open_first_window(self):
        self.root.destroy()
        first_window = tk.Tk()
        FirstWindow(first_window)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = FirstWindow(root)
    root.mainloop()