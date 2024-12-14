#Necessary libraries to import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Load the csv data

df = pd.read_csv('chronic_kidney_disease.csv')

# to view the top 5 patient's record of the data
df.head()

# to view the top 20 patient's record of the data
df.head(20)

# to view the last 5 patient's record of the data
df.tail()

# to view the last 10 patient's record of the data
df.tail(10)

# to check the data dimensions in terms of rows by columns use the df.shape()
df.shape

# to drop/remove any attribute eg 'id' that is not important use df.drop()
df.drop('id', axis = 1, inplace = True)

# to check the effect of the column dropped use the df.head()
df.head()

#This function is useful for quickly getting an overview of the distribution and central tendency of the data.
df.describe()

# df.info() prints information about the dataframe.
df.info()

df

#As we can see that some attributes values such as 'packed_cell_volume', 'white_blood_cell_count' and 'red_blood_cell_count' are object type. 
#We need to change them to numerical dtype.

# converting necessary columns in object type to numerical type

df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')


#confirm the datatype changes with df.info()
df.info()

# Extracting categorical and numerical columns

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# looking at unique values in categorical columns

for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")

# replace incorrect values

df['dm'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

df['cad'] = df['cad'].replace(to_replace = '\tno', value='no')

df['classification'] = df['classification'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})

#print the columns belwo to see their current values
cols = ['dm', 'cad', 'classification']

for col in cols:
    print(f"{col} has {df[col].unique()} values\n")

df.head()

dfnum=df.drop(columns=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification'])

dfnum

from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

lr=LinearRegression()
imp=IterativeImputer(estimator=lr,verbose=2,max_iter=100, tol=1e-10, imputation_order='roman')

dfnum2=imp.fit_transform(dfnum)

dfnum3=pd.DataFrame(dfnum2,columns=('age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc'))

dfnum3.head(5)

df.head()

dfcat =df.drop(columns=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','classification'])

dfcat

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='constant',fill_value='Missing')
dfcat2=imputer.fit_transform(dfcat)

dfcat3=pd.DataFrame(dfcat2,columns=('rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'))

dfcat3.head()

from sklearn.preprocessing import OrdinalEncoder
ordi=OrdinalEncoder()

dfcat4=ordi.fit_transform(dfcat3)

dfcat5=pd.DataFrame(dfcat4,columns=('rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'))

dfcat5

x=pd.concat([dfnum3,dfcat5],axis=1)

x

#our target prediction, y - is to predict if chronic kidney disease or not.
y=df['classification']

a=pd.concat([x,y],axis=1)

a

a.head()

column_names = a.columns[:25]
column_names

u=a.iloc[:,:-1]
v=a['classification']

import os

# Assuming 'df' is your DataFrame and it already includes a 'Label' column with 0 and 1 values

# Replace the numeric labels with text labels in the DataFrame for plotting
df['classification'] = df['classification'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Directory where plots will be saved
output_dir = 'D:/OBU/ML_2024_2025/Dataset_for_class_Practise/Kernel_Plots'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Filter only numeric columns for plotting
features_to_plot = df.select_dtypes(include=['float64', 'int64']).columns

# Set the visual parameters for the plots
sns.set(font_scale=1.5)
sns.set_style("white")

# Generate individual kernel density plots
for feature in features_to_plot:
    plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.kdeplot(data=df, x=feature, hue="classification", fill=True)
    plt.title(f'Kernel Distribution of {feature.upper()}')
    plt.xlabel(feature.title())
    plt.ylabel('Density')
    
    # Save the plot
    plot_filename = f'Kernel_Plot_{feature}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    
    plt.show()


import os

# Assuming 'df' is your DataFrame and it already includes a 'Label' column with 0 and 1 values

# Replace the numeric labels with text labels in the DataFrame for plotting
df['classification'] = df['classification'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Directory where plots will be saved
output_dir = 'D:/OBU/ML 2024_2025/Dataset_for_class_Practise/Violin Plot'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Now, plot the kernel distributions for each numerical feature, distinguishing by 'classification'
features_to_plot = df.columns[:-1]  # Exclude the 'classification' column

# Set the visual parameters for the plots
sns.set(font_scale=1.5)
sns.set_style("white")

# Generate individual plots
for feature in features_to_plot:
    plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.violinplot(data=df, x=feature, hue="classification", fill=True)
    plt.title(f'Violin Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel(feature.title(), fontsize=30)
    plt.ylabel('Density', fontsize=30)
    
    # Manually setting the legend labels
    legend_labels = df['classification'].unique().tolist()  # Get the unique labels for the legend
    legend_title = 'Classification'
    ax.legend(title=legend_title, labels=legend_labels, loc='upper right', prop={'size': 10})
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Violin_Plot_{feature}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    
    plt.show()


import os
import matplotlib.pyplot as plt
import seaborn as sns

# Replace the numeric labels with text labels in the DataFrame for plotting
df['classification'] = df['classification'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Directory where plots will be saved
output_dir = 'D:/OBU/ML 2024_2025/Dataset_for_class_Practise/Box Plot'
os.makedirs(output_dir, exist_ok=True)

# Set the visual parameters for the plots
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Generate box plots for each numerical feature
features_to_plot = df.columns[:-1]  # Exclude the 'classification' column

for feature in features_to_plot:
    plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.boxplot(data=df, x='classification', y=feature, palette="Set3")
    plt.title(f'Box Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel('Classification', fontsize=30)
    plt.ylabel(feature.title(), fontsize=30)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Box_Plot_{feature}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    
    plt.show()


import os
import matplotlib.pyplot as plt
import seaborn as sns

# Replace the numeric labels with text labels in the DataFrame for plotting
df['classification'] = df['classification'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Directory where plots will be saved
output_dir = 'D:/OBU/ML 2024_2025/Dataset_for_class_Practise/Histogram'
os.makedirs(output_dir, exist_ok=True)

# Set the visual parameters for the plots
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Generate histograms for each numerical feature
features_to_plot = df.columns[:-1]  # Exclude the 'classification' column

for feature in features_to_plot:
    plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.histplot(data=df, x=feature, hue='classification', multiple='stack', palette="Set2", kde=False)
    plt.title(f'Histogram of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel(feature.title(), fontsize=30)
    plt.ylabel('Count', fontsize=30)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Histogram_{feature}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    
    plt.show()


import os
import matplotlib.pyplot as plt
import seaborn as sns

# Replace the numeric labels with text labels in the DataFrame for plotting
df['classification'] = df['classification'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Directory where plots will be saved
output_dir = 'D:/OBU/ML 2024_2025/Dataset_for_class_Practise/Swarm Plot'
os.makedirs(output_dir, exist_ok=True)

# Set the visual parameters for the plots
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Generate swarm plots for each numerical feature
features_to_plot = df.columns[:-1]  # Exclude the 'classification' column

for feature in features_to_plot:
    plt.figure(figsize=(8, 6), dpi=600)
    ax = sns.swarmplot(data=df, x='classification', y=feature, palette="Set2", dodge=True)
    plt.title(f'Swarm Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel('Classification', fontsize=30)
    plt.ylabel(feature.title(), fontsize=30)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Swarm_Plot_{feature}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    
    plt.show()







from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,SVMSMOTE
smote=SMOTE()
# Oversample using SMOTE
sm = SVMSMOTE()
u_resample,y_resample=sm.fit_resample(u,v) 


# Standardize the features
scaler = StandardScaler()
u_scaled = scaler.fit_transform(u_resample)

# Train-test split Method
from sklearn.model_selection import train_test_split

# Map labels in y_resample to 0 and 1
y_resample = y_resample.replace({'ckd': 1, 'not ckd': 0})

x_train, x_test, y_train, y_test = train_test_split(u_scaled, y_resample, test_size=0.3, random_state=0)


# 5-fold Cross-vlidation Split Method
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve
fold=ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
fold1=KFold(5)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Assuming 'x_train', 'x_test', 'y_train', and 'y_test' are already defined from your dataset
# Initialize base classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(probability = True),
    "Gaussian Naive Bayes": GaussianNB(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "AdaBoost": AdaBoostClassifier(),
    "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis()
}


# Parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    },
    
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 10]
    },
    
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "Extra Trees": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2']
    },
    
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    
    "Gaussian Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    "Quadratic Discriminant Analysis": {
        'reg_param': [0.0, 0.01, 0.1, 0.5],
        'tol': [1e-4, 1e-3, 1e-2, 1e-1]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "CatBoost": {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    
    #"Stacking Classifier": {
    #    fix the param grid for the stacking classifier for your ML coursework 1 and 2
   # },
    
    
    #"VotingClassifier": {
       # fix the param grid for the stacking classifier for your ML coursework 1 and 2
    #}
}



import os
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, auc, precision_recall_curve, confusion_matrix, 
                             make_scorer)
from sklearn.model_selection import cross_validate, learning_curve, GridSearchCV, KFold

# Define scoring metrics for cross-validation
scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',  # ROC AUC score
    'specificity': make_scorer(recall_score, pos_label=0),  # Custom scorer for specificity
    'pr_auc': 'average_precision'  # Precision-Recall AUC score
}

# Create folders for saving confusion matrices, ROC-AUC plots, Precision-Recall curves, and learning curves
new_folder = "CKD_GridCVOptimizedML_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
precision_recall_folder = os.path.join(new_folder, "precision_recall_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(precision_recall_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Reset lists before appending results to avoid duplicates
results_train_test = []
results_cv = []


# Loop through classifiers (including Stacking and Voting)
for name, clf in classifiers.items():
    print(f"Optimizing {name}...")

    # Apply GridSearchCV for parameter optimization if parameters are defined
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    else:
        # Use default classifier if no grid search is performed
        best_clf = clf.fit(x_train, y_train)

    # Check if result for this model is already saved, skip if already done
    if name not in [result[0] for result in results_train_test]:
        # Train the classifier
        training_start = time.perf_counter()
        best_clf.fit(x_train, y_train)
        training_end = time.perf_counter()
        train_time = training_end - training_start

        # Make predictions
        if hasattr(best_clf, "predict_proba"):
            y_pred_proba = best_clf.predict_proba(x_test)[:, 1]
            y_pred = best_clf.predict(x_test)
        else:
            y_pred = best_clf.predict(x_test)
            y_pred_proba = None
           
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        test_cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = test_cm.ravel()
        spec = tn / (tn + fp)

        # Calculate ROC-AUC score
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None

        # Calculate Precision-Recall AUC score
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
        else:
            pr_auc = None

        # Store results for the best result
        results_train_test.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{acc * 100:.2f}",   # Accuracy (%)
            f"{prec * 100:.2f}",  # Precision (%)
            f"{rec * 100:.2f}",   # Recall (%)
            f"{f1 * 100:.2f}",    # F1-Score (%)
            f"{spec * 100:.2f}",  # Specificity (%)
            f"{roc_auc * 100:.2f}" if roc_auc is not None else 'N/A',  # ROC-AUC (%)
            f"{pr_auc * 100:.2f}" if pr_auc is not None else 'N/A'     # PR AUC (%)
        ])

        # Cross-Validation Evaluation for the best model
        results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=5, scoring=scorers, return_train_score=False)

        cv_acc_mean = results_cv_scores['test_accuracy'].mean() * 100
        cv_acc_std = results_cv_scores['test_accuracy'].std() * 100

        cv_prec_mean = results_cv_scores['test_precision'].mean() * 100
        cv_prec_std = results_cv_scores['test_precision'].std() * 100

        cv_rec_mean = results_cv_scores['test_recall'].mean() * 100
        cv_rec_std = results_cv_scores['test_recall'].std() * 100

        cv_f1_mean = results_cv_scores['test_f1'].mean() * 100
        cv_f1_std = results_cv_scores['test_f1'].std() * 100

        cv_spec_mean = results_cv_scores['test_specificity'].mean() * 100
        cv_spec_std = results_cv_scores['test_specificity'].std() * 100

        cv_pr_auc_mean = results_cv_scores['test_pr_auc'].mean() * 100
        cv_pr_auc_std = results_cv_scores['test_pr_auc'].std() * 100

        cv_roc_auc_mean = results_cv_scores['test_roc_auc'].mean() * 100
        cv_roc_auc_std = results_cv_scores['test_roc_auc'].std() * 100

        # Ensure only one result per model in the cross-validation results
        if name not in [result[0] for result in results_cv]:
            results_cv.append([
                name,
                f"{train_time:.4f}",  # Time in seconds
                f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
                f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
                f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
                f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
                f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
                f"{cv_pr_auc_mean:.2f} ± {cv_pr_auc_std:.2f}",
                f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"  # Added ROC-AUC in Cross-Validation
            ])
# Define columns for train-test and cross-validation results
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)', 'PR AUC (%)']
columns_cv = ['Model', 'Time (s)', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V ROC-AUC (%)', 'C.V PR AUC (%)']

# Convert results to DataFrame and save to Excel
df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns_cv)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_GridCVOptimizedML_Results.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

   
    
print("Results saved successfully in 'CKD_GridCVOptimizedML_Results' folder.")


from sklearn.ensemble import StackingClassifier, VotingClassifier

# Create folders for saving confusion matrices, ROC-AUC plots, Precision-Recall curves, and learning curves
new_folder = "CKD_GridCVOptimizedML_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
precision_recall_folder = os.path.join(new_folder, "precision_recall_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(precision_recall_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Loop through classifiers (including Stacking and Voting)
for name, clf in classifiers.items():
    print(f"Optimizing {name}...")

    # Apply GridSearchCV for parameter optimization if parameters are defined
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    else:
        # Use default classifier if no grid search is performed
        best_clf = clf.fit(x_train, y_train)

    # Train the classifier
    training_start = time.perf_counter()
    best_clf.fit(x_train, y_train)
    training_end = time.perf_counter()
    train_time = training_end - training_start

    # Make predictions
    if hasattr(best_clf, "predict_proba"):
        y_pred_proba = best_clf.predict_proba(x_test)[:, 1]
        y_pred = best_clf.predict(x_test)
    else:
        y_pred = best_clf.predict(x_test)
        y_pred_proba = None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = test_cm.ravel()
    spec = tn / (tn + fp)

    # Calculate ROC-AUC score
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None

    # Calculate Precision-Recall AUC score
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
    else:
        pr_auc = None

    # Ensure that the results for the best model only are saved
    if name not in [result[0] for result in results_train_test]:
        # Store only the best result for each model
        results_train_test.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{acc * 100:.2f}",   # Accuracy (%)
            f"{prec * 100:.2f}",  # Precision (%)
            f"{rec * 100:.2f}",   # Recall (%)
            f"{f1 * 100:.2f}",    # F1-Score (%)
            f"{spec * 100:.2f}",  # Specificity (%)
            f"{roc_auc * 100:.2f}" if roc_auc is not None else 'N/A',  # ROC-AUC (%)
            f"{pr_auc * 100:.2f}" if pr_auc is not None else 'N/A'     # PR AUC (%)
        ])

    # Cross-Validation Evaluation for the best model
    results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=5, scoring=scorers, return_train_score=False)

    cv_acc_mean = results_cv_scores['test_accuracy'].mean() * 100
    cv_acc_std = results_cv_scores['test_accuracy'].std() * 100

    cv_prec_mean = results_cv_scores['test_precision'].mean() * 100
    cv_prec_std = results_cv_scores['test_precision'].std() * 100

    cv_rec_mean = results_cv_scores['test_recall'].mean() * 100
    cv_rec_std = results_cv_scores['test_recall'].std() * 100
    
    cv_f1_mean = results_cv_scores['test_f1'].mean() * 100
    cv_f1_std = results_cv_scores['test_f1'].std() * 100

    cv_spec_mean = results_cv_scores['test_specificity'].mean() * 100
    cv_spec_std = results_cv_scores['test_specificity'].std() * 100

    cv_pr_auc_mean = results_cv_scores['test_pr_auc'].mean() * 100
    cv_pr_auc_std = results_cv_scores['test_pr_auc'].std() * 100
    
    cv_roc_auc_mean = results_cv_scores['test_roc_auc'].mean() * 100
    cv_roc_auc_std = results_cv_scores['test_roc_auc'].std() * 100
    
    # Ensure only one result per model in the cross-validation results
    if name not in [result[0] for result in results_cv]:
        results_cv.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
            f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
            f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
            f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
            f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
            f"{cv_pr_auc_mean:.2f} ± {cv_pr_auc_std:.2f}",
            f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"  # Added ROC-AUC in Cross-Validation
        ])

    # Generate and save confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.title(f'CKD_Confusion Matrix - {name}', fontsize=25)
    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cm_plot_filepath = os.path.join(confusion_matrix_folder, f'confusion_matrix_{name}.png')
    plt.savefig(cm_plot_filepath)
    plt.close()

    # Plot and save ROC-AUC curve if available
    if roc_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title(f'CKD_ROC_AUC - {name}', fontsize=25)
        plt.legend(loc="lower right")
        roc_auc_filepath = os.path.join(roc_auc_folder, f'roc_auc_{name}.png')
        plt.savefig(roc_auc_filepath)
        plt.close()

    # Plot and save Precision-Recall curve if available
    if pr_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.title(f'CKD_Precision-Recall Curve- {name}', fontsize=25)
        plt.legend(loc="lower left")
        pr_curve_filepath = os.path.join(precision_recall_folder, f'precision_recall_{name}.png')
        plt.savefig(pr_curve_filepath)
        plt.close()
              
    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=KFold(5), n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title(f'CKD_Learning Curve - {name}', fontsize=25)
    plt.xlabel('Training Size', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="best")
    plt.tight_layout()
    learning_curve_filepath = os.path.join(learning_curve_folder, f'learning_curve_{name}.png')
    plt.savefig(learning_curve_filepath)
    plt.close()

# Define columns for train-test and cross-validation results
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)', 'PR AUC (%)']
columns_cv = ['Model', 'Time (s)', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V ROC-AUC (%)', 'C.V PR AUC (%)']

# Convert results to DataFrame and save to Excel
df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns_cv)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_GridCVOptimizedML_Results.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_GridCVOptimizedML_Results' folder.")




from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Actual feature names from the dataset
feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 
                 'pot','hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Initialize PCA to retain components that explain 95% of variance
pca = PCA(n_components=0.95, random_state=0)

# Fit PCA on the scaled data
pca.fit(u_scaled)

# Transform the data to the new PCA space
u_pca = pca.transform(u_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Print out the number of components and explained variance
print(f"PCA reduced to {u_pca.shape[1]} components out of {u_scaled.shape[1]} features to explain 95% variance.")
print("Explained variance per component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Component {i + 1}: {ratio:.4f}")

# Plot cumulative explained variance to determine optimal number of components
plt.figure(figsize=(10, 6), dpi =600)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel("Number of PCA Components", fontsize = 20)
plt.ylabel("Cumulative Explained Variance", fontsize = 20)
plt.title("PCA Explained Variance vs. Number of Components", fontsize = 20)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the component loadings for the top PCA components
# We visualize the contribution of the original features to the first two principal components
plt.figure(figsize=(10, 6), dpi = 600)
components = pca.components_[:2]  # Selecting the first two principal components
for i, component in enumerate(components):
    plt.barh(feature_names, component, alpha=0.7, label=f'PC{i+1}')
plt.xlabel('Feature Contribution to Principal Component', fontsize = 20)
plt.ylabel('Features', fontsize = 20)
plt.title('Feature Contribution to First Two Principal Components', fontsize = 20)
plt.legend()
plt.tight_layout()
plt.show()


# Use u_rfe instead of u_scaled for train-test split
x_train, x_test, y_train, y_test = train_test_split(u_pca, y_resample, test_size=0.3, random_state=0)

# Assuming 'x_train', 'x_test', 'y_train', and 'y_test' are already defined from your dataset
# Initialize base classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "AdaBoost": AdaBoostClassifier(),
    "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis()
}

# Parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Extra Trees": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "CatBoost": {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    #Complete the gridsearch parameter for thee Stacking and Voting classifier for your ML coursework 1 and 2
    
    # Stacking Classifier parameter grid
    #"StackingClassifier": {
       
    #},
    
    # Voting Classifier parameter grid
    #"VotingClassifier": {

   # }    
    
}

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_PCA_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
precision_recall_folder = os.path.join(new_folder, "precision_recall_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(precision_recall_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Store results
results_train_test = []
results_cv = []

# To save the best estimators after grid search
best_classifiers = {}

# Loop through classifiers
for name, clf in classifiers.items():
    print(f"Optimizing {name}...")

    # Apply GridSearchCV for parameter optimization if parameters are defined
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    else:
        # Use default classifier if no grid search is performed
        best_clf = clf.fit(x_train, y_train)
    # Store the best estimator
    best_classifiers[name] = best_clf

    print("GridSearchCV optimization complete for all classifiers.")    
    
    # Check if result for this model is already saved, skip if already done
    if name not in [result[0] for result in results_train_test]:
        # Train the classifier
        training_start = time.perf_counter()
        best_clf.fit(x_train, y_train)
        training_end = time.perf_counter()
        train_time = training_end - training_start        
        
     # Store results including PR AUC
    results_train_test.append([
        name,
        f"{train_time:.4f}",  # Time in seconds
        f"{acc * 100:.2f}",   # Accuracy (%)
        f"{prec * 100:.2f}",  # Precision (%)
        f"{rec * 100:.2f}",   # Recall (%)
        f"{f1 * 100:.2f}",    # F1-Score (%)
        f"{spec * 100:.2f}",  # Specificity (%)
        f"{roc_auc * 100:.2f}" if roc_auc is not None else 'N/A',  # ROC-AUC (%)
        f"{pr_auc * 100:.2f}" if pr_auc is not None else 'N/A'     # PR AUC (%)
    ])

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

    cv_acc_mean = results_cv_scores['test_accuracy'].mean() * 100
    cv_acc_std = results_cv_scores['test_accuracy'].std() * 100

    cv_prec_mean = results_cv_scores['test_precision'].mean() * 100
    cv_prec_std = results_cv_scores['test_precision'].std() * 100

    cv_rec_mean = results_cv_scores['test_recall'].mean() * 100
    cv_rec_std = results_cv_scores['test_recall'].std() * 100
    
    cv_f1_mean = results_cv_scores['test_f1'].mean() * 100
    cv_f1_std = results_cv_scores['test_f1'].std() * 100

    cv_spec_mean = results_cv_scores['test_specificity'].mean() * 100
    cv_spec_std = results_cv_scores['test_specificity'].std() * 100

    cv_pr_auc_mean = results_cv_scores['test_pr_auc'].mean() * 100
    cv_pr_auc_std = results_cv_scores['test_pr_auc'].std() * 100
    
    cv_roc_auc_mean = results_cv_scores['test_roc_auc'].mean() * 100
    cv_roc_auc_std = results_cv_scores['test_roc_auc'].std() * 100    

    # Ensure only one result per model in the cross-validation results
    if name not in [result[0] for result in results_cv]:
        results_cv.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
            f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
            f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
            f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
            f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
            f"{cv_pr_auc_mean:.2f} ± {cv_pr_auc_std:.2f}",
            f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"  # Added ROC-AUC in Cross-Validation
        ])
        

    # Generate and save confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.title(f'CKD_PCA_Confusion Matrix - {name}', fontsize=21)
    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cm_plot_filepath = os.path.join(confusion_matrix_folder, f'confusion_matrix_{name}.png')
    plt.savefig(cm_plot_filepath)
    plt.tight_layout()
    plt.close()

    # Plot and save ROC-AUC curve if available
    if roc_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title(f'CKD_LASSO_ROC_AUC - {name}', fontsize=21)
        plt.legend(loc="lower right")
        roc_auc_filepath = os.path.join(roc_auc_folder, f'roc_auc_{name}.png')
        plt.savefig(roc_auc_filepath)
        plt.tight_layout()
        plt.close()

    # Plot and save Precision-Recall curve if available
    if pr_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.title(f'CKD_PCA_PR Curve- {name}', fontsize=21)
        plt.legend(loc="lower left")
        pr_curve_filepath = os.path.join(precision_recall_folder, f'precision_recall_{name}.png')
        plt.savefig(pr_curve_filepath)
        plt.tight_layout()
        plt.close()
              
    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=KFold(5), n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title(f'CKD_PCA_Learning Curve - {name}', fontsize=21)
    plt.xlabel('Training Size', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="best")
    plt.tight_layout()
    learning_curve_filepath = os.path.join(learning_curve_folder, f'learning_curve_{name}.png')
    plt.savefig(learning_curve_filepath)
    plt.close()        

# Define columns for train-test and cross-validation results
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)', 'PR AUC (%)']
columns_cv = ['Model', 'Time (s)', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)',  'C.V ROC-AUC (%)', 'C.V PR AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns_cv)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_PCA.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_PCA_Results' folder.")





from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# Actual feature names from the dataset
feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 
                 'pot','hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Initialize LDA; the number of components is limited by the number of classes - 1
lda = LDA(n_components=None)

# Fit LDA on the scaled data
u_lda = lda.fit_transform(u_scaled, y_resample)

# Explained variance ratio
explained_variance_ratio = lda.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Print out the number of components and explained variance
print(f"LDA reduced to {u_lda.shape[1]} components out of {u_scaled.shape[1]} features.")
print("Explained variance per component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Component {i + 1}: {ratio:.4f}")

# Plot cumulative explained variance to determine optimal number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel("Number of LDA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("LDA Explained Variance vs. Number of Components")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the feature contributions for the top LDA components
plt.figure(figsize=(12, 8))
components = lda.scalings_[:, :2]  # Selecting the first two LDA components
for i, component in enumerate(components.T):  # Plotting the contributions of the first two LDA components
    plt.barh(feature_names, component, alpha=0.7, label=f'LDA Component {i+1}')
plt.xlabel('Feature Contribution to LDA Component')
plt.ylabel('Features')
plt.title('Feature Contribution to First Two LDA Components')
plt.legend()
plt.tight_layout()
plt.show()


# Use u_rfe instead of u_scaled for train-test split
x_train, x_test, y_train, y_test = train_test_split(u_lda, y_resample, test_size=0.3, random_state=0)

# Assuming 'x_train', 'x_test', 'y_train', and 'y_test' are already defined from your dataset
# Initialize base classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "AdaBoost": AdaBoostClassifier(),
    "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis()
}

# Parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Extra Trees": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "CatBoost": {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    #Complete the gridsearch parameter for thee Stacking and Voting classifier for your ML coursework 1 and 2
    
    # Stacking Classifier parameter grid
    #"StackingClassifier": {
       
    #},
    
    # Voting Classifier parameter grid
    #"VotingClassifier": {

   # }    
    
}

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_LDA_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
precision_recall_folder = os.path.join(new_folder, "precision_recall_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(precision_recall_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Store results
results_train_test = []
results_cv = []

# To save the best estimators after grid search
best_classifiers = {}

# Loop through classifiers
for name, clf in classifiers.items():
    print(f"Optimizing {name}...")

    # Apply GridSearchCV for parameter optimization if parameters are defined
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    else:
        # Use default classifier if no grid search is performed
        best_clf = clf.fit(x_train, y_train)
    # Store the best estimator
    best_classifiers[name] = best_clf

    print("GridSearchCV optimization complete for all classifiers.")    
    
    # Check if result for this model is already saved, skip if already done
    if name not in [result[0] for result in results_train_test]:
        # Train the classifier
        training_start = time.perf_counter()
        best_clf.fit(x_train, y_train)
        training_end = time.perf_counter()
        train_time = training_end - training_start        
        
     # Store results including PR AUC
    results_train_test.append([
        name,
        f"{train_time:.4f}",  # Time in seconds
        f"{acc * 100:.2f}",   # Accuracy (%)
        f"{prec * 100:.2f}",  # Precision (%)
        f"{rec * 100:.2f}",   # Recall (%)
        f"{f1 * 100:.2f}",    # F1-Score (%)
        f"{spec * 100:.2f}",  # Specificity (%)
        f"{roc_auc * 100:.2f}" if roc_auc is not None else 'N/A',  # ROC-AUC (%)
        f"{pr_auc * 100:.2f}" if pr_auc is not None else 'N/A'     # PR AUC (%)
    ])

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

    cv_acc_mean = results_cv_scores['test_accuracy'].mean() * 100
    cv_acc_std = results_cv_scores['test_accuracy'].std() * 100

    cv_prec_mean = results_cv_scores['test_precision'].mean() * 100
    cv_prec_std = results_cv_scores['test_precision'].std() * 100

    cv_rec_mean = results_cv_scores['test_recall'].mean() * 100
    cv_rec_std = results_cv_scores['test_recall'].std() * 100
    
    cv_f1_mean = results_cv_scores['test_f1'].mean() * 100
    cv_f1_std = results_cv_scores['test_f1'].std() * 100

    cv_spec_mean = results_cv_scores['test_specificity'].mean() * 100
    cv_spec_std = results_cv_scores['test_specificity'].std() * 100

    cv_pr_auc_mean = results_cv_scores['test_pr_auc'].mean() * 100
    cv_pr_auc_std = results_cv_scores['test_pr_auc'].std() * 100
    
    cv_roc_auc_mean = results_cv_scores['test_roc_auc'].mean() * 100
    cv_roc_auc_std = results_cv_scores['test_roc_auc'].std() * 100    

    # Ensure only one result per model in the cross-validation results
    if name not in [result[0] for result in results_cv]:
        results_cv.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
            f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
            f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
            f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
            f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
            f"{cv_pr_auc_mean:.2f} ± {cv_pr_auc_std:.2f}",
            f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"  # Added ROC-AUC in Cross-Validation
        ])
        

    # Generate and save confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.title(f'CKD_LDA_Confusion Matrix - {name}', fontsize=21)
    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cm_plot_filepath = os.path.join(confusion_matrix_folder, f'confusion_matrix_{name}.png')
    plt.savefig(cm_plot_filepath)
    plt.tight_layout()
    plt.close()

    # Plot and save ROC-AUC curve if available
    if roc_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title(f'CKD_LDA_ROC_AUC - {name}', fontsize=21)
        plt.legend(loc="lower right")
        roc_auc_filepath = os.path.join(roc_auc_folder, f'roc_auc_{name}.png')
        plt.savefig(roc_auc_filepath)
        plt.tight_layout()
        plt.close()

    # Plot and save Precision-Recall curve if available
    if pr_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.title(f'CKD_LDA_PR Curve- {name}', fontsize=21)
        plt.legend(loc="lower left")
        pr_curve_filepath = os.path.join(precision_recall_folder, f'precision_recall_{name}.png')
        plt.savefig(pr_curve_filepath)
        plt.tight_layout()
        plt.close()
              
    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=KFold(5), n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title(f'CKD_LDA_Learning Curve - {name}', fontsize=21)
    plt.xlabel('Training Size', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="best")
    plt.tight_layout()
    learning_curve_filepath = os.path.join(learning_curve_folder, f'learning_curve_{name}.png')
    plt.savefig(learning_curve_filepath)
    plt.close()        

# Define columns for train-test and cross-validation results
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)', 'PR AUC (%)']
columns_cv = ['Model', 'Time (s)', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)',  'C.V ROC-AUC (%)', 'C.V PR AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns_cv)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_LDA.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_LDA_Results' folder.")



from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import numpy as np

# Actual feature names from the dataset
feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 
                 'pot','hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Step 1: Initialize FA to extract all components
fa_temp = FactorAnalysis(n_components=None, random_state=0)

# Fit the initial FA model to extract all components
u_fa_temp = fa_temp.fit_transform(u_scaled)

# Step 2: Calculate the variance explained by each component
explained_variance = np.var(u_fa_temp, axis=0)
cumulative_variance_ratio = np.cumsum(explained_variance / np.sum(explained_variance))

# Find the number of components needed to explain at least 95% variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

print(f"Optimal number of FA components to explain 95% variance: {n_components_95}")

# Step 3: Reinitialize FA with the optimal number of components
fa = FactorAnalysis(n_components=n_components_95, random_state=0)

# Fit FA on the scaled data with the optimal number of components
u_fa = fa.fit_transform(u_scaled)

# Print out the number of components
print(f"FA reduced to {u_fa.shape[1]} components out of {u_scaled.shape[1]} features.")

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.axvline(n_components_95, color='red', linestyle='--', label=f'{n_components_95} components (95% variance)')
plt.xlabel("Number of FA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("FA Explained Variance vs. Number of Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the factor loadings (importance of features in each FA component)
loadings = fa.components_.T  # Transpose to match components to features

plt.figure(figsize=(12, 8))
for i in range(min(2, n_components_95)):  # Plot only the first 2 FA components
    plt.barh(feature_names, loadings[:, i], alpha=0.7, label=f'FA Component {i + 1}')
plt.xlabel('Feature Contribution to FA Component')
plt.ylabel('Features')
plt.title('Feature Contribution to First Two FA Components')
plt.legend()
plt.tight_layout()
plt.show()


# Use u_rfe instead of u_scaled for train-test split
x_train, x_test, y_train, y_test = train_test_split(u_fa, y_resample, test_size=0.3, random_state=0)

# Assuming 'x_train', 'x_test', 'y_train', and 'y_test' are already defined from your dataset
# Initialize base classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "AdaBoost": AdaBoostClassifier(),
    "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis()
}

# Parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Extra Trees": {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "CatBoost": {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    #Complete the gridsearch parameter for thee Stacking and Voting classifier for your ML coursework 1 and 2
    
    # Stacking Classifier parameter grid
    #"StackingClassifier": {
       
    #},
    
    # Voting Classifier parameter grid
    #"VotingClassifier": {

   # }    
    
}

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_FA_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
precision_recall_folder = os.path.join(new_folder, "precision_recall_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(precision_recall_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Store results
results_train_test = []
results_cv = []

# To save the best estimators after grid search
best_classifiers = {}

# Loop through classifiers
for name, clf in classifiers.items():
    print(f"Optimizing {name}...")

    # Apply GridSearchCV for parameter optimization if parameters are defined
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    else:
        # Use default classifier if no grid search is performed
        best_clf = clf.fit(x_train, y_train)
    # Store the best estimator
    best_classifiers[name] = best_clf

    print("GridSearchCV optimization complete for all classifiers.")    
    
    # Check if result for this model is already saved, skip if already done
    if name not in [result[0] for result in results_train_test]:
        # Train the classifier
        training_start = time.perf_counter()
        best_clf.fit(x_train, y_train)
        training_end = time.perf_counter()
        train_time = training_end - training_start        
        
     # Store results including PR AUC
    results_train_test.append([
        name,
        f"{train_time:.4f}",  # Time in seconds
        f"{acc * 100:.2f}",   # Accuracy (%)
        f"{prec * 100:.2f}",  # Precision (%)
        f"{rec * 100:.2f}",   # Recall (%)
        f"{f1 * 100:.2f}",    # F1-Score (%)
        f"{spec * 100:.2f}",  # Specificity (%)
        f"{roc_auc * 100:.2f}" if roc_auc is not None else 'N/A',  # ROC-AUC (%)
        f"{pr_auc * 100:.2f}" if pr_auc is not None else 'N/A'     # PR AUC (%)
    ])

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

    cv_acc_mean = results_cv_scores['test_accuracy'].mean() * 100
    cv_acc_std = results_cv_scores['test_accuracy'].std() * 100

    cv_prec_mean = results_cv_scores['test_precision'].mean() * 100
    cv_prec_std = results_cv_scores['test_precision'].std() * 100

    cv_rec_mean = results_cv_scores['test_recall'].mean() * 100
    cv_rec_std = results_cv_scores['test_recall'].std() * 100
    
    cv_f1_mean = results_cv_scores['test_f1'].mean() * 100
    cv_f1_std = results_cv_scores['test_f1'].std() * 100

    cv_spec_mean = results_cv_scores['test_specificity'].mean() * 100
    cv_spec_std = results_cv_scores['test_specificity'].std() * 100

    cv_pr_auc_mean = results_cv_scores['test_pr_auc'].mean() * 100
    cv_pr_auc_std = results_cv_scores['test_pr_auc'].std() * 100
    
    cv_roc_auc_mean = results_cv_scores['test_roc_auc'].mean() * 100
    cv_roc_auc_std = results_cv_scores['test_roc_auc'].std() * 100    

    # Ensure only one result per model in the cross-validation results
    if name not in [result[0] for result in results_cv]:
        results_cv.append([
            name,
            f"{train_time:.4f}",  # Time in seconds
            f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
            f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
            f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
            f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
            f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
            f"{cv_pr_auc_mean:.2f} ± {cv_pr_auc_std:.2f}",
            f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"  # Added ROC-AUC in Cross-Validation
        ])
        

    # Generate and save confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.title(f'CKD_FA_Confusion Matrix - {name}', fontsize=21)
    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cm_plot_filepath = os.path.join(confusion_matrix_folder, f'confusion_matrix_{name}.png')
    plt.savefig(cm_plot_filepath)
    plt.tight_layout()
    plt.close()

    # Plot and save ROC-AUC curve if available
    if roc_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title(f'CKD_FA_ROC_AUC - {name}', fontsize=21)
        plt.legend(loc="lower right")
        roc_auc_filepath = os.path.join(roc_auc_folder, f'roc_auc_{name}.png')
        plt.savefig(roc_auc_filepath)
        plt.tight_layout()
        plt.close()

    # Plot and save Precision-Recall curve if available
    if pr_auc is not None:
        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.title(f'CKD_LDA_PR Curve- {name}', fontsize=21)
        plt.legend(loc="lower left")
        pr_curve_filepath = os.path.join(precision_recall_folder, f'precision_recall_{name}.png')
        plt.savefig(pr_curve_filepath)
        plt.tight_layout()
        plt.close()
              
    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=KFold(5), n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title(f'CKD_FA_Learning Curve - {name}', fontsize=21)
    plt.xlabel('Training Size', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="best")
    plt.tight_layout()
    learning_curve_filepath = os.path.join(learning_curve_folder, f'learning_curve_{name}.png')
    plt.savefig(learning_curve_filepath)
    plt.close()        

# Define columns for train-test and cross-validation results
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)', 'PR AUC (%)']
columns_cv = ['Model', 'Time (s)', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)',  'C.V ROC-AUC (%)', 'C.V PR AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns_cv)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_FA.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_FA_Results' folder.")





