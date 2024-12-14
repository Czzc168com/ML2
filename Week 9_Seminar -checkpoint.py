import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_dataset_Processed.csv")

df=df.replace('?',np.nan)

df

df1=df.drop(columns=['id','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','Label'])

df1

from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

lr=LinearRegression()
imp=IterativeImputer(estimator=lr,verbose=2,max_iter=100, tol=1e-10, imputation_order='roman')

df2=imp.fit_transform(df1)

dl1=pd.DataFrame(df2,columns=('age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc'))

dl1.head(20)

df3=df.drop(columns=['id','age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','Label'])

df3

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='constant',fill_value='Missing')
df4=imputer.fit_transform(df3)

dl2=pd.DataFrame(df4,columns=('rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'))

dl2.head()

from sklearn.preprocessing import OrdinalEncoder
ordi=OrdinalEncoder()

dl3=ordi.fit_transform(dl2)

dl4=pd.DataFrame(dl3,columns=('rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'))

dl4

x=pd.concat([dl1,dl4],axis=1)

x

y=df['Label']

a=pd.concat([x,y],axis=1)

a

#a=a.sort_values(['bu'],ascending=False)
#a.reset_index(drop=True,inplace=True)

#a.tail(40)

Label={'notckd':0,'ckd':1}
a['Label']=a['Label'].map(Label)

a

a.head(20)

import pandas as pd
column_names= a.columns[:25]
column_names

u=a.iloc[:,:-1]
v=a['Label']

# Save the DataFrame 'a' as a CSV file in a specific directory
a.to_csv(r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\filename.csv', index=False)


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory where you want to save the plots
save_directory = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\Boxen Plot'

# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load your dataset
df = pd.read_csv('filename.csv')

# Replace the numeric labels with text labels in the DataFrame for plotting
df['Label'] = df['Label'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Select the features for the boxen plots
features_to_plot = df.columns[:-1]  # Exclude the 'Label' column

# Set up Seaborn plot style and font size
sns.set(font_scale=1.5)
sns.set_style("white")

# Generate individual boxen plots for each feature
for feature in features_to_plot:
    plt.figure(figsize=(12, 8), dpi=600)
    
    # Use seaborn boxenplot
    ax = sns.boxenplot(data=df, x='Label', y=feature, palette='Set2')
    
    # Set plot title and labels
    plt.title(f'Boxen Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel(feature.title(), fontsize=30)
    plt.ylabel('Density', fontsize=30)
    
    # Adjust x-axis ticks and labels
    plt.xticks(rotation=45, ha='right', fontsize=30)
    
    # Adjust y-axis ticks and labels
    plt.yticks(fontsize=30)

    # Set margin to ensure that there is no overlap of x-axis values
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Boxen_Plot_{feature}.png'
    plt.savefig(os.path.join(save_directory, plot_filename))
    
    # Display the plot
    plt.show()


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory where you want to save the plots
save_directory = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\Ridge Plot'

# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load your dataset
df = pd.read_csv('filename.csv')

# Replace the numeric labels with text labels in the DataFrame for plotting
df['Label'] = df['Label'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Set up Seaborn plot style and font size
sns.set(font_scale=1.5)
sns.set_style("white")

# Generate ridge plots for each feature
for feature in df.columns[:-1]:  # Exclude the 'Label' column
    plt.figure(figsize=(12, 8), dpi=600)
    
    # Use seaborn kdeplot with hue to create ridge plot effect
    ax = sns.kdeplot(data=df, x=feature, hue='Label', fill=True, palette='Set2', alpha=0.7)
    
    # Set plot title and labels
    plt.title(f'Ridge Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel(feature.title(), fontsize=30)
    plt.ylabel('Density', fontsize=30)
    
    # Adjust x-axis ticks and labels
    plt.xticks(rotation=45, ha='right', fontsize=30)
    
    # Adjust y-axis ticks and labels
    plt.yticks(fontsize=30)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Ridge_Plot_{feature}.png'
    plt.savefig(os.path.join(save_directory, plot_filename))
    
    # Display the plot
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('filename.csv')

# Replace the numeric labels with text labels in the DataFrame for plotting
df['Label'] = df['Label'].replace({0: 'Not CKD', 1: 'CKD'}).astype(str)

# Select the features for the violin plots
features_to_plot = df.columns[:-1]  # Exclude the 'Label' column

# Set up Seaborn plot style and font size
sns.set(font_scale=1.5)
sns.set_style("white")

# Generate individual violin plots for each feature
for feature in features_to_plot:
    plt.figure(figsize=(12, 8), dpi=600)
    
    # Use seaborn violinplot with a specified margin for x-axis
    ax = sns.violinplot(data=df, x='Label', y=feature, palette='Set2', fill=True)
    
    # Set plot title and labels
    plt.title(f'Violin Plot of {feature.upper()} Distribution', fontsize=30)
    plt.xlabel(feature.title(), fontsize=30)
    plt.ylabel('Density', fontsize=30)
    
    # Adjust x-axis ticks and labels
    plt.xticks(rotation=45, ha='right', fontsize=30)
    
    # Adjust y-axis ticks and labels
    plt.yticks(fontsize=30)

    # Set margin to ensure that there is no overlap of x-axis values
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.tight_layout()
    plot_filename = f'Violin_Plot_{feature}.png'
    plt.savefig(f'D:/Conference 2024/Grace_Theo/CKD_IntegratedFS_Paper4/Violin Plot/{plot_filename}')
    
    # Display the plot
    plt.show()


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
smote=SMOTE()
# Oversample using SMOTE
sm = SVMSMOTE()
u_resample,y_resample=sm.fit_resample(u,v) 

# Standardize the features
scaler = StandardScaler()
u_scaled = scaler.fit_transform(u_resample)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, make_scorer
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_validate, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Assuming 'u_scaled' and 'y_resample' are already defined

# Train-test split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(u_scaled, y_resample, test_size=0.2, random_state=0)
fold = 10

# Define parameter grids
param_grids = {
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 4, 6]
    },
    'qda': {
        'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0],
        'tol': [1e-4, 1e-3, 1e-2],
        'store_covariance': [True, False]
    },
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 6, 10]
    },
    'logreg': {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    },
    'gnb': {},
    'etc': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2']
    },
    'gdb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'lgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'catboost': {
        'iterations': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 8]
    },
}

# Define classifiers
classifiers = {
    'svm': SVC(probability=True, random_state=0),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=0),
    'qda': QuadraticDiscriminantAnalysis(),
    'tree': DecisionTreeClassifier(random_state=0),
    'logreg': LogisticRegression(random_state=0),
    'gnb': GaussianNB(),
    'etc': ExtraTreesClassifier(random_state=0),
    'gdb': GradientBoostingClassifier(random_state=0),
    'lgb': LGBMClassifier(random_state=0),
    'xgb': XGBClassifier(random_state=0),
    'catboost': CatBoostClassifier(verbose=0, random_state=0),
    'ada': AdaBoostClassifier(random_state=0),
    'bag': BaggingClassifier(random_state=0),
    'stack': StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0))
    ], final_estimator=LogisticRegression()),
    'voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0)),
        ('logreg', LogisticRegression(random_state=0))
    ], voting='soft')
}

# Define custom scorer for specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Store results
results_train_test = []
results_cv = []

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_NoFS_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Loop through classifiers and perform GridSearchCV
for name, clf in classifiers.items():
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=fold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
    else:
        best_clf = clf
    
    # Train-Test Split Evaluation
    training_start = time.perf_counter()
    best_clf.fit(x_train, y_train)
    training_end = time.perf_counter()
    train_time = training_end - training_start
    
    # Make predictions
    if hasattr(best_clf, "predict_proba"):
        y_pred = best_clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = best_clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = test_cm.ravel()
    spec = tn / (tn + fp)
    
    # Calculate ROC-AUC score
    fpr, tpr, thresholds = roc_curve(y_test, best_clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Store all results
    results_train_test.append([name, train_time, acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, roc_auc * 100])

    # Generate confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix_CKD - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_plot_filename = f'confusion_matrix_CKD_{name}.png'
    cm_plot_filepath = os.path.join(confusion_matrix_folder, cm_plot_filename)
    plt.savefig(cm_plot_filepath)
    plt.close()

    # Plot ROC-AUC curve
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC_AUC_CKD - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save roc_auc plot
    roc_auc_filename = f'roc_auc_CKD-{name}.png'
    roc_auc_filepath = os.path.join(roc_auc_folder, roc_auc_filename)
    plt.savefig(roc_auc_filepath)
    plt.close()

    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=fold, n_jobs=-1,
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

    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save learning curve plot
    learning_curve_filename = f'learning_curve_{name}.png'
    learning_curve_filepath = os.path.join(learning_curve_folder, learning_curve_filename)
    plt.savefig(learning_curve_filepath)
    plt.close()

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

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

    cv_roc_auc_mean = results_cv_scores['test_specificity'].mean() * 100  # Assuming AUC is calculated separately
    cv_roc_auc_std = results_cv_scores['test_specificity'].std() * 100

    results_cv.append([
        name,
        f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
        f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
        f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
        f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
        f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
        f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"
    ])

# Convert results to DataFrame and save to Excel
columns1 = ['Model', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V AUC (%)']

# Convert results to DataFrame and save to Excel
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns1)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_Model_Performance.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_Model_Results' folder.")


# Applying Factor Analysis (FA)
# Assuming 'u' is your feature matrix

# Step 1: Perform Factor Analysis and extract eigenvalues
fa = FactorAnalysis(n_components=10, random_state=0)
fa.fit(u)
eigenvalues = fa.noise_variance_ + fa.get_covariance().diagonal()

# Step 2: Apply Kaiser's criterion to determine the number of factors
kaiser_threshold = 1  # Adjust the threshold as needed
num_factors = np.sum(eigenvalues > kaiser_threshold)

# Step 3: Use the determined number of factors in Factor Analysis
fa = FactorAnalysis(n_components=num_factors, random_state=0)
u_fa = fa.fit_transform(u_scaled)
fa_feature_importance = np.abs(fa.components_)

# Choose the desired percentage of variance to retain (e.g., 95%) for PCA
desired_variance_ratio = 0.95 #provide 18 components

pca = PCA(n_components=desired_variance_ratio, random_state=0)
u_pca = pca.fit_transform(u_scaled)
pca_feature_importance = np.abs(pca.components_)



import matplotlib.pyplot as plt
import numpy as np
import os

# Feature names (excluding the class label)
feature_names = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                 'pcv', 'wbcc', 'rbcc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                 'appet', 'pe', 'ane']

# Calculate the sum of absolute values of coefficients for each feature across all PCA components
pca_importance_sum = np.sum(np.abs(pca.components_), axis=0)

# Similarly for FA
fa_importance_sum = np.sum(np.abs(fa.components_), axis=0)


# Get indices of top features
top_pca_indices = np.argsort(pca_importance_sum)[-18:]  # Top 18 features for PCA
top_fa_indices = np.argsort(fa_importance_sum)[-16:]    # Top 16 features for FA

# Convert indices to names
top_pca_features = [feature_names[i] for i in top_pca_indices]
top_fa_features = [feature_names[i] for i in top_fa_indices]


# Create a directory to save plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_IntegratedFS_Paper4\CKD_Model_Results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot and save PCA Features Importance
plt.figure(figsize=(8, 6), dpi=600)
plt.bar(top_pca_features, pca_importance_sum[top_pca_indices], color='#8B4513')  # professional red color
plt.title('Top 18 PCA Features Importance', fontsize=20)
plt.xlabel('PCA Features', fontsize=20)
plt.ylabel('Sum of Absolute Coefficients', fontsize=20)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save PCA plot before showing it
pca_plot_filename = 'PCA_Features_Importance.png'
pca_plot_filepath = os.path.join(save_dir, pca_plot_filename)
plt.savefig(pca_plot_filepath, bbox_inches='tight')
print(f'PCA plot saved to {pca_plot_filepath}')

plt.show()

# Plot and save FA Features Importance
plt.figure(figsize=(8, 6), dpi=600)
plt.bar(top_fa_features, fa_importance_sum[top_fa_indices], color='#8B4513')
plt.title('Top 16 FA Features Importance', fontsize=20)
plt.xlabel('FA Features', fontsize=20)
plt.ylabel('Sum of Absolute Loadings', fontsize=20)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save FA plot before showing it
fa_plot_filename = 'FA_Features_Importance.png'
fa_plot_filepath = os.path.join(save_dir, fa_plot_filename)
plt.savefig(fa_plot_filepath, bbox_inches='tight')
print(f'FA plot saved to {fa_plot_filepath}')

plt.show()


# Print the top features in order of priority for PCA, FA, and LDA
print("Top PCA Features in order of importance:")
sorted_pca_features = [feature_names[i] for i in np.argsort(-pca_importance_sum)[:18]]
print(sorted_pca_features)

print("\nTop FA Features in order of importance:")
sorted_fa_features = [feature_names[i] for i in np.argsort(-fa_importance_sum)[:16]]
print(sorted_fa_features)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, make_scorer
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_validate, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Assuming 'u_scaled' and 'y_resample' are already defined

# Train-test split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(u_pca, y_resample, test_size=0.2, random_state=0)
fold = 10

# Define parameter grids
param_grids = {
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 4, 6]
    },
    'qda': {
        'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0],
        'tol': [1e-4, 1e-3, 1e-2],
        'store_covariance': [True, False]
    },
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 6, 10]
    },
    'logreg': {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    },
    'gnb': {},
    'etc': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2']
    },
    'gdb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'lgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'catboost': {
        'iterations': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 8]
    },
}

# Define classifiers
classifiers = {
    'svm': SVC(probability=True, random_state=0),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=0),
    'qda': QuadraticDiscriminantAnalysis(),
    'tree': DecisionTreeClassifier(random_state=0),
    'logreg': LogisticRegression(random_state=0),
    'gnb': GaussianNB(),
    'etc': ExtraTreesClassifier(random_state=0),
    'gdb': GradientBoostingClassifier(random_state=0),
    'lgb': LGBMClassifier(random_state=0),
    'xgb': XGBClassifier(random_state=0),
    'catboost': CatBoostClassifier(verbose=0, random_state=0),
    'ada': AdaBoostClassifier(random_state=0),
    'bag': BaggingClassifier(random_state=0),
    'stack': StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0))
    ], final_estimator=LogisticRegression()),
    'voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0)),
        ('logreg', LogisticRegression(random_state=0))
    ], voting='soft')
}

# Define custom scorer for specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Store results
results_train_test = []
results_cv = []

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_PCAFS_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Loop through classifiers and perform GridSearchCV
for name, clf in classifiers.items():
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=fold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
    else:
        best_clf = clf
    
    # Train-Test Split Evaluation
    training_start = time.perf_counter()
    best_clf.fit(x_train, y_train)
    training_end = time.perf_counter()
    train_time = training_end - training_start
    
    # Make predictions
    if hasattr(best_clf, "predict_proba"):
        y_pred = best_clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = best_clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = test_cm.ravel()
    spec = tn / (tn + fp)
    
    # Calculate ROC-AUC score
    fpr, tpr, thresholds = roc_curve(y_test, best_clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Store all results
    results_train_test.append([name, train_time, acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, roc_auc * 100])

    # Generate confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix_CKD - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_plot_filename = f'confusion_matrix_CKD_{name}.png'
    cm_plot_filepath = os.path.join(confusion_matrix_folder, cm_plot_filename)
    plt.savefig(cm_plot_filepath)
    plt.close()

    # Plot ROC-AUC curve
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic Curve_CKD - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save roc_auc plot
    roc_auc_filename = f'roc_auc_CKD-{name}.png'
    roc_auc_filepath = os.path.join(roc_auc_folder, roc_auc_filename)
    plt.savefig(roc_auc_filepath)
    plt.close()

    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=fold, n_jobs=-1,
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

    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save learning curve plot
    learning_curve_filename = f'learning_curve_{name}.png'
    learning_curve_filepath = os.path.join(learning_curve_folder, learning_curve_filename)
    plt.savefig(learning_curve_filepath)
    plt.close()

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

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

    cv_roc_auc_mean = results_cv_scores['test_specificity'].mean() * 100  # Assuming AUC is calculated separately
    cv_roc_auc_std = results_cv_scores['test_specificity'].std() * 100

    results_cv.append([
        name,
        f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
        f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
        f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
        f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
        f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
        f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"
    ])

# Convert results to DataFrame and save to Excel
columns1 = ['Model', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V AUC (%)']

# Convert results to DataFrame and save to Excel
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns1)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_PCA_Performance.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_PCAFS_Results' folder.")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, make_scorer
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_validate, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Assuming 'u_scaled' and 'y_resample' are already defined

# Train-test split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(u_fa, y_resample, test_size=0.2, random_state=0)
fold = 10

# Define parameter grids
param_grids = {
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 4, 6]
    },
    'qda': {
        'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0],
        'tol': [1e-4, 1e-3, 1e-2],
        'store_covariance': [True, False]
    },
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 6, 10]
    },
    'logreg': {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    },
    'gnb': {},
    'etc': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2']
    },
    'gdb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'lgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'catboost': {
        'iterations': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 8]
    },
}

# Define classifiers
classifiers = {
    'svm': SVC(probability=True, random_state=0),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=0),
    'qda': QuadraticDiscriminantAnalysis(),
    'tree': DecisionTreeClassifier(random_state=0),
    'logreg': LogisticRegression(random_state=0),
    'gnb': GaussianNB(),
    'etc': ExtraTreesClassifier(random_state=0),
    'gdb': GradientBoostingClassifier(random_state=0),
    'lgb': LGBMClassifier(random_state=0),
    'xgb': XGBClassifier(random_state=0),
    'catboost': CatBoostClassifier(verbose=0, random_state=0),
    'ada': AdaBoostClassifier(random_state=0),
    'bag': BaggingClassifier(random_state=0),
    'stack': StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0))
    ], final_estimator=LogisticRegression()),
    'voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0)),
        ('logreg', LogisticRegression(random_state=0))
    ], voting='soft')
}

# Define custom scorer for specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Store results
results_train_test = []
results_cv = []

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_FAFS_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Loop through classifiers and perform GridSearchCV
for name, clf in classifiers.items():
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=fold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
    else:
        best_clf = clf
    
    # Train-Test Split Evaluation
    training_start = time.perf_counter()
    best_clf.fit(x_train, y_train)
    training_end = time.perf_counter()
    train_time = training_end - training_start
    
    # Make predictions
    if hasattr(best_clf, "predict_proba"):
        y_pred = best_clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = best_clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = test_cm.ravel()
    spec = tn / (tn + fp)
    
    # Calculate ROC-AUC score
    fpr, tpr, thresholds = roc_curve(y_test, best_clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Store all results
    results_train_test.append([name, train_time, acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, roc_auc * 100])

    # Generate confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix_CKD - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_plot_filename = f'confusion_matrix_CKD_{name}.png'
    cm_plot_filepath = os.path.join(confusion_matrix_folder, cm_plot_filename)
    plt.savefig(cm_plot_filepath)
    plt.close()

    # Plot ROC-AUC curve
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic Curve_CKD - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save roc_auc plot
    roc_auc_filename = f'roc_auc_CKD-{name}.png'
    roc_auc_filepath = os.path.join(roc_auc_folder, roc_auc_filename)
    plt.savefig(roc_auc_filepath)
    plt.close()

    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=fold, n_jobs=-1,
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

    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save learning curve plot
    learning_curve_filename = f'learning_curve_{name}.png'
    learning_curve_filepath = os.path.join(learning_curve_folder, learning_curve_filename)
    plt.savefig(learning_curve_filepath)
    plt.close()

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

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

    cv_roc_auc_mean = results_cv_scores['test_specificity'].mean() * 100  # Assuming AUC is calculated separately
    cv_roc_auc_std = results_cv_scores['test_specificity'].std() * 100

    results_cv.append([
        name,
        f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
        f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
        f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
        f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
        f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
        f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"
    ])

# Convert results to DataFrame and save to Excel
columns1 = ['Model', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V AUC (%)']

# Convert results to DataFrame and save to Excel
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns1)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_FA_Performance.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_FAFS_Results' folder.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Assuming the selected features from PCA and FA are available in `u_pca` and `u_fa`
# and `top_pca_indices` and `top_fa_indices` represent the indices of the selected features

# Feature names (excluding the class label)
feature_names = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                 'pcv', 'wbcc', 'rbcc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                 'appet', 'pe', 'ane']

# Convert the transformed PCA features to a DataFrame
df_pca = pd.DataFrame(u_pca, columns=[f'HFCA_{feature_names[i]}' for i in top_pca_indices])

# Convert the transformed FA features to a DataFrame
df_fa = pd.DataFrame(u_fa, columns=[f'HFCA_{feature_names[i]}' for i in top_fa_indices])

# Concatenate the PCA and FA DataFrames along the column axis
hfca_df = pd.concat([df_pca, df_fa], axis=1)

# Remove duplicate columns (keeping the first occurrence)
hfca_df = hfca_df.loc[:, ~hfca_df.columns.duplicated()]

# Sum of absolute loadings for each feature in the PFA DataFrame
feature_importance = np.abs(hfca_df).sum(axis=0)

# Sort the features based on their importance
sorted_features = feature_importance.sort_values(ascending=False)

# Select the top 21 features for plotting
top_features = sorted_features.head(21)

# Plotting the top 21 features
plt.figure(figsize=(12, 8))
plt.bar(top_features.index, top_features.values, color='red')
plt.title('Top 21 HFCA Features Importance')
plt.xlabel('HFCA Features')
plt.ylabel('Sum of Absolute Loadings')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the plot
plt.savefig('top_21_HFCA_features_importance.png')
plt.show()

# Calculate the total number of features integrated
total_features = hfca_df.shape[1]

# Print the total number of features
print(f'Total number of unique features in PCA + FA: {total_features}')

# Print the new DataFrame with the integrated features
print(hfca_df.head())

# Optionally, save the PCA + FA features DataFrame to a CSV file
hfca_df.to_csv('HFCA_features.csv', index=False)

# Perform a train-test split with the new DataFrame
x_train, x_test, y_train, y_test = train_test_split(hfca_df, y_resample, test_size=0.3, random_state=0)

# Print the shapes of the train and test sets
print(f'Train set shape: {x_train.shape}')
print(f'Test set shape: {x_test.shape}')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, make_scorer
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_validate, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Assuming 'u_scaled' and 'y_resample' are already defined

# Train-test split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(hfca_df, y_resample, test_size=0.2, random_state=0)
fold = 10

# Define parameter grids
param_grids = {
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 4, 6]
    },
    'qda': {
        'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0],
        'tol': [1e-4, 1e-3, 1e-2],
        'store_covariance': [True, False]
    },
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 6, 10]
    },
    'logreg': {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    },
    'gnb': {},
    'etc': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2']
    },
    'gdb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'lgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgb': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8]
    },
    'catboost': {
        'iterations': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 8]
    },
}

# Define classifiers
classifiers = {
    'svm': SVC(probability=True, random_state=0),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=0),
    'qda': QuadraticDiscriminantAnalysis(),
    'tree': DecisionTreeClassifier(random_state=0),
    'logreg': LogisticRegression(random_state=0),
    'gnb': GaussianNB(),
    'etc': ExtraTreesClassifier(random_state=0),
    'gdb': GradientBoostingClassifier(random_state=0),
    'lgb': LGBMClassifier(random_state=0),
    'xgb': XGBClassifier(random_state=0),
    'catboost': CatBoostClassifier(verbose=0, random_state=0),
    'ada': AdaBoostClassifier(random_state=0),
    'bag': BaggingClassifier(random_state=0),
    'stack': StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0))
    ], final_estimator=LogisticRegression()),
    'voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=0)),
        ('svc', SVC(probability=True, random_state=0)),
        ('logreg', LogisticRegression(random_state=0))
    ], voting='soft')
}

# Define custom scorer for specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

scorers = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Store results
results_train_test = []
results_cv = []

# Create folders for saving confusion matrices, ROC-AUC plots, and learning curves
new_folder = "CKD_HFCA_Results"
confusion_matrix_folder = os.path.join(new_folder, "confusion_matrices")
roc_auc_folder = os.path.join(new_folder, "roc_auc_curves")
learning_curve_folder = os.path.join(new_folder, "learning_curves")
os.makedirs(confusion_matrix_folder, exist_ok=True)
os.makedirs(roc_auc_folder, exist_ok=True)
os.makedirs(learning_curve_folder, exist_ok=True)

# Loop through classifiers and perform GridSearchCV
for name, clf in classifiers.items():
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=fold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
    else:
        best_clf = clf
    
    # Train-Test Split Evaluation
    training_start = time.perf_counter()
    best_clf.fit(x_train, y_train)
    training_end = time.perf_counter()
    train_time = training_end - training_start
    
    # Make predictions
    if hasattr(best_clf, "predict_proba"):
        y_pred = best_clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = best_clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = test_cm.ravel()
    spec = tn / (tn + fp)
    
    # Calculate ROC-AUC score
    fpr, tpr, thresholds = roc_curve(y_test, best_clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Store all results
    results_train_test.append([name, train_time, acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, roc_auc * 100])

    # Generate confusion matrix
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix_CKD - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_plot_filename = f'confusion_matrix_CKD_{name}.png'
    cm_plot_filepath = os.path.join(confusion_matrix_folder, cm_plot_filename)
    plt.savefig(cm_plot_filepath)
    plt.close()

    # Plot ROC-AUC curve
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic Curve_CKD - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save roc_auc plot
    roc_auc_filename = f'roc_auc_CKD-{name}.png'
    roc_auc_filepath = os.path.join(roc_auc_folder, roc_auc_filename)
    plt.savefig(roc_auc_filepath)
    plt.close()

    # Plot and save learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, cv=fold, n_jobs=-1,
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

    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save learning curve plot
    learning_curve_filename = f'learning_curve_{name}.png'
    learning_curve_filepath = os.path.join(learning_curve_folder, learning_curve_filename)
    plt.savefig(learning_curve_filepath)
    plt.close()

    # Cross-Validation Evaluation
    results_cv_scores = cross_validate(best_clf, x_train, y_train, cv=fold, scoring=scorers, return_train_score=False)

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

    cv_roc_auc_mean = results_cv_scores['test_specificity'].mean() * 100  # Assuming AUC is calculated separately
    cv_roc_auc_std = results_cv_scores['test_specificity'].std() * 100

    results_cv.append([
        name,
        f"{cv_acc_mean:.2f} ± {cv_acc_std:.2f}",
        f"{cv_prec_mean:.2f} ± {cv_prec_std:.2f}",
        f"{cv_rec_mean:.2f} ± {cv_rec_std:.2f}",
        f"{cv_f1_mean:.2f} ± {cv_f1_std:.2f}",
        f"{cv_spec_mean:.2f} ± {cv_spec_std:.2f}",
        f"{cv_roc_auc_mean:.2f} ± {cv_roc_auc_std:.2f}"
    ])

# Convert results to DataFrame and save to Excel
columns1 = ['Model', 'C.V Acc (%)', 'C.V Precision (%)', 'C.V Recall (%)', 'C.V F1-Score (%)', 'C.V Specificity (%)', 'C.V AUC (%)']

# Convert results to DataFrame and save to Excel
columns = ['Model', 'Time (s)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'ROC-AUC (%)']

df_train_test = pd.DataFrame(results_train_test, columns=columns)
df_cv = pd.DataFrame(results_cv, columns=columns1)

with pd.ExcelWriter(os.path.join(new_folder, 'CKD_HFCA_Performance.xlsx')) as writer:
    df_train_test.to_excel(writer, sheet_name='Train-Test Results', index=False)
    df_cv.to_excel(writer, sheet_name='Cross-Validation Results', index=False)

print("Results saved successfully in 'CKD_HFCA_Results' folder.")


import shap
import matplotlib.pyplot as plt
import os
import copy

# Extract feature names from integrated_df
feature_names = hfca_df.columns.tolist()

# Map the original class labels to new labels
class_mapping = {0: 'Not CKD', 1: 'CKD'}
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# List to store SHAP values for each model
shap_values_list = []

# Create a directory to save SHAP plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_HFCA_Results\Summary_Plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through classifiers and generate SHAP dot plots
for label, clf in classifiers.items():
    try:
        # Fit the classifier
        clf.fit(x_train, y_train)
        
        # For faster SHAP values computation, use KernelExplainer on a subset of the training data
        background_summary = shap.sample(x_train, 1)
        
        # Handle potential issues with KernelExplainer
        model_copy = copy.deepcopy(clf)
        
        if hasattr(clf, "predict_proba"):
            explainer = shap.KernelExplainer(model_copy.predict_proba, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model_copy.predict, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        
        # Plot the SHAP summary plot using the feature names from integrated_df
        plt.figure(figsize=(10, 8), dpi=600)
        
        # Plot SHAP values with automatic legend
        shap.summary_plot(shap_values_model, x_test, feature_names=feature_names, plot_type="bar", show=False)
        
        # Customize the legend
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles, ['Not CKD', 'CKD'], loc='lower right', fontsize=22)
        
        # Increase font sizes for x-ticks, y-ticks, and title
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.title(f'SHAP Summary Plot - {label}', fontsize=25)
        
        # Save the SHAP dot plot before showing it
        SHAP_plot_filename = f'SHAP_{label}_summary.png'
        SHAP_plot_filepath = os.path.join(save_dir, SHAP_plot_filename)
        plt.savefig(SHAP_plot_filepath, bbox_inches='tight')
        
        print(f'SHAP dot plot saved to {SHAP_plot_filepath}')
        
        # Show the plot after saving
        plt.show()
    
        # Append SHAP values for the current model to the list
        shap_values_list.append((label, shap_values_model))
    
    except Exception as e:
        print(f"Error processing SHAP for {label}: {str(e)}")


import shap
import matplotlib.pyplot as plt
import os
import copy

# Extract feature names from integrated_df
feature_names = hfca_df.columns.tolist()

# Map the original class labels to new labels
class_mapping = {0: 'Not CKD', 1: 'CKD'}
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# List to store SHAP values for each model
shap_values_list = []

# Create a directory to save SHAP plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_HFCA_Results\Dot_Plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through classifiers and generate SHAP dot plots
for label, clf in classifiers.items():
    try:
        # Fit the classifier
        clf.fit(x_train, y_train)
        
        # For faster SHAP values computation, use KernelExplainer on a subset of the training data
        background_summary = shap.sample(x_train, 1)
        
        # Handle potential issues with KernelExplainer
        model_copy = copy.deepcopy(clf)
        
        if hasattr(clf, "predict_proba"):
            explainer = shap.KernelExplainer(model_copy.predict_proba, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model_copy.predict, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        
        # Check if shap_values_model is a list (indicating multi-output)
        if isinstance(shap_values_model, list):
            # Select the SHAP values for the positive class (CKD)
            shap_values_model = shap_values_model[1]
        
        # Plot the SHAP dot plot using the feature names from integrated_df
        plt.figure(figsize=(10, 8), dpi=600)
        
        # Plot SHAP values as a dot plot
        shap.summary_plot(shap_values_model, x_test, feature_names=feature_names, plot_type="dot", show=False)
        
        # Increase font sizes for x-ticks, y-ticks, and title
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.title(f'SHAP Dot Plot - {label}', fontsize=25)
        
        # Save the SHAP dot plot before showing it
        SHAP_plot_filename = f'SHAP_{label}_dot_plot.png'
        SHAP_plot_filepath = os.path.join(save_dir, SHAP_plot_filename)
        plt.savefig(SHAP_plot_filepath, bbox_inches='tight')
        
        print(f'SHAP dot plot saved to {SHAP_plot_filepath}')
        
        # Show the plot after saving
        plt.show()
    
        # Append SHAP values for the current model to the list
        shap_values_list.append((label, shap_values_model))
    
    except Exception as e:
        print(f"Error processing SHAP for {label}: {str(e)}")




import shap
import matplotlib.pyplot as plt
import os
import copy

# Extract feature names from integrated_df
feature_names = hfca_df.columns.tolist()

# Map the original class labels to new labels
class_mapping = {0: 'Not CKD', 1: 'CKD'}
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# List to store SHAP values for each model
shap_values_list = []

# Create a directory to save SHAP plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_HFCA_Results\Dependency_Plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through classifiers and generate SHAP dependency plots
for label, clf in classifiers.items():
    try:
        # Fit the classifier
        clf.fit(x_train, y_train)
        
        # For faster SHAP values computation, use KernelExplainer on a subset of the training data
        background_summary = shap.sample(x_train, 1)
        
        # Handle potential issues with KernelExplainer
        model_copy = copy.deepcopy(clf)
        
        if hasattr(clf, "predict_proba"):
            explainer = shap.KernelExplainer(model_copy.predict_proba, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model_copy.predict, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        
        # Check if shap_values_model is a list (indicating multi-output)
        if isinstance(shap_values_model, list):
            # Select the SHAP values for the positive class (CKD)
            shap_values_model = shap_values_model[1]

        # Generate dependency plots for each feature
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(10, 8), dpi=600)
            shap.dependence_plot(
                feature, shap_values_model, x_test, interaction_index='auto', show=False
            )
            
            # Increase font sizes for title
            plt.title(f'SHAP Dependency Plot - {label} - {feature}', fontsize=25)
            
            # Save the SHAP dependency plot before showing it
            SHAP_plot_filename = f'SHAP_{label}_dependency_{feature}.png'
            SHAP_plot_filepath = os.path.join(save_dir, SHAP_plot_filename)
            plt.savefig(SHAP_plot_filepath, bbox_inches='tight')
            
            print(f'SHAP dependency plot saved to {SHAP_plot_filepath}')
            
            # Show the plot after saving
            plt.show()

        # Append SHAP values for the current model to the list
        shap_values_list.append((label, shap_values_model))
    
    except Exception as e:
        print(f"Error processing SHAP for {label}: {str(e)}")




import shap
import matplotlib.pyplot as plt
import os
import copy

# Extract feature names from integrated_df
feature_names = hfca_df.columns.tolist()

# Map the original class labels to new labels
class_mapping = {0: 'Not CKD', 1: 'CKD'}
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# Create a directory to save SHAP plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_HFCA_Results\Force_Plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through classifiers and generate SHAP force plots
for label, clf in classifiers.items():
    try:
        # Fit the classifier
        clf.fit(x_train, y_train)
        
        # For faster SHAP values computation, use KernelExplainer on a subset of the training data
        background_summary = shap.sample(x_train, 1)
        
        # Handle potential issues with KernelExplainer
        model_copy = copy.deepcopy(clf)
        
        if hasattr(clf, "predict_proba"):
            explainer = shap.KernelExplainer(model_copy.predict_proba, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model_copy.predict, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        
        # Check if shap_values_model is a list (indicating multi-output)
        if isinstance(shap_values_model, list):
            # Select the SHAP values for the positive class (CKD)
            shap_values_model = shap_values_model[1]
        
        # Generate force plot for each instance in the test set
        for i in range(x_test.shape[0]):
            plt.figure(figsize=(20, 3))
            
            # Create force plot
            shap.force_plot(
                explainer.expected_value[1], shap_values_model[i], x_test.iloc[i],
                feature_names=feature_names, matplotlib=True, show=False
            )
            
            # Save the force plot before showing it
            SHAP_plot_filename = f'SHAP_{label}_force_plot_{i}.png'
            SHAP_plot_filepath = os.path.join(save_dir, SHAP_plot_filename)
            plt.savefig(SHAP_plot_filepath, bbox_inches='tight')
            
            print(f'SHAP force plot saved to {SHAP_plot_filepath}')
            
            # Close the plot to avoid memory issues
            plt.close()

    except Exception as e:
        print(f"Error processing SHAP for {label}: {str(e)}")




import shap
import matplotlib.pyplot as plt
import os
import copy
import numpy as np

# Extract feature names from integrated_df
feature_names = hfca_df.columns.tolist()

# Map the original class labels to new labels
class_mapping = {0: 'Not CKD', 1: 'CKD'}
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# Create a directory to save SHAP plots
save_dir = r'D:\Conference 2024\Grace_Theo\CKD_PCA+FA_Paper3\CKD_HFCA_Results\Waterfall_Plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through classifiers and generate SHAP waterfall plots
for label, clf in classifiers.items():
    try:
        # Fit the classifier
        clf.fit(x_train, y_train)
        
        # For faster SHAP values computation, use KernelExplainer on a subset of the training data
        background_summary = shap.sample(x_train, 1)
        
        # Handle potential issues with KernelExplainer
        model_copy = copy.deepcopy(clf)
        
        if hasattr(clf, "predict_proba"):
            explainer = shap.KernelExplainer(model_copy.predict_proba, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model_copy.predict, background_summary)
            shap_values_model = explainer.shap_values(x_test)
        
        # Check if shap_values_model is a list (indicating multi-output)
        if isinstance(shap_values_model, list):
            # Select the SHAP values for the positive class (CKD)
            shap_values_model = shap_values_model[1]

        # Format SHAP values and features to 2 decimal places
        shap_values_model = np.round(shap_values_model, 2)
        x_test_rounded = x_test.round(2)

        # Generate waterfall plot for each instance in the test set
        for i in range(x_test.shape[0]):
            plt.figure(figsize=(10, 8))
            
            # Create a SHAP values object for the waterfall plot
            shap_explanation = shap.Explanation(values=shap_values_model[i], 
                                                base_values=explainer.expected_value[1], 
                                                data=x_test_rounded.iloc[i], 
                                                feature_names=feature_names)
            
            # Plot the waterfall plot
            shap.plots.waterfall(shap_explanation, show=False)
            
            # Save the waterfall plot
            SHAP_plot_filename = f'SHAP_{label}_waterfall_plot_{i}.png'
            SHAP_plot_filepath = os.path.join(save_dir, SHAP_plot_filename)
            plt.savefig(SHAP_plot_filepath, bbox_inches='tight')
            
            print(f'SHAP waterfall plot saved to {SHAP_plot_filepath}')
            
            # Close the plot to avoid memory issues
            plt.close()

    except Exception as e:
        print(f"Error processing SHAP for {label}: {str(e)}")








