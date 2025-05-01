# --- ML Script: PCA + Hyperparameter Tuning for Multiple Classifiers ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import time
import warnings

# Import necessary components
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Example: Add GradientBoostingClassifier if desired
# from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    roc_curve,
    auc
)
from sklearn.exceptions import UndefinedMetricWarning

# --- Specificity Scorer Function ---
def specificity_score(y_true, y_pred, **kwargs): # Added **kwargs to handle potential extra args from scorers
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
   
    if cm.shape != (2, 2):

        return 0.0 # Or np.nan
    tn, fp, fn, tp = cm.ravel()
    if (tn + fp) == 0:
        return 1.0 if fp == 0 else 0.0
    return tn / (tn + fp)


# Balanced accuracy is often a good primary metric for potentially imbalanced data
scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(specificity_score),
    'precision': make_scorer(precision_score, pos_label=1, zero_division=0),
    'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
    'roc_auc': 'roc_auc' # Use built-in string for AUC scorer
}
# Choose the main metric to optimize GridSearchCV ('refit')
refit_metric = 'balanced_accuracy'

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# --- 1. Configuration ---
print("--- Configuration for ML (PCA + Multi-Classifier Tuning + ROC) ---")
project_dir = '/home/ecren/pd_project' # MODIFY AS NEEDED

# --- INPUTS ---
features_input_path = os.path.join(project_dir, 'features', 'extracted_features_ho_thr25.csv') # Path from previous step

# --- OUTPUTS ---
results_output_dir = os.path.join(project_dir, 'ml_results_pca_tuned') # New output dir
summary_output_filename = 'ml_results_summary_pca_tuned.csv'
plot_dir = os.path.join(results_output_dir, 'plots') # Subdirectory for plots

# --- ML Parameters ---
n_splits = 10 # Number of folds for StratifiedKFold cross-validation
random_state = 42 # Ensures reproducibility
n_jobs = -1 # Use all available CPU cores for GridSearchCV and CVP

# --- Create results directories ---
os.makedirs(results_output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


# --- 2. Load Feature Data ---
print(f"\n--- Loading Features from: {features_input_path} ---")
try:
    features_df = pd.read_csv(features_input_path)
    print(f"Loaded features data with shape: {features_df.shape}")
    if 'Subject' not in features_df.columns or 'Group' not in features_df.columns:
        raise ValueError("Subject/Group column missing.")
    # Check for NaNs in feature columns
    feature_cols_only = features_df.columns.difference(['Subject', 'Group'])
    if features_df[feature_cols_only].isnull().any().any():
        print("WARNING: NaNs detected in feature columns.")
        # Option 1: Fill with 0 (simple, but might not be optimal)
        # features_df[feature_cols_only] = features_df[feature_cols_only].fillna(0)
        # Option 2: Fill with column mean (often better)
        print("Filling NaNs with column means.")
        for col in feature_cols_only:
            if features_df[col].isnull().any():
                 mean_val = features_df[col].mean()
                 features_df[col] = features_df[col].fillna(mean_val)
        # Verify NaNs are gone
        if features_df[feature_cols_only].isnull().any().any():
             print("ERROR: NaNs persist after attempting to fill. Check data.")
             raise SystemExit("Exiting.")
        else:
             print("NaNs filled successfully.")

except FileNotFoundError:
    print(f"ERROR: Features file not found at {features_input_path}. Please verify the path.")
    raise SystemExit("Exiting.")
except Exception as e:
    print(f"ERROR loading features: {e}")
    raise SystemExit("Exiting.")
if features_df.empty:
    raise SystemExit("ERROR: No subjects in features file.")


# --- 3. Prepare Final Data (X and y) ---
print("\n--- Preparing Final Data (X and y) ---")
target_col = 'Group'
# Assuming 'PD' = 1 (positive class), 'Control' = 0 (negative class)
# Adjust if your labels are different
label_map = {'PD': 1, 'Control': 0}

if features_df[target_col].dtype == 'object': # Check if mapping is needed
    y = features_df[target_col].map(label_map)
    print(f"Mapped target labels: PD={label_map['PD']}, Control={label_map['Control']}")
else:
    y = features_df[target_col] # Assume already numeric
    print(f"Using numeric target labels directly. Ensure 1 is positive (PD), 0 is negative (Control).")

# Use all columns except Subject and Group as features
X = features_df.drop(columns=['Subject', target_col])
feature_names = X.columns.tolist()
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts(normalize=True)}")


# --- 4. Define Classifiers, Pipelines, and Parameter Grids ---
print("\n--- Defining Classifiers, Pipelines, and Parameter Grids ---")

# Define pipeline steps (common to all)
pipeline_steps = [
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=random_state)),
    # Placeholder for the classifier
]

# Define classifiers and their specific parameter grids

classifiers = {
    'LogisticRegression': {
        'estimator': LogisticRegression(random_state=random_state, class_weight='balanced', solver='liblinear', max_iter=1000),
        'param_grid': {
            'pca__n_components': [0.85, 0.90, 0.95], # Variance explained
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2']
        }
    },
    'SVC': {
        'estimator': SVC(random_state=random_state, class_weight='balanced', probability=True), # probability=True for ROC AUC
        'param_grid': {
            'pca__n_components': [0.85, 0.90, 0.95],
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto', 0.01, 0.1] # 'scale' and 'auto' are common defaults for rbf
        }
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        'param_grid': {
            'pca__n_components': [0.85, 0.90, 0.95, None], # Include None to test without PCA implicitly via variance
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 3]
        }
    },
    'KNeighbors': {
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'pca__n_components': [0.85, 0.90, 0.95],
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan']
        }
    },
    'GaussianNB': {
        'estimator': GaussianNB(),
        'param_grid': {
            'pca__n_components': [0.85, 0.90, 0.95, None], # Naive Bayes can work well with many features too
            'clf__var_smoothing': [1e-9, 1e-8, 1e-7] # Default is 1e-9
        }
    }
    # Add other classifiers here if needed
    # 'GradientBoosting': {
    #     'estimator': GradientBoostingClassifier(random_state=random_state),
    #     'param_grid': { ... }
    # }
}

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Dictionary to store results
all_results = defaultdict(list)
best_estimators = {}

# --- 5. Run GridSearchCV for each Classifier ---
print(f"\n--- Running GridSearchCV with {n_splits}-Fold CV for {len(classifiers)} Classifiers ---")
total_start_time = time.time()

for clf_name, config in classifiers.items():
    clf_start_time = time.time()
    print(f"\n===== Tuning {clf_name} =====")

    # Create the full pipeline for this classifier
    current_pipeline = Pipeline(pipeline_steps + [('clf', config['estimator'])])

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=current_pipeline,
        param_grid=config['param_grid'],
        scoring=scoring, # Use the dictionary of scorers
        refit=refit_metric, # Optimize for balanced accuracy
        cv=cv_strategy,
        n_jobs=n_jobs, # Use parallel processing
        verbose=1 # Show progress
    )

    # Run Grid Search
    try:
        grid_search.fit(X, y)
    except Exception as e:
        print(f"ERROR during GridSearchCV for {clf_name}: {e}")
        print("Skipping this classifier.")
        continue # Move to the next classifier

    # Store best estimator and results
    best_pipeline = grid_search.best_estimator_
    best_estimators[clf_name] = best_pipeline
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_ # Score for the 'refit' metric
    best_cv_std = grid_search.cv_results_[f'std_test_{refit_metric}'][grid_search.best_index_]

    print(f"\nBest Parameters for {clf_name}: {best_params}")
    print(f"Best Mean {refit_metric.replace('_', ' ').title()} (CV): {best_cv_score:.4f} +/- {best_cv_std:.4f}")

    # --- Evaluate Best Model using Cross-Validation Predictions ---
    print(f"\n--- Evaluating Best {clf_name} using cross_val_predict ---")
    try:
        y_pred_cv = cross_val_predict(best_pipeline, X, y, cv=cv_strategy, n_jobs=n_jobs)
        # Get probabilities ONLY if the classifier supports it AND probability=True was set (or default)
        if hasattr(best_pipeline.named_steps['clf'], 'predict_proba'):
            y_proba_cv = cross_val_predict(best_pipeline, X, y, cv=cv_strategy, method='predict_proba', n_jobs=n_jobs)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba_cv, pos_label=1)
            roc_auc_cvp = auc(fpr, tpr)
        else:
            print(f"  {clf_name} does not support predict_proba, AUC not calculated.")
            y_proba_cv = None
            roc_auc_cvp = np.nan
            fpr, tpr = None, None

    except Exception as e:
        print(f"ERROR during cross_val_predict for {clf_name}: {e}")
        y_pred_cv, y_proba_cv, roc_auc_cvp = None, None, np.nan
        fpr, tpr = None, None

    # Calculate metrics from CVP results
    if y_pred_cv is not None:
        bacc_cvp = balanced_accuracy_score(y, y_pred_cv)
        acc_cvp = accuracy_score(y, y_pred_cv)
        sens_cvp = recall_score(y, y_pred_cv, pos_label=1, zero_division=0)
        spec_cvp = specificity_score(y, y_pred_cv)
        prec_cvp = precision_score(y, y_pred_cv, pos_label=1, zero_division=0)
        f1_cvp = f1_score(y, y_pred_cv, pos_label=1, zero_division=0)
        cm_cvp = confusion_matrix(y, y_pred_cv, labels=[0, 1])

        print("\nDetailed Metrics (from CVP class predictions):")
        print(f"  Overall Balanced Accuracy: {bacc_cvp:.4f}")
        print(f"  Overall Accuracy         : {acc_cvp:.4f}")
        print(f"  Overall Sensitivity      : {sens_cvp:.4f}")
        print(f"  Overall Specificity      : {spec_cvp:.4f}")
        print(f"  Overall Precision        : {prec_cvp:.4f}")
        print(f"  Overall F1-Score         : {f1_cvp:.4f}")
        print(f"  Overall AUC              : {roc_auc_cvp:.4f}" if not np.isnan(roc_auc_cvp) else "  Overall AUC              : N/A")


        # --- Store Results ---
        all_results['Classifier'].append(clf_name)
        all_results['Best_Params'].append(str(best_params)) # Store as string for CSV compatibility
        all_results['Mean_CV_BACC (GridSearch)'].append(best_cv_score)
        all_results['Std_CV_BACC (GridSearch)'].append(best_cv_std)
        all_results['BACC (CVP)'].append(bacc_cvp)
        all_results['Accuracy (CVP)'].append(acc_cvp)
        all_results['Sensitivity (CVP)'].append(sens_cvp)
        all_results['Specificity (CVP)'].append(spec_cvp)
        all_results['Precision (CVP)'].append(prec_cvp)
        all_results['F1_Score (CVP)'].append(f1_cvp)
        all_results['AUC (CVP)'].append(roc_auc_cvp)

        # --- Plot Confusion Matrix ---
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_cvp, annot=True, fmt='d', cmap='Blues', xticklabels=['HC (0)', 'PD (1)'], yticklabels=['HC (0)', 'PD (1)'])
        plt.title(f'Overall CM - Best {clf_name}\n(PCA + Tuned Params)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_plot_path = os.path.join(plot_dir, f'cm_{clf_name}.png')
        try:
            plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
            print(f"\nConfusion matrix plot saved to: {cm_plot_path}")
        except Exception as e:
            print(f"\nWarning: Could not save CM plot for {clf_name}: {e}")
        plt.close() # Close plot to avoid displaying inline if not needed

        # --- Plot ROC Curve ---
        if fpr is not None and tpr is not None:
            plt.figure(figsize=(7, 7))
            plt.plot(fpr, tpr, lw=2, label=f'{clf_name} (AUC = {roc_auc_cvp:.3f})')
            plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Chance (AUC = 0.50)')
            plt.xlim([-0.02, 1.0]); plt.ylim([0.0, 1.02])
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.title(f'ROC Curve - Best {clf_name} (PCA + Tuned Params)')
            plt.legend(loc="lower right"); plt.grid(alpha=0.3)
            roc_plot_path = os.path.join(plot_dir, f'roc_{clf_name}.png')
            try:
                plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
                print(f"ROC curve plot saved to: {roc_plot_path}")
            except Exception as e:
                print(f"\nWarning: Could not save ROC plot for {clf_name}: {e}")
            plt.close() # Close plot

    else: # Handle case where CVP failed
        print("\nSkipping metrics and plots due to CVP errors.")
        # Append NaN values to results
        all_results['Classifier'].append(clf_name)
        all_results['Best_Params'].append(str(best_params))
        all_results['Mean_CV_BACC (GridSearch)'].append(best_cv_score)
        all_results['Std_CV_BACC (GridSearch)'].append(best_cv_std)
        for metric in ['BACC (CVP)', 'Accuracy (CVP)', 'Sensitivity (CVP)', 'Specificity (CVP)', 'Precision (CVP)', 'F1_Score (CVP)', 'AUC (CVP)']:
            all_results[metric].append(np.nan)

    clf_end_time = time.time()
    print(f"----- {clf_name} Tuning & Evaluation Time: {(clf_end_time - clf_start_time):.2f} seconds -----")


# --- 6. Summarize and Save Results ---
print("\n--- Summarizing All Classifier Results ---")
results_df = pd.DataFrame(all_results)
# Sort by the primary CV metric (higher is better)
results_df = results_df.sort_values(by='Mean_CV_BACC (GridSearch)', ascending=False).round(4)

print("\nOverall Results Summary:")
print(results_df)

summary_path = os.path.join(results_output_dir, summary_output_filename)
try:
    results_df.to_csv(summary_path, index=False)
    print(f"\n--- Full results summary saved to: {summary_path} ---")
except Exception as e:
    print(f"\nWarning: Could not save summary results CSV: {e}")



total_end_time = time.time()
print(f"\n--- ML Script (PCA + Multi-Classifier Tuning) Complete ---")
print(f"Total execution time: {(total_end_time - total_start_time) / 60:.2f} minutes.")