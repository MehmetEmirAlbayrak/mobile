import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data Parameters
TEST_SIZE = 0.15  # Reduced to give more training data
VAL_SIZE = 0.1  # of the training set
RANDOM_STATE = 42

# Label Mapping (Standardizing ISCX labels if needed)
# This might need adjustment based on the exact CSV content
LABEL_COLUMN = 'Label' 

# Model Hyperparameters (can be tuned)
RF_PARAMS = {
    'n_estimators': 500,          # Reduced for speed
    'max_depth': 30,              # Limited depth
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'criterion': 'gini',
    'bootstrap': True
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 10,
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

XGB_PARAMS = {
    'n_estimators': 1000,         # Reduced for speed
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.0,
    'min_child_weight': 1,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist'         # CPU (fast)
}

# Search Grids for RandomizedSearchCV
RF_GRID = {
    'n_estimators': [400, 500],
    'max_depth': [25, 30],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

XGB_GRID = {
    'n_estimators': [2000, 2500, 3000],
    'learning_rate': [0.003, 0.005, 0.01],
    'max_depth': [8, 10, 12],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9],
    'gamma': [0, 0.01, 0.05],
    'min_child_weight': [1, 2, 3]
}

LGBM_GRID = {
    'n_estimators': [2500, 3000, 3500],
    'learning_rate': [0.003, 0.005, 0.01],
    'num_leaves': [100, 127, 150],
    'max_depth': [12, 15, 18],
    'min_child_samples': [3, 5, 10],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9]
}

MLP_PARAMS = {
    'hidden_layer_sizes': (200, 100, 50),  # Reduced for memory
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'batch_size': 1024,  # Larger batch = less memory pressure
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 300,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 15
}

LGBM_PARAMS = {
    'n_estimators': 1500,          # Reduced for speed
    'learning_rate': 0.01,
    'num_leaves': 100,
    'max_depth': 15,
    'min_child_samples': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'min_split_gain': 0.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'boosting_type': 'gbdt',
    'importance_type': 'gain'
    # GPU disabled - not available in pip version
}

