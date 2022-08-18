import numpy as np 
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import optuna


def get_lr_coef(regressors, model_variable, train_df, val_df):
    """
    Given training and validation dataframes, fits the best possible linear model(h2o) and returns the coefficients
    """
    h2o.init(nthreads = -1, max_mem_size = "8G", ip = "localhost")
    train_h2o, val_h2o = h2o.H2OFrame(train_df) , h2o.H2OFrame(val_df)
    glm_models = []
    glm_scores = []
    glm_families = []
    for family in ["gaussian","tweedie","poisson","gamma"]:
        try:
            glm = H2OGeneralizedLinearEstimator(family=family,
                        seed=23123,
                        standardize = False,
                        remove_collinear_columns = True,
                        stopping_rounds = 10,
                        nfolds=0)
            glm.train(x=list(regressors),
                    y = model_variable,
                    training_frame = train_h2o,
                    validation_frame = val_h2o)
            glm_models.append(glm)
            glm_scores.append(glm.r2(valid = True))
            glm_families.append(family)
        except:
            pass
    best_model_idx = np.argmax(glm_scores)
    print("Best model score : {0:.3f}({1})".format(glm_scores[best_model_idx], glm_families[best_model_idx]))
    best_glm = glm_models[best_model_idx]
    return(best_glm.coef())


def get_val_tuned_rfe(train_df, val_df, regressors, model_variable, n_trials=20):
    optuna.logging.set_verbosity(0)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
            'verbose':False
        }
        reg = RandomForestRegressor(**params)
        reg.fit(train_df[regressors].values,
                train_df[model_variable].values)
        return(reg.score(val_df[regressors].values,
                val_df[model_variable].values))
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)  
    trial = study.best_trial
    optuna.logging.set_verbosity(20)
    model = RandomForestRegressor(**trial.params)
    return(trial.value, trial.params, model)
        
def get_val_tuned_gbm(train_df, val_df, regressors, model_variable, n_trials=20):
    optuna.logging.set_verbosity(0)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
            'verbose':False
        }
        reg = GradientBoostingRegressor(**params)
        reg.fit(train_df[regressors].values,
                train_df[model_variable].values)
        return(reg.score(val_df[regressors].values,
                val_df[model_variable].values))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)  
    trial = study.best_trial
    optuna.logging.set_verbosity(20)
    model = GradientBoostingRegressor(**trial.params)
    return(trial.value, trial.params, model)