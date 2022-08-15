import numpy as np 
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn import datasets
from sklearn.model_selection import train_test_split


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

diabetes = datasets.load_diabetes(as_frame = True)
df = diabetes["frame"]
regressors = df.columns[0:-1]
train_df, val_df = train_test_split(df, test_size = 0.2)


def get_lr_coef(regressors, model_variable, train_df, val_df):
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
           
print(get_lr_coef(regressors, "target", train_df, val_df))