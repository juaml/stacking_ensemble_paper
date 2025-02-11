#%%
import pandas as pd
import os.path as op
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc
from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator
from sklearn.model_selection import RepeatedKFold
from pprint import pprint  # To print in a pretty way

project_dir = "/home/antogeo/codes/avg_vs_pred_stack/"
df = pd.read_csv(
    op.join(project_dir, "result_files", "analysis", "merged_level0.csv"),
    index_col=0
).reset_index(drop=True)
configure_logging(level="INFO")
# %%
col_age = [x for x in df.columns if "pred_age" in x]
col_mean = [x for x in df.columns if "mean_" in x]
datasets = ["IXI", "eNKI", "CamCAN", "1000Gehirne"]
num_splits = 5
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=200)
#%%
# creator1 = PipelineCreator(problem_type="classification")
# creator1.add("zscore")
# creator1.add(
#     "svm",
#     C=[0.01, 0.1, 1, 10],
#     gamma=[1e-3, 1e-2, "scale", "auto"],
#     kernel=["rbf"],
#     name="model",
# )

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    kernel=["linear"],
    name="model",
)

# creator3 = PipelineCreator(problem_type="classification")
# creator3.add("zscore")
# creator3.add(
#     "rf",
#     max_depth=[2, 3, 4],
#     name="model",
# )

# creator4 = PipelineCreator(problem_type="classification")
# creator4.add("zscore")
# creator4.add(
#     "logit",
#     penalty=["l2", "l1"],
#     dual=[False],
#     solver="liblinear",
#     name="model",
# )

#%%
data_labels = ['1000Gehirne', 'CamCAN', 'IXI', 'eNKI']
scores_age, model_age = run_cross_validation(
    X=col_age, y='train', data=df[df.train.isin(datasets)],
    # cv=cv, model=[creator1, creator2, creator3, creator4],
    cv=cv, model=[creator2],
    return_estimator='all', scoring=['balanced_accuracy'])

print(f"Scores with best hyperparameter: {scores_age['test_accuracy'].mean()}")
pprint(model_age.best_params_)
# %%
scores_gmv, model_gmv = run_cross_validation(
    X=col_mean, y='train', data=df[df.train.isin(datasets)],
    # cv=cv, model=[creator1, creator2, creator3, creator4],
    cv=cv, model=[creator2],
    return_estimator='all', scoring=['balanced_accuracy', 'roc_auc_ovr'])

print(f"Scores with best hyperparameter: {scores_gmv['test_balanced_accuracy'].mean()}")
pprint(model_gmv.best_params_)

# %%
splits = list(cv.split(df[df.train.isin(datasets)][col_mean], 
                       df[df.train.isin(datasets)]))
for scores, rois in zip([scores_age, scores_gmv], [col_age, col_mean]):
    df_accuracy = scores.set_index(
            ['repeat', 'fold'])['test_balanced_accuracy'].unstack()
    df_accuracy.index.name = 'Repeats'
    df_accuracy.columns.name = 'K-fold splits'
    print(df_accuracy)
    y_pred = np.zeros((5, df[df.train.isin(data_labels)].shape[0]), dtype='object')
    for iter, (_, val_ind) in enumerate(splits):
        cv_repeat_id = scores.repeat[iter]
        y_pred_i = scores.estimator[iter].predict(
                    df[df.train.isin(data_labels)].iloc[val_ind][rois])
        y_pred[cv_repeat_id, val_ind] = y_pred_i
    cm = np.zeros((len(df[df.train.isin(data_labels)]['train'].unique()),
                    len(df[df.train.isin(data_labels)]['train'].unique())))
    for cv_pred in y_pred:
        cm = confusion_matrix(df[df.train.isin(data_labels)]['train'].values,
                                cv_pred.astype(str)) + cm
    np.set_printoptions(suppress=True, precision=4)
    cm = cm / 5
    print(cm)

    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                s = cm_sum[i]
                annot[i, j] = '%g/%g' % (c, s)
    df_cm = pd.DataFrame(cm, index=np.array(
                        data_labels), columns=np.array(data_labels))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="YlGnBu", annot=annot, fmt='').set(
                title='All data Confusion matrix\n ')
plt.show()
