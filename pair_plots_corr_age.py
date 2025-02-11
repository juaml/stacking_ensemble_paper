#%%
import pandas as pd
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    # ax = plt.gca()
    # ax.annotate("r = {:.2f} ".format(r),
    #             xy=(.08, .9), xycoords=ax.transAxes,
    #             size=14)
    return r


# %%
def ccc(y_true, y_pred, **kws):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true.astype(float), y_pred.astype(float))[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


project_dir = "/home/antogeo/codes/avg_vs_pred_stack/"
df_all = pd.read_csv(
    op.join(project_dir, "result_files/analysis/merged_level0.csv"),
    index_col=0
)

# %%
# make pairplots
col_age = [x for x in df_all.columns if "pred_age" in x]
col_mean = [x for x in df_all.columns if "mean_" in x]
df_age = df_all[col_age]
df_age = pd.concat([df_age, df_all[['age', 'train']]], axis=1)
df_mean = df_all[col_mean]
df_mean = pd.concat([df_mean, df_all[['age', 'train']]], axis=1)

df_age.columns = df_mean.columns
df_corr_age = df_corr_mean = pd.DataFrame()
datasets = ['IXI', 'eNKI', 'CamCAN', '1000Gehirne']

for ds in datasets:
    corr_mean = corr_age = ccc_mean = ccc_age = []
    corr_mean = df_mean[df_mean.train == ds].corr()["age"][:-1]
    corr_mean['dataset'] = ds
    corr_age = df_age[df_age.train == ds].corr()["age"][:-1]
    corr_age['dataset'] = ds
    df_corr_age = pd.concat([df_corr_age, corr_age.to_frame().T],
                            axis=0, ignore_index=True)
    df_corr_mean = pd.concat([df_corr_mean, corr_mean.to_frame().T],
                             axis=0, ignore_index=True)


#%%
df_age_piv = df_corr_age.pivot_table(columns='dataset')
df_age_piv['what'] = 'L0 age'
df_mean_piv = df_corr_mean.pivot_table(columns='dataset')
numeric_col = df_mean_piv.select_dtypes("number").columns
df_mean_piv['what'] = 'mean GMV'
df_mean_piv[numeric_col] *= -1
df_age_piv.index = df_age_piv.index.str.replace("mean_", "").astype(int)
df_mean_piv.index = df_mean_piv.index.str.replace("mean_", "").astype(int)
df_mean_piv.sort_index(inplace=True)
df_age_piv.sort_index(inplace=True)
# df_age_piv.rename(columns={"1000Gehirne": "1000 Brains"}, inplace=True)
# df_mean_piv.rename(columns={"1000Gehirne": "1000 Brains"}, inplace=True)

#%%
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# subcortex 801-836 

for i, (ax, ds) in enumerate(zip(axs.flat, datasets)):
    x = df_age_piv[ds]
    y = df_mean_piv[ds]
    ax.scatter(x.loc[1:800], y.loc[1:800], s=2, color='gold',
               label="cortex")
    ax.scatter(x.loc[801:836], y.loc[801:836], s=3, color='deeppink',
               label="subcortex")
    ax.scatter(x.loc[837:], y.loc[837:], s=3, color='royalblue',
               label="cerebellum")
    ax.set_title(ds, fontsize=18)
    ax.set_xlim(-0.3, 1)
    ax.set_ylim(-0.3, 1)
    ax.tick_params(labelsize=14)
    ax.axline([0, 0], slope=1, color='silver')
fig.supxlabel("Pearson's r between real age-L0 predicted age\
              \n across subjects", fontsize=18)
fig.supylabel("Pearson's r between real age-mean GMV\
               \n across subjects", fontsize=18)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.show()

# #%%
# sns.set_context('talk', font_scale=0.8)
# g1 = sns.pairplot(df_age_piv, kind="reg",
#                   plot_kws=dict(line_kws={'color': 'red'},
#                                 scatter_kws={'s': 3}))
# g1.set(xlim=(-0.3, 1))
# g1.set(ylim=(-.3, 1))
# # g1.tick_params(labelsize=14)
# g1.map(corrfunc)
# g1.map(ccc)
# g2 = sns.pairplot(df_mean_piv, kind="reg",
#                   plot_kws=dict(line_kws={'color': 'orange'},
#                                 scatter_kws={'s': 3}))
# g2.set(xlim=(-0.3, 1))
# g2.set(ylim=(-.3, 1))
# # g2.tick_params(labelsize=14)
# g2.map(corrfunc)
# g2.map(ccc)
# plt.show()

#%%
datasets = ['IXI', 'eNKI', 'CamCAN', '1000Gehirne']
fig, axs = plt.subplots(4, 4, figsize=(12, 10))

for i, ds1 in enumerate(datasets):
    for j, ds2 in enumerate(datasets):
        if ds1 == ds2:
            axs[i, j].hist(df_age_piv[ds1], bins=30,
                           color="blue", alpha=0.5)
            axs[i, j].hist(df_mean_piv[ds1], bins=30,
                           color="green", alpha=0.5)
            axs[i, j].yaxis.set_tick_params(width=0)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([0, .5])
            axs[i, j].set_xticklabels([0, .5])
        else:
            if i > j:
                axs[i, j].scatter(df_mean_piv[ds1], df_mean_piv[ds2],
                                  c="green", s=1, alpha=0.5)
                axs[i, j].annotate("r = {:.2f}".format(
                                   corrfunc(df_mean_piv[ds1],
                                            df_mean_piv[ds2])),
                                   xy=(-.20, .75), size=10)
                # axs[i, j].annotate("ccc = {:.2f}".format(
                #                    ccc(df_mean_piv[ds1],
                #                        df_mean_piv[ds2])),
                #                    xy=(-.20, .65), size=10)
            else:
                axs[i, j].scatter(df_age_piv[ds1], df_age_piv[ds2],
                                  c="blue", s=1, alpha=0.5)
                axs[i, j].annotate("r = {:.2f}".format(
                                   corrfunc(df_age_piv[ds1],
                                            df_age_piv[ds2])),
                                   xy=(-.20, .75), size=10)
                # axs[i, j].annotate("ccc = {:.2f}".format(
                #                    ccc(df_age_piv[ds1],
                #                        df_age_piv[ds2])),
                #                    xy=(-.20, .65), size=10)
            axs[i, j].plot((-.27, 1), (-.27, 1), '-',
                           color='darkorange', alpha=0.75)
            # axs[i, j].tick_params(axis='both', which='major', labelsize=8)
            plt.setp(axs[i, j], xlim=(-.25, 1.01),  ylim=(-.25, 1.01))
            axs[i, j].set_xticks([-.25, 0, .5, .9])
            axs[i, j].set_yticks([-.25, 0, .5, .9])
            axs[i, j].set_yticklabels(["", 0, ".5", ".9"])
            axs[i, j].set_xticklabels(["", 0, .5, .9])
            # axs[i, j].tick_params(axis='both', which='minor',
            #                       direction="in", pad=-20,
            #                       labelsize=8)
            axs[i, j].tick_params(axis="y", which='major',
                                  direction="in",
                                  labelsize=10, pad=-15)
        axs[i, j].tick_params(axis="x", direction="in",
                              labelsize=10, pad=-12)
        if i == 0:
            axs[i, j].set_title(ds2, fontsize=18)
        if j == 0:
            axs[i, j].set_ylabel(ds1, fontsize=18)
        axs[i, j].xaxis.set_tick_params(width=0.5, length=5)
        axs[i, j].yaxis.set_tick_params(width=0.5, length=5)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()

# %%
