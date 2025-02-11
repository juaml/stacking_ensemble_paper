#%%
import pandas as pd
import numpy as np
import os.path as op
import scipy.stats as stats
# import codes.custom_lib as cl
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = "/home/antogeo/codes/avg_vs_pred_stack/"
df_all = pd.read_csv(
    op.join(project_dir, "result_files/analysis/merged_level0.csv"),
    index_col=0
)
df = pd.read_csv(
    op.join(project_dir, "result_files/analysis/merged_level0.csv"),
    index_col=0
)[["age", "train"]]
#%%
# plot hists of ages for each dataset
ax = plt.gca()
for dset in ["IXI", "eNKI", "CamCAN", "1000Gehirne"]:
    df[df.train == dset]["age"].plot(
        kind="hist", ax=ax, density=False, alpha=0.8, bins=90
    )
plt.legend(["IXI", "eNKI", "CamCAN", "1000Gehirne"])
plt.show()
#%%
df_ixi = df_all[df_all.train == "IXI"]
df_eNKI = df_all[df_all.train == "eNKI"]
df_cc = df_all[df_all.train == "CamCAN"]
df_1000 = df_all[df_all.train == "1000Gehirne"]

age_cols = [col for col in df_all.columns if "pred_age" in col]
mean_cols = [col for col in df_all.columns if "mean_" in col]
#%%
# Calculate Pearson's r between the GMV or l0pred and age for each parcel.
# Then use these 873 values to create a nifti image. Final results are two
# niftis for each dataset: one for GMV and one for L0-pred
for ds, df_ds in zip(
    ["IXI", "eNKI", "CamCAN", "1000Gehirne"], [df_ixi, df_eNKI, df_cc, df_1000]
):
    df_temp = df_ds[mean_cols]
    df_temp["age"] = df_ds["age"].values

    ofile = op.join(project_dir, "niftis", ds + "_corr_gmv_age_acrsub.nii")
    # cl.make_nii(ofile, df_temp.corr()["age"][:-1])

    df_temp = df_ds[age_cols]
    df_temp["age"] = df_ds["age"].values
    ofile = op.join(project_dir, "niftis", ds + "_corr_l0pred_age_acrsub.nii")
    # cl.make_nii(ofile, df_temp.corr()["age"][:-1])
#%%
df_4train = df_all[df_all.train == "1000Gehirne-CamCAN-eNKI-IXI"]

# Calculate Pearson's r between the GMV or l0pred and age for each parcel.
# Then use these 873 values to create a nifti image. Final results are two
# niftis for each dataset: one for GMV and one for L0-pred

df_temp = df_4train[mean_cols]
df_temp["age"] = df_4train["age"].values

ofile = op.join(project_dir, "niftis", "4train_corr_gmv_age_acrsub_negated.nii")
cl.make_nii(ofile, -df_temp.corr()["age"][:-1])
ofile = op.join(project_dir, "niftis", "4train_corr_gmv_age_acrsub.nii")
cl.make_nii(ofile, df_temp.corr()["age"][:-1])

df_temp = df_4train[age_cols]
df_temp["age"] = df_4train["age"].values
ofile = op.join(project_dir, "niftis", "4train_corr_l0pred_age_acrsub.nii")
cl.make_nii(ofile, df_temp.corr()["age"][:-1])
#%%
# VIOLIN plots of GMV and l0pred
df_temp = df_4train[mean_cols]
df_temp["age"] = df_4train["age"].values
df_corr1 = pd.DataFrame(-df_temp.corr()["age"][:-1]).reset_index()
df_corr1["setup"] = "GMV mean (negated)"

df_temp = df_4train[age_cols]
df_temp["age"] = df_4train["age"].values
df_corr2 = pd.DataFrame(df_temp.corr()["age"][:-1]).reset_index()
df_corr2["setup"] = "L0 predictions"
df_corr1 = df_corr1.append(df_corr2).reset_index()

sns.violinplot(data=df_corr1, x='setup', y='age',
               showmeans=False, showmedians=True,
               showextrema=False, legend='full',
               palette="Set2", inner='quart')
plt.legend(title='Correlation of L0 outputs \n and age across subjects',
           loc='upper left')
plt.ylabel('Pearson\'s r')
plt.show()
#%%
# Create niftis of mean corr of GMV and l0pred across datasets
df_all_GMV, df_all_l0pred = pd.DataFrame(), pd.DataFrame()

#%%
for ds, df_ds in zip(
    ["IXI", "eNKI", "CamCAN", "1000Gehirne"], [df_ixi, df_eNKI, df_cc, df_1000]
):
    df_temp = df_ds[mean_cols]
    df_temp["age"] = df_ds["age"].values
    df_all_GMV = pd.concat([df_all_GMV, df_temp.corr()["age"][:-1]], axis=1)

    df_temp = df_ds[age_cols]
    df_temp["age"] = df_ds["age"].values
    df_all_l0pred = pd.concat([df_all_l0pred,
                               df_temp.corr()["age"][:-1]],
                               axis=1)
plt.figure()
plt.scatter(df_all_GMV.mean(axis=1), df_all_l0pred.mean(axis=1), s=2)
# plt.annotate()
plt.show()
#%%
ofile = op.join(project_dir, "niftis", "mean_corr_GMV_acrsub.nii")
cl.make_nii(ofile, df_all_GMV.mean(axis=1))

ofile = op.join(project_dir, "niftis", "mean_corr_L0pred_acrsub.nii")

cl.make_nii(ofile, df_all_l0pred.mean(axis=1))
#%%

# create scatter plots of Pearson's r between L0 pred-age and GMV-age
# for all datasets.
combs = list(
    combinations(
        zip(
            ["IXI", "eNKI", "CamCAN", "1000Gehirne"], [df_ixi, df_eNKI, df_cc, df_1000]
        ),
        2,
    )
)

fig, axs = plt.subplots(2, 3, figsize=(10, 8))

for i, (ax, pair) in enumerate(zip(axs.flat, combs)):
    df_1 = pair[0][1][mean_cols]
    df_1["age"] = df[df.train == pair[0][0]]["age"].values

    df_2 = pair[1][1][mean_cols]
    df_2["age"] = df[df.train == pair[1][0]]["age"].values
    x = df_1.corr()["age"][:-1]
    y = df_2.corr()["age"][:-1]
    ax.scatter(x, y, s=2)
    ax.set_title("{} - {}".format(pair[0][0], pair[1][0]))
    ax.set_xlim(-0.75, 0.25)
    ax.set_ylim(-0.75, 0.25)
    ax.axline(
        (-0.75, -0.75),
        (0.25, 0.25),
        linewidth=2,
        color="k",
        linestyle=(0, (5, 5)),
        label="perfect fit" if i == 0 else "",
    )
    b, a = np.polyfit(x, y, deg=1)
    ax.plot(x, a + b * x, color="r", lw=1.5, label="Regression line" if i == 0 else "")
    r, p = stats.pearsonr(x, y)
    ax.annotate("r = {:.2f}".format(r), xy=(0.7, 0.9), xycoords="axes fraction")
    ax.set_xlabel("Parcelwise r with GMV for {}".format(pair[0][0]))
    ax.set_ylabel("Parcelwise r with GMV for {}".format(pair[1][0]))
    fig.legend()
#%%
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
for i, (ax, pair) in enumerate(zip(axs.flat, combs)):
    print(pair[0][0], pair[1][0])
    df_1 = pair[0][1][age_cols]
    df_1["age"] = df[df.train == pair[0][0]]["age"].values

    df_2 = pair[1][1][age_cols]
    df_2["age"] = df[df.train == pair[1][0]]["age"].values
    x = df_1.corr()["age"][:-1]
    y = df_2.corr()["age"][:-1]
    ax.scatter(x, y, s=2)
    ax.set_title("{} - {}".format(pair[0][0], pair[1][0]))
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-0.01, 1)
    ax.axline(
        (-0.01, -0.01),
        (1, 1),
        linewidth=2,
        color="k",
        linestyle=(0, (5, 5)),
        label="perfect fit" if i == 0 else "",
    )
    b, a = np.polyfit(x, y, deg=1)
    ax.plot(x, a + b * x, color="r", lw=1.5, label="Regression line" if i == 0 else "")
    r, p = stats.pearsonr(x, y)
    ax.annotate("r = {:.2f}".format(r), xy=(0.7, 0.9), xycoords="axes fraction")
    ax.set_xlabel("Parcelwise r with predicted age for {}".format(pair[0][0]))
    ax.set_ylabel("Parcelwise r with predicted age for {}".format(pair[1][0]))
    fig.legend()
plt.show()
