#%%
import pandas as pd
import os.path as op
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Custom function to replace 'setup' based on conditions
def replace_setup(row):
    if '_' in row['train']:
        return replacement_dict_site.get(row['setup'], row['setup'])
    elif '-' in row['train']:
        return replacement_dict_pool.get(row['setup'], row['setup'])
    else:
        return replacement_dict_1train.get(row['setup'], row['setup'])

# rename dictionary
replacement_dict_site = {
    "mean_sitemean": "GMVSL1S",
    "age_sitemean": "PredL0SL1S",
    "l0_oos_from_test_sitemean": "OOSPredSL1S",
    "mean_pooled": "GMVPL1P",
    "age_combined_pooled": "PredL0SL1P",
    "l0_oos_from_test_pooled": "OOSPredSL1P",
}
replacement_dict_pool = {
    "mean_sitemean": "GMVPL1P",
    "age_sitemean": "PredL0PL1P",
    "l0_oos_from_test_sitemean": "OOSPredPL1P",
    "train+2folds": "GMVPL1Pext",
}
replacement_dict_1train = {
    "mean_sitemean": "GMV 1 train",
    "age_sitemean": "Pred 1 train",
    "l0_oos_from_test_sitemean": "OOS 1 train"
}
#%%
# load results
project_path = "/home/antogeo/codes/avg_vs_pred_stack/"
df = pd.read_csv(op.join(
    project_path, "result_files/dfs_scores/merged_results.csv"), 
    index_col=0
)

# add results from train+2folds, append main dataframe and remove w1d
# create df_plot which fascilitates plotting
extended_train = pd.read_csv(
    op.join(project_path, "result_files/dfs_scores/extended_train_all.csv"),
    index_col=0).groupby('Data').mean().reset_index()
extended_train.drop(['fold', 'w1d'], axis=1, inplace=True)
extended_train['setup'] = 'train+2folds'
extended_train.rename(columns={'Data': 'test'}, inplace=True)
extended_train['train'] = [ 'eNKI-IXI-CamCAN', 'eNKI-IXI-1000Gehirne',
                           'eNKI-CamCAN-1000Gehirne', 'IXI-CamCAN-1000Gehirne']
df = df.append(extended_train, ignore_index=True)

# Apply the custom function to the DataFrame
df['setup'] = df.apply(replace_setup, axis=1)

#%%
# make dataframe with results from More et al
More_etal = {
    "setup": ["More et al", "More et al", "More et al", "More et al",
              "More et al", "More et al", "More et al", "More et al",
              "More et al", "More et al", "More et al", "More et al",
              "More et al", "More et al", "More et al", "More et al"],
    "score": [4.57, 4.8, 4.4, 6.08,
              0.78, 0.89, 0.91, 0.79,
              -0.0765, -0.5233, -0.2558, -0.0803,
              0.9081, 0.9477, 0.9564, 0.9416],
    "metric": ["MAE", "MAE", "MAE", "MAE",
               "R2", "R2", "R2", "R2",
               "bias", "bias", "bias", "bias",
               "corr", "corr", "corr", "corr"],
}
df_More = pd.DataFrame(More_etal)
df_More_mean = df_More.groupby('metric', group_keys=False).mean().T

# df = df.append(df_More_mean, ignore_index=True)
# pd.concat([df, pd.DataFrame(df_train2folds).T], ignore_index=True)
#%%

# # edit names of different setups
# df["setup"].replace(
#     [
#         "mean_sitemean",
#         "age_sitemean",
#         "l0_oos_from_test_sitemean",
#         "mean_pooled",
#         "age_combined_pooled",
#         "l0_oos_from_test_pooled",
#     ],
#     [
#         "GMVSL1S",
#         "PredL0SL1S",
#         "OOSPredSL1S",
#         "GMVPL1P",
#         "PredL0SL1P",
#         "OOSPredSL1P",
#     ],
#     inplace=True,
# )
# #%%

df.drop('w1d', axis=1, inplace=True)
df_plot = df.melt(id_vars=["setup", "train", "test"],
                  value_vars=["MAE", "corr", "R2", "bias"])
df_plot.loc[df_plot["train"].isna(), 'number_train'] = "3"
df_plot.loc[df_plot["train"].isna(), 'train'] = "all"
df_plot.loc[(~df_plot["train"].str.contains("-") &
             ~df_plot["train"].str.contains("_")),
            'number_train'] = "1"
df_plot.loc[(~df_plot["train"].str.contains("-") &
             ~df_plot["train"].str.contains("_")),
            'pooled'] = 'both'
df_plot.loc[df_plot.train.str.count("_") == 2, 'number_train'] = "3"
df_plot.loc[df_plot.train.str.count("_") == 2, 'pooled'] = 'no'
df_plot.loc[df_plot.train.str.count("-") == 2, 'number_train'] = "3"
df_plot.loc[df_plot.train.str.count("-") == 2, 'pooled'] = 'yes'

df_plot.loc[df_plot.train.str.count("_") == 1, 'number_train'] = "2"
df_plot.loc[df_plot.train.str.count("_") == 1, 'pooled'] = 'no'
df_plot.loc[df_plot.train.str.count("-") == 1, 'number_train'] = "2"
df_plot.loc[df_plot.train.str.count("-") == 1, 'pooled'] = 'yes'
df_plot = df_plot.rename(columns={"variable": "metric", "value": "score"})


# %%
# Plots with three sites in training ordered by MAE.
# Grouped by setup
df_plot_3 = df_plot[df_plot.number_train == "3"]
# df_conc = pd.concat([df_three_train_sites, df_More], ignore_index=True)

order = (
    df_plot_3[df_plot_3.metric == "MAE"]
    .groupby("setup")["score"]
    .mean()
    .sort_values(ascending=True)
    .index.values
)

fig, axes = plt.subplots(2, 2, figsize=(15, 5))
fig.suptitle('Performance of setups', fontsize=12)
for i, val in enumerate(df_plot_3.metric.unique()):
    ax = sns.barplot(ax=axes[i // 2, i % 2],
                     data=df_plot_3[df_plot_3.metric == val],
                     y="score", x="metric", hue_order=order,
                     palette='pastel',
                     hue="setup", orient="v")
    ax.legend_.remove()
    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=10)

#%%
plt.figure()
plt.title("3 train - 1 test", fontsize=10)
ax = sns.barplot(data=df_conc, x="setup", y="score", hue="metric",
                 order=order, palette='pastel')
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1.05), ncol=4, title=None, frameon=False,
)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=.6)
ax.bar_label(ax.containers[0], fmt='%.2f', label_type='center', fontsize=6)
plt.xticks(rotation=45, fontsize=8)

# Plots with three sites in training ordered by MAE. Grouped by metric
order = (
    df_conc[df_conc.metric == "MAE"]
    .groupby("setup")["score"]
    .sum()
    .sort_values(ascending=True)
    .index.values
)

plt.figure()
plt.title("3 train - 1 test")
ax = sns.barplot(data=df_conc, y="score", x="metric", hue_order=order,
                 hue="setup", orient="v", palette="Set3", errwidth=1)

sns.set_style("whitegrid")
plt.show()
#%%
# Plots with three sites in training ordered by MAE.
# Here only pooled data were used. Grouped by setup

df_three_train_sites = df_plot[
    (df_plot.train.str.count("-") == 2) | (df_plot.train.isnull())]
df_conc = pd.concat([df_three_train_sites, df_More], ignore_index=True)

order = (
    df_conc[df_conc.metric == "MAE"]
    .groupby("setup")["score"]
    .mean()
    .sort_values(ascending=True)
    .index.values
)

plt.figure()
plt.title("3-pooled train - 1 test", fontsize=10)
ax = sns.barplot(data=df_conc, x="setup", y="score", hue="metric",
                 order=order, palette='pastel')
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1.05), ncol=4, title=None, frameon=False,
)
ax.bar_label(ax.containers[0], fmt='%.2f', label_type='center', fontsize=6)
plt.xticks(rotation=45, fontsize=10)

# Plots with three sites in training ordered by MAE. Grouped by metric
order = (
    df_conc[df_conc.metric == "MAE"]
    .groupby("setup")["score"]
    .sum()
    .sort_values(ascending=True)
    .index.values
)

plt.figure()
plt.title("3-pooled train - 1 test")
ax = sns.barplot(data=df_conc, y="score", x="metric", hue_order=order,
                 hue="setup", orient="v", palette="Set3", errwidth=1)

sns.set_style("whitegrid")

plt.show()
#%%
# plot metrics in subplots
sns.set(font_scale=1)
df_mult = df_plot[
    (df_plot.train.str.count("-") == 2) | (df_plot.train.isnull())]
df_conc = pd.concat([df_mult, df_More], ignore_index=True)
order = df_conc.groupby("setup")["score"].sum().sort_values(
                        ascending=True).index.values
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
fig.suptitle('multi train (pooled)- 1 test', fontsize=10)
for i, val in enumerate(df_conc.metric.unique()):
    ax = sns.barplot(ax=axes[i // 2, i % 2],
                     data=df_conc[df_conc.metric == val],
                     y="score", x="metric", hue_order=order,
                     palette='pastel',
                     hue="setup", orient="v")
    ax.legend_.remove()
    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=10)

    
    # axes[i // 2, i % 2].set_xticklabels(df_conc['metric'].unique()[i])
    # axes[i//2, i%2].set_title(df_conc['metric'].unique()[i])

# select those with 2 or 3 pooled train sets and order according to
# best MAE performance and plot grouped by metric
df_mult = df_plot[
    (df_plot.train.str.count("_") == 2) | (df_plot.train.isnull())]
df_conc = pd.concat([df_mult, df_More], ignore_index=True)
order = df_conc.groupby("setup")["score"].sum().sort_values(
                        ascending=True).index.values
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
fig.suptitle("multi train (separate sites)- 1 test", fontsize=10)
for i, val in enumerate(df_conc.metric.unique()):
    ax = sns.barplot(ax=axes[i // 2, i % 2],
                     data=df_conc[df_conc.metric == val],
                     y="score", x="metric", hue_order=order,
                     palette='pastel',
                     hue="setup", orient="v")
    ax.legend_.remove()
    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=10)
plt.show()
#%%
# make df for each training set case (pooled, single, double, triple)
single, double, triple, pooled = [], [], [], []
for train_set in df_plot.train.unique():
    if train_set is not np.nan:
        train_num = len(train_set.split("_"))
        if train_num == 1:
            if "-" in train_set:
                pooled.append(train_set)
            else:
                single.append(train_set)
        elif train_num == 2:
            double.append(train_set)
        elif train_num == 3:
            triple.append(train_set)

mae_pooled = df_plot[(df_plot.train.isin(pooled) * df_plot.metric == "MAE")][
    ["score", "setup", "metric"]
]
mae_pooled["train"] = "pooled"

mae_single = df_plot[(df_plot.train.isin(single) * df_plot.metric == "MAE")][
    ["score", "setup", "metric"]
]
mae_single["train"] = "single"

mae_double = df_plot[(df_plot.train.isin(double) * df_plot.metric == "MAE")][
    ["score", "setup", "metric"]
]
mae_double["train"] = "double"

mae_triple = df_plot[(df_plot.train.isin(triple) * df_plot.metric == "MAE")][
    ["score", "setup", "metric"]
]
mae_triple["train"] = "triple"

df_mae = pd.concat([mae_pooled, mae_single, mae_double, mae_triple], axis=0)

plt.figure()
plt.title("multi train (separate sites)- 1 test")
sns.barplot(data=df_mae, y="score", x="train", hue="setup", orient="v")
plt.legend(loc="best")
plt.show()


# print tables for comparisons

df_all = pd.DataFrame()
res = pd.DataFrame()
mae_pooled = pd.DataFrame()

df_toprint = df_plot[df_plot.train.isin(triple)]
for setup in df_toprint.setup.unique():
    res = (
        df_toprint[df_toprint.setup == setup]
        .drop(["setup", "train"], axis=1)
        .pivot(index="test", columns="metric")
        .rename_axis(None)
        .score
    )
    res["setup"] = setup
    res["pooled"] = "L0 preds"
    df_all = pd.concat([df_all, res], axis=0)
    print(setup)
    print(res, "\n")


pool3 = []
for train_set in df_plot.train.unique():
    if train_set is not np.nan:
        train_num = len(train_set.split("-"))
        if train_num == 3:
            pool3.append(train_set)
mae_pooled = pd.concat([mae_pooled, df_plot[df_plot.train.isin(pool3)]])
mae_pooled["train"] = "pooled"

print("pool 3 datasets before L0 models")

for setup in mae_pooled.setup.unique():
    res = (
        mae_pooled[mae_pooled.setup == setup]
        .drop(["setup", "train"], axis=1)
        .pivot(index="test", columns="metric")
        .rename_axis(None)
        .score
    )
    res["setup"] = setup
    res["pooled"] = "in advance"
    df_all = pd.concat([df_all, res], axis=0)
    print(setup)
    print(res, "\n")
df_all.to_csv("all_3train_res.csv")

# %%
