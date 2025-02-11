#%%
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_path = "/home/antogeo/codes/avg_vs_pred_stack/"
df = pd.read_csv(op.join(
    project_path, "result_files/dfs_scores/merged_results.csv"), index_col=0
)
#%%
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
#         "parcels` mean",
#         "age pred\nper site",
#         "l0 oos-test\nl1 per site",
#         "pool data for\nl0 & l1 ",
#         "l0 per site\nl1 pooled",
#         "l0 oos-test\nl1 pooled ",
#     ],
#     inplace=True,
# )

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
    "l0_oos_from_test_sitemean": "OOSPredPL1P"
}
replacement_dict_1train = {
    "mean_sitemean": "GMV 1 train",
    "age_sitemean": "Pred 1 train",
    "l0_oos_from_test_sitemean": "OOS 1 train"
}
#%%
# Custom function to replace 'setup' based on conditions
def replace_setup(row):
    if '_' in row['train']:
        return replacement_dict_site.get(row['setup'], row['setup'])
    elif '-' in row['train']:
        return replacement_dict_pool.get(row['setup'], row['setup'])
    else:
        return replacement_dict_1train.get(row['setup'], row['setup'])


# Apply the custom function to the DataFrame
df['setup'] = df.apply(replace_setup, axis=1)
#%%
df_plot = df.melt(id_vars=["setup", "train", "test"],
                  value_vars=["MAE", "corr", "R2", "bias", "w1d"])

df_plot = df_plot.rename(columns={"variable": "metric", "value": "score"})
# remove w1d metric
df_plot = df_plot[df_plot.metric != "w1d"]

#%%
# add colum with number of train sets
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

#%%
# order = (
#     df_plot[df_plot.metric == "MAE"]
#     .groupby("setup")["score"]
#     .mean()
#     .sort_values(ascending=True)
#     .index.values
# )
order = ['GMVPL1P',  'GMVSL1S', 'GMV 1 train', 
         'OOSPredPL1P', 'OOSPredSL1S', 'OOSPredSL1P', 'OOS 1 train',
         'PredL0PL1P', 'PredL0SL1S', 'PredL0SL1P', 'Pred 1 train' ]
fig, axes = plt.subplots(2, 2, sharex=False, figsize=(16, 8))
fig.suptitle('impact of # of training sets')

lines = []
labels = []
for i, (metric, ax) in enumerate(zip(["MAE", "corr", "R2", "bias"],
                                     axes.flat)):
    mae_plot = df_plot[df_plot.metric == metric]
    mae_plot['number_train'] = mae_plot['number_train'].astype(int)
    sns.barplot(ax=ax, hue='number_train', y='score', x='setup',
                data=mae_plot, palette='Set2', order=order)
    # pt.RainCloud(ax=ax, hue='number_train', y='score', x='setup',
    #              width_viol=0.3, bw=.1, orient='h',
    #              data=mae_plot, palette='Set2', order=order)
    plt.xticks(rotation=45)
    ax.set_title(metric)
    ax.tick_params(axis='x', labelrotation=45)
    ax.get_legend().remove()
Line, Label = ax.get_legend_handles_labels() 
# print(Label) 
lines.extend(Line) 
labels.extend(Label) 
fig.legend(lines, labels, loc="upper right") 
plt.show()
