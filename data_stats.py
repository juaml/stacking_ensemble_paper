#%%
import pandas as pd
import os.path as op

ds_path = '/home/antogeo/codes/avg_vs_pred_stack/data'
df_ixi = pd.read_csv(op.join(ds_path, 'ixi.subject_list_cat12.8.csv'), index_col=0)
df_enki = pd.read_csv(op.join(ds_path, 'enki.subject_list_cat12.8.csv'), index_col=0)
df_1000br = pd.read_csv(op.join(ds_path, '1000brains.subject_list_cat12.8.csv'), index_col=0)
df_cc = pd.read_csv(op.join(ds_path, 'camcan.subject_list_cat12.8.csv'), index_col=0)
# %%

df_ixi.describe().iloc[1:4]
df_ixi.describe().iloc[7]
df_enki.describe().iloc[1:4]
df_enki.describe().iloc[7]
df_1000br.describe().iloc[1:4]
df_1000br.describe().iloc[7]
df_cc.describe().iloc[1:4]
df_cc.describe().iloc[7]
# %%
