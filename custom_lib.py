import nibabel as nib
import numpy as np
import pandas as pd


def make_nii(output_name, reg_values, contain_zero=False):
    """Takes 873 values representing corresponding to the values to be assigned
    in the BSF873 atlas. The values provided start to fill the nifti regions
    in an ancential way
        Parameters:
                output_name (string): path to the output nifti
                reg_values (pandas series): values to be assigned to the nifti
                contain_zero (bool): whether to include the zero value (outer space)
    """
    if contain_zero:
        a = np.nan
        pd.Series(a)
        reg_values = a.append(reg_values).values
    atlas_img = nib.load("data/BSF_873.nii")
    atlas_flat = np.round(atlas_img.get_fdata().reshape(-1)).astype(np.int)
    all_values = np.unique(atlas_flat)
    values = [s for s in all_values if s != 0]
    values = values[1:]
    # initialize mem
    mat1D = np.zeros(len(atlas_flat), dtype=float)
    for cntr, value in enumerate(values):
        # print(value, cntr)
        mat1D[atlas_flat == value] = reg_values[cntr]
    mat3D = mat1D.reshape(atlas_img.dataobj.shape)
    affine = atlas_img.affine
    empty_header = nib.Nifti1Header()
    stat_map_vol = nib.Nifti1Image(mat3D, affine, empty_header)
    nib.save(stat_map_vol, output_name)
