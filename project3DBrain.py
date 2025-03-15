# from scipy.spatial import cKDTree
# import nibabel as nib
#
# # ==========================================================================
# # Important config options: filenames
# # ==========================================================================
#
# import geometric as geometric
#
# # ===========================
# #  Convenience function for the Glasser parcellation, for debug purposes only...
# # ===========================

from scipy.spatial import cKDTree
import nibabel as nib
import os
import geometric as geometric

def set_up_Glasser360_cortex(base_folder):
    glasser_folder = os.path.join(base_folder, "glasser360")

    glassers_L = nib.load(os.path.join(glasser_folder, 'Glasser360.L.mid.32k_fs_LR.surf.gii'))
    glassers_R = nib.load(os.path.join(glasser_folder, 'Glasser360.R.mid.32k_fs_LR.surf.gii'))
    flat_L = nib.load(os.path.join(glasser_folder, 'Glasser360.L.flat.32k_fs_LR.surf.gii'))
    flat_R = nib.load(os.path.join(glasser_folder, 'Glasser360.R.flat.32k_fs_LR.surf.gii'))
    mapL = nib.load(os.path.join(glasser_folder, 'fsaverage.L.glasser360_fs_LR.func.gii')).agg_data()
    mapR = nib.load(os.path.join(glasser_folder, 'fsaverage.R.glasser360_fs_LR.func.gii')).agg_data()

    return {
        'model_L': glassers_L, 'model_R': glassers_R,
        'flat_L': flat_L, 'flat_R': flat_R,
        'map_L': mapL, 'map_R': mapR
    }

def set_up_cortex(coordinates, base_folder):
    brain_3D_directory = os.path.join(base_folder, "brain_map")

    flat_L = nib.load(os.path.join(brain_3D_directory, 'L.flat.32k_fs_LR.surf.gii'))
    flat_R = nib.load(os.path.join(brain_3D_directory, 'R.flat.32k_fs_LR.surf.gii'))
    model_L = nib.load(os.path.join(brain_3D_directory, 'L.mid.32k_fs_LR.surf.gii'))
    model_R = nib.load(os.path.join(brain_3D_directory, 'R.mid.32k_fs_LR.surf.gii'))

    return {
        'map_L': geometric.findClosestPoints(coordinates, model_L.darrays[0].data)[0].flatten(),
        'map_R': geometric.findClosestPoints(coordinates, model_R.darrays[0].data)[0].flatten(),
        'flat_L': flat_L, 'flat_R': flat_R,
        'model_L': model_L, 'model_R': model_R
    }
