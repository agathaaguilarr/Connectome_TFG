# =================================================================
# =================================================================
# Utility WholeBrain to compute multi-views of the cortex data
# =================================================================
# =================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from plot3DBrain import plotColorView

# =================================================================
# plots the 6-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
#           L-flat,         R-flat
# =================================================================
def multiview6(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):

    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(3, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)
    ax = plt.subplot(3, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)

    # ================== flatmaps
    ax = fig.add_subplot(3, 2, 5)  # left hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'L-flat', cmap=leftCmap, **kwds)
    ax = fig.add_subplot(3, 2, 6)  # right hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'R-flat', cmap=rightCmap, **kwds)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()

# =================================================================
# Functions to plot a 5-view plot, either "star"-shaped or linear
# =================================================================
def finish_multiview5(fig, cmap, display=True, savePath=None, **kwds):
    # ============= Adjust the sizes
    plt.subplots_adjust(left=0.0, right=0.8, bottom=0.0, top=1.0, wspace=0, hspace=0)

    # ============= now, let's add a colorbar...
    # first we normalize
    if 'norm' not in kwds: # if data is not normalized, we do it as the 1st step
        vmin = np.min(data['func_L']) if 'vmin' not in kwds else kwds['vmin']
        vmax = np.max(data['func_L']) if 'vmax' not in kwds else kwds['vmax']
        norm = Normalize(vmin=vmin, vmax=vmax)
    else: # if the data is normalized, the information is extracted from kwds
        norm = kwds['norm']

    PCM = plt.cm.ScalarMappable(norm=norm, cmap=cmap) #this creates the ScalarMappable (connects the normalized data values to dhe colormap)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # this parameter is the dimensions [left, bottom, width, height] of the new axes
    fig.colorbar(PCM, cax=cbar_ax) # adds the color bar to the figure
    # ============ and show!!!
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# multiview5_linear:
#     lh-lateral, rh-lateral, l/r-superior, lh-medial, rh-medial
# =================================================================
def multiview5_linear(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                      suptitle='', figsize=(12, 3), display=True, savePath=None, **kwds):

    # Initialize the Figure and Subplots --> create a figure with 1 row and 5 columns of subplots
    fig, axs = plt.subplots(1, 5, figsize=figsize)

    # plot the left hemisphere views
    plotColorView(axs[0], cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds) #first column (axs[0]): Left hemisphere lateral view
    plotColorView(axs[1], cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds) #second column (axs[1]): Left hemisphere medial view

    # plot superior views --> third column (axs[2]): Combines left and right superior views (whole brain top view)
    plotColorView(axs[2], cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axs[2], cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)

    # plot the right hemisphere views
    plotColorView(axs[3], cortex, data, numRegions, 'Rh-lateral', cmap=cmap, **kwds) #fourth column (axs[3]): Right hemisphere lateral view
    plotColorView(axs[4], cortex, data, numRegions, 'Rh-medial', cmap=cmap, **kwds) #fifth column (axs[4]): Right hemisphere medial view

    return fig

# =================================================================
# plots a 5-view star-shaped plot:
#           lh-lateral,               rh-lateral,
#                       l/r-superior,
#           lh-medial,                rh-medial
# =================================================================
def multiview5_star(cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):

    # Initialize the Figure and Subplots --> creates a 2x3 grid of subplots.
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    plotColorView(axs[0,0], cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds) # axs[0,0]: Top-left, left hemisphere lateral view
    plotColorView(axs[1,0], cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds) # axs[1,0]: Bottom-left, left hemisphere medial view
    plotColorView(axs[0,2], cortex, data, numRegions, 'Rh-lateral', cmap=cmap, **kwds) # axs[0,2]: Top-right, right hemisphere lateral view
    plotColorView(axs[1,2], cortex, data, numRegions, 'Rh-medial', cmap=cmap, **kwds) # axs[1,2]: Bottom-right, right hemisphere medial view
    # === L/R-superior

    # merge the central column of subplots (axs[:,1]) into a single large axis for superior views
    gs = axs[0, 1].get_gridspec()
    # remove the underlying axes
    for ax in axs[:,1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:,1])
    # plot left and right superior views on the merged central axis (whole brain top view)
    plotColorView(axbig, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axbig, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)

    return fig


def multiview5(cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, linear=False, **kwds):
    if not linear: # default mode, if linear = false
        fig = multiview5_star(cortex, data, numRegions, cmap=cmap, suptitle=suptitle, **kwds)
    else: # if linear = true
        fig = multiview5_linear(cortex, data, numRegions, cmap=cmap, suptitle=suptitle, **kwds)
    finish_multiview5(fig, cmap, display=display, savePath=savePath, **kwds)


# =================================================================
# plots the 4-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
# =================================================================
def multiview4(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(2, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)
    ax = plt.subplot(2, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots a left/Right-view plot:
#                       l-medial, l-lateral
# =================================================================
def leftRightView(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                  suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(1, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots a top-view plot:
#                       l/r-superior, and all the others!!!
# =================================================================
def plot_ViewAx(ax, cortex, data, numRegions, view,
                cmap=plt.cm.coolwarm,
                suptitle='', **kwds):
    if view == 'superior':  # this is 'L-superior' + 'R-superior'
        plotColorView(ax, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
        plotColorView(ax, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    else:  # this is 'Lh-medial' / 'Lh-lateral' / 'Rh-medial' / 'Rh-lateral' / 'L-flat' / 'R-flat'
        plotColorView(ax, cortex, data, numRegions, view, suptitle=suptitle, cmap=cmap, **kwds)
    if suptitle == '':
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)


def plot_View(cortex, data, numRegions, view,
              figsize=(15, 10), display=True, savePath=None, **kwd):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    plot_ViewAx(ax, cortex, data, numRegions, view, **kwd)
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots top views for multiple vectors... Now, only the top view.
# All plots have the same limits and use the same colorbar
# =================================================================
def plot_ValuesForAllCohorts(burdens, crtx, cmap, view, orientation='h'):
    if orientation == 'h': sizeX, sizeY = 1, len(burdens)+1
    else: sizeX, sizeY = len(burdens), 1
    fig, axs = plt.subplots(sizeX, sizeY,
                            # we add an extra row to solve a strange bug I found, where the last panel dpes not show the ticks, and do not have the patience to fix
                            gridspec_kw={'wspace': 0.2, 'hspace': 0.2})
    num_regions = len(burdens[list(burdens.keys())[0]])
    vmin = np.min([np.min(burdens[c]) for c in burdens])
    vmax = np.max([np.max(burdens[c]) for c in burdens])
    for c, cohort in enumerate(burdens):
        vect = burdens[cohort]
        data = {'func_L': vect, 'func_R': vect}
        plot_ViewAx(axs[c], crtx, data, num_regions, view, vmin=vmin, vmax=vmax,
                  cmap=cmap, suptitle=cohort, fontSize=15)
    norm = Normalize(vmin=vmin, vmax=vmax)
    PCM = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.77, 0.3, 0.02, 0.4])  # This parameter is the dimensions [left, bottom, width, height] of the new axes.
    fig.colorbar(PCM, cax=cbar_ax)
    fig.tight_layout()
    plt.show()


# =================================================================
# functions to plot multiple mutiview5 plots for different sets,
# but all with a common normalization and a common colorbar
# =================================================================
# This one plots a single multiview5 plot
def plot_multiview5Values(obs, crtx, title, fileName, display, cmap, norm):
    # crtx = setUpGlasser360_cortex()
    # =============== Plot!!! =============================
    data = {'func_L': obs, 'func_R': obs}
    multiview5(crtx, data, 360, cmap, suptitle=title, lightingBias=0.1, mode='flatWire', shadowed=True,
               display=display, savePath=fileName+'.png', norm=norm)


# plots multiple multiview5 plots
def plot_multiview5ValuesForEachChort(burdens, crtx, title, metaName, display, cmap, path):
    vmin = np.min([np.min(burdens[c]) for c in burdens])
    vmax = np.max([np.max(burdens[c]) for c in burdens])
    norm = Normalize(vmin=vmin, vmax=vmax)
    for cohort in burdens:
        fullFileName = path + cohort + metaName
        plot_multiview5Values(burdens[cohort], crtx, title, fullFileName, display, cmap, norm)


# =================================================================
# ================================= module test code
if __name__ == '__main__':
    from matplotlib import cm

    from project3DBrain import set_up_Glasser360_cortex
    crtx = set_up_Glasser360_cortex()

    # =============== Plot!!! =============================
    testData = np.arange(0, 360)
    data = {'func_L': testData, 'func_R': testData}
    # testColors = cm.cividis
    testColors = cm.YlOrBr

    multiview5(crtx, data, 360, testColors,
               linear=True,
               lightingBias=0.1, mode='flatWire', shadowed=True)  # flatWire / flat / gouraud

# ======================================================
# ======================================================
# ======================================================EOF
