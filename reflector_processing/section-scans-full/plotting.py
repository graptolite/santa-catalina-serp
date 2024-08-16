import numpy as np
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")
from scipy.spatial import KDTree
import json

from util_funcs import *

# Conversion between pixels to mm.
pix2mm = 1/1000

##### Loading functions designed to be run after non-volatile data sourcces are established. #####
##### ----- #####

def load_filtered_contours(sample,file_prepend):
    ''' Load contour definitions for the relevant processing pathway.

    sample (str) : sample name/id
    file_prepend (str) : string prepended to files to distinguish the specific processing pathway

    returns contours : numpy array of "small" contours
    larger_contours : numpy array of "large" contours
    '''
    contours = np.load(f"filtered_data/{file_prepend}-{sample}.png.npy",allow_pickle=True)
    larger_contours = np.load(f"filtered_data/{file_prepend}-{sample}.png-larger.npy",allow_pickle=True)
    return contours,larger_contours

def load_areas(sample,file_prepend):
    ''' Load contour definitions for the relevant processing pathway.

    sample (str) : sample name/id
    file_prepend (str) : string prepended to files to distinguish the specific processing pathway

    returns patch_areas (list of numericals) : list of reflector grain areas
    '''
    # Load full data file for the specified processing pathway.
    with open(file_prepend+"-areas.json") as infile:
        data = json.load(infile)
    # Isolate the patch areas.
    patch_areas = data[sample]["patch_areas"]
    return patch_areas

##### ----- #####
class Plotter():
    # Not the most streamlined class.
    def __init__(self,alteration_degree,alteration_desc):
        # Degree of alteration assigned to each section.
        self.alteration_degree = alteration_degree
        # Text descriptions for each level of alteration.
        self.alteration_desc = alteration_desc

    def plot_all(self,plotting_func,file_prepend,figsize=(24,24)):
        ''' Iterate through all the samples described in `alteration_degree` above and plot some property/properties of them in a gridded view where columns are for sections with shared degree of alteration.

        plotting_func (function) : function that takes (fig, subfig, sample, file_prepend, alteration_idx) as input and produces a plot for each sample
        file_prepend (str) : string prepended to files to distinguish the specific processing pathway
        figsize (list-like of 2 numericals) : figure dimensions

        returns fig : figure with everything plotted on
        '''
        fig = plt.figure(figsize=figsize)
        # Samples and their degree of alteration from the static/hardcoded source in this code.
        samples,alterations = list(zip(*self.alteration_degree.items()))
        # The number of different alteration indices is the number of columns.
        n_cols = len(set(alterations))
        # The number of times the most common alteration index appears is the number of rows. Not all rows will be populated.
        n_rows = alterations.count(max(alterations,key=alterations.count))
        # Create grid specification that has a number of columns equal to the number of different types of alteration and a number of rows equal to the number of samples in the most populated degree of alteration.
        gs_0 = fig.add_gridspec(n_rows,n_cols)
        # Iterate through the different degrees of alteration.
        for alteration_idx in range(n_cols):
            # Select samples with the active degree of alteration.
            samples = sorted([k for k,v in self.alteration_degree.items() if v==alteration_idx])
            # Iterate through each sample.
            for i,sample in enumerate(samples):
                # Create a subfigure in the correct location on the toplevel figure.
                subfig = gs_0[i,alteration_idx]
                # Perform plotting for the data relevant to the active sample on this subfigure.
                plotting_func(fig,subfig,sample,file_prepend,alteration_idx)
        return fig

    ##### Graph options - plotting functions matching the specification described by `plot_all` #####
    # Input Parameters:
    # - fig : top level figure
    # - subfigure : subfigure on which to add an axis/axes for plotting sample-specific data
    # - file_prepend : descriptor for the processing pathway
    # - alteration_index : index describing degree of alteration of the sample

    def area_vs_nn_dist(self,fig,subfig,sample,file_prepend,alteration_idx):
        ''' Plot reflector area against nearest neighbour distance.

        Note: the files containing areas and contours must have matching-ordered data.
        '''
        # Create grid specification for the subfigure.
        gs = subfig.subgridspec(1,2,
                                width_ratios=[6,1],
                                hspace=0.1,
                                wspace=0.1)
        # Load large contours.
        _,larger_contours = load_filtered_contours(sample,file_prepend)
        # Convert data into a format suitable for conversion into shapely Polygons.
        larger_contours = [c[:,0,:] for c in larger_contours]
        # Compute centroids of each reflector patch and scale the coordinate system to 1 px = 1 micron.
        centroids = [np.array(Polygon(c).centroid.xy).T[0]/2 for c in larger_contours]

        # Load reflector patch areas.
        areas = load_areas(sample,file_prepend)
        # Construct KD tree using centroids for nearest-neighbour searching.
        tree = KDTree(centroids)
        # Compute nearest-neighbour distances for each reflector's centroid.
        distances = [tree.query(c,2)[0][1]*pix2mm for c in centroids]

        # Scatterplotting area vs nearest neighbour distance.
        ax = fig.add_subplot(gs[0,0])
        ax.scatter(areas,distances)
        ax.set_xlabel(f"Area /mm$^2$")
        ax.set_ylabel(f"Distance to nearest neighbour /mm")
        # Hardcoding limits to permit visual comparison between plots.
        # Note: some data will be cut off.
        ax.set_xlim([0,0.05])
        ax.set_ylim([0,1])
        # Plotting histogram of nearest neighbour distances.
        ax1 = fig.add_subplot(gs[0,1],sharey=ax)
        ax1.tick_params(labelleft=False)
        ax1.hist(distances,bins=100,orientation="horizontal")
        ax1.set_xlabel("Count")

        # Label each plot with the sample it applies to, and if the plot is at the top of a top-level column, label also the degree of alteration.
        if subfig.is_first_row():
            ax.set_title(f"%s altered\n{sample}" % self.alteration_desc[alteration_idx])
        else:
            ax.set_title(f"{sample}")
        return

    def area_distros(self,fig,subfig,sample,file_prepend,alteration_idx):
        ''' Plot reflector area distributions, as well as a exponential fit to the distribution.
        '''
        # Setup a single subfigure axis.
        gs = subfig.subgridspec(1,1)
        ax = fig.add_subplot(gs[0])

        # Hardcode the bin definition.
        max_area = 0.05 # mm2
        bins = np.linspace(0,max_area,100)
        # Compute bin midpoints.
        midpoints = (bins[:-1] + bins[1:])/2

        # Plot area distribution as histogram and return the counts per bin.
        counts,_,_ = ax.hist(load_areas(sample,file_prepend),bins)
        # Find the fit parameters to the distribution assumimg an exponential distribution.
        popt = fit_exp_log_y(midpoints,counts)
        # Plot the fit.
        ax.plot(midpoints,10**exp_func(midpoints,*popt))
        # Describe the fit on the plot (top right).
        ax.text(1,1,"$Count = 10**(%.2f \cdot \exp(%.2f \cdot Area))$" % tuple(popt),transform=ax.transAxes,ha="right",va="top")
        ax.set_yscale("log")
        ax.set_xlabel(f"Area /mm$^2$")
        ax.set_ylabel("Count")
        # Hardcode the x limit.
        ax.set_xlim([0,max_area])

        # Label each plot with the sample it applies to, and if the plot is at the top of a top-level column, label also the degree of alteration.
        if subfig.is_first_row():
            ax.set_title(f"%s altered\n{sample}" % self.alteration_desc[alteration_idx])
        else:
            ax.set_title(f"{sample}")
        return

    def aspect_ratio_distros(self,fig,subfig,sample,file_prepend,alteration_idx):
        ''' Plot aspect ratio distributions.
        '''
        # Setup a single subfigure axis.
        gs = subfig.subgridspec(1,1)
        ax = fig.add_subplot(gs[0])

        # Load large contours.
        _,larger_contours = load_filtered_contours(sample,file_prepend)
        # Extract minimum bounding box dimensions for the reflector patch (outer) contours.
        dimensions = get_dimensions(larger_contours)
        # Extract short axes.
        short_axes = list(map(min,dimensions))
        # Extract long axes.
        long_axes = list(map(max,dimensions))
        # Compute grain aspect ratios.
        aspect_ratios = np.array(long_axes)/np.array(short_axes)
        # Plot histogram.
        ax.hist(aspect_ratios,bins=100,
                histtype="step",edgecolor="k",label="Short")
        # Axes labelling.
        ax.set_xlabel("Aspect Ratio")
        ax.set_ylabel("Count")
        # Set y axis to log scale.
        ax.set_yscale("log")
        # Hardcode x limit.
        ax.set_xlim([0,20])

        # Label each plot with the sample it applies to, and if the plot is at the top of a top-level column, label also the degree of alteration.
        if subfig.is_first_row():
            ax.set_title(f"%s altered\n{sample}" % self.alteration_desc[alteration_idx])
        else:
            ax.set_title(f"{sample}")
        return
