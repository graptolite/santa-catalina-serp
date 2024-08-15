import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,find_peaks
from shapely import intersection,LineString,Point
from PIL import Image,ImageFilter
from io import StringIO
from datetime import datetime
import re
import os

#########################
# Testing options START #
#########################
# n_sigma = 2
# n_selvages = 1
# element = "Al"
# elements = ["Al","Ca","Cr","K","Na","Ni","Si","Ti","C","Mg","Fe"]
# study_area = data_folder = "/home/generic/University/4th-year-project/EDS-maps-Peter-Lindquist/M02/M02-area2/"
# vein_id = "silica-vein-M02-area2.txt"
# vein_window = [-56, 32]
# bkg_windows = [[[4062.24, 4419.36],[3024.36, 2589.12]]]
# margin_window = None #[-257,284]
#######################
# Testing options END #
#######################

#####################
# Utility Functions #
#####################

def linear_func(x,a,b):
    ''' Generic linear function for fitting purposes.
    '''
    return x * a + b

def nearest_in_list(l,x,n=1):
    ''' Find element(s) in a numerical list whose value is (are) closest to a desired value.

    l | :list:-like [Numerical] | List to search in.
    x | Numerical | Value to find the closest in a list to.
    n | :int: | Number of closest-matching values to find, with the found values returned in order of misfit to `x`.

    Returns: :list:
    '''
    # Compute absolute distances ("misfit") from each element in the list to the target number.
    dist = np.abs(np.array(l) - x)
    # Order the list by misfit to the target number.
    _,s_l = zip(*sorted(zip(dist,l)))
    # Return the closes `n` values to the target number.
    nearest = s_l[:n]
    return nearest

def find_rect_mean(rects,pil_image):
    ''' Compute the mean of pixels within rectangle windows in an indexed ("greyscale") image. Pixels are extracted from each rectangle without removal of intersections/repeats.

    rects | :list: [([x0,y0],[x1,y1])] | List of rectangular window specifications (in [x0,y0],[x1,y1] opposing corners format) to find the overall mean (of pixels contained within) from. May have a length of 1 of only 1 rectangle is of interest. The specific opposing-corner vertices pair doesn't matter.
    pil_image | :Pillow.Image: | Pillow image object of EDS map.

    Returns: Numerical
    '''
    # Convert image to an array of [0,255] pixel values.
    image_data = np.array(pil_image.convert("I"))
    # Initialize list to store pixel values extracted from the rectangle(s).
    bkg = []
    # Iterate through rectangles.
    for rect in np.array(rects).astype(int):
        # Extract opposing-corner vertices of the rectangle.
        p1,p2 = rect[0],rect[1]
        # Determine the edges from the opposing-corner coordinates.
        xmin = min(p1[0],p2[0])
        ymin = min(p1[1],p2[1])
        xmax = max(p1[0],p2[0])
        ymax = max(p1[1],p2[1])
        # Extract pixel values from the crop rectangles and store.
        bkg.extend(image_data[ymin:ymax,xmin:xmax].flatten())
    # Find the overall mean of aggregated values from the all crop rectangles.
    rect_mean = np.mean(bkg)
    return rect_mean

def plot_deviations_image(ax,img,center_val):
    ''' Compute and plot (as an image) a 2D array of values representing deviations from a central value. Uses the red (negative) white (zero/center) blue (positive) colormapping.

    ax | :matplotlib.axes.Axes: | Axis to plot the image on.
    img | :Pillow.Image: | Base image whose pixels are to be "recentered". The image doesn't need to be in indexed mode (e.g. can be RGB), but will be converted to indexed mode for local use by this function.
    center_val | Numerical | Central value to find pixel deviations from.

    Returns: (:matplotlib.figure.Figure:,:matplotlib.axes.Axes:)
    '''
    # Convert image to number array in indexed format (pixels represented by number in range [0,255]).
    img_arr = np.array(img.convert("I"))
    # Subtract "background" mean from all pixel values in the image to get an image of deviations.
    img_arr = img_arr-center_val
    # Find maximum absolute deviation from the mean, which will be used to symmetricize the colorbar and ensures zero is colormapped to white.
    max_abs_extent = max(img_arr.max(),abs(img_arr.min()))
    # Display the image with a blue (negative) white (zero) red (positive) colormap.
    img = ax.imshow(img_arr,cmap="bwr",vmin=-max_abs_extent,vmax=max_abs_extent)
    # Extract dimensions of image.
    width_y,width_x = img_arr.shape
    # Crop the plot to the image extent.
    ax.set_xlim([0,width_x])
    ax.set_ylim([0,width_y])
    # Display and label colorbar.
    cbar = plt.colorbar(img,ax=ax)
    cbar.set_label(f"Observed value - background (={int(center_val)})")
    return

def safe_crop(img,crop):
    ''' Crop Pillow image using a ranges-defined rectangle, ensuring the crop rectangle doesn't exceed image boundaries. Returns the cropped image as well as the applied crop specification.

    img | :Pillow.Image: | Pillow image to crop.
    crop | :list: [<xmin>,<xmax>,<ymin>,<ymax>] | Crop region.

    Returns: :Pillow.Image:,:list: [<xmin>,<xmax>,<ymin>,<ymax>]
    '''
    # Get image dimensions that can't be exceeded.
    w,h = img.size
    # Separate axis-specific cropping ranges.
    x_crop_range = np.clip(np.array(crop[:2]),0,w)
    y_crop_range = np.clip(np.array(crop[2:]),0,h)
    # Convert cropping coordinates to coordinates relative to the Pillow image, ensuring they do not exceed the image bounds.
    x_min = min(x_crop_range)
    x_max = max(x_crop_range)
    y_max = h-min(y_crop_range)
    y_min = h-max(y_crop_range)
    # Crop image.
    img = img.crop((x_min,y_min,x_max,y_max))
    return img,list(np.concatenate([x_crop_range,y_crop_range]))

def slice_by_x_window(x,y,x_window):
    ''' Slice x,y coordinates by an x coordinate window into 3 groups: left of x_window, in x_window, right of x_window. There will be an overlap on 1 index by extending the upper end of the slices.

    x | :list:-like | x coordinates. Must be monotonic.
    y | :list:-like | y coordinates.
    x_window | :list: [<x edge 1>,<x edge 2>] | Window of x coordinates to slice the coordinates using.

    Returns: (:list: [:pd.Series:],:list: [:pd.Series:])
    '''
    # Cast x coordinates to pandas series for index finding.
    x = pd.Series(x)
    # Declare function to find the coordinate index corresponding to the closts-matching data x to an input x.
    nearest_idx = lambda val : x.index[x==nearest_in_list(x,val)][0]
    # Find the coordinate indexes for the two window edges.
    edge_idxs = [nearest_idx(x_edge) for x_edge in x_window]
    # Get full list of edges required for coordinate slicing.
    slice_idxs = [0] + sorted(edge_idxs) + [len(x)]
    # Store coordinates in a pandas dataframe for easier slicing.
    coords_df = pd.DataFrame({"x":x,
                              "y":y})
    # Slice coordinates into 3 coordinate groups.
    # There will be an overlap on 1 index on the upper end of the slices.
    sliced_dfs = [coords_df.iloc[a:np.clip(slice_idxs[i+1]+1,0,len(x))] for i,a in enumerate(slice_idxs[:-1])]
    # Separate x and y coordinates.
    x_segments,y_segments = ([df[c] for df in sliced_dfs] for c in ["x","y"])
    return x_segments,y_segments

def denoise_signal(signal):
    ''' Remove some high frequency noise from a signal by passing it through a relatively long-range, second-order savgol filter. Bin width (width of data to fit to the P2) is ~10% of the signal length. Returns the denoised signal and the absolute width of the filter bin.

    Savitzky 2012 https://pubs.acs.org/doi/abs/10.1021/ac00190a744: designed for the "recognition of spectral characteristics such as peaks, valleys"
    Note: first order filter produced bad results, so second order used instead.

    signal | :list:-like | Numerical data array.

    Returns: (:list:-like,:int:)
    '''
    # Compute absolute bin width for savgol filter application.
    savgol_bin_width = int(len(signal)/10)
    # Apply savgol filter to partially denoise the signal.
    signal = savgol_filter(signal,savgol_bin_width,polyorder=2)
    return signal,savgol_bin_width

def load_gmt_profile(element,vein_id):
    ''' Parse a GMT profile of an element and profile specification (with hardcoded file specification) into a pandas dataframe.

    element | :str: | Element of interest.
    vein_id | :str: | Profile filename identifier.

    Returns: :pandas.DataFrame:
    '''
    # Construct filename to GMT profile output path.
    datafile = f"tmp/{element}-{vein_id}-stack.dat"
    # Parse the profile file into a pandas dataframe with suitable column names.
    df = pd.read_csv(datafile,delimiter="\s+",names=["distance","stack_val","deviation","min","max","uncert_lower","uncert_higher"])
    return df

def find_bkg_mean(data_folder,element,bkg_windows):
    ''' Find the background mean of EDS pixels from specified background area windows for a specified element in a specified sample region.

    data_folder | :str: | Path to folder containing EDS maps for a sample region.
    element | :str: | Element of interest.
    bkg_windows | :list: [([x0,y0],[x1,y1])] | List of rectangular window specifications (in [x0,y0],[x1,y1] opposing corners format) to find the overall mean (of pixels contained within) from. May have a length of 1 of only 1 rectangle is of interest. The specific opposing-corner vertices pair doesn't matter.
    '''
    # Load image for the sample region and element of interest.
    img = Image.open(f"{data_folder}/{element}.tiff")
    # Compute the mean of pixels within the rectangle collection.
    mean_val = find_rect_mean(bkg_windows,img)
    return mean_val

#########################################
# Signal Processing and Display Classes #
#########################################

class SignalExtrema():
    ''' Extrema-related methods for 2D (x,y) data.
    '''
    def __init__(self,x,y,prominence=0.3,n_sigma=1):
        '''
        x | :list:-like | x coordinates of data.
        y | :list:-like | y coordinates of data - the signal.
        prominence | :float: | Critical prominence for extrema (turning point) detection.
        n_sigma | :int: | Number of standard deviations away from the mean of a collection of extrema before signal values are treated as being generally more extreme (superextreme) than the extrema collection.
        '''
        self.x = x
        self.signal = y
        # Detect peaks and troughs in the data (and also compute their respective intergroup mean and standard deviation).
        self.peaks_and_troughs = {direction:self.extrema_mean_stdev(direction,prominence) for direction in [1,-1]}
        # After finding this peaks and troughs data...
        # Compute the thresholds for superextremity.
        self.superextreme_thresholds = self.find_thresholds(n_sigma)

    def extrema_mean_stdev(self,direction=1,prominence=0.3):
        ''' Find the mean and standard deviation of detected peaks (with prominence greater than input), as well as the indices of all detected peaks.

        signal | :pandas.Series: or :list: [:pandas.Series:] | Numerical data array (as pandas Series) or list of numerical data arrays (signal segments).
        prominence | :float: | Minimum prominence for peak detections.

        Returns: {"mean": :float:,"stdev": :float:,"idx": :list: [:int:]}
        '''
        signal = self.signal
        # If list of signals, treat each signal separately (e.g. in the case of disjointed signals).
        if isinstance(signal,list):
            # Initialize list to hold extrema detections.
            all_extrema = []
            # Initialize list to hold list indexes of extrema detections.
            all_extrema_idxs = []
            # Initialize variable to hold the required index shifting for subsequent signal segments.
            shift = 0
            # Iterate through each signal segment.
            for s in signal:
                s = direction * np.array(s)
                # Identify extrema indices.
                extrema_idxs,_ = find_peaks(s,height=0,prominence=prominence)
                # Find corresponding signal values for the extrema indices.
                extrema = s[extrema_idxs]
                # Store signal values at extrema indices.
                all_extrema.extend(extrema)
                # Store extrema indices (relative to the full signal/signal comprising joined signal segments).
                all_extrema_idxs.extend(extrema_idxs + shift)
                # Store the signal shift (i.e. where subsequent signal segments start in full-signal index space).
                shift += len(s)
        else:
            # For whole signal arrays.
            signal = direction * np.array(signal)
            # Identify extrema indices.
            all_extrema_idxs,_ = find_peaks(signal,height=0,prominence=prominence)
            # Find corresponding signal values for the extrema indices.
            all_extrema = signal[all_extrema_idxs]
        # Find the mean of detected extrema.
        extrema_mean = np.mean(all_extrema)
        # Find standard deviation of detected extrema.
        extrema_stdev = np.std(all_extrema)
        # Invalidate the standard deviation if it is of a singular datapoint.
        if len(all_extrema_idxs) <= 1:
            extrema_stdev = np.nan
        return {"mean":extrema_mean,"stdev":extrema_stdev,"idx":all_extrema_idxs}

    def find_thresholds(self,n_sigma):
        ''' Compute superextremity thresholds (for both peaks and troughs).

        n_sigma | :int: | Number of standard deviations away from the mean of a collection of extrema before signal values are treated as being generally more extreme (superextreme) than the extrema collection.

        Returns: {<direction>:<threshold :float:>}
        '''
        # Declare function to compute a directed superextreme threshold.
        threshold = lambda data,direction : direction * (data["mean"] + n_sigma * data["stdev"])
        # Check whether a threshold value is valid.
        sense_check = lambda val : not np.isnan(val) and val != 0
        # Find threshold values for both directions.
        thresholds = {direction:threshold(extrema,direction) for direction,extrema in self.peaks_and_troughs.items()}
        # Validate threshold values.
        valid_thresholds = {k:v for k,v in thresholds.items() if sense_check(v)}
        return valid_thresholds

    def find_superextreme_regions(self,signal):
        ''' Find parts of a provided signal that exceed the superextreme in either direction.

        signal | :pandas.Series: | Signal to find superextreme regions over.

        Returns: {<direction>: :pandas.Series:}
        '''
        # Load thresholds.
        thresholds = self.superextreme_thresholds
        # Initialize structure to hold superextreme parts of the signal.
        extreme_regions = {1:[],-1:[]}
        # Iterate through the directed thresholds.
        for direction,critical_val in thresholds.items():
            # Filter for parts of the signal that are superextreme.
            extreme_region = signal[(direction * signal)>(direction*critical_val)]
            # Store the found part(s).
            extreme_regions[direction] = extreme_region
        return extreme_regions

class SignalExtremaPlotter(SignalExtrema):
    ''' Extrema-related plotting methods for 2D (x,y) data. Extends `SignalExtrema`.
    '''
    def __init__(self,ax,x,y,prominence=0.3,n_sigma=1):
        '''
        ax | :matplotlib.axes.Axes: | Axes to place plotted items on.
        '''
        super().__init__(x,y,prominence=0.3,n_sigma=1)
        self.ax = ax
        # String representation of the two extrema directions in 2D.
        self.label = {1:"Peaks",
                      -1:"Troughs"}

    def plot_extrema_detections(self):
        ''' Plot all extrema found in the input signal.

        returns: None
        '''
        # Colormapping by direction.
        c = {1:"g",
             -1:"darkred"}
        # Iterate through the two extrema collections.
        for direction,data in self.peaks_and_troughs.items():
            # Extract the indexes of the active extrema collection's coordinates.
            extrema_idx = data["idx"]
            # If extrema exist ...
            if len(extrema_idx):
                # Declare function to flatten segments into a 1D pandas Series if necessary.
                flatten = lambda segments : pd.concat(segments) if isinstance(segments,list) else segments
                # Cast x and y coordinates data into 1D numpy arrays.
                x = np.array(flatten(self.x))
                y = np.array(flatten(self.signal))
                # Plot the active extrema points with the relevant color.
                self.ax.scatter(x[extrema_idx],y[extrema_idx],zorder=100,c=c[direction],label=f"Detected {self.label[direction]}",marker="+")
        return

    def plot_thresholds(self):
        ''' Plot the superextreme thresholds as horizontal lines.

        Returns: None
        '''
        # Declare direction linestyles.
        extrema_linestyles = {1:"-.",
                              -1:"--"}
        # Iterate through the directed thresholds.
        for direction,threshold in self.superextreme_thresholds.items():
            # Plot a horizontal line at the active threshold value, assigning the relevant linestyle.
            self.ax.axhline(threshold,c="grey",linestyle=extrema_linestyles[direction],label=f"{self.label[direction]} Threshold")
        return

    def plot_superextreme_regions(self,signal,x_coords):
        ''' Plot and label a directed fill if superextreme regions are found in the direction of fill across the signal. Attempt this for both directions.

        signal | :pandas.Series: | Signal to find superextreme regions over. This values of the signal must be evenly spaced on the x axis.
        x_coords | :list:-like | x coordinates corresponding to the signal. x values must be evenly spaced.

        Returns: None
        '''
        # Find superextreme regions.
        extreme_regions = self.find_superextreme_regions(signal)
        # Declare direction-dependent plotting config.
        c = {1:"lightgreen",-1:"pink"}
        change = {1:"Addition",-1:"Removal"}
        # Iterate through the found directed superextreme regions.
        for direction,region in extreme_regions.items():
            # If superextreme regions are found in the active direction...
            if len(region) > 0:
                # Compute the sum of *all* signal area pointing towards the active direction (i.e. not just the area of the superextreme region).
                change_amount = direction * sum(signal[(direction * signal)>0])
                # Fill all parts of the signal that are in the active direction.
                self.ax.fill_between(x_coords,signal,where=(direction * signal) > 0,facecolor=c[direction],label=f"{change[direction]} {change_amount:.2f}",alpha=0.7,zorder=-1000)
        return

class CompositionProfile():
    ''' Specific methods for handling a deviated (i.e. profile values describing deviation from a central value) compositional profile.
    '''
    def __init__(self,x,signal):
        self.x = x
        # Cast signal values to a pandas series.
        self.signal = pd.Series(signal)

    def find_intersections_with_horizontal(self,y_level=0):
        ''' Find the x coordinate locations of intersections between the signal and a horizontal line at prescribed y level (defaulting to zero).

        y_level | Numerical | y level for the horizontal line.

        Returns: :list:
        '''
        # Slightly expand with of x range to capture marginal intersections.
        x0 = min(self.x)-10
        x1 = max(self.x)+10
        # Create horizontal line specification suitable for intersection analysis.
        horizontal_line = LineString([(x0,y_level),(x1,y_level)])
        # Convert vein signal into LineString object for intersection detection.
        signal_line = LineString(zip(self.x,self.signal))
        # Detect intersections between the signal and horizontal line.

        horizontal_intersections = intersection(signal_line,horizontal_line)
        # Extract x coords of found intersections.
        # np.mean ensures long intersections (i.e. signal being flat at zero) are characterized by the flat signal segment's center.
        intersections = sorted(np.array([np.mean(np.array(g.xy)[0]) for g in horizontal_intersections.geoms]).flatten())
        return intersections

    def closest_points_to_window(self,points,x_window):
        ''' Find the closest points to a window on the x axis (defined by start and end coordinate) from a list of x-coordinate points, and then quantize to existing x coordinates of the GMT profile.

        points | :list:-like | x coordinate points from which to search for the closest points to the two x coordinates in `x_window`.
        x_window | :list:-like [<x_edge_1>,<x_edge_2>] | Two x coordinates representing the edges of the inputted vein window to find the closest point(s) to from `points`.

        Returns: :list:
        '''
        try:
            # Find closest points to the two x coordinates from the x window.
            best_match_points = [nearest_in_list(points,x) for x in x_window]
            # Quantize the closest points to x values corresponding to the signal data.
            quantized_match = [nearest_in_list(self.x,x) for x in best_match_points]
        except IndexError:
            # If nearest point matching fails, declare so and return an empty list.
            print("Failed to find suitable matches to %s" % str(x_window))
            return []
        return quantized_match

    def refine_feature_window(self,vein_window,margin_window=None):
        ''' Try and find a sensible feature window (from intersections between the deviated signal and y=0) - i.e. treat the feature as the region between the closest y=0 intersection to the input vein window. The feature is treated as encompassing both the vein, and the inputted vein margin if provided.

        vein_window | :list:-like [<x_edge_1>,<x_edge_2>] | Two x coordinates representing the edges of a (guess) vein window.
        margin_window | :list:-like [<x_edge_1>,<x_edge_2>] | Two x coordinates representing the edges of a (guess) margin window.

        Returns: :np.array: [<x_edge_1>,<x_edge_2>]
        '''
        # Find all intersections between the loaded signal and y=0.
        horizontal_intersections = self.find_intersections_with_horizontal()
        # Assume no refined (i.e. matched to intersections with horizontal) margin exists.
        refined_margin_window = None
        # Refine the input vein window by matching to the closest intersections with the horizontal.
        refined_vein_window = self.closest_points_to_window(horizontal_intersections,vein_window)
        # If an input margin window is provided ...
        if margin_window:
            # Refine the margin window.
            refined_margin_window = self.closest_points_to_window(horizontal_intersections,margin_window)
            # Check if a margin window was produced.
            if len(refined_margin_window):
                # Find intersections that are considered invalid (i.e. equivalent to existing window edges or the data edges).
                invalid_intersections = [x for x in refined_margin_window if x in (refined_vein_window + [min(self.x),max(self.x)])]
                # Check for invalid intersections or non-uniqueness of the refined margin window edges ...
                if len(invalid_intersections) or len(set(refined_margin_window))<=1:
                    # ... and if either is true, invalidate the refined margin window detection.
                    print("Margin window provided, but no suitable window found")
                    refined_margin_window = None
        # Take the refined margin window as describing the feature if valid.
        if refined_margin_window is not None:
            refined_feature_window = refined_margin_window
        elif len(set(refined_vein_window)) <= 1:
            # Otherwise, check if the refined vein window is invalid, and if so, ignore all window refinement.
            refined_feature_window = vein_window
        else:
            # If the refined vein window is valid, then take that as the feature window.
            refined_feature_window = refined_vein_window
        # Cast the resulting feature window to a 1D numpy array.
        refined_feature_window = np.array(refined_feature_window).flatten()
        return refined_feature_window

def profile_element(data_folder,element,vein_window,bkg_windows,margin_window=None,n_sigma=2,vein_id="",savefig=False):
    ''' Specific function that analyses a profile and plots a representation of the stages of profile analysis. Returns the name of file the plot is saved to if file saving is requested, otherwise returns None.

    data_folder | :str: | Path to folder containing EDS maps for a sample region.
    element | :str: | Element of interest.
    vein_window | :list:-like [<x_edge_1>,<x_edge_2>] | Two x coordinates representing the edges of a (guess) vein window.
    bkg_windows | :list: [([x0,y0],[x1,y1])] | List of rectangular window specifications (in [x0,y0],[x1,y1] opposing corners format) to find the overall mean (of pixels contained within) from. May have a length of 1 of only 1 rectangle is of interest. The specific opposing-corner vertices pair doesn't matter.
    margin_window | :list:-like [<x_edge_1>,<x_edge_2>] | Two x coordinates representing the edges of a (guess) margin window.
    n_sigma | :int: | Number of standard deviations away from the mean of a collection of extrema before signal values are treated as being generally more extreme (superextreme) than the extrema collection.
    vein_id | :str: | Unique vein identified.
    savefig | :bool: | Whether to save the figure as an svg and pdf.

    Returns: :str: or None
    '''
    ## DATA HANDLING
    # Load profile data.
    df = load_gmt_profile(element,vein_id)
    x = df["distance"]
    y = df["stack_val"]
    # Find the background mean/center.
    background_center = find_bkg_mean(data_folder,element,bkg_windows)
    # Recenter the profile signal.
    recentered_signal = y - background_center
    # Denoise the signal.
    denoised_signal,savgol_bin_width = denoise_signal(recentered_signal)
    # Refine the window location that captures the feature (vein +/- margin) within the deviated profile (as an instance of CompositionProfile).
    composition_profile = CompositionProfile(x,denoised_signal)
    refined_feature_window = composition_profile.refine_feature_window(vein_window,margin_window)
    # Slice the profile coordinates into segments defined by the feature window edges.
    sliced_x,sliced_y = slice_by_x_window(x,denoised_signal,refined_feature_window)
    # Extract the feature (vein) segment.
    vein_vals,vein_x = sliced_y[1],sliced_x[1]
    # For the matrix segment, if a margin window is provided (in which case the refined feature window also accounts for the margin) ...
    if margin_window:
        # Use the margin window.
        # I.e. the region to find non-vein related extrema across should just be outside of the manually identified feature region as opposed to a less accurate automatic refinement of the vein+margin window.
        sliced_x,sliced_y = slice_by_x_window(x,denoised_signal,margin_window)
    # Extract the matrix segments.
    matrix_vals = [sliced_y[0],sliced_y[2]]
    matrix_x = [sliced_x[0],sliced_x[2]]
    ## PLOTTING
    # Remove any cached figures.
    plt.close("all")
    # Initialize figure.
    fig,ax = plt.subplots(1,1)
    # Plot a horizontal line at y=0, representing the center for the deviation signal.
    ax.axhline(0,c="k",alpha=0.2,zorder=-100)
    # Plot the raw, deviation signal.
    ax.plot(x,recentered_signal,color="#666666",label="raw",zorder=-10000)
    # Plot the same signal after denoising.
    ax.plot(x,denoised_signal,c="darkblue",zorder=100,label=f"denoised (Savgol bin width = {savgol_bin_width})")
    # Initiate a plotter instance for extrema-related properties of the signal.
    signal_extrema_plotter = SignalExtremaPlotter(ax,matrix_x,matrix_vals,n_sigma=n_sigma)
    # Plot the superextreme threshold.
    signal_extrema_plotter.plot_thresholds()
    # Plot the autodetected extrema (in the matrix segments).
    signal_extrema_plotter.plot_extrema_detections()
    # Plot any superextreme regions from the vein segment.
    signal_extrema_plotter.plot_superextreme_regions(vein_vals,vein_x)
    # Plot the input vein window as a filled region.
    ax.axvspan(*vein_window,color="#deeff5",label="Input vein width",zorder=-50000)
    # Plot the input margin window as a filled region if provided.
    if margin_window is not None:
        ax.axvspan(*margin_window,alpha=0.4,color="yellow",label="Input selvage width",zorder=-500000)
    # Plot the edges of the refined feature window.
    ax.scatter(refined_feature_window,[0,0],zorder=10000,color="magenta",marker="|",label="Recovered vein width")
    # Label axes.
    ax.set_xlabel("Distance /px")
    ax.set_ylabel("Value")
    # Add title.
    ax.set_title(f"{element} profile across vein")
    # Add legend.
    ax.legend(loc="upper left",fontsize=6).set_zorder(10e6)
    ## SAVING
    # Check if figure saving is desired or not.
    if savefig:
        # If so, save the figure as both a pdf (for print display) and svg (for web display).
        fname = f"analysed-{element}-profile-{vein_id}"
        for filetype in [".pdf",".svg"]:
            fig.savefig(os.path.join("static",fname+filetype))
        return fname
    return

def profile_element_robust(data_folder,element,vein_window,bkg_windows,margin_window=None,n_sigma=2,vein_id="",savefig=True):
    ''' Wrapper function to avoid blocking errors in case of missing data files. Also determines which matplotlib backend to use depending on whether the figure is to be saved or not. If the figure is to be saved, the selected backend will not permit display.
    '''
    try:
        # Decide on a suitable matplotlib graphics backend.
        if savefig:
            mpl.use("agg")
        else:
            mpl.use("TkAgg")
        # Attempt to execute profile analysis and plotting.
        fname = profile_element(data_folder,element,vein_window,bkg_windows,margin_window,n_sigma,vein_id,savefig=savefig)
        # Return the filename (without file extension) that may also be None (depending on whether figure saving was requested).
        return fname
    except FileNotFoundError as e:
        # If a data-missing failure occurs, print the error rather than raising it as a blocking/fatal event.
        print(f"Missing data: {e}")
        pass
    return


########################################
# EDS Element Abundance Map Processing #
########################################

class EDSMapPlotter():
    ''' Relatively specific (to this study) methods for plotting an EDS map. Plotting will be relative to image coordinates (i.e. 0,0 origin at the top left of the image, with positive y down and positive x right).
    '''
    def __init__(self,data_folder,element,ax=None,crop=None):
        '''
        data_folder | :str: | Folder containing EDS maps for the element of interest.
        element | :str: | Element of interest; used to find the EDS map filename.
        ax | :matplotlib.axes.Axes: | Axis to plot onto.
        crop | :tuple:-like (xmin,xmax,ymin,ymax) | Coordinates for the edges of the rectangular crop region.
        '''
        # Initialize axis if none provided.
        if ax is None:
            _,ax = plt.subplots()
        self.ax = ax
        # Load relevant image.
        img = Image.open(f"{data_folder}/{element}.tiff")
        self.full_img = img
        # Extract image dimensions
        w0,h0 = img.size
        # In the absence of a specified crop, provide a placeholder crop that has no effect on output.
        if crop is None:
            crop = [0,w0,0,h0]
            w,h = w0,h0
        else:
            # Otherwise, cast input crop to an integer list.
            crop = [int(edge) for edge in crop]
            # Crop image using this crop specification.
            img,_ = safe_crop(img,crop)
            # Refresh image dimensions
            w,h = img.size
        # Fix plot coordinate vs image y axis direction
        crop[3],crop[2] = h0 - crop[2],h0 - crop[3]
        # Extract coordinates for the bottom-left corner of the image (treated as the origin).
        self.origin_x = crop[0]
        self.origin_y = crop[2]
        self.element = element
        self.crop = crop
        self.img = img
        self.w0 = w0
        self.h0 = h0
        self.w = w
        self.h = h

    def plot_vein(self,vein_filepath,c="g",alpha=0.5,label=None):
        ''' Plot vein path from a vein specification file.

        vein_filepath | :str: | Filepath to a vein path specification.
        c | color-spec | Color for plotting the vein line.
        alpha | :float: | Alpha value for plotting the vein line.
        label | :str: | Legend label for the vein line.

        Returns: None
        '''
        # If so, load the coordinates of the vein path.
        vein_path_df = pd.read_csv(vein_filepath,sep="\s+",names=range(2))

        # Plot the vein path.
        self.plot_line(vein_path_df,c=c,alpha=alpha,label=label)
        return

    def plot_line(self,xy_df,**kwargs):
        ''' Plot a line described by full-image coords. This function will correct the coordinates for the active crop.

        xy_df | :pandas.DataFrame: | Dataframe of line coordinates arranged in an x and y column in that order.
        kwargs | Kwargs to pass onto ax.plot()

        Returns: None
        '''
        xs = xy_df.iloc[:,0]
        ys = xy_df.iloc[:,1]
        # Correct the coordinates for plotting into Pillow image coordinates (for the EDS map).
        xs = xs - self.origin_x
        ys = self.h0 - ys - self.origin_y
        # Plot corrected coordinates.
        self.ax.plot(xs,ys,**kwargs)
        return

    def plot_map(self):
        ''' Plot the raw EDS map, ensuring it is scaled such that 1 px = 1 data unit.

        Returns: None
        '''
        # Determine plotting bounds for the image to ensure correct positioning relative to the 1 px = 1 GMT unit coordinate alignment
        left = -self.origin_x if self.origin_x<0 else 0
        bottom = -self.origin_y if self.origin_y<0 else 0
        # Plot image with the correct plotting bounds.
        bounds = (left,left+self.w,bottom+self.h,bottom)
        self.ax.imshow(self.img,extent=bounds)
        return

    def save_smoothed_map(self,bkg_windows,fname,blur_radius=10):
        ''' Smooth the EDS map, and plot and save a smoothed map of deviations from a map "background" value of elemental abundance determined from a prescribed window of background material. Returns tthe file that's saved to.

        bkg_windows | :list: [([x0,y0],[x1,y1])] | List of rectangular "background" window specifications (in [x0,y0],[x1,y1] opposing corners format) for finding of the elemental abundance background for the EDS map.
        fname | :str: | Filename of the output SVG.
        blur_radius | Numerical | Radius to perform Gaussian Blurring with.

        Returns: :str:
        '''
        ax = self.ax
        # Compute the "background" elemental abundance using all the supplied background window locations.
        mean_val = find_rect_mean(bkg_windows,pil_image=self.full_img)
        # Apply Gaussian Filter to smooth the image.
        img = self.img.filter(ImageFilter.GaussianBlur(blur_radius))
        # Plot deviation image.
        plot_deviations_image(ax,img,mean_val)
        # Label ticks at image edges with the coordinate in pre-cropped image pixels.
        ax.set_xticks([0,self.w],[int(self.origin_x),int(self.origin_x + self.w)])
        ax.set_yticks([0,self.h],[int(self.origin_y),int(self.origin_y + self.h)])
        # Fix image orientation to max pixel coordinate convention (increasing from top to down).
        ax.invert_yaxis()
        # Set plot title.
        ax.set_title(self.element)
        # Save figure.
        plt.savefig(fname,dpi=600)
        return fname

#############################
# Profile Geometry Handling #
#############################

class Line():
    ''' Class to handle 2D linear (P1) infinite lines represented by gradient and intercept.

    Assumes a 2D Euclidian, Cartesian coordinate system (i.e. line lies on Cartersian plane).
    '''
    def __init__(self,gradient,intercept):
        '''
        gradient | Numerical | dy/dx of the line. Input None for a vertical line.
        intercept | Numerical | y(x=0) of the line. However, if dy/dx is np.nan, this will be x(y=0) instead.
        '''
        self.gradient = gradient
        self.intercept = intercept

    def evaluate_at(self,x):
        ''' Find the corresponding y coordinate on the line given an x coordinate.

        x | Numerical | x coordinate on the line.

        Returns: Numerical
        '''
        if self.gradient != None:
            return self.gradient * x + self.intercept
        else:
            return None

    def intersect(self,other):
        '''
        Find the x,y coordinate of intersection between the current line and another line.

        other | :Line: | The Line instance to find an intersection with.

        Returns: (Numerical,Numerical)
        '''
        # Check if the lines are parallel to each other (or both equal to None for a perfectly vertical line)...
        if self.gradient == other.gradient:
            # ... and if so, raise an error for inability to find an intersection.
            raise ValueError("The two lines are parallel and so will never intercept")

        if self.gradient == None:
            x_intersect = self.intercept
            y_intersect = other.evaluate_at(x_intersect)
        elif other.gradient == None:
            x_intersect = other.intercept
            y_intersect = self.evaluate_at(x_intersect)
        else:
            # Compute the x coordinate of the intersection.
            x_intersect = (other.intercept - self.intercept)/(self.gradient - other.gradient)
            # Compute the corresponding y coordinate. This should be equivalent to other.evaluate_at(x_intersect)
            y_intersect = self.evaluate_at(x_intersect)
        return x_intersect,y_intersect

def line_projection(points,line):
    ''' Project points orthogonally onto a line.

    points | :list: [(px,py)] | List of xy points to project orthogonally onto the line.
    line | :list: [(p1x,p1y),(p2x,p2y)] | Infinite line defined by two coordinates on the line.

    Returns: :list: [(px,py)]
    '''
    # Separate the two coordinates on the line.
    p1,p2 = line
    # Check if both coordinates have the same x value.
    if p1[0] == p2[0]:
        # If so, the line is vertical line.
        line_grad = None
        # Intercept is the vertical line's x intercept.
        line_intercept = p1[0]
    else:
        # Otherwise the line can be expressed by a valid linear function.
        line_grad = (p2[1] - p1[1])/(p2[0] - p1[0])
        line_intercept = p1[1] - line_grad * p1[0]
    # Describe the infinite line passing through the two coordinates used to define `line` using a Line instance.
    infinite_line = Line(line_grad,line_intercept)
    # Initialize list to hold the projected points.
    projected_points = []
    # Iterate through the points.
    for xy in points:
        # Check if the line (which is being projected onto) is horizontal.
        if line_grad == 0:
            # If so, the line normal that passes through the active point is vertical.
            points_grad = None
            # x intercept of this line normal is the x coordinate of the point.
            points_intercept = xy[0]
        else:
            # Otherwise check if the line is vertical.
            if line_grad == None:
                # If so, the line normal is horizontal.
                points_grad = 0
            else:
                # Otherwise, compute the line normal's gradient as typical.
                points_grad = -1/line_grad
            # Compute the line normal's y intercept.
            points_intercept = xy[1] - points_grad * xy[0]
        # Describe the infinite line normal that passes through the active point as a Line instance.
        line_normal_to_point = Line(points_grad,points_intercept)
        # Find where the line and line normal (joining active point) intersect, which represents the point projected onto the line.
        intersection = line_normal_to_point.intersect(infinite_line)
        # Store the projected point.
        projected_points.append(intersection)
    return projected_points

def point_distances(p0,ps):
    ''' Find cartesian distances between points in a list and a reference point.

    p0 | :list: [<x>,<y>] | Reference point
    p1 | :list: [(<x>,<y>)] | List of points.

    Returns: :list: [<distances>]
    '''
    return [np.sqrt((p0[1] - p[1])**2 + (p0[0] - p[0])**2) for p in ps]

def balance_profile(vein_path,profile_line):
    ''' Reposition an provided profile line such that it's centered about the vein path that it's profiling across. Somewhat approximate as it uses circles to aid in finding points of constant distance from a central point, and circles are discretized here.

    vein_path | :list: [(<x>,<y>)] | List of coordinates defining the vein center path.
    profile_line | :list: [(<x0>,<y0>),(<x1>,<y1>)] | Endpoints of a vein profile line.

    Returns: [(<x0>,<y0>),(<x1>,<y1>)]
    '''
    # Cast the profile line and vein path to LineString objects.
    profile_line = LineString(profile_line)
    vein_path = LineString(vein_path)
    # Split the vein path into linear segments.
    vein_path_segments = [LineString(l) for l in zip(vein_path.coords[:-1],vein_path.coords[1:])]
    # Compute the half-length of the profile line (i.e. how long the profile should be either side of the vein path).
    half_len = int(profile_line.length)/2
    # Find the current intersection(s) between the profile line and vein path (by treating each vein path segment separately).
    profile_vein_intersections = [segment.intersection(profile_line) for segment in vein_path_segments]
    # Extract valid intersections and their corresponding vein path segment ("live" segments).
    live = [(s,intersection) for s,intersection in zip(vein_path_segments,profile_vein_intersections) if not intersection.is_empty]
    # Check if there's multiple valid intersections.
    if len(live)>1:
        # If so, display that the profile or vein path is badly defined, resulting in multiple profile-vein intersections.
        print("The vein path (or profile) is badly defined (paths should not intersect more than once)")
    # Consider only the first (and likely only) intersection and live vein path segment.
    live_segment,profile_vein_intersection = live[0]
    # Generate circle centered about intersection with radius = half_len and 4 linear segments per quarter circle (quad_segs).
    half_len_circle = profile_vein_intersection.buffer(half_len,quad_segs=4)
    # Extract the coordinates of the profile - vein segment intersection, which forms the center of the circle.
    x_c,y_c = np.array(profile_vein_intersection.xy).flatten()
    # Separate the x and y coordinates of the live vein path segment.
    x_live,y_live = live_segment.xy
    # Determine if the live segment is aligned to either of the orthogonal coordinate axes.
    if x_live[0] == x_live[1]:
        # Vertical vein segment line -> horizontal profile.
        centered_profile = [(x_c + direction * half_length,y_c) for direction in [-1,1]]
    elif y_live[0] == y_live[1]:
        # Horizontal vein segment line -> vertical profile.
        centered_profile = [(x_c,y_c  + direction * half_length) for direction in [-1,1]]
    else:
        # Otherwise, treat vein segment as a first order polynomial.
        # Compute vein segment gradient.
        gradient = (y_live[1]-y_live[0])/(x_live[1]-x_live[0])
        # Compute the gradient normal to the vein segment.
        normal_gradient = -1/gradient
        # Find y intercept of the line that has the normal gradient and passes through the profile-vein path intersection.
        normal_intercept = y_c - normal_gradient * x_c
        # Find the start and endpoint coordinates of a finite-length line with normal gradient centered on the intersection that exceeds the bounds of the circle.
        x0 = x_c-2*half_len
        x1 = x_c+2*half_len
        y0 = normal_gradient * x0 + normal_intercept
        y1 = normal_gradient * x1 + normal_intercept
        normal = LineString([(x0,y0),(x1,y1)])
        # Find coordinates of the intersections between this finite-length normal line and the constant-radius circle, which produces a suitably oriented profile (normal to local vein path) which matches the length of the input profile.
        intersection_x,intersection_y = normal.intersection(half_len_circle).xy
        # Convert the coordinates from [(x,)],[(y,)] to [(x,y),] format.
        centered_profile = [(intersection_x[i],intersection_y[i]) for i in range(2)]
    return centered_profile

#################################
# GMT Profile Location Plotting #
#################################

def bounding_box(xy_df):
    ''' Find the crop rectangle that bounds a collection of points.

    xy_df | :pandas.DataFrame: | Dataframe of coordinates arranged in an x and y column (in that order).

    Returns: :list: [xmin,xmax,ymin,ymax]
    '''
    xs = xy_df.iloc[:,0]
    ys = xy_df.iloc[:,1]
    x_range = [min(xs),max(xs)]
    y_range = [min(ys),max(ys)]
    return x_range + y_range

def profile_data_to_df(raw_cross_profiles_data):
    ''' Parse GMT profile data (which may be separated into multiple geometries) into a single pandas dataframe.

    raw_cross_profiles_data | :str: | Output of GMT profiling read as a string.

    Returns: :pandas.DataFrame:
    '''
    # Remove geometry separation delimiter lines.
    cross_profiles_data = re.sub("[> ].*?\n","",raw_cross_profiles_data)
    # Parse cleaned data into a pandas dataframe.
    df = pd.read_csv(StringIO(cross_profiles_data),sep="\s+",names=range(5))
    return df

def plot_cross_profiles(vein,element,study_area):
    ''' Plot a vein and the collection of GMT cross profiles onto a background of the element map. Returns the name of the image file that the plot is saved to.

    vein | :str: | Filename for the vein path.
    element | :str: | Element of interest.
    study_area | :str: | Path to the directory containing the element maps.

    Returns: :str:
    '''
    # Load the collection of evenly-spaced profile data as a string.
    cross_profiles = f"tmp/{element}-{vein}-profile.dat"
    with open(cross_profiles) as infile:
        raw_cross_profiles_data = infile.read()
    # Parse the profiles collection into a dataframe.
    profiles_df = profile_data_to_df(raw_cross_profiles_data)
    # Find the bounding box for the profiles collection
    crop_region = bounding_box(profiles_df)
    # Initialize plot.
    fig,ax = plt.subplots()
    # Initialize EDS-specific plotter.
    eds_plotter = EDSMapPlotter(study_area,element,ax,crop_region)
    # Plot the cropped raw EDS map.
    eds_plotter.plot_map()
    # Iterate through the profiles (raw string data).
    for i,profile in enumerate(raw_cross_profiles_data.split(">")):
        # Parse raw string profile specification into dataframe.
        profile_df = profile_data_to_df(profile)
        # Supply a label to only the first profile line.
        label = "Profiles" if i==0 else None
        # Lineplot the first and last point.
        eds_plotter.plot_line(profile_df.iloc[:,[0,1]],color="r",label=label,linewidth=0.5)
    # Plot the vein path.
    eds_plotter.plot_vein(vein,c="c",alpha=1,label="Vein path")
    # Extract the crop rectangle used on the plotted image.
    applied_crop = eds_plotter.crop
    range_x = applied_crop[:2]
    range_y = applied_crop[2:]
    # Label the axes with image coordinates (under the image coordinate system).
    ax.set_xticks([0,eds_plotter.w],range_x)
    ax.set_yticks([0,eds_plotter.h],range_y)
    # Set title to the active element.
    ax.set_title(element)
    # Add legend.
    plt.legend()
    # Save plot as a pdf for print display (no online display necessary).
    fname = f"{vein}-profiles.pdf"
    plt.savefig(fname)
    return fname

#########################
# CONVENIENCE FUNCTIONS #
#########################
# Functions that capture bits of code that are called together in `server.py`.

def smooth_element_and_plot_vein(data_folder,element,crop,bkg_rect,vein_filepath):
    ''' Convenience function for `server.py`. Plots a smoothed EDS map and overlaying the vein path.
    '''
    eds_plotter = EDSMapPlotter(data_folder,element,crop=crop)
    eds_plotter.plot_vein(vein_filepath)
    vein_id = os.path.basename(vein_filepath)
    outfile = eds_plotter.save_smoothed_map(bkg_rect,os.path.join("static",f"{element}-{vein_id}-smoothed.svg"))
    return outfile
