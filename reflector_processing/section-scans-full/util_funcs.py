import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2

def construct_minmax_filter(arr,min_val=None,max_val=None):
    ''' Create min-max boolean filter using an array of values.

    arr : *numpy array* of values
    min_val (numerical) : minimum end of filter
    max_val (numerical) : maximum end of filter

    returns minmax_filter (list of bool) : boolean filter applicable to arr
    '''
    # If no min_val provided, set to the minimum in the array (i.e. no minimum filtering)
    if min_val == None:
        min_val = min(arr)
    # If no max_val provided, set to the maximum in the array (i.e. no maximum filtering)
    if max_val == None:
        max_val = max(arr)
    minmax_filter = np.logical_and(arr <= max_val,arr >= min_val)
    return minmax_filter

def list_of_list_filter(list_of_list,bool_filter):
    ''' Filter a list of list-like objects by a top-level boolean filter.

    list_of_list : list of list-like objects
    bool_filter : top-level boolean filter

    returns : list of list-like objects
    '''
    return [l[0] for l in zip(list_of_list,bool_filter) if l[1]]

def exp_with_first_order_p_func(x,a,b,c,d,e):
    # Exponential fit function with first order polynomial for area distributions.
    return d * np.exp(-c * x + a) + b - e * x

def exp_func(x,a,b):
    # Exponential fit function for area distributions.
    return a * np.exp(b*x)

def bin_values(values,max_value,n_bins,min_value=0):
    ''' Bin and count values based on a min-max range and number of bins.

    values (list of numericals) : values to bin
    max_value (numerical) : maximum bin edge
    n_bins (int) : number of bins
    min_value (numerical) : minimum bin edge

    returns counts (list of numerical) : number of values within each bin
    bins (list of numerical) : bin edges (will be one element longer than bin midpoints)
    midpoints (list of numerical) : bin midpoints
    '''
    # Compute bin edges.
    bins = np.linspace(min_value,max_value,n_bins+1)
    # Compute bin midpoints.
    midpoints = (bins[:-1] + bins[1:])/2
    # Count values within each bin.
    counts,_ = np.histogram(values,bins=bins)
    return counts,bins,midpoints

def fit_exp_log_y(x,y):
    ''' Determine best fit to empirical exponential distribution (where some bins may contain zero counts).

    x (list of numericals) : x value to fit
    y (list of numericals) : y value to fit

    returns fit_params (list of numericals) : fit parameters
    '''
    # Select only datapoints where the bin count is non-zero.
    nonzero_y = y!=0
    y = y[nonzero_y]
    x = x[nonzero_y]
    # Curve fitting using the exponential distribution function.
    popt,_ = curve_fit(exp_func,x,np.log10(y))
    fit_params = popt.tolist()
    return fit_params

def get_dimensions(contours):
    ''' Extract minimum bounding rectangle dimensions from contours.

    contours : list of contours

    returns all_dimensions : list of lists containing dimesions of minimum bounding rectangles
    '''
    all_dimensions = []
    for contour in contours:
        center,dimensions,rotation = cv2.minAreaRect(contour)
        all_dimensions.append(dimensions)
    return all_dimensions
