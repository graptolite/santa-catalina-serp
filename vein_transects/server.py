from flask import Flask,render_template,request,redirect

# !!!!!!
# Need to hide irrelevant maps (non-elemental) in different folder to the EDS maps folder.
# !!!!!!

# TODO:
# - Save and load "options" (i.e. contents of `storage` var)
# - Allow renaming of vein path. - overruled 14th Aug (overcomplicated).

# Note: where there isn't a directed threshold (in the analyzed profile), it means that there were zero or one directed extrema detected with the prominence requested.

# Possible bugs
# sometimes old tiff maps are deleted... Could be related to things outside of this code though...

import re
import subprocess
import os

import numpy as np
import json
import pandas as pd
from PIL import Image
from datetime import datetime
from pathlib import Path

from signal_processing import *
from shapely import LineString

# Make sure all necessary folders for website function are present.
for folder in ["tmp","static"]:
    if not os.path.exists(folder):
        os.mkdir(folder)

def init_storage():
    ''' Generate a starting local storage structure/config.
    '''
    storage = {"element":"placeholder",
               "width":500, # img display width.
               "target_area":None,
               "vein_def":None,
               "profile_length":None,
               "vein_window":None,
               "selvage_window":None,
               "profile_line":None,
               "vein_line":None,
               "crossvein_line":None,
               "crossselvage_line":None,
               "bkg_rect":[],
               "bkg_rect_svgs":[],
               "display_svg":None,
               }
    # Update the blank storage with any options for code testing if present.
    if os.path.exists("testing_options.py"):
        import testing_options
        storage.update(testing_options.storage)
    return storage

# Load the starting storage structure.
storage = init_storage()

# Initialize the flask web app.
app = Flask(__name__)

#####################
# Utility Functions #
#####################

def collate_profiles(prepend_files=[]):
    ''' Combine saved analyzed-profile files for the active vein id into a pdf report (via orgmode -> latex). Returns the name of the orgmode report file.

    prepend_files | :list: [:str:] | Paths to any additional files to prepend to the profiles report.

    Returns: :str:
    '''
    # Extract vein id.
    vein_id = os.path.basename(storage["vein_def"])
    # Find list of files relevant to the active vein that can be eaasily included in a latex document.
    files = sorted([f for f in os.listdir("static") if vein_id in f and not f.endswith(".svg")])
    # Construct org report file name.
    orgfile = f"{vein_id}-out.org"
    # Write the orgmode report to this file.
    with open(orgfile,"w") as outfile:
        outfile.write("\n".join(["#+OPTIONS: toc:nil","#+LATEX_HEADER: \\usepackage[margin=1in]{geometry}",""] + [f"[[./{f}]]" for f in prepend_files] + [f"[[./static/{f}]]" for f in files]))
    # Compile the org report into pdf via latex.
    subprocess.call(["emacs",orgfile,"--batch","-f","org-latex-export-to-pdf","--kill"])
    return orgfile

def tiff_width_height(tifffile):
    ''' Quick but inefficient function to find the dimensions of a tifffile.

    tifffile | :str: | Tiff image to find the dimensions of.
    '''
    w,h = Image.open(tifffile).size
    return w,h

def list_tifffiles(directory):
    ''' Find list of tiff files (excluding hidden ones under the UNIX filesystem) in a folder based on file extension.

    directory | :str: | Path to folder that may contain tiff files.

    Returns: :list: [:str:]
    '''
    tifffiles = [f for f in os.listdir(directory) if (f.endswith(".tiff") or f.endswith(".tif")) and not f.startswith(".")]
    return tifffiles

def elements_from_filenames(filenames):
    ''' Find strings that look like element names from a list of filenames.

    filenames | :list: [:str:] | List of filenames.

    Returns: :list: [:str:]
    '''
    elements = [e for e in [re.search("([A-z]+?)[^A-z]",f).group(1) for f in filenames] if len(e) <= 2]
    return elements

def find_element_tiff(EDS_dir,element):
    ''' Find the corresponding tiff filename(s) to a requested element.

    EDS_dir | :str: | Path to folder containing EDS element maps.
    element | :str: | Element of interest.

    Returns: :list: [<filenames>]
    '''
    # Get list of all tiff files in the EDS folder.
    tifffiles = list_tifffiles(EDS_dir)
    # Find all EDS map tiff files that are named in a way to suggest they represent the element of interest.
    element_tiff = [f for f in tifffiles if re.search(f"{element}[^A-z]",f)]
    return element_tiff

def path_to_svg(coords_list):
    ''' Convert a list of xy coordinates into an SVG path definition/the `d` attribute of a <path> svg tag.

    coords_list | :list:-like [(<x>,<y>),] | Coordinates to be converted to SVG path definition.

    Returns: :str:
    '''
    # Construct list of commands to define the coordinates list as an SVG path.
    path = " ".join(["M"] + [",".join(p) for p in np.array(coords_list).astype(str)])
    return path

def tiff_to_jpg(tifffile,outfile):
    ''' Convert tiff file to jpg file.

    tifffile | :str: | Path to tiff file.
    outfile | :str: | Path to output jpg file.

    Returns: None
    '''
    subprocess.call(["convert","-resize","500",tifffile,outfile])
    return

def svg_to_pdf(svg_file):
    ''' Convert svg file to pdf file and return the pdf file name.

    svg_file | :str: | Path to svg file.

    Returns: :str:
    '''
    outfile = svg_file + ".pdf"
    subprocess.call(["inkscape",svg_file,f"--export-pdf={outfile}",])
    return outfile

#########################################
# HIGHLY SPECIFIC CONVENIENCE FUNCTIONS #
#########################################
# These capture functions procedures that are used multiple times and have a mixture of constant* and variable inputs (between the uses). Only the variable inputs are taken as inputs to these convenience functions.
# *The content of constant inputs can change (i.e. due to being extracted from `storage`), but the way they are referred to remains constant.

def get_jpg_path():
    element = storage["element"]
    region = Path(storage["target_area"]).parts[-1]
    jpg = os.path.join("static",f"{element}-{region}.tiff.jpg")
    return jpg

def force_load_tiff():
    ''' Permit the active element's EDS tiff to be representable in web display by ensuring an existing JPG conversion. If an EDS tiff does not exist for the active element, convert an arbitary element's tiff to JPG to act as a placeholder.
    '''
    tifffiles = [f for f in os.listdir(storage["target_area"]) if (f.endswith(".tiff") or f.endswith(".tif"))]
    element = storage["element"]
    element_tiff = [f for f in tifffiles if f.split(".")[0]==element]
    if len(element_tiff) == 0:
        print(f"No tiff found for {element}")
        storage["element"] = element = "placeholder"
        tifffile = tifffiles[0]
    else:
        tifffile = element_tiff[0]

    tifffile = os.path.join(storage["target_area"],tifffile)
    outfile = get_jpg_path()
    tiff_to_jpg(tifffile,outfile)
    return outfile

def ensure_normalized_eds_name(tiff_source):
    element = storage["element"]
    # Construct normalized filename and filepath.
    normalised_tiff_source = element + ".tiff"
    parent_folder = storage["target_area"]
    norm_path = os.path.join(parent_folder,normalised_tiff_source)
    # Check if the relevant tiff file's name in a normalized format or not.
    if tiff_source != normalised_tiff_source:
        # Rename to the normalized format if not.
        os.rename(os.path.join(parent_folder,tiff_source),
                  norm_path)
    return norm_path

def ensure_eds_map_has_jpg(tiff_source):
    # Load relevant jpg filepath.
    jpg = get_jpg_path()
    # Exit the function if the relevant jpg file already exists.
    # The existence also signifies the validation/normalisation of the .tiff filename (to "<element>.tiff").
    if os.path.exists(jpg):
        return jpg
    norm_path = ensure_normalized_eds_name(tiff_source)
    # Perform the TIFF -> JPG conversion on the (now-)normalized tiff file name for the active element.
    tiff_to_jpg(norm_path,jpg)
    return jpg

def list_available_elements():
    ''' Get list of elements available for analysis for the active EDS target area.
    '''
    return elements_from_filenames(list_tifffiles(storage["target_area"]))

def smooth_element(element):
    ''' Smooth an requested element's EDS map and overlay the vein path, then save. If no element is provided, it defaults to the active element.

    element | :str: | Name of element to plot the smoothed EDS map.
    '''
    # Use active element if none provided.
    if not element:
        element = storage["element"]
    # Call actual plotting function.
    outfile = smooth_element_and_plot_vein(storage["target_area"],element,bbox_vein_and_profile(),storage["bkg_rect"],storage["vein_def"])
    return outfile

def EDS_width_height():
    ''' Find the width and height of the active EDS map.
    '''
    target_area = storage["target_area"]
    element = storage["element"]
    return tiff_width_height(os.path.join(target_area,f"{element}.tiff"))

def invert_y_dir(xy_arr):
    ''' Invert the y direction of an array of xy coordinates based *on the active EDS map dimensions*.

    xy_arr | :np.array: [[<x>,<y>],] | Array of xy coordinates.

    Returns: :np.array: [[<x>,<y>],]
    '''
    # Find dimensions of the active EDS map.
    w_real,h_real = EDS_width_height()
    xy_arr[:,1] = h_real - xy_arr[:,1]
    return xy_arr

def bbox_vein_and_profile():
    ''' Find the bounding box of the active (in `storage`) vein and profile specifications. The bounding box is in [xmin,xmax,ymin,ymax] format.

    Returns: :list: [<xmin>,<xmax>,<ymin>,<ymax>]
    '''
    # Find all the coordinates in the vein and profile lines.
    all_coords = np.vstack([storage["vein_line"],storage["profile_line"]])
    # Separate x and y coordinates.
    xs = all_coords[:,0]
    ys = all_coords[:,1]
    # Extract the bounds into a bounding box.
    bbox = [min(xs),max(xs),min(ys),max(ys)]
    return bbox

def run_gmt_profile():
    ''' Execute GMT profiling with the current config extracted from local storage. Returns the name of the PNG file that contains the result of profiling.
    '''
    element = storage["element"]
    vein_id = os.path.basename(storage["vein_def"])
    # Execute GMT profiling code.
    subprocess.call(["bash","profiling.sh",element,str(storage["profile_length"]),storage["vein_def"],vein_id,storage["target_area"]])
    # Construct output PNG filename (based on hardcoded shared code with `profiling.sh`).
    output_name = f"{element}-profiles-{vein_id}.png"
    return output_name

def analyse_gmt_profile():
    ''' Perform an analysis of the raw GMT profiling output, including profile signal recentering, smoothing, extrema detection etc. Returns the name of the SVG file that contains the result of this analysis.

    Returns: :str:
    '''
    element = storage["element"]
    vein_id = os.path.basename(storage["vein_def"])
    # Execute analysis.
    fname = profile_element_robust(storage["target_area"],element,storage["vein_window"],storage["bkg_rect"],storage["selvage_window"],vein_id=vein_id)
    return fname + ".svg"

def scale_data_to_display(data_xy):
    ''' Scale coordinates from the data (EDS map) reference frame to the web display reference frame.

    data_xy | :list:-like [[<x>,<y>],] | Coordinates in the data reference frame.

    Returns: :numpy.array: [[<x>,<y>],]
    '''
    w_data,_ = EDS_width_height()
    display_xy = np.array(data_xy) * (storage["width"]/w_data)
    return display_xy

def scale_display_to_data(display_xy):
    ''' Scale coordinates from the web display reference frame to the data (EDS map) reference frame.

    display_xy | :list:-like [[<x>,<y>],] | Coordinates in the web display reference frame.

    Returns: :numpy.array: [[<x>,<y>],]
    '''
    w_data,_ = EDS_width_height()
    data_xy = np.array(display_xy) * (w_data/storage["width"])
    return data_xy

def handle_vein_path(xy_path):
    ''' Handle vein path coordinates, saving them to disk as a GMT-readable file, and to local storage.

    xy_path | :list:-like [(<x>,<y>),] | Coordinates definining the veins medial line.

    Returns: None
    '''
    # Declare folder in which vein path definition files are stored.
    path_folder = "vein-paths"
    # Create the folder if it doesn't already exist.
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    # Create unique output filename ("vein id") for the vein definition file.
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    vein_file = os.path.join(path_folder,formatted_time+".txt")
    # Save the vein coordinates into this output file.
    with open(vein_file,"w") as outfile:
        outfile.write("\n".join(["\t".join(p) for p in xy_path.astype(str)]))
    # Update storage with the vein file path and vein path coordinates.
    storage["vein_def"] = vein_file
    storage["vein_line"] = xy_path.tolist()
    return

def handle_profile_path(xy_path):
    ''' Handle across-vein profile coordinates. These coordinates, as well as the length of a linear profile are saved to local.

    xy_path | :list:-like [(<x0>,<y0>),(<x1>,<y1>)] | Coordinates definining the profile across the vein.

    Returns: None
    '''
    # Ensure the profile is linear, taking the first and last point if there are multiple input points defining the profile.
    if len(xy_path) > 2:
        print("Profile path contains more than 2 points (start,end) - taking just the first and last points")
        xy_path = [xy_path[0],xy_path[-1]]
    # Find length of the profile line.
    profile_line = LineString(xy_path)
    # Store profile length.
    storage["profile_length"] = int(profile_line.length)
    # Balance/center the profile about the vein path (assumed to be vein medial).
    standard_profile = balance_profile(storage["vein_line"],xy_path)
    # Store the balanced profile line.
    storage["profile_line"] = standard_profile
    return

def handle_feature(xy_path,feature):
    ''' Handle vein-spanning feature coordinates, which get projected onto the profile to find which part of the profile should be treated as belonging to feature of interest. The coordinates after projection (2D coordinates with respect to the EDS map) and coordinates along the profile ("window") are saved to local.

    xy_path | :list:-like [(<x0>,<y0>),(<x1>,<y1>)] | Coordinates on or near the profile line spanning the feature.
    feature | :str: | Name of the feature type out of {"vein","selvage"}.

    Returns: None
    '''
    # Project the line spanning the feature onto the active profile.
    feature_window_projection = line_projection(xy_path,storage["profile_line"])
    # Store the projected feature span.
    storage[f"cross{feature}_line"] = feature_window_projection
    # Compute the distances between the start of the full profile and the endpoints of the feature span.
    feature_window_distances = point_distances(storage["profile_line"][0],feature_window_projection)
    # Compute the half distance of the full profile line.
    half_line_dist = storage["profile_length"]/2
    # Compute the 1D along-profile coordinates of the feature window with zero at the point where the profile intersects with the vein path (vein center).
    feature_window = sorted((np.array(feature_window_distances)-half_line_dist).astype(int).tolist())
    # Store the first and last point (i.e. enforce a window) of the feature's span as the window to use.
    storage[f"{feature}_window"] = [feature_window[0],feature_window[-1]]
    return

def handle_vein_span(xy_path):
    ''' Handle coordinates representing the span across the vein when projected onto the profile line.
    '''
    handle_feature(xy_path,"vein")
    return

def handle_selvage_span(xy_path):
    ''' Handle coordinates representing the span across the (vein+)selvage when projected onto the profile line.
    '''
    handle_feature(xy_path,"selvage")
    return

@app.route("/")
def init(load_folder=False):
    ''' Backend code for the landing page.
    '''
    # Make sure no residual background rectanges remain.
    clear_bkg_rect()
    print(load_folder)
    # Check if a EDS target area has been loaded.
    if storage["target_area"] and not load_folder:
        # Load a placeholder or valid config-specified element's converted EDS map.
        element = storage["element"]
        region = Path(storage["target_area"]).parts[-1]
        placeholder = f"static/{element}-{region}.tiff.jpg"
        if not os.path.exists(placeholder):
            placeholder = force_load_tiff()
        # Find the loaded EDS map dimensions.
        w,h = Image.open(placeholder).size
        # Scale the EDS map to a pre-specified width for web display.
        width = storage["width"]
        height = (width/w) * h
        # Load page for vein profiling.
        return render_template("vein_profiling.html",width=f"{width}px",height=f"{height}px",element=storage["element"],region=region)
    else:
        # If no EDS target area has been loaded, load a page on which the EDS target area can be specified.
        return render_template("load_path.html")

@app.route("/load_folder", methods=["POST"])
def load_folder():
    ''' Load an provided EDS maps folder into local storage. `redirect` doesn't work at the moment - the page needs to manually refreshed for this update to take effect.
    '''
    # Load folder and remove double quotes around it to isolate the folder name.
    folder = request.data.decode("utf-8").replace("\"","")
    # Assign the provided folder name as the EDS folder path.
    storage["target_area"] = folder
    if os.path.exists(folder):
        return json.dumps({"success":1})
    else:
        return json.dumps({"success":0})

@app.route("/store_path", methods=["POST"])
def store_path():
    # All stored paths should be in coords relative to TIFF.
    xy_path = request.json["xy"]
    path_def = request.json["def"]
    element = storage["element"]
    # Scale web/display coordinates to data coordinates.
    xy_path = scale_display_to_data(xy_path)
    # Account an empty path accompanying the request, which is treated as a valid request for path deletion.
    try:
        # Invert y-axis direction.
        xy_path = invert_y_dir(xy_path)
        path_exists = True
    except IndexError:
        path_exists = False
    # If only one point is provided, not path can be constructed.
    if len(xy_path) == 1:
        path_exists = False
    # Define command dictionary for handling different types of paths.
    # Structure: "<pathname>":[<function to handle path>,<item(s) to delete from storage to clear the path>]
    path_defs = {"vein":[handle_vein_path,"vein_def","vein_line"],
                 "profile":[handle_profile_path,"profile_length","profile_line"],
                 "crossvein":[handle_vein_span,"crossvein_line","vein_window"],
                 "crossselvage":[handle_selvage_span,"crossselvage_line","selvage_window"]}
    # Iterate through the types of paths.
    for candidate_path_def,commands in path_defs.items():
        # Check if the active path type matches the recieved path type.
        if path_def == candidate_path_def:
            # If so, check if a path has been provided.
            if not path_exists:
                # If not, reset the relevant items in storage to clear any pre-existing data for the active path type.
                for delete_item in commands[1:]:
                    storage[delete_item] = None
                # Exit the function.
                return json.dumps({"success":1})
            # Handle the received xy path using the relevant function.
            handling_function = commands[0]
            handling_function(xy_path)
            break
    return json.dumps({"success":1})

@app.route("/load_vein_path", methods=["POST"])
def load_vein_path():
    ''' Load a vein path file and return data in a list [[<x>,<y>],] format (or return a string declaration of failure).
    '''
    # Parse requested filepath into string.
    filepath = json.loads(request.data.decode("utf-8"))
    # Check for the existence of the file and exit the function if not found.
    if not os.path.exists(filepath):
        return json.dumps({"data":""})
    # Load the contents of the file as a string.
    with open(filepath) as infile:
        raw_xy_path = infile.read()
    # Set the vein definition path as the provided filepath.
    storage["vein_def"] = filepath
    # Parse path into a numpy array.
    xy_path = np.array([p.split("\t") for p in raw_xy_path.split("\n") if p]).astype(float)
    # Store vein path specification in GMT coords in list form.
    storage["vein_line"] = xy_path.tolist()
    # Rescale the coordinates to the web display coordinate system.
    transformed_xy_path = scale_data_to_display(xy_path)
    # Issue the path in web display coords as the endpoint response.
    return json.dumps({"data":transformed_xy_path.tolist()})

@app.route("/load_elem", methods=["POST"])
def load_elem():
    ''' Load the requested element into local storage as well as into the web display by providing the relevant tiff -> jpg image for display.
    '''
    # Load element and remove double quotes around it to isolate the element.
    element = request.data.decode("utf-8").replace("\"","").title()
    # Set as active element.
    storage["element"] = element
    # Check if a converted version of the element's EDS map already exists.
    # The existence of this converted map also signifies the validation/normalisation of the .tiff filename (to "<element symbol>.tiff"), so no more needs to be done in this function.
    region = Path(storage["target_area"]).parts[-1]
    if not os.path.exists(get_jpg_path()):
        # Check that an element has been provided.
        if element:
            # If not, convert element map to .jpg and also normalise the .tiff file name to <element symbol>.tiff.
            parent_folder = storage["target_area"]
            # Find EDS filenames that might be relevant for the active element.
            target_files = find_element_tiff(parent_folder,element)
            if len(target_files) < 1:
                # If no relevant EDS maps are found, declare as such.
                print("Failed to find target element map")
                # Set the active element to be a placeholder.
                storage["element"] = "placeholder"
                # Exit the function with an unsuccessful response.
                return json.dumps({"success":0})
            if len(target_files) == 1:
                # If a single relevant map is found, declare success.
                print("Successfully found target element map")
            elif len(target_files) > 1:
                # If multiple relevant maps for the element are found, declare an issue but not lack of success.
                print("Ambiguity over which target element map is desired, selecting arbitrarily")
            # Take the first (and likely only) relevant tiff file for the active element as the one to use for converting to JPG.
            tiff_source = target_files[0]
            ensure_eds_map_has_jpg(tiff_source)
        else:
            # Declare failure if no element was provided.
            print("No element provided")
            # Exit the function with an unsuccessful response.
            return json.dumps({"success":0})
    return json.dumps({"success":1,"region":region})

@app.route("/gmt_profile", methods=["POST"])
def gmt_profile():
    ''' Execute GMT profiling on the currently stored config. This procedure also produces a png file for web display.
    '''
    # Execute GMT profiling.
    output_name = run_gmt_profile()
    # Respond with the PNG filename for display.
    return json.dumps({"filename":output_name})

@app.route("/analyze_profile", methods=["POST"])
def analyze_profile():
    ''' Analyze output of GMT profiling on the currently stored config. This procedure also produces a png file for web display.
    '''
    # Execute GMT profile analysis.
    output_name = analyse_gmt_profile()
    # Respond with the PNG filename for display.
    return json.dumps({"filename":output_name})

@app.route("/plot_svg", methods=["POST"])
def plot_svg():
    ''' Parse stored coordinates for a requested geometry (using the `storage` dictionary key into SVG path definition format to respond with.
    '''
    # Load key from request.
    key = request.json["storage_key"]
    print(key)
    # Should be a key with coords.
    xy_path = np.array(storage[key])
    # Assume the path is defined.
    try:
        # Uninvert GMT-inverted y coords if necessary (i.e. for all coords except the stored vein coords).
        if key != "vein":
            xy_path = invert_y_dir(xy_path)
    except IndexError:
        # In case the path is undefined, return an empty SVG path def.
        return json.dumps({"path":""})
    # Transform data coords to display coords.
    transformed_xy_path = scale_data_to_display(xy_path)
    # Convert path coordinates array to SVG path def.
    path = path_to_svg(transformed_xy_path)
    return json.dumps({"path":path})

@app.route("/save_bkg_rect", methods=["POST"])
def save_bkg_rect():
    ''' Convert the recieved path into a rectangle defined by the bounding box of the first and last points, and store this rectangle as part of the collection of rectangles that represent the background/matrix of active vein in the EDS map.
    '''
    # Receive the two xy coordinates defining opposite corners of the rectangle. If there are more than two coordinates, opposite corners will be taken as the first and last coordinates.
    xy_path = request.json["xy"]
    # Transform the coordinates from the web display to data coordinates (i.e. scaled to 1 unit = 1 px in the EDS map).
    transformed_xy_path = scale_display_to_data(xy_path)
    # Store rectangles to active in data coords.
    try:
        storage["bkg_rect"].append([transformed_xy_path[0],transformed_xy_path[-1]])
    except IndexError:
        # In case only one or zero points are provided.
        return json.dumps({"path":""})
    # Find bounding rectangle of the first and last point in untransformed display coordinates.
    p1,p2 = xy_path[0],xy_path[-1]
    xs = [p1[0],p2[0]]
    ys = [p1[1],p2[1]]
    x0,x1 = min(xs),max(xs)
    y0,y1 = min(ys),max(ys)
    rect = [(x0,y0),(x0,y1),(x1,y1),(x1,y0),(x0,y0)]
    # Convert coordinates list to SVG path def.
    path = path_to_svg(rect)
    # Store rectangle SVG paths to active in display coords.
    storage["bkg_rect_svgs"].append(path)
    # Respond with the SVG collection of active rectangles.
    return json.dumps({"path":" Z ".join(storage["bkg_rect_svgs"])})

@app.route("/clear_bkg_rect", methods=["POST"])
def clear_bkg_rect():
    ''' Reset the collection of active rectangles used to define the background/matrix (non-vein) region of the EDS map.
    '''
    storage["bkg_rect_svgs"] = []
    storage["bkg_rect"] = []
    return json.dumps({"success":1})

@app.route("/save_svg", methods=["POST"])
def save_svg():
    ''' Save a received SVG string to an Inkscape-specific* SVG file after adding the active EDS map as an underlay.

    * in terms of linking rather than embedding the converted EDS map (JPG).
    '''
    # Receive SVG string.
    svg = request.json["svg"]
    # Find the absolute path to the tiff2jpg converted active EDS map.
    cwd = os.getcwd()
    region = Path(storage["target_area"]).parts[-1]
    # Get the loaded element from the webpage.
    element = request.json["element"]
    converted_eds_map_jpg = f"{cwd}/static/{element}-{region}.tiff.jpg"
    # Embed the EDS map into the background (i.e. just after the svg start tag) of the SVG.
    svg = re.sub("(<svg.*?>)",f"""\g<1>
<image xlink:href="{converted_eds_map_jpg}"/>
""",svg)
    # Construct output filename that will permit ordering by name into a chronological ordering within each vein.
    vein_id = os.path.basename(storage["vein_def"])
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    out_svg = os.path.join("tmp",f"0-{vein_id}-{formatted_time}.svg")
    # Save the SVG.
    with open(out_svg,"w") as outfile:
        outfile.write(svg)
    # Store the SVG name.
    storage["display_svg"] = out_svg
    # Respond with the filename of the saved SVG.
    return json.dumps({"savefile":out_svg})

@app.route("/smooth_map", methods=["POST"])
def smooth_map():
    ''' Smooth, recenter and plot the EDS map, and plot the vein path over it.
    '''
    outfile = smooth_element(storage["element"])
    return json.dumps({"filename":outfile})

@app.route("/smooth_all", methods=["POST"])
def smooth_all():
    ''' Smooth, recenter and plot the EDS map (+plot vein path) for all elements available in the active EDS directory.
    '''
    # Reset plotting environment
    plt.close("all")
    # Load the SVG filename for the SVG plot of everything on the web display.
    svg_fname = storage["display_svg"]
    # Convert svg to pdf.
    pdf_fname = svg_to_pdf(svg_fname)
    # Perform EDS smoothing etc. on all available EDS maps.
    outfiles = [smooth_element(element) for element in list_available_elements()]
    # Convert all to pdfs.
    pdf_outfiles = [svg_to_pdf(outfile) for outfile in outfiles]
    # Collate all into one pdf.
    vein_id = os.path.basename(storage["vein_def"])
    subprocess.call(["pdfunite",pdf_fname,*pdf_outfiles,f"smoothed-maps-{vein_id}.pdf"])
    return json.dumps({"success":1})

@app.route("/process_all_elem", methods=["POST"])
def profile_all_elem():
    ''' Execute GMT profiling and subsequent profile analysis on all available elements' EDS maps.
    '''
    # Produce and save a plot of the vein path and all profiles lines across it taken by GMT.
    fname = plot_cross_profiles(os.path.basename(storage["vein_def"]),storage["element"],storage["target_area"])
    # Load the SVG filename for the SVG plot of everything on the web display.
    svg_fname = storage["display_svg"]
    # Convert svg to pdf.
    pdf_fname = svg_to_pdf(svg_fname)
    # Isolate the region id of the active EDS maps.
    region = Path(storage["target_area"]).parts[-1]
    # Path to the active EDS maps.
    parent_folder = storage["target_area"]
    # Generate list of available tiff files in the EDS maps folder.
    target_files = list_tifffiles(parent_folder)
    # Generate list of element names extracted from the available tiff files.
    elements = elements_from_filenames(target_files)
    # Iterate through the tiff files/element names.
    for element,target_file in zip(elements,target_files):
        storage["element"] = element
        # Make sure that the active EDS TIFF has a normalized file name (for GMT profiling).
        ensure_normalized_eds_name(target_file)
        # Execute GMT profiling.
        run_gmt_profile()
        # Execute profile analysis.
        analyze_profile()
    # Collate all profiles for the active vein id alongside the vein path + profiles plot into an org report that gets compiled into a PDF.
    org_fname = collate_profiles([fname])
    # Find the PDF report file name.
    org_pdf_fname = org_fname.replace(".org",".pdf")
    # Collate the plot of the web display items and the profiles repot.
    subprocess.call(["pdfunite",pdf_fname,org_pdf_fname,f"all-profiles-{org_pdf_fname}"])
    return json.dumps({"success":1})

@app.route("/clear_all", methods=["POST"])
def clear_all():
    ''' Completely reset the active storage/config to its state on load.
    '''
    storage = init_storage()
    return json.dumps({"success":1})

@app.route("/load_new_region", methods=["POST"])
def load_new_region():
    ''' Return to the folder loading page to permit loading of a new EDS map region.
    '''
    clear_all()
    return json.dumps({"success":1})

if __name__=="__main__":
    # from waitress import serve
    # serve(app,host="127.0.0.1",port=5000)
    # app.run(debug=True)
    app.run()
