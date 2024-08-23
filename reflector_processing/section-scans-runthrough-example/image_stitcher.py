# Written by Yingbo Li on the 9th Nov 2023
# For autostitching overlapping scans that are grouped into columns and named in order.

import re
import os
import numpy as np
import subprocess
from PIL import Image
from textwrap import wrap

def compute_control_points(scan1,scan2):
    ''' Generate Hugin project file then find and filter out control points, writing to a project file that contains relatively confident control points (though there may still be outliers/anomalies, which means median- rather than mean offset is preferred).

    scan1 | :str: | Filename of the first image.
    scan2 | :str: | Filename of the second image.

    Returns: :str: | filename of project containing cleaned control points
    '''
    control_points_file = "clean_control_points.pto"
    # Create Hugin project file with two sequential photos.
    subprocess.call(["pto_gen",scan1,scan2,"--fov=10","--projection=0","-o","load_scan.pto"],stdout=subprocess.DEVNULL)
    # Find control points (points representing common points in the two photos).
    subprocess.call(["cpfind","load_scan.pto","-o","control_points.pto"],stdout=subprocess.DEVNULL)
    # Initial removal of control points that are considered invalid.
    subprocess.call(["cpclean","control_points.pto","-o",control_points_file],stdout=subprocess.DEVNULL)
    return control_points_file

class HuginProject():
    def __init__(self,project_file):
        ''' project_file | :str: | Path to Hugin project file.
        '''
        self.project_file = project_file
        with open(project_file) as infile:
            self.raw_project = infile.read()
        return

    def extract_multiline_data(self,pre_comment):
        ''' Extract a block of named data (by a header comment) and return the lines of data within that block.

        pre_comment | :str: | Comment immediately after "#" at the header of the data block.

        Returns: :list: [:str: <data lines>]
        '''
        # Find a match for the data block as designated by a comment before it.
        data_match = re.search("#%s([\s\S]+?)\n\n" % pre_comment,self.raw_project)
        # Check if a match was successfully found or not (in which case raise an error).
        if not data_match:
            raise ValueError("%s not found in %s" % (pre_comment,control_points_file))
        # Convert bulk string into lines of data.
        all_lines = data_match.group(1).strip().split("\n")
        # Remove comment lines.
        data_lines = [l for l in all_lines if not l.startswith("#")]
        return data_lines

    def extract_control_points(self):
        ''' Extract control points from Hugin project file into numpy array.

        Returns: :numpy.array: | Array of control point coordinates in format x1,y1,x2,y2.
        '''
        # Extract control point data block.
        control_points_data = self.extract_multiline_data(" control points")
        # Isolate the parts relevant to the control point coordinates.
        control_points = [l.split(" ")[3:-1] for l in control_points_data]
        # Isolate numerical coordinates and convert to numpy array.
        control_points_arr = np.array([[coord[1:] for coord in point] for point in control_points]).astype(float)
        return control_points_arr

    def extract_image_names(self):
        ''' Extract image names in order.

        Returns: :list: [:str:]
        '''
        # Extract image data block.
        image_data = self.extract_multiline_data(" image lines")
        # Isolate image names.
        image_names = [l.split(" ")[-1][1:].replace('"',"") for l in image_data]
        return image_names

def extract_offset(control_points_arr):
    ''' Extract transform offset defined by a collection of control points that may or may not contain anomalous values.

    control_points_arr | :numpy.array: | Array of control point coordinates.

    Returns: (Numerical,Numerical) | x- and y-offset (respectively).
    '''
    # Verify the existence of control points and if not, state that manual control point selection will be necessary.
    try:
        # Find difference in x points.
        dxs = control_points_arr[:,0] - control_points_arr[:,2]
        # Find difference in y points.
        dys = control_points_arr[:,1] - control_points_arr[:,3]

        # Determine transformation (by coordinate difference) necessary to align the two images.
        # Find the median difference in x coordinates.
        dx = int(np.median(dxs))
        # Find the median difference in y coordinates.
        dy = int(np.median(dys))
        return dx,dy
    except IndexError:
        print("\n\nError\n")
        print("Could not autodetect control points")
        print("Enter `clean_control_points.pto` and manually find control points,\nthen paste in the active Python console (without restarting):")
        print(f"""\n\tmerge_images("{scan1}","{scan2}","{out}",generate_pto=False)\n""")
        # Raise IndexError without more error traceback.
        raise IndexError("No control points found") from None
    return None,None

def align_images(f1,f2,dx,dy,out1="img1.tif",out2="img2.tif"):
    ''' Align two images based on the 2D transformative offset of the second image relative to the first using Pillow.

    f1 | :str: | filename of first image
    f2 | :str: | filename of second image
    dx | Numerical | x-offset
    dy | Numerical | y-offset
    out1 | :str: | output filename for aligned first image
    out2 | :str: | output filename for aligned second image

    Returns: (:str:,:str:) | output filename for aligned first and second image (respectively)
    '''
    # Load the two images specified in the Hugin project in RGBA mode.
    img1 = Image.open(f1).convert("RGBA")
    img2 = Image.open(f2).convert("RGBA")

    # Determine the dimensions of the two images.
    w1,h1 = img1.size
    w2,h2 = img2.size

    # Handle offsetting when using a (relatively) parsiminous canvas size.
    if dx < 0:
        dx_mod = abs(dx)
    else:
        dx_mod = 0

    if dy < 0:
        dy_mod = abs(dy)
    else:
        dy_mod = 0

    # Function to create a (relatively) parsimonious canvas.
    newcanvas = lambda : Image.new(mode="RGBA",size=(w1+dx+2*dx_mod,h1+dy+2*dy_mod))
    # Create canvas for first image and place it at relevant position for overlapping the images. Save this "true"-positioned image.
    canvas1 = newcanvas()
    canvas1.paste(img1,(0+dx_mod,0+dy_mod),img1)
    canvas1.save(out1,compression="jpeg")
    # Create canvas for second image and place it at relevant position for overlapping the images. Save this "true"-positioned image.
    canvas2 = newcanvas()
    canvas2.paste(img2,(dx+dx_mod,dy+dy_mod),img2)
    canvas2.save(out2,compression="jpeg")
    return out1,out2

def blend_images(img1,img2,out):
    ''' Blend together two aligned images using Enblend.

    img1 | :str: | filename of first imag
    img2 | :str: | filename of second image
    out | :str: | filename of output

    Returns: :bool: | whether blending produced an output (and was thus successful) or not
    '''
    # Blend together the two images.
    subprocess.call(["enblend",img1,img2,"-o",f"{out}.tif"])
    # Check if the blended output exists and declare if not.
    if not os.path.exists(f"{out}.tif"):
        print("Enblend processing failed.")
        result = False
    else:
        result = True
    return result

def merge_images(scan1,scan2,out="out",generate_pto=True):
    ''' Control function for blending two unaligned images together.

    scan1 | :str: | filename of first image
    scan2 | :str: | filename of second image
    out | :str: | filename of output
    generate_pto | :bool: | whether to regenerate (and overwrite any existing) output Hugin project file

    Returns: :bool: | whether blending was successful or not
    '''
    # Check if the output exists and skip if so.
    if os.path.exists(f"{out}.tif"):
        print(f"Skipping {out}.tif")
        return

    print(f"Working on {scan1} - {scan2}")

    if generate_pto:
        # Turn off in case there's insufficient overlap to autodetect correlation points, in which case the script will break and manual review in `clean_control_points.pto` will be necessary before rerunning this with ***generate_pto=False***
        control_points_file = compute_control_points(scan1,scan2)

    project = HuginProject(control_points_file)
    # Determine image offset necessary to align the second image with the first.
    dx,dy = extract_offset(project.extract_control_points())

    # Load ordered filenames of each photo within the Hugin project.
    imgs = project.extract_image_names()
    f1 = imgs[0]
    f2 = imgs[1]

    # Align then save images.
    out1,out2 = align_images(f1,f2,dx,dy)

    # Blend aligned images.
    outcome = blend_images(out1,out2,out)
    return outcome

def stitch_column(image_files,first_run=False):
    ''' Stitch a column of images into the next level up.

    image_files | :list: [:str:] | list of image filenames in the column
    first_run | :bool: | whether this function is being run on original images that only partially overlap or not

    Returns: :list: [:str:] | list of output image filenames for restitching to increase the number of original images contained in each single file (if necessary)
    '''
    # Find number of images within.
    n_files = len(image_files)
    # Create list to add output file names to.
    outfiles = []

    if first_run:
        # If the stitching is of the raw scanned images themselves (or columns), then all images will be at the same level of stitching and so can be stitched together sequentially
        # E.g. [a,b,c,d,e] ->  [ab,bc,cd,de]
        for i,f in enumerate(image_files[:-1]):
            # Load second image name.
            f1 = image_files[i+1]
            # Construct output image's filename (without extension), which will be the image ids joined together.
            out = (f+f1).replace(".jpg","").replace(".tif","").replace("image","")
            # Stitch the two images together.
            merge_images(f,f1,out)
            # Register/store the output.
            outfiles.append(out + ".tif")

    else:
        # If the stitching is at a higher level, then just stitch together images to ensure overlap by 1 full image.
        # E.g. [ab,bc,cd,de,ef] -> [abc,cde] (ef not changed)
        for i in range(int(len(image_files)//2)):
            f = image_files[i*2]
            f1 = image_files[i*2+1]
            out = (f+f1).replace(".jpg","").replace(".tif","").replace("image","")
            # Split up the output filename into 4 character-long segments then sort them and join them into the output filename.
            out = "".join(sorted(list(set(wrap(out,4)))))
            # If the length of the output file's name is greater than 100 characters, then take just the first and last 4 character-long segments and use those as the name.
            if len(out)>100:
                out = out[:4] + out[-4:]
            merge_images(f,f1,out)
            outfiles.append(out + ".tif")
    # If the number of files is even, then append the final image into the output images.
    # [ab,bc,cd,de,ef] -> [abc,cde,ef] (ef appended)
    if n_files%2 != 0:
        outfiles.append(image_files[-1])
    return outfiles

def stitch_full(sample,do_convert_output=False):
    ''' Stitch all images represented by a sample into one full scan.

    sample | :str: | name of the sample's parent folder (which contains column folders containing image files)
    convert_output | :bool: | whether to convert the output tiff into a trimmed greyscale map and jpg image or not.

    Returns: :string: | filename of the full scan
    '''
    original_dir = os.getcwd()
    # Enter the sample's directory.
    os.chdir(sample)
    # List and string sort all column folders.
    colfolders = sorted([f for f in os.listdir() if f.startswith("col")])

    ## Stitch entire column of images into the maximum level possible (i.e. all original images contained in one file).
    # Create list to add column output file names to.
    col_outfiles = []
    for colfolder in colfolders:
        os.chdir(colfolder)
        # List and string sort all of the valid image files.
        image_files = sorted([f for f in os.listdir() if f.endswith(".jpg") and f.startswith("image")])
        # Enforce image name formatting standard required for this script to work.
        if image_files[0]=="image.jpg":
            image_files[0] = "image0000.jpg"
            os.rename("image.jpg","image0000.jpg")

        # Stitch together lowest level images.
        outfiles = stitch_column(image_files,first_run=True)
        # Iteratively stitch together images of (average) increasing levels until a full column output is produced.
        while len(outfiles)>1:
            outfiles = stitch_column(outfiles)

        print("Handling column")
        # Check if the column output file is present in the sample directory.
        col_outfile = os.path.join("..",outfiles[0])
        if not os.path.exists(col_outfile):
            # If not, move the column output file from the column directory to the sample directory.
            os.rename(outfiles[0],col_outfile)
            # Placeholder file to declare its creation - this can be deleted to force recreation of the column output file.
            with open(outfiles[0],"w") as outfile:
                outfile.write("placeholder")
        col_outfiles.append(outfiles[0])
        # Exit the column directory.
        os.chdir("..")

    # Iteratively combine the column-stitched images until they are all combined into one.
    while len(col_outfiles)>1:
        col_outfiles = stitch_column(col_outfiles)

    if do_convert_output:
        convert_output(col_outfiles[0],os.path.basename(sample))

    # Return to original directory.
    os.chdir(original_dir)
    return

def convert_output(fname,sample):
    ''' Convert an image file into other formats (trimmed greyscale tiff and jpg) with filenames representative of the sample. Must be called immediately after `stitch_full` without changing filepaths.

    fname | :str: | filename of input image
    sample | :str: | name of sample for use in constructing output filenames

    Returns: None
    '''
    # Convert stitching output into sample-named files.
    # Create (compressed) jpg output file which will not contain an alpha channel.
    if not os.path.exists(f"{sample}.jpg"):
        subprocess.call(["convert",fname,"-trim",f"{sample}.jpg"])
    else:
        print(f"{sample}.jpg already exists")
    # Create grayscale output that still contains the alpha channel.
    if not os.path.exists(f"{sample}.tif"):
        subprocess.call(["convert",fname,"-trim","-colorspace","Gray",f"{sample}.tif"])
    else:
        print(f"{sample}.tif already exists")
    return
