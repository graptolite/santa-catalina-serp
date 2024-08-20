import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os

import sys
sys.path.insert(0,os.path.join("..","section-scans-full"))
from util_funcs import *

# Note: this processing does not truncate the maximum area and saves to ./contours-*modified* directories - i.e. the raw data.
# Additional processing in the notebook filters by a maximum area, saving to ./filtered_data/contours-*modified*.png.npy
#        The areas saved in the .json file is after maximum area filtering (i.e. corresponds to contour data in ./filtered_data/contours-*modified*.png.npy)

class AreaProcessor():
    def __init__(self,img_path,pix2mm=None,dilate_erode=None):
        # Path to the manually thresholded scan (image) containing just reflectors.
        self.img_path = img_path
        # Conversion factor from pixels to mm.
        self.pix2mm = pix2mm
        # Whether to perform the dilate then erode on grains. Either None type or the size of the kernel used for dilation and erosion.
        self.dilate_erode = dilate_erode
        # Folder containing manually thresholded scans.
        filename = os.path.basename(self.img_path)

        # Path to folder containing unmodified and unfiltered contours.
        # The initially loaded contours will always the unmodified version as modified contours are produced after filtering and then dilation-erosion and saved to another location.
        self.initial_load_path = os.path.join("contours-unprocessed",filename)
        # Make this folder if necessary.
        if not os.path.exists("contours-unprocessed"):
            os.mkdir("contours-unprocessed")

        # Folder name containing filtered contours after being processed by the requested dilation-erosion pathway.
        if dilate_erode:
            folder = f"contours-modified-{dilate_erode}"
        else:
            folder = "contours-unmodified"
        # Path to folder containing filtered contours after being processed by the requested dilation-erosion pathway.
        self.load_path = os.path.join(folder,filename)
        # Make this folder if necessary.
        if not os.path.exists(folder):
            os.mkdir(folder)

    def _extract_initial_contours(self):
        ''' Extract and save the contours of a thresholded scan. Both the "real size" (slightly smaller than actual reflector patches) and enlarged (extracted from image that's been scaled up by a factor of 2) contours are extracted and saved, which permits patch area computation. No filtering is applied.

        returns contours : list of unfiltered cv2-specification "small" contour patches
        larger_contours : list of unfiltered cv2-specification "large" contour patches
        '''
        # Load image as BGRA image.
        img = cv2.imread(self.img_path,cv2.IMREAD_UNCHANGED)
        # Isolate alpha channel.
        binary_img = img[:,:,3]

        # Fill any holes within reflector patches - this ensures there's a same number of contours and larger contours.
        final_img = binary_fill_holes(binary_img).astype(np.uint8)

        # Even when drawing *external* contours, the raster nature of the array is ignored:
        #   -----
        # 1 |x|x|
        #   -----
        # 0 |x|x|
        #   -----
        #    0 1
        # Becomes [0,0],[1,1], such that the area is 1.
        # This is fixed by determining the number of pixels the patch contour covers in a 2x scaled up image, then performing the operation (larger_contour_areas + 1 - 2 * contour_areas)/2 to find the number of pixels in the original patch.

        # Scale up the image by a factor of two whilst preserving the aspect ratio of the image.
        larger_img = cv2.resize(final_img,tuple(np.array(final_img.shape)*2)[::-1])
        # Extract non-zero areas; note cv2.CHAIN_APPROX_NONE prevents simplification of the vector definition of raster patches (hence these contours can be used to redraw the patches with pixel-precision).
        contours,_ = cv2.findContours(final_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        larger_contours,_ = cv2.findContours(larger_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Save these initial contour definitions for future processing.
        np.save(f"{self.initial_load_path}",np.array(contours,dtype=object))
        np.save(f"{self.initial_load_path}-larger.npy",np.array(larger_contours,dtype=object))
        return contours,larger_contours

    def load_initial_contours(self,force_overwrite=False):
        ''' Load the unfiltered "small" and "large" contours from .npy files and filter to just contours with "small" contour area above a threshold.

        force_overwrite (bool) : whether to regenerate the initial contour definition files even if they already exist.

        returns contours : list of unfiltered cv2-specification "small" contour patches
        larger_contours : list of unfiltered cv2-specification "large" contour patches
        '''
        try:
            # Try loading initial contour definition files if not requested to overwrite them.
            if not force_overwrite:
                contours = np.load(f"{self.initial_load_path}.npy",allow_pickle=True)
                larger_contours = np.load(f"{self.initial_load_path}-larger.npy",allow_pickle=True)
            else:
                # If requested to overwrite the files, overwrite them and reload contours.
                contours,larger_contours = self._extract_initial_contours()
        except FileNotFoundError:
            # If initial contour definition files don't exist, write them and load contours.
            contours,larger_contours = self._extract_initial_contours()
        return contours,larger_contours

    def load_contours(self,force_overwrite=False):
        ''' Load the "small" and "large" contours from .npy files and filter to just contours with "small" contour area above a threshold.

        force_overwrite (bool) : whether to regenerate the processing-path-relevant contour definition files even if they already exist.

        returns contours : list of cv2-specification "small" contour patches
        larger_contours : list of cv2-specification "large" contour patches
        '''
        try:
            if not force_overwrite:
                # Try loading contour definition files if not requested to overwrite them.
                contours = np.load(f"{self.load_path}.npy",allow_pickle=True)
                larger_contours = np.load(f"{self.load_path}-larger.npy",allow_pickle=True)
            else:
                # If requested to overwrite the files, overwrite them and reload contours.
                contours,larger_contours = self._process_contours()
        except FileNotFoundError:
            # If contour definition files don't exist, write them and load contours.
            contours,larger_contours = self._process_contours()
        return contours,larger_contours

    def _draw_contours(self,contours,img,color=(255,255,255,255)):
        ''' Draw cv2-specification contours as white patches on an image.

        contours : cv2-specification contour definitions with pixel units
        img : image (in the form of a numpy array) to overlay the contours on
        color : color of contour patches (defaults to RGBA/BGRA white)

        returns img : image with contours overlain
        '''
        [cv2.fillPoly(img,[np.reshape(c,(c.shape[0],2))],color=color) for c in contours]
        return img

    def _place_contours_on_blank_img(self):
        ''' Generate image containing just contours with area >= 5 px.

        returns contour_img : image with contours plotted as opaque white patches in a transparent background with the same image dimensions as the full sample scan image
        '''
        # Load unfiltered contours.
        contours0,larger_contours0 = self.load_initial_contours()
        # Determine areas of unfiltered_contours.
        areas,units = self.find_areas(contours0,larger_contours0)
        # Hardcode filtering out of "noise" (grains <5 px area).
        size_filter = construct_minmax_filter(areas,5*0.001**2)
        # Filter for just contours >= 5 px in area.
        contours = list_of_list_filter(contours0,size_filter)
        # Load sample image (used to determine image dimensions when displaying the filtered reflectors).
        img = cv2.imread(self.img_path)
        # Create a blank image onto which filtered reflectors are to be plotted.
        blank_img = np.zeros([*img.shape[:2],4])
        # Plot contours onto this blank image.
        contour_img = self._draw_contours(contours,blank_img)
        return contour_img

    def _dilate_erode_patches(self,input_img):
        ''' Expand opaque patch sizes to merge nearby patches (threshold affected by the kernel size) and then erode all patches (including joined ones) to mitigate the effect of adding area to the patches as a whole.

        input_img : cv2 specification image with opaque patches in a transparent background. Must contain numerical data of type np.uint8

        returns output_img : cv2 specification image after dilation-erosion the kernel specified on initiation of this class.
        '''
        # Define kernel as square with dimension self.dilate_erode.
        kernel = np.ones((self.dilate_erode,self.dilate_erode),np.uint8)
        # Attempt to fill fractures to bring grains back to original, unfractured sizes by dilating ; subjective.
        unfrag = cv2.dilate(input_img,kernel,iterations=1)
        # Fill in any internal holes in grains.
        enclaves_filled = binary_fill_holes(unfrag).astype(np.uint8)
        # Attempt to remove extra material added from previous step.
        output_img = cv2.erode(enclaves_filled,kernel)
        return output_img

    def _process_contours(self):
        ''' Process contours through the desired dilation-erosion pathway.

        returns contours : list of cv2-specification "small" contour patches
        larger_contours : list of cv2-specification "large" contour patches
        '''
        # Generate image containing just contours with area >= 5 px.
        contour_img = self._place_contours_on_blank_img()

        # Isolate alpha channel.
        binary_img = contour_img[:,:,3]

        # Fill any holes within reflector patches - this ensures there's a same number of contours and larger contours.
        # Must be uint8 for use by cv2.dilate
        enclaves_filled = binary_fill_holes(binary_img).astype(np.uint8)

        # Perform dilation-erosion as request.
        if self.dilate_erode:
            final_img = self._dilate_erode_patches(enclaves_filled)
        else:
            final_img = enclaves_filled

        # Scale up the image by a factor of two whilst preserving the aspect ratio of the image.
        larger_img = cv2.resize(final_img,tuple(np.array(final_img.shape)*2)[::-1])
        # Extract non-zero areas; note cv2.CHAIN_APPROX_NONE prevents simplification of the vector definition of raster patches.
        contours,_ = cv2.findContours(final_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        larger_contours,_ = cv2.findContours(larger_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Save the contour definitions for future processing.
        np.save(f"{self.load_path}",np.array(contours,dtype=object))
        np.save(f"{self.load_path}-larger.npy",np.array(larger_contours,dtype=object))

        return contours,larger_contours

    def find_areas(self,contours=None,larger_contours=None):
        ''' Compute patch areas using "small" and "large" contours and then convert to mm^2 if necessary.

        contours : list of cv2-specification "small" contour patches
        larger_contours : list of cv2-specification "large" contour patches

        returns patch_areas : list of patch areas
        units : the units that the returned patch areas are in
        '''
        # Load contours if necessary.
        if contours is None or larger_contours is None:
            contours,larger_contours = self.load_contours()

        # Determine areas of the "small" contours.
        contour_areas = np.array(list(map(cv2.contourArea,contours)))
        # Determine areas of the "large" contours.
        larger_contour_areas = np.array(list(map(cv2.contourArea,larger_contours)))
        # Determine areas of the reflector patches.
        patch_areas = (larger_contour_areas + 1 - 2 * contour_areas)/2
        # Convert units of area if needed, and declare whether this conversion happened through the units.
        if self.pix2mm:
            patch_areas = np.array(patch_areas) * (self.pix2mm**2)
            units = "mm"
        else:
            units = "px"
        return patch_areas,units

    def area_studied(self,contours=None):
        ''' Determine the convex hull area in the same units as used to define the coordinates of the polygon.

        contours : list of cv2-specification "small" contour patches

        returns area (numerical) : convex hull area of the list of contour polygons in mm2
        '''
        # Load contours if None provided.
        if contours is None:
            contours,_ = self.load_contours()

        # Flatted the polygons such that the convex hull is for the collection of polygons vertices.
        points = []
        for c in contours:
            points += c[:,0].tolist()
        points = np.array(points)
        # Determine the convex hull of the polygon vertices.
        hull = ConvexHull(points)
        # Extract the points defining the hull's vertices.
        polygon = points[hull.vertices]
        # Find the area of the convex hull as defined by its vertices.
        area = Polygon(polygon).area * (self.pix2mm**2)
        return area
