import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os

class AreaProcessor():
    def __init__(self,img_path,pix2mm=None,dilate_erode=None):
        # Path to the manually thresholded scan containing just reflectors.
        self.img_path = img_path
        # Conversion factor from pixels to mm.
        self.pix2mm = pix2mm
        # Whether to perform the dilate then erode on grains. Either None type or the size of the kernel used for dilation and erosion.
        self.dilate_erode = dilate_erode

        # Folder containing thresholded images, as well as folders that contain contour definition .npy files.
        basedir = os.path.dirname(self.img_path)
        # Name of the thresholded image file.
        filename = os.path.basename(self.img_path)

        # Folder name containing filtered contours after being processed by the requested dilation-erosion pathway.
        if dilate_erode:
            prepend = f"modified-{dilate_erode}-"
        else:
            prepend = "unmodified-"
        # Path to file containing filtered contours after being processed by the requested dilation-erosion pathway.
        self.load_path = os.path.join(basedir,"filtered_data",prepend+filename)

    def _extract_contours(self):
        ''' Extract and save the contours of a thresholded scan. Both the "real size" (slightly smaller than actual reflector patches) and enlarged (extracted from image that's been scaled up by a factor of 2) contours are extracted and saved, which permits patch area computation.

        returns contours : list of cv2-specification "small" contour patches
        larger_contours : list of cv2-specification "large" contour patches
        '''
        # Load image as BGRA image.
        img = cv2.imread(self.img_path,cv2.IMREAD_UNCHANGED)
        # Isolate alpha channel.
        binary_img = alpha_channel = img[:,:,3]

        # Fill any holes within reflector patches - this ensures there's a same number of contours and larger contours.
        # Must be uint8 for use by cv2.dilate
        enclaves_filled = binary_fill_holes(binary_img).astype(np.uint8)
        if self.dilate_erode:
            # Define kernel as square with dimension self.dilate_erode.
            kernel = np.ones((self.dilate_erode,self.dilate_erode),np.uint8)
            # Attempt to fill fractures to bring grains back to original, unfractured sizes by dilating ; subjective.
            unfrag = cv2.dilate(enclaves_filled,kernel,iterations=1)
            # Fill in any internal holes in grains.
            enclaves_filled = binary_fill_holes(unfrag).astype(np.uint8)
            # Attempt to remove extra material added from previous step.
            final_img = cv2.erode(enclaves_filled,kernel)
        else:
            final_img = enclaves_filled

        # Even when drawing *external* contours, the raster nature of the array is ignored:
        #   -----
        # 1 |x|x|
        #   -----
        # 0 |x|x|
        #   -----
        #    0 1
        # Becomes [0,0],[1,1], such that the area is 1.
        # This is fixed by determining the number of pixels the patch contour covers in a 2x scaled up image, then performing the operation (larger_contour_areas + 1 - 2 * contour_areas)/2 to find the number of pixels in the original patch.

        larger_img = cv2.resize(final_img,tuple(np.array(final_img.shape)*2)[::-1])
        # Extract non-zero areas; note cv2.CHAIN_APPROX_NONE prevents simplification of the vector definition of raster patches.
        contours,_ = cv2.findContours(final_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        larger_contours,_ = cv2.findContours(larger_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Save the contour definitions for future processing.
        np.save(f"{self.load_path}",np.array(contours,dtype=object))
        np.save(f"{self.load_path}-larger.npy",np.array(larger_contours,dtype=object))
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
                contours,larger_contours = self._extract_contours()
        except FileNotFoundError:
            # If contour definition files don't exist, write them and load contours.
            contours,larger_contours = self._extract_contours()
        return contours,larger_contours

    def find_areas(self):
        ''' Compute patch areas using "small" and "large" contours and then convert to mm^2 if necessary.

        returns patch_areas : list of patch areas
        units : the units that the returned patch areas are in
        '''
        # Load contours.
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

    def area_studied(self):
        ''' Determine the convex hull area in the same units as used to define the coordinates of the polygon.

        returns area (numerical) : convex hull area of the list of contour polygons in mm2
        '''
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
