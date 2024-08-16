import json

import sys
sys.path.insert(0,"../section-scans-full")
from area_processing import AreaProcessor
from util_funcs import *

pix2mm = 10**-3

class AreaSlicer():
    def __init__(self,sample,dilate_erode,base_folder="../section-scans-full/",base_imgs_folder="../../DATASETS/RL_scans",AP=None):
        # Folder containing folders that contain contour definition .npy files for different processing pathways.
        self.base_folder = base_folder
        # Folder containing thresholded reflector images.
        self.base_imgs_folder = base_imgs_folder
        # Area processor for the sample of interest.
        if AP is None:
            # Default to using the pixel to mm conversion factor of 0.001 defined outside this class.
            AP = AreaProcessor(f"{self.base_folder}/{sample}.png",pix2mm=pix2mm,dilate_erode=dilate_erode)
        # Compute patch areas.
        self.patch_areas,_ = AP.find_areas()
        # Allow this area processor to be called within this class (could probably use class inheritance to better handle this...).
        self.AP = AP

        # File prepend designating the requested dilation-erosion pathway.
        if dilate_erode:
            file_prepend = f"modified-{dilate_erode}"
        else:
            file_prepend = "unmodified"
        self.file_prepend = file_prepend
        # Sample name/id.
        self.sample = sample

    def load_contours(self):
        ''' Load the "small" contours (plotting of which recreates reflector patches) from the necessary load path for the sample and processing path of interest.

        returns contours : list of cv2-specification "small" contour patches
        '''
        contours = np.load(f"{self.AP.load_path}.npy",allow_pickle=True)
        return contours

    def load_img(self):
        ''' Load full image of relevant sample.

        returns img : cv2 specification image.
        '''
        img = cv2.imread(f"{self.base_imgs_folder}/{self.sample}.jpg")
        return img

    def visualise_patches(self,area_slice,RGB,img=None):
        ''' Overlay the patches that have areas within a desired range on the full image of the sample of interest (specified upon initiation of this class).

        area_slice (list of two numericals) : area range in the form [min,max]
        RGB (list of three numericals) : color of patches in format [R,G,B] where each individual color is out of 255
        img : cv2 specification background image onto which contours are overlain

        returns img : cv2 specification image with contours overlain
        '''
        # Load cv2 specification contours.
        contours = self.load_contours()

        # Construct area filter for the contours.
        minmax_filter = construct_minmax_filter(self.patch_areas,*area_slice)
        # Isolate the contours that pass the filter (i.e. have areas within the specified range).
        ranged_contours = list_of_list_filter(contours,minmax_filter)
        # Check if a background image is provided and if not, load the full sample image.
        if img is None:
            img = self.load_img()
        # Overlay all contours onto the background image as patches of the specified color.
        [cv2.fillPoly(img,[np.reshape(c,(c.shape[0],2))],color=RGB[::-1]) for c in ranged_contours]
        return img
