Code written for a UNIX environment with [Python](https://www.python.org/) 3.11 and [R](https://www.r-project.org/) 4.2.2.

The purpose of each notebook with respect to the main text and supplementary information (SI) is stated within each section below.


# AGM\_magnetic\_analysis/notebook.ipynb

Parse magnetic data produced using the Alternating Gradient Magnetometer at Cambridge, including:

-   Vizualising the output of FORC PCA performed in FORCinel.
    -   Figure 3 of main text
-   Producing a Day Plot
    -   Figure 5 of main text
-   Plotting FORC collections and FORC diagrams (though with much less sophistication compared to FORCinel) for quick view.


## Dependencies


### Python (pip) Modules

`codecs`, `cv2`, `io`, `matplotlib.collections`, `matplotlib.patches`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `re`, `scipy.interpolate`, `scipy.signal`, `sys`


# Clc\_orientations/notebook.ipynb

Plot half rose diagrams of manually drawn SVG lines that represent the long axes of different features (most notably clinochlore grains) on a thin section scan.

-   Supports the results of main text (clinochlore and magnetite approximately aligned in a fabric)


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `mplstereonet`, `numpy`, `re`


# EBSD\_mapping/notebook.ipynb

Plot EBSD orientations (about 3 orthogonal axes) with a ternary (R,G,B) colormap.

-   Figure S10 of SI


## Dependencies


### Python (pip) Modules

`io`, `matplotlib.pyplot`, `mpltern.datasets`, `numpy`, `pandas`, `re`


# EPMA\_data\_analysis/notebook.ipynb

For the locations of EPMA spot analyses, see https://doi.org/10.5281/zenodo.13685581.

Process EPMA data.

-   Table S5 and S6 of SI

Match EPMA data to compositions from Webmineral to try and identify minerals.

-   Text S3 of SI

Also compare compositional properties between early- and late-formed clinochlore grains (determined petrographically).

-   Not in main text or SI but may be of interest


## Dependencies


### Python (pip) Modules

`io`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `re`, `requests`


# EPMA\_data\_analysis/magnetite\_as\_oxides.ipynb

Convert element wt% to oxide wt% for the "magnetites" and investigate compositional differences in the magnetites of partially vs heavily serpentinized rocks.

-   Cr comparison: Figure S4 of SI


## Dependencies


### Python (pip) Modules

`../XRF_to_PEM/composition_processor.py`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `sys`


# XRF\_to\_PEM/notebook.ipynb

Normalize raw XRF compositional data for Phase Equilibrium Modelling (PEM), and execute PEM on all serpentinite samples from the XRF data.

-   <span class="underline">P-T</span> history plot for Figure 1 of main text
-   Protolith positions on ultramafic ternary diagrams for Figure S1 of SI
-   Data for Figure 6 of main text and Figures S10-S16 of SI ($PT_{0.7}$ can be changed in the `Serpentinisation Path` section)


## Dependencies


### Python (pip) Modules

`json`, `matplotlib.pyplot`, `matplotlib`, `mpltern`, `numpy`, `os`, `pandas`, `re`, `shutil`, `subprocess`


### System Packages

`wine`


# XRF\_to\_PEM/XRF\_plots.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

-   Produces ternary plots of selected oxide components from XRF data.
    -   Section 5 of `supp.pdf`
-   Plots compositions onto Benard et al.'s composition biplots (for provenance). Not important for the paper but may be of interest for future study.


## Dependencies


### Python (pip) Modules

`IPython.display`, `io`, `matplotlib.pyplot`, `matplotlib`, `mpltern`, `numpy`, `os`, `pandas`, `re`, `subprocess`


### System Packages

`inkscape`


# XRF\_to\_PEM/XRF-PCA.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

Performs PCA on XRF compositions of the 5 samples, identifying possible groups corresponding to different regions of the composition biplots. Not important for the paper but may be of interest for future study.


## Dependencies


### R Packages

`robCompositions`, `dplyr`


### System Packages

`R`


# XRF\_to\_PEM/results.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

-   Plots the output of PEM in volume stackplots.
    -   Figure 6 of main text and Figures S10-S16 of SI


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `subprocess`


### System Packages

`wine`


# density\_color/notebook.ipynb

Process data to find sample density and average color properties, and then compare the two for use as serpentinization indicators.

-   Figure S3 of SI


## Dependencies


### Python (pip) Modules

`PIL`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `scipy.optimize`, `sklearn.metrics`, `sys`


# model\_mag\_effect/notebook.ipynb

Model the effect of subduction-aligned remanent magnetization on hypothetical, surface-measured magnetization.

-   Supports the subsection Subduction Zone Magnetic Structure (of Discussion) of main text.


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `matplotlib`, `numpy`


# reflector\_processing/\*

These notebooks extract reflector (magnetite) grain shapes from reflected light scans. The results of this notebook are not important for the main text or SI but may be of interest to future work.

The full size reflected light scans (.tif files) and derived files should be downloaded beforehand and placed into the created folder `DATASETS/RL_scans` https://doi.org/10.5281/zenodo.13329989.

-   I.e. the folder structure should be:
    -   `DATASETS`
        -   `AGM`
        -   `...`
        -   `RL_scans`
            -   `M08.tif`
            -   `M08.png`
            -   `...`
        -   `...`

This collection of notebooks captures the iterative improvement of the reflector grain extraction code in the following order:

1.  section-scans-runthrough-example/working.ipynb: demonstrate the algorithm on a small example (extract of real data). This algorithm takes the extracted reflector grains and then first performs a dilation (expanding the grain area to merge detected grains that are close together) followed by erode (shrinking the grains by the same amount to remove excess area without removing connections) - this is performed to join together grains that were spurriously disconnected due to imperfections in the magnetite polish surface, imaging etc.
2.  reflector\_processing/section-scans-full/working.ipynb: apply the algorithm (captured in reflector\_processing/section-scans-full/area\_processing.py) to all samples' RL scans and visualize differences between results from different samples.
3.  reflector\_processing/section-scans-full-evaluation/working.ipynb: compare the results of the reflector grain extraction algorithm to manually extracted grains for one instance.
4.  reflector\_processing/section-scans-full-evaluation/further-analysis.ipynb: visually compare the results of the reflector grain extraction algorithm with slightly different parameters to expected to find the optimal parameter. Also develop a new processing algorithm where the fine grains are removed prior to the dilate-erode process. `reflector_processing/section-scans-full/working.ipynb` must have been run for the dilate-erode option in this notebook to work (as it reads data generated by that notebook).
5.  reflector\_processing/section-scans-refined-full/working.ipynb: repeat the process of section-scans-full but with a removal of fine grains ("noise") prior to the dilate-erode process. Also save post-processed contours without truncating maximum area - i.e. saving an extended reflector grain dataset. These extended datasets may also be used by reflector\_grain\_orientations/notebook.ipynb.


## reflector\_processing/section-scans-runthrough-example/working.ipynb


### Dependencies

1.  Python (pip) Modules

    `PIL`, `cv2`, `json`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `scipy.ndimage`, `scipy.optimize`, `scipy.spatial`, `shapely.geometry`, `subprocess`, `textwrap`

2.  System Packages

    `ImageMagick`, `Hugin`


## reflector\_processing/section-scans-full/working.ipynb


### Dependencies

1.  Python (pip) Modules

    `cv2`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy.ndimage`, `scipy.optimize`, `scipy.spatial`, `scipy.stats`, `shapely.geometry`


## reflector\_processing/section-scans-full-evaluation/working.ipynb


### Dependencies

1.  Python (pip) Modules

    `cv2`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy.ndimage`, `scipy.optimize`, `scipy.spatial`, `shapely.geometry`, `sys`


## reflector\_processing/section-scans-full-evaluation/further-analysis.ipynb


### Dependencies

1.  Python (pip) Modules

    `cv2`, `json`, `matplotlib.pyplot`, `numpy`, `os`, `scipy.ndimage`, `scipy.optimize`, `scipy.spatial`, `shapely.geometry`, `sys`


## reflector\_processing/section-scans-refined-full/working.ipynb


### Dependencies

1.  Python (pip) Modules

    `cv2`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy.ndimage`, `scipy.optimize`, `scipy.spatial`, `shapely.geometry`, `sys`


## reflector\_processing/section-scans-Kretz1969/working.ipynb

Notebook applying some methods described in the Kretz 1969 paper to check for the evenness of grain distributions. Results not particularly important beyond demonstrating heterogeneity in spatial distribution of grains. Must be run after `reflector_processing/section-scans-refined-full/working.ipynb` has been run with `dilate_erode = 10`


### Dependencies

1.  Python (pip) Modules

    `cv2`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy.optimize`, `scipy.spatial`, `scipy.stats`, `shapely.geometry`, `shapely`, `sys`


# vein\_transects/server.py

Produce and analyze vein profiles and maps from EDS maps.

-   Table S7 and Figure S19 of SI

For the EDS maps used in this study, see https://doi.org/10.5281/zenodo.13351361.

For plots that have already been made, see https://doi.org/10.5281/zenodo.13685560.


## Dependencies


### Python (pip) Modules

`PIL`, `datetime`, `flask`, `io`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `re`, `scipy.signal`, `shapely`, `subprocess`


### System Packages

`emacs`, `gmt`, `imagemagick`, `inkscape`, `pdflatex`, `pdfunite`
