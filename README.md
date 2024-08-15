
# Table of Contents

1.  [AGM\_magnetic\_analysis/notebook.ipynb](#org03a05c3)
2.  [Clc\_orientations/notebook.ipynb](#org4f35297)
3.  [EBSD\_mapping/notebook.ipynb](#org1eb6062)
4.  [EPMA\_data\_analysis/notebook.ipynb](#org1ca4b10)
5.  [XRF\_to\_PEM/XRF-PCA.ipynb](#org6940395)
6.  [XRF\_to\_PEM/XRF\_plots.ipynb](#org9ca0100)
7.  [XRF\_to\_PEM/notebook.ipynb](#org3ab56b0)
8.  [XRF\_to\_PEM/results.ipynb](#orgbcb6af4)
9.  [density\_color/notebook.ipynb](#org8e94573)
10. [model\_mag\_effect/notebook.ipynb](#orga963203)
11. [reflector\_grain\_orientations/notebook.ipynb](#org038f80e)
12. [vein\_transects/server.py](#org4d5cf15)

Code written for a UNIX environment with Python 3.11 and R 4.2.2.


<a id="org03a05c3"></a>

# AGM\_magnetic\_analysis/notebook.ipynb

Parse magnetic data produced using the Alternating Gradient Magnetometer at Cambridge, including:

-   Producing a Day Plot
    -   Figure 24 of `write.pdf`
-   Plotting FORC collections and FORC diagrams (though with much less sophistication compared to FORCinel).
    -   `raw_forcs.pdf`
-   Vizualising the output of FORC PCA performed in FORCinel.
    -   Figure 26 of `write.pdf`


## Dependencies


### Python (pip) Modules

`codecs`, `cv2`, `io`, `matplotlib.collections`, `matplotlib.patches`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `re`, `scipy.interpolate`, `scipy.signal`, `sys`


<a id="org4f35297"></a>

# Clc\_orientations/notebook.ipynb

Plot half rose diagrams of manually drawn SVG lines that represent the long axes of different features (most notably clinochlore grains) on a thin section scan.

-   Figure 13 of `write.pdf`


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `mplstereonet`, `numpy`, `re`


<a id="org1eb6062"></a>

# EBSD\_mapping/notebook.ipynb

Plot EBSD orientations (about 3 orthogonal axes) with a ternary (R,G,B) colormap.

-   Figure 10 of `supp.pdf`


## Dependencies


### Python (pip) Modules

`io`, `matplotlib.pyplot`, `mpltern.datasets`, `numpy`, `pandas`, `re`


<a id="org1ca4b10"></a>

# EPMA\_data\_analysis/notebook.ipynb

Match EBSD data to compositions from Webmineral to try and identify minerals.

-   Section 2 of `supp.pdf`

Also compare compositional properties between early- and late-formed clinochlore grains (determined petrographically).

-   Figure 5 of `supp.pdf`


## Dependencies


### Python (pip) Modules

`io`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `re`, `requests`


<a id="org6940395"></a>

# XRF\_to\_PEM/XRF-PCA.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

-   Perform PCA on selected oxide components from XRF data.
    -   Figure 16 of `write.pdf`


## Dependencies


### R Packages

`robCompositions`, `dplyr`


### System Packages

`R`


<a id="org9ca0100"></a>

# XRF\_to\_PEM/XRF\_plots.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

-   Produces ternary plots of selected oxide components from XRF data.
    -   Section 5 of `supp.pdf`
-   Plots compositions onto Benard et al.'s composition biplots (for provenance).
    -   Figure 15 of `write.pdf`


## Dependencies


### Python (pip) Modules

`IPython.display`, `io`, `matplotlib.pyplot`, `matplotlib`, `mpltern`, `numpy`, `os`, `pandas`, `re`, `subprocess`


### System Packages

`inkscape`


<a id="org3ab56b0"></a>

# XRF\_to\_PEM/notebook.ipynb

Normalize raw XRF compositional data for Phase Equilibrium Modelling (PEM), and execute PEM on all serpentinite samples from the XRF data.

-   Data for Figure 23 of `write.pdf`


## Dependencies


### Python (pip) Modules

`json`, `matplotlib.pyplot`, `matplotlib`, `mpltern`, `numpy`, `os`, `pandas`, `re`, `shutil`, `subprocess`


### System Packages

`wine`


<a id="orgbcb6af4"></a>

# XRF\_to\_PEM/results.ipynb

Requires `XRF_to_PEM/notebook.ipynb` to be run beforehand.

-   Plots the output of PEM in volume stackplots.
    -   Figure 23 of `write.pdf`


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `subprocess`


### System Packages

`wine`


<a id="org8e94573"></a>

# density\_color/notebook.ipynb

Process data to find sample density and average color properties, and then compare the two for use as serpentinization indicators.

-   Figure 14, Tables 4 and 5 of `write.pdf`


## Dependencies


### Python (pip) Modules

`PIL`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `scipy.optimize`, `sklearn.metrics`, `sys`


<a id="orga963203"></a>

# model\_mag\_effect/notebook.ipynb

Model the effect of subduction-aligned remanent magnetization on hypothetical, surface-measured magnetization.

-   Figure 33 of `write.pdf`
-   Section 11 of `supp.pdf`


## Dependencies


### Python (pip) Modules

`matplotlib.pyplot`, `matplotlib`, `numpy`


<a id="org038f80e"></a>

# reflector\_grain\_orientations/notebook.ipynb

Filter detected reflector grains by area, orientation and elongation, and plot them on top of reflected light scans.

-   `grain-orientation-checking.pdf`


## Dependencies


### Python (pip) Modules

`PIL`, `cv2`, `matplotlib.collections`, `matplotlib.patches`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `shapely`, `tqdm`


<a id="org4d5cf15"></a>

# vein\_transects/server.py

Produce and analyze vein profiles.

-   Figure 19 of `write.pdf`


## Dependencies


### Python (pip) Modules

`PIL`, `datetime`, `flask`, `io`, `json`, `matplotlib.pyplot`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `re`, `scipy.signal`, `shapely`, `subprocess`


### System Packages

`emacs`, `gmt`, `imagemagick`, `inkscape`, `pdflatex`, `pdfunite`
