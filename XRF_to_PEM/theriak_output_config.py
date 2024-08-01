#####
'''
Effectively an enhanced config file handling mineral phase groupings (into more generic names) and colormapping.
Variables intended for export:
 `mergers` - dict for how minerals are to be grouped. {"<broader group>":["<finer group>"]}
 `mineral_colors` - dict colormap for minerals from the broader groups. {"<broader group>":<color spec>}
'''
#####

# Dict describing how to group minerals into broader groups. {"<broader group>":["<finer group>"]}
mergers = {"Cpx":["[cen1]","cen","[jd]","[di]","[kjd1]",
                  "[hen]", # High enstatite is high clinoenstatite (high-temperature orthoenstatite not confirmed experimentally) [Nespolo 2021; https://doi.org/10.1016/B978-0-12-409548-9.12409-1]
                  "[cess]", # Clinoesseneite
                  ],
           "Opx":["[en]","[odi]"],
           "Grt":["[py]", # Pyrope
                  "[alm]",
                  "[andr]",],
           "Mica":["[naph]", # Aspidolite (-Na-trium-ph-logopite)
                   "[phl]",
                   ],
           "Cam":["[tr]","[prgm]"],
           "Tlc":["[tan]", # Talc -n-ormal?
                   ],
           "Clc":["[clin]"],
           "Srp":["[liz]","[atgf]"],
           "Hem":["[dhem]","[hem]"],
           "Mag":["[imt]"],
           "Chu":["[chum]"],
           "Ol":["[fo]","[fa]"],
           "Brc":["[br]"],
           "Gth":["gth"],
           ".Phi_A":["phA"],
           "Ab":["[ab]"],
           "Ilm":["[dilm]"],
           "H2O":["[H2O]"],
           "Lws":["[law]"],
#           "Corun":["[cor]"],
           "Spl":["[nsp]"]
           }

# Describe special plotting colors for the mineral groupings.
special_colors = {"Ol":"20a107ff",
                  "Opx":"a6600cff",
                  "Cpx":"1a4103ff",
                  "Cam":"8be560ff",
                  "Srp":"b5d814ff",
                  "Mag":"165c71ff",
                  "Clc":"e2bd63ff",
                  "Hem":"89231aff",
                  "H2O":"0fd3d3ff",
                  "Tlc":"d2fcdcff",
                  "Grt":"616161ff",
                  "Ilm":"bababaff",
                  "Mica":"343434ff",
                  }

####
# Everything below is not for change under normal circumstances.
####

import matplotlib as mpl
import numpy as np

def list_cmap(name,n):
    ''' Convert a matplotlib colormap into list.

    cmap | :str: | name of the matplotlib colormap.
    n | :int: | number of colors to resample into the list.

    Returns: list [<RGBA numpy arrays>]
    '''
    # Access colormap by name and resample to the desired number of colors.
    cmap = mpl.colormaps[name].resampled(n)
    # Convert colormap to list.
    try:
        cmap_l = list(cmap.colors)
    except AttributeError:
        # In case of LinearSegmentedColormap, which don't have the `colors` attribute.
        cmap_l = [cmap(i) for i in range(n)]
    return cmap_l

def shuffle_cmap_l(cmap_l):
    ''' Shuffle a list of colormaps to avoid similar colors representing subsequent items that are placed next to each other.

    Splits the colormap in half and then alternates between colors from the lower half and upper half going up.

    cmap_l | :list: or :numpy.array: [<color specification>] | list of color specifications in a list colormap.

    Returns: numpy.array [<color specification>]
    '''
    # Force cast the colormap into a list (in case it's a numpy array).
    cmap_l = list(cmap_l)

    # Determine the number of colors in the colormap is odd or not (in preparation for halving)...
    odd = False
    if len(cmap_l)%2 != 0:
        # If so, add a dummy color to the end (repeat the final color).
        odd = True
        cmap_l.append(cmap_l[-1])
    # Find the halfway index of the color list.
    halfway = len(cmap_l)//2
    # Shuffle the color list by halving then alternating colors from the front and end half (and then shaping it to the correct form for a list of colors).
    cmap_l = np.array(list(zip(cmap_l[:halfway],cmap_l[halfway:]))).reshape(len(cmap_l),len(cmap_l[0]))
    if odd:
        # Remove the dummy color added to even out list with odd number of colors.
        cmap_l = cmap_l[:-1]
    return cmap_l

###
# FOR EXPORTED USE: `mergers`
###
# Cover pure phases in addition to endmember phases provided in the mineral groupings.
mergers = {k:v + [s.replace("[","").replace("]","") for s in v] for k,v in mergers.items()}

# Ensure correct format for the hexademical color codes.
special_colors = {k:"#"+v for k,v in special_colors.items()}

# Identify which minerals don't have a specially assigned color.
remaining_minerals = sorted(set(mergers.keys()) - set(special_colors.keys()))
# Assign colors to the remaining minerals based on a shuffled Greys colormap.
cmap = mpl.colormaps["Greys"].resampled(len(remaining_minerals)+4)
colors = [cmap(i+2) for i in range(len(remaining_minerals))]
colors = shuffle_cmap_l(colors)
remaining_colors = {mineral:colors[i] for i,mineral in enumerate(remaining_minerals)}


###
# FOR EXPORTED USE: `mineral_colors`
###
# Combine the specially-assigned colors, "automatically" assigned colors as well as a color for trace components (labelled by "Trace") into a usable {"<mineral>":"<hex color specification>"} colormap.
mineral_colors = special_colors | remaining_colors | {"Trace":"#ff00ffff"}
