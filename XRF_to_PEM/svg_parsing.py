import re
import numpy as np
import pandas as pd

def close_coords(coords_list):
    ''' Make sure a list of coordinates is closed (i.e. the last coord matches the first coord).

    coords_list | :list: [(<coords>)] | List of coordinates (which are themselves :list:-like).

    Returns: :list: [(<coords>)]
    '''
    # Check if the last coordinate is the same as the first.
    if coords_list[-1] != coords_list[0]:
        # If not, append the first coordinate to the end of the coordinate list.
        coords_list.append(coords_list[0])
    return coords_list

def parse_svg(svg_file):
    ''' Read SVG file into a string.

    svg_file | :str: | Path to SVG file.

    Returns: :str:
    '''
    # Load and read SVG file into memory.
    with open(svg_file) as infile:
        svg = infile.read()
    return svg

def find_tag(tag,svg,self_closing=True):
    ''' Find all matches to an SVG tag (including tag contents) under the assumption of no nested tags. Suitable e.g. for paths, ellipses, rects etc. Not necessarily suitable for groups unless the groups have been manually processed.

    tag | :str: | Name of the tag to search for.
    svg | :str: | SVG to search within.
    self_closing | :bool: | Whether the tag is self closing (ends with />) or not (ends with </tag> and contains content).

    Returns: :list: [:str:]
    '''
    # Assume tag closes with </tag>.
    close = f"</{tag}>"
    # Unless the tag is specified to be self closing.
    if self_closing:
        close = "/>"
    # Find all matches for the tag (+contents) and return the list of matches.
    return re.findall(f"<{tag}[\s\S]*?{close}",svg)

def find_tag_nestable(tag,svg):
    ''' Find the first occurrance of a tag of interest and its entire contents (which may contain nested tags of the same type).

    tag | :str: | Name of the tag to search for.
    svg | :str: | SVG to search within.

    Returns: :str:
    '''
    # Find all starting positions of the tag of interest.
    tag_opens = [m.start() for m in re.finditer(f"<{tag}",svg)]
    opens = ["o"] * len(tag_opens)
    # Find all closing positions of the tag of interest.
    tag_closes = [m.end() for m in re.finditer(f"</{tag}>",svg)]
    closes = ["c"] * len(tag_closes)
    # Combine the start and close positions (whilst saving their operation in `tag_operations`)
    tag_positions = tag_opens + tag_closes
    tag_operations = opens + closes
    # Sort the operations by their position.
    sorted_positions,sorted_operations = zip(*sorted(zip(tag_positions,tag_operations)))
    # Initiate tracker for the current tag level.
    level = 0
    # Iterate through the list of operations.
    for pos,op in zip(sorted_positions,sorted_operations):
        # If the tag is an opening tag, increase the level by 1.
        # This is the operation of the first tag.
        if op=="o":
            level += 1
        else:
            # Otherwise the tag is a closing tag, so reduce the level by 1.
            level -= 1
        # Find the tag that closes (by returning the level to zero) the first/opening tag.
        if level == 0:
            closure_pos = pos
            break
    # Extract the part of the SVG string that is fully encompassed by the first instance of the tag of interest.
    full_tag = svg[tag_opens[0]:closure_pos]
    return full_tag

def get_attributes(attribs,svg_tag):
    ''' Extract the value of an attribute or list of attributes from an svg_tag tag.

    attribs | :str: or :list: [:str:] | Attribute name or list of such to extract corresponding value(s) from.
    svg_tag | :str: | SVG tag that contains these attributes.

    Returns: :str: or :tuple: [:str:]
    '''
    # Assume that there are multiple attributes.
    list_attribs = True
    # If only one attribute is provided (i.e. a string rather than list)...
    if isinstance(attribs,str):
        # Create a single-element list.
        attribs = [attribs]
        # Declare that there aren't multiple attributes.
        list_attribs = False
    # Initialize list to store found attribute values in.
    attrib_vals = []
    # Iterate through requested attributes.
    for attrib in attribs:
        # Find the value of the active attribute (defaulting to a blank string if none found).
        attrib_match = re.search(f"\s{attrib}=\"([\s\S]*?)\"",svg_tag)
        attrib_val = ""
        if attrib_match:
            attrib_val = attrib_match.group(1)
        # Store the value.
        attrib_vals.append(attrib_val)
    # Return the values as a tuple (is multiple attributes were requested).
    if list_attribs:
        return tuple(attrib_vals)
    # Otherwise return as a string.
    return attrib_vals[0]

def parse_rect(rect):
    ''' Extract characteristic geometry (width, height, x and y) from an SVG rectangle object. (x,y) for the coordinates for the top left corner of the rectangle.

    rect | :str: | SVG rectangle object/tag.

    Returns: :dict: {"<feature>":<value>}
    '''
    attribs = ["width","height","x","y"]
    return {attrib:float(x) for attrib,x in zip(attribs,get_attributes(attribs,rect))}

def gridify(geoms,n_cols):
    ''' Find the index order for a list of geometries that orders them in a grid by rows then columns.

    geoms | :list: [{"x":<x>,"y":<y> ...}] | List of geometry definitions where the x and y coords are provided in a dictionary.
    n_cols | :int: | Number of columns within the grid.

    Returns: :list: [<index>]
    '''
    # Create array of xy coordinates for all geometries.
    xy_s = np.array([(g["x"],g["y"]) for g in geoms])
    # Save these coordinates in a dataframe structure for easier processing, and sort them in increasing y.
    df = pd.DataFrame(xy_s,columns=["x","y"]).sort_values("y")
    # Find the expected number of rows.
    n_rows = int(len(df)/n_cols)
    # Split the geometries into rows, ordering within-row in the process.
    row_dfs = [df.iloc[0+i*n_cols:n_cols+i*n_cols,:].sort_values("x") for i in range(n_rows)]
    # Find the index order that gives rise to the grid order.
    grid_sorted_idxs = list(pd.concat(row_dfs).index)
    return grid_sorted_idxs
