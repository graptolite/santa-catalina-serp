import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from theriak_output_config import mergers,mineral_colors

# Standardising font sizes.
plt.rcParams.update({"axes.labelsize":12,
                     "axes.titlesize":12,
                     "xtick.labelsize":12,
                     "ytick.labelsize":12,})

###
# Input handling
###

class TheriakAPI():
    def __init__(self,theriak_dir="./theriak",ptx_commandfile="path.txt",directive_file="path.directive"):
        # Directory holding the theriak executable and associated files.
        self.theriak_dir = theriak_dir
        # Name for the file into which P-T-X varying commands are to be saved.
        self.ptx_commandfile = ptx_commandfile
        # Name for the file into which theriak directives will be saved.
        self.directive_file = directive_file
        # Initialize list into which P-T-X varying commands will be accumulated.
        self.commands = []

    def clear_PTX_commands(self):
        ''' Clear commands.

        Returns: None
        '''
        self.commands = []

    def add_PTX_command(self,composition,P,T,n=1):
        ''' Construct the commands for one part of a P-T-X path and save to the commands.

        composition | :str: | Theriak-Domino composition string to use for this P-T path
        P | :list: [Numerical](length=2) or Numerical | Starting (and ending if list) pressure.
        T | :list: [Numerical](length=2) or Numerical | Starting (and ending if list) temperature.
        n | :int: | Number of steps to take along the P-T path.

        Returns: None
        '''
        # In case one out of {P,T} is provided as a single value and n > 1.
        # If P is not provided as a list, force P to be a two-element list.
        if not isinstance(P,list):
            P = [P,P]
        # If T is not provided as a list, force T to be a two-element list.
        if not isinstance(T,list):
            T = [T,T]

        # Construct the basic commands for the P-T-X point.
        command = [f"COMP  {composition}   *",f"TP  {T[0]}  {P[0]}"]
        # If a linear ramp of P-T points is to be visited by this command segment ...
        if n > 1:
            # Construct the command to do so.
            command += [f"TP  {T[1]}  {P[1]}  {n}"]
        # Store the new commands.
        self.commands.extend(command)
        # Print the new commands that were just stored.
        print("Updated commands with command:\n%s" % "\n".join(command))
        return

    def save_PTX_commandfile(self,new_commandfile=None):
        ''' Save all currently stored commands to the (P-T-X) command file.

        commandfile | :str: | Name (not path) of the commandfile if the originally-provided commandfile name is to be changed.

        Returns: None
        '''
        if new_commandfile:
            self.ptx_commandfile = new_commandfile
        # Construct path to where the commandfile is to be saved.
        commandfile = os.path.join(self.theriak_dir,self.ptx_commandfile)
        # Open and write the stored commands into the commandfile.
        with open(commandfile,"w") as outfile:
            outfile.write("\n".join(self.commands))
        # Print the location/name of the commandfile.
        print("Theriak commands saved in %s" % commandfile)
        return

    def create_directive(self,thermodynamic_db="td-test.txt"):
        ''' Construct and save a Theriak directive file.

        thermodynamic_db | :str: | Name of the file containing the thermodynamic database for use by theriak.exe and stored in the same directory as theriak.exe.

        Returns: None
        '''
        directive_file = os.path.join(self.theriak_dir,self.directive_file)
        with open(directive_file,"w") as outfile:
            outfile.write("%s\n%s" % (thermodynamic_db,self.ptx_commandfile))
        print("Theriak directive saved in %s" % directive_file)
        return

    def save_all(self,thermodynamic_db="td-test.txt"):
        ''' Create and save both the (P-T-X) path commandfile as well as the Theriak directive file.

        thermodynamic_db | :str: | Name of the file containing the thermodynamic database for use by theriak.exe and stored in the same directory as theriak.exe.

        Returns: None
        '''
        self.save_PTX_commandfile()
        self.create_directive(thermodynamic_db)
        return

    def check_file_exists(self,filepath,file_purpose,creator_functions=[]):
        ''' Check if a file exists and raise the relevant error if not, which should provide the relevant help message.

        filepath | :str: | Path to the file whose existence is to be checked.
        file_purpose | :str: | Description of the purpose of the file.
        creator_functions | :list: [:str:] | List of the function names that can be used to create the file of interest.

        Returns: None
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError("%s (required: %s) not found. Check that any of the functions {%s} has been run before, or manually check the expected location." % (filepath,file_purpose,", ".join(creator_functions)))
        return

    def execute_theriak(self):
        ''' Execute theriak on the active directive file and then return the output table as a pandas dataframe.

        Returns: :pandas.DataFrame:
        '''
        # Check whether the path commandfile (which the directive file requires) exists or not.
        self.check_file_exists(os.path.join(self.theriak_dir,self.ptx_commandfile),
                               "for the directive file",
                               ["<self>.create_PTX_commandfile()","<self>.save_all()"])
        # Check whether the directive file exists to be called or not.
        self.check_file_exists(os.path.join(self.theriak_dir,self.directive_file),
                               "for calling by theriak.exe",
                               ["<self>.create_PTX_commandfile()","<self>.save_all()"])
        # Store the current directory path ("old working directory").
        oldwd = os.getcwd()
        # Move into the theriak directory.
        os.chdir(self.theriak_dir)
        # Run theriak on the active directory file.
        subprocess.call(["wine","theriak.exe",self.directive_file])
        # Move back to the old working directory.
        os.chdir(oldwd)
        # Parse and return the output table.
        return self.read_theriak_table()

    def read_theriak_table(self):
        ''' Parse the output table from the latest theriak run (or alternative, optionally-provided path) into a pandas dataframe.

        theriak_table | :str: | Filepath to the theriak table.

        Returns: :pandas.DataFrame:
        '''
        theriak_table = os.path.join(self.theriak_dir,"loop_table")
        # Check whether the table file exists to be read.
        self.check_file_exists(theriak_table,
                               "to be read",
                               ["<self>.execute_theriak()"])
        # Load the data outputted from latest theriak run.
        df = read_theriak_table(theriak_table)
        return df

###
# Output handling
###

def read_theriak_table(theriak_table_file):
    ''' Parse the output table from a theriak run into a pandas dataframe.

    theriak_table | :str: | Filepath to the theriak table.

    Returns: :pandas.DataFrame:
    '''
    # Load the data outputted from latest theriak run.
    df = pd.read_csv(theriak_table_file)
    # Clean column names.
    df.columns = [c.replace(" ","") for c in df.columns]
    # Find then remove columns whose content sums to zero.
    empty_cols = df.columns[df.sum()==0].to_list()
    df.drop(empty_cols,axis=1,inplace=True)
    return df

def group_cols(df,groupings=mergers):
    ''' Row-wise summing the numerical contents of different columns into one based on a column grouping specification.

    df | :pandas.DataFrame: | Dataframe containing numerical data in all columns that are to be grouped.
    groupings | :dict: {"<grouped column name>":["<source columns>"]} | Column grouping specification. A source column (columns that are to be combined into a group) cannot belong to more than one group.
    '''
    # Iterate through the grouping specification, which combines source columns into a new, broader-defined column that substitutes the source columns.
    for subs,srcs in groupings.items():
        # Find all source columns (mapping to the actively group column) present in the dataframe.
        active_srcs = list(set([s for s in srcs if s in df.columns]))
        # If the active source columns list is not empty ...
        if len(active_srcs):
            # ... sum the numerical contents of the active source columns into the grouped column.
            df[subs] = df[active_srcs].sum(axis="columns")
            # Remove the active source columns from the dataframe.
            df.drop(active_srcs,axis=1,inplace=True)
    return df

def reorder_df_by_cmap(df,cmap_dict=mineral_colors):
    ''' Reorder a dataframe's columns by their appearance in a dictionary colormap, and isolate only the relevant color mappings.

    df | :pandas.DataFrame: | Dataframe with columns that are partially or all contained within the keys of the dictionary colormap.
    cmap_dict | :dict: {"<column name>":<color spec>} | Dictionary colormap mapping column names to colors.

    Returns: (:pandas.DataFrame:,:dict: {"<column name>":<color spec>})
    '''
    # Find the columns that have a color mapping and list the columns in the order that the appear in the colormap.
    active_maps = [m for m in cmap_dict if m in df.columns]
    # Order the dataframe columns by the colormap-derived ordering.
    df = df[active_maps]
    # Isolate only the relevant color mappings.
    cmap = [cmap_dict[m] for m in active_maps]
    return df,cmap

class TheriakOutput():
    def __init__(self,output_df):
        # Pandas dataframe parsed from the output of a theriak run.
        self.df = output_df

    def extract_conditions(self):
        ''' Isolate the part of the dataframe describing the P-T conditions (as well as step index).

        Returns: :pandas.DataFrame:
        '''
        # Find the names of the columns of interest.
        conditions_columns = [c for c in self.df.columns if c.startswith(":")]
        # Isolate the columns of interest.
        conditions_df = self.df.loc[:,conditions_columns]
        return conditions_df

    def extract_volumes(self,ignore_solids=True):
        ''' Isolate the part of the dataframe describing phase volumes.

        ignore_solids | :bool: | Whether to exclude V_solids from the output dataframe or not (which would otherwise be present if multiple physical phases e.g. a solid and a fluid were simultaneously stabilized).

        Returns: :pandas.DataFrame:
        '''
        # Construct a list of volumes to ignore.
        ignore = ["tot"]
        if ignore_solids:
            ignore.append("solids")
        # Find the names of the volume columns that are not be ignored.
        volume_columns = [c for c in self.df.columns if (c.startswith("V") and c.replace("V_","") not in ignore)]
        # Isolate the volume columns.
        vol_df = self.df.loc[:,volume_columns]
        # Replace the string designations for volume in the column names.
        vol_df.columns = [c.replace("V_","") for c in vol_df.columns]
        return vol_df

    def major_and_trace_volumes(self,critical_fraction=0.05,do_group_cols=True):
        '''
        Obtain and split the volumes dataframe (with V_solids ignored) into a major and trace volume dataframe based on the specified critical volume fraction against which the largest volume fraction of each phase is compared to.

        Volumes are treated as fractions in both of the outputted volume dataframes.

        critical_fraction | :float: | The volume fraction used to differentiate between whether a phase is major or trace. Must be smaller than 1.
        do_group_cols | :bool: | Whether to group the mineral phases or not (under the default grouping of the function `group_cols`).

        Returns: (:pandas.DataFrame:,:pandas.DataFrame:)
        '''
        # Isolate the volumes from the full dataframe.
        vol_df = self.extract_volumes()
        # Perform column grouping if requested.
        if do_group_cols:
            vol_df = group_cols(vol_df)
        # Compute the total volume (in arbitrary units).
        vol_tot = vol_df.sum(axis=1)
        # Initiate a dataframe for trace phase volumes from which major phases are removed.
        trace_df = vol_df.copy()
        # Iterate through the phases with volumes.
        for col in vol_df.columns:
            # Compute the maximum volume fraction of the phase.
            vol_fraction = max(vol_df[col]/vol_tot)
            # Check whether the phase is major or trace and remove the phase from the opposite's dataframe.
            if vol_fraction < critical_fraction:
                vol_df.pop(col)
            else:
                trace_df.pop(col)
        # If trace phases are present.
        if not trace_df.empty:
            # Group the trace phases into one "phase" in the major volume dataframe.
            vol_df["Trace"] = trace_df.sum(axis=1)
        # Normalize the output volumes.
        vol_sum = vol_df.sum(axis=1)
        vol_df = vol_df.div(vol_sum,axis=0)
        trace_df = trace_df.div(vol_sum,axis=0)
        return vol_df,trace_df

    def volume_stackplot(self,x_col,vol_df,vol_cmap,ax=None,legend_cols=1,x_label=""):
        '''
        Create a stackplot of phase volumes from a volume dataframe against a named x variable.

        x_col | :str: | Name of the full dataframe's column for the x variable.
        vol_df | :pandas.DataFrame: | Dataframe with volume fractions.
        vol_cmap | :dict: {"<phase/vol_df column name>":<color spec>} | Colormap for phases/column names in `vol_df`.
        ax | :matplotlib.axes.Axes: | Axis to plot the stackplot onto.
        legend_cols | :int: | Number of columns to split the colormap legend over.
        x_label | :str: | X axis label.

        Returns: :matplotlib.axes.Axes:
        '''
        # Initiate axis if none provided.
        if ax is None:
            _,ax = plt.subplots(1,1)
        # Extract x variable data from the full dataframe.
        x_var = self.df[x_col]
        # If there's only volume data for one point, repeat that point at a slightly shifted x var point to effectively create a bar plot.
        # This is not expected to be relevant to actual display plots and instead is only provided for quick data visualization.
        if len(x_var) == 1:
            x_var = pd.concat([x_var,x_var+1])
            vol_df = pd.concat([vol_df,vol_df])
        # Produce the colormapped volume stackplot.
        ax.stackplot(x_var,vol_df.T.to_numpy(),labels=vol_df.columns,colors=vol_cmap,edgecolor="k",linewidth=0.1)
        # Label the y axis.
        ax.set_ylabel("Volume Fraction")
        # Label the x axis if requested:
        if x_label:
            ax.set_xlabel(x_label)
        # Limit the y axis range to the valid range for a (relative) volume fraction.
        ax.set_ylim([0,1])
        # Display colormap legend with the requested number of entry columns.
        ax.legend(ncol=legend_cols)
        return ax

    def major_and_trace_stackplot(self,x_col=":Temperature",critical_fraction=0.05,group_minerals=True,axs=None):
        ''' Plot major and trace volume stackplots against an x variable (defaulting to temperature) onto the first two axes in a collection of axes.

        x_col | :str: | Name of the full dataframe's column for the x variable.
        critical_fraction | :float: | The volume fraction used to differentiate between whether a phase is major or trace. Must be smaller than 1.
        group_minerals | :bool: | Whether to group the mineral phases or not (under the default grouping of the function `group_cols`).
        axs | :list: [:matplotlib.axes.Axes:] | Collection of axes from which the first two will be plotted onto.

        Returns: :list: [:matplotlib.axes.Axes:]
        '''
        if axs is None:
            _,axs = plt.subplots(2,1)
        major_df,trace_df = self.major_and_trace_volumes(critical_fraction,group_minerals)
        major_df,major_cmap = reorder_df_by_cmap(major_df)
        trace_df,trace_cmap = reorder_df_by_cmap(trace_df)
        self.volume_stackplot(x_col,major_df,major_cmap,axs[0],2)
        if not trace_df.empty:
            self.volume_stackplot(x_col,trace_df,trace_cmap,axs[1],1)
            axs[1].set_ylim(0,critical_fraction*1.1)
        return axs

    def xy_chart(self,x,y,ax=None,plot_style_kwargs_dict={}):
        ''' Plot a chart of x vs y variable after determining the correct way to plot the variables based on their data length.

        x | :list:-like | x data.
        y | :list:-like | y data.
        ax | :matplotlib.axes.Axes: | Axis to plot the chart onto. A new, standalone axis will be generated if none provided.
        plot_styles_kwargs | kwargs to be passed on to the matplotlib plotting function.

        Returns: :matplotlib.axes.Axes:
        '''
        # Initiate axis if none provided.
        if ax is None:
            _,ax = plt.subplots(1,1)
        # Determine the appropriate plotting function depending on the number of datapoints to plot.
        if len(x) == 1:
            plot_func = ax.scatter
        else:
            plot_func = ax.plot
        # Plot the P-T condition(s).
        plot_func(x,y,**plot_style_kwargs_dict)
        return ax

    def plot_PT(self,ax=None,**plot_style_kwargs):
        ''' Plot a chart of the P-T conditions that the dataframe was generated under.

        ax | :matplotlib.axes.Axes: | Axis to plot the chart onto. A new, standalone axis will be generated if none provided.
        plot_styles_kwargs | kwargs to be passed on to the matplotlib plotting function.

        Returns: :matplotlib.axes.Axes:
        '''
        # Set the x var as temperature.
        x = self.df[":Temperature"] # deg C
        # Set the y var as pressure.
        y = self.df[":Pressure"]/1000 # kbar
        # Plot the data.
        ax = self.xy_chart(x,y,ax,plot_style_kwargs)
        # Label axes.
        ax.set_xlabel("T /$^{\circ}$C")
        ax.set_ylabel("P /kbar")
        # Invert y axis such that zero (shallow) is at the top.
        ax.invert_yaxis()
        return ax

    def plot_column(self,column,vs="T",ax=None,y_label=None,**plot_style_kwargs):
        ''' Plot a chart of any column from the dataframe against either temperature (default) or pressure.

        column | :str: or :list: ["<column names>"] | Name of the column(s) in the dataframe to act as the y variable. If multiple columns provided, they must be numerical and will be summed.
        vs | :str: | Shorthand for the physical condition to plot the y variable against. Can only be either "T" or "P".
        y_label | :str: | Y axis label. Defaults to the column(s) name if none provided.
        ax | :matplotlib.axes.Axes: | Axis to plot the chart onto. A new, standalone axis will be generated if none provided.
        plot_styles_kwargs | kwargs to be passed on to the matplotlib plotting function.

        Returns: :matplotlib.axes.Axes:
        '''
        # Set the default y_label if none provided.
        if not y_label:
            y_label = column
        # Isolate the requested x variable column from the dataframe.
        if vs == "T":
            x = self.df[":Temperature"] # deg C
            x_label = "T /$^{\circ}$C"
        elif vs == "P":
            x = self.df[":Pressure"]/1000 # kbar
            x_label = "P /kbar"
        else:
            raise ValueError("`vs` must be \"P\" or \"T\" only")
        if isinstance(column,list):
            # If multiple columns are provided, sum their contents.
            y = self.df[column[0]]
            for c in column[1:]:
                if c in self.df:
                    y += self.df[c].fillna(0)
        else:
            # Isolate the requested y variable column.
            y = self.df[column]
        # Plot the data.
        ax = self.xy_chart(x,y,ax,plot_style_kwargs)# Initiate axis if none provided.
        # Label axes.
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def characterize_output(self,critical_vol_fraction=0.05,axs=None):
        ''' Visually characterize the output. Plot onto a collection of axes: major and trace volume stackplots (after volume normalization), a P-T plot and n_H2O plot (quite specific to this project).

        critical_vol_fraction | :float: | The volume fraction used to differentiate between whether a phase is major or trace. Must be smaller than 1.
        axs | [:matplotlib.axes.Axes:](>3) | Collection of axes to place the plots on. Must contain at least 3 axes (at least 4 axes if the n_H2O plot will be produced.
        '''
        # Initialize an axes collection if none provided.
        if axs is None:
            # Declare the axes collection geometry.
            height_ratios = [6,1,3]
            n_rows = 3
            # Check if a water plot is relevant ...
            if "n_H2O_solids" in self.df or "n_H2O_[H2O]" in self.df:
                # ... and if so, modify the geometry appropriately.
                height_ratios.append(1)
                n_rows += 1
            # Create axes collection.
            _,axs = plt.subplots(n_rows,1,figsize=(5,9),height_ratios=height_ratios,sharex=True)
        # Plot the major and trace volume stackplots in the top two axes.
        self.major_and_trace_stackplot(critical_fraction=critical_vol_fraction,axs=axs[:2])
        # Set the title for the trace volume stackplot.
        axs[1].set_title("Trace Phases",fontsize=12)
        # Remove the y axis label from the trace volume stackplot.
        axs[1].set_ylabel("")
        # Plot P-T path on the next axis.
        self.plot_PT(ax=axs[2],c="k")
        try:
            # Plot H2O-T path on the next axis if relevant.
            self.plot_column(["n_H2O_solids","n_H2O_[H2O]"],ax=axs[3],y_label="n_H2O",c="b")
            y = (self.df["n_H2O_solids"] if "n_H2O_solids" in self.df else 0) + (
                self.df["n_H2O_[H2O]"] if "n_H2O_[H2O]" in self.df else 0)
            axs[3].set_ylim(0,1.1*max(y))
        except (KeyError,IndexError):
            pass
        # Isolate the (previously implicit) x variable.
        x_var = self.df[":Temperature"]
        # Determine x limits, which is the (max,min) range (note the reversal) in case the x variable contains more than one point, otherwise is just close to that single point (x+1,x).
        max_x = max(x_var)
        if len(x_var) == 1:
            max_x += 1
        axs[0].set_xlim([max_x,min(x_var)])
        return axs

# TAPI = TheriakAPI()
# TAPI.add_PTX_command("SI(43.95)MG(47.40)O(140)",[14000,4000],[800,600],10)
# TAPI.save_all()
# df = TAPI.execute_theriak()
# df = TAPI.read_theriak_table()
# output = TheriakOutput(df)
# output.characterize_output()
# plt.show()
