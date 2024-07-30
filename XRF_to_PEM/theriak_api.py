import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from theriak_output_config import mergers,mineral_colors

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

    def add_PTX_command(self,composition,P,T,n):
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

    def save_PTX_commandfile(self):
        ''' Save all currently stored commands to the (P-T-X) command file.

        Returns: None
        '''
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
            raise FileNotFoundError("%s (required: %s) not found. Check that any of the functions {%s} has been run before, or manually check the expected location." % (file_purpose,filepath,", ".join(creator_functions)))
        return

    def execute_theriak(self):
        ''' Execute theriak on the active directive file and then return the output table as a pandas dataframe.

        Returns: pandas.DataFrame
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
        ''' Parse the output table from the latest theriak run into a pandas dataframe.

        Returns: pandas.DataFrame
        '''
        # Check whether the table file exists to be read.
        theriak_table = os.path.join(self.theriak_dir,"loop_table")
        self.check_file_exists(theriak_table,
                               "to be read",
                               ["<self>.execute_theriak()"])
        # Load the data outputted from latest theriak run.
        df = pd.read_csv(theriak_table)
        # Clean column names.
        df.columns = [c.replace(" ","") for c in df.columns]
        # Find then remove columns whose content sums to zero.
        empty_cols = df.columns[df.sum()==0].to_list()
        df.drop(empty_cols,axis=1,inplace=True)
        return df

###
# Output handling
###

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

    Returns: (:pandas.DataFrame:, :dict: {"<column name>":<color spec>})
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

        Returns: pandas.DataFrame
        '''
        # Find the names of the columns of interest.
        conditions_columns = [c for c in self.df.columns if c.startswith(":")]
        # Isolate the columns of interest.
        conditions_df = self.df.loc[:,conditions_columns]
        return conditions_df

    def extract_volumes(self,ignore_solids=True):
        ''' Isolate the part of the dataframe describing phase volumes.

        ignore_solids | :bool: | Whether to exclude V_solids from the output dataframe or not (which would otherwise be present if multiple physical phases e.g. a solid and a fluid were simultaneously stabilized).

        Returns: pandas.DataFrame
        '''
        # Construct a list of volumes to ignore.
        ignore = ["tot"]
        if ignore_solids:
            ignore.append("solids")
        # Find the names of the volume columns that are not be ignored.
        volume_columns = [c for c in self.df.columns if (c.startswith("V") and c.replace("V_","") not in ignore)]
        # Isolate the volume columns.
        vol_df = self.df.loc[:,volume_columns].sort_values(0,axis=1,ascending=False)
        # Replace the string designations for volume in the column names.
        vol_df.columns = [c.replace("V_","") for c in vol_df.columns]
        return vol_df

    def major_and_trace_volumes(self,critical_fraction=0.05,do_group_cols=True):
        '''

        Volumes are treated as fractions in the output.
        '''
        vol_df = self.extract_volumes()
        if do_group_cols:
            vol_df = group_cols(vol_df)
        vol_tot = vol_df.sum(axis=1)
        trace_df = vol_df.copy()
        for col in vol_df.columns:
            vol_fraction = max(vol_df[col]/vol_tot)
            if vol_fraction < critical_fraction:
                vol_df.pop(col)
            else:
                trace_df.pop(col)
        vol_df["Trace"] = trace_df.sum(axis=1)
        # Normalize the output volumes.
        vol_sum = vol_df.sum(axis=1)
        vol_df = vol_df.div(vol_sum,axis=0)
        trace_df = trace_df.div(vol_sum,axis=0)
        return vol_df,trace_df

    def volume_stackplot(self,x_var,vol_df,vol_cmap,ax=None,legend_cols=1):
        '''

        vol_df | :pandas.DataFrame: | Dataframe with volume fractions.
        '''
        # Initiate axis if none provided.
        if ax is None:
            _,ax = plt.subplots(1,1)
        ax.stackplot(x_var,vol_df.T.to_numpy(),labels=vol_df.columns,colors=vol_cmap,edgecolor="k",linewidth=0.1)
        ax.set_ylabel("Volume Fraction")
        ax.set_ylim([0,1])
        ax.legend(ncol=legend_cols)

    def major_and_trace_stackplot(self,x_col=":Temperature",critical_fraction=0.05,group_minerals=True,axs=None):
        if axs is None:
            _,axs = plt.subplots(2,1)
        major_df,trace_df = self.major_and_trace_volumes(critical_fraction,group_minerals)
        major_df,major_cmap = reorder_df_by_cmap(major_df)
        trace_df,trace_cmap = reorder_df_by_cmap(trace_df)
        x_var = self.df[x_col]
        if len(x_var) == 1:
            x_var = pd.concat([x_var,x_var+1])
            major_df = pd.concat([major_df,major_df])
            trace_df = pd.concat([trace_df,trace_df])
        self.volume_stackplot(x_var,major_df,major_cmap,axs[0],2)
        if not trace_df.empty:
            self.volume_stackplot(x_var,trace_df,trace_cmap,axs[1],1)
        return axs

    def xy_chart(self,x,y,ax=None,plot_style_kwargs_dict={}):
        ''' Plot a chart of x vs y variable after determining the correct way to plot the variables based on their data length.

        x | :list:-like | x data.
        y | :list:-like | y data.
        ax | :matplotlib.axes.Axes: | Axis to plot the chart onto. A new, standalone axis will be generated if none provided.
        plot_styles_kwargs | kwargs to be passed on to the matplotlib plotting function.

        Returns: matplotlib.axes.Axes
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

        Returns: matplotlib.axes.Axes
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
        return ax

    def plot_column(self,column,vs="T",ax=None,**plot_style_kwargs):
        ''' Plot a chart of any column from the dataframe against either temperature (default) or pressure.

        column | :str: | Name of the column in the dataframe to act as the y variable.
        vs | :str: | Shorthand for the physical condition to plot the y variable against. Can only be either "T" or "P".
        ax | :matplotlib.axes.Axes: | Axis to plot the chart onto. A new, standalone axis will be generated if none provided.
        plot_styles_kwargs | kwargs to be passed on to the matplotlib plotting function.

        Returns: matplotlib.axes.Axes
        '''
        # Isolate the requested x variable column from the dataframe.
        if vs == "T":
            x = self.df[":Temperature"] # deg C
            xlabel = "T /$^{\circ}$C"
        elif vs == "P":
            x = self.df[":Pressure"]/1000 # kbar
            xlabel = "P /kbar"
        else:
            raise ValueError("`vs` must be \"P\" or \"T\" only")
        # Isolate the requested y variable column.
        y = self.df[column]
        # Plot the data.
        ax = self.xy_chart(x,y,ax,plot_style_kwargs)# Initiate axis if none provided.
        # Label axes.
        ax.set_xlabel(xlabel)
        ax.set_ylabel(column)
        return ax


TAPI = TheriakAPI()
TAPI.add_PTX_command("SI(43.95)MG(47.40)O(140)",14000,800,1)
TAPI.save_all()
df = TAPI.execute_theriak()

output = TheriakOutput(df)
output.plot_PT()
output.plot_column("n_fo")
output.major_and_trace_stackplot()
