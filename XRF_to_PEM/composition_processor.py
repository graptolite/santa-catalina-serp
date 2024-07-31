import pandas as pd
import numpy as np
import re

# Default oxides list passed to CompositionProcessor
oxides = ["SiO2","Al2O3","CaO","MgO","Fe2O3","FeO","K2O","Na2O","TiO2","MnO","H2O","P2O5","CO2","O2"]

# Default dictionary database of atomic masses taken from the IUPAC (Meija et al 2016, Atomic weights of the elements 2013, Pure Appl. Chem. 2016; 88(3): 265–2).
Ar = {"Si":28.085,
      "Ti":47.867,
      "Al":26.982,
      "Fe":55.8452,
      "Mn":54.938,
      "Mg":24.305,
      "Ca":40.0784,
      "Na":22.990,
      "K":39.098,
      "P":30.974,
      "C":12.011,
      "H":1.008,
      "O":15.999,
      }

def normalise_dict_vals(d):
    ''' Normalize the values in a dictionary such that their sum is 100.
    d | :dict: {<key>:<numerical value>} | dictionary with only numerical type values.

    Returns: dict
    '''
    return dict(zip(d.keys(),100*np.array(list(d.values()))/sum(d.values())))

class Molecule():
    def __init__(self,molecule):
        # Molecule name string e.g. Fe2O3.
        self.molecule = molecule

    def __repr__(self):
        return f"<Molecule> {self.molecule}"

    def split_elements(self):
        ''' Split molecular formula into a list of (element,proportional number of atoms) tuples. Supports decimal moles, but not fractional.

        Returns: list
        '''
        import re
        # Search for elements and their relative proportions.
        elements = re.findall("([A-Z][a-z]*)([0-9\.]*)",self.molecule)
        return elements

    def Mr(self,Ar=Ar):
        ''' Compute the molecular mass by summing the masses of all constituent atoms.

        Ar | :dict: {"<Element>":<atomic mass>} | Database of atomic masses for elements.

        Returns: float
        '''
        # Get the list of (element,proportional number of atoms).
        elements = self.split_elements()
        try:
            # Sum the atomic mass of each element multiplied by the number of atoms per molecule under the assumption that all elements have corresponding atomic mass data in the provided Ar database.
            sum_Ar = sum([Ar[element[0]] * (float(element[1]) if len(element[1]) else 1) for element in elements])
        except KeyError:
            raise KeyError("Not all elements in the molecular are accounted for in the Ar database. Check the Ar database and retry.")
        return sum_Ar

class CompositionProcessor():
    ''' Translation of the method from Richard Palin's Excel Spreadsheet into Python (though without any guarantees of complete accuracy). '''
    def __init__(self,correct_apatite=True,ignore_P=True):
        # Whether to remove CaO under the assumption that all P is apatite.
        self.correct_apatite = correct_apatite
        # Whether to remove P2O5 from consideration (after handling any apatite correction). Should usually be set to True.
        self.ignore_P = ignore_P
        # Private (dict) storage for the moles of each oxide.
        self.mol_oxides = None

    def load_composition(self,wt_oxides,oxides):
        ''' Store the computed moles of each oxide into private storage dict in format {"<oxide>":<relative moles>}.

        Kept as a separate function to the mole computation function to permit calling of that function to return a variable.

        wt_oxides | :dict: {"<oxide>":<weight%>} | Oxide weight% data
        oxides | :list: ["<oxide>"] | List of oxides of interest, where each oxide must contain only at most one element other than oxygen, and that other element must be appear at the start of the formula. All other oxides will be skipped.

        Returns: None
        '''
        self.mol_oxides = self.get_moles(wt_oxides,oxides)

    def get_moles(self,wt_oxides,oxides=oxides):
        ''' Compute moles of each oxide from a dictionary database of oxide wt%.

        wt_oxides | :dict: {"<oxide>":<weight%>} | Oxide weight% data
        oxides | :list: ["<oxide>"] | List of oxides of interest, where each oxide must contain only at most one element other than oxygen, and that other element must be appear at the start of the formula. All other oxides will be skipped.

        Returns: dict {"<oxide>":<relative quantity/moles>}
        '''
        # Check for oxides that will be skipped.
        skipped_oxides = sorted(set(wt_oxides) - set(oxides))
        if len(skipped_oxides) > 0:
            print("Oxides that are not being considered in the input formula: %s" % str(skipped_oxides))
        # Initialize dictionary into which mole data will be stored.
        mol_oxides = {}
        # Iterate through the oxides of interest.
        for oxide in oxides:
            # For the oxides of interest that are in the data ...
            if oxide in wt_oxides:
                # ... compute moles (proportional/relative) by dividing weight by oxide Mr.
                mol_oxides[oxide] = wt_oxides[oxide]/Molecule(oxide).Mr()
        return mol_oxides

    def lookup_moles(self,oxide):
        ''' Find the number of moles associated with an oxide from private storage.

        oxide | :str: | Chemical formula of the oxide to search for.

        Returns: int or float
        '''
        return self.mol_oxides[oxide]

    def update_moles(self,oxide,new_moles):
        ''' Update the number of moles associated with an oxide in private storage.

        oxide | :str: | Chemical formula of the oxide to replace.
        new_moles | :int: or :float: | Number of moles to replace the existing data with.

        Returns: None
        '''
        self.mol_oxides[oxide] = new_moles
        return

    def get_standardised_oxides(self):
        ''' Normalize the total (proportional/relative) moles of oxide in private storage to 100.

        Returns: dict {"<oxide>":<relative quantity/moles>}
        '''
        # Check if private storage contains data, otherwise raise an error.
        if self.mol_oxides is None:
            raise UnboundLocalError("No composition is loaded in self.mol_oxides. Load a composition of wt% oxides using load_composition(wt_oxides,oxides).")
        # Check if the P2O5-related options can be applied.
        if not "P2O5" in self.mol_oxides:
            self.correct_apatite = False
            self.ignore_P = False

        # Normalize the total (proportional/relative) moles of oxide in private storage to 100 before any processing (not 100% necessary).
        mol_oxides_proportions = normalise_dict_vals(self.mol_oxides)

        # Combine everything into FeO by forcing any Fe3+ formation to occur due to oxygen contents.
        # Order of calculations is important.
        if "Fe2O3" in mol_oxides_proportions:
            mol_oxides_proportions["O"] = mol_oxides_proportions["Fe2O3"]
            mol_oxides_proportions["FeO"] = 2*mol_oxides_proportions["Fe2O3"] + mol_oxides_proportions["FeO"]
            # Remove Fe2O3 from private storage.
            mol_oxides_proportions.pop("Fe2O3")

        # Check if an apatite correction is needed.
        if self.correct_apatite and mol_oxides_proportions["P2O5"] > 0:
            # Apatite: Ca5(PO4)3 -> 5 CaO, 1.5 P2O5
            # Every 1.5 mol of P2O5 = 1 mol apatite
            # Every 1 mol of apatite = 5 mol CaO
            # Therefore remove 5 mol CaO for every 1.5 mol P2O5
            mol_oxides_proportions["CaO"] -= 5/1.5 * mol_oxides_proportions["P2O5"]

        # If P is to be ignored, remove *before* normalising.
        if self.ignore_P:
            mol_oxides_proportions.pop("P2O5")

        # Normalize the total (proportional/relative) moles of oxide in private storage to 100 after processing.
        mol_oxides_proportions = normalise_dict_vals(mol_oxides_proportions)
        return mol_oxides_proportions

    def get_element_proportions(self,oxides):
        ''' Find the proportions/relative amounts of each element from the oxide amounts.

        oxides | :list: ["<oxide>"] | List of oxides of interest, where each oxide must contain only at most one element other than oxygen, and that other element must be appear at the start of the formula. All other oxides will be skipped.

        Returns: dict {"<element>":<relative amount>}
        '''
        # Retrieve oxide moles data after normalization and application of options.
        mol_oxides_proportions = self.get_standardised_oxides()
        # Initialize dictionary into which mole data will be stored.
        element_proportions = {}
        # Initialize variable that will accumulate oxygen moles.
        oxygen = 0
        # Iterate through all oxides.
        for molecule in mol_oxides_proportions.keys():
            # Extract element proportions data from the oxide into the format [("<element>"),("<proportion>")] where "proportion" is empty if there's just 1 mole of the element per mole of oxide. There should be at most 2 elements per oxide.
            elements = re.findall("([A-Z][a-z]*)([0-9]*)",molecule)
            # If there's more than 2 elements in the oxide, raise error stating that the input is not of the correct form (despite being the correct type).
            if len(elements) > 3:
                raise ValueError("There's more than one major element in the oxide formula %s. Check the `oxides` input variable." % molecule)

            # Find the main element (which should be first in the oxide formula).
            main_element,N = elements[0]
            # Compute the total relative moles of that element in the full formula (undivided into oxides).
            element_proportions[main_element] = mol_oxides_proportions[molecule] * float(N if len(N) else 1)
            # If the oxide formula contains another element ...
            if len(elements) > 1:
                # ... find the remaining element (should be oxygen) and its relative amount.
                should_be_oxygen,N_O = elements[1]
                # Make sure that the remaining element *is* oxygen, and if not, raise an error.
                if should_be_oxygen != "O":
                    raise ValueError("%s does not appear to be an oxide formula. Check the `oxides` input variable." % molecule)
                # Accumulate the correct proportion of oxygen (relative to the full formula) into the `oxygen` variable.
                oxygen += mol_oxides_proportions[molecule] * float(N_O if len(N_O) else 1)

        # Store the amount of oxygen in to the `element_proportions` dictionary database.
        if "O" not in element_proportions:
            element_proportions["O"] = 0
        element_proportions["O"] += oxygen
        return element_proportions

    def theriak_domino_formula(self,wt_oxides=None,oxides=oxides):
        ''' Construct the string formula suitable for Theriak-Domino input.

        wt_oxides | :dict: {"<oxide>":<weight%>} | Oxide weight% data. Provide this if the existing mol oxides data is to be overwritten.
        oxides | :list: ["<oxide>"] | List of oxides of interest, where each oxide must contain only at most one element other than oxygen, and that other element must be appear at the start of the formula. All other oxides will be skipped.

        Returns: str
        '''
        # Load any {"<oxide>":<wt%>} (or suitably castable) composition if provided to this method.
        if wt_oxides is not None:
            self.load_composition(dict(wt_oxides),oxides)
        # Find the proportion of each element (in dictionary format).
        element_proportions = self.get_element_proportions(oxides)
        # Construct and return the Theriak-Domino composition string from the element proportions.
        return "".join(["%s(%.2f)" % (e[0].upper(),e[1]) for e in element_proportions.items() if e[1] > 0])


# Tests (note that there are slight differences in the Ar values used, so there might be some very small differences in output formula).
# CompositionProcessor().theriak_domino_formula({"SiO2":49.05,"TiO2":0.89,"Al2O3":17.83,"FeO":1,"Fe2O3":2.65,"MnO":0.12,"MgO":5.91,"CaO":7.25,"Na2O":3.72,"K2O":1.37,"P2O5":0.25,"H2O":3.60},oxides)
# Returns: 'SI(48.87)AL(20.94)CA(7.39)MG(8.78)FE(6.31)K(1.74)NA(7.19)TI(0.67)MN(0.10)H(23.92)O(170.47)'
# Vs spreadsheet: SI(48.87)AL(20.94)CA(7.39)MG(8.78)FE(6.31)K(1.74)NA(7.19)TI(0.67)MN(0.1)H(23.93)O(170.47)

# CompositionProcessor().theriak_domino_formula({"SiO2":49.05,"TiO2":0.89,"Al2O3":17.83,"FeO":5.19,"MnO":0.12,"MgO":5.91,"CaO":7.25,"Na2O":3.72,"K2O":1.37,"P2O5":0.25,"H2O":3.60},oxides)
# Returns: 'SI(50.37)AL(21.58)CA(7.61)MG(9.05)FE(4.46)K(1.79)NA(7.41)TI(0.69)MN(0.10)H(24.66)O(172.64)'
# Vs spreadsheet: SI(50.37)AL(21.58)CA(7.61)MG(9.05)FE(4.46)K(1.79)NA(7.41)TI(0.69)MN(0.1)H(24.66)O(172.64)

# CompositionProcessor(correct_apatite=False).theriak_domino_formula({"SiO2":49.05,"TiO2":0.89,"Al2O3":17.83,"FeO":5.19,"MnO":0.12,"MgO":5.91,"CaO":7.25,"Na2O":3.72,"K2O":1.37,"P2O5":0.25,"H2O":3.60,"P2O5":0.25},oxides)
# Returns: 'SI(50.19)AL(21.50)CA(7.95)MG(9.01)FE(4.44)K(1.79)NA(7.38)TI(0.69)MN(0.10)H(24.57)O(172.37)'
# Vs spreadsheet: SI(50.19)AL(21.5)CA(7.95)MG(9.01)FE(4.44)K(1.79)NA(7.38)TI(0.68)MN(0.1)H(24.57)O(172.37)
