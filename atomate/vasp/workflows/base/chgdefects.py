# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for charged defects in non-metals:
- bulk calculation
- dielectric calculation (with option to omit)
- many different defects in several charge states (variety + charges used can be varied signficantly)

"""

from fireworks import Workflow, Firework

from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, HSEBSFW, DFPTFW
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs

from atomate.vasp.firetasks.defects import DefectSetupFiretask
from atomate.vasp.fireworks.defects import DefectAnalysisFW

logger = get_logger(__name__)


def get_wf_chgdefects(structure, mpid=None, name="chgdefectwf", user_settings=None,
                        vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", tag="", 
                        conventional=True, diel_flag=True, n_max=128, 
                        oxi_range_dict={}, oxi_state_dict={}, antisites=True,
                        substitutions={}, include_interstitials=False,
                        interstitial_elements=[], chg_type='semiconductor',
                        rerelax_flag=False,
                        hybrid_flag=True, metadata={}):
    """≠≠
    Returns a structure deformation workflow.

    Firework 0 : (optional) re-relax the bulk structure that is input before running rest of 
    Firework 1 : bulk supercell calculation
    Firework 2 : dielectric calculation
    Firework 3 - len(defectcalcs): Optimize the internal structure (fixed volume) 
                                   of each charge+defect combination.


    Args:
        structure (Structure): input structure to have defects run on
        mpid (str): Materials Project id number to be used for parsing of band structures.
            defaults to using vbm/cbm of bulk static calcualtion...
        name (str): some appropriate name for the transmuter fireworks.
        user_settings (dict): User vasp settings for relaxation calculations.
            Caution when modifying this - several default settings are forced (as shown below)
        vasp_cmd (str): Command to run vasp.
        db_file (str): path to file containing the database credentials.
        tag (str): some unique string that will be appended to the names of the fireworks so that
            the data from those tagged fireworks can be queried later during the analysis.
        metadata (dict): meta data
        conventional (bool): flag to use conventional structure (rather than primitive) for supercells,
            defaults to True.
        diel_flag (bool): flag to also run dielectric calculations.
            (required for charge corrections to be run) defaults to True.
        n_max (int): maximum supercell size to consider for supercells
        oxi_range_dict (dict):
            The minimal and maximum oxidation state of each element as a
            dict. For instance {"O":(-2,0)}. If not given, the oxi-states
            of pymatgen are considered.
        oxi_state_dict (dict):
            The oxidation state of the elements in the compound e.g.
            {"Fe":2,"O":-2}. If not given, the oxidation state of each
            site is computed with bond valence sum. WARNING: Bond-valence
            method can fail for mixed-valence compounds.
        antisites (bool):
            If False, don't generate antisites.
        substitutions (dict):
            The allowed substitutions of elements as a dict. If not given,
            intrinsic defects are computed. If given, intrinsic (e.g.,
            anti-sites) and extrinsic are considered explicitly specified.
            Example: {"Co":["Zn","Mn"]} means Co sites can be substituted
            by Mn or Zn.
        include_interstitials (bool):
            If true, do generate interstitial defect configurations
            defaults to False.
        interstitial_elements ([str]):
            List of strings containing symbols of the elements that are
            to be considered for interstitial sites.  The default is an
            empty list, which triggers self-interstitial generation,
            given that include_interstitials is True.
        chg_type (string):
            This is a predefined 'charging' scheme for the defects. 
            Some pre-defined options are 'semiconductor', 'insulator', 
            'manual', and 'ionic'. Detailed code for this can be found 
            in pycdt.core.defectsmaker. Default is 'semiconductor'.
        rerelax_flag (bool):
            Flag to re-relax the input structure for minimizing forces 
            (does volume relaxation of small primitive cell) 
            Default is False (no re-relaxation occurs)
        hybrid_flag (bool):
            Flag to run a single small hybrid bulk structure for band edge shifting.
            Default is True (Hybrid will be calculated)
        longanalyze_flag (bool):
            Flag to run a post-analysis of stable charge states to see if more charge states 
            should be considered (possible stability within the bandgap)
        metadata (dict): meta data; note that key "PyCDTdata" will be overwritten.

    Returns:
        Workflow
    """
    fws, parents = [], []

    #force optimization and dielectric calculations with primitive structure for expediency
    prim_structure = SpacegroupAnalyzer(structure).find_primitive()

    if rerelax_flag:
        vis = MPRelaxSet(prim_structure, user_incar_settings={"EDIFF": .00001, "EDIFFG": -0.001, "ISMEAR":0,
                                                         "SIGMA":0.05, "NSW": 100, "ISIF": 3, "LCHARG":False,
                                                         "ISPIN":2,  "ISYM":2, "LAECHG":False})
        rerelax_fw = OptimizeFW(prim_structure, vasp_input_set=vis,
                                 vasp_cmd=vasp_cmd, db_file=db_file,
                                 job_type="double_relaxation_run",
                                 auto_npar=">>auto_npar<<",
                                 half_kpts_first_relax=True, parents=None)
        fws.append(rerelax_fw)
        parents = [rerelax_fw]
        if hybrid_flag: #only run HSEBSFW hybrid workflow if re-relaxed since it requires a copy-over
            hse_fw = HSEBSFW(prim_structure, parents, vasp_cmd=vasp_cmd, db_file=db_file)
            fws.append( hse_fw)

    elif hybrid_flag: #if not re-relaxing structure but want hybrid then need to run a static primitive struct calc initial
        stat_gap_fw = StaticFW(prim_structure, name="{} gap gga initialize".format(structure.composition.reduced_formula),
                                vasp_cmd=vasp_cmd, db_file=db_file)
        fws.append( stat_gap_fw)
        hse_fw = HSEBSFW(prim_structure, stat_gap_fw, vasp_cmd=vasp_cmd, db_file=db_file)
        fws.append( hse_fw)

    if diel_flag:
        copy_out = True if parents else False
        diel_fw = DFPTFW(prim_structure, name='ionic dielectric', vasp_cmd=vasp_cmd, copy_vasp_outputs=copy_out,
                         db_file=db_file, parents=parents)
        fws.append( diel_fw)

    t = []
    if parents:
        t.append(CopyVaspOutputs(calc_loc= True ))

    t.append(DefectSetupFiretask(structure=prim_structure, max_min_oxi=oxi_range_dict,
                                    oxi_states=oxi_state_dict, antisites_flag=antisites,
                                    substitutions=substitutions,
                                    include_interstitials=include_interstitials,
                                    interstitial_elements=interstitial_elements,
                                    cellmax=n_max, struct_type=chg_type, conventional=conventional))

    setup_fw = Firework(t,parents = parents, name="{} Defect Supercell Setup".format(structure.composition.reduced_formula))
    fws.append(setup_fw)

    analysis_parents = [setup_fw]
    if diel_flag:
        analysis_parents.append(diel_fw)
    analysis_fw = DefectAnalysisFW(prim_structure, mpid=mpid, parents=analysis_parents)
    fws.append(analysis_fw)

    wfname = "{}:{}".format(structure.composition.reduced_formula, name)
    final_wf = Workflow(fws, name=wfname)

    return final_wf
