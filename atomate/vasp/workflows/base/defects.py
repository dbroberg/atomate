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


def get_wf_chg_defects(structure, mpid=None, name="chg_defect_wf", user_settings=None,
                        vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<",
                        conventional=True, diel_flag=True, n_max=128,
                        vacancies={}, antisites={}, substitutions={}, interstitials={},
                        initial_charges={}, rerelax_flag=False, hybrid_flag=True,
                        run_analysis=False):
    """
    Returns a charged defect workflow

    Firework 0 : (optional) re-relax the bulk structure that is input before running rest of workflow
    Firework 1 : (optional) run hybrid calculation of bulk structure to allow for
    Firework 2 : bulk supercell calculation
    Firework 3 : (optional) dielectric calculation
    Firework 4 - len(defectcalcs): Optimize the internal structure (fixed volume)
                                   of each charge+defect combination.

    (note if no re-relaxation required, then 1-len(defectcalcs) will all run at same time...)

    Args:
        structure (Structure): input structure to have defects run on
        mpid (str): Materials Project id number to be used (for storage in metadata).
        name (str): some appropriate name for the workflow.
        user_settings (dict): User vasp settings for relaxation calculations.
            Caution when modifying this - several default settings are forced (as shown below)
        vasp_cmd (str): Command to run vasp.
        db_file (str): path to file containing the database credentials.
        conventional (bool): flag to use conventional structure (rather than primitive) for defect supercells,
            defaults to True.
        diel_flag (bool): flag to also run dielectric calculations.
            (required for charge corrections to be run) defaults to True.
        n_max (int): maximum supercell size to consider for supercells
        vacancies (dict):
            If nothing specified, all vacancies are considered.
            TODO: if more specificity is supplied then limit number of defects created (probably load vacancy pymatgen type)
        antisites (dict):
            If nothing specified, all antisites are considered.
            TODO: if more specificity is supplied then limit number of defects created (probably load Substitution pymatgen type)

        substitutions (dict):
            If nothing specified, NO substitutions defects are considered (default).
            IF substitutions desired then dict gives allowed substitutions:
                Example: {"Co":["Zn","Mn"]} means Co sites (in bulk structure) can be substituted
                by Zn or Mn.

        interstitials (dict):
            If nothing specified, NO interstitial defects are considered (default).
            IF interstitials desired then dict gives allowed interstitials:
                NOTE that two approaches to interstitial generation are available:
                    Option 1 = Manual input of interstitial sites of interest.
                    TODO: make this actually work
                        This is given by the following dictionary type:
                        Example: {<Site_object_1>: ["Zn"], <Site_object_2>: ["Zn","Mn"]}
                         makes Zn interstitial sites on pymatgen site objects 1 and 2, and Mn interstitials on
                         pymatgen site object 2
                    Option 2 = Pymatgen interstital generation of interstitial sites
                        This is given by the following dictionary type:
                        Example: {"Zn": interstitial_generation_method_1}
                         generates Zn interstitial sites using interstitial_generation_method_1 from pymatgen
                        Options for interstitial generation are:  ????

        initial_charges (dict):
            says how to specify initial charges for each defect.
            There are two approaches to charge generation available:
                Option 1 = Manual input of charges of interest.
                TODO: make this actually work
                    This is given by the following dictionary type:
                    Example: {????}
                Option 2 = Pymatgen generation of charges
                TODO: make this actually work
                    This is given by the following dictionary type:
                    Example: {"vacancies": {"Zn": charge_generation_method_1 }, ...}
                        uses charge_generation_method_1 from pymatgen on Zn vacancies...
            Default is to do a fairly restrictive charge generation method:
                for vacancies: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try
                antisites and subs: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try +
                    added to likely charge of substitutional site (closest to zero)
                interstitial: charge zero

        rerelax_flag (bool):
            Flag to re-relax the input structure for minimizing forces 
            (does volume relaxation of small primitive cell) 
            Default is False (no re-relaxation occurs)
        hybrid_flag (bool):
            Flag to run a single small hybrid bulk structure for band edge shifting.
            Default is True (Hybrid will be calculated)
        run_analysis (bool):
            Flag to run an analysis firework at end of workflow to provide overall analysis.
            Default is False (Suggested to run own analysis to do things like add additional charge states
                and smart parsing of defect related values)

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
        if hybrid_flag: #only run HSEBSFW hybrid workflow here if re-relaxed since it requires a copy-over optimized structure
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

    t.append(DefectSetupFiretask(structure=prim_structure, cellmax=n_max, conventional=conventional,
                                 vasp_cmd=vasp_cmd, db_file=db_file,
                                 vacancies=vacancies, antisites=antisites, substitutions=substitutions,
                                 interstitials=interstitials, initial_charges=initial_charges))

    setup_fw = Firework(t,parents = parents, name="{} Defect Supercell Setup".format(structure.composition.reduced_formula))
    fws.append(setup_fw)

    if run_analysis:
        analysis_parents = [setup_fw]
        if diel_flag:
            analysis_parents.append(diel_fw)
        analysis_fw = DefectAnalysisFW(prim_structure, mpid=mpid, parents=analysis_parents)
        fws.append(analysis_fw)

    wfname = "{}:{}".format(structure.composition.reduced_formula, name)
    final_wf = Workflow(fws, name=wfname)

    return final_wf
