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
                        vacancies=[], substitutions={}, interstitials={},
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

        vacancies (list):
            If list is totally empty, all vacancies are considered (default).
            If only specific vacancies are desired then add desired Element symbol to the list
                ex. ['Ga'] in GaAs structure will only produce Galium vacancies

            if NO vacancies are desired, then just add an empty list to the list
                ex. [ [] ]  yields no vacancies

        substitutions (dict):
            If dict is totally empty, all intrinsic antisites are considered (default).
            If only specific antisites/substituions are desired then add vacant site type as key, with list of
                sub site symbol as value
                    ex 1. {'Ga': ['As'] } in GaAs structure will only produce Arsenic_on_Gallium antisites
                    ex 2. {'Ga': ['Sb'] } in GaAs structure will only produce Antimonide_on_Gallium substitutions

            if NO antisites or substitutions are desired, then just add an empty dict
                ex. {None:{}}  yields no antisites or subs


        interstitials (list):
            If list is totally empty, NO interstitial defects are considered (default).
            Option 1 for generation: If one wants to use Pymatgen to predict interstitial
                    then list of pairs of [symbol, generation method (str)] can be provided
                        ex. ['Ga', 'Voronoi'] in GaAs structure will produce Galium interstitials from the
                            Voronoi site finding algorithm
                        NOTE: only options for interstitial generation are "Voronoi" and "Nils"
            Option 2 for generation: If user wants to add their own interstitial sites for consideration
                    the list of pairs of [symbol, Interstitial object] can be provided, where the
                    Interstitial pymatgen.analysis.defects.core object is used to describe the defect of interest


        initial_charges (dict):
            says how to specify initial charges for each defect.
            An empty dict (DEFAULT) is to do a fairly restrictive charge generation method:
                for vacancies: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try
                antisites and subs: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try +
                    added to likely charge of substitutional site (closest to zero)
                interstitial: charge zero
            For non empty dict, charges are specified as:
                initial_charges = {'vacancies': {'Ga': [-3,2,1,0]},
                                   'substitutions': {'Ga': {'As': [0]} },
                                   'interstitials': {}}
                in the GaAs structure this makes vacancy charges in states -3,-2,-1,0; Ga_As antisites in the q=0 state,
                and all other defects will have charges generated in the restrictive automated format stated for DEFAULT

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
            hse_fw = HSEBSFW(structure=prim_structure, parents=parents, name="hse", vasp_cmd=vasp_cmd, db_file=db_file)
            fws.append( hse_fw)

    elif hybrid_flag: #if not re-relaxing structure but want hybrid then need to run a static primitive struct calc initial
        stat_gap_fw = StaticFW(structure=prim_structure, name="{} gap gga initialize".format(structure.composition.reduced_formula),
                                vasp_cmd=vasp_cmd, db_file=db_file)
        fws.append( stat_gap_fw)
        hse_fw = HSEBSFW(structure=prim_structure, parents=stat_gap_fw, name="hse", vasp_cmd=vasp_cmd, db_file=db_file)
        fws.append( hse_fw)

    if diel_flag:
        copy_out = True if parents else False
        diel_fw = DFPTFW(structure=prim_structure, name='ionic dielectric', vasp_cmd=vasp_cmd, copy_vasp_outputs=copy_out,
                         db_file=db_file, parents=parents)
        fws.append( diel_fw)

    t = []
    if parents:
        t.append(CopyVaspOutputs(calc_loc= True ))

    t.append(DefectSetupFiretask(structure=prim_structure, cellmax=n_max, conventional=conventional,
                                 vasp_cmd=vasp_cmd, db_file=db_file,
                                 vacancies=vacancies, substitutions=substitutions,
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
