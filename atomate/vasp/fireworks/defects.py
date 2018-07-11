# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module provides ability to do Defect analysis after defect WF has finished

Makes use of atomate.vasp.firetasks.defects import DefectAnalysisFireTask
Note alot of this structure is taken from pycdt.utils.parse_calculations and pycdt.......
"""

from fireworks import Firework

from atomate.utils.utils import get_logger
from atomate.vasp.firetasks.defects import DefectAnalysisFireTask
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, PassCalcLocs
from atomate.vasp.firetasks.write_inputs import WriteTransmutedStructureIOSet, ModifyIncar
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.parse_outputs import VaspToDb

from pymatgen.io.vasp.sets import MPRelaxSet, MPHSERelaxSet

logger = get_logger(__name__)


class HSETransmuterFW(Firework):

    def __init__(self, structure, transformations, transformation_params=None,
                 vasp_input_set=None, prev_calc_dir=None,
                 name="HSE structure transmuter", vasp_cmd="vasp",
                 copy_vasp_outputs=True, db_file=None, job_type="normal",
                 parents=None, override_default_vasp_params=None, **kwargs):
        """
        Apply the transformations to the input structure, write the input set corresponding
        to the transformed structure, and run an HSE06 vasp RELAXATION (time consuming) on them.

        Note that if a transformation yields
        many structures from one, only the last structure in the list is used.

        Args:
            structure (Structure): Input structure.
            transformations (list): list of names of transformation classes as defined in
                the modules in pymatgen.transformations.
                eg:  transformations=['DeformStructureTransformation', 'SupercellTransformation']
            transformation_params (list): list of dicts where each dict specify the input
                parameters to instantiate the transformation class in the transformations list.
            vasp_input_set (VaspInputSet): VASP input set, used to write the input set for the
                transmuted structure.
            name (string): Name for the Firework.
            vasp_cmd (string): Command to run vasp.
            copy_vasp_outputs (bool): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (string): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            override_default_vasp_params (dict): additional user input settings for vasp_input_set.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        fw_name = "{}-{}".format(structure.composition.reduced_formula, name)
        override_default_vasp_params = override_default_vasp_params or {}
        t = []

        vasp_input_set = vasp_input_set or MPRelaxSet(structure,
                                                       force_gamma=True,
                                                       **override_default_vasp_params)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(
                WriteTransmutedStructureIOSet(transformations=transformations,
                                              transformation_params=transformation_params,
                                              vasp_input_set=vasp_input_set,
                                              override_default_vasp_params=override_default_vasp_params,
                                              prev_calc_dir="."))
        elif copy_vasp_outputs:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
            t.append(
                WriteTransmutedStructureIOSet(structure=structure,
                                              transformations=transformations,
                                              transformation_params=transformation_params,
                                              vasp_input_set=vasp_input_set,
                                              override_default_vasp_params=override_default_vasp_params,
                                              prev_calc_dir="."))
        elif structure:
            t.append(WriteTransmutedStructureIOSet(structure=structure,
                                                   transformations=transformations,
                                                   transformation_params=transformation_params,
                                                   vasp_input_set=vasp_input_set,
                                                   override_default_vasp_params=override_default_vasp_params))
        else:
            raise ValueError("Must specify structure or previous calculation")

        #run GGA relaxation first
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, job_type=job_type))

        #then follow with HSE relaxation
        #TODO: test to make sure this works and modifications are appropriately made...
        hse_vasp_input_set = MPHSERelaxSet(structure,
                                           force_gamma=True, **override_default_vasp_params)
        t.append(ModifyIncar(incar_update=hse_vasp_input_set.incar))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd))


        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file,
                          additional_fields={
                              "task_label": name,
                              "transmuter": {"transformations": transformations,
                                             "transformation_params": transformation_params}
                          }))

        super(HSETransmuterFW, self).__init__(t, parents=parents,
                                           name=fw_name, **kwargs)


class DefectAnalysisFW(Firework):
    def __init__(self, structure, name="Defect Analyzer",
                 mpid=None, parents=None,
                 **kwargs):
        """
        Standard Defect Analysis Firework.
        Args:
            structure (Structure): Bulk Structure. The structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []

        t.append( DefectAnalysisFireTask(mpid=mpid))

        super(DefectAnalysisFW, self).__init__(t, parents=parents, name="{} {}".format(
            structure.composition.reduced_formula, name), **kwargs)

