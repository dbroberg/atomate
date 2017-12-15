# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module provides ability to do Defect analysis after defect FWs finished

Makes use of atomate.vasp.firetasks.defects import DefectAnalysisFireTask
Note alot of this structure is taken from pycdt.utils.parse_calculations and pycdt.......

currently does parsing through a fireworks spec
TODO: make this FW more transparent about DB entry/looking at DB to make sure not to redo previous calculations...
"""

from fireworks import Firework

from atomate.utils.utils import get_logger
from atomate.vasp.firetasks.defects import DefectAnalysisFireTask

logger = get_logger(__name__)


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

