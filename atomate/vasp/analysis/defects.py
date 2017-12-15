#!/usr/bin/env python

"""
This is a practical DefectPhaseDiagram object, analogous to a PhaseDiagram object from pymatgen...

right now it takes fw_metadata
TODO: allow for class to take a list of ComputedEntries, rather than a fw_spec
TODO: fix as_dict() to actually work...
"""

from monty.json import MSONable

import copy
import numpy as np

from pycdt.core.defects_analyzer import DefectsAnalyzer


class DefectPhaseDiagram(MSONable):
    """
    This is similar to a PhaseDiagram object in pymatgen, but has ability to do quick analysis of defect formation energies
    when fed ComputedDefectEntries ( or, for now, fw_metadata...

    uses many of the capabilities from PyCDT's DefectsAnalyzer class...

    SHould be able to get:
        a) stability of charge states for a given defect,
        b) formation ens
        c) [later on] fermi energies...

    Args:
        fw_metadata (list): fw_metadata in update_spec after DefectAnalysisFireTask is run
    """
    def __init__(self, fw_metadata):
        tmpkey = ''
        tmpind = 0
        while not tmpkey:
            if fw_metadata.keys()[tmpind] in ['entry_bulk','_files_prev']:
                tmpind+=1
            else:
                tmpkey = fw_metadata.keys()[tmpind]
        basedat = fw_metadata[tmpkey]['corr_details']['base']
        #load up the defects analyzer objects
        self.fullset_da = {}
        self.da = None #only need one da for looking at stability...
        for mnom, muset in basedat['chem_lims'].items():
            self.fullset_da[mnom] = DefectsAnalyzer( fw_metadata['entry_bulk'], basedat['vbm'], muset,
                                                     basedat['gga_bandgap'])
            for dnom, def_entry in fw_metadata.items():
                if dnom not in ['entry_bulk', '_files_prev']:
                    print dnom,  def_entry['cd'].charge_correction, def_entry['cd'].other_correction
                    self.fullset_da[mnom].add_computed_defect(copy.copy(def_entry['cd']) )
            if not self.da:
                self.da = copy.copy(self.fullset_da[mnom])

        #now get stable defect sets
        self.stable_charges = {} #keys are defect names, items are list of charge states that are stable
        self.finished_charges = {} #keys are defect names, items are list of charge states that are included in the phase diagram
        self.transition_levels = {} #keys are defect names, items are list of [fermi level for transition, previous q, next q] sets
        xlim = (-0.1, self.da._band_gap+.1)
        nb_steps = 10000
        x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/nb_steps)
        for t in self.da._get_all_defect_types():
            print('hey dan:',t, 'tote defs=',len(self.da._defects))
            trans_level = []
            chg_type = []
            prev_min_q, cur_min_q = None, None
            for x_step in x:
                miny = 10000
                for i, dfct in enumerate(self.da._defects):
                    if dfct.name == t:
                        val = self.da._formation_energies[i] + \
                                dfct.charge*x_step
                        if val < miny:
                            miny = val
                            cur_min_q = dfct.charge
                if prev_min_q is not None:
                    if cur_min_q != prev_min_q:
                        trans_level.append((x_step, prev_min_q, cur_min_q))
                    if cur_min_q not in chg_type:
                        chg_type.append(cur_min_q)
                prev_min_q = cur_min_q

            self.stable_charges[dfct.name] = chg_type[:]
            self.finished_charges[dfct.name] = [e.charge for e in self.da._defects if e.name == t]
            self.transition_levels[dfct.name] = trans_level[:]


    def as_dict(self):
        d = {'da': self.da.as_dict(),
             "@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        return d

    # def from_dict(cls, d):
    #     da_dict = d['da']
    #     analyzer = DefectsAnalyzer().from_dict(da_dict)
    #     return analyzer

    @property
    def all_defect_types(self):
        """
        List types of defects existing in the DefectPhaseDiagram
        """
        return self.da._get_all_defect_types()

    @property
    def all_defect_entries(self):
        """
        List all defect entries existing in the DefectPhaseDiagram
        """
        return [e.full_name for e in self.da.defects]

    @property
    def all_stable_entries(self):
        """
        List all stable entries (defect+charge) in the DefectPhaseDiagram
        """
        stable_entries = []
        for t, stabset in self.stable_charges.items():
            for c in stabset:
                stable_entries.append(t + "_" + str(c))
        return stable_entries

    @property
    def all_unstable_entries(self):
        """
        List all unstable entries (defect+charge) in the DefectPhaseDiagram
        """
        return [e for e in self.all_defect_entries if e not in self.all_stable_entries]

    # @property
    # def formation_energies(self, efermi=0.):
    #     """
    #     Give dictionaries of all formation energies at specified efermi in the DefectPhaseDiagram
    #     Default efermi = 0 = VBM energy TODO: need to specify growth condition?
    #     """
    #     return self.da.get_formation_energies(ef=efermi)


    def suggest_charges(self):
        """
        Based on entries, suggested follow up charge states to run
        (to make sure possibilities for charge states have been exhausted)
        """
        fullrecommendset = {}
        for t in self.da._get_all_defect_types():
            print('Consider recommendations for ',t)
            reccomendset = []
            allchgs = self.finished_charges[t]
            stablechgs = self.stable_charges[t]
            for followup_chg in range(min(stablechgs)-1,  max(stablechgs)+2):
                if followup_chg in allchgs:
                    continue
                else:
                    recflag = True
                    for tl in self.transition_levels[t]: #tl has list of [fermilev for trans, prev q, next q]
                        if tl[0] <= 0.1: #check if t.l. is within 0.1 eV of the VBM
                            morepos = int(tl[1])
                            # tmpstrchg = tl[2].split('/') #str(prev_min_q)+'/'+str(cur_min_q)
                            # morepos = int(tmpstrchg[0]) #more positive charge state
                            if  (followup_chg > morepos):
                                print('Wont recommend:',followup_chg,'Because of this trans lev:', tl[1],'/',
                                      tl[2],' at ',tl[0])
                                recflag = False
                        if tl[0] >= (self.da._band_gap - 0.1): #check if t.l. is within 0.1 eV of CBM
                            moreneg = int(tl[2])
                            # tmpstrchg = tl[2].split('/') #str(prev_min_q)+'/'+str(cur_min_q)
                            # moreneg = int(tmpstrchg[1]) #more negative charge state
                            if  (followup_chg < moreneg):
                                print('Wont recommend:',followup_chg,'Because of this trans lev:', tl[1],'/',
                                      tl[2],' at ',tl[0], '(gap = ', self.da._band_gap,'eV)')
                                recflag = False
                    if recflag:
                        reccomendset.append(followup_chg)
            if len(reccomendset):
                print('charges recommending:',reccomendset)
            fullrecommendset[t] = reccomendset[:]

        return fullrecommendset

# class DefectEntry(ComputedEntry):
#     """
#     A simplified defect entry possible...
#     """