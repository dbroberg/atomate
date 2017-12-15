# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module provides ability to setup and parse defects on fly...
Note alot of this structure is taken from pycdt.utils.parse_calculations and pycdt.......

Requirements:
 - bulk calculation is finished (and vasprun.xml + Locpot)
 - a dielectric constant/tensor is provided
 - defect+chg calculation is finished 

 Soft requirements:
 	- Bulk and defect OUTCAR files (if charge correction by Kumagai et al. is desired)
 	- Hybrid bulk bandstructure / simple bulk structure calculation (if bandshifting is desired)
"""

import os
import numpy as np

from monty.json import jsanitize

from pymatgen.io.vasp import Vasprun, Locpot, Poscar
from pymatgen.matproj.rest import MPRester
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry

from fireworks import FiretaskBase, FWAction, explicit_serialize

from pycdt.core.chemical_potentials import MPChemPotAnalyzer
from pycdt.core.defectsmaker import ChargedDefectsStructures
from pycdt.corrections.freysoldt_correction import FreysoldtCorrection
from pycdt.corrections.kumagai_correction import KumagaiBulkInit, KumagaiCorrection
from pycdt.corrections.extra_corrections import ExtraCorrections    #this is just on PyCDT development branch for now
from pycdt.core.defects_analyzer import ComputedDefect

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import TransmuterFW

from monty.serialization import dumpfn
from monty.json import MontyEncoder

logger = get_logger(__name__)

@explicit_serialize
class DefectSetupFiretask(FiretaskBase):
    """
    Run defect supercell setup

    Args:
        structure (Structure): input structure to have defects run on
        cellmax (int): maximum supercell size to consider for supercells
        max_min_oxi (dict):
            The minimal and maximum oxidation state of each element as a
            dict. For instance {"O":(-2,0)}. If not given, the oxi-states
            of pymatgen are considered.
        oxi_states (dict):
            The oxidation state of the elements in the compound e.g.
            {"Fe":2,"O":-2}. If not given, the oxidation state of each
            site is computed with bond valence sum. WARNING: Bond-valence
            method can fail for mixed-valence compounds.
        antisites_flag (bool):
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
        struct_type (string):
            This is a predefined 'charging' scheme for the defects. 
            Some pre-defined options are 'semiconductor', 'insulator', 
            'manual', and 'ionic'. Detailed code for this can be found 
            in pycdt.core.defectsmaker. Default is 'semiconductor'.
        conventional (bool):
            flag to use conventional structure (rather than primitive) for supercells,
            defaults to True.

        vasp_cmd (string): 
            the vasp cmd
        db_file (string):
            the db file
    """
    def run_task(self, fw_spec):
        if os.path.exists("POSCAR"):
            structure =  Poscar.from_file("POSCAR").structure
        else:
            structure = self.get("structure")

        if self.get("conventional", True):
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        #note that ChargedDefectsStructures runs the defectTransformation routines currently in pymatgen.transformations
        def_structs = ChargedDefectsStructures( structure,
                                                max_min_oxi=self.get("max_min_oxi", dict()),
                                                oxi_states=self.get("oxi_states", dict()),
                                                antisites_flag=self.get("antisites_flag", True),
                                                substitutions=self.get("substitutions", dict()),
                                                include_interstitials=self.get("include_interstitials", False),
                                                interstitial_elements=self.get("interstitial_elements", list()),
                                                cellmax=self.get("cellmax", 128),
                                                struct_type=self.get("struct_type", "semiconductor"))


        fws, parents = [], []

        #First Firework is for bulk supercell
        vis = MPStaticSet(def_structs.defects['bulk']['supercell']['structure'],
            user_incar_settings = {"EDIFF":.0001, "EDIFFG": 0.001, "ISMEAR":0, "SIGMA":0.05, "NSW": 0, "ISIF": 2,
                        "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "LAECHG":False} )

        bulk_tag = "{}:bulk_supercell".format(structure.composition.reduced_formula)

        scale_super = def_structs.defects['bulk']['supercell']['size'] #this is just a len 3 array, for cubic supercells..
        #TODO: add ability to do transformation for more abstract supercell shapes...
        supercell_size = scale_super * np.identity(3)
        stat_fw = TransmuterFW(name = bulk_tag, structure=structure,
                               transformations=['SupercellTransformation'],
                               transformation_params=[{"scaling_matrix": supercell_size}],
                               vasp_input_set=vis, copy_vasp_outputs=False, #structure already copied over...
                               vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                               db_file=self.get("db_file", ">>db_file<<"))

        fws.append(stat_fw)

        #iterate through defects+chgs and set up fireworks for them
        stdrd_defect_incar_settings = {"EDIFF":.0001, "EDIFFG":0.001, "IBRION":2, "ISMEAR":0, "SIGMA":0.05, # "NELM":50,
                                       "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "NSW": 100, "ISIF": 2, "LAECHG":False }
        for def_type, deftypelist in def_structs.defects.items():
            if def_type != 'bulk':
                for defcalc in deftypelist:
                    dstruct = defcalc['supercell']['structure']
                    for charge in defcalc['charges']:
                        defect_input_set = MPStaticSet(dstruct,user_incar_settings=stdrd_defect_incar_settings.copy())
                        defect_input_set.user_incar_settings["NELECT"] = defect_input_set.nelect - charge

                        #first scale in the same way as bulksupercell
                        deftrans = ['SupercellTransformation']
                        deftrans_params = [{"scaling_matrix": supercell_size}]

                        #then create defect
                        defsite = defcalc['bulk_supercell_site']
                        #small bit of code to determine potential defect index...
                        ref_bulksupercel = def_structs.defects['bulk']['supercell']['structure']
                        poss_deflist = ref_bulksupercel.get_sites_in_sphere(defsite.coords, 6, include_index=True)
                        poss_defindex = poss_deflist[0][2] #index for defect (assuming vacancy or substitution...)

                        if def_type == 'vacancies':
                            deftrans.append( 'RemoveSitesTransformation')
                            deftrans_params.append( {'indices_to_remove': [poss_defindex]})
                        elif def_type == 'substitutions': #includes antisites..
                            #TODO: needs to be tested for both antisites and subs...
                            deftrans.append( 'ReplaceSiteSpeciesTransformation')
                            deftrans_params.append( {'indices_species_map': {poss_defindex: defsite.specie.symbol}})
                        elif def_type == 'interstitials':
                            #TODO: needs to be tested ...
                            deftrans.append( 'InsertSitesTransformation')
                            deftrans_params.append( {'species': [defsite.specie.symbol],
                                                    'coords': [defsite.frac_coords],
                                                    'coords_are_cartesian': False} )
                        else:
                            print('ERROR RAISED, def_type not recognized: ',def_type)
                            #TODO: make this a legitimate error raiser

                        #TODO here could also insert Transmuter for perturbation function / breaking local symmetry around defect

                        #TODO: would be good to store site_multiplciity and defect bulk_supercell_site object for parsing later on...
                        #           Maybe there could be a DefectEntry passed to database now which is modified during the Analyzer stage?
                        #       defcalc['site_multiplicity'],   defsite
                        def_tag = "{}:{}_{}".format(structure.composition.reduced_formula, defcalc['name'], charge)
                        fw = TransmuterFW(name = def_tag, structure=structure,
                                               transformations=deftrans,
                                               transformation_params=deftrans_params,
                                               vasp_input_set=defect_input_set,
                                               vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                                               copy_vasp_outputs=False, #structure already copied over...
                                               db_file=self.get("db_file", ">>db_file<<"))
                        fws.append(fw)

        # metadata = jsanitize(def_structs.defects, strict=True)
        #TODO: could push metatdata of DefectEntry here...
        return FWAction(detours=fws ) #, update_spec=metadata)


@explicit_serialize
class DefectAnalysisFireTask(FiretaskBase):
    """
    Use the Single Defect Analysis class (below) to loop through local folders,
        and see if additional ANalysis (base, charge, or extra) needs to be done
    """
    def run_task(self, fw_spec):
        """
        ...only meant for single job in database at a time right now...
        TODO: more computationally efficient by saving locpot data etc..
        """

        pat = os.getcwd()
        basepat = os.path.split(pat)[0]
        launchlistset = os.listdir(basepat)

        #assemble defect paths and job types from fw_spec (TODO: this has issues when fw_spec has multiple entries for same defect +chg?)
        globeset = {'defects':[], 'bulk': None, 'diel': None}
        for lset in fw_spec['calc_locs']:
            if "bulk_supercell" in lset['name']:
                globeset['bulk'] = lset['path']
            elif 'dielectric' in lset['name']:
                globeset['diel'] = lset['path']
            elif ('vac_' in lset['name']) or ('as_' in lset['name']) or \
                ('inter_' in lset['name']) or ('sub_' in lset['name']):
                chg = int(lset['name'].split('_')[-1])
                nom = '' #reduce name to something without charge... this is currently hacky way to do this
                nomsplitout = lset['name'].split(':')[-1].split('_')
                for tmpind, tmpnompart in enumerate(nomsplitout):
                    if tmpind != (len(nomsplitout)-1):
                        nom += tmpnompart+'_'
                def_name = nom[:-1]
                chg = int(chg)
                print ('heydan ',def_name, chg)
                # globeset['defects'].append( [lset['name'], chg, lset['path']]) #has firstentry with name_chg...
                globeset['defects'].append( [def_name, chg, lset['path']])

        #load dielectric constant
        #TODO:  should have ability to allow for manual insertion of diel value OR previously determined diel in database
        print ('\tloading diel...')
        dvr = Vasprun(os.path.join(globeset['diel'], 'vasprun.xml.gz'))
        eps_ion = dvr.epsilon_ionic
        eps_stat = dvr.epsilon_static
        eps = []
        for i in range(len(eps_ion)):
            eps.append([e[0]+e[1] for e in zip(eps_ion[i],eps_stat[i])])
        globeset['diel'] = eps
        print ('\tDielectric is ',eps)

        #run defect calculations
        print('path structure assembled. now running chg corrs etc.')
        metadata = {}
        for dset in globeset['defects']:
            dnom = dset[0] #defect name
            chg = dset[1] #charge
            dloc = dset[2] #defect location
            print('parsing',dnom,chg,' at ',dloc)
            #TODO: should have ability to skip correction step if it has already been calculated...
            sda = SingleDefectAnalysis(globeset['bulk'], dloc, globeset['diel'],
                                       dvr.final_structure.sites[0], chg) #second to last entry is defect position, right now it does nothing... so can keep at origin
            corr_details = sda.compile_all(mpid=self.get("mpid", None), print_to_file=False)

            freycorr = corr_details['freysoldt']['total_corr']
            bandfill = corr_details['bandfilling']['mbshift'] #DANNY -> this may include level shifting if not careful...

            def_entry = sda.defvr.get_computed_entry()
            cd_entry = ComputedDefect(def_entry, dvr.final_structure.sites[0], #note fake defect position...  not using true defect position yet...
                           multiplicity=None, supercell_size=[1, 1, 1],   #note fake supercell size and multiplicity... not using these values yet
                           charge=chg, charge_correction=freycorr, other_correction=bandfill, name=dnom)
            metadata[dnom+'_'+str(chg)] = {'cd': cd_entry, 'corr_details':corr_details}
            if 'entry_bulk' not in metadata.keys():
                bulk_entry = ComputedStructureEntry(sda.blkvr.final_structure, sda.blkvr.final_energy,
                                                    data={'locpot_path': os.path.join(globeset['bulk'], 'LOCPOT.gz'),
                                                          'supercell_size': [1, 1, 1]}) #note fake supercell size
                metadata['entry_bulk'] = bulk_entry

        return FWAction( update_spec=metadata)


class SingleDefectAnalysis(object):
    """
    A little unclear whether this will be a Drone or a Firetasker...
    just providing a skeleton here
    TO CONSIDER:
            > is data to be stored in database + in local file structure?


    Contents requried for inputs
        - path to bulk folder OR bulk Pymatgen objects: (vasprun.xml, Locpot
                        + path to OUTCAR file if Kumagai correction desired)
        - a dielectric constant/tensor
        - path to defect folder OR defect Pymatgen objects: (vasprun.xml, Locpot
                        + path to OUTCAR file if Kumagai correction desired)
        - information about defect: (position of defect in structure, charge of defect)

    Quantities that can be produced from this class:
        1) (base_postprocess) = minimum requirements for calculating a Defect Formation Energy
                > parsing of bandstructures for electron chemical potential
                > parsing of phase diagram for atomic chemical potentials
        2) (freysoldt) Charge Correction from Freysoldt et al. (Includes electrostatic term and potential alignment term)
        3) (kumagai) Extension of Freysoldt correction suggested by Kumagai et al. (Includes electrostatic term and potential alignment term)
                > interesting because it takes into account anisotropic dielectric response
        4) (bandfilling) Moss-Burstein filling correction
        5) (bandshifting) [developmental stages] "Shallow level" band edge shifting corrections based on Hybrid Band Structure

    """
    def __init__(self, path_to_bulk, path_to_defect, dielconst, defpos, defchg):
        self.blkpath = path_to_bulk           					                    #REQD by:  kumagai(just for OUTCAR path)
        #quicker way could be to load the follow directly to this class
        # self.blkvr = Vasprun(os.path.join(path_to_bulk, 'vasprun.xml'))			#REQD by:  base_postprocess(if no mpid provided), kumagai(just for structure)
        self.blkvr = Vasprun(os.path.join(path_to_bulk, 'vasprun.xml.gz'))			#REQD by:  base_postprocess(if no mpid provided), kumagai(just for structure)
        # self.blklp = Locpot.from_file(os.path.join(path_to_bulk, 'LOCPOT'))		#REQD by:  freysoldt
        self.blklp = Locpot.from_file(os.path.join(path_to_bulk, 'LOCPOT.gz'))		#REQD by:  freysoldt


        self.defpath = path_to_defect											#REQD by:  kumagai(just for OUTCAR path)
        #quicker way could be to load the follow directly to this class
        # self.defvr = Vasprun(os.path.join(path_to_defect, 'vasprun.xml'))		#REQD by:  freysoldt, kumagai(just for structure)
        self.defvr = Vasprun(os.path.join(path_to_defect, 'vasprun.xml.gz'))		#REQD by:  freysoldt, kumagai(just for structure)
        # self.deflp = Locpot.from_file(os.path.join(path_to_defect, 'LOCPOT'))	#REQD by:  freysoldt
        self.deflp = Locpot.from_file(os.path.join(path_to_defect, 'LOCPOT.gz'))	#REQD by:  freysoldt
        #NOTE that time can also be saved from not loading locpots for charge = 0

        self.diel = dielconst #can be either a number or a 3x3list array		#REQD by:  freysoldt, kumagai

        self.defpos = defpos													#REQD by:  none right now... should be used by freysoldt and kumagai in future
        self.defchg = defchg													#REQD by:  freysoldt, kumagai

    def compile_all(self, mpid=None, print_to_file=True):
        """
        Right now this naively accumulates a global dictionary of:
            1) base_postprocess
            2) FreysoldtCorrection
            3) KumagaiCorrection
            4) bandfilling
        TODO: implement defect level shifting method
        """
        full_compile = {}

        full_compile['base'] = self.base_postprocess(mpid=mpid)
        if not self.defchg:
            full_compile['freysoldt'] = {'electrostatic_corr': 0., 'potalign_corr': 0.,
                        'total_corr': 0., 'three_axis_data': [0.,0.,0.]}
            potalign = 0.
        else:
            full_compile['freysoldt'] = self.freysoldt()
            # full_compile['kumagai'] = self.kumagai()
            potalign = - float(full_compile['freysoldt']['potalign_corr']) / float(self.defchg)

        full_compile['bandfilling'] = self.bandfilling( potalign=potalign)

        if print_to_file:
            pat_dump = os.path.join(self.defpath, 'outfile.json')
            dumpfn(full_compile, pat_dump, cls=MontyEncoder, indent=2)
        else:
            return full_compile


    def base_postprocess(self, mpid=None):
        #if mpid provided then can use MP level data, otherwise use loaded bulk vr file data

        #first get information for electron chemical potential
        if not mpid:
            logger.warning(
                'No mp-id provided, will fetch CBM/VBM details from the '
                'bulk calculation.\nNote that it would be better to '
                'perform real band structure calculation...')
            bandgap = self.blkvr.eigenvalue_band_properties[0]
            vbm = self.blkvr.eigenvalue_band_properties[2]
        else:
            with MPRester() as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if not bs:
                logger.error("Could not fetch band structure!")
                raise ValueError("Could not fetch band structure!")

            vbm = bs.get_vbm()['energy']
            if not vbm:
                try:
                    vbm = bs.efermi #a hack for GGA gap errors
                except:
                    vbm = 0.
            bandgap = bs.get_band_gap()['energy']

        #second get atomic chemical potential information from phase diagram
        substitution_species = {} #TODO: make smarter approach to checking if sub species were calculated
        if mpid:
            cpa = MPChemPotAnalyzer( mpid = mpid, sub_species = substitution_species ) #, mapi_key = mapi_key)
        else:
            cpa = MPChemPotAnalyzer( bulk_ce= self.blkvr.get_computed_entry(),
                                   sub_species = substitution_species) #, mapi_key = mapi_key)

        chem_lims = cpa.analyze_GGA_chempots()
        final_base_parse = {'vbm': vbm, 'gga_bandgap':bandgap, 'chem_lims': chem_lims}

        return final_base_parse


    def freysoldt(self):
        #average the freysoldtcorrection over threeaxes (to try to eliminate potential alignment uncertainty)
        avgcorr = []
        fullcorrset = []
        #TODO: if you set title then can store potential alignment plots for freysoldt...
        for ax in range(3):
            corr_meth = FreysoldtCorrection( ax, self.diel, self.blklp, self.deflp, self.defchg)
            # valset = corr_meth.correction(title=title+'ax'+str(ax+1), partflag='AllSplit')
            valset = corr_meth.correction(partflag='AllSplit')
            avgcorr.append(valset[1])
            fullcorrset.append(valset)
            # if title: #move the file
            #     homepat = os.path.abspath('.')
            #     src = os.path.join(homepat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
            #     dst = os.path.join(dpat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
            #     shutil.move(src, dst)
        freycorrection = {'electrostatic_corr': valset[0], 'potalign_corr': np.mean(avgcorr),
                        'total_corr': valset[0]+np.mean(avgcorr), 'three_axis_data': fullcorrset}

        return freycorrection


    def kumagai(self):
        if os.path.exists(os.path.join( self.blkpath, 'OUTCAR')):
            bulk_outcar_path = os.path.join( self.blkpath, 'OUTCAR')
        elif os.path.exists(os.path.join( self.blkpath, 'OUTCAR.gz')):
            bulk_outcar_path = os.path.join( self.blkpath, 'OUTCAR.gz')

        if os.path.exists(os.path.join( self.blkpath, 'OUTCAR')):
            def_outcar_path = os.path.join( self.blkpath, 'OUTCAR')
        elif os.path.exists(os.path.join( self.blkpath, 'OUTCAR.gz')):
            def_outcar_path = os.path.join( self.blkpath, 'OUTCAR.gz')

        print('at Kumagai. Using outcars:\n',bulk_outcar_path,'\n',def_outcar_path)
        print('\nother inputs =\n',self.diel, self.defchg,type(self.blkvr),
              type(self.defvr),self.blkvr.final_structure.lattice.abc)

        KC=KumagaiCorrection(self.diel, self.defchg, None, None, #these Nones can be gamma and g_sum (below)
                self.blkvr.final_structure, defstructure=self.defvr.final_structure,
                energy_cutoff=520, madetol=0.0001, lengths=self.blkvr.final_structure.lattice.abc,
                bulk_outcar=bulk_outcar_path, defect_outcar=def_outcar_path)

        #substantial time can be saved if KumagaiBulk object below is reused - it only depends on the bulk structure
        KumagaiBulk = KumagaiBulkInit(self.bulkvr.final_structure, KC.dim, self.diel)

        KC.gamma = KumagaiBulk.gamma
        KC.g_sum = KumagaiBulk.g_sum

        #TODO: if you set title then can store potential alignment plots for kumagai...
        # valset = KC.correction(title=title, partflag='AllSplit')
        valset = KC.correction( partflag='AllSplit')

        # if title: #move the file
        #     homepat = os.path.abspath('.')
        #     src = os.path.join(homepat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
        #     dst = os.path.join(dpat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
        #     shutil.move(src, dst)

        kumagaicorrection = {'electrostatic_corr': valset[0], 'potalign_corr': valset[1],
                            'total_corr': valset[2]}

        return kumagaicorrection


    def bandfilling(self, potalign=0.):
        #potalign can be determined from either Kumagai or Freysoldt corrections
        #NOTE that if using above dictionaries, potalign corresponds to  ('potalign_corr' / (-charge))
        bfc = ExtraCorrections(self.defvr, potalign=potalign, bulk_vr=self.blkvr)

        num_hole_vbm, num_elec_cbm, bf_corr = bfc.calculate_bandfill()
        mbshift = bfc.bfcorr #note that mbshift + slshift = bf_corr above normally, but no shifts considered yet
        vbm = bfc.vbm
        gapstates = {'occupied':[[valoccu[0]-vbm, valoccu[1]] for valoccu in bfc.result_occu],
                     'unoccupied': [[valunoccu- vbm] for valunoccu in bfc.result_unocu]}

        extracorrset = {'num_hole_vbm':num_hole_vbm, 'num_elec_cbm':num_elec_cbm,
                        'mbshift': mbshift, 'gapstates': gapstates}
        return extracorrset



    def bandshifting(self):
        #several things are still in development here. Would rather wait to see how we store information before adding
        pass
