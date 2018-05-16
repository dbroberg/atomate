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
from pymatgen import MPRester
from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.generators import VacancyGenerator, SubstitutionGenerator, \
    InterstitialGenerator, VoronoiInterstitialGenerator, SimpleChargeGenerator

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

def optimize_structure_sc_scale(inp_struct, final_site_no):
    """
    A function for optimizing bulk structure size
    (note this copies a function from pycdt.core.defectsmaker)

    TODO: clean this function up to be prettier / make more sense
    TODO: have an option for supercell scaling in a way besides cubic scaling...
    """
    if final_site_no < len(inp_struct.sites):
        final_site_no = len(inp_struct.sites)

    dictio={}
    for k1 in range(1,6):
        for k2 in range(1,6):
            for k3 in range(1,6):
                struct = inp_struct.copy()
                struct.make_supercell([k1, k2, k3])
                if len(struct.sites) > final_site_no:
                    continue

                min_dist = 1000.0
                for a in range(-1,2):
                    for b in range(-1,2):
                        for c in range(-1,2):
                            try:
                                distance = struct.get_distance(0, 0, (a,b,c))
                            except:
                                print (a, b, c)
                                raise
                            if  distance < min_dist and distance>0.00001:
                                min_dist = distance
                min_dist = round(min_dist, 3)
                if min_dist in dictio:
                    if dictio[min_dist]['num_sites'] > struct.num_sites:
                        dictio[min_dist]['num_sites'] = struct.num_sites
                        dictio[min_dist]['supercell'] = [k1,k2,k3]
                else:
                    dictio[min_dist]={}
                    dictio[min_dist]['num_sites'] = struct.num_sites
                    dictio[min_dist]['supercell'] = [k1,k2,k3]
    min_dist = -1.0
    biggest = None
    for c in dictio:
        if c > min_dist:
            biggest = dictio[c]['supercell']
            min_dist = c
    if biggest is None or min_dist < 0.0:
        raise RuntimeError('could not find any supercell scaling vector')
    return biggest






@explicit_serialize
class DefectSetupFiretask(FiretaskBase):
    """
    Run defect supercell setup

    Args:
        structure (Structure): input structure to have defects run on
        cellmax (int): maximum supercell size to consider for supercells
        conventional (bool):
            flag to use conventional structure (rather than primitive) for supercells,
            defaults to True.
        vasp_cmd (string):
            the vasp cmd
        db_file (string):
            the db file

        TODO (for all below readme lists = keep up to date wiht whatever I have listed in atomate.vasp.workflows.base.chgdefects)
        vacancies (list):
            If nothing specified, all vacancies are considered.
            TODO: if more specificity is supplied then limit number of defects created (probably load vacancy pymatgen type)
        antisites (bool):
            If nothing specified, all antisites are considered.
            TODO: if more specificity is supplied then limit number of defects created (probably load Substitution pymatgen type)
        substitutions (dict):
            If nothing specified, NO extrinsic substitutions defects are considered, but ALL intrinsic (antisite) substitutions are considered (default).
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



    """
    def run_task(self, fw_spec):
        if os.path.exists("POSCAR"):
            structure =  Poscar.from_file("POSCAR").structure
        else:
            structure = self.get("structure")

        if self.get("conventional", True):
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        fws, parents = [], []

        cellmax=self.get("cellmax", 128)
        sc_scale = optimize_structure_sc_scale(structure, cellmax)

        #First Firework is for bulk supercell
        bulk_supercell = structure.copy()
        bulk_supercell.make_supercell(sc_scale)
        num_atoms = len(bulk_supercell)

        bulk_incar_settings = {"EDIFF":.0001, "EDIFFG": 0.001, "ISMEAR":0, "SIGMA":0.05, "NSW": 0, "ISIF": 2,
                               "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "LAECHG":False}
        vis = MPStaticSet(bulk_supercell, user_incar_settings =  bulk_incar_settings)

        bulk_tag = "{}:bulk_supercell_{}".format(structure.composition.reduced_formula, num_atoms)

        #TODO: add ability to do transformation for more abstract supercell shapes...
        supercell_size = sc_scale * np.identity(3)
        stat_fw = TransmuterFW(name = bulk_tag, structure=structure,
                               transformations=['SupercellTransformation'],
                               transformation_params=[{"scaling_matrix": supercell_size}],
                               vasp_input_set=vis, copy_vasp_outputs=False, #structure already copied over...
                               vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                               db_file=self.get("db_file", ">>db_file<<"))
        fws.append(stat_fw)


        #Now make defect set
        vacancies = self.get("vacancies", dict())
        antisites = self.get("antisites", dict())
        substitutions = self.get("substitutions", dict())
        interstitials = self.get("interstitials", dict())
        initial_charges  = self.get("initial_charges", dict())


        #track all defect_structures that will be setup/run
        def_structs = []
        #a list with following dict structure for each entry:
        # {'structure': defective structure as supercell,
        # 'charges': list of charges to run,
        # 'transformations': list with pairs of [class for Transformation type, dict for transformation] to create defect (after supercell,
        # 'name': base name (without charge) to be added to firework
        # 'site_multiplicity': site multiplicity of defect AFTER full supercell transformation


        #TODO for all defects belwo could also insert Transmuter for perturbating function / break local symmetry around defect
        if not vacancies:
            #do vacancy set up method...
            copied_sc_structure = bulk_supercell.copy()
            VG = VacancyGenerator(copied_sc_structure)

            for vac_ind, vac in enumerate(VG):
                vac_symbol = vac.site.specie.symbol
                def_name = 'vac_{}_{}'.format(vac_ind+1, vac_symbol)
                dstruct = vac.generate_defect_structure()
                site_mult = vac.multiplicity

                defindex = vac.bulk_structure.index(vac.site)

                transform = [['SupercellTransformation', {"scaling_matrix": supercell_size}],
                             ['RemoveSitesTransformation', {'indices_to_remove': [defindex]}]]


                charges = []
                if initial_charges:
                    if 'vacancies' in initial_charges.keys():
                        if vac_symbol in initial_charges['vacancies']: #NOTE this might get problematic if more than one type of vacancy?
                            charges = initial_charges['vacancies'][vac_symbol]

                if not len(charges):
                    SCG = SimpleChargeGenerator(vac)
                    charges = [v.charge for v in SCG]

                def_structs.append({'name': def_name, 'transformations': transform, 'charges': charges,
                                    'site_multiplicity': site_mult, 'structure': dstruct})


        else:
            #TODO: need to make an option for manual input of vacancy types desired...
            print('nope')
            raise ValueError("DANNY DOESNT KNOW HOW TO DO THIS YET...")



        if antisites:
            #do substitutions set up method....
            copied_sc_structure = bulk_supercell.copy()
            for elt_type in set(bulk_supercell.types_of_specie):
                SG = SubstitutionGenerator(copied_sc_structure, elt_type)
                for as_ind, sub in enumerate(SG):
                    sub_symbol = sub.name.split('_')[1]
                    vac_symbol = sub.name.split('_')[3]
                    def_name = 'as_{}_{}_on_{}'.format(as_ind+1, sub_symbol, vac_symbol)
                    dstruct = sub.generate_defect_structure()
                    site_mult = sub.multiplicity

                    defindex = sub.bulk_structure.index(sub.site)

                    transform = [['SupercellTransformation', {"scaling_matrix": supercell_size}],
                                 ['ReplaceSiteSpeciesTransformation', {'indices_species_map':
                                                                           {defindex: sub_symbol}}]]

                    if initial_charges['antisites']:
                        #TODO: how to interpret input initial charges???
                        raise ValueError("DANNY DOESNT KNOW HOW TO DO THIS YET...")
                    else:
                        charges = []
                        #TODO: do BV method for charge generation?


                    charges = []
                    if initial_charges:
                        if 'substitutions' in initial_charges.keys():
                            if vac_symbol in initial_charges['substitutions']: #NOTE this might get problematic if more than one type of antisite?
                                if sub_symbol in initial_charges['substitutions'][vac_symbol]:
                                    charges = initial_charges['substitutions'][vac_symbol][sub_symbol]

                    if not len(charges):
                        SCG = SimpleChargeGenerator(sub)
                        charges = [v.charge for v in SCG]



                    def_structs.append({'name': def_name, 'transformations': transform, 'charges': charges,
                                        'site_multiplicity': site_mult, 'structure': dstruct})

        else:
            #TODO: need to make an option for manual input of antisite types desired...
            print('nope')
            raise ValueError("DANNY DOESNT KNOW HOW TO DO THIS YET...")

        if substitutions:
            #do substitutions set up method....
            copied_sc_structure = bulk_supercell.copy()
            for elt_type, list_for_subbing in substitutions:
                for sub_elt in list_for_subbing:
                    #sub_elt might need to be an Element type?
                    SG = SubstitutionGenerator(copied_sc_structure, sub_elt)
                    for sub_ind, sub in enumerate(SG):
                        sub_symbol = sub.name.split('_')[1]
                        vac_symbol = sub.name.split('_')[3]
                        if vac_symbol != elt_type:
                            continue
                        def_name = 'sub_{}_{}_on_{}'.format(sub_ind+1, sub_symbol, sub_symbol)
                        dstruct = sub.generate_defect_structure()
                        site_mult = sub.multiplicity
                        defindex = sub.bulk_structure.index(sub.site) #TODO: I dont think this will work??
                        transform = [['SupercellTransformation', {"scaling_matrix": supercell_size}],
                                     ['ReplaceSiteSpeciesTransformation', {'indices_species_map':
                                                                               {defindex: sub_symbol}}]]

                        charges = []
                        if initial_charges:
                            if 'substitutions' in initial_charges.keys():
                                if vac_symbol in initial_charges['substitutions']: #NOTE this might get problematic if more than one type of sub?
                                    if sub_symbol in initial_charges['substitutions'][vac_symbol]:
                                        charges = initial_charges['substitutions'][vac_symbol][sub_symbol]

                        if not len(charges):
                            SCG = SimpleChargeGenerator(sub)
                            charges = [v.charge for v in SCG]


                        def_structs.append({'name': def_name, 'transformations': transform, 'charges': charges,
                                            'site_multiplicity': site_mult, 'structure': dstruct})


        if interstitials:
            #TODO: need to make an option for manual input of interstitials sites desired...
            #
            #     deftrans.append( 'InsertSitesTransformation')
            #     deftrans_params.append( {'species': [defsite.specie.symbol],
            #                             'coords': [defsite.frac_coords],
            #                             'coords_are_cartesian': False} )
            # print('nope')

            #FOR NOW just using simple interstitial generation method...
            #do interstitials set up method....
            copied_sc_structure = bulk_supercell.copy()
            for elt_type in interstitials: #RIGHT now -> interstitials is a list of pymatgen element types?
                IG = VoronoiInterstitialGenerator(copied_sc_structure, elt_type)
                # IG = InterstitialGenerator(copied_sc_structure, sub_elt)
                for inter_ind, inter in enumerate(IG):
                    def_name = 'inter_{}_{}'.format(inter_ind+1, elt_type.symbol)
                    dstruct = inter.generate_defect_structure()
                    site_mult = inter.multiplicity
                    transform = [['SupercellTransformation', {"scaling_matrix": supercell_size}],
                                 ['InsertSitesTransformation', {'species': [inter.site.specie.symbol],
                                                                'coords': [inter.site.frac_coords],
                                                                'coords_are_cartesian': False} ]]

                    charges = []
                    if initial_charges:
                        if 'interstitials' in initial_charges.keys():
                            if elt_type in initial_charges['interstitials']: #NOTE this might get problematic if more than one type of sub?
                                charges = initial_charges['interstitials'][elt_type]

                    if not len(charges):
                        SCG = SimpleChargeGenerator(inter)
                        charges = [v.charge for v in SCG]


                    def_structs.append({'name': def_name, 'transformations': transform, 'charges': charges,
                                        'site_multiplicity': site_mult, 'structure': dstruct})





        stdrd_defect_incar_settings = {"EDIFF":.0001, "EDIFFG":0.001, "IBRION":2, "ISMEAR":0, "SIGMA":0.05,
                                       "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "NSW": 100, "ISIF": 2,
                                       "LAECHG":False }

        metadata = {'multiplicities': {}}
        for defcalc in def_structs:
            # add transformation(s) for creating defect (make supercell then make defect; does not include charge transformation)
            deftrans, deftrans_params = [], []
            for defect_transformation, defect_trans_params in defcalc['transformations']:
                deftrans.append( defect_transformation)
                deftrans_params.append(defect_trans_params)

            #iterate over all charges to be run now
            for charge in defcalc['charges']:
                #apply charge transformation
                #NOTE that this charge transformation has no practical importance for incar...just stylistic?
                chgdeftrans = deftrans[:]
                chgdeftrans.append('ChargedCellTransformation')
                chgdeftrans_params = deftrans_params[:]
                chgdeftrans_params.append({"charge": charge})

                #actually change NELECT incar settings...
                dstruct = defcalc['structure'] #this is supercell structure
                defect_input_set = MPRelaxSet(dstruct, user_incar_settings=stdrd_defect_incar_settings.copy())
                defect_input_set.user_incar_settings["NELECT"] = defect_input_set.nelect - charge

                def_tag = "{}:{}_{}_{}atoms".format(structure.composition.reduced_formula, defcalc['name'],
                                                    charge, num_atoms)

                #storing multiplicity for this sized cell in metadata for parsing purposes later on...
                metadata['metadata']['multiplicities'][def_tag] = defcalc['site_multiplicity']

                fw = TransmuterFW(name = def_tag, structure=structure,
                                       transformations=chgdeftrans,
                                       transformation_params=chgdeftrans_params,
                                       vasp_input_set=defect_input_set,
                                       vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                                       copy_vasp_outputs=False, #structure already copied over...
                                       db_file=self.get("db_file", ">>db_file<<"))
                fws.append(fw)


        return FWAction(detours=fws, update_spec=metadata)


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
