import coffea
from coffea import hist, processor
from typing import Any, Dict, List, Optional
import numpy as np
import awkward as ak
from sklearn.neighbors import KDTree
import pandas
import h5py
import os
import pathlib
import shutil
import warnings

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
            self,
            year: int,
            isMC: int, 
            sample: str,
            output_location: Optional[str]
    ) -> None:

        self.year = year
        self.isMC = isMC
        self.sample = sample
        self.output_location = output_location

        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cutflow_axis   = hist.Cat("cut",   "Cut")
       
        # Events
        deta_axis   = hist.Bin("deta",   r"deltaEta(jj)", 50,0,5)
        njet_axis  = hist.Bin("njet",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
                

        # Dijet axes
        phojet_mjj_axis   = hist.Bin("phojet_mjj",   r"pho + jet mjj (GeV)", 50, 200, 4200)


        _hist_phojet_dict = {
            'phojet_mjj'  : hist.Hist("Counts", dataset_axis, phojet_mjj_axis),
        }

         
        _hist_event_dict = {
                'deta'  : hist.Hist("Counts", dataset_axis, deta_axis),
            }
        
        self.phojet_hists = list(_hist_phojet_dict.keys())
        self.event_hists = list(_hist_event_dict.keys())
    
        _hist_dict = {**_hist_phojet_dict, **_hist_event_dict}
        self._accumulator = processor.dict_accumulator(_hist_dict)
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MultivariatePhotonIdentificationRun2#Training_details_and_working_poi
        self.pho_mva80_barrel = 0.42
        self.pho_mva80_endcap = 0.14

    @property
    def accumulator(self):
        return self._accumulator

    
    def prepare_const(
            self,
            consts: ak.Array,
            mask: ak.Array
    ) -> None:
        tmp = ak.flatten(consts[mask])
        tmp = ak.fill_none(ak.pad_none(tmp,100,clip=True),0)
        return np.array(tmp)

    def dump_pandas(
        self,
        npkin: np.array,
        npextra: np.array,
        npevt: np.array,
        nppf: np.array,
        fname: str,
        location: str,
        subdirs: Optional[List[str]] = None,
    ) -> None:
        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError as err:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                ) from err
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )

        with h5py.File(local_file, "w") as f:
            f.create_dataset("event_info", data=npevt, chunks = True, maxshape=(None, npevt.shape[1]))
            f.create_dataset("jet_kinematics", data=npkin, chunks = True, maxshape=(None, npkin.shape[1]))
            f.create_dataset("jet_extraInfo", data=npextra, chunks = True, maxshape=(None, npextra.shape[1]))
            f.create_dataset("jet_PFCands", data=nppf, chunks = True, maxshape=(None, nppf.shape[1], 4))

        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            shutil.copy(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


        
    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
        if self.isMC:
            output['sumw'][dataset] += ak.sum(events.genWeight)
        
        ############
        # Event level
        
        # Event filters
        filters = ["goodVertices",
                   "globalTightHalo2016Filter",
                   "eeBadScFilter", 
                   "HBHENoiseFilter",
                   "HBHENoiseIsoFilter",
                   "ecalBadCalibFilter",
                   "EcalDeadCellTriggerPrimitiveFilter",
                   "BadChargedCandidateFilter"
                  ]

        triggers = [
                'Photon200'
        ]
 
        if self.year == 2016:
            triggers = [
                'Photon175'
            ]

        for f in filters:
            events = events[(getattr(events.Flag,f) == True)]
        for t in triggers:
            events = events[(getattr(events.HLT,t) == True)]

        ## Electron definition
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        events.Electron = events.Electron[(events.Electron.pt > 10) 
                                          & (abs(events.Electron.eta) < 2.4)
                                          & (events.Electron.cutBased == 2)]
        req_ele = (ak.count(events.Electron.pt, axis=1) == 0)

        
        ## Muon definition
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        events.Muon = events.Muon[(events.Muon.pt > 10) 
                                          & (abs(events.Muon.eta) < 2.4)
                                          & (events.Muon.looseId == 1)]
        req_mu = (ak.count(events.Muon.pt, axis=1) == 0)
        

        ## Photon definition
        events.Photon = events.Photon[(((events.Photon.mvaID_WP80 > self.pho_mva80_barrel) & (abs(events.Photon.eta) <= 1.479))
                                      | ((events.Photon.mvaID_WP80 > self.pho_mva80_endcap) & (abs(events.Photon.eta) <= 2.4) 
                                         & (abs(events.Photon.eta) <= 2.4))) & (events.Photon.pt > 220)]
        req_pho = (ak.count(events.Photon.pt > 220, axis=1) == 1)

        ## Preselection        
        presel = req_pho & req_ele & req_mu
        selev = events[presel]

        # Get PF constituents in right shape --> they are saved as [nevent,[fj1const1,fj1const2,...,fj1constN,fj2const1,...]]
        # But we need them like [nevent, fjs, consts]
        pfidxs = ak.materialized(selev.FatJetPFCands.jetIdx)
        counts = ak.run_lengths(pfidxs)

        nested_pt = ak.unflatten(selev.PFCands.pt, ak.flatten(counts), axis=1)
        nested_eta = ak.unflatten(selev.PFCands.eta, ak.flatten(counts), axis=1)
        nested_phi = ak.unflatten(selev.PFCands.phi, ak.flatten(counts), axis=1)
        nested_m = ak.unflatten(selev.PFCands.mass, ak.flatten(counts), axis=1)

        ## FatJet cuts
        _nearPho = selev.FatJet.delta_r(ak.firsts(selev.Photon)) < 0.8
        kinematic_mask = ((selev.FatJet.pt > 300) & (abs(selev.FatJet.eta) <= 2.4) 
                       & (selev.FatJet.isTight == 1))
        mask_fatjet = ~_nearPho & kinematic_mask
        idx = ak.local_index(mask_fatjet, axis=1)
        flat_idx = idx[mask_fatjet]

        #print(flat_idx)

        nested_pt = self.prepare_const(nested_pt,mask_fatjet)
        nested_eta = self.prepare_const(nested_eta,mask_fatjet)
        nested_phi = self.prepare_const(nested_phi,mask_fatjet)
        nested_m = self.prepare_const(nested_m,mask_fatjet)

        selev.FatJet = selev.FatJet[mask_fatjet]
        req_fatjets = (ak.count(selev.FatJet.pt, axis=1) >= 1)        

        ## FatJets of all events
        nFatJet = ak.count(selev.FatJet.eta,axis=-1)

        # pt, eta, phi, e, msd
        all_fatjet_features = np.stack(
            (
                np.array(np.log(ak.flatten(selev.FatJet.pt))),
                np.array(ak.flatten(selev.FatJet.eta)),
                np.array(ak.flatten(selev.FatJet.phi)),
                np.array(np.log(ak.flatten(selev.FatJet.energy))),
                np.array(ak.flatten(selev.FatJet.msoftdrop))
            ),axis=1)     

        all_fatjets_tau1 = ak.flatten(selev.FatJet.tau1)
        all_fatjets_tau2 = ak.flatten(selev.FatJet.tau2)
        all_fatjets_tau3 = ak.flatten(selev.FatJet.tau3)
        all_fatjets_tau4 = ak.flatten(selev.FatJet.tau4)
        all_fatjets_lsf3 = ak.flatten(selev.FatJet.lsf3)
        all_fatjets_nPF = ak.count_nonzero(nested_pt,axis=1)


        print(selev.SubJet.btagDeepB)
        print(selev.FatJet.subJetIdx1)
        print(selev.SubJet.btagDeepB[selev.FatJet.subJetIdx1])

        all_sj1 = ak.flatten(selev.SubJet.btagDeepB[selev.FatJet.subJetIdx1])
        all_sj2 = ak.flatten(selev.SubJet.btagDeepB[selev.FatJet.subJetIdx2])
        all_sj = np.stack((np.array(all_sj1),np.array(all_sj2)),axis=1)

        
        # tau1, tau2, tau3, tau3, lsf3, max subjet b tag, nPF
        all_extra_features = np.stack((np.array(all_fatjets_tau1),np.array(all_fatjets_tau2),
                                       np.array(all_fatjets_tau3),np.array(all_fatjets_tau4),
                                       np.array(all_fatjets_lsf3),np.amax(all_sj,1),
                                       np.array(all_fatjets_nPF)),axis=1)

        # evtNumber, fjIdx
        all_event_features = np.stack(
            (
                np.array(np.repeat(selev.event,nFatJet)),
                np.array(ak.flatten(flat_idx))
            ),axis=1)

        # pfCands
        all_pfcands = np.stack(
            (
                nested_pt,
                nested_eta,
                nested_phi#,
#                nested_m
            ),axis=2)

        # Write-out

        fname = (
            events.behavior["__events_factory__"]._partition_key.replace("/", "_").replace("%2F","").replace("%3B1","")
            + ".h5"
            )

        subdirs = []
        if "dataset" in events.metadata:
            subdirs.append(f'{events.metadata["dataset"]}')

        self.dump_pandas(all_fatjet_features, 
                         all_extra_features, 
                         all_event_features, 
                         all_pfcands, 
                         fname, self.output_location, subdirs)

        return output

    def postprocess(self, accumulator):
        return accumulator
