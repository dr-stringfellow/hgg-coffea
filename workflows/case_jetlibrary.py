import coffea
from coffea import hist, processor
from typing import Any, Dict, List, Optional
import numpy as np
import awkward as ak
from sklearn.neighbors import KDTree
import pandas

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
            self,
            isMC: int, 
            sample: str,
            output_location: Optional[str]
    ) -> None:

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
        pddf: pandas.DataFrame,
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
        pddf.to_parquet(local_file)
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
                                         & (abs(events.Photon.eta) <= 2.4))) & (events.Photon.pt > 300)]
        req_pho = (ak.count(events.Photon.pt > 300, axis=1) == 1)


        ## Preselection        
        presel = req_pho & req_ele & req_mu
        selev = events[presel]

        # Get PF constituents in right shape --> they are saved as [nevent,[fj1const1,fj1const2,...,fj1constN,fj2const1,...]]
        # But we need them like [nevent, fjs, consts]
        pfidxs = ak.materialized(selev.FatJetPFCands.jetIdx)
        #print(pfidxs)
        counts = ak.run_lengths(pfidxs)
        #print("counts")
        #print(counts)
        nested_pt = ak.unflatten(selev.PFCands.pt, ak.flatten(counts), axis=1)
        nested_eta = ak.unflatten(selev.PFCands.eta, ak.flatten(counts), axis=1)
        nested_phi = ak.unflatten(selev.PFCands.phi, ak.flatten(counts), axis=1)
        nested_m = ak.unflatten(selev.PFCands.mass, ak.flatten(counts), axis=1)
        #print("nested_pt")
        #print(nested_pt)



        ## FatJet cuts
        _nearPho = selev.FatJet.delta_r(ak.firsts(selev.Photon)) < 0.8
        kinematic_mask = ((selev.FatJet.pt > 260) & (abs(selev.FatJet.eta) <= 2.4) 
                       & (selev.FatJet.isTight == 1))
        mask_fatjet = ~_nearPho & kinematic_mask
        #mask_fatjet = kinematic_mask
        idx = ak.local_index(mask_fatjet, axis=1)
        flat_idx = idx[mask_fatjet]

        nested_pt = self.prepare_const(nested_pt,mask_fatjet)
        nested_eta = self.prepare_const(nested_eta,mask_fatjet)
        nested_phi = self.prepare_const(nested_phi,mask_fatjet)
        nested_m = self.prepare_const(nested_m,mask_fatjet)

        
        print(nested_pt)
        print("XXXX")
        print(len(nested_pt))
        #print(nested_pt.shape)




        '''
        print("mask_fatjet")
        print(mask_fatjet)
        print("flat_idx")
        print(flat_idx)
        print("event")
        print(selev.event)
        '''

        selev.FatJet = selev.FatJet[mask_fatjet]
        req_fatjets = (ak.count(selev.FatJet.pt, axis=1) >= 1)


        #print("req_fatjets")
        #print(req_fatjets)
        #print("len(req_fatjets)")
        #print(len(req_fatjets))


        
        ## FatJets of all events - remove first dimension 
        nFatJet = ak.count(selev.FatJet.eta,axis=-1)

        #print("nFatJet")
        #print(nFatJet)
        #print("len(nFatJet)")
        #print(len(nFatJet))
        
        all_fatjets_pt = np.log(ak.flatten(selev.FatJet.pt))

        #print("pt")
        #print(all_fatjets_pt)
        #print("len(pt)")
        #print(len(all_fatjets_pt))

        all_fatjets_eta = ak.flatten(selev.FatJet.eta)
        all_fatjets_phi = ak.flatten(selev.FatJet.phi)
        all_fatjets_e = np.log(ak.flatten(selev.FatJet.energy))
        all_fatjets_msoftdrop = np.log(ak.flatten(selev.FatJet.msoftdrop))
        all_fatjets_eventNumber = np.repeat(selev.event,nFatJet)

        #print("evtNumber")
        #print(all_fatjets_eventNumber)
        #print("len(evtNumber)")
        #print(len(all_fatjets_eventNumber))

        all_fatjets_idx = ak.flatten(flat_idx)

        '''
        print("fjidx")
        print(all_fatjets_idx)


        print("############")
        print("Constituents")

        print(selev.FatJetPFCands.pt)
        print(selev.FatJetPFCands.jetIdx)

        
        all_features = np.stack((np.array(all_fatjets_pt),np.array(all_fatjets_eta),
                                 np.array(all_fatjets_phi),np.array(all_fatjets_e)),axis=1)

        kdtree = KDTree(all_features)

        # FatJet PF candidates
        print("Can it find it?")
        '''

                
        return output

    def postprocess(self, accumulator):
        return accumulator
