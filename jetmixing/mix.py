import os
import json
import argparse
import vector

from workflows.case_jetlibrary import *


class JetMixer():
    def __init__(self, bothjets: bool, events: np.array, events_extra: np.array, library: np.array, events_idx: np.array, library_idx: np.array):

        self.bothjets = bothjets
        self.library = library
        self.events = events
        self.events_orig = events
        self.events_idx = events_idx
        self.events_extra = events_extra
        self.events_extra_orig = events_extra
        self.events_pfs = None
        self.kdtree = None
        self.dist_and_ind = None
        self.closest_ind = None

    def fillKDTree(self):
        self.kdtree = KDTree(self.library[:,:4])

    def computeDistances(self):
        self.dist_and_ind = self.kdtree.query(self.events[:,:4], k=2)

    def replaceJets(self,jet_pfs,jet_extra):
        self.events = self.library[self.dist_and_ind[1][:,0]]
        self.events_extra = jet_extra[self.dist_and_ind[1][:,0]]
        self.events_pfs = jet_pfs[self.dist_and_ind[1][:,0]]


        jet1 = self.events[:int(len(self.events)/2)]
        jet1[:,-1] = np.where(jet1[:,-1]>0,jet1[:,-1],0.1)
        jet2 = self.events[int(len(self.events)/2):]
        jet2[:,-1] = np.where(jet2[:,-1]>0,jet2[:,-1],0.1)

        jet1_vecs = vector.array({
            "pt": np.exp(jet1[:,0]),
            "phi": jet1[:,2],
            "eta": jet1[:,1],
            "M": jet1[:,3],  # some msoftdrop are negative?!
        })
        jet2_vecs = vector.array({
            "pt": np.exp(jet2[:,0]),
            "phi": jet2[:,2],
            "eta": jet2[:,1],
            "M": jet2[:,3],  # some msoftdrop are negative?!
        })


        jet1_orig = self.events_orig[:int(len(self.events_orig)/2)]
        jet1_orig[:,-1] = np.where(jet1_orig[:,-1]>0,jet1_orig[:,-1],0.1)
        jet2_orig = self.events_orig[int(len(self.events_orig)/2):]
        jet2_orig[:,-1] = np.where(jet2_orig[:,-1]>0,jet2_orig[:,-1],0.1)


        jet1_vecs_orig = vector.array({
            "pt": np.exp(jet1_orig[:,0]),
            "phi": jet1_orig[:,2],
            "eta": jet1_orig[:,1],
            "M": jet1_orig[:,3],  # some msoftdrop are negative?!
        })
        jet2_vecs_orig = vector.array({
            "pt": np.exp(jet2_orig[:,0]),
            "phi": jet2_orig[:,2],
            "eta": jet2_orig[:,1],
            "M": jet2_orig[:,3],  # some msoftdrop are negative?!
        })

        #for j1vec in jet1_vecs:
        #    dRs = j1vec.deltaR(jet2_vecs)
        #    print(dRs)

        dijet = jet1_vecs + jet2_vecs
        dijet_orig = jet1_vecs_orig + jet2_vecs_orig        

        print(dijet.mass)
        print(dijet_orig.mass)

    #def dump(self,outfile):
    #    with h5py.File(outfile, "w") as f:
    #        f.create_dataset("event_info", data=npevt, chunks = True, maxshape=(None, npevt.shape[1]))
    #        f.create_dataset("jet_kinematics", data=npkin, chunks = True, maxshape=(None, npkin.shape[1]))
    #        f.create_dataset("jet_extraInfo", data=npextra, chunks = True, maxshape=(None, npextra.shape[1]))
    #        f.create_dataset("jet_PFCands", data=nppf, chunks = True, maxshape=(None, nppf.shape[1], 4))



    '''
    def cleanEvents(self):
        # figure out eventNumbers from the five closest jets to be able to remove overlap (in the end, if the selections are different for SR and GJets, this should do nothing. 
        closest_evtIds = self.library_idx[self.dist_and_ind[1]]
        print(closest_evtIds)
        exit(1)
        exampleIds = np.repeat(np.expand_dims(self.events_idx, axis=1),5,axis=1)

        mask = ak.Array((exampleIds != closest_evtIds))
        counts = ak.sum(mask,axis=1)
        idx = ak.local_index(mask,axis=1)
        unflattened_idx = ak.unflatten(idx[mask],counts)

        print(self.dist_and_ind[1])
        first_idx = np.expand_dims(ak.firsts(unflattened_idx),axis=1)
        print(first_idx)
        final_idx = ak.Array(self.dist_and_ind[1])[ak.Array(first_idx)]
        print(final_idx)


        self.closest_ind = ak.firsts(final_idx)
    '''

def get_kinematics_from_events(eventfile,bothjets):
    with h5py.File(eventfile, "r") as f:
        a_group_key = list(f.keys())[0]
        if bothjets:
            j1 = np.stack((np.log(f['jet_kinematics'][()][:,2]),f['jet_kinematics'][()][:,3],
                           f['jet_kinematics'][()][:,4],f['jet_kinematics'][()][:,5]),axis=1)
            j2 = np.stack((np.log(f['jet_kinematics'][()][:,6]),f['jet_kinematics'][()][:,7],
                           f['jet_kinematics'][()][:,8],f['jet_kinematics'][()][:,9]),axis=1)

            jets = np.concatenate((j1,j2))
            eventIdx_single = f['event_info'][()][:,0]
            eventIdx_double = np.concatenate((eventIdx_single,eventIdx_single))
            return jets, eventIdx_double
        else:
            raise NotImplementedError

def get_pfs_from_events(eventfile,bothjets):
    with h5py.File(eventfile, "r") as f:
        a_group_key = list(f.keys())
        if bothjets:
            pfs1 = f['jet1_PFCands'][()]
            pfs2 = f['jet2_PFCands'][()]
            pfs = np.concatenate((pfs1,pfs2))
            return pfs
        else:
            raise NotImplementedError

def get_extra_from_events(eventfile,bothjets):
    with h5py.File(eventfile, "r") as f:
        a_group_key = list(f.keys())
        if bothjets:
            pfs1 = f['jet1_extraInfo'][()]
            pfs2 = f['jet2_extraInfo'][()]
            pfs = np.concatenate((pfs1,pfs2))
            return pfs
        else:
            raise NotImplementedError

def get_kinematics_from_jetlibrary(jetfile):    
    with h5py.File(jetfile, "r") as f:
        a_group_key = list(f.keys())[0]
        # Getting pT, eta, phi, msoftdrop
        jets = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1],
                         f['jet_kinematics'][()][:,2],f['jet_kinematics'][()][:,4]),axis=1)

        eventIdx = f['event_info'][()][:,0]
        return jets, eventIdx

def get_pfs_from_jetlibrary(jetfile):    
    with h5py.File(jetfile, "r") as f:
        a_group_key = list(f.keys())
        jets = f['jet_PFCands'][()]
        return jets

def get_extra_from_jetlibrary(jetfile):    
    with h5py.File(jetfile, "r") as f:
        a_group_key = list(f.keys())
        jets = f['jet_extraInfo'][()]
        return jets


def mix(infile,jetlibrary,bothjets,outfile):
    
    events, events_idx = get_kinematics_from_events(infile,bothjets)
    library, library_idx = get_kinematics_from_jetlibrary(jetlibrary)
    print(len(libary))
    events_pfs = get_pfs_from_events(infile,bothjets)
    events_extra = get_extra_from_events(infile,bothjets)
    jet_pfs = get_pfs_from_jetlibrary(jetlibrary)
    jet_extra = get_extra_from_jetlibrary(jetlibrary)

    jetmixer = JetMixer(bothjets,events,events_extra,library,events_idx,library_idx)
    jetmixer.fillKDTree()
    jetmixer.computeDistances()
    jetmixer.replaceJets(jet_pfs,jet_extra)
    #jetmixer.dump(outfile)

    #jetmixer.cleanEvents()

    exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('--infile', type=str, default="infile.h5", help="")
    parser.add_argument('--outfile', type=str, default="outfile.h5", help="")
    parser.add_argument('--bothjets', type=bool, default=True, help="Mix both jets or only one?!?!")
    parser.add_argument('--jetlibrary', type=str, default="X", help="")
    options = parser.parse_args()

    mix(options.infile,options.jetlibrary,options.bothjets,options.outfile)
    print(options.infile)



