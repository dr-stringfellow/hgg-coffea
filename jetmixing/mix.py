import os
import json
import argparse

from workflows.case_jetlibrary import *


class JetMixer():
    def __init__(self, bothjets: bool, examples: np.array, library: np.array, examples_idx: np.array, library_idx: np.array):

        self.bothjets = bothjets
        self.examples = examples
        self.examples_idx = examples_idx
        self.library = library
        self.library_idx = library_idx
        self.kdtree = None
        self.dist_and_ind = None
        self.closest_ind = None

    def fillKDTree(self):
        self.kdtree = KDTree(self.library)

    def computeDistances(self):
        self.dist_and_ind = self.kdtree.query(self.examples, k=5)

    def cleanEvents(self):
        closest_evtIds = self.library_idx[self.dist_and_ind[1]]
        exampleIds = np.repeat(np.expand_dims(self.examples_idx, axis=1),5,axis=1)

        mask = ak.Array((exampleIds != closest_evtIds))
        counts = ak.sum(mask,axis=1)
        idx = ak.local_index(mask,axis=1)
        unflattened_idx = ak.unflatten(idx[mask],counts)

        print(self.dist_and_ind[1])
        first_idx = np.expand_dims(ak.firsts(unflattened_idx),axis=1)
        print(first_idx)
        final_idx = ak.Array(self.dist_and_ind[1])[ak.Array(first_idx)]
        print(final_idx)

        '''

        self.closest_ind = ak.firsts(final_idx)
        '''

def get_kinematics_from_eventfile(eventfile,bothjets):
    with h5py.File(eventfile, "r") as f:
        a_group_key = list(f.keys())[0]
        if bothjets:
            j1 = np.stack((np.log(f['jet_kinematics'][()][:,2]),f['jet_kinematics'][()][:,3],
                           f['jet_kinematics'][()][:,4]),axis=1)
            j2 = np.stack((np.log(f['jet_kinematics'][()][:,6]),f['jet_kinematics'][()][:,7],
                           f['jet_kinematics'][()][:,8]),axis=1)
            #print(len(f['jet_kinematics'][()][:,4]))
            jets = np.concatenate((j1,j2))
            eventIdx_single = f['event_info'][()][:,0]
            eventIdx_double = np.concatenate((eventIdx_single,eventIdx_single))
            return jets, eventIdx_double
        else:
            raise NotImplementedError

def get_kinematics_from_jetfile(jetfile):    
    with h5py.File(jetfile, "r") as f:
        a_group_key = list(f.keys())[0]
        jets = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1],
                       f['jet_kinematics'][()][:,2]),axis=1)

        eventIdx = f['event_info'][()][:,0]
        return jets, eventIdx

def mix(infile,jetlibrary,bothjets):
    
    bothjets = True

    examples, examples_idx = get_kinematics_from_eventfile(infile,bothjets)
    library, library_idx = get_kinematics_from_jetfile(jetlibrary)

    jetmixer = JetMixer(bothjets,examples,library,examples_idx,library_idx)
    jetmixer.fillKDTree()
    jetmixer.computeDistances()
    jetmixer.cleanEvents()
    #print(library_idx)
    #print(jetmixer.dist_and_ind[1])
    #print(library_idx[jetmixer.dist_and_ind[1]])
    #print(jetmixer.closest_ind)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('--infile', type=str, default="infile.h5", help="")
    parser.add_argument('--outfile', type=str, default="outfile.h5", help="")
    parser.add_argument('--bothjets', type=bool, default=False, help="")
    parser.add_argument('--jetlibrary', type=str, default="X", help="")
    options = parser.parse_args()

    mix(options.infile,options.jetlibrary,options.bothjets)
    print(options.infile)



