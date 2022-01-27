import os
import json
import argparse

from workflows.case_jetlibrary import *


class JetMixer():
    def __init__(self, examples: np.array, library: np.array):

        self.examples = examples
        self.library = library
        self.kdtree = None
        self.dist_and_ind = None

    def fillKDTree(self):
        self.kdtree = KDTree(self.library)

    def computeDistances(self):
        self.dist_and_ind = self.kdtree.query(self.examples, k=5)
        

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
            return jets
        else:
            raise NotImplementedError

def get_kinematics_from_jetfile(jetfile):    
    with h5py.File(jetfile, "r") as f:
        a_group_key = list(f.keys())[0]
        jets = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1],
                       f['jet_kinematics'][()][:,2]),axis=1)
        return jets


def mix(infile,jetlibrary,bothjets):
    
    examples = get_kinematics_from_eventfile(infile,bothjets)
    library = get_kinematics_from_jetfile(jetlibrary)

    print(len(library))
    jetmixer = JetMixer(examples,library)
    jetmixer.fillKDTree()
    jetmixer.computeDistances()
    print(jetmixer.dist_and_ind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument('--infile', type=str, default="infile.h5", help="")
    parser.add_argument('--outfile', type=str, default="outfile.h5", help="")
    parser.add_argument('--bothjets', type=bool, default=False, help="")
    parser.add_argument('--jetlibrary', type=str, default="X", help="")
    options = parser.parse_args()

    mix(options.infile,options.jetlibrary,options.bothjets)
    print(options.infile)



