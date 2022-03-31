import os
import json
import argparse
import vector
import glob
import matplotlib.pyplot as plt

from workflows.case_jetlibrary import *


class JetMixer():
    def __init__(self, bothjets: bool, events: np.array, events_mjj_and_deta: np.array, events_extra: np.array, library: np.array, library_extra: np.array, library_idx: np.array):

        self.bothjets = bothjets
        self.library = library
        self.library_extra = library_extra
        self.events = events
        self.events_orig = events
        self.events_orig_mjj_and_deta = events_mjj_and_deta
        self.events_extra = events_extra
        self.events_extra_orig = events_extra
        self.event_kinematics = None
        self.events_pfs = None
        self.kdtree = None
        self.dist_and_ind = None
        
    def fillKDTree(self):
        self.kdtree = KDTree(self.library[:,:3])

    def computeDistances(self):
        self.dist_and_ind = self.kdtree.query(self.events[:,:3], k=2)

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
        #jet1_orig[:,-1] = np.where(jet1_orig[:,-1]>0,jet1_orig[:,-1],0.1)
        jet2_orig = self.events_orig[int(len(self.events_orig)/2):]
        #jet2_orig[:,-1] = np.where(jet2_orig[:,-1]>0,jet2_orig[:,-1],0.1)


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

        dijet = jet1_vecs + jet2_vecs
        dijet_orig = jet1_vecs_orig + jet2_vecs_orig        


        self.event_kinematics = np.stack((self.events_orig_mjj_and_deta[:,0],self.events_orig_mjj_and_deta[:,1],
                                          np.exp(jet1[:,0]),jet1[:,1],jet1[:,2],jet1[:,3],np.exp(jet2[:,0]),jet2[:,1],jet2[:,2],jet2[:,3]),axis=1)

        #print(self.event_kinematics)

        self.plot(dijet_orig,dijet)


    def plot(self,dijet_orig,dijet):
        fig,ax = plt.subplots()
        bins = np.linspace(0,4000,50)
        plt.hist(dijet_orig.mass[dijet_orig.mass>0],label='before mixing',bins=bins,histtype='step',density=True)
        plt.hist(dijet.mass[dijet.mass>0],label='after mixing',bins=bins,histtype='step',density=True)
        plt.xlabel("$m_{jj}$ (GeV)")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/dijet.png",bbox_inches='tight',dpi=300)


        fig,ax = plt.subplots()
        bins_pt = np.linspace(300,1500,40)
        plt.hist(np.exp(self.events_orig[:,0]),label='before mixing',bins=bins_pt,histtype='step',density=True)
        plt.hist(np.exp(self.events[:,0]),label='after mixing',bins=bins_pt,histtype='step',density=True)
        plt.hist(np.exp(self.library[:,0]),label='$\gamma$+jets library',bins=bins_pt,histtype='step',density=True)
        plt.xlabel("$p_\mathrm{T}$ (GeV)")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/pt.png",bbox_inches='tight',dpi=300)


        fig,ax = plt.subplots()
        bins_eta = np.linspace(-2.5,2.5,40)
        plt.hist(self.events_orig[:,1],label='before mixing',bins=bins_eta,histtype='step',density=True)
        plt.hist(self.events[:,1],label='after mixing',bins=bins_eta,histtype='step',density=True)
        plt.hist(self.library[:,1],label='$\gamma$+jets library',bins=bins_eta,histtype='step',density=True)
        plt.xlabel("$\eta$")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/eta.png",bbox_inches='tight',dpi=300)

        fig,ax = plt.subplots()
        bins_phi = np.linspace(-3.14,3.14,40)
        plt.hist(self.events_orig[:,2],label='before mixing',bins=bins_phi,histtype='step',density=True)
        plt.hist(self.events[:,2],label='after mixing',bins=bins_phi,histtype='step',density=True)
        plt.hist(self.library[:,2],label='$\gamma$+jets library',bins=bins_phi,histtype='step',density=True)
        plt.xlabel("$\phi$")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/phi.png",bbox_inches='tight',dpi=300)

        fig,ax = plt.subplots()
        bins_msd = np.linspace(0,300,40)
        plt.hist(self.events_orig[:,3],label='before mixing',bins=bins_msd,histtype='step',density=True)
        plt.hist(self.events[:,3],label='after mixing',bins=bins_msd,histtype='step',density=True)
        plt.hist(self.library[:,3],label='$\gamma$+jets library',bins=bins_msd,histtype='step',density=True)
        plt.xlabel("$m_\mathrm{SD}$ (GeV)")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/msd.png",bbox_inches='tight',dpi=300)

        #print(self.events_extra_orig[:,2]/self.events_extra_orig[:,1])

        
        fig,ax = plt.subplots()
        bins_tau32 = np.linspace(0,2,40)
        plt.hist(self.events_extra_orig[:,2]/self.events_extra_orig[:,1],label='before mixing',bins=bins_tau32,histtype='step',density=True)
        plt.hist(np.divide(self.events_extra[:,2], self.events_extra[:,1], out=np.zeros_like(self.events_extra[:,2]),where=self.events_extra[:,1]!=0),label='after mixing',bins=bins_tau32,histtype='step',density=True)
        plt.hist(np.divide(self.library_extra[:,2], self.library_extra[:,1], out=np.zeros_like(self.library_extra[:,2]),where=self.library_extra[:,1]!=0),label='$\gamma$+jets library',bins=bins_tau32,histtype='step',density=True)
        plt.xlabel("$\\tau_{32}$")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.savefig("/home/bmaier/public_html/figs/case/mixing/tau32.png",bbox_inches='tight',dpi=300)
        

    def dump(self,outfile):

        f = h5py.File(outfile, 'w')
        f.create_dataset("jet_kinematics", data=self.event_kinematics, chunks = True, maxshape=(None, self.event_kinematics.shape[1]))
        f.create_dataset("jet1_extra", data=self.events_extra[:int(len(self.events_extra)/2)].astype(np.float32), chunks = True, maxshape=(None, self.events_extra.shape[1]))
        f.create_dataset("jet2_extra", data=self.events_extra[int(len(self.events_extra)/2):].astype(np.float32), chunks = True, maxshape=(None, self.events_extra.shape[1]))
        f.create_dataset("jet1_PFCands", data=self.events_pfs[:int(len(self.events_pfs)/2),:,:3].astype(np.float32), chunks = True, maxshape=(None,self.events_pfs.shape[1], 3))
        f.create_dataset("jet2_PFCands", data=self.events_pfs[int(len(self.events_pfs)/2):,:,:3].astype(np.float32), chunks = True, maxshape=(None,self.events_pfs.shape[1], 3))
        f.close()



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
        #print("WTF")
        #print(len(f['jet_kinematics']))
        if bothjets:
            j1 = np.stack((np.log(f['jet_kinematics'][()][:,2]),f['jet_kinematics'][()][:,3],
                           f['jet_kinematics'][()][:,4],f['jet_kinematics'][()][:,5]),axis=1)
            j2 = np.stack((np.log(f['jet_kinematics'][()][:,6]),f['jet_kinematics'][()][:,7],
                           f['jet_kinematics'][()][:,8],f['jet_kinematics'][()][:,9]),axis=1)

            jets = np.concatenate((j1,j2))
            #eventIdx_single = f['event_info'][()][:,0]
            #eventIdx_double = np.concatenate((eventIdx_single,eventIdx_single))
            return jets#, eventIdx_double
        else:
            raise NotImplementedError

def get_mjj_and_eta_from_events(eventfile):
    with h5py.File(eventfile, "r") as f:
        a_group_key = list(f.keys())[0]
        mjj_and_eta = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1]),axis=1)
        return mjj_and_eta

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
            pfs1 = f['jet1_extra'][()]
            pfs2 = f['jet2_extra'][()]
            pfs = np.concatenate((pfs1,pfs2))
            return pfs
        else:
            raise NotImplementedError

def get_kinematics_from_jetlibrary(folder):
    alljets = None
    alleventIdx = None
    for i in glob.glob("./%s/*"%folder):
        with h5py.File(i, "r") as f:
            a_group_key = list(f.keys())[0]
            if alljets is None:
                # Getting pT, eta, phi, msoftdrop
                alljets = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1],
                                 f['jet_kinematics'][()][:,2],f['jet_kinematics'][()][:,4]),axis=1)
            
                alleventIdx = f['event_info'][()][:,0]
            else:
                tmpjets = np.stack((f['jet_kinematics'][()][:,0],f['jet_kinematics'][()][:,1],
                                 f['jet_kinematics'][()][:,2],f['jet_kinematics'][()][:,4]),axis=1)
            
                tmpeventIdx = f['event_info'][()][:,0]
                alljets = np.concatenate((alljets,tmpjets),axis=0)
                alleventIdx = np.concatenate((alleventIdx,tmpeventIdx),axis=0)

    return alljets, alleventIdx

def get_pfs_from_jetlibrary(folder):    
    allpfs = None
    for i in glob.glob("./%s/*"%folder):
        with h5py.File(i, "r") as f:
            if allpfs is None:
                allpfs = f['jet_PFCands'][()]
            else:
                tmppfs = f['jet_PFCands'][()]
                allpfs = np.concatenate((allpfs,tmppfs),axis=0)
    return allpfs

def get_extra_from_jetlibrary(folder):    
    allextra = None
    for i in glob.glob("./%s/*"%folder):
        with h5py.File(i, "r") as f:
            if allextra is None:
                allextra = f['jet_extraInfo'][()]
            else:
                tmpextra = f['jet_extraInfo'][()]
                allextra = np.concatenate((allextra,tmpextra),axis=0)
                
    return allextra


def mix(infile,jetlibrary,bothjets,outfile):
    
    events = get_kinematics_from_events(infile,bothjets)
    events_mjj_and_eta = get_mjj_and_eta_from_events(infile)
    library, library_idx = get_kinematics_from_jetlibrary(jetlibrary)
    events_pfs = get_pfs_from_events(infile,bothjets)
    events_extra = get_extra_from_events(infile,bothjets)
    jet_pfs = get_pfs_from_jetlibrary(jetlibrary)
    jet_extra = get_extra_from_jetlibrary(jetlibrary)

    jetmixer = JetMixer(bothjets,events,events_mjj_and_eta,events_extra,library,jet_extra,library_idx)
    jetmixer.fillKDTree()
    jetmixer.computeDistances()
    jetmixer.replaceJets(jet_pfs,jet_extra)
    jetmixer.dump(outfile)

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



