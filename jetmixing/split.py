import h5py
import sys
import random
import awkward as ak
import numpy as np
filename = sys.argv[1]
nparts = int(sys.argv[2])

with h5py.File(filename, "r") as f:
#with h5py.File("/work/tier3/bmaier/CASE/privatesamples/Jets_all.h5", "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    #print(f['jet_kinematics'][()][0])
    length = [i for i in range(f['jet_kinematics'].shape[0])]
    random.shuffle(length)
    print(f['jet_kinematics'][()])

    '''
    for i in range(nparts):
        new_kinematics = None
        new_pfcands = None
        new_extra = None
        for j in range(i*int(len(length)/nparts),500+(i)*int(len(length)/nparts)):
            if new_kinematics is not None:
                tmp_kinematics = f['jet_kinematics'][j]
                print(tmp_kinematics)
                print(new_kinematics)
                new_kinematics = np.stack([tmp_kinematics,new_kinematics],axis=0)
                tmp_pfcands = f['jet_PFCands'][j]
                new_pfcands = np.stack([tmp_pfcands,new_pfcands],axis=0)
                tmp_extra = f['jet_extraInfo'][j]
                new_extra = np.stack([tmp_extra,new_extra],axis=0)
            else:
                new_kinematics = f['jet_kinematics'][j]
                new_pfcands = f['jet_PFCands'][j]
                new_extra = f['jet_extraInfo'][j]
                
                
            print(j)

        print("Writing file %i" % i)
        print(new_pfcands)
        print(new_pfcands.shape)
        with h5py.File(filename.replace(".h5","_split_%i.h5" % i), "w") as nf:
            nf.create_dataset("jet_kinematics", data=new_kinematics, chunks = True, maxshape=(None, new_kinematics.shape[1]))
            nf.create_dataset("jet_extraInfo", data=new_extra, chunks = True, maxshape=(None, new_extra.shape[1]))
            nf.create_dataset("jet_PFCands", data=new_pfcands, chunks = True, maxshape=(None, new_pfcands.shape[1], 3))
    '''
    
    new_kinematics = ak.Array(f['jet_kinematics'])[length]
    new_pfcands = ak.Array(f['jet_PFCands'])[length]
    new_extra = ak.Array(f['jet_extraInfo'])[length]
    #print(new)
    #print(f['jet_kinematics'].shape)



    for i in range(nparts):
        small_kinematics = np.array(new_kinematics[i*int(len(length)/nparts):(i+1)*int(len(length)/nparts)])
        small_pfcands = np.array(new_pfcands[i*int(len(length)/nparts):(i+1)*int(len(length)/nparts)])
        small_extra = np.array(new_extra[i*int(len(length)/nparts):(i+1)*int(len(length)/nparts)])
        
        print("Writing file %i" % i)
        with h5py.File(filename.replace(".h5","_split_%i.h5" % i), "w") as nf:
            nf.create_dataset("jet_kinematics", data=small_kinematics, chunks = True, maxshape=(None, small_kinematics.shape[1]))
            nf.create_dataset("jet_extraInfo", data=small_extra, chunks = True, maxshape=(None, small_extra.shape[1]))
            nf.create_dataset("jet_PFCands", data=small_pfcands, chunks = True, maxshape=(None, small_pfcands.shape[1], 4))

    
    # Get the data
    #data = list(f[a_group_key])
