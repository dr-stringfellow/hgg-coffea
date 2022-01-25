import os
import json
import argparse

#Import coffea specific features
from coffea.processor import run_uproot_job, futures_executor

#SUEP Repo Specific
from workflows.case_jetlibrary import *

#Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--chunk', type=int, default=10000, help="")
options = parser.parse_args()

out_dir = os.getcwd()
modules_era = []
#Run the SUEP code. Note the xsection as input. For Data the xsection = 1.0 from above
modules_era.append(NanoProcessor(isMC=options.isMC,sample=options.dataset,output_location="/mnt/hadoop/scratch/bmaier/"))

for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 10,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize=options.chunk
    )


