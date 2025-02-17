import os
import sys
import json
import argparse
import time

import numpy as np

import uproot
from coffea import hist
from coffea.nanoevents import NanoEventsFactory
from coffea.util import load, save
from coffea import processor
from workflows import workflows, taggers


def validate(file):
    try:
        fin = uproot.open(file)
        return fin["Events"].num_entries
    except:
        print("Corrupted file: {}".format(file))
        return


def check_port(port):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        available = True
    except:
        available = False
    sock.close()
    return available


def get_main_parser():
    parser = argparse.ArgumentParser(
        description="Run Hgg Workflows on NanoAOD using coffea processor files"
    )
    # Inputs
    parser.add_argument(
        "--wf",
        "--workflow",
        dest="workflow",
        choices=list(workflows.keys()),
        help="Which processor to run",
        required=True,
    )
    parser.add_argument(
        "--ts",
        "--tagger-set",
        dest="taggers",
        nargs="+",
        default=None,
        choices=list(taggers.keys()),
        help="The tagger set to apply to this run.",
    )
    parser.add_argument(
        "--meta",
        "--metaconditions",
        dest="metaconditions",
        choices=os.listdir(os.path.join(os.path.dirname(__file__), "metaconditions")),
        help="What metaconditions to load",
        required=True,
    )
    parser.add_argument(
        "--systs",
        "--systematics",
        dest="systematics",
        default=False,
        action="store_true",
        help="Run systematic variations and store to output.",
    )
    parser.add_argument(
        "--no-trigger",
        dest="use_trigger",
        default=True,
        action="store_false",
        help="Turn off trigger selection",
    )
    parser.add_argument(
        "-d",
        "--dump",
        default=None,
        help="Path to dump parquet outputs to (default: None)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"output.coffea",
        help="Output filename (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        "--json",
        dest="samplejson",
        default="dummy_samples.json",
        help="JSON file containing dataset and file locations (default: %(default)s)",
    )

    # Scale out
    parser.add_argument(
        "--executor",
        choices=[
            "iterative",
            "futures",
            "parsl/slurm",
            "parsl/condor",
            "dask/condor",
            "dask/slurm",
            "dask/lpc",
            "dask/lxplus",
            "dask/casa",
        ],
        default="futures",
        help="The type of executor to use (default: %(default)s). Other options can be implemented. "
        "For example see https://parsl.readthedocs.io/en/stable/userguide/configuring.html"
        "- `parsl/slurm` - tested at DESY/Maxwell"
        "- `parsl/condor` - tested at DESY, RWTH"
        "- `dask/slurm` - tested at DESY/Maxwell"
        "- `dask/condor` - tested at DESY, RWTH"
        "- `dask/lpc` - custom lpc/condor setup (due to write access restrictions)"
        "- `dask/lxplus` - custom lxplus/condor setup (due to port restrictions)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=12,
        help="Number of workers (cores/threads) to use for multi-worker executors "
        "(e.g. futures or condor) (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--scaleout",
        type=int,
        default=6,
        help="Number of nodes to scale out to if using slurm/condor. Total number of "
        "concurrent threads is ``workers x scaleout`` (default: %(default)s)",
    )
    parser.add_argument(
        "--max-scaleout",
        dest="max_scaleout",
        type=int,
        default=250,
        help="The maximum number of nodes to adapt the cluster to. (default: %(default)s)",
    )
    parser.add_argument(
        "--voms",
        default=None,
        type=str,
        help="Path to voms proxy, accessible to worker nodes. By default a copy will be made to $HOME.",
    )
    # Debugging
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Do not process, just check all files are accessible",
    )
    parser.add_argument("--skipbadfiles", action="store_true", help="Skip bad files.")
    parser.add_argument(
        "--only", type=str, default=None, help="Only process specific dataset or file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N files of each dataset in sample JSON",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=500000,
        metavar="N",
        help="Number of events per process chunk",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Max number of chunks to run in total",
    )
    return parser


if __name__ == "__main__":
    metaCondsPath = os.path.join(os.path.dirname(__file__), "metaconditions")
    parser = get_main_parser()
    args = parser.parse_args()

    if args.output == parser.get_default("output"):
        args.output = f'hists_{args.workflow}_{(args.samplejson).replace("/","_").rstrip(".json")}.coffea'

    # load dataset
    xrootd_pfx = "root://"
    xrd_pfx_len = len(xrootd_pfx)
    with open(args.samplejson) as f:
        sample_dict = json.load(f)
    for key in sample_dict.keys():
        sample_dict[key] = sample_dict[key][: args.limit]
    if args.executor == "dask/casa":
        for key in sample_dict.keys():
            sample_dict[key] = [
                path.replace(
                    path[xrd_pfx_len : xrd_pfx_len + path[xrd_pfx_len:].find("/")]
                    + "/",
                    "xcache",
                )
                for path in sample_dict[key]
            ]

    # For debugging
    if args.only is not None:
        if args.only in sample_dict.keys():  # is dataset
            sample_dict = dict([(args.only, sample_dict[args.only])])
        if "*" in args.only:  # wildcard for datasets
            _new_dict = {}
            print("Will only process the following datasets:")
            for k, v in sample_dict.items():
                if k.lstrip("/").startswith(args.only.rstrip("*")):
                    print("    ", k)
                    _new_dict[k] = v
            sample_dict = _new_dict
        else:  # is file
            for key in sample_dict.keys():
                if args.only in sample_dict[key]:
                    sample_dict = dict([(key, [args.only])])

    # Scan if files can be opened
    if args.validate:
        start = time.time()
        from p_tqdm import p_map

        all_invalid = []
        for sample in sample_dict.keys():
            _rmap = p_map(
                validate,
                sample_dict[sample],
                num_cpus=args.workers,
                desc=f"Validating {sample[:20]}...",
            )
            _results = list(_rmap)
            counts = np.sum([r for r in _results if np.isreal(r)])
            all_invalid += [r for r in _results if type(r) == str]
            print("Events:", np.sum(counts))
        print("Bad files:")
        for fi in all_invalid:
            print(f"  {fi}")
        end = time.time()
        print("TIME:", time.strftime("%H:%M:%S", time.gmtime(end - start)))
        if input("Remove bad files? (y/n)") == "y":
            print("Removing:")
            for fi in all_invalid:
                print(f"Removing: {fi}")
                os.system(f"rm {fi}")
        sys.exit(0)

    # load workflow
    if args.workflow in workflows:
        wf_taggers = None
        if args.taggers is not None:
            for tagger in args.taggers:
                if tagger not in taggers.keys():
                    raise NotImplemented
            wf_taggers = [taggers[tagger]() for tagger in args.taggers]
        with open(os.path.join(metaCondsPath, args.metaconditions)) as f:
            processor_instance = workflows[args.workflow](
                json.load(f),
                args.systematics,
                args.use_trigger,
                args.dump,
                wf_taggers,
            )  # additional args can go here to configure a processor
    else:
        raise NotImplemented

    if args.executor not in ["futures", "iterative", "dask/lpc", "dask/casa"]:
        """
        dask/parsl needs to export x509 to read over xrootd
        dask/lpc uses custom jobqueue provider that handles x509
        """
        if args.voms is not None:
            _x509_path = args.voms
        else:
            try:
                _x509_localpath = (
                    [
                        l
                        for l in os.popen("voms-proxy-info").read().split("\n")
                        if l.startswith("path")
                    ][0]
                    .split(":")[-1]
                    .strip()
                )
            except:
                raise RuntimeError(
                    "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
                )
            _x509_path = os.environ["HOME"] + f'/.{_x509_localpath.split("/")[-1]}'
            os.system(f"cp {_x509_localpath} {_x509_path}")

        env_extra = [
            "export XRD_RUNFORKHANDLER=1",
            f"export X509_USER_PROXY={_x509_path}",
            f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}',
            f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
        ]
        condor_extra = [
            f'source {os.environ["HOME"]}/.bashrc',
        ]

    #########
    # Execute
    if args.executor in ["futures", "iterative"]:
        if args.executor == "iterative":
            _exec = processor.iterative_executor
        else:
            _exec = processor.futures_executor
        output = processor.run_uproot_job(
            sample_dict,
            treename="Events",
            processor_instance=processor_instance,
            executor=_exec,
            executor_args={
                "skipbadfiles": args.skipbadfiles,
                "schema": processor.NanoAODSchema,
                "workers": args.workers,
            },
            chunksize=args.chunk,
            maxchunks=args.max,
        )
    elif "parsl" in args.executor:
        import parsl
        from parsl.providers import LocalProvider, CondorProvider, SlurmProvider
        from parsl.channels import LocalChannel
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor
        from parsl.launchers import SrunLauncher
        from parsl.addresses import address_by_hostname, address_by_query

        if "slurm" in args.executor:
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_slurm",
                        address=address_by_hostname(),
                        prefetch_capacity=0,
                        provider=SlurmProvider(
                            channel=LocalChannel(script_dir="logs_parsl"),
                            launcher=SrunLauncher(),
                            max_blocks=(args.scaleout) + 10,
                            init_blocks=args.scaleout,
                            partition="all",
                            worker_init="\n".join(env_extra),
                            walltime="00:120:00",
                        ),
                    )
                ],
                retries=20,
            )
        elif "condor" in args.executor:
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_condor",
                        address=address_by_query(),
                        # max_workers=1,
                        provider=CondorProvider(
                            nodes_per_block=1,
                            init_blocks=1,
                            max_blocks=1,
                            worker_init="\n".join(env_extra + condor_extra),
                            walltime="00:20:00",
                        ),
                    )
                ]
            )
        else:
            raise NotImplementedError

        dfk = parsl.load(htex_config)

        output = processor.run_uproot_job(
            sample_dict,
            treename="Events",
            processor_instance=processor_instance,
            executor=processor.parsl_executor,
            executor_args={
                "skipbadfiles": True,
                "schema": processor.NanoAODSchema,
                "config": None,
            },
            chunksize=args.chunk,
            maxchunks=args.max,
        )

    elif "dask" in args.executor:
        from dask_jobqueue import SLURMCluster, HTCondorCluster
        from distributed import Client
        from dask.distributed import performance_report

        if "lpc" in args.executor:
            env_extra = [
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ]
            from lpcjobqueue import LPCCondorCluster

            cluster = LPCCondorCluster(
                transfer_input_files="/srv/workflows/",
                ship_env=True,
                env_extra=env_extra,
            )
        elif "lxplus" in args.executor:
            n_port = 8786
            if not check_port(8786):
                raise RuntimeError(
                    "Port '8786' is not occupied on this node. Try another one."
                )
            import socket

            cluster = HTCondorCluster(
                cores=1,
                memory="2GB",  # hardcoded
                disk="1GB",
                death_timeout="60",
                nanny=False,
                scheduler_options={"port": n_port, "host": socket.gethostname()},
                job_extra={
                    "log": "dask_job_output.log",
                    "output": "dask_job_output.out",
                    "error": "dask_job_output.err",
                    "should_transfer_files": "Yes",
                    "when_to_transfer_output": "ON_EXIT",
                    "+JobFlavour": '"workday"',
                },
                extra=["--worker-port {}".format(n_port)],
                env_extra=env_extra,
            )
        elif "slurm" in args.executor:
            cluster = SLURMCluster(
                queue="all",
                cores=args.workers,
                processes=args.workers,
                memory="200 GB",
                retries=10,
                walltime="00:30:00",
                env_extra=env_extra,
            )
        elif "condor" in args.executor:
            cluster = HTCondorCluster(
                cores=args.workers,
                memory="4GB",
                disk="4GB",
                env_extra=env_extra,
            )

        if args.executor == "dask/casa":
            client = Client("tls://localhost:8786")
            import shutil

            shutil.make_archive("workflows", "zip", base_dir="workflows")
            client.upload_file("workflows.zip")
        else:
            cluster.adapt(minimum=args.scaleout, maximum=args.max_scaleout)
            client = Client(cluster)
            print("Waiting for at least one worker...")
            client.wait_for_workers(1)
        with performance_report(filename="dask-report.html"):
            output = processor.run_uproot_job(
                sample_dict,
                treename="Events",
                processor_instance=processor_instance,
                executor=processor.dask_executor,
                executor_args={
                    "client": client,
                    "skipbadfiles": args.skipbadfiles,
                    "schema": processor.NanoAODSchema,
                    "retries": 3,
                },
                chunksize=args.chunk,
                maxchunks=args.max,
            )

    save(output, args.output)

    print(output)
    print(f"Saving output to {args.output}")
