#!/usr/bin/env python
import os, sys, re
from pathlib import Path

jobstring  = '''#!/bin/bash
ulimit -c 0 -S
ulimit -c 0 -H
set -e

# Experiment executable config
export CVMFS_PARENT_DIR=""
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-ubuntu2204-gcc11-opt/setup.sh
source /cvmfs/sft-cygno.infn.it/config/setup_digi.sh
chmod a+x reconstruction.py
COMMAND
'''

def makeCondorFile(rootfiles,outdirs,runs,srcfiles,logdir,options):
    dummy_exec = open(options.outdir+'/dummy_exec.sh','w')
    dummy_exec.write('#!/bin/bash\n')
    dummy_exec.write('bash $*\n')
    dummy_exec.close()

    condor_file_name = options.outdir+'/condor_submit.condor'
    condor_file = open(condor_file_name,'w')
    condor_file.write('''+SingularityImage = "/cvmfs/sft-cygno.infn.it/dockers/images/cygno-wn_v2.4.sif"
+SingularityBind = "/cvmfs/:/cvmfs/"
Requirements = HasSingularity

Executable = {de}
Log        = {ld}/$(ProcId).log
Output     = {ld}/$(ProcId).out
Error      = {ld}/$(ProcId).error
getenv      = True
next_job_start_delay = 1
environment = "LS_SUBCWD={here}"
request_cpus = {cpu}
should_transfer_files   = YES
preserve_relative_paths = True
+CygnoUser = "{user}"\n
'''.format(de=dummy_exec.name,
           ld=os.path.abspath(logdir),
           cpu=options.threads, user=os.environ['USERNAME'], here=os.environ['PWD'] ) )
    for i,rf in enumerate(rootfiles):
        condor_file.write(f'transfer_input_files = {rf} \n')
        condor_file.write(f'transfer_output_files = reco_run{runs[i]:05d}_3D.root,reco_run{runs[i]:05d}_3D.txt\n')
        condor_file.write(f'transfer_output_remaps = "reco_run{runs[i]:05d}_3D.root = {outdirs[i]}/reco_run{runs[i]:05d}_3D.root,reco_run{runs[i]:05d}_3D.txt = {outdirs[i]}/reco_run{runs[i]:05d}_3D.txt"\n')
        condor_file.write(f'arguments = {srcfiles[i]} \nqueue \n\n')
        
    condor_file.close()
    return condor_file_name


def find_files_with_dirs(root_dir, extension):
    print("Navigating into ",root_dir," tp find ",extension," files. It can take time...")
    root_dir = Path(root_dir).resolve()
    extension = extension if extension.startswith('.') else f'.{extension}'
    
    results = []

    for path in root_dir.rglob(f"*{extension}"):
        relative = path.relative_to(root_dir)        
        # directories only (exclude filename)
        dirs = list(relative.parent.parts)
        filename = path.name        
        results.append((dirs, filename))
        
    return results




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir", help="base directory where the ditized files are (nested: path/subpath1/...histograms.root")
    parser.add_argument("-o", "--outdir", type=str, default="./", help='output directory');
    parser.add_argument("-c", "--ce", type=int, default=2, help="Computing element in condor to use")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of CPUs to request")
    args = parser.parse_args()

    print("SUBMIT RECO")
    absopath  = os.path.abspath(args.outdir)
    
    logdir = absopath+'/logs/'
    if not os.path.isdir(logdir):
        os.system('mkdir -m 777 -p {od}'.format(od=logdir))
    
    files = find_files_with_dirs(args.inputdir,".root")
    print("List of input files done. Now creating the jobs")

    rootfiles,outdirs,runs,srcfiles = [],[],[],[]
    for dirs,f in files:
        outdir = '/'.join([args.outdir]+[d for d in dirs])
        print("outdir will be ",outdir)
        os.system(f'mkdir -m 777 -p {outdir}')
        inputfile = '/'.join([args.inputdir]+[d for d in dirs]+[f])
        run = int(re.search(r'\d+', f).group())

        print(f"Creating job for run: {run} to be saved in {outdir}\n")

        job_file_name = outdir+"/reco.sh"
        tmp_file = open(job_file_name, 'w')

        tmp_filecont = jobstring
        cmd = f"python3 reconstruction.py configFile_MC.txt --pdir plots --max-entries -1 -j{args.threads} -r {run} -t ./"
        tmp_filecont = tmp_filecont.replace('COMMAND',cmd)
        tmp_file.write(tmp_filecont)
        tmp_file.close()
        
        srcfiles.append(job_file_name)
        rootfiles.append(inputfile)
        outdirs.append(outdir)
        runs.append(run)

    cf = makeCondorFile(rootfiles,outdirs,runs,srcfiles,logdir,args)
    subcmd = f'source $CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/config/cygno_htc -s {cf} {args.ce}'
    
    print ("DONE")

        
