#!/usr/bin/env python
import os, sys, re, math, glob
from pathlib import Path

ENDPOINT_URL='https://s3.cr.cnaf.infn.it:7480/'

jobstring  = '''#!/bin/bash
ulimit -c 0 -S
ulimit -c 0 -H

# Experiment executable config
echo "STARTING THE RECONSTRUCTION JOB NOW..."
export CVMFS_PARENT_DIR=""
export PATH=$CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/script:$PATH
export PYTHONPATH="${PYTHONPATH}:$CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/packages/py/Ubuntu22.04_Py3.11.9/"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-ubuntu2204-gcc11-opt/setup.sh
source /cvmfs/sft-cygno.infn.it/config/setup_digi.sh
if [ ! -L /usr/include/numpy ]; then
   ln -s $CVMFS_PARENT_DIR/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-ubuntu2204-gcc11-opt/lib/python3.9/site-packages/numpy/core/include/numpy/ /usr/include/numpy
   echo "numpy link necessary. Done."
fi
echo "Untar the code:"
tar xzvf code.tgz
mkdir plots
echo "setup done, now the main program:"
'''

def makeCodeTgz(recodir):
    filesToTransfer = [str(f) for f in Path(recodir).glob(f"*.py")]
    filesToTransfer.extend([str(f) for f in Path(recodir).glob(f"*.txt")])
    filesToTransfer = [os.path.basename(f) for f in filesToTransfer]
    filesToTransfer.extend(["cython_cygno.pyx","cythonize.sh","cython_cygno.c","cython_cygno.cpython-39-x86_64-linux-gnu.so"])
    pedestal_files = [str(p.relative_to(recodir)) for p in (Path(recodir) / "pedestals").glob("*.txt")]
    filesToTransfer.extend(pedestal_files)
    
    dirsToTransfer = ["data","debug_code","cluster","__pycache__","modules_config"]
    print(f"Making a tar.gz file to transfer only the necessary files: {filesToTransfer} and directories: {dirsToTransfer}...")
    os.system(f"tar -czf code.tgz -C {recodir} {' '.join(filesToTransfer)} {' '.join(dirsToTransfer)}")

def makeInputList(fileswithdir,inputdir):
    wget_cmds = []
    for ifwdir in fileswithdir:
        ifile=f'{inputdir}/{ifwdir}'
        inputcloud = re.sub(r'^.*?(?=cygno-)', '', ifile)
        inputcloud = os.path.normpath(inputcloud)
        full_url = f"{ENDPOINT_URL}cygno:{inputcloud}"
        full_url = re.sub(r'(?<!:)/{2,}', '/', full_url) # remove eventual last // wich prevents wget from cloud
        wget_cmds.append(f"wget {full_url}")
    return wget_cmds
        
def makePreSign(jobdir,subdir,jobnumber,outfile,options):
    BUCKET=options.bucket
    TAG=f'{options.storagedir}/{subdir}/job_{jobnumber}'
    print(f"jobdir = {jobdir}, path_jobdir = {Path(jobdir).name}, TAG = {TAG}")
    FILETOKEN='/tmp/token'
    
    cmd = f'/cvmfs/sft-cygno.infn.it/config/lib/presigned.py -u {ENDPOINT_URL} -b {BUCKET} -t {TAG} {outfile} -f {FILETOKEN} > {jobdir}/presign_job{jobnumber}.json'
    print(f"generating presigned url for: with command: {cmd}")
    os.system(cmd)

def makeCondorFile(srcfiles,options,batchn=None):
    dummy_exec = open(f'{jobdir}/dummy_exec.sh','w')
    dummy_exec.write('#!/bin/bash\n')
    dummy_exec.write('bash $*\n')
    dummy_exec.close()

    if not batchn:
        condor_file_name = f'{jobdir}/submit.condor'
    else:
        condor_file_name = f"{jobdir}/submit_{batchn}.condor"
    condor_file = open(condor_file_name,'w')
    condor_file.write('''+SingularityImage = "/cvmfs/sft-cygno.infn.it/dockers/images/cygno-wn_v2.4.sif"
+SingularityBind = "/cvmfs/:/cvmfs/"
Requirements = HasSingularity

Executable = {de}
Log        = {ld}/$(ClusterId).$(ProcId).log
Output     = {ld}/$(ClusterId).$(ProcId).out
Error      = {ld}/$(ClusterId).$(ProcId).error
getenv      = True
next_job_start_delay = 1
request_cpus = {cpu}
should_transfer_files   = YES
preserve_relative_paths = True
+CygnoUser = "{user}"\n
'''.format(de=dummy_exec.name,
           ld=f'{os.path.abspath(options.outdir)}/jobs',
           cpu=options.threads, user=os.environ['USERNAME'], here=jobdir ) )
    for i,src in enumerate(srcfiles):
        srcdir = Path(src).parent
        json_string = ', '.join([str(f) for f in Path(srcdir).glob("*.json")])
        print(f"isrcfile = {i}, src={src}")
        condor_file.write(f'transfer_input_files = code.tgz, {os.path.abspath(src)}, /cvmfs/sft-cygno.infn.it/config/lib/s3upload_put.py, {json_string}\n') # "trailing / is impoprtant: in this way the content of the dir is transferred, the dir itself not
        condor_file.write(f'arguments = {os.path.basename(src)} \nqueue \n\n')
    condor_file.close()

def find_files_with_dirs(root_dir, extension):
    print("Navigating into ",root_dir," to find ",extension," files. It can take time...")
    root_dir = Path(root_dir).resolve()
    extension = extension if extension.startswith('.') else f'.{extension}'
    
    results = []

    for path in root_dir.rglob(f"*{extension}"):
        relative = path.relative_to(root_dir)        
        # directories only (exclude filename)
        dirs = list(relative.parent.parts)
        filename = path.name        
        results.append('/'.join(dirs)+"/"+filename)
        
    return results

def print_submission_cmd(jobdir,ce):
    condor_files = sorted(glob.glob(f"{jobdir}/*.condor"))
    script_name = f"{jobdir.replace('jobs/','')}/submit_all.sh"
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n\n")
        for file in condor_files:
            f.write(f"cygno_htc -s {file} {ce}\n")
    print(f"Submit {len(condor_files)} clusters with the script {script_name}.")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputdir", help="base directory where the ditized files are (nested: path/subpath1/...histograms.root")
    parser.add_argument("-o", "--outdir", type=str, default="./", help='output directory');
    parser.add_argument("-c", "--ce", type=int, default=2, help="Computing element in condor to use")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of CPUs to request")
    parser.add_argument("-j", "--jobrange", nargs=2, type=int, metavar=("jobmin","jobmax"), help="range di job da sottomettere")
    parser.add_argument("-B", "--batchsize", type=int, default=None, help="divide each cluster of jobs in batch of the size batch-size")
    parser.add_argument("-b", "--bucket", type=str, default="cygno-analysis", help='bucket in the cloud where to store the output');
    parser.add_argument("-s", "--storagedir", type=str, default="users/dimarcoe/reco/fe_zcone", help='output directory in the cloud');
    args = parser.parse_args()

    print("SUBMIT RECO")
    absopath  = os.path.abspath(args.outdir)

    jmin=0; jmax=1e4
    if args.jobrange:
        jmin=args.jobrange[0]
        jmax=args.jobrange[1]
    
    jobdir = absopath+'/jobs/'
    if not os.path.isdir(jobdir):
        os.system('mkdir -m 777 -p {od}'.format(od=jobdir))

    makeCodeTgz(os.environ["PWD"])
    
    files = find_files_with_dirs(args.inputdir,".root")
    wgets = makeInputList(files,args.inputdir)
    print(f"List of {len(files)} input files done. Now creating the jobs.")
    
    srcfiles = []
    for j,f in enumerate(files):
        if j<jmin or j>jmax: continue
        outdir = f'{jobdir}/{Path(f).parent}'
        os.system(f'mkdir -m 777 -p {outdir}')
        subdir = Path(f).parent
        inputfile = Path(f).name
        run = int(re.search(r"Run(\d+)\.root", f).group(1))
        makePreSign(outdir,subdir,j,f'reco_run{run:05d}_3D.root',args)

        print(f"Creating job #{j} for run: {run} to be saved in {outdir}\n")

        job_file_name = outdir+"/reco.sh"
        tmp_file = open(job_file_name, 'w')

        tmp_filecont = jobstring
        tmp_filecont += f'\n{wgets[j]}\n'
        tmp_filecont += f"python3 reconstruction.py configFile_MC.txt --pdir plots --max-entries -1 -j{args.threads} -r {run} -t ./ \n"
        tmp_filecont += f"./s3upload_put.py presign_job{j}.json\n"
        tmp_filecont += "\necho DONE.\n"
        tmp_file.write(tmp_filecont)
        tmp_file.close()
        
        srcfiles.append(job_file_name)

    if not args.batchsize:
        cf = makeCondorFile(srcfiles,args)
        subcmd = f'cygno_htc -s {cf} {args.ce}'
        print(f"Scripts prepared in {absopath}") 
        print (f"To submitting jobs run:   '{subcmd}'")

    else:
        chunk = args.batchsize
        chunks = [(srcfiles[i:i+chunk]) for i in range(0, len(srcfiles), chunk)]
        for c, sf in enumerate(chunks, 1):
            print(f"Chunk {c}")
            cf = makeCondorFile(sf,args,c)
        print_submission_cmd(jobdir,args.ce)
    print ("DONE")

        
