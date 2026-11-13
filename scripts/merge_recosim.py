#!/usr/bin/env python
from os import system,access,F_OK,popen
from submit_condor_recosim import find_files_with_dirs

def printAndExec(cmd,onlyprint=False):
    print(cmd)
    if not onlyprint:
        result = popen(cmd).read()
        print(result)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputdir", help="base directory where the reconstructed files are (nested: path/subpath1/...histograms.root")
    parser.add_argument('--outputname', default="merged.root", type=str, help='output file name')
    args = parser.parse_args()

    print("Start..")
    files = find_files_with_dirs(args.inputdir,".root")

    outputfn = args.outputname
    result = sorted(files)
    print(f"===> List of {len(result)} files to be hadded:")
    print(result)
    print("================================")

    if access(outputfn,F_OK):
        print("skipping",outputfn)
        exit(0)
    if len(result) > 24:
        system(f"mkdir -p {args.inputdir}/Chunks")
        filesperintermediate = int((len(result))**0.5)
        subres = []
        nextone = 0
        while nextone < len(result):
            subres.append(result[nextone:nextone+filesperintermediate])
            nextone += filesperintermediate
        mediumlist = []    
        for i in range(len(subres)):
            mediumfile = outputfn.replace(".root",f"{args.inputdir}/intermediate{i}.root")
            mediumlist.append(mediumfile)
            if access(mediumfile,F_OK):
                print("skipping",mediumfile)
                continue
            cmd = "hadd %s %s" % (f"{args.inputdir}/{mediumfile}"," ".join([f"{args.inputdir}/{fnn}" for fnn in subres[i]]))
            printAndExec(cmd)
        cmd = "hadd %s %s" % (f"{args.inputdir}/{outputfn}"," ".join(mediumlist)) 
        printAndExec(cmd)
    else:    
        cmd = "hadd %s %s" % (f"{args.inputdir}/{outputfn}"," ".join([f"{args.inputdir}/{fnn}" for fnn in result]))
        printAndExec(cmd)
        
    # print("Now moving all the chunks files in Chunks and remove the intermediate files...")
    # system("mv %s {args.inputdir}/Chunks" % " ".join([fnn for fnn in result]))
    # system("rm {args.inputdir}/*intermediate*root")

