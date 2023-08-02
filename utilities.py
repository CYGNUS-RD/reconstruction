#!/usr/bin/env python
import subprocess

import os,sys,optparse,csv,resource
import numpy as np
import ROOT,math,uproot
import swiftlib as sw
import matplotlib.pyplot as plt            
from cameraChannel import cameraTools, cameraGeometry

font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }

class utils:
    def __init__(self):
        pass

    def dynamicProfileBins(self,hits,coord='x',relError=0.1):
        minPixels = max(1,1/relError/relError)
        index = 0 if coord=='x' else 1
        xmin=min([h[index] for h in hits])
        xmax=max([h[index] for h in hits])
        x=int(xmin)
        xedges=[x]
        integral=0
        while x<xmax:
            if integral<minPixels:
                integral += sum([int(h[index])==int(x) for h in hits])
            else:
                xedges.append(x)
                integral=0
            x+=1
        xedges.append(int(xmax))
        return xedges


    def dynamicProfileBins_v2(self,hits,coord='x',relError=0.1):
        import numpy as np
        hits = np.array(hits)

        minPixels = max(1,1/relError/relError)
        index = 0 if coord=='x' else 1
        h = hits.T[index].astype(int)
        xmin=min(h)
        xmax=max(h)
        x=xmin
        xedges=[x]
        integral=0

        xunique, xcounts = np.unique(h,return_counts=True)

        if xmin>=0:
            xrange = list(range(0, xmax+1))
            c = np.zeros(len(xrange),dtype = int)
            c[xunique] = xcounts
            c = c[xmin:-1]
        else:
            xrange = list(range(0, (xmax+1)-xmin))
            c = np.zeros(len(xrange),dtype = int)
            c[xunique-xmin] = xcounts

        for ind, x in enumerate(range(xmin,xmax)):
            if integral<minPixels:
                integral += c[ind]
            else:
                xedges.append(x)
                integral=0
        xedges.append(xmax)
        return xedges

    def rotate_around_point(self, hit, dir, pivot, inverse=False):
        x,y = hit[:-1]
        ox, oy = pivot
        cos,sin = dir
        if inverse: cos = -1*cos
        qx = ox + cos * (x - ox) + sin * (y - oy)
        qy = oy - sin * (x - ox) + cos * (y - oy)
        return qx, qy

    def gen_rand_limit(self, x1, x2, y1, y2, maxx=2048, maxy=2048):
        import random
        # generate x, y O(1)
        # --x
        left = random.randrange(0, x1)
        right = random.randrange(x2+1, maxx)
        withinx = random.randrange(x1, x2+1)
        # adjust probability of a point outside the box columns
        # a point outside has probability (1/(maxx-w)) v.s. a point inside has 1/w
        # the same is true for rows. adjupx/y adjust for this probability 
        w = abs(x2-x1)
        h = abs(y2-y1)
        adjpx = ((maxx - w)/w/2)
        x = random.choice([left, right] * adjpx + [withinx])
        # --y
        top = random.randrange(0, y1)
        bottom = random.randrange(y2+1, maxy)
        withiny = random.randrange(y1, y2+1)
        if x == left or x == right:
            adjpy = ((maxy- h)/h/2)
            y = random.choice([top, bottom] * adjpy + [withiny])
        else:
            y = random.choice([top, bottom])
        return x, y 

    def get_git_revision_hash(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    def calcVignettingMap(self,run,pedfile,outfile,maxImages=1000,rebin=12,det='lime',daq='midas'):

        ################ GEOMETRY ###
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=det),'r')
        geometryParams = eval(geometryPSet.read())
        cg = cameraGeometry(geometryParams)
        ctools = cameraTools(cg)
        #############################
        
        # pedestal map, full reso
        pedrf_fr = uproot.open(pedfile)
        pedarr_fr = pedrf_fr['pedmap'].values().T
        noisearr_fr = pedrf_fr['pedmap'].errors().T
        
        outname_base = os.path.basename(outfile).split('.')[0]
        tf_out = ROOT.TFile.Open(outname_base+'.root','recreate')

        N = cg.npixx
        nx=int(N/rebin); ny=int(N/rebin);
        normmap = ROOT.TH2D('normmap_{det}'.format(det=det),'normmap',nx,0,N,nx,0,N)
        summap = normmap.Clone('summap_{det}'.format(det=det))
        
        mapsum = np.zeros((nx,nx))

        USER = os.environ['USER']
        if sw.checkfiletmp(int(run),'root'):
            infile = "/tmp/%s/histograms_Run%05d.root" % (USER,int(run))
        else:
            print ('Downloading file: ' + sw.swift_root_file('Data', int(run)))
            infile = sw.swift_download_root_file(sw.swift_root_file('Data', int(run)),int(run))
           
        tf_in = sw.swift_read_root_file(infile)
        
        framesize = 216 if det=='lime' else 0

        #this was a special case with 3 pictures with different orientations
        #files = ["~/Work/data/cygnus/run03930.root","~/Work/data/cygnus/run03931.root","~/Work/data/cygnus/run03932.root"]
        #for f in files:
        #tf_in = ROOT.TFile(infile)
        
        # first calculate the mean 
        for i,key in enumerate(tf_in.keys()):
            iev = i if daq != 'midas'  else i/2 # when PMT is present
            if 'pic' not in key: continue
            if maxImages>-1 and i<len(tf_in.keys())-maxImages: continue
            arr = tf_in[key].values()
            print("Calc pixel sums with event: ",key)
            
            # Upper Threshold full image
            #img_cimax = np.where(arr < 300, arr, 0)
            img_cimax = arr
            img_fr_sub = ctools.pedsub(img_cimax,pedarr_fr)
            img_fr_zs  = ctools.zsfullres(img_fr_sub,noisearr_fr,nsigma=1)
            
            # for lime, remove the borders of the sensor
            if det=='lime':
                #img_fr_zs[:framesize,:]=0
                #img_fr_zs[-framesize:,:]=0
                img_fr_zs[:,:framesize]=0
                img_fr_zs[:,-framesize:]=0
                
            img_rb_zs  = ctools.arrrebin(img_fr_zs,rebin)
            mapsum = np.add(mapsum,img_rb_zs)
        print (mapsum)
     
        # calc the normalized map wrt the center area
        CA = 16
        central_square = mapsum[int((N-CA)/rebin/2):int((N+CA)/rebin/2),int((N-CA)/rebin/2):int((N+CA)/rebin/2)]
        print (central_square)
        norm = np.mean(central_square)
        print ("Now normalizing to the central area value = ",norm)
        mapnorm = mapsum / float(norm)
        if det=='lime':
            framesize_rb = int(framesize/rebin)
            #mapnorm[:framesize_rb,:]=1
            #mapnorm[-framesize_rb:,:]=1
            mapnorm[:,:framesize_rb]=1
            mapnorm[:,-framesize_rb:]=1
        
        # now save in a persistent ROOT object. Threshold to 1
        for ix in range(nx):
            for iy in range(nx):
                normmap.SetBinContent(ix+1,iy+1,min(mapnorm[ix,iy],1.));
                summap.SetBinContent(ix+1,iy+1,mapsum[ix,iy]);
     
        tf_out.cd()
        normmap.Write()
        summap.Write()
        tf_out.Close()
        print("Written the mean map with rebinning {rb}x{rb} into file {outf}.".format(rb=rebin,outf=outfile))

    def getVignette1D(self,filevignette,det='lime'):

        tf_in = ROOT.TFile.Open(filevignette)
        vignettemap = tf_in.Get('normmap_{det}'.format(det=det))
        xmax = vignettemap.GetXaxis().GetBinLowEdge(vignettemap.GetNbinsX()+1)
        rmax = xmax/math.sqrt(2)
        if int(xmax)==2048:
            det = 'lemon'
            extrap = 'lime'
        else:
            det = 'lime'
            extrap = 'lemon'
        vignettemap_meas = vignettemap.Clone('normmap_'+det)
        vignettemap_meas.SetDirectory(0)
        arr = np.array(vignettemap)
        tf_in.Close()

        nbinsx = arr.shape[0]
        nbinsR = int(nbinsx/math.sqrt(2))
        binsizeR = rmax/nbinsR
        print("rmax = ",rmax," nbinsr = ", nbinsR,"  binsizer = ",binsizeR)
        
        centerx = nbinsx/2; centery = centerx

        x = np.arange(0, nbinsx)
        y = np.arange(0, nbinsx)
        
        tf_out = ROOT.TFile.Open('vign1d.root','recreate')

        vign1d = ROOT.TH1F('vign1d','',nbinsR,0,rmax)

        for ib in range(nbinsR):
            rlow  = vign1d.GetXaxis().GetBinLowEdge(ib+1)
            rhigh = vign1d.GetXaxis().GetBinLowEdge(ib+2)
            maskInner = (x[np.newaxis,:]-centerx)**2 + (y[:,np.newaxis]-centery)**2  < (ib+1)**2
            maskOuter = (x[np.newaxis,:]-centerx)**2 + (y[:,np.newaxis]-centery)**2 >= ib**2
            mask = (maskInner == 1) & (maskOuter == 1)
            vals = arr[mask]
            mean = np.mean(vals) 
            meanerr = np.std(vals)/math.sqrt(len(vals))
            print ("bin = ",ib,"\trlow = ",rlow,"\trhigh = ",rhigh,"\tmean = ",mean,"\tin = ",np.count_nonzero(maskInner),"\tout=",np.count_nonzero(maskOuter),"\tnumber=",np.count_nonzero(mask))
            vign1d.SetBinContent(ib+1,mean)
            vign1d.SetBinError(ib+1,meanerr)

        vign1d.SetLineColor(ROOT.kBlack)
        vign1d.SetMarkerColor(ROOT.kBlack)
        vign1d.SetMarkerSize(0.3)
        vign1d.SetMarkerStyle(ROOT.kFullCircle)
        vign1d.SetLineWidth(1)
        vign1d.SetMinimum(0)
        vign1d.GetXaxis().SetTitle("Distance from center (pixels)")
        vign1d.GetYaxis().SetTitle("Avg. LY ratio")

        # now make the vignette map for the other camera (stretching the measured one)
        print ("Now extrapolating from ",det," to the other camera...")
        xmax2 = 2304 if det == 'lemon' else 2048
        nbins2 = int(xmax2/binsizeR)
        vignettemap_stretched = ROOT.TH2F('normmap_'+extrap,'',nbins2,0,xmax2,nbins2,0,xmax2)
        stretch_factor = xmax2/xmax
        center2 = nbins2/2
        for ix in range(nbins2):
            for iy in range(nbins2):
                r_meas = math.hypot(ix-center2,iy-center2)/stretch_factor * binsizeR
                i1d = vign1d.GetXaxis().FindFixBin(r_meas)
                vignettemap_stretched.SetBinContent(ix+1,iy+1,vign1d.GetBinContent(i1d))

        tf_out.cd()
        vignettemap_meas.Write()        
        vignettemap_stretched.Write()        
        vign1d.Write()
        tf_out.Close()

    def plotVignetteMap(self,filein,name='summap_lime'):
        tf = uproot.open(filein)
        vignette = np.rot90(tf[name].values())
        fig = plt.figure(figsize=(12,12))
        plt.imshow(vignette,cmap='binary',origin='upper',vmin=350,vmax=800 )
        plt.xlabel('x (pixels)', font, labelpad=20)
        plt.ylabel('y (pixels)', font, labelpad=20)
        plt.ylim(250,2000)
        plt.gca().invert_yaxis()
        plt.savefig('%s.pdf' % name)
        
    def setPedestalRun(self,options,detector):
        if not hasattr(options,"pedrun"):
            runlog='runlog_%s.csv' % (detector)
            with open("pedestals/%s"%runlog,"r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                # This skips the first row (header) of the CSV file.
                next(csvreader)
                for row in reversed(list(csvreader)):
                    runkey,runtype,comment = row[:3]
                    if row[-12].strip()!='': # >= run 3
                        pedestal_flag = int(row[-12]) # count from the end, because the field [1] is a txt run description that sometimes has ","...
                    else:
                        pedestal_flag = (":PED:" in runtype)
                    nevents = int(row[-2]) if (str(row[-2]).strip()not in ["NULL",'']) else 0
                    if int(runkey)<=int(options.run) and pedestal_flag and nevents>=100:
                        options.pedrun = int(runkey)
                        print("Will use pedestal run %05d which has comment: '%s' and n of events: '%d'" % (int(runkey),comment,int(nevents)))
                        break
            assert hasattr(options,"pedrun"), ("Didn't find the pedestal corresponding to run %d in pedestals/%s. Check the csv runlog dump!"%(options.run,runlog))
        setattr(options,'pedfile_fullres_name', 'pedestals/pedmap_run%s_rebin1.root' % (options.pedrun))


    def setPedestalRun_v2(self,options,detector):
        import pandas as pd
        if not hasattr(options,"pedrun"):
            runlog='runlog_%s_auto.csv' % (detector)

 

            df = pd.read_csv('pedestals/%s'%runlog)
            dffilter = ((df["number_of_events"] >= 100) & (df["pedestal_run"] == 1) & (df["run_number"] <= int(options.run)) & (df["HV_STATE"] == 0))
            runkey = df.run_number[dffilter].values.tolist()[-1]
            comment = df.run_description[dffilter].values.tolist()[-1]
            nevents = df.number_of_events[dffilter].values.tolist()[-1]
            options.pedrun = int(runkey)
            if runkey:
                print("Will use pedestal run %05d which has comment: '%s' and n of events: '%d'" % (int(runkey),comment,int(nevents)))
            else:
                print("Didn't find the pedestal corresponding to run %d in pedestals/%s. Check the csv runlog dump!" % (options.run, runlog))
        setattr(options,'pedfile_fullres_name', 'pedestals/pedmap_run%s_rebin1.root' % (options.pedrun))

        
    def peak_memory_usage(self):
        """Return peak memory usage in MB"""
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        factor_mb = 1 / 1024
        if sys.platform == "darwin":
            factor_mb = 1 / (1024 * 1024)
        return mem * factor_mb
    
    def conversion_env_variables(self, dslow, odb, i = 0, j = 0):

        """
        if i == 'P1UIn5':
            print(odb.data['History']['Display']['GasSystem']['humidity']['Variables'])
            print(odb.data['History']['Display']['GasSystem']['humidity']['Formula'])
            conversion = odb.data['History']['Display']['GasSystem']['humidity']['Formula'][1]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
        """
        if i == 'P1UIn1':
            conversion = odb.data['History']['Display']['Environment']['Temperature']['Formula'][0]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P0IIn0':
            conversion = odb.data['History']['Display']['Environment']['Temperature']['Formula'][1]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P0IIn5':
            conversion = odb.data['History']['Display']['Environment']['Pressure']['Formula'][0]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P0IIn3':
            conversion = odb.data['History']['Display']['Environment']['Pressure']['Formula'][0]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
        """             
        if i == 'P3IIn0':
            conversion = odb.data['History']['Display']['GasSystem']['Filters']['Formula'][0]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P3IIn1':
            conversion = odb.data['History']['Display']['GasSystem']['Filters']['Formula'][1]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P3IIn2':
            conversion = odb.data['History']['Display']['GasSystem']['Filters']['Formula'][2]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
                    
        if i == 'P3IIn3':
            conversion = odb.data['History']['Display']['GasSystem']['Filters']['Formula'][3]
            dslow[i][j] = eval(conversion.replace('x',str(dslow[i][j])))
        """    
        return dslow
    
    def read_env_variables(self, bank, dslow, odb, j=0):
        import midas.file_reader
        from datetime import datetime
        import numpy as np
        from matplotlib import pyplot as plt
        import cygno as cy
        import time
        import pandas as pd
        
        slow = cy.daq_slow2array(bank)
        #print(slow)
        dslow.loc[len(dslow)] = slow
        #print(dslow)
        for i in dslow.keys():
            dslow = self.conversion_env_variables(dslow, odb, i, j)           
        j = j+1
            
        return dslow
        


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    parser = optparse.OptionParser(usage='usage: %prog [opts] ', version='%prog 1.0')
    parser.add_option('', '--make'   , type='string'       , default='calcVignette' , help='run utilities.py (options = calcVignette)')
    (options, args) = parser.parse_args()

    if options.make == 'calcVignette':
        run = 5890
        pedfile = 'pedestals/pedmap_run5861_rebin1.root'
        ut = utils()
        ut.calcVignettingMap(run,pedfile,"vignette_run%05d.root" % run,det='lime',rebin=8,maxImages=10000)

    if options.make == 'vignette1d':
        ut = utils()
        ut.getVignette1D('vignette_run04117.root')

    if options.make == 'plotVignette':
        ut = utils()
        ut.plotVignetteMap("vignette_run05890.root","summap_lime")

        
