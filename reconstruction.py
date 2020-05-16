#!/usr/bin/env python

import os,math,sys,random
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
from root_numpy import hist2array
from cameraChannel import cameraTools

from snakes import SnakesProducer
from output import OutputTree
from treeVars import AutoFillTreeProducer
import swiftlib as sw

import utilities
utilities = utilities.utils()

class analysis:

    def __init__(self,options):
        self.xmax = 2048
        self.rebin = options.rebin        
        self.options = options
        self.pedfile_fullres_name = options.pedfile_fullres_name
        self.tmpname = options.tmpname
        
        if not os.path.exists(self.pedfile_fullres_name):
            print("WARNING: pedestal file with full resolution ",self.pedfile_fullres_name, " not existing. First calculate them...")
            self.calcPedestal(options,1)
        if not options.justPedestal:
           print("Pulling pedestals...")
           # first the one for clustering with rebin
           ctools = cameraTools()
           # then the full resolution one
           pedrf_fr = ROOT.TFile.Open(self.pedfile_fullres_name)
           self.pedmap_fr = pedrf_fr.Get('pedmap').Clone()
           self.pedmap_fr.SetDirectory(0)
           self.pedarr_fr = hist2array(self.pedmap_fr)
           self.noisearr_fr = ctools.noisearray(self.pedmap_fr)
           pedrf_fr.Close()

    # the following is needed for multithreading
    def __call__(self,evrange=(-1,-1,-1)):
        if evrange[0]==-1:
            outfname = self.options.outFile
        else:
            outfname = '{base}_chunk{ij}.root'.format(base=self.options.outFile.split('.')[0],ij=evrange[0])
        self.beginJob(outfname)
        self.reconstruct(evrange)
        self.endJob()
        
    def beginJob(self,outfname):
        # prepare output file
        self.outputFile = ROOT.TFile.Open(outfname, "RECREATE")
        # prepare output tree
        self.outputTree = ROOT.TTree("Events","Tree containing reconstructed quantities")
        self.outTree = OutputTree(self.outputFile,self.outputTree)
        self.autotree = AutoFillTreeProducer(self.outTree)

        self.outTree.branch("run", "I")
        self.outTree.branch("event", "I")
        if self.options.camera_mode:
            self.autotree.createCameraVariables()
            self.autotree.createClusterVariables('cl')
            self.autotree.createClusterVariables('sc')
        if self.options.pmt_mode:
            self.autotree.createPMTVariables()

    def endJob(self):
        self.outTree.write()
        self.outputFile.Close()
        
    def getNEvents(self):
        tf = sw.swift_read_root_file(self.tmpname) #tf = ROOT.TFile.Open(self.rfile)
        ret = len(tf.GetListOfKeys()) if self.options.daq!='midas' else int(len(tf.GetListOfKeys())/2) 
        tf.Close()
        return ret

    def calcPedestal(self,options,alternativeRebin=-1):
        maxImages=options.maxEntries
        nx=ny=self.xmax
        rebin = self.rebin if alternativeRebin<0 else alternativeRebin
        nx=int(nx/rebin); ny=int(ny/rebin); 
        #pedfilename = 'pedestals/pedmap_ex%d_rebin%d.root' % (options.pedexposure,rebin)
        pedfilename = 'pedestals/pedmap_run%s_rebin%d.root' % (options.run,rebin)
        
        pedfile = ROOT.TFile.Open(pedfilename,'recreate')
        pedmap = ROOT.TH2D('pedmap','pedmap',nx,0,self.xmax,ny,0,self.xmax)
        pedmapS = ROOT.TH2D('pedmapsigma','pedmapsigma',nx,0,self.xmax,ny,0,self.xmax)

        pedsum = np.zeros((nx,ny))
        
        tf = sw.swift_read_root_file(self.tmpname)
        #tf = ROOT.TFile.Open(self.rfile)

        # first calculate the mean 
        numev = 0
        for i,e in enumerate(tf.GetListOfKeys()):
            iev = i if self.options.daq != 'midas'  else i/2 # when PMT is present
            if iev in self.options.excImages: continue

            if maxImages>-1 and i<len(tf.GetListOfKeys())-maxImages: continue
            name=e.GetName()
            obj=e.ReadObj()
            if not obj.InheritsFrom('TH2'): continue
            print("Calc pedestal mean with event: ",name)
            if rebin>1:
                obj.RebinX(rebin);
                obj.RebinY(rebin); 
            arr = hist2array(obj)
            pedsum = np.add(pedsum,arr)
            numev += 1
        pedmean = pedsum / float(numev)

        # now compute the rms (two separate loops is faster than one, yes)
        pedsqdiff = np.zeros((nx,ny))
        for i,e in enumerate(tf.GetListOfKeys()):
            iev = i if self.options.daq != 'midas'  else i/2 # when PMT is present
            if iev in self.options.excImages: continue

            if maxImages>-1 and i<len(tf.GetListOfKeys())-maxImages: continue
            name=e.GetName()
            obj=e.ReadObj()
            if not obj.InheritsFrom('TH2'): continue
            print("Calc pedestal rms with event: ",name)
            if rebin>1:
                obj.RebinX(rebin);
                obj.RebinY(rebin); 
            arr = hist2array(obj)
            pedsqdiff = np.add(pedsqdiff, np.square(np.add(arr,-1*pedmean)))
        pedrms = np.sqrt(pedsqdiff/float(numev-1))

        # now save in a persistent ROOT object
        for ix in range(nx):
            for iy in range(ny):
                pedmap.SetBinContent(ix+1,iy+1,pedmean[ix,iy]);
                pedmap.SetBinError(ix+1,iy+1,pedrms[ix,iy]);
                pedmapS.SetBinContent(ix+1,iy+1,pedrms[ix,iy]);
        tf.Close()

        pedfile.cd()
        pedmap.Write()
        pedmapS.Write()
        pedmean1D = ROOT.TH1D('pedmean','pedestal mean',500,97,103)
        pedrms1D = ROOT.TH1D('pedrms','pedestal RMS',500,0,5)
        for ix in range(nx):
            for iy in range(ny):
               pedmean1D.Fill(pedmap.GetBinContent(ix,iy)) 
               pedrms1D.Fill(pedmap.GetBinError(ix,iy)) 
        pedmean1D.Write()
        pedrms1D.Write()
        pedfile.Close()
        print("Pedestal calculated and saved into ",pedfilename)


    def reconstruct(self,evrange=(-1,-1,-1)):

        ROOT.gROOT.Macro('rootlogon.C')
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kRainBow)
        savErrorLevel = ROOT.gErrorIgnoreLevel; ROOT.gErrorIgnoreLevel = ROOT.kWarning

        tf = sw.swift_read_root_file(self.tmpname)
        #tf = ROOT.TFile.Open(self.rfile)
        #c1 = ROOT.TCanvas('c1','',600,600)
        ctools = cameraTools()
        print("Reconstructing event range: ",evrange[1],"-",evrange[2])
        # loop over events (pictures)
        for iobj,key in enumerate(tf.GetListOfKeys()) :
            iev = iobj if self.options.daq != 'midas'  else int(iobj/2) # when PMT is present
            #print("max entries = ",self.options.maxEntries)
            if self.options.maxEntries>0 and iev==max(evrange[0],0)+self.options.maxEntries: break
            if sum(evrange[1:])>-2:
                if iev<evrange[1] or iev>evrange[2]: continue

            name=key.GetName()
            obj=key.ReadObj()

            # Routine to skip some images if needed
            if iev in self.options.excImages: continue

            if self.options.debug_mode == 1 and iev != self.options.ev: continue

            if obj.InheritsFrom('TH2'):
                if self.options.daq == 'btf':
                    run,event=(int(name.split('_')[0].split('run')[-1].lstrip("0")),int(name.split('_')[-1].lstrip("0")))
                elif self.options.daq == 'h5':
                    run,event=(int(name.split('_')[0].split('run')[-1]),int(name.split('_')[-1]))
                else:
                    run,event=(int(name.split('_')[1].split('run')[-1].lstrip("0")),int(name.split('_')[-1].split('ev')[-1]))
                print("Processing Run: ",run,"- Event ",event,"...")
                self.outTree.fillBranch("run",run)
                self.outTree.fillBranch("event",event)

            if self.options.camera_mode:
                if obj.InheritsFrom('TH2'):
     
                    pic_fullres = obj.Clone(obj.GetName()+'_fr')
                    img_fr = hist2array(pic_fullres)

                    # Upper Threshold full image
                    img_cimax = np.where(img_fr < self.options.cimax, img_fr, 0)
                    
                    # zs on full image + saturation correction on full image
                    if self.options.saturation_corr:
                    	#print("you are in saturation correction mode")
                    	img_fr_sub = ctools.pedsub(img_cimax,self.pedarr_fr)
                    	img_fr_satcor = ctools.satur_corr(img_fr_sub) 
                    	img_fr_zs  = ctools.zsfullres(img_fr_satcor,self.noisearr_fr,nsigma=self.options.nsigma)
                    	img_rb_zs  = ctools.arrrebin(img_fr_zs,self.rebin)
                    
                    # skip saturation and set satcor =img_fr_sub 
                    else:
                    	#print("you are in poor mode")
                    	img_fr_sub = ctools.pedsub(img_cimax,self.pedarr_fr)
                    	img_fr_satcor = img_fr_sub  
                    	img_fr_zs  = ctools.zsfullres(img_fr_satcor,self.noisearr_fr,nsigma=self.options.nsigma)
                    	img_rb_zs  = ctools.arrrebin(img_fr_zs,self.rebin)
                    
                    
                    # Cluster reconstruction on 2D picture
                    algo = 'DBSCAN'
                    if self.options.type in ['beam','cosmics']: algo = 'HOUGH'
                    snprod_inputs = {'picture': img_rb_zs, 'pictureHD': img_fr_satcor, 'picturezsHD': img_fr_zs, 'pictureOri': img_fr, 'name': name, 'algo': algo}
                    plotpy = options.jobs < 2 # for some reason on macOS this crashes in multicore
                    snprod_params = {'snake_qual': 3, 'plot2D': False, 'plotpy': False, 'plotprofiles': False}
                    snprod = SnakesProducer(snprod_inputs,snprod_params,self.options)
                    clusters,snakes = snprod.run()
                    self.autotree.fillCameraVariables(img_fr_zs)
                    self.autotree.fillClusterVariables(snakes,'sc')
                    self.autotree.fillClusterVariables(clusters,'cl')
                    
            if self.options.pmt_mode:
                if obj.InheritsFrom('TGraph'):
                    # PMT waveform reconstruction
                    from waveform import PeakFinder,PeaksProducer
                    wform = tf.Get('wfm_'+'_'.join(name.split('_')[1:]))
                    # sampling was 5 GHz (5/ns). Rebin by 5 (1/ns)
                    pkprod_inputs = {'waveform': wform}
                    pkprod_params = {'threshold': options.threshold, # min threshold for a signal (baseline is -20 mV)
                                     'minPeakDistance': options.minPeakDistance, # number of samples (1 sample = 1ns )
                                     'prominence': options.prominence, # noise after resampling very small
                                     'width': options.width, # minimal width of the signal
                                     'resample': options.resample,  # to sample waveform at 1 GHz only
                                     'rangex': (self.options.time_range[0],self.options.time_range[1]),
                                     'plotpy': options.pmt_plotpy
                    }
                    pkprod = PeaksProducer(pkprod_inputs,pkprod_params,self.options)
                    
                    peaksfinder = pkprod.run()
                    self.autotree.fillPMTVariables(peaksfinder,0.2*pkprod_params['resample'])
                    
            # fill reco tree (just once/event, and the TGraph is analyses as last)
            if (self.options.daq == 'midas' and obj.InheritsFrom('TGraph')) or self.options.daq != 'midas':
                self.outTree.fill()

        ROOT.gErrorIgnoreLevel = savErrorLevel

                
if __name__ == '__main__':
    
    from optparse import OptionParser
    #from debug_code.tools_lib import inputFile
    
    parser = OptionParser(usage='%prog h5file1,...,h5fileN [opts] ')
    parser.add_option('-r', '--run', dest='run', default='00000', type='string', help='run number with 5 characteres')
    parser.add_option('-j', '--jobs', dest='jobs', default=1, type='int', help='Jobs to be run in parallel (-1 uses all the cores available)')
    parser.add_option(      '--max-entries', dest='maxEntries', default=-1, type='float', help='Process only the first n entries')
    parser.add_option(      '--pdir', dest='plotDir', default='./', type='string', help='Directory where to put the plots')
    #parser.add_option(      '--tmppath', dest='tmpname', default='/tmp', type='string', help='Directory where to keep tmp histograms')
    
    (options, args) = parser.parse_args()
    
    f = open(args[0], "r")
    params = eval(f.read())
    
    for k,v in params.items():
        setattr(options,k,v)
    if options.debug_mode == 1:
        setattr(options,'outFile','reco_run%s_%s_debug.root' % (options.run, options.tip))
        if options.ev: options.maxEntries = options.ev + 1
        #if options.daq == 'midas': options.ev +=0.5 
    else:
        setattr(options,'outFile','reco_run%05d_%s.root' % (int(options.run), options.tip))
    setattr(options,'pedfile_name', 'pedestals/pedestals/pedmap_run%s_rebin%d.root' % (options.pedrun,options.rebin))
    setattr(options,'pedfile_fullres_name', 'pedestals/pedmap_run%s_rebin1.root' % (options.pedrun))
    
    #inputf = inputFile(options.run, options.dir, options.daq)

    if sw.checkfiletmp(int(options.run), options.tmpname):
       #print("I am checking if there is a file in: " + options.tmpname)
       options.tmpname = options.tmpname + "/histograms_Run%05d.root" % int(options.run)
    
    #if there is no file in given tmp directory  = options.tmpname it will set tmpname as "/tmp" and look for histograms there
    else:
        #print ('Downloading file: ' + sw.swift_root_file(options.tag, int(options.run)))
        options.tmpname = sw.swift_download_root_file(sw.swift_root_file(options.tag, int(options.run)),int(options.run),options.tmpname)
    
    if options.justPedestal:
        ana = analysis(options)
        print("Pedestals done. Exiting.")
        sw.swift_rm_root_file(options.tmpname)
        sys.exit(0)     
    
    ana = analysis(options)
    nev = ana.getNEvents() if options.maxEntries == -1 else int(options.maxEntries)
    print("This run has ",nev," events.")
    print("Will save plots to ",options.plotDir)
    os.system('cp utils/index.php {od}'.format(od=options.plotDir))
    
    nThreads = 1
    if options.jobs==-1:
        import multiprocessing
        nThreads = multiprocessing.cpu_count()
    else:
        nThreads = options.jobs

    if nThreads>1:
        print ("RUNNING USING ",nThreads," THREADS.")
        nj = int(nev/nThreads)
        chunks = [(ichunk,i,min(i+nj-1,nev)) for ichunk,i in enumerate(range(0,nev,nj))]
        print(chunks)
        from multiprocessing import Pool
        pool = Pool(nThreads)
        ret = pool.map(ana, chunks)
        print("Now hadding the chunks...")
        base = options.outFile.split('.')[0]
        os.system('{rootsys}/bin/hadd -f {base}.root {base}_chunk*.root'.format(rootsys=os.environ['ROOTSYS'],base=base))
        os.system('rm {base}_chunk*org.root'.format(base=base))
    else:
        ana.beginJob(options.outFile)
        ana.reconstruct()
        ana.endJob()

    #### FOR SOME REASON THIS DOESN'T WORK IN BATCH.
    # now add the git commit hash to track the version in the ROOT file
    # tf = ROOT.TFile.Open(options.outFile,'update')
    # githash = ROOT.TNamed("gitHash",str(utilities.get_git_revision_hash()).replace('\n',''))
    # githash.Write()
    # tf.Close()
    
    if options.donotremove == False:
        sw.swift_rm_root_file(options.tmpname)
