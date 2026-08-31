#! /usr/bin/env python3

import os
cythonized=False
for fname in os.listdir('.'):
        if fname.endswith('.so'):
          cythonized=True
          break
          
if cythonized==False:
          os.system('sh cythonize.sh')
        
from concurrent import futures
from subprocess import Popen, PIPE
import signal,time

import math,sys,random,re,gc
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
import uproot
from cameraChannel import cameraTools, cameraGeometry
import midas.file_reader
import h5py

from snakes import SnakesProducer
from output import OutputTree
from treeVars import AutoFillTreeProducer
import swiftlib as sw
import cygno as cy

import pandas as pd

import utilities
utilities = utilities.utils()

from waveform import PMTreco

class analysis:

    def __init__(self,options):
        self.rebin = options.rebin        
        self.options = options
        if options.camera_mode:
            self.pedfile_fullres_name = options.pedfile_fullres_name
        self.tmpname = options.tmpname
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=options.geometry),'r')
        geometryParams = eval(geometryPSet.read())
        self.cg = cameraGeometry(geometryParams)
        self.xmax = self.cg.npixx
        self.ymax = self.cg.npixy

        eventContentPSet = open('modules_config/reco_eventcontent.txt')
        self.eventContentParams = eval(eventContentPSet.read())
        for k,v in self.eventContentParams.items():
            setattr(self.options,k,v)
        

        if options.camera_mode:
            if not os.path.exists(self.pedfile_fullres_name):
                print("WARNING: pedestal file with full resolution ",self.pedfile_fullres_name, " not existing. First calculate them...")
                self.calcPedestal(options,1)
            if not options.justPedestal:
                print("Pulling pedestals...")
                # first the one for clustering with rebin
                ctools = cameraTools(self.cg)
                # then the full resolution one
                pedrf_fr = uproot.open(self.pedfile_fullres_name)
                self.pedarr_fr   = pedrf_fr['pedmap'].values().T
                self.noisearr_fr = pedrf_fr['pedmap'].errors().T
                if options.vignetteCorr:
                    self.vignmap = ctools.loadVignettingMap()
                else:
                    self.vignmap = np.ones((self.ymax, self.xmax))

        ## Dictionary with the PMT parameters found in config_file
        self.pmt_params = {
            'ch_to_read' : options.board_pmt_channels,             
            'threshold': options.threshold,
            'height_RMS': options.height_RMS, 
            'minPeakDistance': options.minPeakDistance, 
            'prominence': options.prominence,
            'fixed_prom': options.fixed_prom, 
            'width': options.width,
            'resample': options.resample,
            'plotpy': options.pmt_plotpy,
            'wf_in_tree': options.pmt_wf_in_tree,
            'pmt_verb':  options.pmt_verbose,
            'pmt_outdir': options.plotDir,
            'include_gem':  options.include_gem,
            'ch_to_read_gem' : options.board_gem_channels,             
        }
        if options.debug_mode == 1:
            self.pmt_params['pmt_verb']=3
            self.pmt_params['plotpy']= True
        if options.pmt_mode and not options.board_pmt_channels:
            print('\nIt seems you are trying to analyse the PMT signals without selecting their channels. Untoggle PMT mode or add channels.\n ANALYSIS FAILED')
            sys.exit()
        if options.include_gem and not options.pmt_mode:
            print('\nIt seems you are trying to analyse the GEM signals without the PMT Mode On. This does not work.\n ANALYSIS FAILED')
            sys.exit()
        if options.include_gem and not options.board_gem_channels:
            print('\nIt seems you are trying to analyse the GEM signals without selecting their channels. Untoggle GEM mode or add channels.\n ANALYSIS FAILED')
            sys.exit()


    # the following is needed for multithreading
    def __call__(self,evrange=(-1,-1,-1)):
        if evrange[0]==-1:
            outfname = '{outdir}/{base}'.format(base=self.options.outFile,outdir=options.outdir)
        else:
            outfname = '{outdir}/{base}_chunk{ij}.root'.format(base=self.options.outFile.split('.')[0],ij=evrange[0],outdir=self.options.outdir)
        self.beginJob(outfname)
        self.reconstruct(evrange)
        self.endJob()
        
    def beginJob(self,outfname):
        # prepare output file
        ROOT.EnableThreadSafety()
        self.outputFile = ROOT.TFile.Open(outfname, "RECREATE")
        print("Opening out file: ",outfname," self.outputFile = ",self.outputFile)
        ROOT.gDirectory.cd()
        # prepare output tree
        if options.camera_mode or options.environment_variables:
            self.outputTree = ROOT.TTree("Events","Tree containing reconstructed quantities")
            self.outTree = OutputTree(self.outputFile,self.outputTree)
            self.autotree = AutoFillTreeProducer(self.outTree,self.eventContentParams)

        ## Prepare PMT waveform Tree (1 event = 1 waveform)
        if options.pmt_mode:
            self.outputTree_pmt = ROOT.TTree("PMT_Events","Tree containing reconstructed PMT quantities")
            self.outTree_pmt = OutputTree(self.outputFile,self.outputTree_pmt)
            self.autotree_pmt = AutoFillTreeProducer(self.outTree_pmt,self.eventContentParams)

            ## Prepare PMT average waveform Tree (1 event = 1 averaged waveform using 4 PMTs)
            ## Only does average if there are more than one channel
            if len(self.options.board_pmt_channels) > 1:
                self.outputTree_pmt_avg = ROOT.TTree("PMT_Avg_Events","Tree containing the average PMT waveforms of 4 channels")
                self.outTree_pmt_avg = OutputTree(self.outputFile,self.outputTree_pmt_avg)
                self.autotree_pmt_avg = AutoFillTreeProducer(self.outTree_pmt_avg,self.eventContentParams)

            if options.include_gem:
                self.outputTree_gem = ROOT.TTree("GEM_Events","Tree containing reconstructed GEM quantities")
                self.outTree_gem = OutputTree(self.outputFile,self.outputTree_gem)
                self.autotree_gem = AutoFillTreeProducer(self.outTree_gem,self.eventContentParams)

        if self.options.camera_mode:
            self.outTree.branch("run", "I", title="run number")
            self.outTree.branch("event", "I", title="event number")
            self.outTree.branch("pedestal_run", "I", title="run number used for pedestal subtraction")
            self.autotree.createCameraVariables()
            self.autotree.createTimeCameraVariables()
            self.autotree.createClusterVariables('sc')
        if self.options.save_MC_data:
#           self.outTree.branch("MC_track_len","F")
            self.outTree.branch("eventnumber","I")
            self.outTree.branch("particle_type","I")
            self.outTree.branch("energy","F")
            self.outTree.branch("phi_initial","F")
            self.outTree.branch("theta_initial","F")
            self.outTree.branch("MC_x_vertex","F")
            self.outTree.branch("MC_y_vertex","F")
            self.outTree.branch("MC_z_vertex","F")
            self.outTree.branch("MC_x_vertex_end","F")
            self.outTree.branch("MC_y_vertex_end","F")
            self.outTree.branch("MC_z_vertex_end","F")
            self.outTree.branch("MC_3D_pathlength","F")
            self.outTree.branch("MC_2D_pathlength","F")
            
        if options.environment_variables: self.autotree.createEnvVariables()
        if self.options.pmt_mode:
            self.autotree_pmt.createPMTVariables(self.pmt_params)
            self.autotree_pmt.createTimePMTVariables()

            if len(self.options.board_pmt_channels) > 1:
                self.autotree_pmt_avg.createPMTVariables_average(self.pmt_params)            
                self.autotree_pmt_avg.createTimePMTVariables()

            if options.include_gem:
                self.autotree_gem.createPMTVariables(self.pmt_params)   ## We base GEM analysis on PMT, for now.


    def endJob(self):
        if options.camera_mode or options.environment_variables:
            self.outTree.write()
        
        if self.options.pmt_mode:
            self.outTree_pmt.write()
            if len(self.options.board_pmt_channels) > 1:
                self.outTree_pmt_avg.write()            
            if options.include_gem:
                self.outTree_gem.write()
        
        self.outputFile.Close()
        
    def getNEvents(self,options):
        if options.rawdata_tier == 'root':
            tf = sw.swift_read_root_file(self.tmpname)
            pics = [k for k in tf.keys() if 'pic' in k]
            return len(pics)
        elif options.rawdata_tier == 'h5':
            tf = sw.swift_read_h5_file(self.tmpname)
            pics = [k for k in tf.keys() if 'pic' in k]
            print("n events:", len(pics))
            return len(pics)
            
        run,tmpdir,tag = self.tmpname
        mf = sw.swift_download_midas_file(run,tmpdir,tag)     #you download the file here so that in multithread does not confuse if it downloaded or not
        if options.offline==False:
            df = cy.read_cygno_logbook(tag=options.tag,start_run=run-2000,end_run=run+1)
        else:
            runlog='runlog_%s_auto.csv' % (options.tag)
            df = pd.read_csv('pedestals/%s'%runlog)
        if df.run_number.isin({int(options.run)}).any():
           dffilter = df["run_number"] == int(options.run)
           try:
              evs = int(df.number_of_events[dffilter].values.tolist()[0])
              return evs
           except ValueError:
              print('Probably number of events line in data frame is empty. Opening and counting the file events\n')
     
        evs =0
        for mevent in mf:
            if mevent.header.is_midas_internal_event():
               continue
            else:
                keys = mevent.banks.keys()
            for iobj,key in enumerate(keys):
                name=key
                if name.startswith('CAM'):
                    evs += 1
        return evs

    def calcPedestal(self,options,alternativeRebin=-1):
        maxImages=options.maxEntries
        nx=self.xmax
        ny=self.ymax
        rebin = self.rebin if alternativeRebin<0 else alternativeRebin
        nx=int(nx/rebin); ny=int(ny/rebin); 
        pedfilename = 'pedestals/pedmap_run%s_rebin%d.root' % (options.pedrun,rebin)
        
        pedfile = ROOT.TFile.Open(pedfilename,'recreate')
        pedmap = ROOT.TH2D('pedmap','pedmap',nx,0,self.xmax,ny,0,self.ymax)
        pedmapS = ROOT.TH2D('pedmapsigma','pedmapsigma',nx,0,self.xmax,ny,0,self.ymax)

        pedsum = np.zeros((ny,nx))
        pedmean = np.zeros((ny,nx))
        pedrms = np.zeros((ny,nx))

        if not options.zeroped:
           if (options.rawdata_tier == 'root' or options.rawdata_tier == 'h5') and options.pedrun==1:
               tmpdir = '{tmpdir}'.format(tmpdir=options.tmpdir if options.tmpdir else "/tmp/")
               if not sw.checkfiletmp(int(options.pedrun),'root',tmpdir):
                   print ('Downloading file: ' + sw.swift_root_file(options.tag, int(options.pedrun)) + 'into ' + tmpdir)
                   pedfilename = sw.swift_download_root_file(sw.swift_root_file(options.tag, int(options.pedrun)),int(options.pedrun),tmpdir)
               else:
                   pedfilename = sw.swift_download_root_file(sw.swift_root_file(options.tag, int(options.pedrun)),int(options.pedrun),tmp=tmpdir,justName=True)                
               tf = sw.swift_read_root_file(pedfilename)
               keys = tf.keys()
               mf = [0] # dummy array to make a common loop with MIDAS case
           else:
               mf = sw.swift_download_midas_file(int(options.pedrun),'/tmp/',options.tag)
               #mf = self.tmpname

        if not options.zeroped:

           # first calculate the mean 
           numev = 0
           if  options.rawdata_tier == 'midas' or options.pedrun>1:
               mf.jump_to_start()
               for mevent in mf:
                   if mevent.header.is_midas_internal_event():
                       continue
                   else:
                       keys = mevent.banks.keys()
                   for iobj,key in enumerate(keys):
                       name=key
                       if name.startswith('CAM'):
                           justSkip=False
                           if (numev in self.options.excImages) and self.options.justPedestal: justSkip=True
                           if (maxImages>-1 and numev>min(len(keys),maxImages)) and self.options.justPedestal: break
                           if numev>250: break # no need to compute pedestals with >250 evts
                               
                           if numev%20 == 0:
                               print("Calc pedestal mean with event: ",numev)
                           numev += 1
                           if justSkip:
                               continue
                           arr,_,_ = cy.daq_cam2array(mevent.banks[key])
                           if rebin>1:
                               ctools.arrrebin(arr,rebin)
                           pedsum = np.add(pedsum,arr)
           else:
               #print ("keys = ",keys)
               for i,name in enumerate(keys):
                   if 'pic' in name:
                       patt = re.compile('\\S+run(\\d+)_ev(\\d+)')
                       m = patt.match(name)
                       run = int(m.group(1))
                       event = int(m.group(2))
                       justSkip=False
                       if (numev in self.options.excImages) and self.options.justPedestal: justSkip=True
                       if (maxImages>-1 and numev>min(len(keys),maxImages)) and self.options.justPedestal: break
                       if numev>250: break # no need to compute pedestals with >250 evts
    
                       if event%20 == 0:
                           print("Calc pedestal mean with event: ",event)
                       numev += 1
                       if justSkip:
                           continue
                       arr = utilities.rootflip(tf,name,options.tag)                    #necessary to uniform root raw data to midas. This is a vertical flip (raw data differ between ROOT and MIDAS formats)
                       pedsum = np.add(pedsum,arr)
    
           pedmean = pedsum / float(numev)
    
           # now compute the rms (two separate loops is faster than one, yes)
           pedsqdiff = np.zeros((ny,nx))
           numev = 0
           if  options.rawdata_tier == 'midas':
               mf.jump_to_start()
               for mevent in mf:
                   if mevent.header.is_midas_internal_event():
                       continue
                   else:
                       keys = mevent.banks.keys()
                   for iobj,key in enumerate(keys):
                       name=key
                       if name.startswith('CAM'):
                           justSkip=False
                           if (numev in self.options.excImages) and self.options.justPedestal: justSkip=True
                           if (maxImages>-1 and numev>min(len(keys),maxImages)) and self.options.justPedestal: break
                           if numev>250: break # no need to compute pedestals with >250 evts 
                
                           if numev%20 == 0:
                               print("Calc pedestal rms with event: ",numev)
                           numev += 1
                           if justSkip:
                                continue
    
                           arr,_,_ = cy.daq_cam2array(mevent.banks[key])
                           if rebin>1:
                               ctools.arrrebin(arr,rebin)
                           pedsqdiff = np.add(pedsqdiff, np.square(np.add(arr,-1*pedmean)))
           else:
               for i,name in enumerate(keys):
                   if 'pic' in name:
                       patt = re.compile('\\S+run(\\d+)_ev(\\d+)')
                       m = patt.match(name)
                       run = int(m.group(1))
                       event = int(m.group(2))
                       justSkip=False
                       if (numev in self.options.excImages) and self.options.justPedestal: justSkip=True
                       if (maxImages>-1 and numev>min(len(keys),maxImages)) and self.options.justPedestal: break
                       if numev>250: break # no need to compute pedestals with >250 evts
                       if justSkip:
                        continue
    
                       if event%20 == 0:
                           print("Calc pedestal rms with event: ",event)
                       numev += 1
                       arr = utilities.rootflip(tf,name,options.tag)                     #see cycle above on pedmean
                       pedsqdiff = np.add(pedsqdiff, np.square(np.add(arr,-1*pedmean)))
                   
           pedrms = np.sqrt(pedsqdiff/float(numev-1))

        else:
                print("===> PEDESTAL IS DUMMY: 0 MEAN AND 0 RMS (typically for simulated events")

        # now save in a persistent ROOT object
        # the inversion of x and y from array to histogram is correct: [row][columns] to x,y
        for iy in range(ny):
            for ix in range(nx):
                pedmap.SetBinContent(ix+1,iy+1,pedmean[iy,ix]);
                pedmap.SetBinError(ix+1,iy+1,pedrms[iy,ix]);
                pedmapS.SetBinContent(ix+1,iy+1,pedrms[iy,ix]);

        pedfile.cd()
        pedmap.Write()
        pedmapS.Write()
        if self.cg.cameratype == 'Quest':
            pedmean1D = ROOT.TH1D('pedmean','pedestal mean',500,195,220)
        else:    
            pedmean1D = ROOT.TH1D('pedmean','pedestal mean',500,97,103)
        pedrms1D = ROOT.TH1D('pedrms','pedestal RMS',1000,0,15)
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
        
        ctools = cameraTools(self.cg)
        print("Reconstructing event range: ",evrange[1],"-",evrange[2])
        self.outputFile.cd()

        flag_oxy=False              #Flag-check to see if oxygen variable exist
        env_varf = open('modules_config/env_variables.txt','r')
        env_var = eval(env_varf.read())
        if self.options.rawdata_tier == 'root':
            tf = sw.swift_read_root_file(self.tmpname)
            keys = tf.keys()
            mf = [0] # dummy array to make a common loop with MIDAS case
        elif self.options.rawdata_tier == 'h5':
            tf = sw.swift_read_h5_file(self.tmpname)
            keys = tf.keys()
            mf = [0] # dummy array to make a common loop with MIDAS case


        elif self.options.rawdata_tier == 'midas':
            run,tmpdir,tag = self.tmpname
            mf = sw.swift_download_midas_file(run,tmpdir,tag)
            
            ## Necessary to read the ODB to retrieve some info necessary for the waveform analysis
            ## Seems to repeat the opening process but *doesn't* slow down the code.
            if self.options.pmt_mode == 1:        
                
                odb,corrected,channels_offsets,camera_exposure = utilities.get_odb_pmt_info(mf,self.options,run)

            mf.jump_to_start()
            dslow = pd.DataFrame()
            if self.options.environment_variables:
        
                odb = cy.get_bor_odb(mf)
                
                header_environment = odb.data['Equipment']['Environment']['Settings']['Names Input']
                header_gas_system = odb.data['Equipment']['GasSystem']['Settings']['Names']
                header_oxygen = header_gas_system[265]
                if header_oxygen == env_var['oxygen']: 
                    flag_oxy=True
                else:
                    header_oxygen = env_var['oxygen']
                
                value_variables = odb.data['Equipment']['Environment']['Variables']
                value_gas_system = odb.data['Equipment']['GasSystem']['Variables']
                value_oxygen = value_gas_system['Demand'][265]
                if not flag_oxy:
                    value_oxygen=-99
                
                doxygen = pd.DataFrame([value_oxygen], columns=[f"{header_oxygen}"])
                dslow = pd.DataFrame(columns = header_environment)
                dslow.loc[len(dslow)] = value_variables['Input']
                dslow = pd.merge(dslow,doxygen,left_index=True,right_index=True)

                for i in dslow.keys():
                    dslow = utilities.conversion_env_variables(dslow, odb, env_var, i, j_env = 0)
                   
                try:
                   self.autotree.fillEnvVariables(dslow.take([0]))
                   if not self.options.camera_mode:
                            self.outTree.fill()
                except:
                   print("WARNING: could not fill dslow variables.")   
                #print(dslow)

                j_env = 1

        if self.options.save_MC_data:
            mc_variables = [
                "eventnumber", "particle_type", "energy", "phi", "theta",
                "x_vertex", "y_vertex", "z_vertex",
                "x_vertex_end", "y_vertex_end", "z_vertex_end",
                "proj_track_2D", "track_length_3D"
            ]
            mc_table = tf['event_info'].arrays(mc_variables)
            print(mc_table)
 
            branch_map = {
                "eventnumber":      "eventnumber",
                "particle_type":    "particle_type",
                "energy":           "energy",
                "phi_initial":      "phi",
                "theta_initial":    "theta",
                "MC_x_vertex":      "x_vertex",
                "MC_y_vertex":      "y_vertex",
                "MC_z_vertex":      "z_vertex",
                "MC_x_vertex_end":  "x_vertex_end",
                "MC_y_vertex_end":  "y_vertex_end",
                "MC_z_vertex_end":  "z_vertex_end",
                "MC_2D_pathlength": "proj_track_2D",
                "MC_3D_pathlength": "track_length_3D",
            }

        numev = 0
        event=0
        camera_read = False         #only useful for midas read 
        pmt_read = False            #only useful for midas read... FIX: is it fine to use only camera_read in the for of mevent but before keys loop? probably yes
        if self.options.pmt_mode == 0:
            pmt_read = True
        
        exist_pmt = False
        exist_cam = False
        fails_count = 0
        timestamp = -1
        timestamp0 = 0

        for mevent in mf:
            if self.options.rawdata_tier == 'midas':
                if mevent.header.is_midas_internal_event():
                    continue
                else:
                    keys = mevent.banks.keys()

            if camera_read and pmt_read:
                numev +=1   
            camera_read = False         #only useful for midas read 
            pmt_read = False            #only useful for midas read
            if self.options.pmt_mode == 0:
                pmt_read = True
            else:
                if exist_cam and not exist_pmt:
                    fails_count +=1
                    if fails_count==3:
                        print('\nCareful: you set the PMT analysis ON but no PMT bank was found. Are you sure PMT data is available for this run?\n ANALYSIS FAILED')
                        sys.exit()
                    else:
                         exist_pmt = False
                         exist_cam = False   
            
            if self.options.rawdata_tier == 'midas':
                timestamp=mevent.header.timestamp

            for iobj,key in enumerate(keys):
                name=key
                camera = False
                pmt = False
                #print(name)

                if self.options.rawdata_tier == 'root':
                    if 'pic' in name:
                        patt = re.compile('\\S+run(\\d+)_ev(\\d+)')
                        m = patt.match(name)
                        run = int(m.group(1))
                        event = int(m.group(2))
                        img_fr = utilities.rootflip(tf,key,self.options.tag)     #necessary to uniform root raw data to midas. This is a vertical flip (raw data differ between ROOT and MIDAS formats)
                        camera=True

                elif self.options.rawdata_tier == 'h5':
                    if 'pic' in name:
                        patt = re.compile('\\S+run(\\d+)_ev(\\d+)')
                        m = patt.match(name)
                        run = int(m.group(1))
                        event = int(m.group(2))
                        img_fr = utilities.rootflip(tf,key,self.options.tag)                   #structure for h5 copied from ROOT as it was in the past. Unsure if it is correct
                        camera=True

                elif self.options.rawdata_tier == 'midas':
                    run = int(self.options.run)

                    if name == 'TIME':
                        timestamp0 = mevent.banks[key].data[0]
                        #print(timestamp0,timestamp)
                    if timestamp<10000:             #10000 in seconds is 2h 46 min. This time should be larger than the run duration
                        timestamp = timestamp0*1000 + timestamp
                    
                    if name.startswith('CAM'):
                        camera_read = True
                        exist_cam = True
                        if options.camera_mode:
                            img_fr,_,_ = cy.daq_cam2array(mevent.banks[key])
                            camera=True

                    elif name.startswith('MSRD') and self.options.environment_variables: 
                        if mevent.header.event_id == 6:
                            dslow = utilities.read_env_variables(mevent.banks[key], name, dslow, odb, env_var, j_env=j_env)
                            if not flag_oxy:                            #check if not oxygen      
                                dslow.iloc[-1, -1]=-99
                            self.autotree.fillEnvVariables(dslow.take([j_env]))
                            j_env = j_env + 1
                            if not self.options.camera_mode:
                                if self.options.jobs != 1:
                                    if numev >= evrange[1]: 
                                        self.outTree.fill()
                                else:
                                    self.outTree.fill()
                    
                    elif name.startswith('INPT') and self.options.environment_variables:
                        if mevent.header.event_id == 5:
                            dslow = utilities.read_env_variables(mevent.banks[key], name, dslow, odb, env_var, j_env=j_env)
                            self.autotree.fillEnvVariables(dslow.take([j_env]))
                            j_env = j_env + 1
                            if not self.options.camera_mode:
                                if self.options.jobs != 1:
                                    if numev >= evrange[1]: 
                                        self.outTree.fill()
                                else:
                                    self.outTree.fill()
                    
                    elif name.startswith('DGH0'):
                        pmt_read = True
                        exist_pmt = True
                        fast_digitizer = False
                        slow_digitizer = False
                        if self.options.pmt_mode:
                            header=cy.daq_dgz_full2header(mevent.banks[key], verbose=False)
                            # sample_rate = header.sampling_rate

                            ## Care: if tag is MC$blabla, the tag for the digitizer will have to be changed to LNGS or something
                            waveform_f, waveform_s = cy.daq_dgz_full2array(mevent.banks['DIG0'], header, verbose=False, corrected=corrected, ch_offset=channels_offsets,tag=self.options.tag)

                            for idigi,digitizer in enumerate(header.boardNames):

                                if str(digitizer) == '1742' and len(waveform_f):  

                                    fast_digitizer = True
                                    nChannels_f  = header.nchannels[idigi]
                                    nTriggers_f = len(header.TTT[idigi])
                                    TTTs_f = header.TTT[idigi]

                                elif str(digitizer) == '1720' and len(waveform_s):
                                    
                                    slow_digitizer = True
                                    nChannels_s  = header.nchannels[idigi]
                                    nTriggers_s = len(header.TTT[idigi])
                                    TTTs_s = header.TTT[idigi]

                            pmt = True

                    event=numev

                justSkip = False
                if event<evrange[1]: justSkip=True
                if event>evrange[2]: return # avoids seeking up to EOF which with MIDAS is slow
                if event in self.options.excImages: justSkip=True
                if self.options.debug_mode == 1 and event != self.options.ev: justSkip=True
                if justSkip:
                    continue

                if self.options.camera_mode:
                    if camera==True:
                        print("Processing Run: ",run,"- Event ",event,"Camera...")
                        self.outTree.fillBranch("run",run)
                        self.outTree.fillBranch("event",event)
                        self.outTree.fillBranch("pedestal_run", int(self.options.pedrun))
                    
                        testspark=2*100*self.cg.npixx*self.cg.npixy+9000000		
                        if np.sum(img_fr)>testspark:
                            print("Run ",run,"- Event ",event," has spark, will not be analyzed!")
                            continue

                        if self.options.save_MC_data:
                            print(f'Analyzing event {event}')
                            if event >= len(mc_table['eventnumber']):
                                print(f"WARNING: camera image index {event} exceeds "
                                      f"MC table length ({len(mc_table['eventnumber'])}). "
                                      f"Skipping MC fill.")
                            else:
                                for branch_name, field_name in branch_map.items():
                                    self.outTree.fillBranch(branch_name, mc_table[event][field_name])
         
                        # Upper Threshold full image
                        img_cimax = np.where(img_fr < self.options.cimax, img_fr, 0)
                        
                        # zs on full image + saturation correction on full image or skip it
                        t_pre0 = time.perf_counter()
                        img_fr_sub = ctools.pedsub(img_cimax,self.pedarr_fr)
                        t_pre1 = time.perf_counter()
                        if self.options.saturation_corr:
                            #print("you are in saturation correction mode")
                            img_fr_satcor = ctools.satur_corr(img_fr_sub) 
                        else:
                            #print("you are in poor mode")
                            img_fr_satcor = img_fr_sub
                        t_pre2 = time.perf_counter()
                        img_fr_zs  = ctools.zsfullres(img_fr_satcor,self.noisearr_fr,nsigma=self.options.nsigma)
                        t_pre3 = time.perf_counter()
                        img_fr_zs_acc = ctools.acceptance(img_fr_zs,self.cg.ymin,self.cg.ymax,self.cg.xmin,self.cg.xmax)
                        t_pre4 = time.perf_counter()
                        img_rb_zs  = ctools.arrrebin(img_fr_zs_acc,self.rebin)
                        t_pre5 = time.perf_counter()
                            
                        t_pedsub = t_pre1 - t_pre0
                        t_saturation = t_pre2 - t_pre1
                        t_zerosup = t_pre3 - t_pre2
                        t_xycut = t_pre4 - t_pre3
                        t_rebin = t_pre5 - t_pre4
                            
                        # Cluster reconstruction on 2D picture
                        algo = 'DBSCAN'
                        if self.options.rawdata_tier == 'midas':
                            name = name + '_run' + str(run)+ '_' + str(event)
                        snprod_inputs = {'picture': img_rb_zs, 'pictureHD': img_fr_satcor, 'picturezsHD': img_fr_zs, 'pictureOri': img_fr, 'vignette': self.vignmap, 'name': name, 'algo': algo}
                        plotpy = self.options.jobs < 2 # for some reason on macOS this crashes in multicore
                        snprod_params = {'snake_qual': 3, 'plot2D': False, 'plotpy': False, 'plotprofiles': False}
                        t_DBSCAN_0 = time.perf_counter()
                        snprod = SnakesProducer(snprod_inputs,snprod_params,self.options,self.cg)
                        t_DBSCAN_1 = time.perf_counter()
                        snakes, t_DBSCAN, t_variables, lp_len, t_medianfilter, t_noisered = snprod.run()
                        t_DBSCAN_2 = time.perf_counter()
                        if self.options.debug_mode == 1:
                            print(f"1. DBSCAN run + variables calculation in {t_DBSCAN_2 - t_DBSCAN_1:0.4f} seconds")
                        self.autotree.fillCameraVariables(img_fr_zs,timestamp)
                        t_DBSCAN_3 = time.perf_counter()
                        if self.options.debug_mode == 1:
                            print(f"fillCameraVariables in {t_DBSCAN_3 - t_DBSCAN_2:0.4f} seconds")
                        self.autotree.fillClusterVariables(snakes,'sc')
                        t_DBSCAN_4 = time.perf_counter()
                        self.autotree.fillTimeCameraVariables(t_variables, t_DBSCAN, lp_len, t_pedsub, t_saturation, t_zerosup, t_xycut, t_rebin, t_medianfilter, t_noisered)
                        if self.options.debug_mode == 1:
                            print(f"fillClusterVariables in {t_DBSCAN_4 - t_DBSCAN_3:0.04f} seconds")
                            print()
                        del img_fr_sub,img_fr_satcor,img_fr_zs,img_fr_zs_acc,img_rb_zs
                        self.outTree.fill()
                        del img_fr
                        
         
                if self.options.pmt_mode:
                    if pmt == True:
                        print("Processing Run: ",run,"- Event ",event,"PMT...")
                        t00_wave =  time.perf_counter()
                        chs_to_analyse = len(self.options.board_pmt_channels)
                        fast_sampling = 1024
                        slow_sampling = 4000

                        ## Fast waveforms
                        if fast_digitizer:
                            if self.options.debug_mode == 1:
                                print("Number of fast triggers: {}".format(nTriggers_f))

                            for trg in range(nTriggers_f):    

                                insideGE = 0
                                # Uses the TTTs to check this condition
                                if (TTTs_f[trg] * 8.5/1000/1000) >= 180 and (TTTs_f[trg] * 8.5/1000/1000) <= (camera_exposure*1000):
                                    insideGE = 1

                                # Prepare the weighted average waveform 
                                sing_weig_avg_fast_wf = [ [0]*fast_sampling for _ in range(chs_to_analyse)]  
                                fast_wf_weights_snr = [0] * chs_to_analyse
                                weight_average_wf = [0]* fast_sampling

                                for ichf,chf in enumerate(self.options.board_pmt_channels):

                                    indx = trg * nChannels_f + chf
                                    waveform_info = { 'run' : run, 'event': event, 'channel' : chf, 'trigger' : trg , 'GE' : insideGE, 'sampling' : "fast", 'TTT' : (TTTs_f[trg]*8.5/1000./1000.)}

                                    t0_waveforms = time.perf_counter()

                                    fast_waveform = PMTreco(waveform_info, waveform_f[indx], self.pmt_params)
                                    fast_waveform.__repr__()

                                    t1_waveforms = time.perf_counter()
                                    t_waveforms = t1_waveforms - t0_waveforms

                                    self.autotree_pmt.fillPMTVariables(fast_waveform) 
                                    self.autotree_pmt.fillTimePMTVariables(t_waveforms)
                                    self.outTree_pmt.fill()

                                    # Weighted averaged waveform (weight = SNR)
                                    snr_ratio = fast_waveform.getSignalToNoise()
                                    fast_wf_weights_snr[ichf] = snr_ratio
                                    sing_weig_avg_fast_wf[ichf] = waveform_f[indx]

                                    # If one wants to visualize the new weighted waveforms,
                                    # meaning how much they actual weight for the final average, 
                                    # Ask David for the script changes

                                    if len(self.options.board_pmt_channels) > 1 and chf == self.options.board_pmt_channels[-1]:

                                        fast_wf_weights_snr = [ (x / max(fast_wf_weights_snr)) for x in fast_wf_weights_snr ]   # Normalization of the weights
                                        
                                        for k in range(chs_to_analyse):
                                        
                                            for j in range(fast_sampling):

                                                weight_average_wf[j] += sing_weig_avg_fast_wf[k][j]*fast_wf_weights_snr[k]/sum(fast_wf_weights_snr)

                                        waveform_info_fast_wei_avg = { 'run' : run, 'event': event, 'channel' : 9, 'trigger' : trg, 'GE' : 9, 'sampling' : "fast"}

                                        t0_waveforms = time.perf_counter()
                                        fast_waveform_wei_avg = PMTreco(waveform_info_fast_wei_avg, weight_average_wf, self.pmt_params)
                                        t1_waveforms = time.perf_counter()
                                        t_waveforms = t1_waveforms - t0_waveforms

                                        self.autotree_pmt_avg.fillPMTVariables_average(fast_waveform_wei_avg)
                                        self.autotree_pmt_avg.fillTimePMTVariables(t_waveforms)
                                        #fast_waveform_wei_avg.__repr__()           ## Verbose of averaged waveform
                                        self.outTree_pmt_avg.fill()

                                        del waveform_info_fast_wei_avg
                                        del fast_waveform_wei_avg

                                    del waveform_info
                                    del fast_waveform
                                
                                # GEM readout. Only available for fast digitizer
                                # No computing time properties for GEM for now.
                                if options.include_gem:
                                    for ichf_gem,chf_gem in enumerate(self.options.board_gem_channels):

                                        indx = trg * nChannels_f + chf_gem
                                        waveform_info = { 'run' : run, 'event': event, 'channel' : chf_gem, 'trigger' : trg , 'GE' : insideGE, 'sampling' : "fast", 'TTT' : (TTTs_f[trg]*8.5/1000./1000.)}

                                        fast_gem_waveform = PMTreco(waveform_info, waveform_f[indx], self.pmt_params)
                                        # fast_gem_waveform.__repr__()

                                        self.autotree_gem.fillPMTVariables(fast_gem_waveform) 
                                        self.outTree_gem.fill()

                                        del fast_gem_waveform

                            del waveform_f

                        # Slow waveforms
                        if slow_digitizer:
                            if self.options.debug_mode == 1:
                                print("Number of slow triggers: {}".format(nTriggers_s))

                            for trg in range(nTriggers_s):    

                                insideGE = 0
                                if (TTTs_s[trg] * 8.5/1000/1000) >= 180 and (TTTs_s[trg] * 8.5/1000/1000) <= (camera_exposure*1000):
                                    insideGE = 1

                                sing_weig_avg_slow_wf = [ [0]* slow_sampling for _ in range(chs_to_analyse)]  
                                slow_wf_weights_snr = [0] * chs_to_analyse
                                weight_average_wf = [0]* slow_sampling

                                for ichs,chs in enumerate(self.options.board_pmt_channels):

                                    indx = trg * nChannels_s + chs
                                    waveform_info = { 'run' : run, 'event': event, 'channel' : chs, 'trigger' : trg , 'GE' : insideGE , 'sampling' : "slow", 'TTT' : (TTTs_s[trg]*8.5/1000./1000.)}
                                    
                                    t0_waveforms = time.perf_counter()

                                    slow_waveform = PMTreco(waveform_info, waveform_s[indx], self.pmt_params)
                                    slow_waveform.__repr__()

                                    t1_waveforms = time.perf_counter()
                                    t_waveforms = t1_waveforms - t0_waveforms

                                    self.autotree_pmt.fillPMTVariables(slow_waveform) 
                                    self.autotree_pmt.fillTimePMTVariables(t_waveforms)
                                    self.outTree_pmt.fill()

                                    snr_ratio = slow_waveform.getSignalToNoise()
                                    slow_wf_weights_snr[ichs] = snr_ratio
                                    sing_weig_avg_slow_wf[ichs] = waveform_s[indx]

                                    if len(self.options.board_pmt_channels) > 1 and chs == self.options.board_pmt_channels[-1]:

                                        slow_wf_weights_snr = [ (x / max(slow_wf_weights_snr)) for x in slow_wf_weights_snr ]   # Normalization of the weights

                                        for k in range(chs_to_analyse):

                                            for j in range(slow_sampling):

                                                weight_average_wf[j] += sing_weig_avg_slow_wf[k][j]*slow_wf_weights_snr[k]/sum(slow_wf_weights_snr)

                                        waveform_info_slow_wei_avg = { 'run' : run, 'event': event, 'channel' : 9, 'trigger' : trg, 'GE' : 9, 'sampling' : "slow"}

                                        t0_waveforms = time.perf_counter()
                                        slow_waveform_wei_avg = PMTreco(waveform_info_slow_wei_avg, weight_average_wf, self.pmt_params)
                                        t1_waveforms = time.perf_counter()
                                        t_waveforms = t1_waveforms - t0_waveforms

                                        self.autotree_pmt_avg.fillPMTVariables_average(slow_waveform_wei_avg)
                                        self.autotree_pmt_avg.fillTimePMTVariables(t_waveforms)
                                        #slow_waveform_wei_avg.__repr__()           ## Verbose of averaged waveform
                                        self.outTree_pmt_avg.fill()

                                        del waveform_info_slow_wei_avg
                                        del slow_waveform_wei_avg

                                    del waveform_info
                                    del slow_waveform

                                # ... There is no slow board for GEM signals 
                                # for ichs_gem,chs_gem in enumerate(self.options.board_gem_channels):

                            del waveform_s

                        del header
                        
                        t01_wave =  time.perf_counter()
                        if self.options.debug_mode == 1:
                            print(f'PMT Reco Code Took: {t01_wave - t00_wave} seconds')
                        # END of `if pmt`
                # END of `if self.options.pmt_mode`
                
        gc.collect()
             
        ROOT.gErrorIgnoreLevel = savErrorLevel
                
if __name__ == '__main__':
    from optparse import OptionParser
    t0 = time.perf_counter()
    parser = OptionParser(usage='%prog h5file1,...,h5fileN [opts] ')
    parser.add_option('-r', '--run', dest='run', default='00000', type='string', help='run number with 5 characteres')
    parser.add_option('-j', '--jobs', dest='jobs', default=1, type='int', help='Jobs to be run in parallel (-1 uses all the cores available)')
    parser.add_option(      '--max-entries', dest='maxEntries', default=-1, type='int', help='Process only the first n entries')
    parser.add_option(      '--first-event', dest='firstEvent', default=-1, type='int', help='Skip all the events before this one')
    parser.add_option(      '--pdir', dest='plotDir', default='./', type='string', help='Directory where to put the plots')
    parser.add_option('-t',  '--tmp',  dest='tmpdir', default=None, type='string', help='Directory where to put the input file. If none is given, /tmp/<user> is used')
    parser.add_option(      '--max-hours', dest='maxHours', default=-1, type='float', help='Kill a subprocess if hanging for more than given number of hours.')
    parser.add_option('-o', '--outname', dest='outname', default='reco', type='string', help='prefix for the output file name')
    parser.add_option('-d', '--outdir', dest='outdir', default='.', type='string', help='Directory where to save the output file')
    parser.add_option(      '--git', dest='githash', default=None, type='string', help='git hash of the version of the reco code in use which you may want to give manually')
        
    (options, args) = parser.parse_args()
    
    f = open(args[0], "r")
    params = eval(f.read())

    for k,v in params.items():
        setattr(options,k,v)

    run = int(options.run)
    
    if options.debug_mode == 1:
        setattr(options,'outFile','%s_run%d_%s_debug.root' % (options.outname, run, options.tip))
        #if options.ev: options.maxEntries = options.ev + 1
        #if options.daq == 'midas': options.ev +=0.5 
    else:
        setattr(options,'outFile','%s_run%05d_%s.root' % (options.outname, run, options.tip))
    # FIX: if only pmt mode, don't need to compute pedestal
    if options.camera_mode:
        utilities.setPedestalRun(options)        
        
    try:
        USER = os.environ['USER']
    except:
        try:
          USER = os.environ['JUPYTERHUB_USER']
        except:
          USER = "autoreco"
    #tmpdir = '/mnt/ssdcache/' if os.path.exists('/mnt/ssdcache/') else '/tmp/'
    # it seems that ssdcache it is only mounted on cygno-login, not in the batch queues (neither in cygno-custom)
    tmpdir = '/tmp'
    os.system('mkdir -p {tmpdir}/{user}'.format(tmpdir=tmpdir,user=USER))
    tmpdir = '{tmpdir}/{user}/'.format(tmpdir=tmpdir,user=USER) if not options.tmpdir else options.tmpdir+"/"
    if sw.checkfiletmp(int(options.run),options.rawdata_tier,tmpdir):
        if options.rawdata_tier=='root':
            prefix = 'histograms_Run'
            postfix = 'root'
        elif options.rawdata_tier=='h5':
            prefix = 'histograms_Run'
            postfix = 'h5'
        else:
            prefix = 'run'
            postfix = 'mid.gz'
        options.tmpname = "%s/%s%05d.%s" % (tmpdir,prefix,int(options.run),postfix)
    
    else:
        if options.rawdata_tier == 'root':
            file_url = sw.swift_root_file(options.tag, int(options.run))
            print ('Downloading file: ' + file_url)
            options.tmpname = sw.swift_download_root_file(file_url,int(options.run),tmpdir)
        else:
            print ('Downloading MIDAS.gz file for run ' + options.run)
    # in case of MIDAS, download function checks the existence and in case it is absent, downloads it. If present, opens it
    if options.rawdata_tier == 'midas':
        ## need to open it (and create the midas object) in the function, otherwise the async run when multithreaded will confuse events in the two threads
        options.tmpname = [int(options.run),tmpdir,options.tag]		#This line needs to be corrected if MC data will be in midas format. Not foreseen at all
    if options.justPedestal:
        ana = analysis(options)
        print("Pedestals done. Exiting.")
        if options.donotremove == False:
            sw.swift_rm_root_file(options.tmpname)
        sys.exit(0)

    ana = analysis(options)
    nev = ana.getNEvents(options)
    print("\nThis run has ",nev," events.")
    if options.debug_mode == 1: print('DEBUG mode activated. Only event',options.ev,'will be analysed')
    # FIX: the option for saving plots should only be ON only if camera mode is ON
    print("I Will save plots to ",options.plotDir)
    #os.system('cp utils/index.php {od}'.format(od=options.plotDir))
    os.system('mkdir -p {pdir}'.format(pdir=options.plotDir))
    
    nThreads = 1
    if options.jobs==-1:
        import multiprocessing
        nThreads = min(multiprocessing.cpu_count(),nev)
    else:
        nThreads = min(options.jobs,nev)

    t1 = time.perf_counter()
    firstEvent = 0 if options.firstEvent<0 else options.firstEvent
    lastEvent = nev if options.maxEntries==-1 else min(nev,firstEvent+options.maxEntries)
    if options.debug_mode == 1: lastEvent = min(nev,int(options.ev))
    
    print ("Analyzing from event %d to event %d" %(firstEvent,lastEvent))
    base = options.outFile.split('.')[0]
    if nThreads>1:
        print ("RUNNING USING ",nThreads," THREADS.")
        nj = int(nev/nThreads) if options.maxEntries==-1 else max(int((lastEvent-firstEvent)/nThreads),1)
        chunks = [(ichunk,i,min(i+nj-1,nev)) for ichunk,i in enumerate(range(firstEvent,lastEvent,nj))]
        if len(chunks)>nThreads:
            chunks[-2] = (chunks[-2][0],chunks[-2][1],chunks[-1][2])
            del chunks[-1]
        print("Chunks = ",chunks)
        with futures.ProcessPoolExecutor(nThreads) as executor:
            futures_list = [executor.submit(ana,c) for c in chunks]
            for future in futures.as_completed(futures_list):
                # retrieve the result. This is crucial, because result() does not exit until the process is completed.
                future.result()
        print("Now hadding the chunks...")
        os.system('hadd -k -f {outdir}/{base}.root {outdir}/{base}_chunk*.root'.format(base=base, outdir=options.outdir))
        os.system('rm {outdir}/{base}_chunk*.root'.format(base=base, outdir=options.outdir))
    else:
        evrange=(-1,firstEvent,lastEvent)
        ana(evrange)
    t2 = time.perf_counter()
    if options.debug_mode == 1:
        print(f'Reconstruction Code Took: {t2 - t1} seconds')

    # now add extra information
    tf = ROOT.TFile.Open("{outdir}/{base}.root".format(base=base, outdir=options.outdir),'update')
    # now add parameters of the reconstruction
    utilities.Param_storage(tf,base,args[0],options)
    # now add the git commit hash to track the version in the ROOT file
    if options.githash != None:
       githash=ROOT.TNamed("gitHash",options.githash)
       githash.Write()       
    else:
       try:
          githash = ROOT.TNamed("gitHash",str(utilities.get_git_revision_hash()).replace("\\n'","").replace("b'",""))
          githash.Write()
       except:
          print('No githash provided nor githash found (no .git folder?)') 
    # now add the time of reconstruction
    total_time = ROOT.TNamed("total_time", str(t2-t1))
    total_time.Write()
    tf.Close()
    
    if options.donotremove == False:
        if options.rawdata_tier=='midas':
            prefix = 'run'
            postfix = 'mid.gz'
            options.tmpname = "%s/%s%05d.%s" % (tmpdir,prefix,int(options.run),postfix)
        sw.swift_rm_root_file(options.tmpname)
    
    t3 = time.perf_counter()
    if options.debug_mode == 1:
           print(f'Total time the Code Took: {t3 - t0} seconds')
