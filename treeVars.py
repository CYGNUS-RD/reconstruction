import numpy as np

class AutoFillTreeProducer:
    def __init__(self,tree):
        self.outTree = tree

    def createPMTVariables(self):
        self.outTree.branch('pmt_integral', 'F')
        self.outTree.branch('pmt_tot', 'F')
        self.outTree.branch('pmt_amplitude', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_time', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_prominence', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_fwhm', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_hm', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_risetime', 'F', lenVar='nPeak')
        self.outTree.branch('pmt_falltime', 'F', lenVar='nPeak')

    def fillPMTVariables(self,peakFinder,sampleSize):
        self.outTree.fillBranch('pmt_integral',peakFinder.getIntegral()*sampleSize)        
        self.outTree.fillBranch('pmt_tot',peakFinder.getTot())
        self.outTree.fillBranch('pmt_amplitude',peakFinder.getAmplitudes())
        self.outTree.fillBranch('pmt_time',peakFinder.getPeakTimes())
        self.outTree.fillBranch('pmt_prominence',peakFinder.getProminences())
        self.outTree.fillBranch('pmt_fwhm',peakFinder.getFWHMs())
        self.outTree.fillBranch('pmt_hm',peakFinder.getHMs())
        self.outTree.fillBranch('pmt_risetime',peakFinder.getTimes('rise'))
        self.outTree.fillBranch('pmt_falltime',peakFinder.getTimes('fall'))

    def createCameraVariables(self):
        self.outTree.branch('cmos_integral', 'F')
        self.outTree.branch('cmos_mean', 'F')
        self.outTree.branch('cmos_rms', 'F')

    def createClusterVariables(self,name='track'):
        chars = list(name)
        start = chars[0]; rest = chars[1:]
        sizeStr = 'n'+start.upper()+''.join(rest)
        self.outTree.branch('{name}_size'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_nhits'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_integral'.format(name=name),     'F', lenVar=sizeStr)
        # filled only for the supercluster
        if name=='sc':
            self.outTree.branch('{name}_energy'.format(name=name), 'F', lenVar=sizeStr)
            self.outTree.branch('{name}_pathlength'.format(name=name),    'F', lenVar=sizeStr)
        self.outTree.branch('{name}_length'.format(name=name),       'F', lenVar=sizeStr)
        self.outTree.branch('{name}_width'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_longrms'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_latrms'.format(name=name),       'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lfullrms'.format(name=name),     'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tfullrms'.format(name=name),     'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lp0amplitude'.format(name=name), 'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lp0prominence'.format(name=name),'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lp0fwhm'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lp0mean'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tp0fwhm'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_iteration'.format(name=name),    'F', lenVar=sizeStr)
        self.outTree.branch('{name}_xmean'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_ymean'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_xmax'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_xmin'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_ymax'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_ymin'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_nclu'.format(name=name),         'F', lenVar=sizeStr)
        self.outTree.branch('{name}_pearson'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tgaussamp'.format(name=name),    'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tgaussmean'.format(name=name),   'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tgausssigma'.format(name=name),  'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tchi2'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_tstatus'.format(name=name),      'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lgaussamp'.format(name=name),    'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lgaussmean'.format(name=name),   'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lgausssigma'.format(name=name),  'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lchi2'.format(name=name),        'F', lenVar=sizeStr)
        self.outTree.branch('{name}_lstatus'.format(name=name),      'F', lenVar=sizeStr)

    def fillCameraVariables(self,pic):
        self.outTree.fillBranch('cmos_integral',np.sum(pic))
        self.outTree.fillBranch('cmos_mean',np.mean(pic))
        self.outTree.fillBranch('cmos_rms',np.std(pic))

    def fillClusterVariables(self,clusters,name='track'):
        chars = list(name)
        start = chars[0]; rest = chars[1:]
        sizeStr = 'n'+start.upper()+''.join(rest)
        self.outTree.fillBranch('{name}_size'.format(name=name),     [cl.size() for cl in clusters])
        self.outTree.fillBranch('{name}_nhits'.format(name=name),    [cl.sizeActive() for cl in clusters])
        self.outTree.fillBranch('{name}_integral'.format(name=name), [cl.integral() for cl in clusters])
        # filled only for the supercluster
        if name=='sc':
            self.outTree.fillBranch('{name}_energy'.format(name=name), [cl.calibratedEnergy for cl in clusters])
            self.outTree.fillBranch('{name}_pathlength'.format(name=name),    [cl.pathlength for cl in clusters])
        self.outTree.fillBranch('{name}_length'.format(name=name),   [cl.shapes['long_width'] for cl in clusters])
        self.outTree.fillBranch('{name}_width'.format(name=name),    [cl.shapes['lat_width'] for cl in clusters])
        self.outTree.fillBranch('{name}_longrms'.format(name=name),  [cl.shapes['longrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_latrms'.format(name=name),   [cl.shapes['latrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_lfullrms'.format(name=name), [cl.shapes['long_fullrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_tfullrms'.format(name=name), [cl.shapes['lat_fullrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0amplitude'.format(name=name), [cl.shapes['long_p0amplitude'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0prominence'.format(name=name), [cl.shapes['long_p0prominence'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0fwhm'.format(name=name),   [cl.shapes['long_p0fwhm'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0mean'.format(name=name),   [cl.shapes['long_p0mean'] for cl in clusters])
        self.outTree.fillBranch('{name}_tp0fwhm'.format(name=name),   [cl.shapes['lat_p0fwhm'] for cl in clusters])
        self.outTree.fillBranch('{name}_iteration'.format(name=name), [cl.iterations() for cl in clusters])
        self.outTree.fillBranch('{name}_xmean'.format(name=name),     [cl.shapes['xmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymean'.format(name=name),     [cl.shapes['ymean'] for cl in clusters])
        self.outTree.fillBranch('{name}_xmax'.format(name=name),      [cl.shapes['xmax'] for cl in clusters])
        self.outTree.fillBranch('{name}_xmin'.format(name=name),      [cl.shapes['xmin'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymax'.format(name=name),      [cl.shapes['ymax'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymin'.format(name=name),      [cl.shapes['ymin'] for cl in clusters])
        self.outTree.fillBranch('{name}_nclu'.format(name=name),      [cl.getNclu() for cl in clusters])
        self.outTree.fillBranch('{name}_pearson'.format(name=name),   [cl.getPearson() for cl in clusters])
        self.outTree.fillBranch('{name}_tgaussamp'.format(name=name),   [cl.shapes['tgaussamp'] for cl in clusters])
        self.outTree.fillBranch('{name}_tgaussmean'.format(name=name),  [cl.shapes['tgaussmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_tgausssigma'.format(name=name), [cl.shapes['tgausssigma'] for cl in clusters])
        self.outTree.fillBranch('{name}_tchi2'.format(name=name),       [cl.shapes['tchi2'] for cl in clusters]) 
        self.outTree.fillBranch('{name}_tstatus'.format(name=name),     [cl.shapes['tstatus'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgaussamp'.format(name=name),   [cl.shapes['lgaussamp'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgaussmean'.format(name=name),  [cl.shapes['lgaussmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgausssigma'.format(name=name), [cl.shapes['lgausssigma'] for cl in clusters])
        self.outTree.fillBranch('{name}_lchi2'.format(name=name),       [cl.shapes['lchi2'] for cl in clusters]) 
        self.outTree.fillBranch('{name}_lstatus'.format(name=name),     [cl.shapes['lstatus'] for cl in clusters])



        
