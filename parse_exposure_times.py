import os
import glob
import xmltodict
import numpy as np

def parse_exposure_times(indir):
  if len(glob.glob(os.path.join(indir,'*.xml')))==1:
    xmlfiles = glob.glob(os.path.join(indir,'*.xml'))
    print("Reading exposure times from XML file...")
    for xmlfile in xmlfiles:
      with open(xmlfile) as fd:
        doc = xmltodict.parse(fd.read())

        channelColors = doc['Template']['Channels']['string']
        print(channelColors)
        print(channelColors[1])

        cycleInfo = doc['Template']['Exposures']['ExposureItem']
        print(len(cycleInfo))
        cycleInfo = [entry for entry in cycleInfo if entry['Active'] == 'true' and int(entry['Cycle']) > 0]
        ncycles = len(cycleInfo)
        
        nchannels = len(cycleInfo[0]['WaveLength']['decimal'])
        print(nchannels)
        
        exposureTimes = np.zeros((ncycles, nchannels), dtype=np.float32)
        for cy,info in enumerate(cycleInfo):
          for ch,color in enumerate(channelColors):
            duration = float(info['ExposuresTime']['decimal'][ch])
            exposureTimes[cy,ch] = duration
        
        print(exposureTimes)
        
        wavelengthEX = np.zeros(nchannels, dtype=np.float32)
        for ch,color in enumerate(channelColors):
          wavelengthEX[ch] = float(info['WaveLength']['decimal'][ch])
        
        print(wavelengthEX)
        
        exposurefile = os.path.join(indir, 'exposure_times.bin')
        with open(exposurefile, 'wb') as file:
          file.write(exposureTimes.tobytes())
          file.write(wavelengthEX.tobytes())
  
  else:
    exposure_times_txt = os.path.join(indir,'exposure_times.txt')
    print("Reading exposure times from txt file...")
    
    exposure_times = np.genfromtxt(exposure_times_txt, skip_header=1, delimiter=',')
    if np.any(np.isnan(exposure_times)):
      exposure_times = np.genfromtxt(exposure_times_txt, skip_header=1, delimiter='\t')
    
    print(exposure_times)
    
    if np.any(np.isnan(exposure_times)):
      print("Error: file '{}' has unexpected format".format(exposure_times_txt))
    else:
      exposure_times = exposure_times[:,1:].astype(np.float32)
      wavelength_ex = np.array([358, 488, 550, 650], dtype=np.float32)
      
      nc = exposure_times.shape[1]
      if len(wavelength_ex) > nc: wavelength_ex = wavelength_ex[0:nc]

      print(wavelength_ex)
      
      exposurefile = os.path.join(indir, 'exposure_times.bin')
      
      with open(exposurefile, 'wb') as file:
        file.write(exposure_times.tobytes())
        file.write(wavelength_ex.tobytes())


if __name__ == '__main__':
  indir = 'X:/temp/CODEX raw/20201008_test'
  parse_exposure_times(indir)



  
