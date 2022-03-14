import multiprocessing as mp

from ctypes import *
from _ctypes import FreeLibrary

import os
import sys
import glob
import re
import toml
import numpy as np
import itertools
from timeit import default_timer as timer
from time import sleep
import humanfriendly
from PyCaffeinate import PyCaffeinate

os.environ['VIPS_WARNING'] = '0'

if os.name=='nt':
  libc = cdll.msvcrt
  CRISP_path = os.path.join(os.getcwd(),'CRISP.dll')
  if os.path.isfile(CRISP_path):
    if hasattr(os, 'add_dll_directory'):
      for p in os.getenv('PATH').split(';'):
        if p not in ['','.'] and os.path.isdir(p): os.add_dll_directory(p)
    libCRISP = CDLL(CRISP_path)
  else: print('Unable to find CRISP.dll')
else:
  libc = cdll.LoadLibrary(_ctypes.util.find_library('c'))
  libCRISP = CDLL('CRISP.dylib')

c_driftcomp_3d = libCRISP.driftcomp_3d_tiled_overlap_focus
c_driftcomp_3d.restype = c_float
c_driftcomp_3d.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_char_p, c_char_p, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]


def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib
    
def cstr(string):
  return c_char_p(string.encode('ascii'))

def driftcomp(out, tid, job, dims, params, indir, dark, flat):
  ncy, nz = dims
  reg, pos = job
  ch = params['reference_channel']
  
  indir = cstr(indir)
  inpattern = params['inpattern'].format(region=reg, position=pos, channel=ch)
  inpattern = cstr(f"cyc%03d_reg{reg:03d}/{inpattern}")
  
  mode = 1 # 0 or 1
  
  status = c_driftcomp_3d(indir, inpattern, reg, pos, ncy, nz, dark, flat, tid, mode, params['a1'], params['a2'], params['a3'], params['a4'], params['h1'], params['h2'], params['h3'], params['h4'], params['h5'], params['p1'], params['p2'])
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, indir, dims, params, dark, flat, jobs = args
  
  if dark: dark = cstr(dark)
  if flat: flat = cstr(flat)
  
  process = mp.current_process()
  out.put('{}> pid {:5d} got job list {}'.format(tid, process.pid, jobs), False)
  
  sleep(tid * 20)
  
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job), False)
      
      status = driftcomp(out, tid, job, dims, params, indir, dark, flat)
      
      if status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job), False)
    if status: break
  
  t1 = timer()
  free_libs([libCRISP])
  out.put(f'{tid}> joblist complete, elapsed {t1-t0:.0f}s', False)

def dispatch_jobs(indir, joblist, dims, params, dark, flat, max_threads=1):
  tstart = timer()
  manager = mp.Manager()

  nc = 0 # completed jobs
  nj = len(joblist)
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to 3d driftcomp {} image stacks'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  
  q = manager.Queue()
  
  with mp.Pool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, indir, dims, params, dark, flat, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 

    nc = 0
    remainingtime0 = None
    while rs._number_left > 0 or not q.empty():
      try:
        msg = q.get(True, 600)
        if isinstance(msg, str):
          print(msg)
        else:
          if msg[0] == 0: # success
            completed_jobs.append(msg[2])
            nc = len(completed_jobs)
            remainingtime1 = (timer() - tstart)/nc * (nj-nc)
            remainingtime = np.mean([remainingtime0, remainingtime1]) if remainingtime0 else remainingtime1
            remainingtime0 = remainingtime1
            timestring = '' if nc==nj else ' ({} remaining)'.format(humanfriendly.format_timespan(remainingtime, max_units=2))
            print('## progress: {} of {} [{:.1f}%]{}'.format(nc, nj, 100*nc/nj, timestring))
          else:
            failed_jobs.append(msg[2])
            print('%%%% Job {} from worker {} failed! %%%%'.format(msg[2], msg[1]))
          
      except mp.queues.Empty:
        print('Message queue is empty - is the program stalled?')
        break
        
    nc = len(completed_jobs)
    if(rs._number_left == 0):
      if(nc == nj): print('Finished - processed {} tiles'.format(nc))
      else: print('Incomplete - processed {} of {} tiles'.format(nc, nj))
    else:
      print('Queue timeout - {} workers stuck'.format(rs._number_left))
      if(nc == nj): print('Processed {} tiles'.format(nc))
      else: print('Processed {} of {} tiles'.format(nc, nj))
    
    if(nc != nj):
      print('Failed jobs: {}'.format(nj-nc))
      for job in joblist:
        if not job in completed_jobs:
          print('  region: {}, position: {:02d}'.format(*job))
      print()
    
    p.close()
    p.join()
    

def get_folders(root):
  pattern1 = re.compile('cyc\d+_reg\d+.*')
  pattern2 = re.compile('(cyc\d+_reg\d+)')
  
  directories = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
  cycle_region_dirs = [d for d in directories if pattern1.match(d)]
  renamed = [pattern2.match(d).group(0) for d in cycle_region_dirs]
  
  for d1,d2 in zip(cycle_region_dirs, renamed):
    if d1 != d2 and not os.path.exists(os.path.join(root,d2)):
      print("  renamed input directory '{}' -> '{}'".format(d1, d2))
      os.rename(os.path.join(root, d1), os.path.join(root, d2))
  
  return renamed

def check_files(indir):
  print("Processing '{}'".format(indir))
  folders_in = get_folders(indir)

  if not os.path.exists(os.path.join(indir, 'driftcomp')): os.mkdir(os.path.join(indir, 'driftcomp'))
  
  d_in = {}
  pattern1 = re.compile('cyc(\d+)_reg(\d+)')
  for f in folders_in:
    m = pattern1.match(f)
    cyc = int(m.groups()[0])
    reg = int(m.groups()[1])
    key = 'c{}_r{}'.format(cyc,reg)
    d_in[key] = os.path.join(indir, f)
  
  return d_in

def main(indir=None, max_threads=2):
  if not indir or not os.path.isdir(indir):
    print("Error: '{}' is not a valid directory".format(indir))
    return
  
  d_in = check_files(indir)
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))

  w = config['dimensions']['width']
  h = config['dimensions']['height']
  z = config['dimensions']['slices']

  regions   = {reg+1 for reg in range(config['dimensions']['regions'])}
  positions = {pos+1 for pos in range(config['dimensions']['gx']*config['dimensions']['gy'])}
  cycles    = {cyc+1 for cyc in range(config['dimensions']['cycles'])}
  channels  = set(config['microscope']['wavelengthEM'].keys())
  reference_channel = config['microscope'].get('reference_channel', 1)
  inpattern  = config['setup'].get('inpattern', '{region}_{position:05d}_Z%03d_CH{channel:d}.tif')
  
  if config['correction']['correct_darkfield']:
    dark = config['correction']['darkfield_images']
    if isinstance(dark, list): dark = dark[reference_channel-1]
    dark = os.path.join(indir, dark)
  else:
    dark = None
  
  if config['correction']['correct_flatfield']:
    flat = config['correction']['flatfield_images']
    if isinstance(flat, list): flat = flat[reference_channel-1]
    flat = os.path.join(indir, flat)
  else:
    flat = None
  
  if z<4:
    print('Image stack size is too small, aborting!')
    return
  
  for r in regions:
    # allocate binary output files to prevent potential race conditions
    offsetfile = os.path.join(indir, 'driftcomp', 'region{:02d}_offsets.bin'.format(r))
    if not os.path.isfile(offsetfile) or os.path.getsize(offsetfile) < max(positions)*max(cycles)*4*4:
      np.full(max(positions)*max(cycles)*4, np.nan, dtype=np.float32).tofile(offsetfile)
    
    tx = (w + 256//4 + 256-1) // 256
    ty = (h + 256//4 + 256-1) // 256
    ztilefile = os.path.join(indir, 'driftcomp', 'region{:02d}_ztiles.bin'.format(r))
    if not os.path.isfile(ztilefile) or os.path.getsize(ztilefile) < max(positions)*ty*tx*max(cycles)*z*4:
      np.full(max(positions)*ty*tx*max(cycles)*z, np.nan, dtype=np.float32).tofile(ztilefile)
  
  jobs = list(itertools.product(regions, positions))
  params = {'inpattern': inpattern, 'reference_channel': reference_channel, 'a1': 0.5, 'a2': 0.5, 'a3': 0.5, 'a4': 0.5, 'h1': 8.0, 'h2': 0.75, 'h3': 0.3, 'h4': 0.3, 'h5': 0.9, 'p1': 1.5, 'p2': 2.0}
  
  dispatch_jobs(indir, jobs, (len(cycles), z), params, dark, flat, max_threads)
  print("Completed processing '{}'".format(indir))
  
if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()

  dirs = []
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20191211_run20_long_preveh')
  dirs.extend(sys.argv[1:])
  
  t0 = timer()
  for indir in dirs:
    main(indir)
  free_libs([libCRISP])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  


  
