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

c_driftcomp = libCRISP.driftcomp_2d
c_driftcomp.restype = c_float
c_driftcomp.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def cstr(string):
  return c_char_p(string.encode('ascii'))

def driftcomp(out, tid, job, dims, params, indir):
  ncy, nz = dims
  reg, pos = job
  ch = params['reference_channel']
  
  indir = cstr(indir)
  inpattern = params['inpattern'].format(region=reg, position=pos, channel=ch)
  inpattern = cstr(f"cyc%03d_reg{reg:03d}/{inpattern}")
  
  mode = 1 # mode&1: normalize image stack 5 to loosen stitch restrictions
  
  status = c_driftcomp(indir, inpattern, reg, pos, ncy, nz, 0, tid, mode, params['a1'], params['a2'], params['a3'], params['a4'], params['h1'], params['h2'], params['h3'], params['h4'], params['h5'], 0, 0, 0)
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, indir, dims, params, jobs = args

  sleep(tid * 15)
  
  process = mp.current_process()
  out.put('{}> pid {:5d} got job list {}'.format(tid, process.pid, jobs), False)
  
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job), False)
      
      status = driftcomp(out, tid, job, dims, params, indir)
      
      if status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job), False)
    if status: break
  
  t1 = timer()
  free_libs([libCRISP])
  out.put(f'{tid}> joblist complete, elapsed {t1-t0:.0f}s', False)

def dispatch_jobs(indir, joblist, dims, params, max_threads=1):
  tstart = timer()
  manager = mp.Manager()
  
  nc = 0
  nj = len(joblist)
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to 2d driftcomp {} image stacks'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  
  q = manager.Queue()
  
  with mp.Pool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, indir, dims, params, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 

    remainingtime0 = None
    while rs._number_left > 0 or not q.empty():
      try:
        msg = q.get(True, 120)
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

def main(indir, params=None, max_threads=2):
  print("Processing '{}'".format(indir))
  
  if not os.path.exists(os.path.join(indir, 'driftcomp')): os.mkdir(os.path.join(indir, 'driftcomp'))
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))

  w = config['dimensions']['width']
  h = config['dimensions']['height']
  z = config['padding']['zout']

  regions   = {reg+1 for reg in range(config['dimensions']['regions'])}
  positions = {pos+1 for pos in range(config['dimensions']['gx']*config['dimensions']['gy'])}
  cycles    = {cyc+1 for cyc in range(config['dimensions']['cycles'])}
  channels  = set(config['microscope']['wavelengthEM'].keys())
  reference_channel = config['setup'].get('reference_channel', 1)
  inpattern  = '{region}_{position:05d}_Z%03d_CH{channel:d}.tif'
  
  if config.get('extended_depth_of_field') and config['extended_depth_of_field'].get('enabled', True):
    if config['extended_depth_of_field'].get('register_edf', False) or not config['extended_depth_of_field'].get('save_zstack', True):
      z = 1
      inpattern = '{region}_{position:05d}_EDF_CH{channel:d}.tif'
  
  for r in regions:
    # preallocate output files to prevent potential race conditions
    offsetfile = os.path.join(indir, 'driftcomp', 'region{:02d}_offsets.bin'.format(r))
    if not os.path.isfile(offsetfile) or os.path.getsize(offsetfile) != max(positions)*max(cycles)*4*4:
      np.full(max(positions)*max(cycles)*4, np.nan, dtype=np.float32).tofile(offsetfile)
    
    zscorefile = os.path.join(indir, 'driftcomp', 'region{:02d}_zscores.bin'.format(r))
    if not os.path.isfile(zscorefile) or os.path.getsize(zscorefile) != max(positions)*max(cycles)*z*4:
      np.full(max(positions)*max(cycles)*z, np.nan, dtype=np.float32).tofile(zscorefile)
  
  jobs = list(itertools.product(regions, positions))
  if params:
    p = params
    params = {'a1': p[0], 'a2': p[1], 'a3': p[2], 'a4': p[3], 'h1': p[4], 'h2': p[5], 'h3': p[6], 'h4': p[7], 'h5': p[8]}
  else:
    params = {'a1': 0.25, 'a2': 1.0, 'a3': 0.5, 'a4': 1.0, 'h1': 6.0, 'h2': 0.7, 'h3': 0.0, 'h4': 0.35, 'h5': 1.0}
  
  params.update({'inpattern': inpattern, 'reference_channel': reference_channel})
  
  score = dispatch_jobs(indir, jobs, (len(cycles), z), params, max_threads)
  
  print("Completed processing '{}'".format(indir))
  return score

if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()
  
  dirs = []
  dirs.append('X:/deconvolved/20211224_cartilage_final_2_offsetplus015')
  
  dirs.extend(sys.argv[1:])
  
  t0 = timer()
  for indir in dirs:
    main(indir, [0.25, 1.0, 0.5, 1.0, 6, 0.9, 1.0, 0.0, 3]) # v1
  
  free_libs([libCRISP])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  



  
