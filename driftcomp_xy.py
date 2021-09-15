import multiprocessing as mp

from ctypes import *
from _ctypes import FreeLibrary

import os
import glob
import re
import itertools
import numpy as np
from shutil import copyfile
from timeit import default_timer as timer
from time import sleep
import humanfriendly
from PyCaffeinate import PyCaffeinate

if os.name=='nt':
  libc = cdll.msvcrt
  CRISP_path = os.path.join(os.getcwd(),'CRISP.dll')
  if os.path.isfile(CRISP_path):
    for p in os.getenv('Path').split(';'):
      if p in ['','.']:
        continue
      elif os.path.isdir(p): os.add_dll_directory(p)
    libCRISP = CDLL(CRISP_path)
  else: print('Unable to find CRISP.dll')
else:
  libc = cdll.LoadLibrary(_ctypes.util.find_library('c'))
  libCRISP = CDLL('CRISP.dylib')

c_driftcomp = libCRISP.driftcomp_2d
c_driftcomp.restype = c_float
c_driftcomp.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def driftcomp(out, tid, job, dims, params, indir):
  ncy, nz = dims
  reg, pos = job
  
  c_indir = c_char_p(indir.encode('ascii'))
  
  status = c_driftcomp(c_indir, reg, pos, ncy, nz, 0, tid, 1, params['a1'], params['a2'], params['a3'], params['a4'], params['h1'], params['h2'], params['h3'], params['h4'], params['h5'])
  
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
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.0f}s'.format(tid, t1-t0), False)

def dispatch_jobs(indir, joblist, dims, params, max_threads=1):
  tstart = timer()
  manager = mp.Manager()
  
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
  folders_in = get_folders(indir)

  if not os.path.exists(os.path.join(indir, 'driftcomp')): os.mkdir(os.path.join(indir, 'driftcomp'))
  
  cycles = set()
  regions = set()
  positions = set()
  slices = set()
  channels = set()

  d_in = {}
  pattern1 = re.compile('cyc(\d+)_reg(\d+)')
  for f in folders_in:
    m = pattern1.match(f)
    cyc = int(m.groups()[0])
    reg = int(m.groups()[1])
    cycles.add(cyc)
    regions.add(reg)
    key = 'c{}_r{}'.format(cyc,reg)
    d_in[key] = os.path.join(indir, f)
  
  imagelist_z = glob.glob(os.path.join(indir,folders_in[0],'*_*_Z*_CH*.tif'))
  imagelist_e = glob.glob(os.path.join(indir,folders_in[0],'*_*_EDF_CH*.tif'))

  if len(imagelist_z) > 0:
    pattern2 = re.compile('.*_(\d+)_Z(\d+)_CH(\d).tif')
    for f in imagelist_z:
      m = pattern2.match(f)
      positions.add(int(m.groups()[0]))
      slices.add(int(m.groups()[1]))
      channels.add(int(m.groups()[2]))
  else:
    pattern2 = re.compile('.*_(\d+)_EDF_CH(\d).tif')
    slices = {1}
    for f in imagelist_e:
      m = pattern2.match(f)
      positions.add(int(m.groups()[0]))
      channels.add(int(m.groups()[1]))
  
  z = len(slices)
  
  print('channels:\t', channels)
  print('cycles: \t', cycles)
  print('regions:\t', regions)
  print('positions:\t', positions)
  print('slices:\t', z)
  print()
  
  return d_in, channels, cycles, regions, positions, z

def main(indir, params=None, max_threads=2):
  print("Processing '{}'".format(indir))
  d_in, channels, cycles, regions, positions, z = check_files(indir)
  
  for r in regions:
    # allocate binary output files to prevent a potential race condition
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
  
  score = dispatch_jobs(indir, jobs, (len(cycles), z), params, max_threads)
  
  print("Completed processing '{}'".format(indir))
  return score

if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()
  
  dirs = []
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190513_run8_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190517_run9_postclo_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190523_run10_postveh_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190610_run11_postclo_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190802_run12_preveh_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190816_run13_postclo_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190820_run14_postveh_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20190905_run15_28monTA_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20191018_run17_postveh3_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20191028_run18_preveh2_decon') # v5 to be processed
  #dirs.append('N:/Colin/20191106 deconvolved reprocessed/20191104_run19_postclo_decon') #
  
  dirs.append('G:/20190523_run10_postveh_decon_ca24') # v5 to be processed
  
  dirs.extend(sys.argv[1:])
  
  t0 = timer()
  for indir in dirs:
    #main(indir, [0.5, 0.5, 0.5, 0.5, 8.0, 0.75, 0.0, 0.5, 0.9]) # v1
    #main(indir, [0.25, 1, 1, 1, 6.0, 0.7, 0, 0.35, 1.0])         # v2 best
    #main(indir, [0.25, 1, 1, 1, 6.0, 0.7, 0, 0.3, 0.8])         # v3
    #main(indir, [0.5, 1, 1, 1, 6.0, 0.7, 0, 0.35, 1.0])         # v4
    main(indir, [0.25, 1, 0.5, 1, 6.0, 0.7, 0, 0.35, 1.0])      # v5 best
    #main(indir, [0.25, 0.5, 0.5, 1, 6.0, 0.7, 0, 0.35, 1.0])      # v6 worse
    #main(indir, [0.25, 1, 0.5, 0.5, 6.0, 0.7, 0, 0.35, 1.0])      # v7
    #main(indir, [0.5, 1, 0.5, 1, 6.0, 0.7, 0, 0.35, 1.0])      # v8
    #main(indir, [0.25, 1, 0.25, 1, 6.0, 0.7, 0, 0.35, 1.0])      # v9
  free_libs([libCRISP, libc])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  



  
