import multiprocessing as mp

from ctypes import *
from _ctypes import FreeLibrary

import sys
import os
import glob
import re
import toml
import itertools
import numpy as np
from skimage import io
from timeit import default_timer as timer
from time import sleep
import humanfriendly
from PyCaffeinate import PyCaffeinate

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

edge_alignment_3d = libCRISP.edge_alignment_3d
edge_alignment_3d.restype = c_float
edge_alignment_3d.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_char_p, c_char_p, c_int, c_int]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def cstr(string):
  return c_char_p(string.encode('ascii'))

def align_edges(out, tid, job, dims, indir, dark, flat):
  ox, oy, gx, gy, nz = dims
  reg, cyc = job
  
  mode = 1 # 0 or 1
  
  status = edge_alignment_3d(cstr(indir), reg, cyc, gx, gy, nz, ox, oy, dark, flat, tid, mode)
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, indir, dims, dark, flat, jobs = args
  
  if dark: dark = cstr(dark)
  if flat: flat = cstr(flat)
  
  process = mp.current_process()
  out.put('{}> pid {:5d} got job list {}'.format(tid, process.pid, jobs), False)
  
  sleep(tid * 30)
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job), False)
      
      status = align_edges(out, tid, job, dims, indir, dark, flat)
      
      if  status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job), False)
  
    if status: break
  
  t1 = timer()
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.1f}s'.format(tid, t1-t0), False)

def dispatch_jobs(indir, joblist, dims, dark, flat, max_threads=1):
  tstart = timer()
  manager = mp.Manager()

  nc = 0 # completed jobs
  nj = len(joblist)
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to align {} cycle-regions'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  
  q = manager.Queue()
  
  with mp.Pool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, indir, dims, dark, flat, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 
    
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
    if(rs._number_left == 0):
      nc = len(completed_jobs)
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
  
  imagelist = glob.glob(os.path.join(indir,folders_in[0],'*_*_Z*_CH*.tif'))
  
  pattern2 = re.compile('.*_(\d+)_Z(\d+)_CH(\d).tif')
  for f in imagelist:
    m = pattern2.match(f)  
    positions.add(int(m.groups()[0]))
    slices.add(int(m.groups()[1]))
    channels.add(int(m.groups()[2]))
  
  img = io.imread(os.path.join(indir, folders_in[0], imagelist[0]))
  h, w = img.shape
  z = len(slices)
  
  print('channels:\t', channels)
  print('cycles: \t', cycles)
  print('regions:\t', regions)
  print('positions:\t', positions)
  print('slices:\t', z)
  print()
  
  return d_in, channels, cycles, regions, positions, z, h, w

def main(indir=None, max_threads=1):
  if not indir or not os.path.isdir(indir):
    print("Error: '{}' is not a valid directory".format(indir))
    return
  
  d_in, channels, cycles, regions, positions, z, h, w = check_files(indir)
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))
  
  ox = config['dimensions']['overlap_x']
  oy = config['dimensions']['overlap_y']
  
  gx = config['dimensions']['gx']
  gy = config['dimensions']['gy']
  
  if config['correction']['correct_darkfield']:
    dark = config['correction']['darkfield_images']
    if isinstance(dark, list): dark = dark[0]
    dark = os.path.join(indir, dark)
  else:
    dark = None
  
  if config['correction']['correct_flatfield']:
    flat = config['correction']['flatfield_images']
    if isinstance(flat, list): flat = flat[0]
    flat = os.path.join(indir, flat)
  else:
    flat = None
  
  if z<4:
    print('Image stack size is too small, aborting!')
    return
  
  for r in regions:
    # allocate binary output files to prevent potential race conditions
    offsetfile = os.path.join(indir, 'driftcomp', 'region{:02d}_edge_alignments.bin'.format(r))
    if not os.path.isfile(offsetfile) or os.path.getsize(offsetfile) < max(cycles)*max(positions)*2*4*4:
      np.full(max(cycles)*max(positions)*2*4, np.nan, dtype=np.float32).tofile(offsetfile)
  
  jobs = list(itertools.product(regions, cycles))
  
  dispatch_jobs(indir, jobs, (ox, oy, gx, gy, z), dark, flat, max_threads)
  print("Completed processing '{}'".format(indir))

if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()
  
  dirs = []
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20200130_run22_long_preveh')
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20200202_run23_long_preveh')
  #dirs.append('N:/Colin/codex_training_nov_19')
  dirs.append('N:/CODEX raw/Mouse Sk Muscle/20190802_run12_preveh')
  dirs.extend(sys.argv[1:])
  
  t0 = timer()
  for indir in dirs: main(indir)
  free_libs([libCRISP, libc])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  


  
