import multiprocessing as mp

from ctypes import *
from _ctypes import FreeLibrary

import sys
import os
import re
import toml
import itertools
import numpy as np
from skimage import io
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

edge_alignment_3d = libCRISP.edge_alignment_3d
edge_alignment_3d.restype = c_float
edge_alignment_3d.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_bool, c_int, c_int, c_int, c_float, c_float, c_char_p, c_char_p, c_int, c_int]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def cstr(string):
  return c_char_p(string.encode('ascii'))

def align_edges(out, tid, job, dims, p):
  ox, oy, gx, gy, nz, nc = dims
  reg, cy, ch = job
  
  snake = p['snake']
  dark = p['dark']
  flat = p['flat']

  if dark: dark = cstr(os.path.join(p['indir'], dark[ch-1]))
  if flat: flat = cstr(os.path.join(p['indir'], flat[ch-1]))
  
  indir = cstr(p['indir'])
  inpattern = re.sub('{position:([0-9 d]+)}', '%\\1', p['inpattern'])
  inpattern = inpattern.format(region=reg, channel=ch)
  inpattern = cstr(f"cyc%03d_reg{reg:03d}/{inpattern}")
  
  mode = 1 # 0 or 1

  status = edge_alignment_3d(indir, inpattern, reg, cy, ch, nc, snake, gx, gy, nz, ox, oy, dark, flat, tid, mode)
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, dims, params, jobs = args
  out.put('{}> pid {:5d} got job list {}'.format(tid, mp.current_process().pid, jobs), False)
  
  sleep(tid * 30)
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job), False)
      
      status = align_edges(out, tid, job, dims, params)
      
      if  status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job), False)
  
    if status: break
  
  t1 = timer()
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.1f}s'.format(tid, t1-t0), False)

def dispatch_jobs(joblist, dims, params, max_threads=1):
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
    rs = p.map_async(process_jobs, [(q, j, dims, params, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 

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
  
  d_in = {}
  pattern1 = re.compile('cyc(\d+)_reg(\d+)')
  for f in folders_in:
    m = pattern1.match(f)
    cyc = int(m.groups()[0])
    reg = int(m.groups()[1])
    key = 'c{}_r{}'.format(cyc,reg)
    d_in[key] = os.path.join(indir, f)
  
  print()
  
  return d_in

def main(indir=None, max_threads=1):
  if not indir or not os.path.isdir(indir):
    print("Error: '{}' is not a valid directory".format(indir))
    return
  
  d_in = check_files(indir)
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))

  w = config['dimensions']['width']
  h = config['dimensions']['height']
  z = config['dimensions']['slices']
  
  ox = config['dimensions']['overlap_x']
  oy = config['dimensions']['overlap_y']
  
  gx = config['dimensions']['gx']
  gy = config['dimensions']['gy']
  snake = config['dimensions'].get('snake', True)
  
  regions   = {reg+1 for reg in range(config['dimensions']['regions'])}
  positions = {pos+1 for pos in range(gx*gy)}
  cycles    = {cyc+1 for cyc in range(config['dimensions']['cycles'])}
  channels  = set(list(config['setup'].get('alignment_channels', [1])))
  
  inpattern  = config['setup']['inpattern']
  
  if config['correction']['correct_darkfield']:
    dark = config['correction']['darkfield_images']
    if not isinstance(dark, list): dark = [dark] * max(channels)
  else:
    dark = None
  
  if config['correction']['correct_flatfield']:
    flat = config['correction']['flatfield_images']
    if not isinstance(flat, list): flat = [flat] * max(channels)
  else:
    flat = None
  
  if z<4:
    print('Image stack size is too small, aborting!')
    return
  
  for r in regions:
    # allocate binary output files to prevent potential race conditions
    offsetfile = os.path.join(indir, 'driftcomp', 'region{:02d}_edge_alignments.bin'.format(r))
    num_entries = max(cycles)*max(channels)*max(positions)*2*4
    if not os.path.isfile(offsetfile) or os.path.getsize(offsetfile) < num_entries*4:
      np.full(num_entries, np.nan, dtype=np.float32).tofile(offsetfile)
  
  jobs = list(itertools.product(regions, cycles, channels))

  params = {'indir': indir, 'inpattern': inpattern, 'snake': snake, 'dark': dark, 'flat': flat}
  
  dispatch_jobs(jobs, (ox, oy, gx, gy, z, max(channels)), params, max_threads)
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
  


  
