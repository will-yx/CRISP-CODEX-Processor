import multiprocessing as mp
import multiprocessing.pool

from ctypes import *
from _ctypes import FreeLibrary

import os
import itertools
import numpy as np
from time import sleep
from timeit import default_timer as timer
import humanfriendly
from PyCaffeinate import PyCaffeinate

import toml


class NoDaemonProcess(mp.Process):
  @property
  def daemon(self):
    return False

  @daemon.setter
  def daemon(self, value):
    pass

class NoDaemonContext(type(mp.get_context())):
  Process = NoDaemonProcess

class MyPool(multiprocessing.pool.Pool):
  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super(MyPool, self).__init__(*args, **kwargs)


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

c_dice_mosaic = libCRISP.dice_mosaic
c_dice_mosaic.restype = c_int
c_dice_mosaic.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]


def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib
  
def dice_mosaic(tid, job, indir, outdir, gx, gy, border, slice_offset):
  reg, cy, ch, sl = job
  
  c_indir = c_char_p(indir.encode('ascii'))
  c_outdir = c_char_p(outdir.encode('ascii'))
  
  status = c_dice_mosaic(c_indir, c_outdir, reg, cy, ch, sl, slice_offset, gx, gy, border[0], border[1])
  
  return status

def process_jobs(args):
  t0 = timer()
  out, tid, d_in, d_out, gx, gy, border, slice_offset, jobs = args
  process = mp.current_process()
  out.put('{}> pid {:5d} got {} jobs'.format(tid, process.pid, len(jobs)))

  sleep(5 * tid)
  
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job))
	  
      proc = mp.Process(target=dice_mosaic, args=(tid, job, d_in, d_out, gx, gy, border, slice_offset))
      proc.start()
      proc.join()
      status = proc.exitcode
      
      if status == 0: out.put('{}> done'.format(tid))
      else: out.put('{}> error processing image ({})!'.format(tid, status))
      
      if status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job))
    if status: break
  
  t1 = timer()
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.2f}s'.format(tid, t1-t0))



def dispatch_jobs(d_in, d_out, joblist, gx, gy, border, slice_offset=0, max_threads=1):
  tstart = timer()
  manager = mp.Manager()
  
  nj = len(joblist)
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to dice {} images'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  
  q = manager.Queue()
  
  with MyPool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, d_in, d_out, gx, gy, border, slice_offset, jobs) for j,jobs in enumerate(joblist_per_thread)]) # 
    
    remainingtime0 = None
    while rs._number_left > 0 or not q.empty():
      try:
        msg = q.get(True, 60)
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
        if nc < nj: print('Message queue is empty - is the program stalled?')
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
          print('  region: {}, cycle: {:02d}, channel: {}, slice: {:02d}'.format(*job))
      print()
    
    p.close()
    p.join()


def main(indir, outdir, config, max_threads=4):
  outdir = os.path.join(outdir, 'processed', 'tiles')
  if not os.path.exists(outdir): os.makedirs(outdir)
  
  zout = config['padding']['zout']
  gx   = config['dimensions']['gx']
  gy   = config['dimensions']['gy']
  nreg = config['dimensions']['regions']
  ncy  = config['dimensions']['cycles']
  nch  = len(config['microscope']['channelNames'])
  
  ztrim = 2
  
  nzout = (zout - 2*ztrim) if zout > 8 else zout
  zregister = (zout+1) >> 1
  z1 = zregister - (nzout >> 1)
  z2 = zregister + (nzout >> 1)
  slices = set(range(z1, z2+1))
  
  regions  = {reg+1 for reg in range(nreg)}
  cycles   = {-ncy} # process all cycles at once   # {cy+1 for cy in range(ncy)}
  channels = {-nch} # process all channels at once # {ch+1 for ch in range(nch)}
  
  slice_offset = -ztrim if zout > 8 else 0 # input slice 3 becomes output slice 1
  
  jobs = list(itertools.product(regions, cycles, channels, slices))
  dispatch_jobs(indir, outdir, jobs, gx, gy, [0, 0], slice_offset, max_threads)



if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()

  stitchdir = 'N:/CODEX processed/20211209_VEGF_regen_run2/stitched'
  finaldir = 'N:/CODEX processed/20211209_VEGF_regen_run2'
  config = toml.load(os.path.join(finaldir, 'CRISP_config.toml')) 
  
  t0 = timer()
  main(stitchdir, finaldir, config, max_threads=8)
  free_libs([libCRISP, libc])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  



  
