import multiprocessing as mp

from ctypes import *
from _ctypes import FreeLibrary

import os
from glob import glob
import shutil
import re
import itertools
import toml
import binascii
import numpy as np
from skimage import io
from time import sleep
from timeit import default_timer as timer
import humanfriendly
from parse_exposure_times import *
from PyCaffeinate import PyCaffeinate

os.environ['VIPS_WARNING'] = '0'

class psf_params(Structure):
  _fields_ = [
              ('zpos', c_float),
              ('ti0', c_float),
              ('tg', c_float), ('tg0', c_float),
              ('ng', c_float), ('ng0', c_float),
              ('ni', c_float), ('ni0', c_float),
              ('ns', c_float),
              ('resXY', c_float),
              ('resZ', c_float),
              ('NA', c_float),
              ('lambda0', c_float), ('lambdaEM', c_float), ('fwhmEM', c_float),
              ('nbasis', c_int), ('nrho', c_int),
              ('nr', c_int), ('accuracy', c_int), ('tolerance', c_float),
              ('sigma_xy', c_float), ('sigma_z', c_float),
              ('extend_psf', c_bool),
              ('center_psf', c_bool),
              ('zshift_psf', c_float),
              ('zpad', c_int),
              ('blind', c_bool)
             ]

class deconvolution_params(Structure):
  _fields_ = [
              ('w', c_int), ('h', c_int), ('z', c_int),
              ('nx', c_int), ('ny', c_int), ('nz', c_int),
              ('correct_darkfield', c_bool), ('correct_flatfield', c_bool),
              ('iterations', c_int),
              ('blind', c_bool),
              ('psf_update_interval', c_int),
              ('subtract_low_tile', c_bool),
              ('subtract_low_fraction', c_float),
              ('subtract_low_global', c_bool),
              ('subtract_low_value', c_float),
              ('conserve_intensity', c_bool),
              ('denoise_residual', c_float),
              ('L_TM_obj', c_float), ('L_TV_obj', c_float), ('B_TV_obj', c_float),
              ('L_TM_psf', c_float), ('L_TV_psf', c_float), ('B_TV_psf', c_float),
              ('host_obj_init', c_bool),
              ('dering_object', c_float),
              ('zout', c_int), ('zpad', c_int), ('zpad_min', c_int), ('zpad_max', c_int),
              ('zmin_project_alpha', c_float),
              ('ax', c_float), ('ay', c_float),
              ('aw', c_float), ('ah', c_float),
              ('ztrim1', c_int), ('ztrim2', c_int), ('ztrim1edf', c_int), ('ztrim2edf', c_int),
              ('zshift', c_float), ('tzshift', c_uint64),
              ('edf', c_bool), ('save_zstack', c_bool),
              ('subband_filter', c_bool), ('majority_filter', c_bool), ('edf_clamp', c_int), 
              ('drc0', c_float), ('drc1', c_float)
             ]
  
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

c_free_psf = libCRISP.free_global_psf
c_free_psf.restype = None
c_free_psf.argtypes = [c_int]

c_kernelsize_cufft = libCRISP.kernelsize_cufft
c_kernelsize_cufft.restype = c_int
c_kernelsize_cufft.argtypes = [c_int, c_int]

c_generate_psf_GibsonLanni = libCRISP.generate_psf_GibsonLanni
c_generate_psf_GibsonLanni.restype = POINTER(c_float)
c_generate_psf_GibsonLanni.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, POINTER(psf_params)]

c_generate_psf_vectorial = libCRISP.generate_psf_vectorial
c_generate_psf_vectorial.restype = POINTER(c_float)
c_generate_psf_vectorial.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, POINTER(psf_params)]

c_generate_psf_gaussian = libCRISP.generate_psf_gaussian
c_generate_psf_gaussian.restype = POINTER(c_float)
c_generate_psf_gaussian.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, POINTER(psf_params)]

c_process_imagestack = libCRISP.process_imagestack
c_process_imagestack.restype = c_float
c_process_imagestack.argtypes = [POINTER(deconvolution_params), c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, POINTER(c_float), POINTER(c_float), c_int]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def cstr(string):
  return c_char_p(string.encode('ascii'))

def generate_psf(out, tid, dims, p_psf, wavelength):
  nz, ny, nx, w, h = dims
  z_psf = nz if p_psf.extend_psf else (nz - p_psf.zpad)
  p_psf.lambdaEM = wavelength[0]
  p_psf.fwhmEM = wavelength[1] * 0
  out.put('{}> generating PSF for wavelength: {:.0f}nm'.format(tid, p_psf.lambdaEM * 1e3))
  out.put('{}> generating PSF of size {} x {} x {}'.format(tid, z_psf, ny, nx))
  #c_psf = c_generate_psf_vectorial(tid, w, h, nx, ny, nz, pointer(p_psf))
  c_psf = c_generate_psf_GibsonLanni(tid, w, h, nx, ny, nz, pointer(p_psf))
  #c_psf = c_generate_psf_gaussian(tid, w, h, nx, ny, nz, pointer(p_psf))
  return c_psf

def deconvolve_imagestack(out, tid, job, p, p_RL, c_psf):
  t0 = timer()
  
  nz, ny, nx  = p_RL.nz, p_RL.ny, p_RL.nx
  cy, ch, reg, pos = job

  p_RL.ax = p['ca_xy'][ch][0] if p['ca_xy'] else 0
  p_RL.ay = p['ca_xy'][ch][1] if p['ca_xy'] else 0

  p_RL.zshift = p['zshifts'][reg][cy-1, pos-1] + p['czshifts'][ch] if p['zshifts'] else 0
  c_tzshifts = np.ascontiguousarray(p['tzshifts'][reg][cy-1, pos-1] + p['czshifts'][ch], dtype=np.float32).ctypes.data_as(POINTER(c_float)) if p['tzshifts'] else None

  key = 'c{}_r{}'.format(cy, reg)
  indir  = cstr(p['d_in' ][key])
  outdir = cstr(p['d_out'][key])
      
  inpattern  = cstr(p[ 'inpattern'].format(cycle=cy, region=reg, position=pos, channel=ch))
  outpattern = cstr(p['outpattern'].format(cycle=cy, region=reg, position=pos, channel=ch))
  
  dark = cstr(p['dark'][ch]) if p['dark'] else None
  flat = cstr(p['flat'][ch]) if p['flat'] else None
  
  status = c_process_imagestack(p_RL, indir, outdir, inpattern, outpattern, dark, flat, nx, ny, nz, reg, pos, ch, c_psf, c_tzshifts, tid)
  
  if status==1: return status
  if status==0: out.put('{}> processed imagestack in {:.1f}s'.format(tid, timer() - t0))
  return status

def process_jobs(args):
  t0 = timer()
  
  out, tid, p, jobs = args                             
  out.put('{}> pid {:5d} got {} jobs'.format(tid, mp.current_process().pid, len(jobs)))
  
  # make copies of the ctypes structs to avoid thread conflicts
  p_psf =           psf_params.from_buffer_copy(p['p_psf'])
  p_RL  = deconvolution_params.from_buffer_copy(p['p_RL'])
  
  psf_ch = None
  c_psf  = None

  sleep(30 * tid)
  
  for job in jobs:
    for attempts in reversed(range(3)):
      out.put('{}> processing job {}'.format(tid, job))
      cy, ch, reg, pos = job
      if ch != psf_ch or p_RL.blind:
        psf_ch = ch
        c_psf = generate_psf(out, tid, (p_RL.nz, p_RL.ny, p_RL.nx, p_RL.w, p_RL.h), p_psf, p['wavelengths'][ch])
      
      status = deconvolve_imagestack(out, tid, job, p, p_RL, c_psf)
      
      if status == 0: break
      if attempts > 0: sleep(30)
     
    out.put((status, tid, job))
    if status: break
  
  t1 = timer()
  c_free_psf(tid)
  free_libs([libCRISP, libc])
  out.put('{}> joblist complete, elapsed {:.0f}s'.format(tid, t1-t0))
  out.put((100, tid, None))


def dispatch_jobs(outdir, params, joblist, resume=False, max_threads=1):
  struct_keys = ['p_psf', 'p_RL']
  printable_params = {k: v for k, v in params.items() if k not in struct_keys}
  config_bytes = bytes(repr(printable_params).encode('ascii'))
  for k in struct_keys:
    config_bytes += bytes(params[k])
  config_hash = binascii.crc32(config_bytes)
  
  progress_dict = {}
  progressfile = os.path.join(outdir, 'deconvolution_progress.npz')
  if os.path.isfile(progressfile):
    progress_dict = np.load(progressfile, allow_pickle=True, fix_imports=False)['data'].item()

    if resume:
      total_jobcount = len(joblist)
      joblist = [job for job in joblist if progress_dict.get(repr(job)) != config_hash]
    
      print('  Resuming deconvolution: {} of {} jobs completed, {} remaining'.format(total_jobcount-len(joblist), total_jobcount, len(joblist)))
  
  tstart = timer()
  manager = mp.Manager()
  
  nj = len(joblist)
  if nj < 1:
    print('No work to do!')
    return
  
  na = nj
  nt = np.min([mp.cpu_count(), nj, max_threads if max_threads>0 else nj])
  
  print('Using {} threads to deconvolve {} stacks'.format(nt, nj))
  
  jobs_per_thread = [nj//nt + (t < (nj%nt)) for t in range(nt)]
  print('jobs per thread:', jobs_per_thread)
  print()
  
  job_start = [0]; job_start.extend(np.cumsum(jobs_per_thread))
  joblist_per_thread = [joblist[job_start[t]:job_start[t+1]] for t in range(nt)]
  completed_jobs = []
  failed_jobs = []
  nc = 0
  
  q = manager.Queue()
  
  with mp.Pool(processes=nt) as p:
    rs = p.map_async(process_jobs, [(q, j, params, jobs) for j,jobs in enumerate(joblist_per_thread)])
    
    remainingtime0 = None
    while rs._number_left > 0 or not q.empty():
      try:
        msg = q.get(True, 300)
        if isinstance(msg, str):
          print(msg)
        else:
          if msg[0] == 0: # success
            progress_dict.update({repr(msg[2]): config_hash})
            np.savez_compressed(progressfile+'.tmp.npz', data=progress_dict)
            os.replace(progressfile+'.tmp.npz', progressfile)
            
            completed_jobs.append(msg[2])
            nc = len(completed_jobs)
            remainingtime1 = (timer() - tstart)/nc * (na-nc)
            remainingtime = np.mean([remainingtime0, remainingtime1]) if remainingtime0 else remainingtime1
            remainingtime0 = remainingtime1
            timestring = '' if nc==na else ' ({} remaining)'.format(humanfriendly.format_timespan(remainingtime, max_units=2))
            print('## progress: {} of {} [{:.1f}%]{}'.format(nc, na, 100*nc/na, timestring))
          elif msg[0] == 100:
            print('Worker {} has completed its jobs'.format(msg[1]))
          else:
            failed_jobs.append(msg[2])
            print('%%%% Job {} from worker {} failed (status: {})! %%%%'.format(msg[2], msg[1], msg[0]))
          
      except mp.queues.Empty:
        print('Message queue is empty - is the program stalled?')
        break
    if(rs._number_left == 0):
      nc = len(completed_jobs)
      if(nc == na): print('Finished - processed {} tiles'.format(nc))
      else: print('Incomplete - processed {} of {} tiles'.format(nc, na))
    else:
      print('Queue timeout - {} workers stuck'.format(rs._number_left))
      if(nc == na): print('Processed {} tiles'.format(nc))
      else: print('Processed {} of {} tiles'.format(nc, na))
    
    if(nc != nj):
      print('Failed jobs: {}'.format(nj-nc))
      for job in joblist:
        if not job in completed_jobs:
          print('  cycle: {:02d}, channel: {}, region: {}, position: {:02d}'.format(*job))
      print()
    
    p.close()
    p.join()

def get_folders(root):
  pattern1 = re.compile('cyc\d+_reg\d+.*')
  pattern2 = re.compile('(cyc\d+_reg\d+)')
  
  directories = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
  cycle_region_dirs = [d for d in directories if pattern1.match(d)]
  renamed = [pattern2.match(d).group(0) for d in cycle_region_dirs]
  
  return cycle_region_dirs, renamed

def check_files(indir, outdir):
  folders_in, folders_out = get_folders(indir)
  
  if not os.path.exists(outdir): os.makedirs(outdir)
  for d in folders_out:
    d = os.path.join(outdir, d)
    if not os.path.exists(d): os.makedirs(d)
  
  parse_exposure_times(indir)
  
  if 1:
    miscfiles = glob.glob(os.path.join(indir,'*.*'))
    for file in miscfiles:
      if os.path.isfile(file):
        fname = os.path.basename(file)
        outfile = os.path.join(outdir, fname)
        print('copying {}'.format(fname, os.path.basename(outdir)))
        shutil.copy(file, outfile)
    print()

  print('cycle_region folders:')
  print(folders_in)
  print()
  
  d_in = {}
  d_out = {}
  pattern1 = re.compile('cyc(\d+)_reg(\d+)')
  for f, f2 in zip(folders_in, folders_out):
    m = pattern1.match(f)
    cyc = int(m.groups()[0])
    reg = int(m.groups()[1])
    key = 'c{}_r{}'.format(cyc,reg)
    d_in[key] = os.path.join(indir, f)
    d_out[key] = os.path.join(outdir, f2)
  
  return d_in, d_out

def load_config(indir):
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))
  print(config)
  print()
  
  wavelengths  = config['microscope']['wavelengthEM']
  fwhms        = config['microscope']['fwhmEM']
  wavelengths  = {int(k) : (v * 1e-3, fwhms[k]) for k,v in wavelengths.items()}
  channels     = wavelengths.keys()
  
  cycles    = config['dimensions']['cycles']
  regions   = config['dimensions']['regions']
  positions = config['dimensions']['gx'] * config['dimensions']['gy']
  snake = config['dimensions'].get('snake', True)
  
  w = config['dimensions']['width']
  h = config['dimensions']['height']
  z = config['dimensions']['slices']
  
  p_psf = psf_params()
  p_psf.resXY   = config['microscope']['resXY']
  p_psf.resZ    = config['microscope']['resZ']
  p_psf.NA      = config['microscope']['NA']
  p_psf.lambda0 = min([v[0] for v in wavelengths.values()])
  
  if config.get('gibson_lanni'):
    p_psf.zpos   = config['gibson_lanni']['zpos']
    p_psf.ti0    = config['gibson_lanni']['ti0']
    p_psf.tg     = config['gibson_lanni']['tg']
    p_psf.tg0    = config['gibson_lanni']['tg0']
    p_psf.ng     = config['gibson_lanni']['ng']
    p_psf.ng0    = config['gibson_lanni']['ng0']
    p_psf.ni     = config['gibson_lanni']['ni']
    p_psf.ni0    = config['gibson_lanni']['ni0']
    p_psf.ns     = config['gibson_lanni']['ns']
    p_psf.nbasis = config['gibson_lanni']['nbasis']
    p_psf.nrho   = config['gibson_lanni']['nrho']
  
  if config.get('vectorial'):
    p_psf.nr        = config['vectorial'].get('radius', 256)
    p_psf.accuracy  = config['vectorial'].get('accuracy', 9)
    p_psf.tolerance = config['vectorial'].get('tolerance', 0.1)
  
  if config.get('gaussian'):
    p_psf.sigma_xy = config['gaussian'].get('sigma_xy', 1.0)
    p_psf.sigma_z  = config['gaussian'].get('sigma_z',  1.0)
  
  p_psf.extend_psf = config['padding'].get('extend_psf', False)
  p_psf.center_psf = config['padding'].get('center_psf', True)
  p_psf.zshift_psf = config['padding'].get('zshift_psf', 0)
  
  p_psf.blind = config['deconvolution'].get('blind') or False
  p_psf.zpad = c_kernelsize_cufft(int(round(z + z * config['padding']['zpad_min'])), int(round(z + z * config['padding']['zpad_max']))) - z
  
  p_RL = deconvolution_params()
  p_RL.w = w
  p_RL.h = h
  p_RL.z = z
  
  p_RL.nx = c_kernelsize_cufft(int(round(w*1.05)), int(round(w*1.5)))
  p_RL.ny = c_kernelsize_cufft(int(round(h*1.05)), int(round(h*1.5)))
  p_RL.nz = z + p_psf.zpad
  
  if config.get('correction'):
    p_RL.correct_darkfield = config['correction'].get('correct_darkfield', False)
    p_RL.correct_flatfield = config['correction'].get('correct_flatfield', False)
    
    dark = config['correction']['darkfield_images'] if p_RL.correct_darkfield else ''
    flat = config['correction']['flatfield_images'] if p_RL.correct_flatfield else ''
    
    if not isinstance(dark, list): dark = [dark] * len(channels)
    if not isinstance(flat, list): flat = [flat] * len(channels)
    
    if p_RL.correct_darkfield: dark = {c: os.path.join(indir, f) for c,f in zip(channels, dark)}
    if p_RL.correct_flatfield: flat = {c: os.path.join(indir, f) for c,f in zip(channels, flat)}
    
    if len(dark) != len(channels): raise('Wrong number of darkfield images were given!')
    if len(flat) != len(channels): raise('Wrong number of flatfield images were given!') 
  
  p_RL.iterations = config['deconvolution']['iterations']
  p_RL.blind = config['deconvolution'].get('blind', False)
  p_RL.psf_update_interval = config['deconvolution'].get('psf_update_interval', 1)
  
  p_RL.denoise_residual = config['deconvolution'].get('denoise_residual', 0) 
  p_RL.L_TM_obj = config['deconvolution'].get('L_TM_obj', 0)
  p_RL.L_TV_obj = config['deconvolution'].get('L_TV_obj', 0)
  p_RL.B_TV_obj = config['deconvolution'].get('B_TV_obj', 0)
  p_RL.L_TM_psf = config['deconvolution'].get('L_TM_psf', 0)
  p_RL.L_TV_psf = config['deconvolution'].get('L_TV_psf', 0)
  p_RL.B_TV_psf = config['deconvolution'].get('B_TV_psf', 0)
  
  p_RL.host_obj_init = config['deconvolution'].get('host_obj_init', False)
  p_RL.dering_object = config['padding'].get('dering_object', -2)
  p_RL.zpad = p_psf.zpad
  p_RL.zout = config['padding']['zout'] or z
  p_RL.zmin_project_alpha = config['padding'].get('zmin_project_alpha', 0)
  
  p_RL.ztrim1 = config['padding']['ztrim1']
  p_RL.ztrim2 = config['padding']['ztrim2']
  p_RL.ztrim1edf = config['padding'].get('ztrim1edf', 0)
  p_RL.ztrim2edf = config['padding'].get('ztrim2edf', 0)
  p_RL.zshift = 0

  if p_RL.z != p_RL.zout + p_RL.ztrim1 + p_RL.ztrim2 + p_RL.ztrim1edf + p_RL.ztrim2edf:
    raise ValueError('Error: Number of input z slices is not the sum of output plus trimmed slices!') 

  p_RL.edf = True
  p_RL.save_zstack = True
  if config.get('extended_depth_of_field'):
    p_RL.edf  = config['extended_depth_of_field'].get('enabled', True)
    p_RL.save_zstack = config['extended_depth_of_field'].get('save_zstack', True)
    p_RL.subband_filter = config['extended_depth_of_field'].get('subband_filter', False)
    p_RL.majority_filter = config['extended_depth_of_field'].get('majority_filter', False)
    p_RL.edf_clamp = config['extended_depth_of_field'].get('clamp_values', False)
  
  if config.get('dynamic_range_compression'):
    p_RL.drc0 = config['dynamic_range_compression'].get('drc0', 0)
    p_RL.drc1 = config['dynamic_range_compression'].get('drc1', 0)
  
  if(config.get('chromatic_aberration_correction') and config['chromatic_aberration_correction']['enabled']):
    p_RL.aw = config['chromatic_aberration_correction']['width']
    p_RL.ah = config['chromatic_aberration_correction']['height']
    
    ca_xy = {c: (config['chromatic_aberration_correction']['ax'][c-1],
                 config['chromatic_aberration_correction']['ay'][c-1]) for c in channels}
  else:
    ca_xy = None
  
  if config['padding'].get('frame_zshift', True):
    zshifts = {}
    for r in range(1,1+regions):
      zshiftsfile = os.path.join(indir, 'driftcomp', 'region{:02d}_zshifts.bin'.format(r))
      zshifts[r] = np.fromfile(zshiftsfile, dtype=np.float32).reshape(cycles, positions)
  else: zshifts = None
  
  if config['padding'].get('tiled_zshift', True):
    tzshifts = {}
    tx = (w + 256//4 + 256-1) // 256
    ty = (h + 256//4 + 256-1) // 256
    for r in range(1,1+regions):
      tile_zshiftsfile = os.path.join(indir, 'driftcomp', 'region{:02d}_tile_zshifts.bin'.format(r))
      tzshifts[r] = np.fromfile(tile_zshiftsfile, dtype=np.float32).reshape(cycles, positions, ty*tx)
  else: tzshifts = None
  
  if config['microscope'].get('channelZPos'):
    channel_z_positions = config['microscope']['channelZPos']
    czshifts = {int(k) : -channel_z_positions[k] / p_psf.resZ for k in fwhms.keys()}
  else: czshifts = {int(k) : 0 for k in wavelengths.keys()}

  inpattern  = config['setup'].get( 'inpattern', '{region}_{position:05d}_Z%03d_CH{channel:d}.tif')
  outpattern = config['setup'].get('outpattern', '{region}_{position:05d}_Z%%03d_CH{channel:d}.tif')
  
  regions   = {reg+1 for reg in range(regions)}
  positions = {pos+1 for pos in range(positions)}
  cycles    = {cyc+1 for cyc in range(cycles)}
  channels  = set(channels)
  
  if 1:
    params = {'inpattern': inpattern, 'outpattern': outpattern, 'snake': snake}
    params.update({'p_psf': p_psf, 'p_RL': p_RL})
    params.update({'dark': dark, 'flat': flat})
    params.update({'wavelengths': wavelengths, 'ca_xy': ca_xy})
    params.update({'zshifts': zshifts, 'tzshifts': tzshifts, 'czshifts': czshifts})
    return regions, positions, cycles, channels, params
  
  return p_psf, p_RL, wavelengths, ca_xy, zshifts, tzshifts, czshifts, dark, flat, regions, positions, cycles, channels


def main(indir, outdir, max_threads=2, override=0):
  d_in, d_out = check_files(indir, outdir)
  
  regions, positions, cycles, channels, params = load_config(indir)
  params.update({'d_in': d_in, 'd_out': d_out})
  
  if params['p_RL'].w<64 or params['p_RL'].h<64 or params['p_RL'].z<4:
    print('Image stack size is too small, aborting!')
    return 1
  
  jobs = list(itertools.product(cycles, channels, regions, positions))
  resume = True # check deconvolution progress file for remaining jobs
  
  if override: # Perform deconvolution on a specific set of tiles
    jobs = override #(cycles, channels, regions, positions)
    print('Override mode: deconvolving {}'.format(jobs))
    resume = False # do not use resume for override
  
  dispatch_jobs(outdir, params, jobs, resume, max_threads)
  
  print("Completed processing '{}'".format(indir))
  
  return 0

if __name__ == '__main__':
  pyCaffeinate = PyCaffeinate()
  pyCaffeinate.preventSleep()
  
  ###########################################################################################
  indir  = 'N:/CODEX raw/Mouse Sk Muscle/20190513_run08'
  tempdir = 'X:/temp/deleteme/20190513_run08_testing'
  override_jobs = False
  ###########################################################################################
  
  t0 = timer()
  main(indir, tempdir, max_threads=2, override=override_jobs)
  free_libs([libCRISP, libc])
  t1 = timer()
  elapsed = humanfriendly.format_timespan(t1-t0)
  print('Total run time: {}'.format(elapsed))
  
  pyCaffeinate.allowSleep()
  
