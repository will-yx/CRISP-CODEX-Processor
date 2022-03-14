from ctypes import *
from _ctypes import FreeLibrary

import sys
import os
import toml
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

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

c_findpeaks1 = libCRISP.peak_max_simple_c
c_findpeaks1.restype = c_float
c_findpeaks1.argtypes = [POINTER(c_float), c_int]

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def reorder_serpentine(data, serpentine=True):
  if not serpentine: return data
  
  gy, gx, *_ = data.shape
  for j in range(gy):
    if j%2: # odd rows need to be reversed
      for i in range(gx//2):
        data[j,i], data[j,gx-1-i] = data[j,gx-1-i].copy(), data[j,i].copy()
  return data

def reorder_grid(data):
  gy, gx, ty, tx, *_ = data.shape
  newshape = (gy*ty, gx*tx, *data.shape[4:])
  out = np.empty_like(data).reshape(newshape)
  for gj in range(gy):
    for gi in range(gx):
      for tj in range(ty):
        for ti in range(tx):
          out[gj*ty+tj,gi*tx+ti] = data[gj,gi,tj,ti].copy()
  return out

def reorder_ungrid(data, gy, gx, ty, tx):
  out = np.empty((gy,gx,ty,tx), dtype=data.dtype)
  data = data.reshape((gy*ty,gx*tx))
  for gj in range(gy):
    for gi in range(gx):
      for tj in range(ty):
        for ti in range(tx):
          out[gj,gi,tj,ti] = data[gj*ty+tj,gi*tx+ti].copy()
  return out

def find_peak(zscores):
  return c_findpeaks1(zscores.ctypes.data_as(POINTER(c_float)), len(zscores))


def force_converge(ax, dpH, wtH, dpV, wtV, miniter=1000, maxiter=100000):
  if np.any(np.isnan(dpH)): print("Error: NaN value encountered in dpH");
  if np.any(np.isnan(wtH)): print("Error: NaN value encountered in wtH");
  if np.any(np.isnan(dpV)): print("Error: NaN value encountered in dpV");
  if np.any(np.isnan(wtV)): print("Error: NaN value encountered in wtV");
  
  gy, gx = wtH.shape
  npos = gy*gx
  
  converged = False
  maxd = 0.0
  scale = 0.5 # 0.8 is unstable!
  scale_factor = 0.999
  
  p = np.zeros_like(wtH)
  for it in range(maxiter):
    F = np.zeros_like(wtH)
    for j in range(gy):
      for i in range(gx):
        divisor = 1e-6
        
        if i>0:
          F[j,i] += (dpH[j,i] - (p[j,i]-p[j,i-1])) * wtH[j,i]
          divisor += wtH[j,i]
        if j>0:
          F[j,i] += (dpV[j,i] - (p[j,i]-p[j-1,i])) * wtV[j,i]
          divisor += wtV[j,i]
        
        if i<gx-1:
          F[j,i] -= (dpH[j,i+1] - (p[j,i+1]-p[j,i])) * wtH[j,i+1]
          divisor += wtH[j,i+1]
        if j<gy-1:
          F[j,i] -= (dpV[j+1,i] - (p[j+1,i]-p[j,i])) * wtV[j+1,i]
          divisor += wtV[j+1,i]
        
        F[j,i] /= divisor

    F *= scale
    maxd = np.max(F, axis=None)
    p += F
    scale *= scale_factor
    
    if (miniter > 0 and it >= miniter) and (scale < 1e-6 or maxd < 1e-6):
      converged = True
      print('positions converged after {} iterations, maxd: {:.1e}'.format(it, maxd))
      break
  
  if not converged: print('failed to converge after {} iterations, maxd: {:.1e}'.format(maxiter, maxd))
  
  p_ = np.ma.average(p, weights=(wtH+wtV), axis=None)
  p -= p_
  print('p{}_: {:+6.2f}'.format(ax, p_))
  
  print(' p{}:'.format(ax))
  for row in p: print(''.join(['  {:+5.1f}'.format(v) for v in row]))
  
  return p
  

def calculate_tile_alignment(indir, nreg=5, ncy=18, channels=[1], gx=5, gy=7, w=1920, h=1440, z=33, ox=0.3, oy=0.3, serpentine=True, show=0):
  npos = gy*gx
  nch = len(channels)
  
  for r in range(1,1+nreg):
    offsetfile = os.path.join(indir, 'region{:02d}_edge_alignments.bin'.format(r))
    if not os.path.isfile(offsetfile): continue
    
    offsets = np.fromfile(offsetfile, dtype=np.float32)
    offsets = offsets.reshape((ncy, max(channels), 2, 4, gy, gx))
    offsets = offsets[:,[c-1 for c in channels],:,:,:,:] # keep only the alignment channels
    
    dxHs = offsets[:,:,0,0,:,:]
    dyHs = offsets[:,:,0,1,:,:]
    dzHs = offsets[:,:,0,2,:,:]
    wtHs = offsets[:,:,0,3,:,:]

    dxVs = offsets[:,:,1,0,:,:]
    dyVs = offsets[:,:,1,1,:,:]
    dzVs = offsets[:,:,1,2,:,:]
    wtVs = offsets[:,:,1,3,:,:]
    
    if np.any(np.isnan(offsets)):
      print("Error: NaN value encountered in '{}'".format(offsetfile));
      return 1

    def filter_dxyzs(dx, dy, dz, wt):
      dx = dx.reshape(ncy,nch,npos)
      dy = dy.reshape(ncy,nch,npos)
      dz = dz.reshape(ncy,nch,npos)
      wt = wt.reshape(ncy,nch,npos)
      
      # compute weighted average across all channels
      ws = np.maximum(wt, 0) + 1e-12
      dx = np.average(dx, weights=ws, axis=1)
      dy = np.average(dy, weights=ws, axis=1)
      dz = np.average(dz, weights=ws, axis=1)
      wt = np.nanmean(wt, axis=1)
      
      #compute weighted average across all cycles
      dx_ = np.ma.average(dx, weights=wt, axis=0)
      dy_ = np.ma.average(dy, weights=wt, axis=0)
      dz_ = np.ma.average(dz, weights=wt, axis=0)
      wt_ = np.nanmean(wt, axis=0)
      
      dx_std = np.sqrt(np.cov(dx.flat, aweights=wt.flat))
      dy_std = np.sqrt(np.cov(dy.flat, aweights=wt.flat))
      dz_std = np.sqrt(np.cov(dz.flat, aweights=wt.flat))
      wt_std = np.std(wt.flat)
      
      print('  avgs | wt: {:.1e}, dx: {:.1f}, dy: {:.1f}, dz: {:.1f}'.format(np.mean(wt_), np.mean(dx_), np.mean(dy_), np.mean(dz_)))
      print('stdevs | wt: {:.1e}, dx: {:.1f}, dy: {:.1f}, dz: {:.1f}'.format(wt_std, dx_std, dy_std, dz_std))
      
      dx_cutoff = min(50, max(9, dx_std * 2))
      dy_cutoff = min(50, max(9, dy_std * 2))
      dz_cutoff = min(5, max(3, dz_std * 2))
      wt_cutoff = (wt_ - wt_std * 2)
      
      mask = np.zeros_like(wt, dtype=np.uint8)
      mask[np.abs(dx - dx_) > dx_cutoff] = 1
      mask[np.abs(dy - dy_) > dy_cutoff] = 1
      mask[np.abs(dz - dz_) > dz_cutoff] = 1
      
      mask[np.abs(dx) > 40] = 1
      mask[np.abs(dy) > 40] = 1
      mask[np.abs(dz) >  5] = 1
      
      mask[wt < wt_cutoff] = 1

      print('Valid Points:')
      print()
      print(np.array_str(1 - mask).replace('0',' ').replace('1','#'))
      print()
      
      np.putmask(dx, mask, np.outer(np.ones(ncy, dtype=np.float32), dx_))
      np.putmask(dy, mask, np.outer(np.ones(ncy, dtype=np.float32), dy_))
      np.putmask(dz, mask, np.outer(np.ones(ncy, dtype=np.float32), dz_))
      np.putmask(wt, mask, np.outer(np.ones(ncy, dtype=np.float32), np.zeros_like(wt_)))

      if 0:
        cy = 8
        for row in dx[cy].reshape(gy,gx): print(''.join(['  {:+.1f}'.format(v) for v in row]))
        print()
        for row in dy[cy].reshape(gy,gx): print(''.join(['  {:+.1f}'.format(v) for v in row]))
        print()
        for row in dz[cy].reshape(gy,gx): print(''.join(['  {:+.1f}'.format(v) for v in row]))
        print()
        for row in wt[cy].reshape(gy,gx): print(''.join(['  {:+.1e}'.format(v*10000) for v in row]))
        print()

      dims = (ncy, gy, gx)
      return dx.reshape(dims), dy.reshape(dims), dz.reshape(dims), wt.reshape(dims)
    
    dxHs, dyHs, dzHs, wtHs = filter_dxyzs(dxHs, dyHs, dzHs, wtHs)
    dxVs, dyVs, dzVs, wtVs = filter_dxyzs(dxVs, dyVs, dzVs, wtVs)
    
    px = np.empty((gy,gx,ncy), dtype=np.float32)
    py = np.empty((gy,gx,ncy), dtype=np.float32)
    pz = np.empty((gy,gx,ncy), dtype=np.float32)
    for cy in range(ncy):
      print('\nCycle {:02d}'.format(cy+1))
      px[:,:,cy] = reorder_serpentine(force_converge('x', dxHs[cy], wtHs[cy], dxVs[cy], wtVs[cy]), serpentine)
      py[:,:,cy] = reorder_serpentine(force_converge('y', dyHs[cy], wtHs[cy], dyVs[cy], wtVs[cy]), serpentine)
      pz[:,:,cy] = reorder_serpentine(force_converge('z', dzHs[cy], wtHs[cy], dzVs[cy], wtVs[cy]), serpentine)
    
    if np.any(np.isnan(px)) or np.any(np.isnan(py)) or np.any(np.isnan(pz)):
      print('Error: NaN value encountered in output positions');
      return 1
    
    if 1:
      pzfile = os.path.join(indir, 'region{:02d}_tile_pzs.bin'.format(r))
      np.ascontiguousarray(pz.flat).tofile(pzfile)
      pxyzfile = os.path.join(indir, 'region{:02d}_tile_pxyzs.bin'.format(r))
      np.concatenate([px,py,pz], axis=None).tofile(pxyzfile)
    else: print('\n\nWARNING - DISPLAY ONLY - NO FILES HAVE BEEN WRITTEN!\n\n')
  
  print('  finished processing', indir)
  print()
  

def main(dirs=[], show=False):
  dirs.extend(sys.argv[1:])
  
  for basedir in dirs:
    indir = os.path.join(basedir, 'driftcomp')
    if not os.path.isdir(indir):
      print("Error: driftcomp directory '{}' not found!".format(indir))
      continue
    
    config = toml.load(os.path.join(basedir, 'CRISP_config.toml'))
    
    w    = config['dimensions']['width']
    h    = config['dimensions']['height']
    z    = config['dimensions']['slices']
    ox   = config['dimensions']['overlap_x']
    oy   = config['dimensions']['overlap_y']
    gx   = config['dimensions']['gx']
    gy   = config['dimensions']['gy']
    ncy  = config['dimensions']['cycles']
    nreg = config['dimensions']['regions']
    snake = config['dimensions'].get('snake', True)

    channels = config['setup'].get('alignment_channels', [1])
    
    calculate_tile_alignment(indir, nreg, ncy, channels, gx, gy, w, h, z, ox, oy, snake, show=show)

if __name__ == '__main__':
  dirs = []
  dirs.append('N:/CODEX raw/Mouse Sk Muscle/20190513_run08')
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20200130_run22_long_preveh')
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20200202_run23_long_preveh')
  #dirs.append('N:/Colin/codex_training_nov_19')
  #dirs.append('N:/CODEX raw/Human Muscle/20200210 human_run2_regen_2')
  main(dirs, show=True)

free_libs([libCRISP, libc])












