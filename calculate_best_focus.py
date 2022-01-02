from ctypes import *
from _ctypes import FreeLibrary

import sys
import os
import toml
import numpy as np
from scipy import interpolate, signal

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

def reorder_serpentine(data):
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

def find_peak_spline(zscores):
  xs = np.linspace(1, len(zscores)-2, (len(zscores)-3)*10+1)
  
  spline = interpolate.InterpolatedUnivariateSpline(range(len(zscores)), zscores)
  yinterp = spline(xs)
  
  peak = xs[np.argmax(yinterp)]

  if 0 and np.max(yinterp) > 1e-12:
    plt.plot(range(len(zscores)), zscores)
    plt.plot(xs, yinterp)
    
    print(np.argmax(zscores))
    print(peak)
    plt.show()
  
  return peak

def interpolate_zscores(zscores, dz, us):
  xs = np.linspace(0, len(zscores)-1, (len(zscores)-1)*us+1)
  
  spline = interpolate.InterpolatedUnivariateSpline(range(len(zscores)), zscores, ext=3)
  yinterp = spline(xs-dz)
  
  if 0 and abs(dz) > 4 and np.max(yinterp) > 1e-6:
    print(dz)
    plt.plot(dz+np.array(range(len(zscores))), zscores)
    plt.plot(xs, yinterp)
    plt.show()
    
  return yinterp


def interpolate_zscores_plot(zscores, dz, us):
  xs = np.linspace(0, len(zscores)-1, (len(zscores)-1)*us+1)
  
  spline = interpolate.InterpolatedUnivariateSpline(range(len(zscores)), zscores, ext=3)
  yinterp = spline(xs-dz)
  
  plt.plot(dz+np.array(range(len(zscores))), zscores)
  plt.plot(xs, yinterp)
  
  return yinterp

def calculate_best_focus(indir, nreg=5, ncy=18, gx=5, gy=7, w=1920, h=1440, z=33, ox=0.3, oy=0.3, serpentine=True, show=0):
  npos = gy*gx
  
  us = 10
  nx = 256
  ny = 256
  
  tx = (w + nx//4 + nx-1) // nx
  ty = (h + ny//4 + ny-1) // ny

  nts = ty*tx
  
  ngrid = gy*gx*ty*tx
  
  zu = (z-1) * us + 1
  
  total_raw_var_sum = 0
  total_new_var_sum = 0
  surface_var_sum = 0
  
  for r in range(1,1+nreg):
    offsetfile = os.path.join(indir, 'region{:02d}_offsets.bin'.format(r))
    if not os.path.isfile(offsetfile): return 1
    offsets = np.fromfile(offsetfile, dtype=np.float32)
    offsets = offsets.reshape((gy, gx, 4, ncy))
    if serpentine: offsets = reorder_serpentine(offsets)
    driftz = offsets[:,:,2,:]
    driftw = offsets[:,:,3,:]
    
    tile_cycle_zscorefile = os.path.join(indir, 'region{:02d}_ztiles.bin'.format(r))
    if not os.path.isfile(tile_cycle_zscorefile): return 1
    position_cycle_zscores = np.fromfile(tile_cycle_zscorefile, dtype=np.float32)
    position_cycle_zscores = position_cycle_zscores.reshape((gy, gx, ty, tx, ncy, z))
    if serpentine: position_cycle_zscores = reorder_serpentine(position_cycle_zscores)
    
    if np.any(np.isnan(offsets)):
      print("Error: NaN value encountered in '{}'".format(offsetfile));
      return 1
    
    if np.any(np.isnan(position_cycle_zscores)):
      print("Error: NaN value encountered in '{}'".format(tile_cycle_zscorefile));
      return 1
    
    position_weights = np.array([np.median([np.max(zscores) for zscores in position_cycle_zscores.reshape(npos,nts*ncy,z)[pos]]) for pos in range(npos)])
    #print('Position weights:')
    #for row in position_weights.reshape(gy,gx): print(''.join(['  {:.2e}'.format(v) for v in row]))
    
    grid_cycle_zscores = reorder_grid(position_cycle_zscores)
    flat_cycle_zscores = grid_cycle_zscores.reshape(ngrid, ncy, z)
    
    grid_weights = np.array([np.median([np.max(zscores) for zscores in flat_cycle_zscores[pos]]) for pos in range(ngrid)])
    grid_weights /= np.mean(grid_weights.flat)
    #print('Grid weights:')
    #for row in grid_weights.reshape(gy*ty,gx*tx): print(''.join(['  {:.1e}'.format(v) for v in row]))
    
    cycle_weights = np.array([np.median([np.max(zscores) for zscores in flat_cycle_zscores[:,cy,:]]) for cy in range(ncy)])
    #print('Cycle weights:')
    #for wt in cycle_weights: print(''.join(['    {:.2e}'.format(wt)]))
    
    driftz_grid = np.array([[driftz[(p//(gx*tx))//ty, (p%(gx*tx))//tx, cy] for p in range(ngrid)] for cy in range(ncy)])
    
    #cycle_mean_zdrift = [np.average(driftz[:,:,cy].flat, weights=position_weights) for cy in range(ncy)]
    #print(cycle_mean_zdrift)

    if 0:
      for pos in [999]:
        print('pos: {} of {}'.format(pos+1, ngrid))
        for cy in range(ncy):
          interpolate_zscores_plot(flat_cycle_zscores[pos,cy], 0, us)
        plt.show()
    
    cycle_zpeaks = np.empty((ncy, ngrid), dtype=np.float32)
    cycle_grid_zs = np.empty((ncy, ngrid, zu), dtype=np.float32)
    for cy in range(ncy):
      grid_zscores = flat_cycle_zscores[:,cy]
      cycle_zpeaks[cy] = np.array([find_peak(grid_zscores[pos]) for pos in range(ngrid)])
      #cycle_zpeaks[cy] = np.array([find_peak_spline(grid_zscores[pos]) for pos in range(ngrid)])
      cycle_grid_zs[cy] = np.array([interpolate_zscores(grid_zscores[pos], 0, us) for pos in range(ngrid)])
    
    focusvolume = np.array([np.average(cycle_grid_zs[:,pos,:], weights=cycle_weights, axis=0) for pos in range(ngrid)])
    focusplane = np.array([find_peak(profile)/us for profile in focusvolume])
    focuserror = np.array([np.std([find_peak_spline(profile)/us for profile in cycle_grid_zs[:,pos,:]]) for pos in range(ngrid)])
    print('mean cycle z error: {:.1e}'.format(np.mean(focuserror)))
    
    if 0: focusweight = np.array([np.log(1 + np.max(profile)*1e6) for profile in focusvolume])
    else: focusweight = np.array([pow(0.01 + std, -0.5) for std in focuserror])
    avgerr = np.mean(focuserror)
    
    if 0:
      print('Focus plane:')
      for row in focusplane.reshape(gy*ty,gx*tx): print(''.join(['  {:.1f}'.format(v) for v in row]))
      print('Focus error:')
      for row in focuserror.reshape(gy*ty,gx*tx): print(''.join(['  {:.1f}'.format(v) for v in row]))
      print('Focus weight:')
      for row in focusweight.reshape(gy*ty,gx*tx): print(''.join(['  {:.1f}'.format(v) for v in row]))
      print()
      print('  Average error: {:.4f}'.format(avgerr))
      print()
    
    # fit a surface to the best focus tiles
    if show:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
    
    fx = (w//tx) - ((w//tx)*(tx-1) + nx - w) / (tx-1)
    fy = (h//ty) - ((h//ty)*(ty-1) + ny - h) / (ty-1)
    
    txs = [int(round(gi*fx + nx/2)) for gi in range(tx)]
    tys = [int(round(gj*fy + ny/2)) for gj in range(ty)]
    
    X = np.array([[(i//tx)*w*(1-ox) + txs[i%tx] for i in range(gx*tx)] for j in range(gy*ty)])
    Y = np.array([[(j//ty)*h*(1-oy) + tys[j%ty] for i in range(gx*tx)] for j in range(gy*ty)])
    
    Z = focusplane.reshape(gy*ty,gx*tx).copy()
    
    wmin = np.percentile(grid_weights.flat, 10) * 0.1
    W = np.array([(max(wmin,gw))**0.1 * fw**0.5 for gw, fw in zip(grid_weights.flat, focusweight)])

    WS = sorted(W)
    knee = 3*len(W)//4
    cutoff = WS[knee] * 0.45
    while knee > 0 and WS[knee] > cutoff: knee -= 1
    
    percentile = 100 * np.max([0.05, knee / len(W)])
    threshold = np.percentile(W, percentile) if knee else max(np.percentile(W, 50) * 0.5, np.percentile(W, 25))
    
    Z[W.reshape(gy*ty,gx*tx) < threshold] = np.NaN
    W[W < threshold] = threshold
    Z0 = Z.copy()
    while np.isnan(Z).any():
      Zi = Z.copy()
      for y in range(gy*ty):
        for x in range(gx*tx):
          if np.isnan(Zi[y,x]):
            zsum, zwt = 0,0
            radius = 10
            for dy in range(-radius,radius+1):
              for dx in range(-radius,radius+1):
                if y+dy < 0 or y+dy >= gy*ty: continue
                if x+dx < 0 or x+dx >= gx*tx: continue
                if np.isfinite(Zi[y+dy,x+dx]):
                  wt = 1 / np.sqrt(dy*dy + dx*dx)
                  zsum += wt * Zi[y+dy,x+dx]
                  zwt  += wt
            if zwt > 10: Z[y,x] = zsum / zwt
    
    print('Removed {:.1f}% of points due to low weight'.format(100 * np.count_nonzero(np.isnan(Z0))/Z0.size))
    
    spline2D = interpolate.bisplrep(X.flat, Y.flat, Z, w=W, s=ngrid**3)
    ZI = np.array([interpolate.bisplev(xi, yi, spline2D) for xi,yi in zip(X.flat,Y.flat)]).reshape(gy*ty, gx*tx)
    
    ZI = np.clip(ZI, 0, z-1)
    ZIvalid = ZI[np.isfinite(Z0)].flat
    zimean = np.mean(ZIvalid)
    zimin = np.min(ZIvalid)
    zimax = np.max(ZIvalid)
    
    print('Surface mean: {:.1f} ({:+.1f} slices from center)'.format(zimean, zimean - (z-1)/2))
    print('Surface min: {:.1f}, max: {:.1f}'.format(zimin, zimax))
    print('Surface var: {:.1f}'.format(np.var(ZIvalid)))
    surface_var_sum += np.var(ZIvalid)
    
    if show:
      warnings.filterwarnings('ignore', message='Z contains NaN values. This may result in rendering artifacts.')

      XS = np.argsort(X, axis=1).reshape(Z.shape)
      YS = np.argsort(Y, axis=0).reshape(Z.shape)
      
      XP = np.sort(X, axis=1)
      YP = np.sort(Y, axis=0)
      Z0P = np.array([[Z0[YS[j][i]][XS[j][i]] for i in range(gx*tx)] for j in range(gy*ty)])
      ZIP = np.array([[ZI[YS[j][i]][XS[j][i]] for i in range(gx*tx)] for j in range(gy*ty)])
      
      plt.title('Region {:}'.format(r))
      ax.plot_surface(XP, -YP, ZIP)
      #ax.plot_surface(X, -Y, Z, alpha=0.5)
      ax.plot_surface(X, -Y, Z0, alpha=0.5)
      
      ax.set_xlabel('X Axis')
      ax.set_ylabel('Y Axis')
      ax.set_zlabel('Z Axis')
      ax.set_zlim(0, z-1)
    
    target_z = (z-2 + (z&1)) / 2 - 0.5 # shift center by a half slice because there seems to be a consistent bias
    print(' slices: {}, middle: {}, target: {}'.format(z, (z-1)/2, target_z))
    
    raw_var_sum = 0
    new_var_sum = 0
    cycle_grid_zshifts = np.empty((ncy, ngrid), dtype=np.float32)
    cycle_position_zshifts = np.empty((ncy, npos), dtype=np.float32)
    for cy in range(ncy):
      raw_positions = cycle_zpeaks[cy]
      
      cycle_grid_zshifts[cy] = ((target_z - ZI).flat) - driftz_grid[cy]
      cycle_grid_zshifts[cy] = np.clip(cycle_grid_zshifts[cy], -z/2, z/2)
      
      raw_var_sum += np.var(raw_positions)
      new_var_sum += np.var(cycle_grid_zshifts[cy])
      
      if show and 0:
        for row in cycle_zpeaks[cy].reshape(gy*ty,gx*tx): print(''.join(['  {:.1f}'.format(v) for v in row]))
        print('  np.mean(cycle_zpeaks[cy]): ', np.mean(cycle_zpeaks[cy]))
        
        cymean = np.mean(cycle_grid_zshifts[cy])
        print('cycle {:02d} mean zdrift: {:5.2f}, adjusted: {:5.2f}'.format(cy+1, cycle_mean_zdrift[cy], cymean))
        print('             raw std: {:5.2f},  new std: {:5.2f}'.format(np.std(raw_positions), np.std(cycle_grid_zshifts[cy])))
        
        for row in raw_positions.reshape(gy*ty,gx*tx):
          print('  cy{:02d}'.format(cy+1),('  {:.1f}'*gx*tx).format(*row))
        print()
        
        for row in cycle_grid_zshifts[cy].reshape(gy*ty,gx*tx):
          print('  cy{:02d}'.format(cy+1),('  {:.1f}'*gx*tx).format(*row))
        print()
      
      cycle_ungrid_zshifts = reorder_ungrid(cycle_grid_zshifts[cy], gy, gx, ty, tx)
      
      cycle_position_zshifts[cy] = np.array([np.average(zs, weights=ws) for zs,ws in zip(cycle_ungrid_zshifts.reshape(npos, nts), W.reshape(npos, nts))])
      cycle_position_zshifts[cy] = reorder_serpentine(cycle_position_zshifts[cy].reshape(gy,gx)).flat
      
      cycle_grid_zshifts[cy] = reorder_serpentine(cycle_ungrid_zshifts).flat
    
    if show and 0:
      for cy,dz in enumerate(cycle_mean_zdrift):
        print('  cy{:02} dz: {:+.1f}'.format(cy+1,dz))
      print()
    
    total_raw_var_sum += raw_var_sum
    total_new_var_sum += new_var_sum
    print('  var: {:4.1f} -> {:4.1f}'.format(raw_var_sum, new_var_sum))
    print()
    
    if not show:
      zshiftsfile = os.path.join(indir, 'region{:02d}_zshifts.bin'.format(r))
      np.ascontiguousarray(cycle_position_zshifts.flat).tofile(zshiftsfile)
      
      tile_zshiftsfile = os.path.join(indir, 'region{:02d}_tile_zshifts.bin'.format(r))
      np.ascontiguousarray(cycle_grid_zshifts.flat).tofile(tile_zshiftsfile)
    else: print('\nWARNING - DISPLAY ONLY - NO FILES HAVE BEEN WRITTEN!\n\n')
    if show: plt.show()
  print('  total var: {:4.1f} -> {:4.1f}'.format(total_raw_var_sum, total_new_var_sum))
  print('  surface var: {:.1f}'.format(surface_var_sum))
  print('  finished processing', indir)
  print()

  return 0
  

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

    print(indir)
    res = calculate_best_focus(indir, nreg, ncy, gx, gy, w, h, z, ox, oy, snake, show=show)
    
    if res != 0:
      asdf

if __name__ == '__main__':
  dirs = []
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20190802_run12_preveh')
  #dirs.append('N:/CODEX raw/Mouse Sk Muscle/20200130_run22_long_preveh')
  #dirs.append('N:/Colin/codex_training_nov_19')
  #dirs.append('N:/CODEX raw/Human Muscle/20200622_human_run07_MTJ_E')
  dirs.append('N:/CODEX raw/Mouse Sk Muscle/20211209_VEGF_regen_run2')
  main(dirs, show=True)

free_libs([libCRISP, libc])












