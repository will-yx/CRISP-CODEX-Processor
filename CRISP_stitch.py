import os
import toml
import numpy as np
import subprocess
from time import sleep
from timeit import default_timer as timer

def unique(sequence):
  seen = set()
  for value in sequence:
    if value in seen: continue
    seen.add(value)
    yield value

def stitch_region(indir, outdir, reg, params, gx, gy, ox, oy, ncy, nc, nzout, zregister, blankstring):
  ha = params[0]
  hb = params[1]
  hc = params[2]
  hd = params[3]
  he = params[4]
  
  o3 = params[5]
  o4 = params[6]
  o5 = params[7]
  o6 = params[8]
  o9 = params[9]
  o0 = params[10]
  
  pH = params[11]
  pJ = params[12]
  
  p1 = params[13]
  p2 = params[14]
  p3 = params[15]
  p4 = params[16]

  ax  = [0.0, 0.0, 0.0, 0.0]
  ay  = [0.0, 0.0, 0.0, 0.0]

  env = dict(os.environ)
  env.update({'VIPS_WARNING': '0'})
  
  command = 'CRISP_stitch.exe --nc {} --ncy {} --gx {} --gy {} -r {}'.format(nc, ncy, gx, gy, reg)
  command += ' --ox {:.4f} --oy {:.4f}'.format(ox, oy)
  command += ' -i "{}" -o "{}" --zr 0 --quiet'.format(indir, outdir)
  command += blankstring
  command += ' --update-ca --blend0'
  command += ''.join([' -a {:.4f} -a {:.4f}'.format(x, y) for x, y in zip(ax, ay)])
  command += ' --ha {} --hb {} --hc {} --hd {} --he {}'.format(ha,hb,hc,hd,he)
  command += ' -3 {} -4 {} -5 {:d} -6 {} -9 {:d} -0 {:d}'.format(o3,o4,o5,o6,o9,o0)
  command += ' -H {} -J {} -L {} -T {} -U 0.0 -W {}'.format(pH,pJ, 3, 2, 0.25)
  command += ' -! {} -@ {} -# {} -$ {}'.format(p1,p2,p3,p4)
  
  errsum = 0.0
  z1 = zregister - (nzout >> 1)
  z2 = zregister + (nzout >> 1)
  slices = list(unique([0, zregister] + list(range(z1, z2+1))))
  
  print('Stitching slices:', slices)
  for z in slices:
    print("Stitching region {}, slice {}".format(reg, z))
    stitch_command = command + ' --load --save -z {}'.format(z)
    print("Executing command: '{}'".format(stitch_command))
    t0 = timer()
    with subprocess.Popen(stitch_command, env=env) as proc:
      v = proc.wait()
    t1 = timer()
    print('Elapsed: {:.0f}s'.format(t1-t0))
    print('Result: {}'.format(v))
  
  return errsum


def stitch_main(indir, outdir, params):
  t0 = timer()
  
  print("Stitching directory {}".format(indir))
  rawscores = []
  name = os.path.basename(os.path.normpath(indir))
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml'))
  
  zout = config['padding']['zout']
  ox   = config['dimensions']['overlap_x']
  oy   = config['dimensions']['overlap_y']
  gx   = config['dimensions']['gx']
  gy   = config['dimensions']['gy']
  ncy  = config['dimensions']['cycles']
  nreg = config['dimensions']['regions']
  nc = len(config['microscope']['channelNames'])
  
  dcdir = os.path.join(indir, 'driftcomp')
  if ncy > 1 and not os.path.isdir(dcdir):
    print("Error: driftcomp directory '{}' not found!".format(dcdir))
    return 1
  
  if config.get('background_subtraction'):
    blankcycles = config['background_subtraction']['blankcycles']
    blankstring = ''.join([' --blank {}'.format(c) for c in blankcycles])
  else:
    blankstring = ''
  
  nzout = (zout - 2 - 2) if zout > 8 else zout
  zregister = (zout+1) >> 1
  
  if config.get(extended_depth_of_field):
    if config['extended_depth_of_field'].get('enabled', True) and not config['extended_depth_of_field'].get('save_zstack', True):
      nzout = 1
      zregister = 0
  
  if not os.path.exists(outdir): os.makedirs(outdir)
  for reg in range(1,1+nreg):
    if not os.path.isfile(os.path.join(outdir, 'region{:03d}_registration.bin'.format(reg))): continue
    err = stitch_region(indir, outdir, reg, params, gx, gy, ox, oy, ncy, nc, nzout, zregister, blankstring)
    rawscores.append(err)
    print('Score: {}'.format(err))
  
  print('Raw errors for run {}:'.format(os.path.basename(os.path.normpath(indir))))
  print([int(s) for s in rawscores])
  
  t1 = timer()
  
  print('\n\n')
  print('Elapsed time: {:.1f}s'.format(t1-t0))
  print('\n\n')
  
  return 0


##########

def main(indir, outdir):
  params = [40, 0.80, 0.50, 0.200, 1.0, 1.00, 4.00, 100, 1.0, 7, 1, 0.05, 0.5, 0.2, 0.03, 0.4, 0.04]
  return stitch_main(indir, outdir, params)

if __name__ == '__main__':
  indir = 'X:/20190802_run12_preveh_decon'
  outdir = 'X:/stitched/20190802_run12_preveh'
  outdir = 'X:/stitched/20211209_VEGF_regen_run2_RAW_d333_f333b'
  main(indir, outdir)
 





