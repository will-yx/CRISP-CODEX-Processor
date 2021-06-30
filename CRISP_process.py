import os
import toml
from glob import glob
from shutil import copyfile
from distutils.dir_util import copy_tree

from edge_alignment import main as edge_alignment
from calculate_tile_positions import main as calculate_tile_positions
from driftcomp_z import main as driftcomp_z
from calculate_best_focus import main as calculate_best_focus
from deconvolve import main as deconvolve
from driftcomp_xy import main as driftcomp_xy
from CRISP_register import main as CRISP_register
from CRISP_stitch import main as CRISP_stitch
from dice_mosaics import main as dice_mosaics

#####################################
#SET PARAMETERS
scratch_drive = 'X:/'
input_dir  = 'X:/temp/CODEX raw'
output_dir = 'X:/temp/CODEX processed'
#List of folder names for each multicycle
runs = ['20201008_test']
#####################################

for run in runs:
  indir     = os.path.join(input_dir, run)
  decondir  = os.path.join(scratch_drive, 'deconvolved', run)
  stitchdir = os.path.join(scratch_drive, 'stitched', run)
  finaldir  = os.path.join(output_dir, run)
  
  if not os.path.exists(indir):
    print("Error: input directory '{}' does not exists".format(indir))
    break
  if not os.path.exists(decondir): os.makedirs(decondir)
  if not os.path.exists(stitchdir): os.makedirs(stitchdir)
  if not os.path.exists(finaldir): os.makedirs(finaldir)
  
  config = toml.load(os.path.join(indir, 'CRISP_config.toml')) 
  
  #Step 1: Z-drift compensation and best-focus finding
  edge_alignment(indir, max_threads=1)
  calculate_tile_positions([indir])
  driftcomp_z(indir, max_threads=1)
  calculate_best_focus([indir])
  
  #Step 2: RL deconvolution
  deconvolve(indir, decondir, max_threads=2)
  
  #Step 3: XY-drift compensation
  driftcomp_xy(decondir, max_threads=4)
  
  #Step 4: Stitch
  CRISP_register(decondir, stitchdir)
  CRISP_stitch(decondir, stitchdir)
  
  #Step 5: Dice stitched mosaics for output
  dice_mosaics(stitchdir, finaldir, config, max_threads=8)

  break
  
  print("Copying config files to '{}'".format(finaldir))
  to_copy = []
  to_copy.extend(glob(os.path.join(indir, '*.xml')))
  to_copy.extend(glob(os.path.join(indir, '*.txt')))
  to_copy.extend(glob(os.path.join(indir, '*.json')))
  to_copy.extend(glob(os.path.join(indir, '*.toml')))
  for file in to_copy:
    copyfile(file, os.path.join(finaldir, os.path.basename(file)))
  
  print("Copying stitched images to '{}'".format(finaldir))
  copy_tree(stitchdir, os.path.join(finaldir, 'stitched'))
