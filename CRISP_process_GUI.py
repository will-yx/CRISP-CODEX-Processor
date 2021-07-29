import os
import toml
import numpy as np
from glob import glob
from shutil import copyfile
from distutils.dir_util import copy_tree

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from edge_alignment import main as edge_alignment
from calculate_tile_positions import main as calculate_tile_positions
from driftcomp_z import main as driftcomp_z
from calculate_best_focus import main as calculate_best_focus
from deconvolve import main as deconvolve
from driftcomp_xy import main as driftcomp_xy
from CRISP_register import main as CRISP_register
from CRISP_stitch import main as CRISP_stitch
from dice_mosaics import main as dice_mosaics
from updateJSON import updateJSON

class Checkbar(Frame):
  def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
    Frame.__init__(self, parent)
    self.vars = []
    for pick in picks:
      var = IntVar()
      chk = Checkbutton(self, text=pick, variable=var)
      chk.pack(side=side, anchor=anchor, expand=YES)
      self.vars.append(var)
  def setall(self):
    for child in self.winfo_children():
      child.select()
  def state(self):
    return map((lambda var: var.get()), self.vars)

class MainWindow(Tk):
  def __init__(self, parent):
    Tk.__init__(self, parent)
    self.parent = parent
    self.setup()

    #self.grid(column=0, row=0, sticky=(N, W, E, S))
    self.columnconfigure(0, weight=1)
    self.rowconfigure(0, weight=1)
  
  def setup(self):
    self.title('CRISP: Multiplex Imaging Processor')

    self.frame = ttk.Frame(self, padding='10 30 10 10')
    self.frame.grid(column=0, row=0) #sticky=(N, W, E, S)
    self.columnconfigure(0, weight=10)
    self.rowconfigure(0, weight=10)

    input_dir = StringVar()
    output_dir = StringVar()
    scratch_dir = StringVar()
    config_file = StringVar()

    ttk.Label(self.frame, text="Input folder").grid(column=1, row=1, sticky=(E), pady=(0, 10))
    ttk.Button(self.frame, text=" ... ", command=lambda:self.selectfolder(input_dir)).grid(column=2, row=1, sticky=W, pady=(0, 10))
    ttk.Label(self.frame, textvariable=input_dir).grid(column=3, row=1, sticky=(W), pady=(0, 10))

    ttk.Label(self.frame, text="Output folder").grid(column=1, row=4, sticky=(E), pady=(0, 10))
    ttk.Button(self.frame, text=" ... ", command=lambda:self.selectfolder(output_dir)).grid(column=2, row=4, sticky=W, pady=(0, 10))
    ttk.Label(self.frame, textvariable=output_dir).grid(column=3, row=4, sticky=(W), pady=(0, 10))

    ttk.Label(self.frame, text="Temporary files folder (scratch drive)").grid(column=1, row=7, sticky=(E), pady=(0, 10))
    ttk.Button(self.frame, text=" ... ", command=lambda:self.selectfolder(scratch_dir)).grid(column=2, row=7, sticky=W, pady=(0, 10))
    ttk.Label(self.frame, textvariable=scratch_dir).grid(column=3, row=7, sticky=(W), pady=(0, 10))

    ttk.Label(self.frame, text="Image Config File (CRISP_config.toml)").grid(column=1, row=8, sticky=(E))
    ttk.Button(self.frame, text="Create from template", command=lambda:self.makeconfig(input_dir, config_file)).grid(column=2, row=8, sticky=W)
    ttk.Button(self.frame, text="Edit config file", command=lambda:self.editconfig(input_dir, config_file)).grid(column=3, row=8, sticky=W)
    ttk.Label(self.frame, textvariable=config_file).grid(column=2, row=9, columnspan=3, sticky=(W), pady=(0, 10))
    
    if os.path.exists('CRISP_GUI.tmp.npy'):
      prev_settings = np.load('CRISP_GUI.tmp.npy')
      print('Previous settings loaded')
      input_dir.set(prev_settings[0])
      output_dir.set(prev_settings[1])
      scratch_dir.set(prev_settings[2])
    
    ttk.Label(self.frame, text="Step 1: Drift compensation and best focus determination").grid(column=1, row=9, sticky=(E), pady=(0, 0))
    option1 = Checkbar(self.frame, ['Tile Alignment', '3D-drift compensation', 'Find Best Focus'])
    option1.grid(column=2, row=9, sticky=W, columnspan=2, pady=(0, 0))
    option1.setall()
    
    ttk.Label(self.frame, text="Step 2: Richarson-Lucy Deconvolution").grid(column=1, row=10, sticky=(E), pady=(0, 0))
    option2 = Checkbar(self.frame, ['Deconvolution'])
    option2.grid(column=2, row=10, sticky=W, columnspan=2, pady=(0, 0))
    option2.setall()
    
    ttk.Label(self.frame, text="Step 3: Opitmize X-Y drift comp").grid(column=1, row=11, sticky=(E), pady=(0, 0))
    option3 = Checkbar(self.frame, ['X-Y drift compensation'])
    option3.grid(column=2, row=11, sticky=W, columnspan=2, pady=(0, 0))
    option3.setall()
    
    ttk.Label(self.frame, text="Step 4: Register and stitch").grid(column=1, row=12, sticky=(E), pady=(0, 0))
    option4 = Checkbar(self.frame, ['Register and Stitch'])
    option4.grid(column=2, row=12, sticky=W, columnspan=2, pady=(0, 0))
    option4.setall()
    
    ttk.Label(self.frame, text="Step 5: Dice montage for output").grid(column=1, row=13, sticky=(E), pady=(0, 20))
    option5 = Checkbar(self.frame, ['Dice'])
    option5.grid(column=2, row=13, sticky=W, columnspan=2, pady=(0, 20))
    option5.setall()

    self.btnDeconvolve = ttk.Button(self.frame, text="Start...", command=lambda:self.process(input_dir.get(), output_dir.get(), scratch_dir.get(), [list(option1.state()),list(option2.state()),list(option3.state()),list(option4.state()),list(option5.state())]))
    self.btnDeconvolve.grid(column=2, row=15, sticky=E)
    
    self.btnDeconvolve = ttk.Button(self.frame, text="Help", command=lambda:self.readme())
    self.btnDeconvolve.grid(column=4, row=15, sticky=E)
    
  def selectfolder(self, x, count=0):
    x.set(filedialog.askdirectory())
    if count:
      count.set(len(glob.glob(x.get()+'/*.tif')))
      self.logger(count.get()+' tif files found')
    
  def makeconfig(self, indir, config):
    if indir.get():
      if os.path.exists(indir.get()+'/CRISP_config.toml'):
        MsgBox = messagebox.askquestion("Warning","CRISP_config.toml exists in the output directory \n Overwite?")
        if MsgBox == 'yes':
          copyfile('Configuration files/CRISP_config.toml', indir.get()+'/CRISP_config.toml')
          config.set(indir.get()+'/CRISP_config.toml')
          messagebox.showinfo('Template config file copied','Edit config file to match experiment parameters.')
        else:
          messagebox.showinfo('Message','Not overwritten')
      else:
        copyfile('Configuration files/CRISP_config.toml', indir.get()+'/CRISP_config.toml')
        config.set(indir.get()+'/CRISP_config.toml')
        messagebox.showinfo('Template config file copied','Edit config file to match experiment parameters.')
    else: messagebox.showinfo('Warning','Select input folder')

  def editconfig(self, path, config):
    if path.get():
      if os.path.exists(path.get()+'/CRISP_config.toml'):
        config.set(path.get()+'/CRISP_config.toml')
        os.startfile(path.get()+'/CRISP_config.toml')
      else: messagebox.showinfo('CRISP_config.toml not found in Input folder.','Click "Create from Template" or copy config file to the input folder.')
    else: messagebox.showinfo('Warning','Select input folder')
    
  def readme(self):
    if os.path.exists("README.txt"):
      os.startfile('README.txt')
    else: messagebox.showinfo('Error','Could not load Read Me file')

  def progress_update(self):
    self.progress_bar['value'] += 1
    self.update()

  def logger(self, string):
    self.log.insert(END, string+'\n')
    self.log.see(END)
    self.update()
  
  def process(self, input_path, output_path, scratch_drive, options):
    np.save('CRISP_GUI.tmp',[input_path,output_path,scratch_drive])
    if input_path.endswith('/'):
      run = input_path.split('/')[-2]
    else: run = input_path.split('/')[-1]
    indir = input_path
    decondir = scratch_drive + '/deconvolved/' + run
    stitchdir = scratch_drive + '/stitched/' + run
    finaldir = output_path + '/' + run
    
    '''
    def disableButtons(self):
      for child in self.frame.winfo_children():
        if 'button' in str(child):
          child.configure(state='disabled')
  
    def enableButtons(self):
      for child in self.frame.winfo_children():
        if 'button' in str(child):
          child.configure(state='enable')
    '''
    
    if not os.path.exists(indir):
      raise("Error: input directory '{}' does not exists".format(indir))
    if not os.path.exists(decondir): os.makedirs(decondir)
    if not os.path.exists(stitchdir): os.makedirs(stitchdir)
    if not os.path.exists(finaldir): os.makedirs(finaldir)

    config = toml.load(os.path.join(indir, 'CRISP_config.toml'))
    try:
      threading = dict((x.strip(),y.strip()) for x,y in (line[:-1].split('=') for line in open('max_threads.txt').readlines() if line.startswith('max_threads')))
      max_threads_edge_alignment=int(threading['max_threads_edge_alignment'])
      max_threads_z_driftcomp=int(threading['max_threads_z_driftcomp'])
      max_threads_deconvolution=int(threading['max_threads_deconvolution'])
      max_threads_xy_driftcomp=int(threading['max_threads_xy_driftcomp'])
      max_threads_dicing=int(threading['max_threads_dicing'])
      print(max_threads_edge_alignment,max_threads_z_driftcomp,max_threads_deconvolution,max_threads_xy_driftcomp,max_threads_dicing)
    except:
      print('max_thread.txt not found... continuing with default parallelization settings')
      max_threads_edge_alignment=1
      max_threads_z_driftcomp=1
      max_threads_deconvolution=2
      max_threads_xy_driftcomp=4
      max_threads_dicing=8

    #Step 1: Z-drift compensation and best-focus finding
    if options[0][0] == 1:
      print('Performing edge aligment:')
      edge_alignment(indir, max_threads=max_threads_edge_alignment)
      print('Optimizing tile positions:')
      calculate_tile_positions([indir])
    if options[0][1]  == 1:
      print('Performing Z-drift compensation:')
      driftcomp_z(indir, max_threads=max_threads_z_driftcomp)
    if options[0][2]  == 1:
      print('Finding best focused Z:')
      calculate_best_focus([indir])

    #Step 2: RL Deconvolution
    if options[1][0] == 1:
      print('Running Deconvolution:')
      deconvolve(indir, decondir, max_threads=max_threads_deconvolution)

    #Step 3: XY-drift compensation
    if options[2][0] == 1:
      print('Optimizing X-Y drift compensation:')
      driftcomp_xy(decondir, max_threads=max_threads_xy_driftcomp)

    #Step 4: Stitch
    if options[3][0] == 1:
      print('Registering image positions:')
      CRISP_register(decondir, stitchdir)
      print('Stitching montages:')
      CRISP_stitch(decondir, stitchdir)
      print("Copying stitched images to '{}'".format(finaldir))
      copy_tree(stitchdir, os.path.join(finaldir, 'stitched'))
    
    #Step 5: Dice stitched mosaics for output
    if options[4][0] == 1:
      print('Dicing output images:')
      dice_mosaics(stitchdir, finaldir, config, max_threads=max_threads_dicing)
      print("Dicing complete...")

    #Update experiment.json for MAV
    updateJSON(indir, os.path.join(indir, 'CRISP_config.toml'))
    
    print("Copying config files to '{}'".format(finaldir))
    to_copy = []
    to_copy.extend(glob(os.path.join(indir, '*.xml')))
    to_copy.extend(glob(os.path.join(indir, '*.txt')))
    to_copy.extend(glob(os.path.join(indir, '*.json')))
    to_copy.extend(glob(os.path.join(indir, '*.toml')))
    for file in to_copy:
      copyfile(file, os.path.join(finaldir, os.path.basename(file)))

    print("CRISP Process complete!")

    for child in self.frame.winfo_children():
        if 'button' in str(child):
          child.configure(state='normal')

def GUI():
  root = MainWindow(None)
  root.mainloop()

if __name__=='__main__':
  GUI()
