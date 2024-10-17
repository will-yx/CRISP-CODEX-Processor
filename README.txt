CRISP CODEX Processor

**System Requirements**
x86 processor
32GB RAM (64GB+ recommended)
Nvidia CUDA-compatible Graphics Card (required)
	-1080Ti or 2080Ti with 11GB VRAM or 3060 with 12GB VRAM (3090+ with 24GB VRAM recommended)
	-Video RAM determines the maximum size and Z-stack size for processing
SSD or NAS are recommended for improved file read and write speeds
**use of a local 2TB+ NVMe SSD (PCIe4) as the scratch drive is highly recommended**
Windows 10+ or Windows Server 2016+ (64-bit)

**Installation**
Software Requirements
CUDA 11.2
https://developer.nvidia.com/cuda-toolkit-archive
**you can have concurrent versions of CUDA on your computer

vips 8.11.4
https://libvips.github.io/libvips/install.html
https://github.com/libvips/build-win64-mxe/releases/tag/v8.11.4
**make sure to add path to the bin folder to system environment variables

Visual Studio Community
https://visualstudio.microsoft.com/vs/community/

Microsoft Visual C++ Redistributable 2015-2019
https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads

Python 3.8+
https://www.python.org/

**Set NotePad or NotePad++ to default app to open .toml files**
**Currently CODEX MAV 1.3.0.308 is compatible (https://sites.imagej.net/Akoya/plugins/). Changes to the experiment.json file are needed for compatibility with later versions

We recommend using a conda virtual environment
1. install miniconda3
*note the installation path and make sure it is added to your systems environment variables
2. in a command prompt 
	conda create -n CRISP python=3.9
3. activate the CRISP environment
	conda activate CRISP
4. nagivate to the CRISP folder then use pip to install required python packages
	pip install -r requirements.txt
5. in NotePad or NotePad++ edit CRISP-CODEX-Processor/conda_activate_CRISP.bat to your miniconda installation path
6. OPTIONAL: edit CRISP-CODEX-Processor/max_threads.txt to optimally match GPU VRAM and imaging parameters

**Input Image File Folder Structure**
	Run directory
		|
		->Cycle 1 Region 1 images ["cyc001_reg001"]
		|	|
		|	-> Tiled stacked multichannel images (grayscale 16-bit) file name pattern ["*_00001_Z001_CH1.tif"]
		|		["*_00001_Z002_CH1.tif"] ... (single Z, single channel grayscale 16-bit images; naming formats can be modified in the CRISP_config.toml
		|
		->Cycle 1 Region 2 images ["cyc001_reg002"]
		|	|
		|	-> Tiled stacked multichannel images (grayscale 16-bit) file name pattern ["*_00001_Z001_CH1.tif"]
		|		["*_00001_Z002_CH1.tif"] ... 
		|
		...
		|
		->channelNames.txt
		->exposure_times.txt
		->CRISP_config.toml
		-> *opitonal* darkfield.tif
		-> *opitonal* flatfield.tif
		-> *opitonal* CODEX experiment config file.XML
		
**Experimental Configuration**
Copy and configure CRISP_config.toml to match experimental settings. 
(Optional) if experiment was performed on an Akoya CODEX instrument, copy CODEX template XML file to the folder containing the output images.
(Optional) generate darkfield and flatfield images for shade correction; see below for instructions.
	The provided example is for images captured on a Keyence microscope using: 
	- a 20x 0.75NA widefield, non-immersion (air) objective lens (Nikon)
	- high resolution settings
	- 30% image overlaps
	- PSF generation using the Gibson-Lanni method has been optimized by a global optimizer to calculate experimental values versus the theoretical values
	
	Edit the following sections:
		[dimensions] # experimental parameters
		width, height, slices, gx, gy, cycles, regions, overlap_x, overlap_y
	  
		[microscope] # microscope characteristics and settings
		magnification, NA, resXY, resZ
    
		[correction] # shading correction using experimental darkfield and flatfield images
		correct_darkfield, correct_flatfield
		
		[padding]			# pads Z-stacks for deconvolution, trims out of focus slices from top and bottom of stack
		zout 				# squashes imagestack down to zout slices
		ztrim1				# trim slices from the beginning of the stack
		ztrim2	 			# trim slices from the end of the stack
		
		[deconvolution]
		iterations
		host_obj_init = false 		# Slower! Store one of the arrays on the CPU to conserve VRAM; ONLY USE IF NOT ENOUGH GPU VRAM
		
		[extended_depth_of_field]
		enabled = true            # compute EDF image; default true
		save_zstack = false        # save non-EDF images; for 2D analysis with a single Z plane or 3D analysis with higher number of output Z slices
		register_edf = true      # use EDF slice for registration when stitching; default true
		# defaults registers and saves EDF images only, save_zstack to true if analyzing in 3D or imaging thicker or high density tissues
		
		[dynamic_range_compression]
		drc0, drc1
		
		[background_subtraction]
		 # perform background subtraction using up to 3 cycles
		 # cycles are specified as 1-indexed: 1 is the first cycle, -1 is the last cycle, -2 is the penultimate cycle, etc.
		blankcycles = [2, -3, -2] # indicate which cycles are blank cycles; set to [] to disable background subtraction
		 # optimal imaging set up:
		 # [1] Bleach cycle, [2] 1st blank cycle (max exposure times), marker cycles..., end blank cycles ( [-2] max exposure only, or [-3, -2] min and max exposures), [-1] non-CODEX stains
		 # consistent exposure times across cycles with max-max [2, -2] background subtraction yields the best results, staining intensity should be determined by titering antibodies

*** Background subtraction and exposure time for cycles: if imaging was performed on an Akoya CODEX instrument, import the XML file from the configuration. 
If not, please supply a text file as exposure_times.txt in the following format:

Cycle	CH1	CH2	CH3	CH4
1	5	33.333	40	66.667
2	5	333.333	500	250
...

**3 ways to run**
1. With a GUI window:
	Run CRISP_CODEX.bat 
	**you may need to edit CRISP-CODEX-Processor/conda_activate_CRISP.bat to point to the correct miniconda path
	Select input, output, and scratch drive folders
	Create and edit the config file
	*If reprocessing or re-starting after an error, uncheck steps that already successfully completed*
	Click "Start..."
	
2. In jupyter notebook: 
	make sure CRISP_config.toml file is properly configured in the input directory 
	>import CRISP_process_GUI as CRISP
	>CRISP.process(input_path, output_path, scratch_dir)

3. In a python IDE (IDLE):
	Edit CRISP_process.py
	- point scratch_drive to a scratch drive with sufficient space (Deconvolved images and full-resolution stitched montages will be saved here; ideally this is a 2TB NVME drive)
	- point input_path to the parent folder of a multicycle run
	- point output_path to a folder for final processed images (this can be on a NAS, images will be saved similar to the output format of the Akoya CODEX processor)
	save edits to CRISP_process.py
	run module (F5)
	
IMPORTANT: Current parallel processing is threaded for graphics cards with 11GB of VRAM. If you have less graphics memory, edit function calls in CRISP_process or CRISP_process_GUI to use fewer threads.

**Darkfield and Flatfield correction**
To generate darkfield images of sensor noise:
1. Match capture settings to your experiment (i.e. "High Resolution" "1x1 binning" "12dB gain") 
2. Reduce excitation light to as low as possible
3. Use a relatively short exposure time (25ms)
4. Capture 100-500 images (multipoint or 3D capture) without slides or samples
5. Use ImageJ/FIJI to generate a median signal from these images
6. Save this median to the raw images folder
- darkfield will not likely change between experiments, however will change if different sensor settings are used (i.e. different binning or sensor gain)

To generate flatfield images:
1. Add small amounts of FITC or fluorescent dye in between two coverslips or on a slide, making sure it is evenly distributed
2. Place onto the microscope
3. Adjust exposure settings by monitoring the image histogram so that the signal is not saturated
4. Capture 100-500 images (multipoint or 3D capture) at various positions on the sample holder
5. Use ImageJ/FIJI to generate a median signal from these images
6. Save this median to the raw images folder
- flatfield can be different in each channel and uneven illumination can occur with sample chamber orientation, this correction is limited to different signal intensity caused by lens abberations or microscope alignment
- significant uneven illumination after this step may require preprocessing such as rolling circle background subtraction prior to processing

Release notes:
v0.7
-updates for compatibility with 3 slices
-automatically determine threads for deconvolution
-updated default params in config for increased throughput and EDF output
-updated README
v0.6
-recompiled with CUDA 11.2
-updated DLL loading for Python 3.7+
-added step to update experiment.json files to be compatible with Akoya MAV
-fixed an error with background subtraction using early and late cycles to account for photobleaching
v0.5.2
-changed TIFF compression from LZW to deflate
-added temp file to load previous settings
-added setting file to externally set number of max threads for each process
v0.5.1
-fixed bug in output directory path
-added checkbox options for skipping already performed steps
v0.5.0a
-minor fixes to reading configuration files from XML
