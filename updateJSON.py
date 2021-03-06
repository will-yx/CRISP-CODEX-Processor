import os, datetime, toml, json
import pandas as pd

def updateJSON(indir, toml_file):
    indir = os.path.normpath(indir)
    exp_name = os.path.basename(indir)
    CRISPtoml = toml.load(open(toml_file))
    channelnames_file = os.path.join(indir, 'channelnames.txt')
    exposuretimes_file = os.path.join(indir, 'exposure_times.txt')
    expjson_file = os.path.join(indir, 'experiment.json')
    if os.path.isfile(expjson_file) and not os.path.isfile(os.path.join(indir, 'experiment_bkup.json')):
        os.rename(expjson_file, os.path.join(indir, 'experiment_bkup.json'))
    
    json_dict = {}
    json_dict['version'] = 'CRISP'
    json_dict['name'] = exp_name
    json_dict['dateProcessed'] = str(datetime.datetime.now())
    json_dict['path'] = indir

    #dimensions
    json_dict['numRegions'] = CRISPtoml['dimensions']['regions']
    json_dict['numCycles'] = CRISPtoml['dimensions']['cycles']
    json_dict['numTiles'] = CRISPtoml['dimensions']['gx']*CRISPtoml['dimensions']['gy']
    json_dict['numZPlanes'] = CRISPtoml['padding']['zout']
    json_dict['numOriginalPlanes'] = CRISPtoml['dimensions']['slices']
    json_dict['numChannels'] = len(CRISPtoml['microscope']['channelNames'])
    json_dict['regionWidth'] = CRISPtoml['dimensions']['gx']
    json_dict['regionHeight'] = CRISPtoml['dimensions']['gy']
    json_dict['tileWidth'] = CRISPtoml['dimensions']['width']
    json_dict['tileHeight'] = CRISPtoml['dimensions']['height']
    json_dict['tileOverlapX'] = CRISPtoml['dimensions']['overlap_x']
    json_dict['tileOverlapY'] = CRISPtoml['dimensions']['overlap_y']
    
    #processing settings
    json_dict['tilingMode'] = 'snakeRows'
    json_dict['backgroundSubtractionMode'] = list(CRISPtoml['background_subtraction'].keys())[0]
        ## some settings are hard coded because they do not apply to CRISP
    json_dict['driftCompReferenceCycle'] = 0
    json_dict['driftCompReferenceChannel'] = 0 
    json_dict['bestFocusReferenceCycle'] = 0
    json_dict['bestFocusReferenceChannel'] = 0
    json_dict['numSubTiles'] = 0
    json_dict['focusingOffset'] = 0
    json_dict['useBackgroundSubtraction'] = bool(CRISPtoml['background_subtraction'])
    json_dict['useShadingCorrection'] = any([CRISPtoml['correction']['correct_darkfield'],CRISPtoml['correction']['correct_flatfield']])
    json_dict['use3dDriftCompensation'] = True
    json_dict['useBleachMinimizingCrop'] = False
    json_dict['useBlindDeconvolution'] = CRISPtoml['deconvolution']['blind']
    json_dict['useDiagnosticMode'] = False
    json_dict['multipointMode'] = False
    json_dict['HandEstain'] = False
    
    #channelNames
    channelNames = pd.read_csv(channelnames_file, sep='\t', header=None)
    json_dict['channelNames'] = {"channelNamesArray":channelNames.values.tolist()}
    
    #exposure times
    exp_times = pd.read_csv(exposuretimes_file, sep=',', header=None)
    json_dict['exposureTimes'] = {"exposureTimesArray":exp_times.values.tolist()}
    
    #microscope settings
    json_dict['numerical_aperture'] = CRISPtoml['microscope']['NA']
    json_dict['per_pixel_XY_resolution'] = CRISPtoml['microscope']['resXY']*1000
    json_dict['z_pitch'] = CRISPtoml['microscope']['resZ']*1000
    json_dict['focusing_offset'] = 0
    json_dict['tiling_mode'] = 'snake'
    json_dict['deconvolution'] = 'CRISP_{}'.format(CRISPtoml['deconvolution']['iterations'])
    
    #channel, region indices
    json_dict['channel_names'] = ['CH{}'.format(i+1) for i in range(json_dict['numChannels'])]
    json_dict['region_names'] = ['reg{}'.format(i+1) for i in range(json_dict['numRegions'])]
    json_dict['regIdx'] = [i+1 for i in range(json_dict['numRegions'])]
    
    #redundant info
    json_dict['cycle_lower_limit'] = 1
    json_dict['cycle_upper_limit'] = CRISPtoml['dimensions']['cycles']
    json_dict['num_cycles'] = CRISPtoml['dimensions']['cycles']
    json_dict['region_width'] = CRISPtoml['dimensions']['gx']
    json_dict['region_height'] = CRISPtoml['dimensions']['gy']
    json_dict['num_z_planes'] = CRISPtoml['padding']['zout']
    json_dict['tile_width'] = int(CRISPtoml['dimensions']['width']-(CRISPtoml['dimensions']['width']*CRISPtoml['dimensions']['overlap_x']))
    json_dict['tile_height'] = int(CRISPtoml['dimensions']['height']-(CRISPtoml['dimensions']['height']*CRISPtoml['dimensions']['overlap_y']))
    json_dict['tile_overlap_X'] = int(CRISPtoml['dimensions']['width']*CRISPtoml['dimensions']['overlap_x'])
    json_dict['tile_overlap_Y'] = int(CRISPtoml['dimensions']['height']*CRISPtoml['dimensions']['overlap_y'])
    json_dict['drift_comp_channel'] = 0
    json_dict['driftCompReference'] = 0
    json_dict['best_focus_channel'] = 1
    json_dict['bgsub'] = bool(CRISPtoml['background_subtraction'])
    
    with open(expjson_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent = 4)

if __name__ == '__main__':
    indir = "FILEPATH"
    toml_file = os.path.join(indir, 'CRISP_config.toml')
    updateJSON(indir, toml_file)