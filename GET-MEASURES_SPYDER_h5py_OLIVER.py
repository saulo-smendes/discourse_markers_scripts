# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:42:54 2022

@author: Saulo Mendes Santos
"""

from __future__ import print_function
import sys
import subprocess
import pkg_resources

required = {'audiofile', 'opensmile', 'h5py', 'soundfile', 'praat-parselmouth', 'tgt', 'pysptk', 'librosa', 'pyworld', 'helper', 'scikit-fda', 'matlab'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

print("Update done!")

#!pip install -U cupy

#Import modules
import h5py
import os
import numpy as np
import pandas as pd
import audiofile
import parselmouth
import statistics
import matplotlib.pyplot as plt
#import seaborn as sns
import soundfile
#import helper
#import pdas
#import maps_f0
import pathlib
#from tqdm.notebook import tqdm
import matlab
import parselmouth
from parselmouth.praat import call
import numpy as np
#import statistics
import matplotlib.pyplot as plt
#import seaborn as sns
import librosa
import audiofile
import matlab
import matlab.engine
import audiofile
import opensmile
import resampy
import os
import uuid
import pathlib
import numpy
import resampy
import soundfile
import datetime
import sys
from librosa.feature import mfcc
import librosa
import matplotlib.pyplot as plt
import librosa.display
import scipy

print("Importing libraries done!")

from os.path import exists
#!jupyter labextension install @jupyter-widgets/jupyterlab-manager
#!jupyter nbextension enable --py widgetsnbextension

#### GLOBALS ####
WD = os.path.normpath('/home/mendessantos/GET-MEASURES')
PATH = os.path.normpath('/home/mendessantos/todos/')
#RESULT_FILE = os.path.normpath('/srv/tannatdata/saulo/Corpus/RESULT_TODOS_20231114_v2_teste.hdf5')
RESULT_FILE = os.path.normpath('/home/mendessantos/RESULT_TODOS_20231114_v2_teste.hdf5')
VERIFY = True
#corpus='APLAWDW'
os.chdir(WD)
error_log_name = "/srv/tannatdata/saulo/Corpus/error_log_todos.txt"
unprocessed_log_name = "/srv/tannatdata/saulo/Corpus/unprocessed_log_todos.txt"



print("Globals definition done!")

#### FUNCTION DEFINITION ####

#Import modules
# Sox is actually not required so don't worry if Python don't find it

#Set pandas to display max 4 columns
#pd.set_option('display.max_columns', 4)

"""
F0 via ACF and SHS methods
Probability of Voicing
"""

# Define a function to write a error log file

def write_error(path, file_name, error, algo, log_name=error_log_name):
    # Get current date time
    current_time = datetime.datetime.now()
    # Write error in log file
    with open(log_name, 'a+') as f:
        f.writelines(str(file_name) + "\t" + str(algo) + "\t" + str(current_time) + "\t" + str(os.path.join(path, file_name)) + "\t" + str(error) + "\n")

def check_extraction(var, length, dset, log_name=unprocessed_log_name):
    current_time = datetime.datetime.now()
    if len(var) > length:
        dset.attrs.modify("processed", value=True)
    else:
        dset.attrs.modify("processed", value=False)
        with open(log_name, 'a+') as f:
            f.writelines(str(dset.name) + "\t" + str(current_time) + str("\n"))

#### OPENSMILE ####
def get_opensmile(path, hf, verbose=True, verify=True):
    #Instantiate feature extractor eGeMAPSv02
    #?Build a for loop
    #P?enser comment prendre les mesures des différents features extractors
    #feature_set=opensmile.FeatureSet.ComParE_2016 (6.5k features)
    #Get measures for the whole file
    #smile = opensmile.Smile(
    #    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    #    feature_level=opensmile.FeatureLevel.Functionals,
    #    num_channels=1
    #    )
    #smile.feature_names

    #create a feature extractor for low-level descriptors (LLDs) - eGeMAPSv02
    smile_lld_basic = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors#,
        #channels=1 # Change from num_channels
        )

    smile_lld_complete = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors#,
        #channels=1 # Change from num_channels
        )

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                file_name = file.replace('.wav', '')
                signal, sampling_rate = audiofile.read(os.path.join(root, file))


                features_lld_complete = smile_lld_complete.process_signal(
                    signal,
                    sampling_rate
                    )

                features_lld_complete.reset_index(level=['start', 'end'], inplace=True)

                # Only create new group if it does not exist in arborescence
                root_group = os.path.join(root, file_name)
                if str(root_group) not in hf:
                    grp = hf.create_group(os.path.join(root, file_name))
                else:
                    grp = hf[str(root_group)]

                # And we only run the extractior if the dataset is not in the group:
                if "opensmile" not in grp:
                    try:
                        signal, sampling_rate = audiofile.read(os.path.join(root, file))
                        features_lld_complete = smile_lld_complete.process_signal(signal, sampling_rate)
                        features_lld_complete.reset_index(level=['start', 'end'], inplace=True)
                        end = np.asarray(features_lld_complete['end'].dt.total_seconds())
                        start = np.asarray(features_lld_complete['start'].dt.total_seconds())
                        timeframes = (start + end) / 2
                        f0 = features_lld_complete['F0final_sma']

                        # Save a result txt file on the same folder of audio
                        #results_path = os.path.join(root, file_name) + '_opensmile_results.txt'
                        var1 = np.array(timeframes)
                        var2 = np.array(f0)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("opensmile", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)

                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="opensmile", log_name=error_log_name)
                        continue
                else:
                    print("Dataset opensmile is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "opensmile"))]
                        check_extraction(var=dset[0], length=5, dset=dset)
                    #with open(results_path, 'a+') as f:
                    #    f.writelines('timeframe' + '\t' + 'f0')
                    #    for i in range(len(f0)):
                    #        f.writelines('\n' + str(round(timeframes[i], 3)) + '\t' + str(round(f0[i], 2)))

                if verbose:
                    print("File {0} done for Opensmile".format(file))

#get_opensmile(PATH)

#Define max and min f0 for pitch object following Hirst
def hirst_procedure(pitch_object, f0min=40, f0max=800, unit="Hertz"):
    q25 = call(pitch_object, "Get quantile", 0, 0, 0.25, unit)
    q75 = call(pitch_object, "Get quantile", 0, 0, 0.75, unit)
    minf0 = (q25 * 0.75/10)*10
    maxf0 = (q75 * 1.5/10)*10
    # Edit to avoid having maxf0 or minf0 == 0
    if (maxf0 == 0 or np.isnan(maxf0)):
        maxf0 = 800
    if (minf0 == 0 or np.isnan(minf0)):
        minf0 = 40
    return minf0, maxf0

#Get pitch default
def get_pitch(voiceID, f0min=40, f0max=800, unit="Hertz", hirst=True):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.005, f0min, f0max) #create a praat pitch object
    if hirst:
        f0min_hirst, f0max_hirst = hirst_procedure(pitch)
        _ = call(pitch, "Remove")
        pitch = call(sound, "To Pitch", 0.005, f0min_hirst, f0max_hirst) #create a praat pitch object
    frame_times = call(pitch, "List all frame times")
    f0_values = call(pitch, "List values in all frames", unit)
    _ = call(pitch, "Remove")
    return frame_times, f0_values

#Get pitch auto-correlation
def get_pitch_ac(voiceID, f0min=40, f0max=800, unit="Hertz", hirst=True):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch (ac)", 0.005, f0min, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f0max) #create a praat pitch object
    if hirst:
        f0min_hirst, f0max_hirst = hirst_procedure(pitch)
        _ = call(pitch, "Remove")
        pitch = call(sound, "To Pitch (ac)", 0.005, f0min_hirst, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f0max_hirst) #create a praat pitch object
    frame_times = call(pitch, "List all frame times")
    f0_values = call(pitch, "List values in all frames", unit)
    _ = call(pitch, "Remove")
    return frame_times, f0_values

#Get pitch cross-correlation
def get_pitch_cc(voiceID, f0min=40, f0max=800, unit="Hertz", hirst=True):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0.005, f0min, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f0max) #create a praat pitch object
    if hirst:
        f0min_hirst, f0max_hirst = hirst_procedure(pitch)
        _ = call(pitch, "Remove")
        pitch = call(sound, "To Pitch (cc)", 0.005, f0min_hirst, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f0max_hirst) #create a praat pitch object
    frame_times = call(pitch, "List all frame times")
    f0_values = call(pitch, "List values in all frames", unit)
    _ = call(pitch, "Remove")
    return frame_times, f0_values

#Get pitch SPINET
def get_pitch_spinet(voiceID, f0max=800, unit="Hertz"):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch (SPINET)", 0.005, 0.04, 70, 5000, 250, f0max, 15)
    frame_times = call(pitch, "List all frame times")
    f0_values = call(pitch, "List values in all frames", unit)
    _ = call(pitch, "Remove")
    return frame_times, f0_values

#Get pitch Sub-harmonic summation
def get_pitch_shs(voiceID, f0max=800, unit="Hertz"):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch (shs)", 0.005, 50, 15, 1250, 15, 0.84, f0max, 48)
    frame_times = call(pitch, "List all frame times")
    f0_values = call(pitch, "List values in all frames", unit)
    _ = call(pitch, "Remove")
    return frame_times, f0_values

# Get measures of Harmonicity, Intensity, Spectral emphasis ?

# Get measures of Harmonicity, Intensity, Spectral emphasis ?

#Get Intensity
def get_intensity(voiceID, f0min=70, subtract_mean="yes"):
    sound = parselmouth.Sound(voiceID) # read the sound
    intensity = call(sound, "To Intensity", f0min, 0.005, subtract_mean)
    frame_times = call(intensity, "List all frame times")
    intensity_values = []
    for timestep in frame_times:
        intensity_value = call(intensity, "Get value at time...", timestep, 'cubic')
        intensity_values.append(intensity_value)
    _ = call(intensity, "Remove")
    return frame_times, intensity_values

#Get Spectral Emphasis

#Get spectral emphasis
# Traunmüller and Erikson 2000

def get_spectralemphasis(voiceID, f0_values, time_frames):
    sound = parselmouth.Sound(voiceID) # read the sound
    se_values = []
    frames = []
    for i in range(1, len(time_frames)):
        starttime = time_frames[i-1]
        endtime = time_frames[i]
        mean_time = (endtime + starttime) / 2
        extract = call(sound, "Extract part...", starttime, endtime, "rectangular", 1, "no")
        spectrum = call(extract, "To Spectrum", "yes")
        meanf0 = (f0_values[i] + f0_values[i-1]) / 2
        if not np.isnan(meanf0):
            low_band = meanf0 * 1.43
            se = call(spectrum, "Get band energy difference...", 0, low_band, 0, 0)
        else:
            se = call(spectrum, "Get band energy difference...", 0, 400, 0, 0)
        se_values.append(se)
        frames.append(mean_time)

    _ = call(extract, "Remove")
    _ = call(spectrum, "Remove")
    return frames, se_values

def get_cpps(sound, f0min=75, f0max=1000, time_step = 0.005):
    # Go through all the wav files in the folder and estimate f0 using different algorithms
    cepstrogram = call(sound, "To PowerCepstrogram", f0min, 0.005, 5000, 50) #create a praat cesptrogram obje     ct
    total_duration = call(cepstrogram, "Get total duration")
    step = 0.0
    times = []
    cpps = []
    while step <= total_duration:
        pc_slice = call(cepstrogram, "To PowerCepstrum (slice)", step)
        cpp = call(pc_slice, "Get peak prominence", f0min, f0max, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow")
        times.append(step)
        cpps.append(cpp)
        step += time_step
    # Append results
    return times, cpps


def get_praat(path, hf, verbose=True, verify=True):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                # Create the HDF5 group
                file_name = file.replace('.wav', '')
                root_group = os.path.join(root, file_name)
                if str(root_group) not in hf:
                    grp = hf.create_group(os.path.join(root, file_name))
                else:
                    grp = hf[str(root_group)]



                # Read file and create sound object
                try:
                    sound = parselmouth.Sound(os.path.join(root, file))
                except Exception as e:
                    print("File {0} not opened: {1}".format(file, e))

                # Auto-correlation
                if "praat_ac" not in grp:
                    try:
                        times_ac, f0_ac = get_pitch_ac(sound, f0min=75, f0max=1000, unit="Hertz", hirst=False)
                        var1 = np.array(times_ac)
                        var2 = np.array(f0_ac)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("praat_ac", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="praat_ac", log_name=error_log_name)
                else:
                    print("Dataset praat_ac is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "praat_ac"))]
                        check_extraction(var=dset[0], length=5, dset=dset)


                # Cross-correlation
                if "praat_cc"not in grp:
                    try:
                        times_cc, f0_cc = get_pitch_cc(sound, f0min=75, f0max=1000, unit="Hertz", hirst=False)
                        var1 = np.array(times_cc)
                        var2 = np.array(f0_cc)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("praat_cc", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="praat_cc", log_name=error_log_name)
                else:
                    print("Dataset praat_cc is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "praat_ac"))]
                        check_extraction(var=dset[0], length=5, dset=dset)


                # SHS
                if "praat_shs" not in grp:
                    try:
                        times_shs, f0_shs = get_pitch_shs(sound, f0max=1000, unit="Hertz")
                        var1 = np.array(times_shs)
                        var2 = np.array(f0_shs)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("praat_shs", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="praat_shs", log_name=error_log_name)
                else:
                    print("Dataset praat_shs is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "praat_shs"))]
                        check_extraction(var=dset[0], length=5, dset=dset)


                # Intensity
                if "praat_intensity" not in grp:
                    try:
                        times_intensity, intensity = get_intensity(sound, f0min=75, subtract_mean="yes")
                        var1 = np.array(times_intensity)
                        var2 = np.array(intensity)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("praat_intensity", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="praat_intensity", log_name=error_log_name)
                else:
                    print("Dataset praat_intensity is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "praat_intensity"))]
                        check_extraction(var=dset[0], length=5, dset=dset)


                # Spectral emphasis
                if "praat_se" not in grp:
                    try:
                        times_se, se = get_spectralemphasis(sound, f0_ac, times_ac)
                        var1 = np.array(times_se)
                        var2 = np.array(se)
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("praat_se", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="praat_se", log_name=error_log_name)
                else:
                    print("Dataset praat_se is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "praat_se"))]
                        check_extraction(var=dset[0], length=5, dset=dset)

                # CPPS
                if "cpps" not in grp:
                    try:
                        times, cpps = get_cpps(sound, f0min=75, f0max=1000, time_step = 0.005)
                        var1 = np.array(times).ravel()
                        var2 = np.array(cpps).ravel()
                        arr = np.array([var1, var2])
                        dset = grp.create_dataset("cpps", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                        del var1, var2, arr
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="cpps", log_name=error_log_name)
                else:
                    print("Dataset cpps is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "cpps"))]
                        check_extraction(var=dset[0], length=5, dset=dset)



                if verbose:
                    print("File {0} done for Praat".format(file))
                # End of definition


def hirst_procedure_yin(x):
    q25 = np.nanquantile(x, 0.25)
    q75 = np.nanquantile(x, 0.75)
    minf0 = (q25 * 0.75/10)*10
    maxf0 = (q75 * 1.5/10)*10
    if (maxf0 == 0 or np.isnan(maxf0)):
        maxf0 = 800
    if (minf0 == 0 or np.isnan(minf0)):
        minf0 = 40
    return minf0, maxf0

def call_pyin(sound, hirst_procedure=True):
    f0_yin, voiced_flag, voiced_probs = librosa.pyin(sound, fmin=75, fmax=1000, center=True)
    if hirst_procedure:
        MIN, MAX = hirst_procedure_yin(f0_yin)
        del f0_yin
        f0_yin, voiced_flag, voiced_probs = librosa.pyin(sound, fmin=int(MIN), fmax=int(MAX))
    return f0_yin, voiced_flag, voiced_probs


def get_pyin_mfcc(path, hf, verbose=True, verify=True):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                file_name = file.replace('.wav', '')
                # Create or load the group
                root_group = os.path.join(root, file_name)
                if str(root_group) not in hf:
                    grp = hf.create_group(root_group)
                else:
                    grp = hf[str(root_group)]

                # Load file
                try:
                    sound, fs = librosa.load(os.path.join(root, file))
                except:
                    continue

                # PYIN
                if "pyin" not in grp:
                    try:
                        f0_yin, voiced_flags, voiced_probs = call_pyin(sound, hirst_procedure=False)
                        timeframes = librosa.times_like(f0_yin) # Check why we have a "petit décalage" here
                        var1 = np.array(timeframes)
                        var2 = np.array(f0_yin)
                        var3 = np.array(voiced_flags)
                        arr = np.array([var1, var2, var3])
                        dset = grp.create_dataset("pyin", data=arr)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="pyin", log_name=error_log_name)
                else:
                    print("Dataset pyin is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "pyin"))]
                        check_extraction(var=dset[0], length=5, dset=dset)

                # MFCC
                if "mfcc" not in grp:
                    try:
                        hop_length = int(0.005 * fs)
                        mfcc = librosa.feature.mfcc(y=sound, sr=fs, hop_length=hop_length)
                        dset = grp.create_dataset("mfcc", data=mfcc)
                        if verify:
                            check_extraction(var=var1, length=5, dset=dset)
                    except Exception as e:
                        print(e)
                        write_error(path=root, file_name=file, error=e, algo="mfcc", log_name=error_log_name)
                else:
                    print("Dataset mfcc is already in arborescence of file {0}".format(file_name))
                    if verify:
                        dset = hf[str(os.path.join(grp.name, "mfcc"))]
                        check_extraction(var=dset[0], length=5, dset=dset)


                if verbose:
                    print("Audio file {0} done for pyin and mfcc".format(file_name))


def batch_process_deprecated(algorithm, files_path=PATH, algo_name='YAAPT', algo_folder='YAAPT_v4.0', file_format=".wav", verbose=False, matlab_audioread=False):
    """
    algorithm: a PDA function that returns time, fundamental frequency, and voicedness probability.
    path: the path to the folder where audio files are store.
    file_format: file format of the audio files to be batch processed.
    verbose: defaults to True. Prints the progress of the process.
    
    Returns a dictionary whose keys are the names of the files processed without the file formats
    and a list of lists contaning time frames, fundamental frequency, and voicedness probability.
    """

    with matlab.engine.start_matlab() as eng:
        algopath = pathlib.Path.cwd() / algo_folder
        eng.cd(str(algopath))
    
        # Walk through files
        for root, dirs, files in os.walk(files_path):
            for file in files:
                if file.endswith(file_format):
                    file_name = file.replace(file_format, '')
                    if matlab_audioread:
                        var1, var2, var3 = algorithm(root, file, eng)
                    else:    
                        sig , sr = soundfile.read(os.path.join(root, file))
                        var1, var2, var3 = algorithm(sig, sr, eng)
                    
                    # Save a result txt file on the same folder of audio
                    results_path = os.path.join(root, file_name) + '_'+ str(algo_name) + '_results.txt'
                    with open(results_path, 'a+') as f:
                        # On traite Creaky et les autres differemment
                        if algo_name == 'CREAKY':
                            f.writelines('timeframe' + '\t' + 'creaky_decision' + '\t' + 'creaky_prob')
                            for i in range(len(var1)):
                                f.writelines('\n' + str(var1[i]) + '\t' + str(var3[i]) + '\t' + str(var2[i]))
                        # Pour tous les autres algorithmes utilisant la fonction
                        else:
                            f.writelines('timeframe' + '\t' + 'f0' + '\t' + 'voice_prob')
                            for i in range(len(var1)):
                                f.writelines('\n' + str(var1[i]) + '\t' + str(var2[i]) + '\t' + str(var3[i]))
                        
                        
                    if verbose:
                        print('Audio file {0} done'.format(file_name))
            
    return 0


#### TENTATIVE H5PY-FILE-WRITTER BATCH PROCESS FUNCTION ####
# https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
# https://www.hdfgroup.org/solutions/hdf5/
# https://docs.h5py.org/en/stable/
# https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
# 
def batch_process_h5(algorithm, files_path, hf, algo_name='YAAPT', algo_folder='YAAPT_v4.0', file_format=".wav", verbose=False, matlab_audioread=False, verify=True):
    """
    algorithm: a PDA function that returns time, fundamental frequency, and voicedness probability.
    path: the path to the folder where audio files are store.
    file_format: file format of the audio files to be batch processed.
    verbose: defaults to True. Prints the progress of the process.

    Returns a dictionary whose keys are the names of the files processed without the file formats
    and a list of lists contaning time frames, fundamental frequency, and voicedness probability.
    """

    # The root h5py file must be created before we enter the loop and it must passed as a parameter of the batch_process_h5 function
    # TODO:
    # Modify the function so that it will receive a list of algorithms, algo_names, algo_folders, and true/false matlab_audioread



    # Estrutura
    # ROOT > FILE > NOISE_CONDITION (?) > NOISE_LEVEL (?) > ALGO_NAME > DATASET # ANY ATTRIBUTES ?
    # You can build a big if else strucuture to test if noise conditions and levels of SNR are in the name of the files like
    # "results" in "ad00a9_16kHz_Noise_Ventilator_0dB_opensmile_results.txt", which returns True.
    # Engine is the last item in the hierarchy. I doesn't make sense creating and opening a new engine inside batch_process
    # Make a loop that calls batch_process_h5 for each PDA algorithm, having the h5 file as a parameter
    # This will be better because thereafter you can test whether the dataset already exists in the hierarchy

    with matlab.engine.start_matlab() as eng:
        algopath = pathlib.Path.cwd() / algo_folder
        eng.cd(str(algopath))

        # Walk through files
        for root, dirs, files in os.walk(files_path):
            for file in files:
                if file.endswith(file_format):

                    # Create a group inside the h5's file hierarchy which corresponds to root

                    file_name = file.replace(file_format, '')

                    # Only create new group if it does not exist in arborescence
                    #if str(os.path.join(root, file_name)) not in hf:
                    root_group = os.path.join(root, file_name)
                    if str(root_group) not in hf:
                        grp = hf.create_group(root_group)
                    else:
                        grp = hf[str(root_group)]
                        #print(hf[str(root_group)].name)
                    #except:
                    #    pass

                    # Change here to test if a group/test exist in the hdf5 file
                    # compared_name = file_name + '_'+ str(algo_name) + '_results.txt'


                    # VERY IMPORTANT: TEST IF DATASET EXISTS IN ABORESCENCE
                    dset_name = str(algo_name)
                    dset_exist = dset_name in grp
                    if not dset_exist:
                        # GET DATA
                        if matlab_audioread:
                            try:
                                var1, var2, var3 = algorithm(root, file, eng)
                                var1 = np.array(var1)
                                var2 = np.array(var2)
                                var3 = np.array(var3)
                                arr = np.array([var1, var2, var3])
                                # We only create a dataset at the end of the arborescence if and only if there's no error in f0 estimations
                                dset = grp.create_dataset(str(algo_name), data=arr)
                                if verify:
                                    check_extraction(var=var1, length=5, dset=dset)
                                if verbose:
                                    print("Audio file {0} done for {1}".format(file_name, dset_name))
                            except Exception as e:
                                print(e)
                                continue

                        else:
                            try:
                                sig , sr = soundfile.read(os.path.join(root, file))
                                var1, var2, var3 = algorithm(sig, sr, eng)
                                var1 = np.array(var1)
                                var2 = np.array(var2)
                                var3 = np.array(var3)
                                arr = np.array([var1, var2, var3])
                                # We only create a dataset at the end of the arborescence if and only if there's no error in f0 estimations
                                dset = grp.create_dataset(str(algo_name), data=arr)
                                if verify:
                                    check_extraction(var=var1, length=5, dset=dset)
                                if verbose:
                                    print("Audio file {0} done for {1}".format(filename, dset_name))
                            except Exception as e:
                                print(e)
                                continue

                    else:
                        print("Dataset {0} is already in arborescence of file {1}".format(dset_name, file_name))
                        if verify:
                            dset = hf[str(os.path.join(grp.name, dset_name))]
                            check_extraction(var=dset[0], length=5, dset=dset)

                    #if compared_name not in done_files:        
                    #    results_path = os.path.join(root, file_name) + '_'+ str(algo_name) + '_results.txt'
                    #    with open(results_path, 'a+') as f:
                    #        f.writelines('timeframe' + '\t' + 'f0' + '\t' + 'voice_prob')
                    #        for i in range(len(var1)):
                    #            f.writelines('\n' + str(np.round(var1[i], 3)) + '\t' + 
                    #                         str(np.round(var2[i], 2)) + '\t' + 
                    #                         str(np.round(var3[i], 3)))



#TODO: faire le rounding des resultats
# COLOCAR DOIS TIPOS DE RETURN: UM NO QUE DÁ CERTO E OUTRO NA EXCEÇÃO
# Used by get_hnr
def yin(signal, samplerate, eng):
    if samplerate != 16000:
        signal = resampy.resample(signal, samplerate, 16000)
        samplerate = 16000
    struct= eng.yin(matlab.double(signal.tolist()), matlab.double([16000]), nargout=1)
    f0 = 440 * 2**np.asarray(struct['f0']._data)
    hopsize = struct['hop']
    time = np.arange(len(f0))*hopsize/samplerate
    p = np.ones(len(f0))

    return time, f0, p

def yin_deprecated_2(root, file, eng):
    # Output validation done
    try:
        tmpfilepath = os.path.join(root, file)
        struct = eng.yin(str(tmpfilepath), nargout=1)
        f0 = 440 * 2**np.asarray(struct['f0']._data)
        f0 = f0.ravel()
        print(type(f0))
        print(f0.shape)
        time = np.arange(len(f0))*struct['hop']/struct['sr']
        print(type(time))
        print(time.shape)
        p = np.ones(len(f0))
    except Exception as e:
        # TODO: write not processed files into a log txt file
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="YIN", log_name=error_log_name)
        time = 0
        f0 = 0
        prob = 0
    return time, f0, p

# Used for regular extraction
def yin2(root, file, eng):
    # Output validation done
    try:
        tmpfilepath = os.path.join(root, file)
        struct = eng.yin(str(tmpfilepath), nargout=1)
        f0 = 440 * 2**np.asarray(struct['f0']._data)
        f0 = f0.ravel()
        time = np.arange(len(f0))*struct['hop']/struct['sr']
        prob = np.ones(len(f0))
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="YIN", log_name=error_log_name)
        raise e



def pefac(root, file, eng):
    # Output validation done
    try:
        f0, time, prob = eng.matlab_audioread(root, file, nargout=3)
        f0 = np.asarray(f0._data).ravel()
        time = np.asarray(time._data).ravel()
        prob = np.asarray(prob._data).ravel()
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="PEFAC", log_name=error_log_name)
        raise e


def bana(root, file, eng):
    try:
        f0, sig, sr = eng.matlab_audioread(root, file, nargout=3)
        f0 = np.asarray(f0).ravel()
        length = (len(sig) / sr)
        time = np.linspace(0, length, num=len(f0))
        prob = np.ones(len(f0))
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="BANA", log_name=error_log_name)
        raise e


def straight(root, file, eng):
    try:
        f0, sig, sr = eng.matlab_audioread(root, file, nargout=3)
        f0 = np.asarray(f0).ravel()
        length = (len(sig) / sr)
        time = np.linspace(0, length, num=len(f0))
        prob = np.ones(len(f0))
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="STRAIGHT", log_name=error_log_name)
        raise e

# We have a problem with SRH: sampling rate is way too small. While straight is getting 985 points for a certain file, SRH is getting only 85 points.
# TODO: check how we can solve this problem by modifying parameters of matlab script
def srh(root, file, eng):
    try:
        f0, time, decision, prob, sig, sr = eng.matlab_audioread(root, file, nargout=6)
        f0 = np.asarray(f0).ravel()
        time = np.asarray(time).ravel()
        decision = np.asarray(decision).ravel()
        #print(type(time))
        #print("time done")
        #f0 = f0._data
        #print(f0.shape)
        #print(f0)
        #print("f0 done")
        #decision = decision._data
        #print("decision done")
        return time, f0, decision
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="SRH", log_name=error_log_name)
        raise e


def swipep(root, file, eng):
    try:
        f0, time, prob = eng.matlab_audioread(root, file, nargout=3)
        f0 = np.asarray(f0).ravel()
        time = np.asarray(time).ravel()
        prob = np.asarray(prob).ravel()
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="SWIPEP", log_name=error_log_name)
        raise e


def swipe(root, file, eng):
    try:
        f0, time, prob = eng.matlab_audioread_swipe(root, file, nargout=3)
        f0 = np.asarray(f0).ravel()
        time = np.asarray(time).ravel()
        prob = np.asarray(prob).ravel()
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="SWIPE", log_name=error_log_name)
        raise e


def rapt(root, file, eng):
    # https://stackoverflow.com/questions/34155829/how-to-efficiently-convert-matlab-engine-arrays-to-numpy-ndarray
    try:
        f0, time, sig, sr  = eng.matlab_audioread(root, file, nargout=4)
        f0 = np.array(f0._data).ravel()
        #print(len(f0))
        length = (len(sig) / sr)
        time = np.linspace(0, length, num=len(f0))
        #time = np.array(time._data).ravel()
        #print(len(time))
        #print(time)
        #print(time2)
        prob = np.ones(len(f0))
        #print(len(prob))
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="RAPT", log_name=error_log_name)
        raise e


def yaapt(root, file, eng):
    try:
        f0, numfrms, signal, samplerate = eng.fuckmatlab(root, file, nargout=4)
        f0 = np.asarray(f0).ravel()
        signal = np.asarray(signal).ravel()
        samplerate = np.asarray(samplerate)
        length = (len(signal) / samplerate)
        time = np.linspace(0, length, num=len(f0))
        prob = np.ones(len(f0))
        return time, f0, prob
    except Exception as e:
        print("File not processed: {0}".format(file))
        write_error(path=root, file_name=file, error=e, algo="YAAPT", log_name=error_log_name)
        raise e


import os
import uuid
import pathlib
import numpy
import resampy
import soundfile

# Define wrapper for PAPD
# Define wrapper for PAPD
def get_hnr(path, hf, file_format=".wav", verbose=True, verify=True):
    """

    """

    with matlab.engine.start_matlab() as eng:
        algopath = pathlib.Path.cwd() / 'PAPD_Limsi_v1.1'
        eng.cd(str(algopath))
        # Walk through files
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(file_format):
                    file_name = file.replace(file_format, '')

                    # Create group
                    root_group = os.path.join(root, file_name)
                    if str(root_group) not in hf:
                        grp = hf.create_group(root_group)
                    else:
                        grp = hf[str(root_group)]

                    if "hnr" not in grp:
                        # Check file conformity
                        # PADP assumes that sample rate is max 16k
                        try:
                            signal , samplerate = soundfile.read(os.path.join(root, file))
                            if samplerate != 16000:
                                signal = resampy.resample(signal, samplerate, 16000)
                                samplerate = 16000

                            tmpfilepath = pathlib.Path.cwd() / (str(uuid.uuid1()) + '.wav')
                            soundfile.write(str(tmpfilepath), signal, samplerate)

                            # We don't need to do that again if we have already calculated YIN and it is in the arborescence
                            if "yin" in grp:
                                f0 = grp['yin'][1]
                            else:
                                struct = eng.yin(matlab.double(signal.tolist()), matlab.double([16000]), nargout=1)
                                f0 = 440 * 2**np.asarray(struct['f0']._data)
                                hopsize = struct['hop']
                                timeframes = np.arange(len(f0))*hopsize/samplerate

                            # Get HNR
                            _, _, hnr, _ = eng.lance_papd(str(tmpfilepath), f0, nargout=4)
                            length = (len(signal) / samplerate)

                            #Save file
                            var1 = np.array(hnr).ravel()
                            var2 = np.linspace(0, length, num=len(var1))
                            arr = np.array([var1, var2])
                            dset = grp.create_dataset("hnr", data=arr)
                            if verify:
                                check_extraction(var=var1, length=5, dset=dset)
                            #if tmpfilepath.exists():
                            #    tmpfilepath.unlink()
                            if verbose:
                                print('Audio file {0} done'.format(file_name))

                        except Exception as e:
                            print(e)
                            write_error(path=root, file_name=file, error=e, algo="hnr", log_name=error_log_name)
                        if tmpfilepath.exists():
                            tmpfilepath.unlink()

                    else:
                        print("Dataset hnr is already in arborescence of file {0}".format(file_name))
                        if verify:
                            dset = hf[str(os.path.join(grp.name, "opensmile"))]
                            check_extraction(var=dset[0], length=5, dset=dset)


# A toolbox Deep Learning deve estar instalada no Matlab
# https://fr.mathworks.com/help/install/ug/install-noninteractively-silent-installation.html
def detect_creaky(signal, samplerate, eng):
    
    if samplerate != 16000:
        signal = resampy.resample(signal, samplerate, 16000)
        samplerate = 16000
    
    
    signal_ml = matlab.double(signal.tolist())
    creaky_prob, creaky_decision = eng.detect_creaky_voice(signal_ml, float(samplerate), nargout=2)
    creaky_prob = np.asarray(np.asarray(creaky_prob).ravel().tolist()[0:len(np.asarray(creaky_prob).ravel()):2])
    creaky_decision = np.asarray(np.asarray(creaky_decision).ravel().tolist()[0:len(np.asarray(creaky_decision).ravel()):2])
    length = (len(signal) / samplerate)
    time = np.linspace(0, length, num=len(creaky_decision))
        #print(creaky_decision)
    #except:
    #    creaky_prob = np.nan
    #    creaky_decision = np.nan
    #    print("File not processed")    
    
    return time, creaky_prob, creaky_decision


print("Functions definition done!")

#### GET FEATURES ####
#PATH = os.path.normpath('/srv/tannatdata/saulo/Corpus/Oliver/')

with h5py.File(RESULT_FILE, 'a') as f:
    """
    batch_process_h5(detect_creaky, hf=f, files_path=PATH, algo_name='creaky', algo_folder='covarep-master/glottalsource/creaky_voice_detection', file_format=".wav", verbose=True, matlab_audioread=True)
    get_opensmile(path=PATH, hf=f, verbose=True, verify=True)
    get_praat(path=PATH, hf=f, verbose=True, verify=True)
    get_pyin_mfcc(path=PATH, hf=f, verbose=True, verify=True)
    batch_process_h5(yin2, hf=f, files_path=PATH, algo_name='yin', algo_folder='YIN', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(pefac, hf=f, files_path=PATH, algo_name='pefac', algo_folder='PEFAC', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(algorithm=srh, hf=f, files_path=PATH, algo_name='srh', algo_folder='covarep-master/glottalsource', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(algorithm=straight, hf=f, algo_name='straight', files_path=PATH, algo_folder='legacy_STRAIGHT-master/src', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(algorithm=bana, hf=f, algo_name='bana', files_path=PATH, algo_folder='BANA', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(algorithm=swipep, hf=f, algo_name='swipep', files_path=PATH, algo_folder='SWIPE/swipe-master', file_format=".wav", verbose=False, matlab_audioread=True)
    batch_process_h5(algorithm=swipe, hf=f, algo_name='swipe', files_path=PATH, algo_folder='SWIPE/swipe-master', file_format=".wav", verbose=False, matlab_audioread=True)
    batch_process_h5(algorithm=rapt, hf=f, algo_name='rapt', files_path=PATH, algo_folder='RAPT', file_format=".wav", verbose=True, matlab_audioread=True)
    batch_process_h5(algorithm=yaapt, hf=f, algo_name='yaapt', files_path=PATH, algo_folder='YAAPT_v4.0', file_format=".wav", verbose=True, matlab_audioread=True)
    get_hnr(path=PATH, hf=f, file_format=".wav", verbose=True, verify=True)
    """
    batch_process_h5(algorithm=detect_creaky,
                algo_name='creaky',
                files_path=PATH,
                algo_folder='covarep-master/glottalsource/creaky_voice_detection', 
                file_format=".wav", 
                verbose=False, 
                matlab_audioread=False)


print("extraction done!")
