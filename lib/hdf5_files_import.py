import sys
sys.path.insert(1, r'.\SCA_utilities')
# from sca_DataKeys import ProjectDataSetTags

import os.path
import sys
import h5py
import numpy as np
import random
from glob import glob

def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	# Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	# Y_attack = np.array(in_file['Attack_traces/labels'])


	traces =np.concatenate((X_profiling, X_attack), axis=0)
	if load_metadata == False:
		    return traces
	else:
            # print(in_file['Attack_traces/metadata'][0])
            # print()
            # print(in_file['Attack_traces/metadata'][1])
            return traces, in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata']


def load_raw_ascad(ascad_database_file, idx_srt, idx_end, start, end, load_metadata=False ):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load raw traces
	idx = np.arange(idx_srt, idx_end, dtype=np.uint32)
	idx = idx.tolist()
	traces = np.array(in_file['Profiling_traces/traces'][idx, start:end], dtype=np.int8)
	if load_metadata == False:
		return traces
	else:
		return traces, in_file['Profiling_traces/metadata'] #[idx]    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Be careful 
            # print(in_file['Attack_traces/metadata'][0])
            # print()
            # print(in_file['Attack_traces/metadata'][1])

    	#return traces, in_file['Profiling_traces/metadata'] #[idx]


def load_ascad_metadata(ascad_database_file, num_trc):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# index
	profiling_index = [n for n in range(0, num_trc)]
	
	return in_file['metadata'][profiling_index]


def read_hdf5_proj(database_file, idx_srt, idx_end, start, end, load_plts=True, load_cpts=True ):
	check_file_exists(database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database_file)
		sys.exit(-1)
	# Load raw traces
	# profiling_index = [n for n in range(0, num_trc)]
	idx = np.arange(idx_srt, idx_end, dtype=np.uint32)
	idx = idx.tolist()
	traces = np.array(in_file[ProjectDataSetTags.TRACES.value][idx, start:end])

	if load_plts == True and load_cpts == True:
		return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx], in_file[ProjectDataSetTags.CIPHER_TEXT.value][idx]
	elif load_plts == True and load_cpts == False:
		return traces, in_file[ProjectDataSetTags.PLAIN_TEXT.value][idx]
	elif load_plts == False and load_cpts == True:
		return traces, in_file[ProjectDataSetTags.CIPHER_TEXT.value][idx]
	else:
		return traces

def read_parameters_from_file(param_filename):
	#read parameters for the extract_traces function from given filename
	#TODO: sanity checks on parameters
	param_file = open(param_filename,"r")

	#FIXME: replace eval() by ast.linear_eval()
	my_parameters= eval(param_file.read())

	model_file = my_parameters["model_file"]
	ascad_database = my_parameters["ascad_database"]
	num_traces = my_parameters["num_traces"]
	return model_file, ascad_database, num_traces


def read_multi_h5(h5_file, num_trc, srt, end):
	# Read multiple h5(trace) files
	files_trace = glob(h5_file)
	traces = []
	for filename in files_trace:
		f = h5py.File(filename, 'r')
		a_group_key = list(f.keys())[0]
		traces.extend(  f[a_group_key][0:num_trc, srt:end]  )
	traces = np.asarray( traces )
	return traces

def read_single_h5(database_file, idx_srt, idx_end, start, end ):
	check_file_exists(database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database_file)
		sys.exit(-1)
	# Load raw traces
	idx = np.arange(idx_srt, idx_end, dtype=np.uint32)
	idx = idx.tolist()
	traces = np.array(in_file['MTD'][idx, start:end], dtype=np.float32)
	
	return traces

	
	
def read_multi_plt(plt_file, num_trc):
	# read multiple plaintext files
	files_plt = glob(plt_file)
	plt = []
	for filename in files_plt:
		# valid when has only one test file, otherwise will read [0:num_trc] in every text file
		lines_to_read = np.arange( 0, num_trc, 1 )
		with open(filename, 'r') as f:
			for position, line in enumerate(f):
				if position in lines_to_read:
					plt.append(list(map(str,line.split())))
				else:
					break

	plt = np.asarray(plt)
	return plt


# test code
import time

def main():
	if len(sys.argv)!=2:
		#default parameters values
		ascad_database=traces_file=r'D:\Data\ASCAD_data\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5'
		# num_traces=2000
	else:
		#get parameters from user input
		model_file, ascad_database, num_traces = read_parameters_from_file(sys.argv[1])

	# check data
	# (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(ascad_database_file=traces_file, load_metadata=True)
	# traces, metadata = load_raw_ascad(ascad_database_file=traces_file, load_metadata=True)

	try:
		input("Press enter to exit ...")
	except SyntaxError:
		pass





if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))
