import sys, os
import math 
import os.path
from collections import OrderedDict
import argparse
import numpy as np 
import tiledb
import random 
ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}
reverse_dict={}
for key in ltrdict:
    cur_val=ltrdict[key]
    reverse_dict[tuple(cur_val)]=key

#def one_hot_encode(seqs):
#    return np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])

def one_hot_encode(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper()

    one_hot_map = np.identity(5)[:, :-1]

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds].reshape((len(seqs), seq_len, 4))


def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]

def dinuc_shuffle(seq):
    #get list of dinucleotides
    nucs=[]
    for i in range(0,len(seq),2):
        nucs.append(seq[i:i+2])
    #generate a random permutation
    random.shuffle(nucs)
    return ''.join(nucs) 


def revcomp(seq):
    seq=seq[::-1].upper()
    comp_dict=dict()
    comp_dict['A']='T'
    comp_dict['T']='A'
    comp_dict['C']='G'
    comp_dict['G']='C'
    rc=[]
    for base in seq:
        if base in comp_dict:
            rc.append(comp_dict[base])
        else:
            rc.append(base)
    return ''.join(rc)






class DefaultOrderedDictWrapper(object):
    def __init__(self, factory):
        self.ordered_dict = OrderedDict()
        assert hasattr(factory, '__call__')
        self.factory = factory

    def __getitem__(self, key):
        if key not in self.ordered_dict:
            self.ordered_dict[key] = self.factory() 
        return self.ordered_dict[key]

def enum(**enums):
    class Enum(object):
        pass
    to_return = Enum
    for key,val in enums.items():
        if hasattr(val, '__call__'): 
            setattr(to_return, key, staticmethod(val))
        else:
            setattr(to_return, key, val)
    to_return.vals = [x for x in enums.values()]
    to_return.the_dict = enums
    return to_return


def combine_enums(*enums):
    new_enum_dict = OrderedDict()
    for an_enum in enums:
        new_enum_dict.update(an_enum.the_dict)
    return enum(**new_enum_dict)


    
    
def coords_to_tdb_indices(coords,tdb_instance):
    '''
    coords is a tuple (chrom, start, stop)
    '''
    num_chroms=tdb_instance.meta['num_chroms']
    for i in range(num_chroms):
        if tdb_instance.meta['chrom_'+str(i)]==coords[0]:
            chrom_offset=tdb_instance.meta['offset_'+str(i)]
            tdb_index_start=chrom_offset+coords[1]
            tdb_index_end=chrom_offset+coords[2]
            return (tdb_index_start,tdb_index_end)
    raise Exception("chrom name:"+str(coords[0])+" not found in tdb array")


def tdb_indices_to_coords(indices,tdb_instance):
    '''
    indices is a list of tdb indices     
    '''
    pass


def transform_data_type(inputs,num_inputs):
    if inputs is None:
        inputs=[None]*num_inputs
    elif inputs is []:
        inputs=[None]*num_inputs
    else:
        assert(len(inputs)==num_inputs)
        transformed=[]
        for i in range(num_inputs):
            transformed.append([]) 
            cur_inputs=inputs[i].split(',')
            for j in cur_inputs: 
                if str(j).lower()=="none":
                    transformed[i].append(None)
                else:
                    transformed[i].append(float(j))
    return transformed


def transform_data_type_min(inputs,num_inputs):
    if inputs is None:
        transformed=[-math.inf]*num_inputs
    elif inputs is []:
        transformed=[-math.inf]*num_inputs
    else:
        assert(len(inputs)==num_inputs)
        transformed=[]
        for i in range(num_inputs):
            transformed.append([]) 
            cur_inputs=inputs[i].split(',')
            for j in cur_inputs: 
                if str(j).lower()=="none":
                    transformed[i].append(-math.inf)
                else:
                    transformed[i].append(float(j))
    return transformed

def transform_data_type_max(inputs,num_inputs):
    if inputs is None:
        transformed=[math.inf]*num_inputs
    elif inputs is []:
        transformed=[math.inf]*num_inputs
    else:
        assert(len(inputs)==num_inputs)
        transformed=[]
        for i in range(num_inputs):
            transformed.append([]) 
            cur_inputs=inputs[i].split(',')
            for j in cur_inputs: 
                if str(j).lower()=="none":
                    transformed[i].append(math.inf)
                else:
                    transformed[i].append(float(j))
    return transformed

