"""script to create a custom pytorch dataset using the annotation and candidate csv files"""
import pandas as pd
import numpy as np
import os
import glob
import csv
import functools
import SimpleITK as sitk
from collections import namedtuple

# Importing utility functions and logging configuration
from util.util import xyztuple, xyz2irc
from util.disk import getcache
from util.logconf import logging

# Set up logging for debugging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Named tuple to store candidate information
CandidateInfoTuple = namedtuple('CandidateInfoTuple', ['isNodule_bool', 'diameter_mm', 'series_uid', 'center_xyz'])

# Cache for storing intermediate results to improve performance
raw_cache = getcache('c10cache')

# Function to retrieve candidate information from CSV files
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # Get all .mhd files in directories matching 'subset*'
    mhd_path = glob.glob('subset*/*.mhd')
    # Create a set of available series UIDs from filenames
    presentOnDisk = {os.path.split(p)[-1][:-4] for p in mhd_path}

    # Dictionary to store nodule diameters
    diameter_dict = {}
    with open('annotations.csv') as f:
        for row in list(csv.reader(f))[1:]:  # Skip the header
            series_uid = row[0]
            anncenter_xyz = tuple(float(x) for x in row[1:4])
            anndiam_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append((anncenter_xyz, anndiam_mm))

    # List to store candidate information
    candidateInfoList = []
    with open('candidates.csv') as f:
        for row in list(csv.reader(f))[1:]:  # Skip the header
            series_uid = row[0]

            # Skip candidates that do not have corresponding .mhd files if required
            if series_uid not in presentOnDisk and requireOnDisk_bool:
                continue

            candDiam_mm = 0.0
            isNodule_bool = bool(int(row[4]))  # Convert string to boolean
            candCentre_xyz = tuple(float(x) for x in row[1:4])

            # Check if the candidate matches an annotation (i.e., it's a nodule)
            for ann in diameter_dict.get(series_uid, []):
                center, diameter = ann
                for i in range(3):
                    delta = abs(candCentre_xyz[i] - center[i])
                    if delta > diameter / 4:
                        break
                else:
                    candDiam_mm = diameter
                    break

            # Add candidate information to the list
            candidateInfoList.append(CandidateInfoTuple(isNodule_bool, candDiam_mm, series_uid, candCentre_xyz))

    # Sort candidates in descending order
    candidateInfoList.sort(reverse=True)
    return candidateInfoList

# Class to represent a CT scan and provide access to voxel data
class Ct:
    def __init__(self, series_uid):
        # Locate the corresponding .mhd file
        mhd_path = glob.glob('subset*/{}.mhd'.format(series_uid))[0]

        # Read the .mhd image and convert to a numpy array
        ct_mhd = sitk.ReadImage(mhd_path)
        array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # Clip Hounsfield Units (HU) to a standard range
        array.clip(-1000, 1000, array)

        self.series_uid = series_uid
        self.hu_a = array

        # Store spatial information
        self.origin_xyz = xyztuple(*ct_mhd.GetOrigin())
        self.voxsize_xyz = xyztuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    # Extract a candidate region from the CT scan
    def getRawCand(self, center_xyz, width_irc):
        # Convert XYZ coordinates to IRC (Index-Row-Column)
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.voxsize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start = int(round(center_val - width_irc[axis] / 2))
            stop = int(start + width_irc[axis])
            slice_list.append(slice(start, stop))

        # Extract the region of interest (ROI)
        ctChunk = self.hu_a[tuple(slice_list)]
        return ctChunk, center_irc

# Cache for the Ct object to avoid reloading the same scan
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

# Cached function to extract candidate regions from CT scans
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCand(center_xyz, width_irc)
    return ct_chunk, center_irc

# Custom dataset class for lung cancer candidates
class LunaDataset():
    def __init__(self, val_stride=0, isValset_bool=None, series_uid=None):
        self.candInfo_list = copy.copy(getCandidateInfoList())

        # Filter by a specific series if provided
        if series_uid:
            self.candInfo_list = [x for x in self.candInfo_list if x.series_uid == series_uid]

        # Separate validation and training sets
        if isValset_bool:
            assert val_stride > 0, val_stride
            self.candInfo_list = self.candInfo_list[::val_stride]
            assert self.candInfo_list
        else:
            del self.candInfo_list[::val_stride]

    def __len__(self):
        return len(self.candInfo_list)

    def __getitem__(self, ndx):
        # Retrieve candidate information
        candInfo_tup = self.candInfo_list[ndx]
        width_irc = [32, 48, 48]

        # Extract the candidate region
        candidate_a, center_irc = getCtRawCandidate(candInfo_tup.series_uid, candInfo_tup.center_xyz, width_irc)

        # Convert to PyTorch tensor and add a channel dimension
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32).unsqueeze(0)

        # Create a binary label tensor (nodule or not)
        pos_t = torch.tensor([not candInfo_tup.isNodule_bool, candInfo_tup.isNodule_bool], dtype=torch.long)

        return candidate_t, pos_t
