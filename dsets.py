import pandas as pd
import numpy as np
import os
import glob
import csv
import functools
import SimpleITK as sitk
from collections import namedtuple

from util.util import xyztuple, xyz2irc
from util.disk import getcache
from util.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple('CandidateInfoTuple', ['isNodule_bool', 'diameter_mm', 'series_uid', 'center_xyz'])
raw_cache = getcache('c10cache')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool = True):
    mhd_path = glob.glob('subset*/*.mhd')
    presentOnDisk = {os.path.split(p)[-1][:-4] for p in mhd_path}
   
    diameter_dict = {}
    with open('annotations.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            anncenter_xyz = tuple(float(x) for x in row[1:4])
            anndiam_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append((anncenter_xyz, anndiam_mm))

    candidateInfoList = []
    with open('candidates.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk and requireOnDisk_bool:
                continue
            candDiam_mm = 0.0
            isNodule_bool = bool(row[4])
            candCentre_xyz = tuple(float(x) for x in row[1:4])
            for ann in diameter_dict.get(series_uid, []):
                center, diameter = ann
                for i in range(3):
                    delta = abs(candCentre_xyz[i] - center[i])
                    if delta > diameter/4:
                        break
                else:
                    candDiam_mm = diameter
                    break
            candidateInfoList.append(CandidateInfoTuple(isNodule_bool, candDiam_mm, series_uid, candCentre_xyz))
    candidateInfoList.sort(reverse = True)
    return candidateInfoList

class ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32)
        array.clip(-1000, 1000, array)
        self.series_uid = series_uid
        self.hu_a = array
        self.origin_xyz = xyztuple(*ct_mhd.GetOrigin())
        self.voxsize_xyz = xyztuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)
    def getRawCand(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.voxsize_xyz, self.direction_a)
        slice_list = []
        for axis, center_val in enumerate(center_xyz):
            start = int(round(center_val - width_irc[axis]/2))
            stop = int(start + width_irc[axis])
            slice_list.append(slice(start, stop))
        ctChunk = self.hu_a[tuple(slice_list)]
        return ctChunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCand(series_uidcenter_xyz, width_irc)
    return ct_chunk, center_irc
                

class LunaDataset():
    def __init__(self, val_stride = 0, isValset_bool = None, series_uid = None):
        self.candInfo_list = copy.copy(getCandInfoList())
        if series_uid:
            self.candInfo_list = [x for x in self.candInfo_list if x.series_uid  == series_uid]
        if isValset:
            assert val_stride > 0, val_stride
            self.candInfo_list = self.candInfo_list[::val_stride]
            assert self.candInfo_list
        else:
            del self.candInfo_list[::val_stride]

    def __len__(self):
        return len(self.candInfo_list)
    
    def __getitem__(self, ndx):
        candInfo_tup = self.candInfo_list[ndx]
        width_irc = [32,48,48]
        candidate_a, center_irc = getCtRawCandidate(candInfo_tup.series_uid, candInfo_tup.center_xyz, width_irc)
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze()
        pos_t = torch.tensor([not candInfo_tup.isNodule_bool, candInfo_tup.isNodule_bool], dtype = torch.long)
        return candidate_t, pos_t