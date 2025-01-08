import collections
import numpy as np



from logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


collections.irctuple = namedtuple('IrcTuple', ['index', 'row', 'couple'])
collections.xyztuple = namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(cord_irc, origin_xyz, voxsize_xyz, direction_a):
    cri = np.array(cord_irc)[::-1]
    origin_xyz = np.array(origin_xyz)
    voxsize_xyz = np.array(voxsize_xyz)
    cord_xyz = (direction_a @ cri*voxsize_xyz) + origin_xyz
    return irctuple(*cord_xyz)
def xyz2irc(cord_xyz, origin_xyz, voxsize_xyz, direction_a):
    cord_a = np.array(cord_xyz)
    voxsize_xyz = np.array(voxsize_xyz)
    origin_xyz = np.array(origin_xyz)
    irc = ((cord_a - origin_xyz) @ np.linalg.inverse(direction_a))/voxsize_xyz
    irc = np.round(irc)
    return xyztuple(irc[2], irc[1], irc[0])
