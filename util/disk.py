import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO

from util.longconf import logging
log = logging.getLogger('__name__')
log.setLevel(logging.DEBUG)

class gzipdisk(Disk):
    def Store(self, value, key, read = None):
        if type(value) == BytesType:
            if read:
                value = value.read()
                read = False

        str_io = BytesIO()
        gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

        for offset in range(0, len(value), 2**30):
            gz_file.write(value[offset:offset+2**30])
        gz_file.close()

        value = str_io.getvalue()

        return super(gzipdisk, self).store(value, read)

    def fetch(value, read, mode, filename):
        value = super(gzipdisk, self).fetch(value, mode, read, filename)

        if mode == 'MODE_BINARY':
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            new = BytesIO()
    
            while True:
                uncompressed = gz_file.read(2**30)
                if uncompressed:
                    new.write(uncompressed)
                else:
                    break
            value = new.getvalue()
        return value

def getcache(scope_str):
    return FanoutCache('cache/'+scope_str, disk=GzipDisk, shards=64, timeout=1, size_limit=3e11)