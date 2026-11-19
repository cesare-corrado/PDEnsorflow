#!/usr/bin/env python
"""
    RoeWriter: writes a binary ionic state file (`.roe`) byte for byte as the
    reference simulator writes it, so that the reference can restart from it
    (start_statef) and so that a file read with RoeReader and written back is
    identical to the original.

    The layout is described in gpuSolve.IO.readers.roereader. The content to
    write is a dict with the keys RoeReader.read() returns; 'created' may be
    None (the current wall-clock time is used, as the reference does) and
    'fingerprint' entries are not needed: the writer takes them from the
    content when present and requires them only in format 3.

    The names follow the reference too (see the module functions below):
        <write_statef>.<tsav_ext>.roe   a state saved at a time of tsav[];
                                        tsav_ext defaults to the time, %.1f
        checkpoint.<time %.1f>.roe      interval checkpointing (chkpt_intv)
        exit.save.<time %.3f>.roe       the state saved when a run is stopped

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import struct
import time

import numpy as np

from gpuSolve.IO.readers.roereader import ROE_FORMATS
from gpuSolve.IO.readers.roereader import ROE_GLOBAL_NAMES
from gpuSolve.IO.readers.roereader import ROE_MAGIC
from gpuSolve.IO.readers.roereader import record_dtype


# Extension of every state file of the reference.
ROE_EXTENSION : str = '.roe'

# Rows handled per pass when regions have records of different sizes (see
# RoeReader: the scatter index has rows x record-size entries).
_CHUNK_ROWS : int = 32768


def tsav_label(tsav: float) -> str:
    """ tsav_label(tsav) returns the default tsav_ext of a save time: the time
        with one decimal, as the reference's stringify() prints it ("%.1f").
        Python's '%' formatting and C's printf both round the binary value
        exactly, so the two agree on every input, ties included.
    """
    return('%.1f' % tsav)


def tsav_file_name(write_statef: str, tsav_ext: str) -> str:
    """ tsav_file_name(write_statef, tsav_ext) returns the name of a state saved
        at a time of tsav[]: "<write_statef>.<tsav_ext>.roe"
    """
    return('{}.{}{}'.format(write_statef, tsav_ext, ROE_EXTENSION))


def checkpoint_file_name(ctime: float) -> str:
    """ checkpoint_file_name(ctime) returns the name of an interval checkpoint:
        "checkpoint.<ctime %.1f>.roe". The reference prints the time of the
        step that triggered the checkpoint, so ctime is that step's time.
    """
    return('checkpoint.%.1f%s' % (ctime, ROE_EXTENSION))


def exit_file_name(ctime: float) -> str:
    """ exit_file_name(ctime) returns the name of the state saved when a run is
        stopped: "exit.save.<ctime %.3f>.roe"
    """
    return('exit.save.%.3f%s' % (ctime, ROE_EXTENSION))


def encode_records(state: dict, fields: list, packed: bool) -> np.ndarray:
    """ encode_records(state, fields, packed) turns {field name: values, one per
        node} into the raw records of one state structure, a uint8 array
        (nodes, record size). fields is the ordered (name, C type) list; each
        value is cast to the field's stored type (a Gatetype is rounded to
        float32, as the reference stores it). Padding bytes are zero, as in
        the reference's zero-filled state tables.
    """
    dtype   = record_dtype(fields, packed)
    missing = [name for name in dtype.names if name not in state]
    if len(missing) > 0:
        raise ValueError('cannot encode the records: no values for {}'.format(', '.join(missing)))
    nnodes  = np.size(state[dtype.names[0]])
    records = np.zeros(nnodes, dtype=dtype)
    for name in dtype.names:
        values = np.reshape(np.asarray(state[name]), (-1,))
        if values.size != nnodes:
            raise ValueError('cannot encode the records: {} has {} values, {} expected'.format(
                name, values.size, nnodes))
        records[name] = values.astype(dtype.fields[name][0], copy=False)
    return(records.view(np.uint8).reshape(nnodes, dtype.itemsize))


class RoeWriter:
    """
    class RoeWriter: writes the content of a binary ionic state file.
    """

    def __init__(self, config: dict = None):
        self._fname : str       = 'state.roe'
        # 1 is what release 19.0 of the reference writes and reads
        self._format : int      = 1
        # width of the node count in formats 2 and 3: the reference's global
        # index type, 32 bits unless it was built with 64-bit indices
        self._index_bytes : int = 4
        # the reference writes its own commit count in a tissue run and 0 in a
        # single-cell run; 0 claims no revision
        self._revision : int    = 0

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_fname(self, fname: str):
        """ set_fname(fname) sets the output file name """
        self._fname = fname

    def fname(self) -> str:
        """ fname() returns the output file name """
        return(self._fname)

    def set_format(self, fmt: int):
        """ set_format(fmt) sets the format written when the content does not carry one """
        self._format = fmt

    def format(self) -> int:
        """ format() returns the format written when the content does not carry one """
        return(self._format)

    def set_index_bytes(self, index_bytes: int):
        """ set_index_bytes(index_bytes) sets the node-count width (4 or 8) of formats 2 and 3 """
        self._index_bytes = index_bytes

    def index_bytes(self) -> int:
        """ index_bytes() returns the node-count width of formats 2 and 3 """
        return(self._index_bytes)

    def set_revision(self, revision: int):
        """ set_revision(revision) sets the revision written when the content does not carry one """
        self._revision = revision

    def revision(self) -> int:
        """ revision() returns the revision written when the content does not carry one """
        return(self._revision)

    def write(self, content: dict, fname: str = None):
        """ write(content, fname) writes content (the dict RoeReader.read()
            returns) to fname (default: fname()). 'format', 'revision',
            'created' and 'index_bytes' are taken from the content when it has
            them, so a file read and written back is identical; the writer's
            settings fill in what is missing.
        """
        outname = fname if fname is not None else self._fname
        try:
            fmt = int(content.get('format', self._format))
            if fmt not in ROE_FORMATS:
                raise ValueError('cannot write {}: format {} is not supported ({})'.format(
                    outname, fmt, ROE_FORMATS))
            nnodes = int(content['num_nodes'])
            mask   = np.reshape(np.asarray(content['mask']), (-1,)).astype(np.int8)
            if mask.size != nnodes:
                raise ValueError('cannot write {}: the mask has {} values for {} nodes'.format(
                    outname, mask.size, nnodes))
            header  = self.__header(content, fmt, nnodes)
            globals_ = self.__globals(content['globals'], nnodes, outname)
            regions = self.__regions(content['regions'], fmt, outname)
            folder  = os.path.dirname(outname)
            if len(folder) > 0:
                os.makedirs(folder, exist_ok=True)
            # write next to the target, then rename: a run killed while writing
            # leaves the previous file intact instead of a truncated one
            partial = '{}.partial'.format(outname)
            with open(partial, 'wb') as fstate:
                fstate.write(header)
                fstate.write(struct.pack('<i', len(globals_)))
                for name, values in globals_:
                    fstate.write(self.__string(name))
                    fstate.write(values.tobytes())
                fstate.write(regions)
                fstate.write(mask.tobytes())
                self.__write_records(fstate, content, mask, outname)
            os.replace(partial, outname)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __header(self, content: dict, fmt: int, nnodes: int) -> bytes:
        """ everything before the global vectors. The time is stored as
            float32, as the reference does: a time that float32 cannot hold
            exactly is rounded here, and the reader of the file must snap it
            back to its time step.
        """
        revision = int(content.get('revision', self._revision))
        created  = content.get('created', None)
        if created is None:
            created = int(time.time())
        index_bytes = 4 if fmt == 1 else int(content.get('index_bytes', self._index_bytes))
        if index_bytes not in (4, 8):
            raise ValueError('index_bytes must be 4 or 8, not {}'.format(index_bytes))
        data = self.__string(ROE_MAGIC)
        data += struct.pack('<II', fmt, revision)
        data += struct.pack('<q', int(created))
        data += np.float32(content['time']).astype('<f4').tobytes()
        data += struct.pack('<i' if index_bytes == 4 else '<q', nnodes)
        return(data)

    def __globals(self, vectors: dict, nnodes: int, outname: str) -> list:
        """ the global vectors as (name, float64 bytes), in the order the
            reference writes them. A name it does not know goes last, in the
            order given, so that a file read back keeps what it carried.
        """
        known   = [name for name in ROE_GLOBAL_NAMES if name in vectors]
        unknown = [name for name in vectors.keys() if name not in ROE_GLOBAL_NAMES]
        ordered : list = []
        for name in known + unknown:
            values = np.reshape(np.asarray(vectors[name], dtype='<f8'), (-1,))
            if values.size != nnodes:
                raise ValueError('cannot write {}: {} has {} values for {} nodes'.format(
                    outname, name, values.size, nnodes))
            ordered.append((name, values))
        return(ordered)

    def __regions(self, regions: list, fmt: int, outname: str) -> bytes:
        """ the table of ionic regions and their plugins """
        data = struct.pack('<i', len(regions))
        for region in regions:
            data += self.__imp(region['model'], region['size'], region.get('fingerprint'),
                               fmt, outname)
            data += struct.pack('<i', len(region['plugins']))
            for plugin in region['plugins']:
                # 'stored_size' reproduces a release-19.0 table (see RoeReader);
                # without it the table holds the true size
                data += self.__imp(plugin['name'], plugin.get('stored_size', plugin['size']),
                                   plugin.get('fingerprint'), fmt, outname)
        return(data)

    def __imp(self, name: str, size: int, fprint: int, fmt: int, outname: str) -> bytes:
        """ one (name, record size, fingerprint) entry of the region table """
        data = self.__string(name) + struct.pack('<i', int(size))
        if fmt >= 3:
            if fprint is None:
                raise ValueError('cannot write {}: format 3 needs the fingerprint of {}'.format(
                    outname, name))
            data += struct.pack('<Q', int(fprint))
        return(data)

    def __write_records(self, fstate, content: dict, mask: np.ndarray, outname: str):
        """ the per-node records, in canonical node order """
        regions = content['regions']
        records = content['records']
        sizes   = np.array([reg['size'] + sum(p['size'] for p in reg['plugins'])
                            for reg in regions], dtype=np.int64)
        for region, rows in enumerate(records):
            expected = (int(np.count_nonzero(mask == region)), int(sizes[region]))
            if tuple(np.shape(rows)) != expected:
                raise ValueError('cannot write {}: region {} has records of shape {}, {} '
                                 'expected'.format(outname, region, np.shape(rows), expected))
        if np.all(sizes == sizes[0]):
            out = np.empty((mask.size, int(sizes[0])), dtype=np.uint8)
            for region, rows in enumerate(records):
                out[mask == region] = rows
            fstate.write(out.tobytes())
            return
        offsets = np.concatenate(([0], np.cumsum(sizes[mask])[:-1])).astype(np.int64)
        out     = np.empty(int(np.sum(sizes[mask])), dtype=np.uint8)
        for region, rows in enumerate(records):
            step  = np.arange(int(sizes[region]), dtype=np.int64)
            nodes = np.flatnonzero(mask == region)
            for first in range(0, nodes.size, _CHUNK_ROWS):
                block = nodes[first:first + _CHUNK_ROWS]
                out[offsets[block][:, None] + step[None, :]] = rows[first:first + block.size]
        fstate.write(out.tobytes())

    def __string(self, text: str) -> bytes:
        """ a string as the reference stores it: int32 length + characters """
        encoded = text.encode('ascii')
        return(struct.pack('<i', len(encoded)) + encoded)
