#!/usr/bin/env python
"""
    RoeReader: reads a binary ionic state file (`.roe`, a "MIIF dump") as the
    reference simulator writes it for a tissue run (tsav, chkpt_intv) and reads
    it back with start_statef.

    Layout, every value little endian, nothing between the items:

        magic             int32 length + characters, "Dump_MIIF_ID", no NUL
        format            uint32: 1, 2 or 3
        revision          uint32: revision of the program that wrote the file
        created           int64 (time_t): wall-clock time of the write
        time              float32: simulation time of the state, in ms
        num_nodes         int32 (format 1) or the build's global index type,
                          int32 or int64 (formats 2 and 3)
        num_globals       int32
        num_globals x     name (int32 length + characters), then num_nodes
                          float64 values in canonical node order
        num_regions       int32
        num_regions x     model name, record size (int32), fingerprint (uint64,
                          format 3 only), number of plugins (int32), then per
                          plugin: name, record size (int32), fingerprint
                          (uint64, format 3 only)
        mask              num_nodes int8: region index of each node, canonical
                          order
        records           one record per node, canonical order: the model's
                          state, then each plugin's state, in the order above

    A record holds the fields of the model's state structure, in the order of
    the structure. Formats 1 and 2 copy the C structure as it is in memory, so
    the fields carry C alignment and the record ends with the padding that
    rounds it to the largest alignment (the padding bytes are zero: the
    reference allocates its state tables zero-filled). Format 3 writes the same
    fields packed, without padding, and adds a fingerprint of the field list,
    so that a restart against a model whose fields changed is refused.

    Release 19.0 of the reference (format 1) stores a wrong plugin size in the
    region table: the size of the region's model, not the plugin's (its
    dump_state writes the model's sv_size in the plugin loop), while the node
    records carry the plugin's true size. The reader recognises that case,
    every plugin entry equal to its model's size and a file size that does not
    add up, and recovers the true sizes (see __resolve_plugin_sizes). Each
    plugin entry keeps both: 'size', the bytes of its records, and
    'stored_size', the value of the table, so a file written back is
    identical. That release also restores its own files with the wrong
    offsets, so it cannot restart from a file of its own that has a plugin.

    The reader does not need to know the models to read a file: the records are
    returned as raw bytes, region by region. region_state() decodes them once
    the caller supplies the model's field list (see record_dtype()).

    Canonical order is the node order of the mesh files, which is also the
    order of every nodal array gpuSolve hands to its users.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import struct

import numpy as np


# The string that opens every file.
ROE_MAGIC : str = 'Dump_MIIF_ID'

# The formats this reader understands. 1 is written by release 19.0 of the
# reference; 3 by its later sources (packed records with fingerprints); 2 only
# existed between the two and is read like 1 with a wider node count.
ROE_FORMATS : tuple = (1, 2, 3)

# The global nodal quantities a file can carry, in the order the reference
# writes them (its IMP data types, limpet/models/imp_list.txt).
ROE_GLOBAL_NAMES : tuple = ('Vm', 'Lambda', 'delLambda', 'Tension', 'Ke', 'Nae', 'Cae', 'Iion',
                            'tension_component', 'illum')

# C type of a state-structure field -> (numpy dtype, type code). The type code
# is the one the reference hashes into the format-3 fingerprint (typeID in its
# limpet/src/python/limpetcommon.py). Gatetype is single precision and
# GlobalData_t double precision in the reference's builds; every model gpuSolve
# provides uses only those two.
ROE_FIELD_TYPES : dict = {'Gatetype':     ('<f4', 0),
                          'Real':         ('<f8', 1),
                          'float':        ('<f4', 2),
                          'char':         ('i1',  3),
                          'int':          ('<i4', 4),
                          'double':       ('<f8', 5),
                          'short':        ('<i2', 6),
                          'GlobalData_t': ('<f8', 7),
                          'long':         ('<i8', 8),
                          'bool':         ('u1',  9)}

# Bounds used to tell a 32-bit node count from a 64-bit one in formats 2 and 3,
# which do not record the width (it is a build option of the reference).
_MAX_NAME_LENGTH : int = 256

# Rows handled per pass when regions have records of different sizes: the
# gather index has rows x record-size entries, so this keeps it near 100 MB.
_CHUNK_ROWS : int = 32768


def record_dtype(fields: list, packed: bool) -> np.dtype:
    """ record_dtype(fields, packed) returns the numpy dtype of one record of a
        state structure. fields is the ordered list of (name, C type), C type
        a key of ROE_FIELD_TYPES. packed=False reproduces the C layout
        (formats 1 and 2: aligned fields, trailing padding); packed=True puts
        the fields side by side (format 3).
    """
    names   = [name for name, _ctype in fields]
    formats = [ROE_FIELD_TYPES[ctype][0] for _name, ctype in fields]
    return(np.dtype({'names': names, 'formats': formats}, align=not packed))


def fingerprint(fields: list) -> int:
    """ fingerprint(fields) returns the format-3 fingerprint of a state
        structure: 64-bit FNV-1a over, per field in order, its name, a zero
        byte (so that {"ab", "c"} and {"a", "bc"} differ), its size and its type
        code, both as 4 little-endian bytes (IonIfBase::sv_fingerprint)
    """
    value : int = 0xcbf29ce484222325
    prime : int = 0x100000001b3
    for name, ctype in fields:
        dtype, code = ROE_FIELD_TYPES[ctype]
        payload = (name.encode('ascii') + b'\x00'
                   + struct.pack('<I', np.dtype(dtype).itemsize) + struct.pack('<I', code))
        for byte in payload:
            value = ((value ^ byte) * prime) & 0xffffffffffffffff
    return(value)


class RoeReader:
    """
    class RoeReader: utility to read a binary ionic state file.
    Provided a file name, this class returns its content as a dict:

        'format', 'revision', 'created'   header values, as stored
        'time'                            float32 time, widened to float
        'num_nodes'                       int
        'index_bytes'                     4 or 8: width of the node count
        'globals'                         {name: float64 array}, file order
        'regions'                         [{'model', 'size', 'fingerprint',
                                            'plugins': [{'name', 'size',
                                                         'stored_size',
                                                         'fingerprint'}]}]
        'mask'                            int8 array, region of each node
        'records'                         [uint8 array (nodes of the region,
                                           record size)], one per region, rows
                                           in canonical node order

    'fingerprint' is None in formats 1 and 2, which do not store it. 'size' is
    the number of bytes of a record; a plugin's 'stored_size' is the value of
    the region table, which differs from 'size' in files of release 19.0.
    """

    def __init__(self, config: dict = None):
        self._index_bytes : int   = None
        # {plugin name: bytes per node}, for a file whose region table holds
        # wrong plugin sizes and whose own size cannot settle them
        self._plugin_sizes : dict = None
        self.__fname : str      = None
        self.__content : dict   = None

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_index_bytes(self, index_bytes: int):
        """ set_index_bytes(index_bytes) fixes the width (4 or 8) of the node
            count of formats 2 and 3. None (the default) detects it.
        """
        self._index_bytes = index_bytes

    def index_bytes(self) -> int:
        """ index_bytes() returns the node-count width set by the user (None: detected) """
        return(self._index_bytes)

    def set_plugin_sizes(self, plugin_sizes: dict):
        """ set_plugin_sizes(plugin_sizes) gives the record size of plugins,
            {name: bytes per node}, for a release-19.0 file whose size alone
            cannot settle them (two different plugins in one file)
        """
        self._plugin_sizes = plugin_sizes

    def plugin_sizes(self) -> dict:
        """ plugin_sizes() returns the plugin record sizes given by the user """
        return(self._plugin_sizes)

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def content(self) -> dict:
        """ content() returns the content of the last file read; None before """
        return(self.__content)

    def time(self) -> float:
        """ time() returns the simulation time of the state, in ms """
        return(self.__content['time'])

    def num_nodes(self) -> int:
        """ num_nodes() returns the number of nodes of the state """
        return(self.__content['num_nodes'])

    def region_nodes(self, region: int) -> np.ndarray:
        """ region_nodes(region) returns the canonical indices of the nodes of
            a region, in the order of its records
        """
        return(np.flatnonzero(self.__content['mask'] == region))

    def region_state(self, region: int, fields: list, plugin: int = None) -> dict:
        """ region_state(region, fields, plugin) decodes the records of a region
            and returns {field name: array, one value per node of the region}.
            fields is the ordered (name, C type) list of the state structure of
            the region's model (plugin None) or of its plugin number `plugin`.
            The arrays keep the stored precision (float32 for a Gatetype).
        """
        try:
            info   = self.__content['regions'][region]
            packed = self.__content['format'] >= 3
            dtype  = record_dtype(fields, packed)
            if plugin is None:
                start, size, name = 0, info['size'], info['model']
            else:
                start = info['size'] + sum(p['size'] for p in info['plugins'][:plugin])
                size  = info['plugins'][plugin]['size']
                name  = info['plugins'][plugin]['name']
            if dtype.itemsize != size:
                raise ValueError('{}: region {} ({}) stores {} bytes per node, the fields given '
                                 'describe {}'.format(self.__fname, region, name, size,
                                                      dtype.itemsize))
            raw    = np.ascontiguousarray(self.__content['records'][region][:, start:start + size])
            values = raw.view(dtype).reshape(-1)
            return({field: np.array(values[field]) for field in dtype.names})
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def read(self, fname: str) -> dict:
        """ read(fname) reads the state file fname, checks that its parts are
            consistent and that nothing is missing or left over, and returns
            its content (see the class description)
        """
        try:
            with open(fname, 'rb') as fstate:
                content = self.__read_header(fstate, fname)
                content['globals'] = self.__read_globals(fstate, content['num_nodes'], fname)
                content['regions'] = self.__read_regions(fstate, content['format'], fname)
                content['mask']    = self.__read_mask(fstate, content, fname)
                content['records'] = self.__read_records(fstate, content, fname)
            self.__content = content
            self.__fname   = fname
            return(self.__content)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __read_header(self, fstate, fname: str) -> dict:
        """ reads everything before the global vectors """
        magic = self.__read_string(fstate, fname)
        if magic != ROE_MAGIC:
            raise ValueError('{}: not an ionic state file (it opens with "{}", not "{}")'.format(
                fname, magic, ROE_MAGIC))
        fmt, revision = struct.unpack('<II', self.__read_bytes(fstate, 8, fname))
        if fmt not in ROE_FORMATS:
            raise ValueError('{}: format {} is not supported (known: {})'.format(
                fname, fmt, ROE_FORMATS))
        created, = struct.unpack('<q', self.__read_bytes(fstate, 8, fname))
        ctime,   = struct.unpack('<f', self.__read_bytes(fstate, 4, fname))
        index_bytes = 4 if fmt == 1 else self.__index_width(fstate, fname)
        code        = '<i' if index_bytes == 4 else '<q'
        num_nodes,  = struct.unpack(code, self.__read_bytes(fstate, index_bytes, fname))
        if num_nodes <= 0:
            raise ValueError('{}: invalid number of nodes {}'.format(fname, num_nodes))
        return({'format': fmt, 'revision': revision, 'created': created,
                'time': float(ctime), 'num_nodes': num_nodes, 'index_bytes': index_bytes})

    def __index_width(self, fstate, fname: str) -> int:
        """ the node count of formats 2 and 3 is the reference's global index
            type, 32 or 64 bits by build option, and the file does not say
            which. Unless the user fixed it, each width is tried: the right one
            is followed by a plausible global-vector count and, when there is a
            global vector, by the length and the name of a known one. 64 bits
            are tried first: a 64-bit count read as 32 bits leaves its upper
            half, zero, where the vector count goes, and "no global vector"
            would be accepted; a 32-bit count read as 64 bits instead takes the
            vector count as its upper half and then fails the name check.
        """
        if self._index_bytes is not None:
            if self._index_bytes not in (4, 8):
                raise ValueError('index_bytes must be 4 or 8, not {}'.format(self._index_bytes))
            return(self._index_bytes)
        start = fstate.tell()
        peek  = fstate.read(8 + 4 + 4 + _MAX_NAME_LENGTH)
        fstate.seek(start)
        for width in (8, 4):
            if len(peek) < width + 4:
                continue
            count, = struct.unpack('<i' if width == 4 else '<q', peek[:width])
            nglob, = struct.unpack('<i', peek[width:width + 4])
            if count <= 0 or nglob < 0 or nglob > len(ROE_GLOBAL_NAMES):
                continue
            if nglob == 0:
                return(width)
            length, = struct.unpack('<i', peek[width + 4:width + 8])
            name    = peek[width + 8:width + 8 + max(length, 0)]
            if 0 < length <= _MAX_NAME_LENGTH and name.decode('ascii', 'replace') in ROE_GLOBAL_NAMES:
                return(width)
        raise ValueError('{}: cannot tell whether the node count is 32 or 64 bits wide; '
                         'set index_bytes'.format(fname))

    def __read_globals(self, fstate, num_nodes: int, fname: str) -> dict:
        """ reads the global nodal vectors. A name this reader does not know is
            kept: the reference skips such a vector on a restart, and only the
            caller knows whether it matters.
        """
        count, = struct.unpack('<i', self.__read_bytes(fstate, 4, fname))
        if count < 0 or count > len(ROE_GLOBAL_NAMES):
            raise ValueError('{}: invalid number of global vectors {}'.format(fname, count))
        vectors : dict = {}
        for _index in range(count):
            name   = self.__read_string(fstate, fname)
            values = np.frombuffer(self.__read_bytes(fstate, 8 * num_nodes, fname), dtype='<f8')
            vectors[name] = values.astype(np.float64)
        return(vectors)

    def __read_regions(self, fstate, fmt: int, fname: str) -> list:
        """ reads the table of ionic regions and their plugins """
        count, = struct.unpack('<i', self.__read_bytes(fstate, 4, fname))
        if count <= 0 or count > 127:
            # the mask is one signed byte per node, so 127 regions at most
            raise ValueError('{}: invalid number of ionic regions {}'.format(fname, count))
        regions : list = []
        for _index in range(count):
            model, size, fprint = self.__read_imp(fstate, fmt, fname)
            nplug, = struct.unpack('<i', self.__read_bytes(fstate, 4, fname))
            if nplug < 0:
                raise ValueError('{}: invalid number of plugins {} for {}'.format(fname, nplug, model))
            plugins : list = []
            for _plug in range(nplug):
                name, psize, pprint = self.__read_imp(fstate, fmt, fname)
                plugins.append({'name': name, 'size': psize, 'stored_size': psize,
                                'fingerprint': pprint})
            regions.append({'model': model, 'size': size, 'fingerprint': fprint,
                            'plugins': plugins})
        return(regions)

    def __read_imp(self, fstate, fmt: int, fname: str) -> tuple:
        """ reads one (name, record size, fingerprint) entry of the region table """
        name  = self.__read_string(fstate, fname)
        size, = struct.unpack('<i', self.__read_bytes(fstate, 4, fname))
        if size < 0:
            raise ValueError('{}: invalid record size {} for {}'.format(fname, size, name))
        fprint = None
        if fmt >= 3:
            fprint, = struct.unpack('<Q', self.__read_bytes(fstate, 8, fname))
        return((name, size, fprint))

    def __read_mask(self, fstate, content: dict, fname: str) -> np.ndarray:
        """ reads the region index of each node and checks it names a region """
        nnodes = content['num_nodes']
        mask   = np.frombuffer(self.__read_bytes(fstate, nnodes, fname), dtype=np.int8).copy()
        nreg   = len(content['regions'])
        if np.any(mask < 0) or np.any(mask >= nreg):
            raise ValueError('{}: the region mask names a region outside 0..{}'.format(fname, nreg - 1))
        return(mask)

    def __read_records(self, fstate, content: dict, fname: str) -> list:
        """ reads the per-node records and splits them by region. The size of
            the rest of the file must be exactly the sum of the records: a
            shorter file is truncated, a longer one is not what the header says.
        """
        mask   = content['mask']
        counts = np.bincount(mask, minlength=len(content['regions'])).astype(np.int64)
        rest   = os.fstat(fstate.fileno()).st_size - fstate.tell()
        self.__resolve_plugin_sizes(content, counts, rest, fname)
        sizes  = np.array([reg['size'] + sum(p['size'] for p in reg['plugins'])
                           for reg in content['regions']], dtype=np.int64)
        total  = int(np.sum(counts * sizes))
        if rest != total:
            raise ValueError('{}: {} bytes of node records expected, {} found'.format(
                fname, total, rest))
        data = np.fromfile(fstate, dtype=np.uint8, count=total)
        if np.all(sizes == sizes[0]):
            # every node has a record of the same size: one reshape, then the
            # rows of each region are a boolean selection, with no index array
            rows = data.reshape(-1, int(sizes[0]))
            return([rows[mask == region] for region in range(sizes.size)])
        return(self.__split_mixed(data, mask, sizes))

    def __resolve_plugin_sizes(self, content: dict, counts: np.ndarray, rest: int, fname: str):
        """ recovers the true plugin record sizes of a release-19.0 file (see
            the module description). Nothing is changed unless the table does
            not add up to the file size AND every plugin entry carries its
            model's size, the signature of that release. The sizes come from
            set_plugin_sizes() when given; otherwise, when the file has a single
            plugin kind, from the one equation the file size gives:
                rest = sum over regions of nodes x (model size + plugins x size)
        """
        regions = content['regions']
        plugins = [plug for reg in regions for plug in reg['plugins']]
        if len(plugins) == 0 or content['format'] >= 3:
            return
        stated = int(np.sum(counts * np.array([reg['size'] + sum(p['size'] for p in reg['plugins'])
                                               for reg in regions], dtype=np.int64)))
        if stated == rest:
            return
        if not all(plug['stored_size'] == reg['size'] for reg in regions for plug in reg['plugins']):
            return
        known = dict(self._plugin_sizes) if self._plugin_sizes is not None else {}
        kinds = sorted(set(plug['name'] for plug in plugins) - set(known.keys()))
        if len(kinds) > 1:
            raise ValueError('{}: the region table holds wrong plugin sizes and the file size '
                             'cannot settle {} plugins ({}); give them with set_plugin_sizes'.format(
                                 fname, len(kinds), ', '.join(kinds)))
        if len(kinds) == 1:
            fixed = sum(int(counts[i]) * (reg['size'] + sum(int(known[p['name']]) for p in reg['plugins']
                                                            if p['name'] in known))
                        for i, reg in enumerate(regions))
            users = sum(int(counts[i]) * sum(1 for p in reg['plugins'] if p['name'] == kinds[0])
                        for i, reg in enumerate(regions))
            if users == 0 or (rest - fixed) <= 0 or (rest - fixed) % users != 0:
                raise ValueError('{}: the region table holds wrong plugin sizes and the file size '
                                 'gives no whole size for {}'.format(fname, kinds[0]))
            known[kinds[0]] = (rest - fixed) // users
        for plug in plugins:
            plug['size'] = int(known[plug['name']])

    def __split_mixed(self, data: np.ndarray, mask: np.ndarray, sizes: np.ndarray) -> list:
        """ splits records of different sizes, node by node, in chunks so that
            the gather index stays small on a large mesh
        """
        offsets = np.concatenate(([0], np.cumsum(sizes[mask])[:-1])).astype(np.int64)
        records : list = []
        for region in range(sizes.size):
            size  = int(sizes[region])
            nodes = np.flatnonzero(mask == region)
            out   = np.empty((nodes.size, size), dtype=np.uint8)
            step  = np.arange(size, dtype=np.int64)
            for first in range(0, nodes.size, _CHUNK_ROWS):
                block = nodes[first:first + _CHUNK_ROWS]
                out[first:first + block.size] = data[offsets[block][:, None] + step[None, :]]
            records.append(out)
        return(records)

    def __read_string(self, fstate, fname: str) -> str:
        """ reads a string stored as int32 length + characters, no terminator """
        length, = struct.unpack('<i', self.__read_bytes(fstate, 4, fname))
        if length < 0 or length > _MAX_NAME_LENGTH:
            raise ValueError('{}: invalid string length {}'.format(fname, length))
        return(self.__read_bytes(fstate, length, fname).decode('ascii'))

    def __read_bytes(self, fstate, count: int, fname: str) -> bytes:
        """ reads exactly count bytes, or reports where the file ends """
        data = fstate.read(count)
        if len(data) != count:
            raise ValueError('{}: the file ends at byte {} while {} more bytes were expected'.format(
                fname, fstate.tell(), count - len(data)))
        return(data)
