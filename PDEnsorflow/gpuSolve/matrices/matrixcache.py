import os
import json
import hashlib
import numpy as np
from gpuSolve._version import __version__


# Bumped whenever the meaning or the layout of the stored arrays changes. A
# file written by an older layout is rejected like a key mismatch, so an old
# cache left on disk costs one reassembly and never a wrong matrix.
MATRIX_CACHE_FORMAT_VERSION : int = 1

# Read in blocks rather than whole: the binary meshes this cache is meant for
# are of the order of 1 GB, and the digest does not need them in memory at once.
_HASH_BLOCK_BYTES : int = 16 * 1024 * 1024


class MatrixCache:
    """ A cache of the assembled mass and stiffness matrices on disk.

    Assembling the operators of a large mesh costs minutes (on a 4.0 M-node,
    23 M-tetrahedron mesh: 28 s of mesh connectivity, 1 s of sparsity pattern
    and 131 s of assembly), and a study that runs the same mesh and the same
    conductivities many times over, varying only the stimulus or the duration,
    pays it again on every invocation. Nothing that enters the two matrices
    changes between those runs, so they are stored once and read back.

    What is stored is the host CSR form of the assembly: one row pointer and
    one column array shared by both matrices, one values array each, and the
    node renumbering. The renumbering is part of the cache because the solver
    uses it for the whole run and not only during the assembly, and recomputing
    it would need the sparsity pattern, hence the mesh connectivity, which is
    what the cache exists to skip.

    The system matrix A, the right-hand side matrix and the preconditioner are
    NOT stored: each is one sparse axpby on an existing pattern, so rebuilding
    them costs milliseconds and keeps dt and theta out of the cache key. A
    convergence study varies exactly those two, and a cache that invalidated on
    them would be of no use for it.

    The key holds everything the matrices depend on: the content digest of the
    mesh file, the per-region sigma_l, sigma_t and beta, whether the nodes are
    renumbered, and the gpuSolve version. It is compared whole, and ANY
    difference means reassembling: a cache that silently substituted the wrong
    conductivities would run physics nobody asked for and say nothing, which is
    far worse than being slow.
    """

    def __init__(self, config: dict = None):
        self._fname : str          = ''
        self._key : dict          = None
        self._matrix_names : tuple = ('mass', 'stiffness')
        if config is not None:
            for attribute, value in config.items():
                if '_'+attribute in self.__dict__.keys():
                    setattr(self, '_'+attribute, value)
                else:
                    print('warning: attribute {} not in MatrixCache'.format(attribute))

    def fname(self) -> str:
        """ fname() returns the name of the cache file """
        return(self._fname)

    def set_fname(self, fname: str):
        """ set_fname(fname) sets the name of the cache file """
        self._fname = fname

    def key(self) -> dict:
        """ key() returns the key the file is required to carry """
        return(self._key)

    def set_key(self, key: dict):
        """ set_key(key) sets the key the file is required to carry """
        self._key = key

    def matrix_names(self) -> tuple:
        """ matrix_names() returns the names of the matrices the file stores """
        return(self._matrix_names)

    def set_matrix_names(self, matrix_names: tuple):
        """ set_matrix_names(matrix_names) sets the names of the stored matrices """
        self._matrix_names = tuple(matrix_names)

    def active(self) -> bool:
        """ active() is True when a cache file has been named """
        return(len(self._fname) > 0)

    def file_fingerprint(self, fname: str) -> dict:
        """ file_fingerprint(fname) returns the size and the sha256 digest of
            the content of fname.
            The content is hashed, not the name and not the modification time:
            a mesh rewritten in place keeps both of those, and would be read
            from the cache as if it were the old one. Hashing 1 GB costs a
            couple of seconds against the minutes the cache saves.
        """
        digest = hashlib.sha256()
        nbytes = 0
        with open(fname, 'rb') as fin:
            while True:
                block = fin.read(_HASH_BLOCK_BYTES)
                if len(block) == 0:
                    break
                nbytes += len(block)
                digest.update(block)
        return({'sha256': digest.hexdigest(), 'bytes': nbytes})

    def version_string(self) -> str:
        """ version_string() returns the gpuSolve version, for the key """
        return('.'.join(['{}'.format(x) for x in __version__]))

    def load(self) -> dict:
        """ load() returns the stored arrays when the file exists and carries
            this cache's key, and None otherwise.
            None is the signal to assemble: a missing file, an unreadable one,
            a file of another format version and a key that differs in any
            field are all answered the same way, by recomputing. Only an
            inconsistency WITHIN a file that the key accepted raises, because
            that is a corrupt cache rather than a stale one, and continuing
            with it would mean solving on a mangled operator.
        """
        if not self.active() or not os.path.isfile(self._fname):
            return(None)
        # the members are read inside the try and checked outside it, so that
        # a file that cannot be read falls back to the assembly while a file
        # that was read and does not hold a consistent matrix raises
        try:
            with np.load(self._fname, allow_pickle=False) as stored:
                if int(stored['format_version']) != MATRIX_CACHE_FORMAT_VERSION:
                    print('matrix cache {}: format version {} is not {}; reassembling'.format(
                        self._fname, int(stored['format_version']), MATRIX_CACHE_FORMAT_VERSION), flush=True)
                    return(None)
                if str(stored['key']) != self.__key_string():
                    print('matrix cache {}: the key does not match this run; '
                          'reassembling'.format(self._fname), flush=True)
                    return(None)
                content = {name: stored[name] for name in stored.files}
        except Exception as err:
            # a truncated or non-npz file is a stale cache, not a reason to stop
            print('matrix cache {}: cannot be read ({}); reassembling'.format(self._fname, err), flush=True)
            return(None)
        arrays = self.__arrays_from_content(content)
        print('matrix cache {}: loaded {} entries on {} nodes'.format(
            self._fname, arrays['nnz'], arrays['npt']), flush=True)
        return(arrays)

    def save(self, arrays: dict, renumbering: dict = None):
        """ save(arrays,renumbering) writes the CSR arrays, the renumbering and
            the key to the cache file.
            The file is written beside its final name and then renamed, which is
            atomic on one file system: a reader of a cache being rewritten sees
            either the whole old file or the whole new one. Several short runs
            of the same study are launched together, so a half-written file is
            a case that does happen.
        """
        if not self.active():
            return
        to_store = {'format_version': np.asarray(MATRIX_CACHE_FORMAT_VERSION, dtype=np.int64),
                    'key': np.asarray(self.__key_string()),
                    'npt': np.asarray(int(arrays['npt']), dtype=np.int64),
                    'nnz': np.asarray(int(arrays['nnz']), dtype=np.int64),
                    'indptr': arrays['indptr'].astype(np.int64),
                    'col': arrays['col'].astype(np.int32)}
        for matr_name in self._matrix_names:
            to_store['values_'+matr_name] = arrays['values'][matr_name].astype(np.float32)
        if renumbering is not None:
            to_store['perm']  = np.asarray(renumbering['perm']).astype(np.int32)
            to_store['iperm'] = np.asarray(renumbering['iperm']).astype(np.int32)
        tmpname = '{}.tmp.{}'.format(self._fname, os.getpid())
        try:
            with open(tmpname, 'wb') as fout:
                np.savez(fout, **to_store)
            os.replace(tmpname, self._fname)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            if os.path.isfile(tmpname):
                os.remove(tmpname)
            raise
        print('matrix cache {}: written ({:3.2f} MB)'.format(
            self._fname, os.path.getsize(self._fname) / (1024.0 * 1024.0)), flush=True)

    def __key_string(self) -> str:
        """ the key as one canonical string, so that two keys that hold the same
            values compare equal whatever order their dicts were built in, and
            so that the comparison is one string comparison rather than a walk
            of a nested structure of numpy scalars
        """
        return(json.dumps(self._key, sort_keys=True, default=str))

    def __arrays_from_content(self, content: dict) -> dict:
        """ takes the members read from the file and checks that they describe
            one matrix shape. The shapes are checked rather than trusted: the
            key says the file was written for this problem, not that it was
            written completely, and a column array one entry short would give a
            matrix that is wrong without being invalid. It raises, because a
            file the key accepted cannot be answered by reassembling: it is a
            damaged cache and not a stale one.
        """
        npt    = int(content['npt'])
        nnz    = int(content['nnz'])
        indptr = content['indptr']
        col    = content['col']
        if indptr.shape != (npt + 1,):
            raise ValueError('matrix cache {}: indptr has shape {}, expected {}'.format(
                self._fname, indptr.shape, (npt + 1,)))
        if col.shape != (nnz,):
            raise ValueError('matrix cache {}: col has shape {}, expected {}'.format(
                self._fname, col.shape, (nnz,)))
        if int(indptr[-1]) != nnz:
            raise ValueError('matrix cache {}: indptr ends at {}, expected nnz = {}'.format(
                self._fname, int(indptr[-1]), nnz))
        values = {}
        for matr_name in self._matrix_names:
            vals = content['values_'+matr_name]
            if vals.shape != (nnz,):
                raise ValueError('matrix cache {}: the values of {} have shape {}, expected {}'.format(
                    self._fname, matr_name, vals.shape, (nnz,)))
            values[matr_name] = vals
        renumbering = None
        if 'perm' in content.keys():
            perm  = content['perm']
            iperm = content['iperm']
            if perm.shape != (npt,) or iperm.shape != (npt,):
                raise ValueError('matrix cache {}: perm/iperm have shapes {}/{}, expected {}'.format(
                    self._fname, perm.shape, iperm.shape, (npt,)))
            renumbering = {'perm': perm, 'iperm': iperm}
        return({'npt': npt, 'nnz': nnz, 'indptr': indptr, 'col': col,
                'values': values, 'renumbering': renumbering})
