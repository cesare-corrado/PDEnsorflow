import os
import numpy as np
import tensorflow as tf


class BaseWriter:
    """
    Class BaseWriter
    Base class for the GPU-resident solution writers.

    It implements a variable-size container that receives the solution
    (TensorFlow GPU tensors) at given time steps and keeps it in GPU memory.
    Because TensorFlow allocates GPU memory greedily, the buffer is flushed
    to disk in chunks instead of probing the hardware memory:
      * every_N    : if not None, the chunk is flushed every N received solutions
      * max_chunk_mb : if every_N is None, the chunk is flushed as soon as the
                       accumulated data exceeds max_chunk_mb megabytes.
    At the end of the run, all the chunks are aggregated and saved to disk as a
    single (NumPy) array.

    Derived classes only implement the format specific saving logic by
    overriding _write_chunk(), _aggregate() and _final_path().
    """

    _MB = 1024 * 1024  # bytes in one megabyte (MiB)

    def __init__(self, config: dict = None):
        self._fname        : str   = 'out'
        self._every_N      : int   = None      # iteration-based flush trigger
        self._max_chunk_mb : float = 500       # memory-based flush trigger (MB)
        self._buffer       : list  = []        # GPU-resident list of tensors
        self._chunk_files  : list  = []        # temporary chunk files on disk
        self._counter      : int   = 0         # nb of received solutions
        self._chunk_index  : int   = 0         # nb of flushed chunks
        self._chunk_bytes  : int   = 0         # estimated bytes in current buffer

        if(config is not None):
            for attribute in ['_fname','_every_N','_max_chunk_mb']:
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    def set_fname(self, fname: str):
        """ set_fname(fname) sets the output file name """
        self._fname = fname

    def set_every_N(self, every_N: int):
        """ set_every_N(every_N) sets the iteration-based flush trigger """
        self._every_N = every_N

    def set_max_chunk_mb(self, max_chunk_mb: float):
        """ set_max_chunk_mb(max_chunk_mb) sets the memory-based flush trigger (MB) """
        self._max_chunk_mb = max_chunk_mb

    def counter(self) -> int:
        """ counter(): returns the number of received solutions """
        return(self._counter)

    def nb_chunks(self) -> int:
        """ nb_chunks(): returns the number of chunks flushed so far """
        return(self._chunk_index)

    def add_solution(self, data):
        """add_solution(data): receives one solution (a TensorFlow GPU tensor or
        a numpy array) and stores it in the GPU buffer. The chunk is flushed to
        disk when the flush condition (every_N or max_chunk_mb) is met.
        """
        # keep the solution on the GPU (tf.identity makes a
        # snapshot, so in-place updates of the source tensor do not corrupt the
        # buffer) instead of forcing a host transfer with .numpy() at every step
        tensor = tf.identity(tf.convert_to_tensor(data))
        self._buffer.append(tensor)
        self._counter    += 1
        self._chunk_bytes += self._estimate_frame_bytes(tensor)
        if self._should_flush():
            self.flush_chunk()

    def flush_chunk(self):
        """flush_chunk(): stacks the buffered GPU tensors, transfers them to the
        host with a single .numpy() call and hands the chunk to the format
        specific writer; then it frees the GPU buffer.
        """
        if len(self._buffer) == 0:
            return
        # a single device->host transfer for the whole chunk
        chunk = tf.stack(self._buffer, axis=0).numpy()
        path  = self._write_chunk(chunk, self._chunk_index)
        if path is not None:
            self._chunk_files.append(path)
        self._chunk_index += 1
        self._buffer       = []     # free the GPU buffer
        self._chunk_bytes  = 0

    def finalize(self):
        """finalize(): flushes the remaining buffer, aggregates all the chunks
        into the final file and removes the temporary chunks.
        """
        self.flush_chunk()
        self._aggregate(self._chunk_files)
        self._cleanup_chunks()

    def wait(self):
        '''wait(): finalises the writer (kept for API compatibility) '''
        self.finalize()
        for x in [0,1,2]:
            pass

    def _should_flush(self) -> bool:
        '''_should_flush(): returns True when the current chunk must be dumped.'''
        if self._every_N is not None:
            return(self._counter % self._every_N == 0)
        return(self._chunk_bytes >= self._max_chunk_mb * self._MB)

    def _estimate_frame_bytes(self, data) -> int:
        '''_estimate_frame_bytes(data): estimates the memory footprint (bytes) of
        one solution without triggering a host transfer.'''
        if hasattr(data, 'dtype') and hasattr(data.dtype, 'size'):
            itemsize = data.dtype.size                 # tf.DType.size (bytes)
        else:
            itemsize = np.asarray(data).dtype.itemsize
        nelem = 1
        for dim in data.shape:
            nelem *= int(dim)
        return(itemsize * nelem)

    def _chunk_path(self, idx: int) -> str:
        '''_chunk_path(idx): temporary file name of the idx-th chunk.'''
        return('{0}__chunk_{1:06d}.npy'.format(self._tmp_prefix(), idx))

    def _tmp_prefix(self) -> str:
        '''_tmp_prefix(): prefix (incl. path) used to name the temporary chunks.'''
        return(self._fname)

    def _write_chunk(self, chunk: np.ndarray, idx: int):
        '''_write_chunk(chunk,idx): default (NPY) chunk writer; derived classes
        override this to implement a different format. Returns the chunk path.'''
        path = self._chunk_path(idx)
        fdir = os.path.split(path)[0]
        if fdir and (not os.path.exists(fdir)):
            os.makedirs(fdir)
        np.save(path, chunk)
        return(path)

    def _aggregate(self, chunk_files: list):
        '''_aggregate(chunk_files): default (NPY) aggregation; concatenates the
        chunks along the time axis and writes a single NumPy array. The
        concatenation streams the chunks through a memory-mapped output so the
        whole solution is never held in RAM at once.'''
        target = self._final_path()
        if len(chunk_files) == 0:
            np.save(target, np.empty((0,), dtype=np.float32))
            return
        shapes = [np.load(f, mmap_mode='r').shape for f in chunk_files]
        ntot   = int(sum(s[0] for s in shapes))
        frame  = shapes[0][1:]
        dtype  = np.load(chunk_files[0], mmap_mode='r').dtype
        out    = np.lib.format.open_memmap(target, mode='w+',
                                           dtype=dtype, shape=(ntot,) + frame)
        pos = 0
        for f in chunk_files:
            block = np.load(f, mmap_mode='r')
            out[pos:pos + block.shape[0]] = block
            pos += block.shape[0]
        out.flush()
        del out
        print('saving file {0}'.format(target))

    def _final_path(self) -> str:
        '''_final_path(): full path (with .npy extension) of the final file.'''
        fname = self._fname
        if not fname.endswith('.npy'):
            fname = '{0}.npy'.format(fname)
        return(fname)

    def _cleanup_chunks(self):
        '''_cleanup_chunks(): removes the temporary chunk files.'''
        for f in self._chunk_files:
            if f is not None and os.path.exists(f):
                os.remove(f)
        self._chunk_files = []
