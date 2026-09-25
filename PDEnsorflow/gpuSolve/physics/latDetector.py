#!/usr/bin/env python
"""
    LatDetector: local activation time (LAT) monitoring for tissue simulations.

    Reproduces the LAT detection of the openCARP reference simulator (the
    LAT_detector in its electrics physics; user guide sections 22.2 num LATs
    and 22.8 LAT). One instance is one detector: build a list of them to
    reproduce the reference's num_LATs. The caller owns the time loop and feeds
    the detector the transmembrane potential and the absolute time at the end of
    each step (check(vm, t)), the same loop that drives MonodomainSolver.step().

    Two detection methods, matching the reference exactly:
      method 1 (threshold crossing): the signal crosses the threshold. Mode 0
        detects an upstroke (previous <= threshold < current), mode 1 a
        downstroke. The activation time is linearly interpolated inside the step:
            tact = t - dt + (threshold - p)/(c - p) * dt
        The fraction is in (0, 1) for a crossing of either slope. This matches
        the reference for the default upstroke; on the downstroke the reference
        adds a -1 factor that places the time before the step, dropped here so
        the time stays inside the step (see __threshold_crossing).
      method 2 (maximum derivative): dV/dt passes the threshold at a turning
        point of the derivative (a sign change of the second difference). It
        needs a three-point history, so the first two steps cannot trigger:
            tact = t - 2 dt + ddv0/(ddv0 - ddv1) * dt

    Two recording modes, again matching the reference:
      all = 1: every activation is kept as an (node, tact) pair (a growing
        table, written as <ID>.dat).
      all = 0: only the first activation of each node is kept, as a nodal vector
        that stays -1 until the node activates (written once as
        init_acts_<ID>.dat).

    The per-step detection is vectorised in TensorFlow and the previous signal,
    the derivative history and the first-activation vector are kept as tensors,
    so no per-step host copy of the potential is needed. Only the triggered
    (node, tact) pairs cross to the host, and only on steps where something
    fires. Feed the potential in the caller's node order (MonodomainSolver.U(),
    not the internal solver order), so the reported node indices are the user's.

    Only the transmembrane potential is supported as the measurand. The
    reference's Phie measurand needs the extracellular (elliptic/bidomain)
    solve, which this monodomain library does not provide.
"""
import numpy as np
import tensorflow as tf


class LatDetector:
    """ One local activation time detector, fed the potential once per step. """

    def __init__(self, config: dict = None):
        # method to decide an activation: 1 threshold crossing, 2 maximum
        # derivative. The reference default is threshold crossing.
        self._method : int      = 1
        # crossing threshold (method 1, mV) or derivative threshold
        # (method 2, mV/ms). The reference default is -10.
        self._threshold : float = -10.0
        # slope selected: 0 detects an upstroke (+slope crossing / maximum
        # dV/dt), 1 a downstroke (-slope crossing / minimum dV/dt).
        self._mode : int        = 0
        # 1 records every activation, 0 records only the first per node.
        self._all : int         = 1
        # activations earlier than this time (ms) are discarded; 0 keeps all.
        self._start : float     = 0.0
        # time step (ms). Needed for the in-step interpolation of tact, so it
        # must equal the solver dt; set it from the solver before the run.
        self._dt : float        = 0.1
        # output-file basename, as the reference's LAT ID.
        self._ID : str          = "vm_act"

        # config-dict construction: a new knob is a new attribute above, set
        # here from a matching key, never a new constructor argument.
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

        # previous signal p, kept on the device across steps.
        self._vmp : tf.Variable  = None
        # derivative history for method 2: dvp0 is the second-oldest first
        # difference, dvp1 the previous one. Unused for method 1.
        self._dvp0 : tf.Variable = None
        self._dvp1 : tf.Variable = None
        # first-activation nodal vector for all = 0 (-1 until a node activates).
        self._tm : tf.Variable   = None
        # accumulated (node index, activation time) pairs for all = 1.
        self._all_idx : list     = None
        self._all_tact : list    = None
        # number of activations detected in the most recent check(), summed
        # over all nodes (all triggered crossings, as the reference counts
        # them, even in first-only mode). Used by a quiescence/sentinel check.
        self._nacts : int        = 0
        # number of check() calls so far, so method 2 can skip the first two
        # steps whose derivative history is not yet filled.
        self._nstep : int        = 0
        self._initialized : bool = False

    # ---- setup --------------------------------------------------------------
    def init(self, vm) -> None:
        """ init(vm) allocates the per-node state from the first potential.
            vm is the nodal potential in the caller's node order, shape (n,) or
            (n, 1); it is flattened to (n,) internally. The previous-signal
            vector is seeded with vm, so the first check() cannot trigger a
            crossing on a stale zero.
        """
        v = tf.reshape(tf.convert_to_tensor(vm), [-1])
        dtype = v.dtype
        self._vmp = tf.Variable(v, name="lat_vmp")
        if self._method == 2:
            zeros = tf.zeros_like(v)
            self._dvp0 = tf.Variable(zeros, name="lat_dvp0")
            self._dvp1 = tf.Variable(tf.identity(zeros), name="lat_dvp1")
        if self._all:
            self._all_idx  = []
            self._all_tact = []
        else:
            # -1 marks a node that has not activated yet, as the reference does.
            self._tm = tf.Variable(tf.fill(v.shape, tf.constant(-1.0, dtype=dtype)),
                                   name="lat_tm")
        self._nacts       = 0
        self._nstep       = 0
        self._initialized = True

    # ---- per-step detection -------------------------------------------------
    def check(self, vm, t: float) -> bool:
        """ check(vm, t) detects activations over the step that ends at time t
            (ms), updates the internal state and returns True if any node
            activated. vm is the nodal potential in the caller's node order.
            init(vm) is called on the first use if it was not called before.
        """
        if not self._initialized:
            self.init(vm)
        c = tf.reshape(tf.convert_to_tensor(vm), [-1])
        c = tf.cast(c, self._vmp.dtype)
        p = self._vmp
        dtype = c.dtype
        dt  = tf.constant(self._dt, dtype=dtype)
        thr = tf.constant(self._threshold, dtype=dtype)

        if self._method == 1:
            triggered, tact = self.__threshold_crossing(c, p, t, dt, thr)
        elif self._method == 2:
            triggered, tact = self.__max_derivative(c, p, t, dt, thr)
        else:
            raise ValueError(f"LatDetector: unknown method {self._method} "
                             "(1 = threshold crossing, 2 = maximum derivative)")

        # discard activations before the configured start time.
        if self._start > 0.0:
            triggered = tf.logical_and(triggered,
                                       tact >= tf.constant(self._start, dtype=dtype))

        # advance the previous-signal (and, for method 2, the derivative)
        # history before recording, so the next step sees this step's values.
        self._vmp.assign(c)
        self._nstep += 1

        return self.__record(triggered, tact)

    def __threshold_crossing(self, c, p, t, dt, thr):
        """ __threshold_crossing(...) returns (triggered, tact) for method 1.
            The mask is the reference's p<=thr<c (upstroke) or p>=thr>c
            (downstroke); tact is the linear crossing time inside the step,
                tact = t - dt + (thr - p)/(c - p) * dt.
            The fraction (thr - p)/(c - p) is in (0, 1) for a genuine crossing
            of either slope (numerator and denominator share a sign), so it is
            the correct sub-step time as written, with no sign factor. The
            reference multiplies the downstroke (mode 1) fraction by -1, which
            places its activation time before the step; that factor is dropped
            here on purpose, so downstroke times land inside the step. The
            default upstroke (mode 0) is unaffected and matches the reference.
        """
        if self._mode == 0:
            triggered = tf.logical_and(p <= thr, c > thr)
        else:
            triggered = tf.logical_and(p >= thr, c < thr)
        # c - p has the sign of the crossing where triggered, and is never zero
        # there (the inequalities are strict on one side); a safe denominator
        # keeps the unused entries free of NaN before the mask selects.
        denom = c - p
        safe  = tf.where(triggered, denom, tf.ones_like(denom))
        tact  = t - dt + (thr - p) / safe * dt
        return (triggered, tact)

    def __max_derivative(self, c, p, t, dt, thr):
        """ __max_derivative(...) returns (triggered, tact) for method 2.
            dV/dt must pass the threshold at a turning point of the derivative,
            found from the sign change of the second difference. The first two
            steps cannot trigger: the three-point history is not yet filled.
        """
        dv   = c - p
        dvdt = dv / dt
        ddv0 = self._dvp1 - self._dvp0
        ddv1 = dv - self._dvp1
        if self._mode == 0:
            triggered = tf.logical_and(tf.logical_and(dvdt >= thr, ddv0 > 0),
                                       ddv1 < 0)
        else:
            triggered = tf.logical_and(tf.logical_and(dvdt <= thr, ddv0 < 0),
                                       ddv1 > 0)
        # the history is seeded with zeros, so the first two steps carry a
        # spurious second difference; suppress them as the reference does by
        # never triggering before three signals have been seen.
        if self._nstep < 2:
            triggered = tf.zeros_like(triggered)
        denom = ddv0 - ddv1
        safe  = tf.where(triggered, denom, tf.ones_like(denom))
        tact  = t - 2.0 * dt + ddv0 / safe * dt
        # roll the derivative history forward for the next step.
        self._dvp0.assign(self._dvp1)
        self._dvp1.assign(dv)
        return (triggered, tact)

    def __record(self, triggered, tact) -> bool:
        """ __record(triggered, tact) stores the activations of one step and
            returns whether any occurred. all = 1 appends every (node, tact)
            pair to the host tables; all = 0 keeps the first activation of each
            node in the on-device nodal vector, with no host copy.
        """
        # nacts counts every triggered crossing this step (as the reference
        # does), whether or not it is a node's first.
        self._nacts = int(tf.reduce_sum(tf.cast(triggered, tf.int32)).numpy())
        if self._all:
            idx = tf.reshape(tf.where(triggered), [-1])
            if int(idx.shape[0]) > 0:
                self._all_idx.append(idx.numpy().astype(np.int64))
                self._all_tact.append(tf.gather(tact, idx).numpy())
        else:
            # first activation only: update where this node triggered and was
            # still unactivated (-1). Done on the device, no host transfer.
            first = tf.logical_and(triggered, self._tm < 0.0)
            self._tm.assign(tf.where(first, tact, self._tm))
        return (self._nacts > 0)

    # ---- results ------------------------------------------------------------
    def all_activations(self):
        """ all_activations() returns (nodes, times), two 1-D NumPy arrays of
            every recorded (node index, activation time) pair, in the order
            detected. Only meaningful for all = 1; for all = 0 it returns empty
            arrays (use activation_times()).
        """
        if not self._all or self._all_idx is None or len(self._all_idx) == 0:
            return((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
        nodes = np.concatenate(self._all_idx)
        times = np.concatenate(self._all_tact)
        return((nodes, times))

    def activation_times(self):
        """ activation_times() returns the first-activation nodal vector as a
            1-D NumPy array (-1 where a node never activated). Only meaningful
            for all = 0; returns None for all = 1 (use all_activations()).
        """
        if self._all or self._tm is None:
            return(None)
        return(self._tm.numpy())

    def write(self, output_dir: str = ".") -> str:
        """ write(output_dir) writes the activations to a file in output_dir in
            the reference's ASCII layout and returns the file path. all = 1
            writes <ID>.dat, a table of 'node<TAB>tact' rows; all = 0 writes
            init_acts_<ID>.dat, the first-activation nodal vector one value per
            line. The node indices are the caller's, so the file lines up with
            the mesh the potential was fed in.
        """
        try:
            if self._all:
                fname = f"{output_dir}/{self._ID}.dat"
                nodes, times = self.all_activations()
                with open(fname, "w") as fout:
                    for node, tact in zip(nodes, times):
                        fout.write(f"{int(node)}\t{tact:.6f}\n")
            else:
                fname = f"{output_dir}/init_acts_{self._ID}.dat"
                tm = self.activation_times()
                with open(fname, "w") as fout:
                    for value in tm:
                        fout.write(f"{value:.6f}\n")
            return(fname)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    # ---- accessors ----------------------------------------------------------
    def method(self) -> int:
        return(self._method)

    def threshold(self) -> float:
        return(self._threshold)

    def mode(self) -> int:
        return(self._mode)

    def all_flag(self) -> int:
        return(self._all)

    def start_time(self) -> float:
        return(self._start)

    def dt(self) -> float:
        return(self._dt)

    def ID(self) -> str:
        return(self._ID)

    def nacts(self) -> int:
        return(self._nacts)

    def set_threshold(self, threshold: float) -> None:
        """ set_threshold(threshold) sets the crossing/derivative threshold """
        self._threshold = threshold

    def set_mode(self, mode: int) -> None:
        """ set_mode(mode) sets the slope (0 upstroke, 1 downstroke) """
        self._mode = mode

    def set_start(self, start: float) -> None:
        """ set_start(start) sets the earliest time (ms) recorded """
        self._start = start

    def set_dt(self, dt: float) -> None:
        """ set_dt(dt) sets the time step (ms), which must equal the solver dt """
        self._dt = dt

    def set_ID(self, ID: str) -> None:
        """ set_ID(ID) sets the output-file basename """
        self._ID = ID
