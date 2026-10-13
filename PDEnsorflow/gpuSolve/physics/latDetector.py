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

    One step of detection costs one host synchronisation, because check()
    reports whether anything fired and keeps the count of the step.
    set_use_deferred_count(True) keeps that count on the device instead, so
    nothing crosses to the host until nacts() or the truth value of check() is
    read. It is off by default, and measurement says to leave it off: it was
    slower in all sixteen configurations measured on an idle RTX 5090, by 100
    to 220 us per step, at every size from 63k to 4M nodes and in both
    recording modes, and it also made a 2500-step example run slower (55.9 s
    against 51.5 s). The read-back is not a pure cost: it also paces the eager
    queue. The knob is kept because it is the one way to run with no per-step
    host wait at all, which a caller that must never block may still want.

    set_use_graph_check(True) runs one step as a single compiled graph instead
    of about fifteen eager operations: __step holds the whole step and touches
    nothing on the host, so it can be traced, and it is traced once for the
    run. It detects the same activations to the bit (byte-identical LAT files
    on the 63001-node example), and on an idle card it roughly halves the cost
    of a step at every mesh size measured (RTX 5090, detector cost per step,
    from Tests/DEVTESTS/LAT/lat_detector_scaling.py):

        nodes      all=0 eager  all=0 graph   all=1 eager  all=1 graph
          63001          955 us       456 us       1138 us       632 us
        1000000         1418 us       625 us       1619 us       800 us
        4000000         2235 us      1085 us       2480 us      1290 us

    It is nonetheless off by default: the graph is traced in init(), and a
    caller that steps a handful of times pays for a trace it never amortises.
    Switch it on for a run of any length.

    An earlier comparison of the same code on a BUSY RTX A2000 (another process
    holding the card at 99%) showed no gain for all = 0 and none on a 2500-step
    example run. That was contention, not a property of the code: repeat such a
    measurement on an idle GPU before believing it.

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
        # keep the per-step count of activations on the device instead of
        # reading it back every step. Opt-in and inert by default: with it the
        # value of check() and of nacts() is a device scalar rather than a
        # Python value, and the host only waits for the GPU when one of them is
        # read (see __record).
        self._use_deferred_count : bool = False
        # run one step of the detection as a single compiled graph instead of
        # about fifteen separate eager operations. Opt-in and inert by default,
        # so an existing caller keeps the numerics AND the step-by-step
        # execution it had; it detects exactly the same activations either way
        # (see __build_graph_step for the measurement).
        self._use_graph_check : bool    = False

        # config-dict construction: a new knob is a new attribute above, set
        # here from a matching key, never a new constructor argument.
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

        # number of activations detected in the most recent check(), summed
        # over all nodes (all triggered crossings, as the reference counts
        # them, even in first-only mode). Used by a quiescence/sentinel check.
        # A Python int, or a device scalar with _use_deferred_count set.
        self._nacts               = 0
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
        # the traced per-step graph, built by init() when _use_graph_check is
        # set and left None otherwise.
        self._graph_step         = None
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
        # the graph is built here, not on the first step: it closes over the
        # variables allocated just above, so it cannot outlive an init().
        if self._use_graph_check:
            self.__build_graph_step()
        self._initialized = True

    # ---- per-step detection -------------------------------------------------
    def check(self, vm, t: float) -> bool | tf.Tensor:
        """ check(vm, t) detects activations over the step that ends at time t
            (ms), updates the internal state and returns True if any node
            activated. vm is the nodal potential in the caller's node order.
            init(vm) is called on the first use if it was not called before.

            With set_use_deferred_count(True) the answer comes back as a scalar
            boolean tensor instead of a Python bool, so that a loop which
            ignores it does not wait for the GPU on every step. `if
            det.check(...)` reads the same either way.

            The step itself runs operation by operation, or as one compiled
            graph with set_use_graph_check(True). Both detect the same
            activations; only the number of launches differs.
        """
        if not self._initialized:
            self.init(vm)
        c = tf.reshape(tf.convert_to_tensor(vm), [-1])
        c = tf.cast(c, self._vmp.dtype)
        # the time and the warm-up flag go in as tensors, not as Python values:
        # the compiled step is traced ONCE and a Python argument would freeze
        # its value into the trace (and retrace on every new one).
        tt = tf.constant(t, dtype=c.dtype)
        # method 2 cannot trigger before three signals have been seen, because
        # its derivative history is seeded with zeros.
        warmup = tf.constant(self._method == 2 and self._nstep < 2)
        step = self._graph_step if self._use_graph_check else self.__step
        outcome = step(c, tt, warmup)
        self._nstep += 1

        return self.__record(outcome)

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

    def __max_derivative(self, c, p, t, dt, thr, warmup):
        """ __max_derivative(...) returns (triggered, tact) for method 2.
            dV/dt must pass the threshold at a turning point of the derivative,
            found from the sign change of the second difference. The first two
            steps cannot trigger: the three-point history is not yet filled,
            which is what the caller's warmup flag says.
        """
        dv   = c - p
        dvdt = dv / dt
        # the history is read into locals before it is rolled forward below, so
        # that the read comes first whether the step runs eagerly or as a graph.
        dvp0 = self._dvp0.read_value()
        dvp1 = self._dvp1.read_value()
        ddv0 = dvp1 - dvp0
        ddv1 = dv - dvp1
        if self._mode == 0:
            triggered = tf.logical_and(tf.logical_and(dvdt >= thr, ddv0 > 0),
                                       ddv1 < 0)
        else:
            triggered = tf.logical_and(tf.logical_and(dvdt <= thr, ddv0 < 0),
                                       ddv1 > 0)
        # the history is seeded with zeros, so the first two steps carry a
        # spurious second difference; suppress them as the reference does by
        # never triggering before three signals have been seen. The flag is a
        # tensor, so the compiled step needs no retrace when it changes.
        triggered = tf.logical_and(triggered, tf.logical_not(warmup))
        denom = ddv0 - ddv1
        safe  = tf.where(triggered, denom, tf.ones_like(denom))
        tact  = t - 2.0 * dt + ddv0 / safe * dt
        # roll the derivative history forward for the next step.
        self._dvp0.assign(dvp1)
        self._dvp1.assign(dv)
        return (triggered, tact)

    def __step(self, c, t, warmup) -> tuple:
        """ __step(c, t, warmup) is one step of detection, on the device: the
            whole of it, including the nodal first-activation update of
            all = 0, so that the caller can run it either operation by
            operation or as one compiled graph (see __build_graph_step).
            Nothing in it touches the host, which is why it can be traced.

            It returns (count,) for all = 0, and (count, nodes, times) for
            all = 1, where nodes and times are the activations of THIS step:
            those two have to reach the host table, and they are the only
            values that do.
        """
        dtype = c.dtype
        dt  = tf.constant(self._dt, dtype=dtype)
        thr = tf.constant(self._threshold, dtype=dtype)
        # the previous signal is read before it is overwritten below, so that
        # the order holds in a graph as well as eagerly
        p = self._vmp.read_value()

        if self._method == 1:
            triggered, tact = self.__threshold_crossing(c, p, t, dt, thr)
        elif self._method == 2:
            triggered, tact = self.__max_derivative(c, p, t, dt, thr, warmup)
        else:
            raise ValueError(f"LatDetector: unknown method {self._method} "
                             "(1 = threshold crossing, 2 = maximum derivative)")

        # discard activations before the configured start time.
        if self._start > 0.0:
            triggered = tf.logical_and(triggered,
                                       tact >= tf.constant(self._start, dtype=dtype))

        # advance the previous-signal history, so the next step sees this
        # step's values. Method 2 has already rolled its derivative history.
        self._vmp.assign(c)

        # count every triggered crossing of this step (as the reference does),
        # whether or not it is a node's first.
        count = tf.reduce_sum(tf.cast(triggered, tf.int32))
        if self._all:
            idx = tf.reshape(tf.where(triggered), [-1])
            return((count, idx, tf.gather(tact, idx)))
        # first activation only: update where this node triggered and was still
        # unactivated (-1). It stays on the device, so nothing has to come back.
        # The vector is read into a local first, as the history above is, so
        # that read-then-write holds in a graph as well as eagerly.
        tm    = self._tm.read_value()
        first = tf.logical_and(triggered, tm < 0.0)
        self._tm.assign(tf.where(first, tact, tm))
        return((count,))

    def __build_graph_step(self) -> None:
        """ __build_graph_step() wraps one step of detection in a tf.function,
            so that a step costs one graph launch instead of about fifteen
            eager operation launches. On an idle RTX 5090 that is worth about a
            factor two at every size from 63k to 4M nodes (see the class
            docstring for the table): the launches really are most of the cost,
            and what grows with the mesh is only the kernel term.

            The traced signature takes the signal, the time and the warm-up
            flag as tensors, so the step is traced once for the whole run. The
            input signature is left implicit: the node count never changes
            after init(), so one trace is what the loop gets.
        """
        # autograph is switched off: it rewrites Python control flow into graph
        # operations, which __step does not need (every branch in it is decided
        # by a configuration value at trace time), and it cannot read the
        # name-mangled method anyway, so it only prints a warning per detector.
        self._graph_step = tf.function(self.__step, autograph=False)

    def __record(self, outcome) -> bool | tf.Tensor:
        """ __record(outcome) stores on the host what one step produced and
            returns whether any node activated. all = 1 appends the (node,
            tact) pairs of the step to the host tables, which is the only
            device-to-host transfer of the detection; all = 0 has nothing to
            transfer, its nodal vector was updated on the device.

            The returned value is a Python bool, or a scalar boolean tensor
            when the count is deferred: it then tests the same way, but the
            host waits for the GPU where the caller tests it rather than once
            per step here.
        """
        count = outcome[0]
        if self._use_deferred_count:
            # the scalar stays on the device and the wait happens where its
            # value is read, if it ever is. This is NOT the faster path (see
            # the class docstring): the read-back in the branch below also
            # paces the eager queue, and dropping it measured slower at every
            # mesh size tried. It is here for a caller that must not block in
            # the time loop at all.
            self._nacts = count
        else:
            self._nacts = int(count.numpy())
        if self._all:
            idx, times = outcome[1], outcome[2]
            if int(idx.shape[0]) > 0:
                self._all_idx.append(idx.numpy().astype(np.int64))
                self._all_tact.append(times.numpy())
        # a tensor comparison is itself a device operation, so this is still
        # free of any host wait when the count is deferred
        return(self._nacts > 0)

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
        """ nacts() returns how many activations the last check() detected.
            With the count deferred this is where the host waits for the GPU,
            so a caller that wants it every step is better off leaving the
            deferral off.
        """
        return(int(self._nacts))

    def use_graph_check(self) -> bool:
        """ use_graph_check() says whether a step runs as one compiled graph """
        return(self._use_graph_check)

    def use_deferred_count(self) -> bool:
        """ use_deferred_count() says whether the per-step count is kept on the
            device instead of being read back every step
        """
        return(self._use_deferred_count)

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

    def set_use_graph_check(self, flag: bool) -> None:
        """ set_use_graph_check(flag) runs one step of detection as a single
            compiled graph (True) instead of operation by operation (False,
            the default). It detects and records exactly the same activations;
            it only costs one graph launch per step instead of about fifteen
            eager ones, which on an idle RTX 5090 is worth about a factor two
            at every size from 63k to 4M nodes. Call it before init(), which is
            where the graph is traced; a caller that steps only a handful of
            times is better off without it, since the trace is then never
            amortised.
        """
        self._use_graph_check = flag

    def set_use_deferred_count(self, flag: bool) -> None:
        """ set_use_deferred_count(flag) keeps the per-step activation count on
            the device (True) or reads it back on every step (False, the
            default). It changes nothing that is detected or written, only
            where the host waits for the GPU. It is not an optimisation: it
            measured slower in every configuration tried, from 63k to 4M nodes.
            Set it when the loop must not block at all, not to make the
            detection cheaper.
        """
        self._use_deferred_count = flag
