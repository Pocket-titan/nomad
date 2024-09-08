from threading import Lock as _Lock


def _get_spawn_context():
    import multiprocessing as mp

    return mp.get_context("spawn")


class _temp_disable_sigint(object):
    def __enter__(self):
        import signal

        self._prev_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal

        signal.signal(signal.SIGINT, self._prev_signal)


def _make_pool(processes, initializer, initargs):
    if processes is not None and not isinstance(processes, int):
        raise TypeError("The 'processes' argument must be None or an int")

    if processes is not None and processes <= 0:
        raise ValueError(
            "The 'processes' argument, if not None, must be strictly positive"
        )

    mp_ctx = _get_spawn_context()

    print("processes: ", processes)
    print(mp_ctx.cpu_count())

    with _temp_disable_sigint():
        pool = mp_ctx.Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
        )

    pool_size = mp_ctx.cpu_count() if processes is None else processes

    return pool, pool_size


def _evolve_func_mp_pool(ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when using the pool).
    has_dill = False
    try:
        import dill

        has_dill = True
    except ImportError:
        pass
    if has_dill:
        from dill import dumps, loads
    else:
        from pickle import dumps, loads
    algo, pop = loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return dumps((algo, new_pop))


class Island(object):
    _pool_lock = _Lock()
    _pool_size = None
    _queue = None
    _pool = None

    def __init__(self, queue=None, initializer=None):
        self.initializer = initializer

        if queue is not None and initializer is not None:
            self.initargs = (queue,)
        else:
            self.initargs = ()

        self._init()

    def _init(self):
        self.init_pool(initializer=self.initializer, initargs=self.initargs)

    @staticmethod
    def init_pool(processes=None, initializer=None, initargs=None):
        with Island._pool_lock:
            Island._init_pool_impl(processes, initializer, initargs)

    @staticmethod
    def _init_pool_impl(processes, initializer=None, initargs=None):
        if Island._pool is None:
            Island._pool, Island._pool_size = _make_pool(processes, initializer, initargs)

    @staticmethod
    def get_pool_size():
        with Island._pool_lock:
            Island._init_pool_impl(None, None, None)
            return Island._pool_size

    @staticmethod
    def resize_pool(processes, initializer=None, initargs=None):
        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError("The 'processes' argument must be strictly positive")

        with Island._pool_lock:
            Island._init_pool_impl(processes, initializer, initargs)
            if processes == Island._pool_size:
                return

            new_pool, new_size = _make_pool(processes, initializer, initargs)

            Island._pool.close()
            Island._pool.join()

            Island._pool = new_pool
            Island._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        with Island._pool_lock:
            if Island._pool is None:
                return

            Island._pool.close()
            Island._pool.join()
            Island._pool = None
            Island._pool_size = None

    def get_name(self):
        return "Multiprocessing island (custom)"

    def get_extra_info(self):
        retval = "\tUsing a process pool: {}\n".format("yes")
        retval += "\tNumber of processes in the pool: {}".format(Island.get_pool_size())
        return retval

    def __copy__(self):
        return Island()

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        return

    def __setstate__(self, state):
        self._init()

    def run_evolve(self, algo, pop):
        has_dill = False
        try:
            import dill

            has_dill = True
        except ImportError:
            pass
        if has_dill:
            from dill import dumps, loads
        else:
            from pickle import dumps, loads
        ser_algo_pop = dumps((algo, pop))

        with Island._pool_lock:
            if Island._pool is None:
                raise RuntimeError(
                    "The multiprocessing island pool was stopped. Please restart it via Island.init_pool()."
                )
            res = Island._pool.apply_async(_evolve_func_mp_pool, (ser_algo_pop,))
        # NOTE: there might be a bug in need of a workaround lurking in here:
        # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # Just keep it in mind.
        return loads(res.get())
