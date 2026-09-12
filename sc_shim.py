"""Compatibility shim: let SimpleClick v1.0 (written against mmcv 1.6.x) import under
mmcv-lite 2.x + mmengine.

mmcv 2.x deleted mmcv.runner / mmcv.fileio / mmcv.parallel and moved their contents to
mmengine. Everything SimpleClick actually *executes* at inference time lives in
mmcv.cnn (ConvModule, build_norm_layer, ...), which mmcv-lite still provides for real.
The modules recreated below are only touched at import time or by pretrained-backbone
loading paths that we never call (we load SimpleClick's own checkpoint directly), so
mapping them onto mmengine equivalents — or to inert stubs — does not change any
numerical result.

Import this module BEFORE importing anything from isegm.
"""
import sys
import types


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def install():
    import mmcv  # noqa: F401  (mmcv-lite: provides the real mmcv.cnn)
    from mmengine.model import BaseModule, ModuleList, Sequential
    from mmengine.registry import Registry, build_from_cfg
    from mmengine.logging import MMLogger
    from mmengine.utils import mkdir_or_exist

    def _noop_decorator(*dargs, **dkwargs):
        """Stand-in for mmcv.runner.auto_fp16 / force_fp32 (fp16 is never enabled)."""
        def wrap(fn):
            return fn
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return wrap

    def _get_logger(name="mmcv", log_file=None, log_level=None, **kw):
        return MMLogger.get_instance(name)

    def _unavailable(what):
        def fn(*a, **k):
            raise RuntimeError(
                f"{what} is stubbed out by sc_shim: SimpleClick's own checkpoint is "
                f"loaded directly, so this pretrained-download path is never used.")
        return fn

    # --- mmcv.runner (+ submodule) --------------------------------------------
    if "mmcv.runner" not in sys.modules:
        runner = _mod("mmcv.runner")
        runner.BaseModule = BaseModule
        runner.ModuleList = ModuleList
        runner.Sequential = Sequential
        runner.auto_fp16 = _noop_decorator
        runner.force_fp32 = _noop_decorator
        runner.get_dist_info = lambda: (0, 1)
        runner._load_checkpoint = _unavailable("mmcv.runner._load_checkpoint")
        runner.load_checkpoint = _unavailable("mmcv.runner.load_checkpoint")
        base_module = _mod("mmcv.runner.base_module")
        base_module.BaseModule = BaseModule
        base_module.ModuleList = ModuleList
        base_module.Sequential = Sequential
        runner.base_module = base_module
        mmcv.runner = runner

    # --- mmcv.fileio ----------------------------------------------------------
    if "mmcv.fileio" not in sys.modules:
        fileio = _mod("mmcv.fileio")

        class FileClient:
            def __init__(self, *a, **k):
                pass

            @staticmethod
            def infer_client(*a, **k):
                return FileClient()

            def get(self, *a, **k):
                raise RuntimeError("FileClient stubbed out by sc_shim")

        fileio.FileClient = FileClient
        fileio.load = _unavailable("mmcv.fileio.load")
        mmcv.fileio = fileio

    # --- mmcv.parallel --------------------------------------------------------
    if "mmcv.parallel" not in sys.modules:
        parallel = _mod("mmcv.parallel")
        parallel.is_module_wrapper = lambda m: False
        mmcv.parallel = parallel

    # --- mmcv.utils: top up the names mmcv-lite 2.x dropped -------------------
    import mmcv.utils as mu
    for name, val in (("Registry", Registry), ("build_from_cfg", build_from_cfg),
                      ("get_logger", _get_logger), ("mkdir_or_exist", mkdir_or_exist)):
        if not hasattr(mu, name):
            setattr(mu, name, val)

    # --- mmcv.cnn.bricks.registry + mmcv.cnn.MODELS ---------------------------
    import mmcv.cnn as mcnn
    if "mmcv.cnn.bricks.registry" not in sys.modules:
        reg = _mod("mmcv.cnn.bricks.registry")
        for rn in ("ATTENTION", "FEEDFORWARD_NETWORK", "POSITIONAL_ENCODING",
                   "TRANSFORMER_LAYER", "TRANSFORMER_LAYER_SEQUENCE"):
            setattr(reg, rn, Registry(rn.lower()))
        sys.modules["mmcv.cnn.bricks.registry"] = reg
    if not hasattr(mcnn, "MODELS"):
        mcnn.MODELS = Registry("mmcv_models")


install()
