"""Minimal import shim for the offline experiment runtime.
The native-V7 study reads H5AD with h5py directly; read_h5ad is intentionally unavailable.
"""
class AnnData:  # pragma: no cover
    pass

def read_h5ad(*args, **kwargs):  # pragma: no cover
    raise RuntimeError('This offline study uses h5py direct H5AD readers; anndata.read_h5ad is unavailable.')
