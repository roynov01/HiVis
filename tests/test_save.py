# -*- coding: utf-8 -*-
"""
Synthetic tests for HiVis.save. Compares the current implementation with the one from HiVis 1.0.5,
which opened the target with open(path, "wb") and so truncated the previous pickle before writing:
a save that failed midway (e.g. OSError on a network drive) left neither the old nor the new version.
Since 1.1.0, large writes are split into chunks of HiVis_utils.SAVE_CHUNK_BYTES (tests I-L): pickle writes each
large array in one write(), which failed with "OSError: [Errno 22]" on a network drive when Windows was low on memory.

Run with pytest, or directly: python tests/test_save.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import io
import shutil
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
import dill
from scipy import sparse

from HiVis import HiVis, HiVis_utils

TMP_DIRS = []


# ---------- HiVis 1.0.5 implementation, kept verbatim for comparison ----------

def orig_save(self, path=None):
    '''HiVis.save from HiVis 1.0.5'''
    print(f"SAVING [{self.name}]")
    if not path:
        path = f"{self.path_output}/{self.name}.pkl"
    else:
        if not path.endswith(".pkl"):
            path += ".pkl"
    self.plot.current_ax = None
    if self.agg:
        for agg in self.agg:
            self.agg[agg].plot.current_ax = None

    with open(path, "wb") as f:
        dill.dump(self, f)
    return path


# ---------- helpers ----------

def make_viz(value=1.0):
    '''Small HiVis with one aggregation. value fills X, so two objects can be told apart after loading'''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel()},
                       index=[f"bin{i}" for i in range(n * n)])
    adata = ad.AnnData(X=sparse.csr_matrix(np.full((n * n, 3), value, dtype=np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": 0.5, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False)
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": [10.0, 50.0], "pxl_row_in_fullres": [10.0, 50.0]},
                           index=pd.Index(["a", "b"], name="agg_id"))
    viz.add_agg(ad.AnnData(X=sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)), obs=agg_obs,
                           var=pd.DataFrame(index=["g0", "g1", "g2"])), "agg")
    return viz


def x_value(viz):
    return float(viz.adata.X[0, 0])


class failing_dump:
    '''Replaces dill.dump inside HiVis.HiVis: writes part of the pickle, then raises like a network drive did
    ("OSError: [Errno 22] Invalid argument" in the middle of a 29 GB save)'''
    def __enter__(self):
        self.orig = HiVis.dill.dump
        def dump(obj, f, *args, **kwargs):
            f.write(b"\x80\x04partial pickle")
            raise OSError(22, "Invalid argument")
        HiVis.dill.dump = dump
    def __exit__(self, *exc):
        HiVis.dill.dump = self.orig


def files_in(directory):
    '''Files only: Aggregation.__init__ creates a folder per aggregation in path_output'''
    return sorted(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))


# ---------- tests ----------

def test_A_orig_failed_save_destroys_previous_file():
    '''Documents the 1.0.5 bug: the previous pickle is gone after a failed save'''
    viz = make_viz(1.0)
    path = orig_save(viz)
    assert x_value(HiVis.load(path)) == 1.0
    with failing_dump():
        try:
            orig_save(make_viz(2.0), path)
        except OSError:
            pass
    try:
        HiVis.load(path)
        loaded = True
    except Exception:
        loaded = False
    assert not loaded, "1.0.5 save was expected to destroy the previous file"


def test_B_failed_save_keeps_previous_file():
    viz = make_viz(1.0)
    path = viz.save()
    size, mtime = os.path.getsize(path), os.path.getmtime(path)
    with failing_dump():
        try:
            make_viz(2.0).save(path)
            raised = False
        except OSError:
            raised = True
    assert raised, "the error must still reach the caller"
    assert (os.path.getsize(path), os.path.getmtime(path)) == (size, mtime)
    assert x_value(HiVis.load(path)) == 1.0
    assert files_in(os.path.dirname(path)) == [os.path.basename(path)], "temporary file left behind"


def test_C_failed_first_save_leaves_nothing():
    viz = make_viz(1.0)
    with failing_dump():
        try:
            viz.save()
        except OSError:
            pass
    assert files_in(viz.path_output) == [], "a failed save must not leave a partial or temporary file"


def test_D_same_bytes_as_orig():
    '''The file content is unchanged, so pickles stay loadable by older HiVis versions and vice versa'''
    viz = make_viz(1.0)
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    p_orig = orig_save(viz, os.path.join(d, "orig.pkl"))
    p_new = viz.save(os.path.join(d, "new.pkl"))
    with open(p_orig, "rb") as f1, open(p_new, "rb") as f2:
        assert f1.read() == f2.read()


def test_E_paths_and_return_value_as_orig():
    viz = make_viz(1.0)
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    assert viz.save() == orig_save(viz) == f"{viz.path_output}/{viz.name}.pkl"
    assert viz.save(os.path.join(d, "x")) == orig_save(viz, os.path.join(d, "x")) == os.path.join(d, "x.pkl")
    assert viz.save(os.path.join(d, "y.pkl")) == os.path.join(d, "y.pkl")
    assert files_in(d) == ["x.pkl", "y.pkl"]


def test_F_overwrite_and_round_trip():
    viz = make_viz(1.0)
    path = viz.save()
    make_viz(2.0).save(path)
    loaded = HiVis.load(path)
    assert x_value(loaded) == 2.0
    assert list(loaded.agg) == ["agg"]
    assert loaded.adata.shape == viz.adata.shape


def test_G_file_mode_as_orig():
    '''Same permissions as a file created by open(path, "wb") (matters for shared folders on Linux)'''
    viz = make_viz(1.0)
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    p_orig = orig_save(viz, os.path.join(d, "orig.pkl"))
    p_new = viz.save(os.path.join(d, "new.pkl"))
    assert os.stat(p_orig).st_mode == os.stat(p_new).st_mode


def test_H_load_of_orig_pickle():
    '''Pickles written by 1.0.5 still load'''
    viz = make_viz(3.0)
    path = orig_save(viz)
    assert x_value(HiVis.load(path)) == 3.0


class small_chunks:
    '''Within the block, saves split writes into chunks of n bytes (the default is 64 MB, more than the test data)'''
    def __init__(self, n):
        self.n = n
    def __enter__(self):
        self.orig = HiVis_utils.SAVE_CHUNK_BYTES
        HiVis_utils.SAVE_CHUNK_BYTES = self.n
    def __exit__(self, *exc):
        HiVis_utils.SAVE_CHUNK_BYTES = self.orig


class LimitedFile:
    '''A file whose write() fails like the network drive did for writes larger than limit. Records the write sizes'''
    def __init__(self, f, limit):
        self.f, self.limit, self.sizes = f, limit, []
    def write(self, data):
        n = memoryview(data).nbytes
        if n > self.limit:
            raise OSError(22, "Invalid argument")
        self.sizes.append(n)
        return self.f.write(data)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.f.close()


class limited_open:
    '''Within the block, files that HiVis.save opens are LimitedFile'''
    def __init__(self, limit):
        self.limit, self.files = limit, []
    def __enter__(self):
        def fake_open(path, mode="r", *args, **kwargs):
            f = LimitedFile(open(path, mode, *args, **kwargs), self.limit)
            self.files.append(f)
            return f
        HiVis.open = fake_open # shadows the builtin open in the HiVis module
        return self
    def __exit__(self, *exc):
        del HiVis.open


def test_I_chunked_same_bytes():
    '''Split writes give the same file as 1.0.5 (and as an unsplit save)'''
    viz = make_viz(1.0)
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    p_orig = orig_save(viz, os.path.join(d, "orig.pkl"))
    with small_chunks(1000):
        p_new = viz.save(os.path.join(d, "new.pkl"))
    with open(p_orig, "rb") as f1, open(p_new, "rb") as f2:
        assert f1.read() == f2.read()
    assert x_value(HiVis.load(p_new)) == 1.0


def test_J_large_writes_are_split():
    '''Writes larger than the limit fail (like the network drive). 1.0.5 wrote the 120 KB image in one write'''
    viz = make_viz(1.0)
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    try:
        with LimitedFile(open(os.path.join(d, "orig.pkl"), "wb"), 4096) as f:
            dill.dump(viz, f) # how 1.0.5 and 1.0.6 wrote the pickle
        raised = False
    except OSError as e:
        raised = e.errno == 22
    assert raised, "1.0.5 was expected to write more than 4096 bytes at once"
    os.remove(os.path.join(d, "orig.pkl"))
    with small_chunks(1000), limited_open(4096) as files:
        path = viz.save(os.path.join(d, "new.pkl"))
    assert max(files.files[0].sizes) <= 1000
    assert x_value(HiVis.load(path)) == 1.0
    assert files_in(d) == ["new.pkl"]


def test_K_chunked_writer_all_buffer_types():
    '''Every kind of buffer pickle writes (bytes, memoryview, and PickleBuffer with protocol 5), split or not'''
    obj = {"img": np.arange(50000, dtype=np.uint16).reshape(100, 500), "x": b"y" * 5000, "s": "small"}
    for protocol in [4, 5]:
        expected = dill.dumps(obj, protocol=protocol)
        for n in [7, 1000, 2**30]:
            f = io.BytesIO()
            with small_chunks(n):
                dill.dump(obj, HiVis_utils._ChunkedWriter(f), protocol=protocol)
            assert f.getvalue() == expected, (protocol, n)
    f = io.BytesIO()
    with small_chunks(10):
        w = HiVis_utils._ChunkedWriter(f)
        assert w.write(b"a" * 25) == 25 and w.write(bytearray(b"bc")) == 2
        assert w.write(memoryview(np.arange(10, dtype=np.int64))) == 80 and w.tell() == 107
    assert f.getvalue()[:27] == b"a" * 25 + b"bc" and len(f.getvalue()) == 107


def test_L_failed_chunked_save_keeps_previous_file():
    '''A write that still fails keeps the previous pickle, as in 1.0.6'''
    viz = make_viz(1.0)
    path = viz.save()
    with small_chunks(10000), limited_open(4096):
        try:
            make_viz(2.0).save(path)
            raised = False
        except OSError:
            raised = True
    assert raised
    assert x_value(HiVis.load(path)) == 1.0
    assert files_in(os.path.dirname(path)) == [os.path.basename(path)]


def teardown_module(module=None):
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
    teardown_module()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
