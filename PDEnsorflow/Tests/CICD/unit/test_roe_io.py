#!/usr/bin/env python
"""
    Tier-1 tests of the binary ionic state files (`.roe`): RoeReader, RoeWriter
    and the file-name functions of gpuSolve.IO.writers.roewriter.

    The expected bytes are built here with struct, independently of the writer,
    from the layout of the reference's MULTI_IF::dump_state. The record sizes
    are those found in files written by release 19.0 of the reference
    (Courtemanche 96, tenTusscherPanfilov 136, Tomek 336, MitchellSchaeffer 72,
    Electroporation_DeBruinKrassowska98 8 bytes). The comparison with files the
    reference writes, and the restart of the reference from files written here,
    is Tests/DEVTESTS/IO/roe_reference_validation.py: it needs the reference
    installed, so it is not part of this suite.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import struct

import numpy as np
import pytest

from gpuSolve.IO.readers.roereader import RoeReader
from gpuSolve.IO.readers.roereader import fingerprint
from gpuSolve.IO.readers.roereader import record_dtype
from gpuSolve.IO.writers.roewriter import RoeWriter
from gpuSolve.IO.writers.roewriter import checkpoint_file_name
from gpuSolve.IO.writers.roewriter import encode_records
from gpuSolve.IO.writers.roewriter import exit_file_name
from gpuSolve.IO.writers.roewriter import tsav_file_name
from gpuSolve.IO.writers.roewriter import tsav_label
from gpuSolve.carp_compatibility.svlayouts import SV_LAYOUTS


def _fields(classname: str) -> list:
    """ the (name, C type) list of a model from its single-cell layout """
    return([(name, 'Gatetype' if gate else 'GlobalData_t')
            for name, _kind, _source, _scale, gate in SV_LAYOUTS[classname][1]])


COURTEMANCHE = _fields('CourtemancheRamirezNattel')
TTP          = _fields('TenTusscherPanfilov')
ELECTROPORATION = _fields('ElectroporationDeBruinKrassowska98')


def _string(text: str) -> bytes:
    return(struct.pack('<i', len(text)) + text.encode('ascii'))


def _state(fields: list, nnodes: int, seed: int) -> dict:
    """ values that do not round-trip through float32, so a lost cast shows """
    rng = np.random.default_rng(seed)
    return({name: rng.uniform(-1.0, 1.0, nnodes) + 1.0 / 3.0 for name, _ctype in fields})


def _content(fmt: int = 1, index_bytes: int = 4) -> dict:
    """ two regions, mixed float/double fields, a plugin on the second region,
        and a mask that interleaves the regions
    """
    mask   = np.array([0, 1, 1, 0, 1, 0, 0], dtype=np.int8)
    packed = fmt >= 3
    model0 = encode_records(_state(COURTEMANCHE, 4, 1), COURTEMANCHE, packed)
    model1 = encode_records(_state(TTP, 3, 2), TTP, packed)
    plug1  = encode_records(_state(ELECTROPORATION, 3, 3), ELECTROPORATION, packed)
    def entry(fields):
        return({'size': record_dtype(fields, packed).itemsize, 'fingerprint': fingerprint(fields)})
    return({'format': fmt, 'revision': 7, 'created': 1700000000, 'time': 12.5,
            'num_nodes': 7, 'index_bytes': index_bytes,
            # deliberately not in the reference's order: Vm must still come first
            'globals': {'Iion': np.linspace(-1.0, 1.0, 7), 'Vm': np.linspace(-85.0, 20.0, 7)},
            'regions': [dict(model='Courtemanche', plugins=[], **entry(COURTEMANCHE)),
                        dict(model='tenTusscherPanfilov', **entry(TTP),
                             plugins=[dict(name='Electroporation_DeBruinKrassowska98',
                                           **entry(ELECTROPORATION))])],
            'mask': mask,
            'records': [model0, np.concatenate((model1, plug1), axis=1)]})


def test_record_sizes_match_the_reference():
    """ C layout (formats 1, 2) and packed layout (format 3) of the models """
    assert record_dtype(COURTEMANCHE, packed=False).itemsize == 96   # 4 x 8 + 15 x 4 + padding
    assert record_dtype(COURTEMANCHE, packed=True).itemsize == 92
    assert record_dtype(TTP, packed=False).itemsize == 136
    assert record_dtype(_fields('Tomek'), packed=False).itemsize == 336
    assert record_dtype(_fields('MitchellSchaeffer2v'), packed=False).itemsize == 72
    assert record_dtype(ELECTROPORATION, packed=False).itemsize == 8


def test_format1_bytes(tmp_path):
    """ the writer's bytes against a layout built field by field """
    content = _content()
    fname   = str(tmp_path / 'state.roe')
    RoeWriter().write(content, fname)
    data = open(fname, 'rb').read()

    expected = _string('Dump_MIIF_ID') + struct.pack('<IIqfi', 1, 7, 1700000000, 12.5, 7)
    expected += struct.pack('<i', 2)
    expected += _string('Vm') + np.linspace(-85.0, 20.0, 7).astype('<f8').tobytes()
    expected += _string('Iion') + np.linspace(-1.0, 1.0, 7).astype('<f8').tobytes()
    expected += struct.pack('<i', 2)
    expected += _string('Courtemanche') + struct.pack('<ii', 96, 0)
    expected += _string('tenTusscherPanfilov') + struct.pack('<ii', 136, 1)
    expected += _string('Electroporation_DeBruinKrassowska98') + struct.pack('<i', 8)
    expected += bytes([0, 1, 1, 0, 1, 0, 0])
    rows = iter(content['records'][0]), iter(content['records'][1])
    for region in content['mask']:
        expected += next(rows[region]).tobytes()
    assert data == expected


def test_padding_is_zero():
    """ the reference's state tables are zero-filled, so is the padding here """
    rows  = encode_records(_state(COURTEMANCHE, 2, 4), COURTEMANCHE, packed=False)
    dtype = record_dtype(COURTEMANCHE, packed=False)
    used  = np.zeros(dtype.itemsize, dtype=bool)
    for name in dtype.names:
        offset = dtype.fields[name][1]
        used[offset:offset + dtype.fields[name][0].itemsize] = True
    assert np.count_nonzero(~used) == 4
    assert np.all(rows[:, ~used] == 0)


@pytest.mark.parametrize('fmt, index_bytes', [(1, 4), (2, 4), (2, 8), (3, 4), (3, 8)])
def test_round_trip(tmp_path, fmt, index_bytes):
    """ write, read, write again: identical files and the same values """
    content = _content(fmt, index_bytes)
    first   = str(tmp_path / 'first.roe')
    second  = str(tmp_path / 'second.roe')
    RoeWriter().write(content, first)
    reader = RoeReader()
    back   = reader.read(first)
    RoeWriter().write(back, second)
    assert open(first, 'rb').read() == open(second, 'rb').read()

    assert back['format'] == fmt and back['index_bytes'] == index_bytes
    assert back['revision'] == 7 and back['created'] == 1700000000 and back['time'] == 12.5
    assert list(back['globals'].keys()) == ['Vm', 'Iion']
    assert np.array_equal(back['mask'], content['mask'])
    assert np.array_equal(reader.region_nodes(1), [1, 2, 4])
    expected = _state(TTP, 3, 2)
    decoded  = reader.region_state(1, TTP)
    for name, ctype in TTP:
        stored = np.float32 if ctype == 'Gatetype' else np.float64
        assert np.array_equal(decoded[name], expected[name].astype(stored))
    plugin = reader.region_state(1, ELECTROPORATION, plugin=0)
    assert np.array_equal(plugin['n'], _state(ELECTROPORATION, 3, 3)['n'])
    if fmt >= 3:
        assert back['regions'][1]['fingerprint'] == fingerprint(TTP)
    else:
        assert back['regions'][1]['fingerprint'] is None


def test_release19_plugin_table(tmp_path):
    """ release 19.0 stores the model's size for a plugin; the true size is
        recovered from the file size, and the file is written back unchanged
    """
    content = _content()
    content['regions'][1]['plugins'][0]['stored_size'] = 136
    fname = str(tmp_path / 'v19.roe')
    RoeWriter().write(content, fname)
    data = open(fname, 'rb').read()
    assert _string('Electroporation_DeBruinKrassowska98') + struct.pack('<i', 136) in data

    reader = RoeReader()
    back   = reader.read(fname)
    plugin = back['regions'][1]['plugins'][0]
    assert plugin['size'] == 8 and plugin['stored_size'] == 136
    assert np.array_equal(reader.region_state(1, ELECTROPORATION, plugin=0)['n'],
                          _state(ELECTROPORATION, 3, 3)['n'])
    again = str(tmp_path / 'again.roe')
    RoeWriter().write(back, again)
    assert open(again, 'rb').read() == data

    # without stored_size the table holds the true size, which the reference
    # needs to restore the plugin
    del back['regions'][1]['plugins'][0]['stored_size']
    RoeWriter().write(back, again)
    assert RoeReader().read(again)['regions'][1]['plugins'][0]['stored_size'] == 8


def test_release19_two_plugin_kinds_need_sizes(tmp_path):
    """ two unknown plugin sizes cannot come from one file size """
    content = _content()
    defib = [('Ki', 'GlobalData_t'), ('__sl_i2c_local', 'GlobalData_t')]
    content['regions'][1]['plugins'][0]['stored_size'] = 136
    content['regions'][1]['plugins'].append({'name': 'Defib_AshiharaTrayanova', 'size': 16,
                                             'stored_size': 136})
    content['records'][1] = np.concatenate(
        (content['records'][1], encode_records(_state(defib, 3, 5), defib, False)), axis=1)
    fname = str(tmp_path / 'two.roe')
    RoeWriter().write(content, fname)
    with pytest.raises(ValueError, match='set_plugin_sizes'):
        RoeReader().read(fname)
    reader = RoeReader({'plugin_sizes': {'Defib_AshiharaTrayanova': 16}})
    back   = reader.read(fname)
    assert [p['size'] for p in back['regions'][1]['plugins']] == [8, 16]


def test_float32_time(tmp_path):
    """ the time is stored as float32, as the reference stores it """
    content = _content()
    content['time'] = 1234.567
    fname = str(tmp_path / 't.roe')
    RoeWriter().write(content, fname)
    ctime = RoeReader().read(fname)['time']
    assert ctime == float(np.float32(1234.567)) and ctime != 1234.567


def test_fingerprint_is_fnv1a():
    """ FNV-1a 64: the offset basis for no input, and the standard vector of
        the one-byte input "a" for the byte sequence of a field
    """
    assert fingerprint([]) == 0xcbf29ce484222325
    value = 0xcbf29ce484222325
    for byte in b'a':
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    assert value == 0xaf63dc4c8601ec8c
    expected = 0xcbf29ce484222325
    for byte in b'a\x00' + struct.pack('<II', 8, 7):
        expected = ((expected ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    assert fingerprint([('a', 'GlobalData_t')]) == expected
    assert fingerprint([('a', 'Gatetype')]) != expected


def test_refused_files(tmp_path):
    """ a foreign file and a truncated one are refused with the file name """
    fname = str(tmp_path / 'bad.roe')
    with open(fname, 'wb') as fbad:
        fbad.write(_string('Not_a_dump') + bytes(32))
    with pytest.raises(ValueError, match='not an ionic state file'):
        RoeReader().read(fname)
    RoeWriter().write(_content(), fname)
    data = open(fname, 'rb').read()
    with open(fname, 'wb') as fbad:
        fbad.write(data[:-3])
    with pytest.raises(ValueError, match='bytes of node records expected'):
        RoeReader().read(fname)


def test_file_names():
    """ the names the reference gives its state files """
    assert tsav_label(7.3) == '7.3'
    assert tsav_label(15) == '15.0'
    assert tsav_label(0.25) == '0.2'        # printf rounds the binary value, ties to even
    assert tsav_file_name('state', tsav_label(780.0)) == 'state.780.0.roe'
    assert tsav_file_name('state', 'paced') == 'state.paced.roe'
    assert checkpoint_file_name(10.0) == 'checkpoint.10.0.roe'
    assert exit_file_name(12.5) == 'exit.save.12.500.roe'
