#!/usr/bin/env python
"""
    roe_reference_validation.py: checks RoeReader and RoeWriter against the
    reference simulator (openCARP), which must be installed (`openCARP` and
    `mesher` on the PATH).

    On a 525-node slab with two ionic regions it runs three cases to 20 ms,
    saving states with tsav = 7.3, 15 ms and chkpt_intv = 10 ms:
        A  Courtemanche | tenTusscherPanfilov + Electroporation_DeBruinKrassowska98
           (records mixing float and double fields, so C padding, and a plugin)
        B  Tomek | Tomek
        C  MitchellSchaeffer | MitchellSchaeffer
    and checks, for every state file the reference wrote:
        1. its name is the one the roewriter functions give;
        2. Vm matches the vm.igb frame at the same time (float32 precision);
        3. read and written back, the file is byte-identical;
        4. decoded field by field and encoded again from the values, the file
           is byte-identical except for the creation time and the revision.
    Then the reference restarts at 10 ms from the file of step 4, and its Vm to
    20 ms must equal the uninterrupted run bit for bit. For case A the table
    gets the plugin's true size first: release 19.0 writes the model's size
    there, cannot restart from its own file, and does restart from the
    corrected one (the original file is also tried, to show it).

    Usage: python roe_reference_validation.py [work_dir]
    work_dir defaults to a new temporary folder; nothing is written in the
    repository.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

from gpuSolve.IO.readers.igbreader import IGBReader
from gpuSolve.IO.readers.roereader import RoeReader
from gpuSolve.IO.readers.roereader import record_dtype
from gpuSolve.IO.writers.roewriter import RoeWriter
from gpuSolve.IO.writers.roewriter import checkpoint_file_name
from gpuSolve.IO.writers.roewriter import encode_records
from gpuSolve.IO.writers.roewriter import tsav_file_name
from gpuSolve.IO.writers.roewriter import tsav_label
from gpuSolve.carp_compatibility.svlayouts import SV_LAYOUTS

# (name, C type) lists by the reference's model name
FIELDS = {section: [(name, 'Gatetype' if gate else 'GlobalData_t')
                    for name, _kind, _source, _scale, gate in entries]
          for section, entries in SV_LAYOUTS.values()}

CASES = {'A': ['-imp_region[0].im', 'Courtemanche', '-imp_region[1].im', 'tenTusscherPanfilov',
               '-imp_region[1].plugins', 'Electroporation_DeBruinKrassowska98'],
         'B': ['-imp_region[0].im', 'Tomek', '-imp_region[1].im', 'Tomek'],
         # the MitchellSchaeffer potential is normalised (0..1): a smaller stimulus
         'C': ['-imp_region[0].im', 'MitchellSchaeffer', '-imp_region[1].im', 'MitchellSchaeffer',
               '-stim[0].pulse.strength', '1.0']}

TSAV        = (7.3, 15.0)
CHECKPOINTS = (0.0, 10.0, 20.0)
RESTART     = 10.0
SPACEDT     = 0.1

PAR = """meshname  = slab
bidomain  = 0
dt        = 25
timedt    = 1
spacedt   = {spacedt}
num_phys_regions       = 1
phys_region[0].name    = "myocard"
phys_region[0].ptype   = 0
phys_region[0].num_IDs = 2
phys_region[0].ID[0]   = 1
phys_region[0].ID[1]   = 2
num_gregions           = 1
gregion[0].num_IDs     = 2
gregion[0].ID[0]       = 1
gregion[0].ID[1]       = 2
num_imp_regions        = 2
imp_region[0].num_IDs  = 1
imp_region[0].ID[0]    = 1
imp_region[1].num_IDs  = 1
imp_region[1].ID[0]    = 2
num_stim               = 1
stim[0].crct.type      = 0
stim[0].ptcl.start     = 1.0
stim[0].ptcl.duration  = 2.0
stim[0].pulse.strength = 60.0
stim[0].elec.p0[0]     = -1100
stim[0].elec.p0[1]     = -300
stim[0].elec.p0[2]     = -300
stim[0].elec.p1[0]     = -850
stim[0].elec.p1[1]     = 300
stim[0].elec.p1[2]     = 300
"""


def make_mesh(work: str):
    """ a 2 x 0.4 x 0.4 mm slab at 100 um, tag 2 on the half x > 0 """
    subprocess.check_call(['mesher', '-size[0]', '0.2', '-size[1]', '0.04', '-size[2]', '0.04',
                           '-resolution[0]', '100', '-resolution[1]', '100',
                           '-resolution[2]', '100', '-mesh', 'slab'],
                          cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pts   = np.loadtxt(os.path.join(work, 'slab.pts'), skiprows=1)
    lines = open(os.path.join(work, 'slab.elem')).read().split('\n')
    nelem = int(lines[0])
    out   = [lines[0]]
    for line in lines[1:nelem + 1]:
        tokens = line.split()
        nodes  = [int(t) for t in tokens[1:5]]
        tokens[5] = '2' if pts[nodes, 0].mean() > 0.0 else '1'
        out.append(' '.join(tokens))
    with open(os.path.join(work, 'slab.elem'), 'w') as felem:
        felem.write('\n'.join(out) + '\n')
    with open(os.path.join(work, 'base.par'), 'w') as fpar:
        fpar.write(PAR.format(spacedt=SPACEDT))


def run_reference(work: str, simid: str, case: str, extra: list) -> bool:
    """ runs the reference on case, returns True when it finished cleanly """
    with open(os.path.join(work, simid + '.log'), 'w') as flog:
        code = subprocess.call(['openCARP', '+F', 'base.par'] + CASES[case]
                               + ['-simID', simid, '-tend', '20'] + extra,
                               cwd=work, stdout=flog, stderr=subprocess.STDOUT)
    return(code == 0)


def vm_frames(fname: str) -> np.ndarray:
    reader = IGBReader()
    reader.read(fname)
    return(reader.data().reshape(reader.nt(), -1))


def reencode(reader: RoeReader, content: dict) -> dict:
    """ the content rebuilt from decoded field values only, as a caller that
        holds the state in its own arrays would build it. stored_size keeps
        the release-19.0 plugin entry so that the bytes can be compared.
    """
    records : list = []
    regions : list = []
    for index, region in enumerate(content['regions']):
        model = FIELDS[region['model']]
        parts = [encode_records(reader.region_state(index, model), model, False)]
        plugins : list = []
        for jplug, plugin in enumerate(region['plugins']):
            pfields = FIELDS[plugin['name']]
            parts.append(encode_records(reader.region_state(index, pfields, plugin=jplug),
                                        pfields, False))
            plugins.append({'name': plugin['name'],
                            'size': record_dtype(pfields, False).itemsize,
                            'stored_size': record_dtype(model, False).itemsize})
        records.append(np.concatenate(parts, axis=1))
        regions.append({'model': region['model'], 'size': record_dtype(model, False).itemsize,
                        'plugins': plugins})
    return({'time': content['time'], 'num_nodes': content['num_nodes'],
            'globals': content['globals'], 'regions': regions, 'mask': content['mask'],
            'records': records})


def check_case(work: str, case: str) -> bool:
    folder = os.path.join(work, case + '_full')
    names  = ([tsav_file_name('state', tsav_label(t)) for t in TSAV]
              + [checkpoint_file_name(t) for t in CHECKPOINTS])
    found  = sorted(f for f in os.listdir(folder) if f.endswith('.roe'))
    ok     = sorted(names) == found
    print('case {}: names {} ({})'.format(case, 'match' if ok else 'DIFFER', ', '.join(found)))
    frames = vm_frames(os.path.join(folder, 'vm.igb'))
    for name in names:
        path    = os.path.join(folder, name)
        reader  = RoeReader()
        content = reader.read(path)
        dvm     = np.max(np.abs(content['globals']['Vm'] - frames[int(round(content['time'] / SPACEDT))]))
        RoeWriter().write(content, os.path.join(work, 'copy.roe'))
        original = open(path, 'rb').read()
        raw_same = open(os.path.join(work, 'copy.roe'), 'rb').read() == original
        rebuilt  = os.path.join(folder, 'ours_' + name)
        RoeWriter().write(reencode(reader, content), rebuilt)
        data     = open(rebuilt, 'rb').read()
        # bytes 16..27 hold the revision and the creation time of the header
        body_same = (len(data) == len(original) and data[:16] == original[:16]
                     and data[28:] == original[28:])
        ok = ok and raw_same and body_same and dvm < 1.0e-5 * max(1.0, np.max(np.abs(frames)))
        print('  {:22s} t = {:5.2f}  max|Vm - igb| = {:.1e}  copy {}  rebuilt {}'.format(
            name, content['time'], dvm, 'identical' if raw_same else 'DIFFERS',
            'identical but header time/revision' if body_same else 'DIFFERS'))
    return(ok)


def check_restart(work: str, case: str, source: str, label: str, expect: bool) -> bool:
    simid = '{}_restart_{}'.format(case, label)
    clean = run_reference(work, simid, case, ['-start_statef', source])
    if not clean:
        print('  restart from {}: the reference failed{}'.format(
            source, ' (expected)' if not expect else ''))
        return(not expect)
    full = vm_frames(os.path.join(work, case + '_full', 'vm.igb'))
    rest = vm_frames(os.path.join(work, simid, 'vm.igb'))
    # the restarted run writes its frames from the restart time on
    first = int(round(RESTART / SPACEDT))
    same  = np.array_equal(rest, full[first:first + rest.shape[0]])
    print('  restart from {}: Vm 10..20 ms {} the uninterrupted run'.format(
        source, 'bit-identical to' if same else 'DIFFERS from'))
    return(same == expect)


def main(work: str) -> bool:
    os.makedirs(work, exist_ok=True)
    print('work folder:', work)
    make_mesh(work)
    extra = (['-num_tsav', str(len(TSAV))]
             + sum([['-tsav[{}]'.format(i), str(t)] for i, t in enumerate(TSAV)], [])
             + ['-chkpt_start', '0', '-chkpt_intv', str(RESTART)])
    ok = True
    for case in CASES:
        if not run_reference(work, case + '_full', case, extra):
            print('case {}: the reference run failed, see {}_full.log'.format(case, case))
            return(False)
        ok = check_case(work, case) and ok
    ours = 'ours_' + checkpoint_file_name(RESTART)
    for case in ('B', 'C'):
        ok = check_restart(work, case, os.path.join(case + '_full', ours), 'ours', True) and ok
    # case A: the reference's own file, then ours with the plugin's true size
    ok = check_restart(work, 'A', os.path.join('A_full', checkpoint_file_name(RESTART)),
                       'reference', False) and ok
    fixed   = os.path.join('A_full', 'fixed_' + checkpoint_file_name(RESTART))
    content = RoeReader().read(os.path.join(work, 'A_full', ours))
    for region in content['regions']:
        for plugin in region['plugins']:
            del plugin['stored_size']
    RoeWriter().write(content, os.path.join(work, fixed))
    ok = check_restart(work, 'A', fixed, 'fixed', True) and ok
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return(ok)


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='roe_validation_')
    sys.exit(0 if main(os.path.abspath(folder)) else 1)
