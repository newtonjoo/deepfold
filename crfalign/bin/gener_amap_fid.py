#!/usr/bin/env /user/deepfold/anaconda3/envs/unifold/bin/python

import os, sys
sys.path.insert(0,'/user/deepfold/users/newton/deepfold')
err = sys.stderr

from os.path import exists
import numpy as np
import gzip

from unifold.data.mmcif_parsing import parse as parse_mmcif_string
from unifold.data.templates import _get_atom_positions as get_atom_positions
from Bio.PDB import protein_letters_3to1

mmcif_path = '/mnt/user/protein/database/pdb/data/structures/divided/mmCIF'
mmcif_path = '/gpfs/database/casp15/pdb/data/structures/divided/mmCIF'

ca_ca = 3.80209737096

# 37 atom types
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

def gen_amap_fid (code, outfile=sys.stdout, force=True):
    if outfile != sys.stdout:
        if force or not exists(outfile):
            out = open(outfile, 'w')
        else:
            print ('%s is exist, break' % outfile, file=err)
            return
    else:
        out = sys.stdout

    code = code  # 1tonA, 1ton_A, 1to9_B (CSD, MSE), 2zfo_D, 6rm3_LN0
    name = code.replace('_','')
    pid = name[:4]
    fid = name[4:]      # chain_id
    dv  = pid[1:3]

    #print (code, pid, fid, dv)

    cif_path = '%s.cif' % (code)
    if exists(cif_path):
        print (cif_path, 'exist...', file=err)
    else:
        cif_path=f'{mmcif_path}/{dv}/{pid}.cif.gz'
        print (cif_path, 'exist...', file=err)

    # mmcif parser
    if cif_path.endswith('.gz'):
        cif_string = gzip.open(cif_path, 'rb').read().decode('utf-8')
    else:
        cif_string = open(cif_path, 'r').read()

    mmcif_obj = parse_mmcif_string(
            file_id=pid, mmcif_string=cif_string).mmcif_object

    residues = mmcif_obj.seqres_to_structure[fid]
    residue_names = [residues[t].name for t in range(len(residues))]

    # 7c03_A, 4tza_A, ... (due to special residues)
    #sequence = ''.join([protein_letters_3to1[a] for a in residue_names])
    sequence = []
    for aaa in residue_names:
        try:
            a = protein_letters_3to1[aaa]
            if len(a) > 1: a = 'X'
        except:
            a = 'X'
            print (f'Unknown residue: {aaa} to X in {code}', file=err)
        sequence.append(a)
    sequence = ''.join(sequence)

    chain = mmcif_obj.structure[fid]
    #cid = mmcif_obj.chain_id_map[fid]

    _, mask = get_atom_positions (mmcif_obj, fid, max_ca_ca_distance=float('inf'))

    # backbone mask and CA mask
    bb_mask = np.array([False]*37) ; 
    for i in [0,1,2,4]: bb_mask[i] = True

    bb = 0
    ca = 0
    b1 = 0

    for i in range(len(sequence)):
        res_at_position = residues[i]

        resname = res_at_position.name

        if res_at_position.is_missing: 
            resnum   = '-'
            ins_code = ' '
        else:
            res = chain[(res_at_position.hetflag,
                         res_at_position.position.residue_number,
                         res_at_position.position.insertion_code)]
            _, resnum, ins_code = res.id
            #resnam2 = res.resname
        #print (f'{pid:<4} {i+1:>4} {sequence[i]} {resname:>4} {resnum:>4}{ins_code} {mask[i]} {mask[i][bb_mask]} {mask[i][ca_mask]}')

        nbb= sum(mask[i][bb_mask])

        if nbb == 0:
            bb_flag = False
            print (f'{i+1:>5} {sequence[i]:1}    -    -    - 0       -   -   - - {mask[i]}', file=out)
            continue

        bb_count = 0
        if nbb == 4: 
            bb += 1 ; bb_str = f'{bb:4}' ; bb_count +=1
        else:
            bb_str = '   -'

        if mask[i][1] == 1: 
            ca += 1 ; ca_str = f'{ca:4}' ; bb_count +=1   # CA check
        else:
            ca_str = '   -'

        if nbb > 0 :
            b1 += 1 ; b1_str = f'{b1:4}' ; bb_count +=1
        else:
            b1_str = '   -'

        a = resname         # seqres
        #b = resnam2         # pdb
        try: 
            b = protein_letters_3to1[a]       # 2,3,4 letters are possible
        except:
            b = a
        if len(b) == 1 or len(b) == 4: b = a

        c = sequence[i]     # one letter
        print (f'{i+1:>5} {sequence[i]} {bb_str} {ca_str} {b1_str} {bb_count:1}   {resnum:>4}{ins_code} {a:3} {b:>3} {c:1} {mask[i]}', file=out)

    return

if __name__ == '__main__':

    code = sys.argv[1]
    gen_amap_fid (code)
