import subprocess
def get_tm(pdb1, pdb2):
    """
    Caclulate the tm score of pdb1, pdb2.

    Args:
        pdb1 : model pdb file path, str
        pdb2 : native pdb file path, str

    Returns:
        tm score, float
    """
    cmd = 'tmscore'+' '+ pdb1 +' ' + pdb2 
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    stdout, stderr = p.communicate()
    outputs = stdout.decode().split('\n')
    tm = 0.0
    for line in outputs:
        if line.startswith('TM-score'):
            tm = float(line.split(' ')[5])
            return tm
