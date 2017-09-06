import numpy as np

def convert_embed_to_np(emb_file, np_file):
    print 'Convert %s to %s' % (emb_file, np_file)
    with open(emb_file) as f:
        lines = f.readlines()
    t = lines[0].split()
    m = int(t[0])
    n = int(t[1])
    mat = np.zeros((m, n))
    for line in lines[1:]:
        t = line.rstrip().split()
        r = int(t[0])
        li = [ float(x) for x in t[1:] ]
        mat[r-1] = li
    np.save(np_file, mat)

