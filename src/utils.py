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

def format_edgelist(edg_file):
    '''
    Converts an edgelist with arbitrarily labelled vertices
    to edgelist with vertices labeled 1...<num_vertices>
    '''
    with open(edg_file) as f:
        lines = f.readlines()

    # grab all vertices and sort them
    temp = []
    for line in lines:
        str_v = line.split()
        temp.append(int(str_v[0]))
        temp.append(int(str_v[1]))
    vertices = np.unique(temp)

    # create mapping of original vertex
    # labels to 1...<num_vertices>
    mapping = {}
    for x in range(len(vertices)):
        mapping[vertices[x]] = x+1

    # convert edgelist with new labels
    outfile = '%s_cleaned.edgelist' % edg_file.split('.')[0]
    with open(outfile, 'w') as f:
        for line in lines:
           str_v = line.split()
           output = '%d\t%d\n' % (mapping[int(str_v[0])], mapping[int(str_v[1])])
           f.write(output)

