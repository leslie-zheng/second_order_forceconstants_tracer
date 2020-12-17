#!/bin/python

"""
Forcre Constant Tracer
Created on 12/16/2020
"""

__author__="Zheng Jiongzhi"
__version__="0.1_r"
__email__="jzhengap@connect.ust.hk"

print(__version__, __author__,__email__)

import h5py
import sys
import numpy as np
import yaml
from optparse import OptionParser
from phonopy.interface.vasp import read_vasp
from phonopy.structure.cells import get_supercell, Primitive, get_smallest_vectors, get_reduced_bases
from phonopy.structure.atoms import PhonopyAtoms
import matplotlib.pylab as pl
import matplotlib.cm as cm
import itertools

# Read FORCE_CONSTANTS
def parse_FORCE_CONSTANTS(filename):
    """

    :param filename:
    :return:
    """
    fcfile = open(filename)
    num = int((fcfile.readline().strip().split())[0])
    force_constants = np.zeros((num, num, 3, 3), dtype=float)
    for i in range(num):
        for j in range(num):
            fcfile.readline()
            tensor = []
            for k in range(3):
                tensor.append([float(x) for x in fcfile.readline().strip().split()])
            force_constants[i, j] = np.array(tensor)
    return force_constants

# get distance between the random atom in supercell and the center atom
def get_distance(atoms, a0, a1, tolerance=1e-5):
    """
    Return the shortest distance between a pair of atoms in PBC
    :param atoms:
    :param a0:
    :param a1:
    :param tolerance:
    :return:
    """
    reduced_bases = get_reduced_bases(atoms.get_cell(), tolerance)
    scaled_pos = np.dot(atoms.get_positions(), np.linalg.inv(reduced_bases))
    for pos in scaled_pos:
        pos -= np.rint(pos)
    distance = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                distance.append( np.linalg.norm(np.dot(scaled_pos[a0] - scaled_pos[a1] + np.array([i,j,k]), reduced_bases)))
    return min(distance)

def get_equivalent_smallest_vectors(atom_num_supercell,
                                    atom_number_primitive,
                                    supercell,
                                    primitive_lattice,
                                    symprec=1.0e-5):
    """

    :param atom_num_supercell:
    :param atom_number_primitive:
    :param supercell:
    :param primitive_lattice:
    :param symprec:
    :return:
    """
    distances = []
    differences = []
    reduced_bases = get_reduced_bases(supercell.get_cell, symprec)
    positions = np.dot(supercell.get_positions(), np.linalg.inv(reduced_bases))

    for pos in positions:
        pos -= np.rint(pos)

    p_pos = positions[atom_number_primitive]
    s_pos = positions[atom_num_supercell]

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                diff = s_pos + np.array([i, j, k]) - p_pos
                differences.append(diff)
                vec = np.dot(diff, reduced_bases)
                distances.append(np.linalg.norm(vec))
    minimum = min(distances)
    smallest_vectors = []
    for i in range(27):
        if abs(minimum - distances[i]) < symprec:
            relative_scale = np.dot(reduced_bases, np.linalg.inv(primitive_lattice))
            smallest_vectors.append(np.dot(differences[i], relative_scale))

    return smallest_vectors



parser = OptionParser()
parser.add_option("--mode", dest="mode", type=int, default=2,
                  help="The mode of the force constants, harmonic (2) or anharmonic (3)")
parser.add_option("-c", "--poscar", dest="poscar", type="string", default="POSCAR",
                  help="The POSCAR file of the unit-cell")
parser.add_option("--dim", dest="dim", type="string", default=None,
                  help="Supercell dimension")
parser.add_option("-o", dest="output", type="string", default=None,
                  help="Output file name of the plot (in PDF format)")
parser.add_option("--center", dest="center_atom", type=int, default=0,
                  help="The center atom (in the unit-cell) for the force constants to print out")
parser.add_option("--legend", dest="is_legend", action="store_true",
                  help="If set, the legend of the atoms are shown")
parser.add_option("--ddfine", "--distance_define", dest="ddefine", type="string", default="max",
                  help="Set the distance definition for fc3 (min, max, cir or hydraulic)")
parser.add_option("--dirt", "--dir", dest="dirt", type="string", default=None,
                  help="The direction tensor (e.g. 'xx', 'yy', 'zz' for fc2 or 'xxx', 'yyy', 'zzz' for fc3). If not specify, a generalized borm "
                       "is calculated. ")
parser.add_option("--rot", "--rot_mat", "--rotation_matrix", dest="rotation_matrix", type="string", default=None,
                  help="A rotation matrix to transform the original force constants to a rotated coordinate. ")
parser.add_option("--projection", "--proj", dest="projection", type="string", default=None,
                  help="The projection direction of the distance vector.")
parser.add_option("-r", "--relative", dest="is_relative", action="store_true",
                  help="Set the relative value as the vertical axis.")
parser.add_option("--log", dest="is_log", action="store_true",
                  help="print in logarithm scale (vertical).")
parser.add_option("--ymin", dest="ymin", type=float, default=1e-5,
                  help="Minimum of y value in plot. ")
parser.add_option("--ymax", dest="ymax", type="float", default=None,
                  help="Maximum of y value in plot.")
parser.add_option('--xreverse', dest="xreverse", type="string", default=None,
                  help="export the atom triplet corresponding to the largest fc within the given distance range"
                       "(e.g. '3.4 3.5'). ")

(options, args) = parser.parse_args()
mode = options.mode

#specify the filename
filename1=None
if len(args) == 0:
    print("The force constants file is specified automatically! ")
    if mode == 2:
        filename = "fc2.hdf5"
    elif mode == 3:
        filename = "fc3.hdf5"
    else:
        print("Force constant modes other than 2 and 3 have not been implemented!")
        sys.exit(1)
elif len(args) == 1:
    filename = args[0]
else:
    print("Two force constants are compared")
    filename = args[0]
    filename1 = args[1]

#dimension of the supercell
if options.dim is not None:
    dim = [int(x) for x in options.dim.split()]
    if len(dim) == 9:
        dim = np.array(dim).reshape(3, 3)
    elif len(dim) == 3:
        dim = np.diag(dim)
    else:
        print("Error! Number of elements of DIM tag has to be 3 or 9.")
        sys.exit(1)
    if np.linalg.det(dim) < 1:
        print ("Error! Determinant of supercell matrix has to be positive. ")
        sys.exit(1)
else:
    rotmat = None

#Extract the maximum force constants and the corresponding interacting pair or triplet

if options.xreverse is not None:
    reverse_range = map(float, options.xreverse.replace(","," ").split())
    if len(reverse_range) == 1:
        reverse_range = [reverse_range[0]-0.01, reverse_range[0] + 0.01]
    else:
        reverse_range = [reverse_range[0], reverse_range[1]]

# Direction

dir_map={"x":0, "y":1, "z":2}
dirt = None
if options.dirt is not None:
    dirt_string = options.dirt.strip().replace(",", " ").replace(" "," ")
    print(dirt_string)
    for key in dirt_string:
        if key not in dir_map.keys():
            print("Error: The direction tensor is set incorrectly!")
            sys.exit(1)
    dirt = [dir_map[s] for s in dirt_string]
    print (dirt)
    if len(dirt) > 0:
        if mode == 2:
            if len(dirt) == 1:
                dirt = [dirt[0], dirt[0]]
            elif len(dirt) > 2:
                print("Warning! The given direction tensor is ambiguous"
                      "Only the first two component is extracted !")
                dirt = dirt[:2]
        if mode == 3:
            if len(dirt) == 1:
                dirt = [dirt[0], dirt[0], dirt[0]]
            elif len(dirt) >3:
                print("Warning! The given direction tensor is ambiguous"
                      "Only the first three components are extracted!")
                dirt = dirt[:3]
            elif len(dirt) == 2:
                print("Error! The given direction tensor component are not enough! ")
                sys.exit(1)

if options.projection is not None:
    projection_string = options.projection.strip().replace(",", " ").replace(" "," ")
    if projection_string[0] not in dir_map.keys():
        print("Error! The projection is set incorrectly!")
        sys.exit(1)
    projection = dir_map[projection_string[0]]
else:
    projection = None
if options.dirt is not None and options.is_log:
    print("Warning! The dirt and log arguments are not compatible"
          "All the values on the graph are set as positive!")

#center
center_atom = options.center_atom

# read the force constants

if filename.find('hdf5') != -1:
    f = h5py.File(filename, 'r')
    try:
        if mode == 2:
            if 'fc2' in f.keys():
                fc = f['fc2'][:]
            elif 'force_constants' in f.keys():
                fc = f['force_constants'][:]
        elif mode == 3:
            fc = f['fc3'][:]
    except KeyError:
        print("Error: Force constants related keys do not exist."
              "Maybe the mode is set incorrectly.")
        sys.exit(1)
else:
    if mode == 2:
        fc = parse_FORCE_CONSTANTS(filename)
#        print(fc)
    else:
        print("Only harmonic force constants can be read from a txt-format file")
        sys.exit(1)

if filename1 is not None:
    if filename1.find('hdf5') != -1:
        f = h5py.File(filename1, 'r')
        try:
            if mode == 2:
                if 'fc2' in f.keys():
                    fcp = f['fc2'][:]
                elif 'force_constants' in f.keys():
                    fcp = f['force_constants'][:]
            elif mode == 3:
                fcp = f['fc3'][:]
        except KeyError:
            print("Error! Force constants related to keys do not exist."
                  "Maybe the mode is set incorrectly! ")
            sys.exit(1)
    else:
        if mode == 2:
            fcp = parse_FORCE_CONSTANTS(filename)
        else:
            print("Only harmonic force constants can be read from a txt-format file")
            sys.exit(1)
    fc -= fcp


#POSCAR

cell = read_vasp(options.poscar)
#print (cell)
supercell = get_supercell(cell, dim)
#print (supercell)
s2u_map = supercell.get_supercell_to_unitcell_map()
#print (s2u_map)
chemical_symbols = supercell.get_chemical_symbols()
#print(chemical_symbols)
chemicals, index = np.unique(chemical_symbols,return_index=True)
print (chemicals, index)
chemicals=chemicals[np.argsort(index)]
print(chemicals)
chemap = []

if options.rotation_matrix is not None:
    lattice = cell.get_cell()
    positions = cell.get_scaled_positions()
    rpos = (positions-positions[0]) #relative positions
    rpos = np.where(np.abs(rpos) > 0.5, rpos-np.sign(rpos), rpos)
    rpos_cart = np.dot(rpos, lattice) # cartesian relative positions
    new_pos = np.dot(rpos_cart, rotmat.T) #new relation positions in Cartesian coordinate
    print("The cartesian coordinate of atoms after rotation:")
    for s, p in zip(cell.get_chemical_symbols(), new_pos):
        print("%5s: %15.5f: %15.5f: %15.5f" %(s, p[0], p[1], p[2]))
for c_unique in chemicals:
    che_temp = []
    for i, c_all in enumerate(chemical_symbols):
        if c_unique == c_all:
            che_temp.append(i)
    chemap.append(che_temp)
    print(chemap)
primitive = Primitive(supercell, np.linalg.inv(dim), 1e-5)
num_atom_prim = primitive.get_number_of_atoms()
if not 0 <= center_atom < num_atom_prim:
    print ("The center atom %d is out of the index range (%d-->%d)" %(center_atom, 0, num_atom_prim-1))
    sys.exit(1)
print("The center atom in the unitcell:")
symbol = primitive.get_chemical_symbols()[center_atom]
pos = primitive.get_scaled_positions()[center_atom]
print ("Index: %d; Symbol: %s; Position: [%15.6f, %15.6f, %15.6f]" %(center_atom, symbol, pos[0], pos[1], pos[2]))
print("Center atom in the supercell:")
num_atom_super = supercell.get_number_of_atoms()
index_super = primitive.get_primitive_to_supercell_map()[center_atom]
pos_super = supercell.get_scaled_positions()[index_super]
print("Index: %d; Symbol: %s; Position: [%15.6f, %15.6f, %15.6f]" %(index_super, symbol, pos_super[0], pos_super[1], pos_super[2]))

markers=itertools.cycle(".,ov^<>12348s*p+xD")
force_in_range = []
pair_in_range = []
dist_in_range = []
if mode == 2:
    if not options.is_relative:
        unit="eV/A^2"
    else:
        unit="arbi"
    distance = np.zeros(num_atom_super, dtype=float)
    fc_relative_trace = np.zeros(num_atom_super, dtype=float)
    if options.is_relative:
        self_trace = np.linalg.norm(fc[index_super, index_super])
        print ("Self-interaction strength for the center atom: %f (eV/A^2)" %self_trace)
        if np.abs(self_trace) < 1e-7:
            print("Warning: Self-interaction strength is too small"
                  "changing to absolute value instead")
            self_trace = 1.0
    else:
        self_trace = 1.0
    for i in np.arange(num_atom_super):
        if projection is not None:
            vectors = \
            get_equivalent_smallest_vectors(i, index_super,
                                            supercell=supercell,
                                            primitive_lattice= primitive.get_cell(),
                                            symprec=1.0e-5)
            vectors = np.dot(vectors, primitive.get_cell())
            project_vector = np.eye(3)[projection]
            distance[i] = np.abs(np.dot(vectors, project_vector)).main()
        else:
             distance[i] = get_distance(supercell, index_super, i)
             #print (distance[i])
        if dirt is not None:
            if options.rotation_matrix is not None:
                fc_temp = np.einsum("ab, cd, bd -> ac", rotmat, rotmat, fc[index_super, i])[dirt(0), dirt[1]]
            else:
                fc_temp = fc[index_super, i, dirt[0], dirt[1]]

        else:
            fc_temp = np.linalg.norm(fc[index_super, i])

        if options.is_log:
            fc_relative_trace[i] = np.abs(fc_temp / self_trace) + 1e-10

        else:
            fc_relative_trace[i] = fc_temp / self_trace
        if options.xreverse is not None:
            if distance[i] < reverse_range[1] and distance[i] > reverse_range[0]:
                pair_in_range.append(index_super, i)
                force_in_range.append(fc_relative_trace[i])
                dist_in_range.append(distance[i])

    if options.xreverse is not None:
        ind = np.argmax(force_in_range)
        print ("Max force component within the given distance range %f to %f" %tuple(reverse_range))
        print ("pair:", pair_in_range[ind])
        print ("Max force:", force_in_range[ind])
        sys.exit(0)
    distance_map={}
    for i, c in enumerate(chemicals):
        for j in chemap[i]:
            distance_map.setdefault(c, []).append([distance[j], fc_relative_trace[j]])
    print ("Distance (A) and bond strength (%s)" %unit)
    for i, c in enumerate(chemicals):
        dist = np.array(distance_map[c])
        dargsort = np.argsort(dist[:,0])
        distance_map[c] = np.vstack((dist[:,0][dargsort], dist[:,1][dargsort])).T
        print ("%15s %15s " %("dist(A)", symbol+'-'+c), end='')
    print()
    lenght=0
    while True:
        all_empty=True
        for i, c in enumerate(chemicals):
            dist = distance_map[c]
            if len(dist)<=lenght:
                print("%15s %15s " %(" "," "))
            else:
                all_empty=False
                print("%15.7f %15.7f" %tuple(abs(dist[lenght])), end='')
        print()
        lenght+=1
        if all_empty:
            break
    colors = cm.rainbow(np.linspace(0, 1, len(chemicals)))
    for i in np.arange(len(chemicals)):
        marker = next(markers)
        if options.is_legend:
            pl.scatter(distance[chemap[i]], np.abs(fc_relative_trace[chemap[i]]), label=symbol+"-"+chemicals[i], color=colors[i], marker=marker)
        else:
            pl.scatter(distance[chemap[i]], fc_relative_trace[chemap[i]], label=symbol+"-"+chemicals[i], color=colors[i], marker=marker)
    pl.xlabel("Distance(A)")
    pl.ylabel("Bond strength (%s)"%unit)
    if options.is_legend:
        pl.legend()
    if options.is_log:
        pl.yscale("log")
    if options.output == None:
        output = filename.split(".")[0]
        if filename1 is not None:
            output += "-"+filename.split(".")[0]
        if options.dirt is not None:
            output += "-%s"%dirt_string
        if options.projection is not None:
            output += "-%s.pdf"%symbol
    else:
        output = options.output
    if options.ymin is not None:
        pl.ylim(ymin=options.ymin)
    if options.ymax is not None:
        pl.ylim(ymax=options.ymax)

    pl.savefig(output)

#    pl.savefig(output)








