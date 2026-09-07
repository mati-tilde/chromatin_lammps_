# =================================
# Chromatin LAMMPS Model
# =================================

## What this repository does:
This repository contains the code and input files needed to generate
LAMMPS `.data` files for the chromatin model of the drosophila.
A `.data` file contains topological information, coordinates and 
epigenetic signals for all Bead objects.

The user only needs to provide:
1. the experimental chromosome annotation file (.BED file; .bw files for epigenetic signals
    are already included in pipeline);
2. the desired geometry (helix, random walk fully operational; persistent random walk in progress);
3. a few simulation parameters (see below).

The internal Python modules handle the following actions:
parsing, chromosome construction by geometry of choice, signal annotation,
coordinate generation, and LAMMPS formatting.

## Quick start:

### 1. Enter the model directory.
cd chromatin_model/

### 2. Choose the geometry.
Open 'writer.py' and set:

GEOMETRY = "random_walk"

or

GEOMETRY = "helix"

(for future reference, GEOMETRY = "persistent_random_walk")

### 3. Set the main parameters.
NOTE: The choice of parameters is quite "fixed" for now; working on making
it a free choice, for the model to be independent.

TESTED PARAMETERS (validated in short lammps runs).
In 'writer.py', set parameters according to geometry of choice:

-----------------------------------------------------------------------------
if: GEOMETRY = "helix" :

HELIX_RADIUS = 5.0
HELIX_HEIGHT = 68
HELIX_TURNS = 50
(helix drift is set by default, working on making it independent as well.)
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
if: GEOMETRY = "random_walk" :
STEP_SIZE = 1.0
SEED = 42
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
for: CYLINDER BOUNDARY (needed for random walk only):
CYLINDER_RADIUS = 8.0
CYLINDER_HEIGHT = 68.0
-----------------------------------------------------------------------------

### 4. Generate the LAMMPS data file.
Use:
python writer.py

The resulting file is (accordingly):

"rw_genome.data"

or

"helix_genome.data"

or the "informed_..._genome.data" if signals are implemented. 

### 5. Expected output.
After running 'python writer.py', the generated '.data' file should contain:

. 26512 atoms
. 26506 bonds
. 2 atom types
. 1 bond type
. 20 frozen endpoint atoms

and, if informed of signals:

. 26512 rows of four signals each in the order:
  H3K27AC - H3K9AC - H3K27ME3 - H3K9ME3


## Geometry options.
A full genome consists of two copies of chromosomes 2, 3 and X; each chromosome (and copy) is
generated according to the geometry of choice and displaced inside a nucleus in a way that
reduces overlap between chromosomes.
This is done differently according to geometry.

### 1. Random walk:
Implies: six chromosomes independently generated with a self-avoiding random walk geometry, each
built inside a cylinder (see "cylinder boundary" above) to provide confinement in space. The
boundary can be removed, should one want, but this configuration has not been tested yet.
Each chromosome is then translated in space in a hexagon-like shape, to fit inside the nucleus.

### 2. Helix:
Implies: haploid generated with a helix geometry of given dimensions for each arm of each chromosome,
with guaranteed continuity at arms junction. Each arm is built with a drift (drift is 5 for chr2 and chr3,
0 for chrX). After construction, each chromosome is copied and then translated in space to fit inside
the nucleus.

### 3. Persistent random walk:
Work in progress.

Status:
. chromosome parsing and coordinate generation -- DONE
. translation in space -- DONE
. self-avoidance -- in progress


## Frozen atoms.
The first and last bead of each chromosome arm are assigned
the frozen atom type, 2.

This is fixed in:
'writer_funcs.py', in function 'write_atoms';
'writer.py', in N_ATOM_TYPES = 2.

For the diploid genome this corresponds to 20 frozen atoms.

These atoms are subsequently fixed in LAMMPS with:

group for_freeze type 2
fix freeze for_freeze setforce 0.0 0.0 0.0

## Signal section.
Retrieves the four epigenetic signals that are stored in each
Bead object and allows lammps to access them.


## LAMMPS simulation.
To run a simulation, provide the correct .data file in lammps:

fix signals all property/atom d2_signals 4 ghost yes
read_data informed_..._genome.data fix signals NULL Signals


## Repository structure.

### Fundamental pipeline.
. 'writer.py'
  Main entry point for generating a LAMMPS '.data' file.

. 'builder.py'
  Builds chromosome and diploid genome coordinates.

. 'geometry.py'
  Implements the available geometry generators and boundary constraints.

. 'beads_class.py'
  Defines a class containing all information for beads.

. 'genome_parser.py'
  Reads the experimental '.BED' chromosome annotation.

. 'writer_funcs.py'
  Handles formatting and writing of the LAMMPS data file.

. 'genome_constants.py'
  Contains chromosome sizes, layouts and global parameters.

. 'epigenetic_track.py', 'func_epigen.py'
  Handle epigenetic signal processing.

. 'signal_annotator.py'
  Reads .bw files and assigns epigenetic signals to beads.

. 'lammps_genome.in', 'informed_lammps.in'
  Input file for LAMMPS.