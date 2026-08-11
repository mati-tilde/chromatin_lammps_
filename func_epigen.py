# FUNCTIONS FOR script.py
import statistics

#-------------------------------------------------------------------------
# write_track_info: writes basic information of a track to an output file
#-------------------------------------------------------------------------

def write_track_info(track): 

	o_file = f"{track.name}_info.txt"

	with open(o_file, "w") as file:

        	file.write(f"Track: {track.name}\n")
        	file.write("=" * 40 + "\n\n")


        	file.write("HEADER\n")
        	file.write(str(track.header()))
        	file.write("\n\n")

        	file.write("CHROMOSOMES\n")				#allows to have a full list of all chromosomes in the file and their length

        	for chrom, length in track.chromosomes().items():       #for gives a loop over the pairs of ID and length
	                file.write(f"{chrom}\t{length}\n")

	print(f"{o_file} has been successfully created.")

#-------------------------------------------------
# write_signal: retrives signal data from the file
#-------------------------------------------------

def write_signal(track, chromosome, start, end):

    signal = track.get_signal(
        chromosome,
        start,
        end
    )

    intervals = track.get_intervals(
        chromosome,
        start,
        end
    )

    o_file = (
        f"{track.name}_signal_"
        f"{chromosome}.txt"
    )

    with open(o_file, "w") as file:

        file.write(f"Track: {track.name}\n")
        file.write(f"Chromosome: {chromosome}\n")
        file.write("=" * 40 + "\n\n")

        file.write("INTERVALS\n")
        file.write("Start\tEnd\tSignal\n")

        for interval in intervals:
            file.write(
                f"{interval[0]}\t{interval[1]}\t{interval[2]}\n"
            )

        file.write("\n\n")

        file.write("PER-BASE SIGNAL\n")
        file.write(f"Region: {start} - {end}\n")
        file.write("Position\tSignal\n")

        for i, value in enumerate(signal):
            file.write(
                f"{start+i}\t{value}\n"
            )

    print(f"{o_file} has been successfully created.")

#------------------------------------------------------------
# write_stats: gives statistics info on a specific chromosome
#------------------------------------------------------------

def write_stats(track, chromosome, start, end):

    mean = track.stats(
        chromosome,
        start,
        end,
        "mean"
    )

    maximum = track.stats(
        chromosome,
        start,
        end,
        "max"
    )

    minimum = track.stats(
        chromosome,
        start,
        end,
        "min"
    )

    o_file = (
        f"{track.name}_stats_"
        f"{chromosome}.txt"
    )

    with open(o_file, "w") as file:

        file.write(f"Track: {track.name}\n")
        file.write(f"Chromosome: {chromosome}\n")
        file.write(f"Region: {start}-{end}\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Mean:\t{mean[0]}\n")
        file.write(f"Maximum:\t{maximum[0]}\n")
        file.write(f"Minimum:\t{minimum[0]}\n")

    print(f"{o_file} has been successfully created.")

#----------------------------------------------------------------------
# genome_stats: gives statistic info on all the chromosomes in the file
#----------------------------------------------------------------------


def genome_stats(track):

    chromosomes = track.chromosomes()

    statistics = ["mean", "std", "min", "max"]

    results = {}

    for chrom, length in chromosomes.items():

        results[chrom] = {}

        for stat in statistics:
            results[chrom][stat] = track.stats(
                chrom,
                0,
                length,
                stat
            )[0]

    return results

# to use this result in the script: 
# results = genome_stats(selected_track)
# print(results["211000022278031"]["mean"])


#-----------------------------------------------------------
# write_genome_stats: writes to file the genome_stats result
#-----------------------------------------------------------

def write_genome_stats(track):

    results = genome_stats(track)

    o_file = f"{track.name}_genome_stats.txt"

    with open(o_file, "w") as file:

        file.write(f"Track: {track.name}\n")
        file.write("=" * 40 + "\n\n")

        file.write(
            "Chromosome\tMean\tStd\tMin\tMax\n"
        )

        for chrom, stats in results.items():

            file.write(
                f"{chrom}\t"
                f"{stats['mean']}\t"
                f"{stats['std']}\t"
                f"{stats['min']}\t"
                f"{stats['max']}\n"
            )

    print(f"{o_file} has been successfully created.")


#--------------------------------------------------------------------------------------------
# write_bead_stats: writes a statistics summary to file according to the right number of bins
#--------------------------------------------------------------------------------------------

def write_bead_stats(track, chromosomes, signal):

    signal = [								# replacing None values in the bw file
        0 if value is None else value
        for value in signal
    ]

    if isinstance(chromosomes, str):

        length = track.chrom_length(chromosomes)
        chrom_name = chromosomes

    else:

        length = 0

        for chrom in chromosomes:
            length += track.chrom_length(chrom)

    chrom_name = "+".join(chromosomes)
    
    resolution = length / len(signal)

    o_file = f"{track.name}_{chrom_name}_summary.txt"

    with open(o_file, "w") as file: 
        file.write(f"Track: {track.name}\n")
        file.write(f"Chromosomes(s): {chrom_name}\n")
        file.write("=" *40 + "\n")

        file.write("Chromosome attributes:\n")
        file.write(f"Chromosome length: {length} bp\n")
        file.write(f"Number of beads: {len(signal)}\n")
        file.write(f"Resolution: {resolution} bp/beads\n")
        file.write("\n\n")
        file.write("Statistical analysis:\n")
        file.write(f"Mean:  {statistics.mean(signal):.3f}\n")
        file.write(f"Std. deviation:  {statistics.stdev(signal):.3f}\n")
        file.write(f"Maximum:  {max(signal):.3f}\n")
        file.write(f"Minimum:  {min(signal):.3f}\n")

    print(f"{o_file} has been created successfully.")

#------------------------------------------------------------------------------
# read_beads: reads the file init_drosophila.data and stores atom types and ids
#------------------------------------------------------------------------------

def read_beads(filename, molecule_id):

    atom_ids = []
    atom_types = []

    reading = False

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()                  # remove white spaces

            if line.startswith("Atoms"):
                reading = True
                continue

            if not reading:
                continue                         # ignore everything before Atoms

            if line == "":
                continue                         # skip empty line after Atoms

            if line.startswith("Bonds"):
                break                            # stop reading atoms

            parts = line.split()                 # divide values into separate parts

            atom_id = int(parts[0])
            molecule = int(parts[1])
            atom_type = int(parts[2])

            if molecule == molecule_id:
                atom_ids.append(atom_id)
                atom_types.append(atom_type)

    return atom_ids, atom_types

#-----------------------------------------------------------------------
# chrom_length: defines the length of the chromosome from the .data file
#-----------------------------------------------------------------------

def chrom_length(filename, molecule_id):

    atom_ids, atom_types = read_beads(
        filename,
        molecule_id
    )

    return len(atom_ids) 

#----------------------------------------------------
# comparts_theta: assigns binary values to atom types
#----------------------------------------------------

def comparts_theta(atom_types): 

    theta = []
    
    for atom_type in atom_types: 

        if atom_type == 1:
            theta.append(1)

        elif atom_type == 2:
            theta.append(-1)

        elif atom_type == 3:
            theta.append(0)

        elif atom_type == 4:
            theta.append(0)

        else:
            raise ValueError(f"Unknown atom type: {atom_type}")


    return theta




























