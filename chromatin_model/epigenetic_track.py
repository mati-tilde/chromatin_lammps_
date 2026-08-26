import pyBigWig

# class for epigeentic tracks containing signals related to acetilation and metilation.
class EpigeneticTrack:

    def __init__(self, name, filename):				# defining the constructor
        self.name = name
        self.filename = filename
        self.bw = pyBigWig.open(filename)

# methods: 
    def header(self):
        return self.bw.header()

    def chromosomes(self):					# allows to obtain the entire list of chromosomes and their length from a track
        return self.bw.chroms()

    def chrom_length(self, chromosome):				# allows to identify one chromosome and obtain their length
        return self.bw.chroms(chromosome)

    def get_signal(self, chromosome, start, end):		# allows to obtain per-pair signal in a specific chromosome
        return self.bw.values(
            chromosome,
            start,
            end
        )

    def get_intervals(self, chromosome, start, end):		# allows to obtain the intervals as they were stored in the pyBigWig file
        return self.bw.intervals(
            chromosome,
            start,
            end
        )

    def stats(self, chromosome, start, end, stat="mean"):	# allows to evaluate statistical results for a specific chromosome
        return self.bw.stats(
            chromosome,
            start,
            end,
            type=stat
        )

    def bead_signal(self, chromosome, n_beads, stat="mean"):	# for a single chrom: to make a plot according to our resolution of 10kbp per bead 

        length = self.chrom_length(chromosome)

        return self.bw.stats(
            chromosome,
            0,
            length,
            type=stat,
            nBins=n_beads
        )

    def multi_bead_signal(self, chromosomes, n_beads):
        
        total_length = 0

        for chrom in chromosomes:				# merging the length of the chromosome 
            total_length += self.chrom_length(chrom)

        signal=[]

        #print(total_length)
        
        for chrom in chromosomes:				# assigning beads to each part of the chrom according to their length
            length = self.chrom_length(chrom)

            beads = round(n_beads * length / total_length)
            
            arm_signal = self.bead_signal(chrom, beads)
            signal.extend(arm_signal)

        #print(len(signal))

        assert len(signal) == n_beads				# sanity check

        return signal



    def close(self):						# closes the file.
        self.bw.close()
