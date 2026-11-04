EXPERIMENTAL HI-C MAPS -- WHAT WE KNOW (as of 4 November 2026)
==================================================================


1. AVAILABLE MAPS
-----------------
- Four merged maps, 10 kb resolution:
    WT stage 5, WT stage 10, DCA stage 10, DCD stage 10.
  DCA and DCD are mutants.
  Availability of ChIP tracks for the mutants and for stage 5?
- Five replicates of WT stage 10 (files WT_ST10_REP1 ... REP5, 10 kb).
- Target chosen for fitting: WT stage 10 (same stage as the BED compartments and the ChIP tracks).
- File examined in detail: WT_ST10.10000_exp.npz (merged WT stage 10).
  "_exp" means "experimental": the content is contact frequencies, not observed/expected :)


2. FILE FORMAT
--------------
- .npz containing one scipy sparse matrix (keys: indices, indptr, format, shape, data).
- One genome-wide matrix per file (all chromosomes in one matrix).
- Symmetric (both triangles stored).
- Entries are not whole numbers: normalized values, not raw counts!
- No NaN entries. Bins without data appear as rows/columns of zeros.


3. GENOME ASSEMBLY AND BIN LAYOUT
---------------------------------
- Assembly: dm6 (drosophila melanogaster sixth version of the reference genome assembly). Resolution: 10 kb.
- Merged map: 13758 x 13758 bins, in the order of the BED file:
      chr2L 2352 | chr2R 2529 | chr3L 2812 | chr3R 3208 | chr4 135 | chrX 2355 | chrY 367
  13758 = 13256 (the model's chr2L, chr2R, chr3L, chr3R, chrX) + 135 (chr4) + 367 (chrY).
  chr4 and chrY bin counts match the BED file. Layout has been confirmed.
- Replicates: 13391 x 13391 = 13758 - 367: the same layout WITHOUT chrY.
  chrY is the last chromosome, so all other bins have the same index in merged map and replicates.
- The model does not contain chr4 (and chrY), which sits between chr3R and chrX in the experimental
  order: model bins and experimental bins have different indices beyond chr3R.
- Bin index within an arm = 0-based 10 kb bin, interval [bin*10000, (bin+1)*10000) in 0-based coordinates.


4. NORMALIZATION
----------------
- Merged map: balanced (all biases have been already worked out). Row sums of all non-empty bins are identical
  (variation 0.000).
- Replicates: balanced, row-sum variation 0.008-0.010 (probably balanced before chrY
  was removed, or with a different tolerance). Small enough to be irrelevant.
- Values are relative contact frequencies with an arbitrary overall scale (balancing fixes row
  sums to a constant!), not absolute contact probabilities.


5. CONTENT BY GENOMIC DISTANCE (merged map, genome-wide, relative to s = 1)
--------------------------------------------------------------------------
    s (bins of 10 kb)   0       1    2       3       4       5       10       30       100      300       1000
    merged            0.8934  1   0.5185  0.3419  0.2484  0.1905  0.07863  0.01808  0.00615  0.001991  0.0002287
- The main diagonal (s = 0, a bin with itself) is filled.
- Values decay steadily with s: contact frequencies (an observed/expected map would be flat ~1).
- chr2 only, after masking and with p(s=1) = 1:
    P_exp(s): s=4: 0.249, s=10: 0.0791, s=30: 0.0182, s=100: 0.0060, s=300: 0.0021, s=1000: 0.000247
  Local power-law exponents of P(s) ~ s^-alpha (alpha taken as slope on a log-log plot):
    (just a first indication from six P(s) values -- i'll compute P(s) at every distance next)
    alpha ~ 1.1 between 100 kb and 1 Mb (s = 10-100); alpha ~ 1.4 between 1 and 10 Mb (s = 100-1000).
    finer (same calculation between consecutive values): 1.34 (s 10-30), 0.92 (30-100), 0.96 (100-300), 1.78 (300-1000).


6. EMPTY BINS (no contacts at all)
----------------------------------
- Merged map, per chromosome: chr2L 39, chr2R 43, chr3L 58, chr3R 27, chr4 10, chrX 47, chrY 149
  (total 373). Most likely the bins removed by the balancing (low coverage / unmappable).
- Replicates, per chromosome:
              chr2L  chr2R  chr3L  chr3R  chr4  chrX
    REP1       37     44     61     32     10    50
    REP2       39     46     61     32     10    58
    REP3       38     48     60     32     10    51
    REP4       38     48     60     32     10    51   (identical to REP3!!! see below)
    REP5       40     45     59     30     10    54
  The sets of empty bins are similar but not identical between files.
- chr2 (4881 bins): 82 empty in the merged map; 91 empty in the merged map or in at least one of
  REP1, REP2, REP3, REP5; valid bins (data everywhere): 4790.
- Relation with the BED file (chr2): 752 bins are absent from the BED; 54 of the 82 empty bins of the
  merged map are among them (62 of the 91 when replicates are included). So:
    - 28 bins empty in the merged map ARE in the BED;
    - about 700 bins absent from the BED DO have Hi-C data.
  Empty bins and BED-less bins are different sets; only empty bins need masking.


7. REPLICATES
-------------
- REP1: first copy truncated during transfer (zip header present, end of archive missing, 74 MB);
  re-transferred copy is fine.
- REP3 and REP4: identical files (same size, 94,800,618 bytes; identical diagnostics to every digit).
  Using both would underestimate the uncertainty. Distinct replicates available: REP1, REP2, REP3, REP5.
- Diagonal means relative to s = 1:
              s=0     s=2     s=3     s=4     s=5     s=10     s=30     s=100     s=300     s=1000
    REP1    0.9208  0.5233  0.3461  0.2523  0.1935  0.07967  0.0188   0.00624   0.002123  0.0002418
    REP2    0.9647  0.5252  0.3467  0.2507  0.1933  0.07825  0.01759  0.005781  0.00196   0.0002277
    REP3    0.9233  0.4978  0.3226  0.234   0.1785  0.07273  0.01649  0.00563   0.001956  0.0002079
    REP5    0.8206  0.5255  0.3496  0.2554  0.198   0.08277  0.01948  0.006282  0.002051  0.0002694
    merged  0.8934  0.5185  0.3419  0.2484  0.1905  0.07863  0.01808  0.00615   0.001991  0.0002287
  The replicates decay like the merged map; REP3 is systematically a little lower at s >= 2,
  REP5 a little higher.
- Cis fractions: REP1 0.8590, REP2 0.8672, REP3 0.8596, REP5 0.8223 (merged 0.8489).
- Consistency with the merged map (chr2, valid bins, each map scaled to p(s=1) = 1):
  mean of REP1/2/3/5 divided by the merged map: replicates and merged map are normalized consistently.


8. NOISE BETWEEN REPLICATES (chr2, REP1/2/3/5, valid bins, joint mask)
----------------------------------------------------------------------
- Relative noise sigma(s) / <p>(s), with sigma^2(s) = mean over the diagonal of the per-pixel
  variance across replicates:
    s=4: 0.218   s=10: 0.404   s=30: 1.00   s=100: 1.34   s=300: 2.10   s=1000: 6.47
- Without the joint mask (so mask from the merged map only): 0.237, 0.416, 1.03, 1.36, 2.11, 6.49.
  The joint mask removes the artificial part at short range (~8%); the long-range noise is genuine.
- Beyond ~300 kb (s >~ 30), the spread between replicates of a single pixel is as large as or larger
  than its mean: single pixels at large genomic distance carry little information!
- Noise grows with the signal at fixed s (counting statistics): a single sigma per diagonal
  under-weights the noise of high-contact pixels (compartments, domains) and over-weights that of
  low-contact pixels --> "insensitive to high-contact regions"

