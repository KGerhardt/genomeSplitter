# genomeSplitter
Python tool for quickly splitting a genome into usefully sized pieces

genomeSplitter is both a class accessible through the python API and a command line tool for quickly splitting genomes into chunks.

# Purpose

The original intent of genomeSplitter is threefold:
* Split very large genomes into evenly sized pieces for efficient parallel processing with python
* Split genomes of any size into chunks small enough to fit into limited RAM while processing in parallel
* Split very large contigs in a genome (e.g.< >2Gbp lungfish chromosomes) into chunks with overlaps so that aligners with 32-bit sequence length limits can function and not miss alignments due to breaking sequences at a bad place.

# Dependencies

* pyfastx (https://pubmed.ncbi.nlm.nih.gov/33341884/), available @  https://github.com/lmdu/pyfastx, https://pypi.org/project/pyfastx/, https://anaconda.org/bioconda/pyfastx

# Function

# API: 

```python
from genome_splitter import genomeSplitter

genome_file = 'path/to/genome'
outdir = 'output_directory/to/create'
threads = 4
chnk = 25_000_000
olap = 1_000_000

gs = genomeSplitter(genome_file = genome,
                  output_directory = outdir,
                  chunk_size = chnk,
                  overlap_size = olap,
                  processors = threads,
                  smart = False,
                  post_index = True,
                  verbose = True
                  )
paths_to_output_files = gs.run()
```

# Command line

```bash
python3 split_genomes.py --genome path/to/genome --output_directory output_directory/to/create --processors 4 --index_outputs --smart
```
