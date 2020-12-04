#!/bin/bash
db_ingest --tiledb_metadata inputs.tsv \
          --array_name db \
          --chrom_sizes hg38.chrom.sizes \
          --attribute_config_file attribs.txt \
          --coord_tile_size 10000 \
          --task_tile_size 1 \
          --write_chunk 10000000 \
          --threads 40 \
          --max_queue_size 50 \
          --max_mem_g 200
