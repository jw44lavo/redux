nextflow.preview.dsl=2

/*  PYTHON  */
include {
  python_species_filter;
  python_return_similar_pairs
} from "$baseDir/modules/python.nf"

/*  SHOVILL  */
include {
  shovill_assemble
} from "$baseDir/modules/shovill.nf"

/*  SOURMASH  */
include {
  sourmash_compute_signatures;
  sourmash_compare_signatures;
  sourmash_plot_comparison;
  sourmash_classify_signatures
} from "$baseDir/modules/sourmash.nf"


/************************************************************
* WORKFLOW  *************************************************
************************************************************/
workflow {
log.info ""
log.info ""
log.info """\
R E D U N D A N T - A N D - S P E C I E S - C H E C K
=====================================================
Start date:             ${params.date}
"""
log.info ""
log.info ""

  main:
    genome_size           = params.genome_size
    max_cores_per_process = params.max_cores_per_process
    jaccard_threshold     = params.jaccard_threshold
    database              = params.database
    reads                 = params.reads

    ch_reads = Channel.fromFilePairs(reads)

  /*  ASSEMBLY  */
    shovill_assemble(
      ch_reads,
      genome_size,
      max_cores_per_process
    )

  /*  SIGNATURES  */
    sourmash_compute_signatures(
      shovill_assemble.out
    )

  /*  COMPARISON ALL VS ALL */
    sourmash_compare_signatures(
      sourmash_compute_signatures.out.collect(),
      max_cores_per_process
    )

  /*  PLOT JACCARD INDEX HEATMAP */
    sourmash_plot_comparison(
      sourmash_compare_signatures.out[0]
    )

  /*  IDENTIFY SIMILAR ASSEMBLIES BY JACCARD INDEX  */
    python_return_similar_pairs(
      sourmash_compare_signatures.out[1],
      jaccard_threshold
    )

  /*  CLASSIFICATION  */
    sourmash_classify_signatures(
      sourmash_compute_signatures.out,
      database
    )
  /*  SPECIES FILTER  */
    python_species_filter(
      sourmash_classify_signatures.out,
    )

}