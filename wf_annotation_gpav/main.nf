nextflow.preview.dsl=2

include {
  get_coverages;
  get_coverages_csv;
  ggplot_coverages;
  seqtk_subsample;
  shovill_assemble_reads;
  prodigal_translate_contigs;
  plass_assemble_paired_end_reads;
  plass_merge_reads;
  mmseqs2_search as mmseqs2_search_plass;
  mmseqs2_search as mmseqs2_search_shovill;
  mmseqs2_search_directly;
  mmseqs2_convert_direct_output;
  create_result_file;
  get_number_of_annotations_over_coverage_csv;
  ggplot_number_of_annotations_over_coverage;
  get_gpa;
  machine_learning_classifier
} from "$baseDir/modules/redux_processes.nf"


workflow subwf_train {
  take:
    ch_reads
    db_mmseqs2
    e_value_cutoff
    max_cores_per_process
    pfam_dat

  main:
    plass_merge_reads(
        ch_reads,
        max_cores_per_process
    )

    mmseqs2_search_directly(
        plass_merge_reads.out,
        db_mmseqs2,
        e_value_cutoff,
        max_cores_per_process
    )

    mmseqs2_convert_direct_output(
        mmseqs2_search_directly.out[0],
        mmseqs2_search_directly.out[1],
        db_mmseqs2,
        max_cores_per_process
    )

    get_gpa(
      mmseqs2_convert_direct_output.out,
      pfam_dat
    )
  emit:
    get_gpa.out
}

workflow subwf_test {
  take:
    ch_reads
    db_mmseqs2
    e_value_cutoff
    max_cores_per_process
    ch_proportions
    genome_size
    pfam_dat
    path_to_original_reads

  main:
    seqtk_subsample(
      ch_reads,
      ch_proportions,
      genome_size,
      path_to_original_reads
    )

    // mmseqs shovill prodigal
    // begin
    shovill_assemble_reads(
      ch_reads.concat(seqtk_subsample.out),
      genome_size,
      max_cores_per_process
    )

    prodigal_translate_contigs(
      shovill_assemble_reads.out
    )

    mmseqs2_search_shovill(
      prodigal_translate_contigs.out,
      db_mmseqs2,
      e_value_cutoff,
      max_cores_per_process
    )
    // end

    // mmseqs direct
    // begin
    plass_merge_reads(
        ch_reads.concat(seqtk_subsample.out),
        max_cores_per_process
    )

    mmseqs2_search_directly(
      plass_merge_reads.out,
      db_mmseqs2,
      e_value_cutoff,
      max_cores_per_process
    )

    mmseqs2_convert_direct_output(
      mmseqs2_search_directly.out[0],
      mmseqs2_search_directly.out[1],
      db_mmseqs2,
      max_cores_per_process
    )
    // end
    
    // mmseqs plass
    // begin
    plass_assemble_paired_end_reads(
        ch_reads.concat(seqtk_subsample.out),
        max_cores_per_process
    )

    mmseqs2_search_plass(
        plass_assemble_paired_end_reads.out,
        db_mmseqs2,
        e_value_cutoff,
        max_cores_per_process
    )
    //end

    create_result_file(
        mmseqs2_search_shovill.out.concat(mmseqs2_search_plass.out, mmseqs2_convert_direct_output.out)
    )

    get_number_of_annotations_over_coverage_csv(
      create_result_file.out.collect()
    )

    /*
    ggplot_number_of_annotations_over_coverage(
      get_number_of_annotations_over_coverage_csv.out
    )
    */

    get_gpa(
      mmseqs2_convert_direct_output.out,
      pfam_dat
    )

  emit:
   get_gpa.out
}

workflow subwf_coverage_examination {
  take:
      ch_train
      ch_test
      path_to_original_reads
      genome_size

  main:
    get_coverages(
      ch_train.concat(ch_test),
      path_to_original_reads,
      genome_size
    )

    get_coverages_csv(
      get_coverages.out.collect()
    )

    ggplot_coverages(
      get_coverages_csv.out
    )
}

/************************************************************
* WORKFLOW  *************************************************
************************************************************/
workflow {
log.info ""
log.info ""
log.info """\
R E D U X - W F
=========================
Start date:             ${params.date}

"""
log.info ""
log.info ""

  main:
    // dependencies
    genome_size            = params.genome_size
    e_value_cutoff         = params.e_value_cutoff
    db_mmseqs2             = params.db_mmseqs2
    pfam_dat               = params.pfam_dat
    max_cores_per_process  = params.max_cores_per_process
    path_to_file_system    = params.path_to_file_system
    ch_proportions         = Channel.fromList(params.proportions)
    path_to_original_reads = params.path_to_original_reads

    // input data
    ch_train = Channel.fromFilePairs(params.data_train, checkIfExists: true)
    ch_test  = Channel.fromFilePairs(params.data_test, checkIfExists: true)

    subwf_coverage_examination(
      ch_train,
      ch_test,
      path_to_original_reads,
      genome_size
    )

    subwf_train(
      ch_train,
      db_mmseqs2,
      e_value_cutoff,
      max_cores_per_process,
      pfam_dat
    )

    subwf_test(
      ch_test,
      db_mmseqs2,
      e_value_cutoff,
      max_cores_per_process,
      ch_proportions,
      genome_size,
      pfam_dat,
      path_to_original_reads
    )
}