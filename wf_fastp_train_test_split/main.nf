nextflow.preview.dsl=2

include {
  create_folders;
  fastp_qc;
  get_train_test_csv;
  get_train_test_channels;
} from "$baseDir/modules/redux_processes.nf"

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
/*
log.info """
This workflow already run through once. It should not be started again,
without beeing aware of the consequences. This workflow would split up
samples in a train and a test set, altough the split was already done.
This would result in a bigger test set and a wrong train test split,
because some samples would be in each set. So if you run this workflow
again, specify a new path_to_file_system, delete the old outputs or live
with the consequences.
Bye.
  """
*/

    // dependencies
    genome_size           = params.genome_size
    e_value_cutoff        = params.e_value_cutoff
    db_mmseqs2            = params.db_mmseqs2
    pfam_dat              = params.pfam_dat
    max_cores_per_process = params.max_cores_per_process
    path_to_file_system   = params.path_to_file_system
    test_size             = params.test_size

    // input data
    ch_reads    = Channel.fromFilePairs(params.data, checkIfExists: true)
    ch_samples  = Channel.fromPath(params.data)
  
    /*
    // create folders for outputs in specefied path
    create_folders(
      path_to_file_system
    )

    // quality control of all samples using fastp, includes adapter and quality trimming
    fastp_qc(
      ch_reads,
      max_cores_per_process
    )

    get_train_test_csv(
      ch_samples.collect(),
      test_size
    )
    */

    ch_reads  = Channel.fromFilePairs("/scr/k61san/johann/redux/workflow/reads/reads_all/*_{1,2}_fastp.fastq.gz", checkIfExists: true)
    train_csv = "/scr/k61san/johann/redux/workflow/metadata/train.csv"
    test_csv  = "/scr/k61san/johann/redux/workflow/metadata/test.csv"

    get_train_test_channels(
        ch_reads,  //fastp_qc.out,
        train_csv, //get_train_test_csv.out[0],  //train.csv
        test_csv   //get_train_test_csv.out[1] //test.csv
    )

}