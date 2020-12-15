path_to_file_system = params.path_to_file_system

process get_coverages {
  /*
  same problem as described at "seqtk_subsample"
  */
  publishDir "$path_to_file_system/coverages", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  
  input:
    tuple val(id), file(reads)
    val path_to_original  // explained in process description
    val genome_size

  output:
    path "${id}_sample_coverage.out", type: "file"
  script:
  """
  bash $baseDir/scripts/get_coverage.sh $id $path_to_original $genome_size > ${id}_sample_coverage.out
  """
}

process get_coverages_csv {
  publishDir "$path_to_file_system/results", mode: "copy"
  conda "$baseDir/environments/redux.yml"

  input:
    file coverage_files

  output:
    path "coverages.csv", type: "file"

  script:
    """
    python $baseDir/scripts/get_coverages_dataframe.py $coverage_files
    """
}

process ggplot_coverages {
  publishDir "$path_to_file_system/results", mode: "copy"
  conda "$baseDir/environments/redux.yml"

  input:
    file coverages_csv

  output:
    path "*.png", type: "file"

  script:
    """
    Rscript $baseDir/scripts/plot_coverages.R $coverages_csv
    """
}

process seqtk_subsample {
  /*
  The specified script takes the length of the first read in the original read file to calculate
  the number of needed reads to match a certain coverage. If the length of the first read in the 
  preprocessed read file was taken, the calculation of the number of needed reads would be randomly
  wrong. For example, the first read could be 48 bases long, but all others 100. So there would be
  much more information than wanted. By taking the length of the first read in the original read
  file, the amount of information is always underestimated, because preprocessed reads are always
  less or equal in length. So you always get less information than wanted. This is much easier to
  interpret in downstream analyses.
  You could use more complex algorithms to subsample at a specific coverage. This method was choosen
  cause of simplicity and velocity.
  The number of reads to subsample is calculated by the following equation:
    #reads = (coverage * genome_size)/(2 * read_length)"
  */

  publishDir "$path_to_file_system/reads/reads_test", mode: "copy", pattern: "*_test.fastq.gz"
  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id), file(reads)
    each proportion // subsample at this coverage
    val genome_size
    val path_to_original  // explained in process description

  output:
    tuple val("${proportion}_${id}"), file("${proportion}_${id}*.fastq.gz")
  
  script:
    """
    bash $baseDir/scripts/get_proportional_reads.sh ${reads[0]} $proportion $genome_size $id $path_to_original
    bash $baseDir/scripts/get_proportional_reads.sh ${reads[1]} $proportion $genome_size $id $path_to_original
    """
}

process plass_assemble_paired_end_reads{
  publishDir "$path_to_file_system/assemblies", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  errorStrategy "ignore"

  input:
    tuple val(id), file(reads)
    val max_cores_per_process

  output:
    path "${id}_plass.fas", type: "file"
  
  script:
    """
    plass assemble ${reads[0]} ${reads[1]} ${id}_plass.fas tmp --threads $max_cores_per_process
    """
}

process plass_merge_reads {
  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id), file(reads)
    val max_cores_per_process

  output:
    tuple val(id), file("${id}*")

  script:
    """
    plass mergereads ${reads[0]} ${reads[1]} ${id} --threads $max_cores_per_process
    """
}

process mmseqs2_search{
  publishDir "$path_to_file_system/annotations", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  errorStrategy 'ignore'

  input:
    file assembly
    val db
    each e_value
    val max_cores_per_process

  output:
    path "${assembly.baseName}_mmseqs2_${e_value}.tsv", type: "file"
  
  script:
    """
    mmseqs easy-search $assembly $db ${assembly.baseName}_mmseqs2_${e_value}.tsv tmp -e $e_value --threads $max_cores_per_process --format-output "target,bits,evalue" --remove-tmp-files
    """
}

process mmseqs2_search_directly{
  conda "$baseDir/environments/redux.yml"
  errorStrategy 'ignore'

  input:
    tuple val(id), file(reads)
    val db
    each e_value
    val max_cores_per_process

  output:
    tuple val(id), file(reads)
    tuple val("${id}_direct_mmseqs2_${e_value}"), file("${id}_direct_mmseqs2_${e_value}*")

  script:
    """
    mmseqs search $id $db ${id}_direct_mmseqs2_${e_value} tmp -e $e_value --threads $max_cores_per_process --remove-tmp-files
    """
}

process mmseqs2_convert_direct_output {
  publishDir "$path_to_file_system/annotations", mode: "copy"
  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id_reads), file(reads)
    tuple val(id_annotations), file(annotations)
    val db
    val max_cores_per_process

  output:
    path "${id_annotations}.tsv", type: "file"
  
  script:
    """
    mmseqs convertalis $id_reads $db ${id_annotations} ${id_annotations}.tsv --format-output "target,bits,evalue" --threads $max_cores_per_process
    """
}

process create_result_file {
  publishDir "$path_to_file_system/uniq_counts", mode: "copy"
  conda "$baseDir/environments/redux.yml"

  input:
    file annotation

  output:
    path "${annotation.baseName}_uniq_count.out", type: "file"

  script:
    """
    echo \$(cut -f1 ${annotation} | sort | uniq | wc -l) > ${annotation.baseName}_uniq_count.out
    """
}

process get_number_of_annotations_over_coverage_csv {
  publishDir "$path_to_file_system/results", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  
  input:
    file count_files

  output:
    path "annotation_count.csv", type: "file"

  script:
    """
    python $baseDir/scripts/get_sequential_annotation_dataframe.py $count_files
    """
}

process ggplot_number_of_annotations_over_coverage {
  publishDir "$path_to_file_system/results", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  
  input:
    file csv_file

  output:
    path "*.png", type: "file"

  script:
    """
    Rscript $baseDir/scripts/plot_annotations_over_coverage.R $csv_file
    """
}

process get_gpa {
  publishDir "$path_to_file_system/gpav", mode: "copy"
  conda "$baseDir/environments/redux.yml"

  input:
    file data 
    val dat
  
  output:
    path "${data.baseName}_gpa.tsv", type: "file"

  script:
    """
    python $baseDir/scripts/get_gpav_from_annotation_tsv.py --input $data --pfam $dat --output ${data.baseName}_gpa.tsv
    """
}

process shovill_assemble_reads {
  publishDir "$path_to_file_system/assemblies", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  errorStrategy "ignore"

  input:
    tuple val(id), file(reads)
    val genome_size
    val max_cores_per_process

  output:
    path "${id}_shovill.fasta", type: "file"
  
  script:
    """
    shovill --R1 ${reads[0]} --R2 ${reads[1]} --gsize $genome_size --outdir ${id} --cpus $max_cores_per_process
    mv ${id}/contigs.fa ${id}_shovill.fasta
    """
}

process prodigal_translate_contigs {
  conda "$baseDir/environments/redux.yml"
  errorStrategy "ignore"

  input:
    file assembly

  output:
    path "${assembly.baseName}_prodigal.fasta", type: "file"
  
  script:
    """
    prodigal -i ${assembly} -a ${assembly.baseName}_prodigal.fasta
    """
}