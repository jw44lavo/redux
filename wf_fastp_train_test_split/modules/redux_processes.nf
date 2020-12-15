path_to_file_system = params.path_to_file_system

process create_folders {
  input:
    val wf_path

  script:
    """
    mkdir $wf_path/gpav -p
    mkdir $wf_path/annotations -p
    mkdir $wf_path/coverages -p
    mkdir $wf_path/uniq_counts -p
    mkdir $wf_path/reads -p
    mkdir $wf_path/reads/reads_all -p
    mkdir $wf_path/reads/reads_train -p
    mkdir $wf_path/reads/reads_test -p
    mkdir $wf_path/assemblies -p
    mkdir $wf_path/mlearning -p
    mkdir $wf_path/recommender -p
    mkdir $wf_path/metadata -p
    mkdir $wf_path/results -p
    """
}

process fastp_qc {
  publishDir "$path_to_file_system/reads/reads_all", mode: "move", pattern: "*.gz"
  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id), file(read)
    val max_cores_per_process

  output:
    tuple val(id), file("${id}*_fastp.fastq.gz")

  script:
    """
    fastp -i ${read[0]} -I ${read[1]} -o ${read[0].simpleName}_fastp.fastq.gz -O ${read[1].simpleName}_fastp.fastq.gz -w $max_cores_per_process
    """
}

process get_train_test_csv {
  publishDir "$path_to_file_system/metadata", mode: "copy"
  conda "$baseDir/environments/redux.yml"
  
  input:
    file samples
    val test_size

  output:
    path "train.csv", type: "file"
    path "test.csv", type: "file"

  script:
    """
    python $baseDir/scripts/get_train_test_csv.py --input $samples --test_proportion $test_size
    """
}

process get_train_test_channels {
  publishDir "$path_to_file_system/reads/reads_train", mode: "move", pattern: "*_train.fastq.gz"
  publishDir "$path_to_file_system/reads/reads_test", mode: "move", pattern: "*_test.fastq.gz"
  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id), file(reads)
    val train_samples
    val test_samples

  output:
    tuple val(id), file("${id}*.fastq.gz")

  script:
    """
    if grep -q $id $test_samples; then
        bash $baseDir/scripts/symlink_original_file.sh ${reads[0]} ${reads[0].simpleName}_test.fastq.gz
        bash $baseDir/scripts/symlink_original_file.sh ${reads[1]} ${reads[1].simpleName}_test.fastq.gz

    else
        bash $baseDir/scripts/symlink_original_file.sh ${reads[0]} ${reads[0].simpleName}_train.fastq.gz
        bash $baseDir/scripts/symlink_original_file.sh ${reads[1]} ${reads[1].simpleName}_train.fastq.gz
    fi
    """
}