path_to_file_system = params.path_to_file_system

process sourmash_compute_signatures {
  conda "$baseDir/environments/redux.yml"

  input:
    file fasta

  output:
    file "${fasta.baseName}.sig"
  
  script:
    """
    sourmash compute -k 31 --scaled=1000 $fasta -o ${fasta.baseName}.sig
    """
}

process sourmash_classify_signatures {
  publishDir "$path_to_file_system/signatures", mode: "copy"

  conda "$baseDir/environments/redux.yml"

  input:
    file signature
    val db

  output:
    path "${signature.baseName}_classification.out", type: "file"

  script:
  """
  sourmash lca classify --db $db --query $signature -o ${signature.baseName}_classification.out
  """
}

process sourmash_compare_signatures {
  publishDir "$path_to_file_system/results", mode: "copy"

  conda "$baseDir/environments/redux.yml"

  label 'big_cpu'

  input:
    file assemblies
    val max_cores_per_process

  output:
    path "*cmp*", type: "file"
    path "matrix.csv", type: "file"

  script:
    """
    sourmash compare -p $max_cores_per_process $assemblies -o cmp --csv matrix.csv
    """
}

process sourmash_plot_comparison {
  publishDir "$path_to_file_system/results", mode: "copy"

  conda "$baseDir/environments/redux.yml"

  input:
    file cmp

  output:
    path "*", type: "file"

  script:
    """
    sourmash plot ${cmp[0]} --labels
    """
}