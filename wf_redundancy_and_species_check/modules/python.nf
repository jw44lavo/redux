path_to_file_system = params.path_to_file_system

process python_species_filter{
  input:
    file classification

  script:
    """
    python3 $baseDir/scripts/species_filter.py $classification
    """
}

process python_return_similar_pairs {
  publishDir "$path_to_file_system/results", mode: "copy"

  input:
    file csv
    val threshold

  output:
    path "similar_pairs.tsv", type: "file"

  script:
    """
    python3 $baseDir/scripts/return_similar_pairs.py $csv $threshold
    """
}