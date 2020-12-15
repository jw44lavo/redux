path_to_file_system = params.path_to_file_system

process shovill_assemble {
    publishDir "$path_to_file_system/assemblies", pattern: "*.fasta", mode: "copy"

  conda "$baseDir/environments/redux.yml"

  input:
    tuple val(id), file(reads)
    val genome_size
    val max_cores_per_process

  output:
    path "${id}.fasta", type: "file"
  
  script:
    """
    shovill --R1 ${reads[0]} --R2 ${reads[1]} --gsize $genome_size --outdir ${id} --cpus $max_cores_per_process --trim
    mv ${id}/contigs.fa ${id}.fasta
    """
}