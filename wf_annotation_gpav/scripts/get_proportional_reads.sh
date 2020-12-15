
fastq=$1
proportion=$2
genome_size=$3
id=$4
path_to_original=$5


length=$(zcat "$path_to_original"/"$id"_1.fastq.gz | head -2 | tail -1 | awk '{print length($0)}')
number_of_reads=$(echo "("$proportion"*"$genome_size")/(2*"$length")" | bc)
seqtk sample -s100 "$fastq" "$number_of_reads" | gzip > "$proportion"_"$fastq"