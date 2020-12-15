id=$1
path=$2
g_size=$3

length=$(zcat "$path"/"$id"_1.fastq.gz | head -2 | tail -1 | awk '{print length($0)}')
number=$(zcat "$path"/"$id"_1.fastq.gz | wc -l)
number=$(echo "$number"/4 | bc)
number=$(echo "("$number"*"$length"*2)/("$g_size")" | bc)
echo "$number"

