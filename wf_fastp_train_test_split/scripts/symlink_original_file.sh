o_path=$(readlink -e "$1")

ln -s "$o_path" "$2"
