folder1=$1
folder2=$2


# Get the number of files in each folder
count_folder1=$(find "$folder1" -type f ! -name "*.txt" | wc -l)
count_folder2=$(find "$folder2" -type f ! -name "*.txt"  | wc -l)

# Check if the counts are the same
if [ "$count_folder1" -ne "$count_folder2" ]; then
    echo "Error: The number of files in $folder1 ($count_folder1) is not the same as in $folder2 ($count_folder2)"
    
    export "$3=1"
else
    echo "Number of files in $folder1 and $folder2 is the same: $count_folder1"
    export "$3=0"
fi

echo "All good!"