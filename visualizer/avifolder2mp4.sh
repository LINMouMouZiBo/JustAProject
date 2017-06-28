do
        for file in "$folder"/*.avi
        do
#		echo "${file[@]%.avi}".mp4
                avconv -i "$file" -c:v libx264 -c:a copy "${file[@]%.avi}".mp4
        done
done
