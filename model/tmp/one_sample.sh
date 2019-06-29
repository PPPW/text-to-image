for c in $(cat classes.txt); do
    c=$(echo "$c" | sed -e 's/[]\/$*.^|[]/\\&/g')
    grep "$c" txt.txt | head -1
done
