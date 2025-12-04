mkdir -p Resampled

for f in /home/sandipandhar/Desktop/DAC-Code/*.wav; do
    fname=$(basename "$f")
    ffmpeg -i "$f" -ar 24000 "Resampled/$fname"
done
