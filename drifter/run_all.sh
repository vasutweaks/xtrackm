#/bin/bash
for sat in "GFO" "TP+J1+J2+J3+S6A" "S3A" "S3B" "TPN+J1N+J2N+J3N" "ERS1+ERS2+ENV+SRL" "HY2A" "HY2B"
do
echo $sat
python3 p.scatter.track_guVSdrifter_u.py $sat
done
