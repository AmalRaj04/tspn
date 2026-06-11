#!/bin/bash

mkdir -p data/raw/wits/json

while read ISO CODE
do
    for YEAR in 2015 2016 2017 2018 2019 2020 2021
    do
        OUT="data/raw/wits/json/${ISO}_${YEAR}.json"

        if [ -s "$OUT" ]; then
            echo "Skipping ${ISO} ${YEAR}"
            continue
        fi

        echo "Downloading ${ISO} ${YEAR}"

        curl -L \
        "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/TRN/reporter/${CODE}/partner/000/product/ALL/year/${YEAR}/datatype/reported?format=JSON" \
        -o "$OUT"

        sleep 1
    done

done << EOF
aus 36
aut 40
bgr 100
bra 76
che 756
cyp 196
cze 203
dnk 208
est 233
fin 246
grc 300
hrv 191
hun 348
idn 360
ind 356
irl 372
jpn 392
kor 410
ltu 440
lux 442
lva 428
mex 484
mlt 470
pol 616
prt 620
rou 642
rus 643
svk 703
svn 705
swe 752
tur 792
twn 158
EOF