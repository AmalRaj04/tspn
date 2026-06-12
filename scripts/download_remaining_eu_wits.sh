#!/bin/bash

mkdir -p data/raw/wits/json

download_country () {
    ISO=$1
    CODE=$2

    echo "======================================="
    echo "$ISO ($CODE)"
    echo "======================================="

    LOWER=$(echo "$ISO" | tr '[:upper:]' '[:lower:]')

    for YEAR in 2018 2019 2020 2021
    do
        OUTFILE="data/raw/wits/json/${LOWER}_${YEAR}.json"

        if [ -f "$OUTFILE" ]; then
            echo "Skipping $ISO $YEAR"
            continue
        fi

        echo "Downloading $ISO $YEAR"

        curl -L \
        "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/TRN/reporter/${CODE}/partner/000/product/ALL/year/${YEAR}/datatype/reported?format=JSON" \
        -o "$OUTFILE"

        sleep 2
    done
}

download_country AUT 040
download_country BGR 100
download_country CYP 196
download_country CZE 203
download_country DNK 208
download_country EST 233
download_country FIN 246
download_country GRC 300
download_country HRV 191
download_country HUN 348
download_country IRL 372
download_country LTU 440
download_country LUX 442
download_country LVA 428
download_country MLT 470
download_country POL 616
download_country PRT 620
download_country ROU 642
download_country SVK 703
download_country SVN 705
download_country SWE 752

echo ""
echo "Finished downloading remaining EU WITS files."