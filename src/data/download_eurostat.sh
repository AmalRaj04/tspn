mkdir -p data/raw/eurostat_ppi

for geo in AT BE BG CY CZ DE DK ES EE FI FR EL HR HU IE IT LT LU LV MT NL PL PT RO SK SI SE; do
  for nace in B C D; do
    echo "Downloading ${geo}_${nace}"

    curl -s "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m?format=JSON&geo=${geo}&nace_r2=${nace}&s_adj=NSA&unit=I15&indic_bt=PRC_PRR_DOM" \
      -o "data/raw/eurostat_ppi/${geo}_${nace}.json"
  done
done
