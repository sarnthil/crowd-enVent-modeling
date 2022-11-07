.PHONY: download split deduplicate all

all: download deduplicate split

sources/crowd-enVent_validation.tsv sources/crowd-enVent_generation.tsv:
	mkdir sources
	mkdir tmp
	curl -L https://www.romanklinger.de/data-sets/crowd-enVent2022.zip >tmp/crowd-enVent2020.zip
	cd tmp; unzip crowd-enVent2020.zip
	cp tmp/corpus/crowd-enVent_validation.tsv sources/
	cp tmp/corpus/crowd-enVent_generation.tsv sources/
	rm -r tmp

download: sources/crowd-enVent_validation.tsv sources/crowd-enVent_generation.tsv

sources/crowd-enVent-train.tsv sources/crowd-enVent-test.tsv sources/crowd-enVent-val.tsv:
	python scripts/create_splits.py

split: sources/crowd-enVent-train.tsv sources/crowd-enVent-test.tsv sources/crowd-enVent-val.tsv

deduplicate: sources/crowd-enVent_validation_deduplicated.tsv

sources/crowd-enVent_validation_deduplicated.tsv: sources/crowd-enVent_validation.tsv sources/crowd-enVent_generation.tsv
	python scripts/deduplicate-validation-data.py --validation sources/crowd-enVent_validation.tsv --generation sources/crowd-enVent_generation.tsv --output sources/crowd-enVent_validation_deduplicated.tsv
