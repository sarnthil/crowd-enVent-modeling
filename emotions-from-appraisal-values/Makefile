.phony: clean all train eval

TRAINDATA = workdata/train+val.tsv
TESTDATA = sources/crowd-enVent_validation_deduplicated.tsv
MODEL = workdata/simple.joblib


all: train eval

clean:
	rm -rf workdata/*

workdata/train+val.tsv:
	cat sources/crowd-enVent-train.tsv sources/crowd-enVent-val.tsv >workdata/train+val.tsv


train:
	for seed in 1234 2142 42 23 512 ; do \
		python scripts/cli.py -v train $(TRAINDATA) --output workdata/simple-$$seed.joblib --seed $$seed ; \
	done
	for seed in 1234 2142 42 23 512 ; do \
		python scripts/cli.py -v train $(TRAINDATA) --bin --output workdata/binned-$$seed.joblib --seed $$seed ; \
	done
	for seed in 1234 2142 42 23 512 ; do \
		python scripts/cli.py -v train $(TRAINDATA) --scale --output workdata/scaled-$$seed.joblib --seed $$seed ; \
	done

eval:
	for seed in 1234 2142 42 23 512 ; do \
		echo seed $$seed ; \
		python scripts/cli.py -v evaluate sources/crowd-enVent_validation_deduplicated.tsv --model workdata/scaled-$$seed.joblib ; \
		python scripts/cli.py -v evaluate sources/crowd-enVent_validation_deduplicated.tsv --model workdata/binned-$$seed.joblib ; \
		python scripts/cli.py -v evaluate sources/crowd-enVent_predicted_gold.tsv --model workdata/scaled-$$seed.joblib ; \
		python scripts/cli.py -v evaluate sources/crowd-enVent_predicted_gold.tsv --model workdata/binned-$$seed.joblib ; \
	done
