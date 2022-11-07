.PHONY: all

all: TAE_human_cls.tsv TA_human_reg.tsv TE_human_cls.tsv TA_human_cls.tsv

TAE_human_cls.tsv: model_validators.py
	python3 scripts/model_validators.py --dump-raw outputs/scores/TAE_human_cls.tsv

TA_human_reg.tsv: model_validators.py
	python3 scripts/model_validators.py --metric spearman --dump-raw outputs/scores/TA_human_reg.tsv

TE_human_cls.tsv: TAE_human_cls.tsv
	cut -d '	' -f 1-3 <outputs/scores/TAE_human_cls.tsv >outputs/scores/TE_human_cls.tsv

TA_human_cls.tsv: TAE_human_cls.tsv
	cut -d '	' -f 1,4-99 <outputs/scores/TAE_human_cls.tsv >outputs/scores/TA_human_cls.tsv
