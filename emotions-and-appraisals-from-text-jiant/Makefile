.PHONY: prepare train

prepare:
	python create_task_configs.py

train: jiant
	# for example
	python crowdenvent_single.py emo_cls
	python crowdenvent_single.py suddenness_cls
	python crowdenvent_single.py suddenness_reg

jiant:
	git clone git@github.com:nyu-mll/jiant.git
	cd jiant; git checkout bc7a7840ae42aa6addce1913a57f0703205e090b ; pip install -e . ; cd ..
	cp jiant-scripts/crowdenvent.py jiant/jiant/tasks/lib/crowdenvent.py
	cp jiant-scripts/retrieval.py jiant/jiant/tasks/retrieval.py
	cp jiant-scripts/core.py jiant/jiant/tasks/evaluate/core.py
