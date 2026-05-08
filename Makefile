# Makefile

.PHONY: setup destroy

setup:
	bash infra/setup.sh

destroy:
	bash infra/teardown.sh