$ pip list --not-required --format=freeze > pip_freeze_not_required_requirements.txt
$ pip freeze > pip_freeze.txt
$ conda env export --from-history -f environment_from_history.yml
$ conda env export -f environment.yml
