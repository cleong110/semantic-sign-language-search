# basically this is me trying to find the exact requirements for signCLIP to work. It requires Python 3.8.8 I believe

$ pip list --not-required --format=freeze > pip_freeze_not_required_requirements.txt
$ pip freeze > pip_freeze.txt
$ conda env export --from-history -f environment_from_history.yml
$ conda env export -f environment.yml
