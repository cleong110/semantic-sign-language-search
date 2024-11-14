Scripts, files, etc to setup [SignCLIP](https://github.com/J22Melody/fairseq/tree/main/examples/MMPT) locally. 

* [setup_signclip.sh](setup_signclip.sh) attempts to clone SignCLIP and install requirements and put files in the right places. You need to have model files and .yaml files in the expected paths or it won't work, and getting the right combination of requirements is tricky (See [env_files/README.md](env_files/README.md) for an enviroment I actually got to work eventually)
* [demo_sign_embed_and_save.py](demo_sign_embed_and_save.py) will embed every .pose file in a folder with every model you give it. It assumes the .yaml files are in the expected places, the models are in the right folders, and the requirements are installed in the current environment.
