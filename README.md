This repo is meant to be an attempt to implement SC2 reinforcement learning agents using the algorithms developed in OpenAI's baselines repo. 

Note: I am currently working off a forked version of baselines so I can inject some additional tweaks to the algorithms, but this should work just fine using the regular baselines package.


In order to install the baselines fork on Windows you have to do the following:
1) Clone my fork (or the base version) and navigate to the directory you cloned it into.
2) Install this forked version of OpenAI's atari-py (atari py doesn't support windows currently): 
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
``
3) Download and run both MPI installers from the following link: https://www.microsoft.com/en-us/download/details.aspx?id=56727
4) Install baselines using the following command:
```
pip install -e .
```