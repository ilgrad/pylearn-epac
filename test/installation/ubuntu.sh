#! bin/sh

FILENAME=epac
BRANCHNAME=documents

git clone git@github.com:neurospin/pylearn-epac.git $FILENAME
cd $FILENAME
git fetch origin
git checkout -b $BRANCHNAME origin/$BRANCHNAME

# Installation script:
sudo apt-get install python-pip

sudo apt-get install python-setuptools python-dev build-essential libatlas-dev python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

sudo pip install scikit-learn
sudo pip install soma-workflow

sudo python setup.py install

