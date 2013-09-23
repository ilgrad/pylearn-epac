.. _installation:


Introduction
------------

EPAC depends on scikit-learn, dill, joblib, and soma-workflow (optionally run on DRM system and torque/pbs has been tested). The figure below shows EPAC's dependencies.
EPAC has been tested on python 2.7 so that we recommand that run EPAC on python 2.7
or its latest version, but not with python 3.0.
In this section, we will present how to install EPAC on ubuntu and manually on the other platforms.

.. img_dependencies:: ./images/dependencies.png

.. figure:: ./images/dependencies.png
   :scale: 50 %
   :align: center
   :alt: EPAC's dependencies


Ubuntu
------

First of all, you need to install some softwares for EPAC:


.. code-block:: guess

    sudo apt-get install python-pip
    sudo apt-get install python-setuptools python-dev build-essential libatlas-dev python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    sudo pip install scikit-learn
    sudo pip install soma-workflow


Download and install **dill** from https://github.com/uqfoundation/dill. EPAC needs the latest code of dill.

.. code-block:: guess
   
    git clone https://github.com/uqfoundation/dill
    cd dill
    python setup.py build
    sudo python setup.py install
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
Download and install **joblib** from https://github.com/joblib/joblib. EPAC needs the latest code of joblib.

Finally, EPAC can be downloaded from github and you can run installation script for your local user.

.. code-block:: guess

    git clone https://github.com/neurospin/pylearn-epac.git
    cd pylearn-epac
    sudo python setup.py install


Other platforms
---------------

On the other platforms which support python, you can manually install EPAC according to your system configuration.

**scikit-learn**: EPAC depends on scikit-learn which is a manchine learning libary. To use EPAC, scikit-learn should be installed on your computer. Please goto http://scikit-learn.org/ to install scikit-learn.

**soma-workflow** (optionally): you can install soma-workflow so that EPAC can run on the DRM system (torque/pbs). To install soma-workflow, please goto http://brainvisa.info/soma/soma-workflow for documentation, and https://pypi.python.org/pypi/soma-workflow for installation.

**dill**: Download and install **dill** from https://github.com/uqfoundation/dill. EPAC needs the latest code of dill.

**joblib**: download and install **joblib** from https://github.com/joblib/joblib.

**EPAC**: download EPAC from github to ``$EPACDIR`` and set enviroment variable ``$PYTHONPATH`` that contains ``$EPACDIR`` (EPAC directory), and ``$PATH`` contains $EPACDIR/bin

.. code-block:: guess

    EPACDIR=epac
    git clone https://github.com/neurospin/pylearn-epac.git $EPACDIR
    export PYTHONPATH=$EPACDIR:$PYTHONPATH
    export PATH=$EPACDIR/bin:$PATH


Now, you can start to use EPAC for machine learning.

