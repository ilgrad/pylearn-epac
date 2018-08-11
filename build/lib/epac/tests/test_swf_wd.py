# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:07:47 2013

@author: jinpeng.li@cea.fr
"""

import tempfile
import os
import sys
import socket
import os.path
from distutils.version import LooseVersion as V
import dill
import joblib
from epac.errors import NoSomaWFError

# Import dill if installed and recent enough, otherwise falls back to pickle
from distutils.version import LooseVersion as V
try:
    errmsg = "Falling back to pickle. "\
             "There may be problem when running soma-workflow on cluster "\
             "using EPAC\n"
    import dill as pickle
    if V(pickle.__version__) < V("0.2a"):
        sys.stderr.write("warning: dill version is too old to use. " + errmsg)
except ImportError:
    import pickle
    sys.stderr.write("warning: Cannot import dill. " + errmsg)
# Try to import soma_workflow and raise error if impossible
try:
    from soma_workflow.client import Job, Workflow
    from soma_workflow.client import Helper, FileTransfer
    from soma_workflow.client import WorkflowController
    import soma_workflow.constants as constants
except ImportError:
    errmsg = "No soma-workflow is found. "\
        "Please verify your soma-worklow"\
        "on your computer (e.g. PYTHONPATH) \n"
    sys.stderr.write(errmsg)
    sys.stdout.write(errmsg)
    raise NoSomaWFError


if __name__ == '__main__':

    if V(dill.__version__) < V("0.2a"):
        raise ValueError("dill version is too old to use, please use version "
                         "on https://github.com/uqfoundation/dill")
    if V(joblib.__version__) < V("0.7.1"):
        raise ValueError("joblib version is too old to use, please use version"
                         " on https://github.com/joblib/joblib")

    tmp_work_dir_path = tempfile.mkdtemp()
    cur_work_dir = os.getcwd()
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    test_filepath = u"./onlytest.txt"
    test_bash_script = u"./testbs.sh"
    os.chdir(tmp_work_dir_path)
    fileout = open(test_bash_script, "w+")
    filecontent = """#!/bin/bash
echo %s
""" % test_bash_script
    fileout.write(filecontent)
    fileout.close()
    os.chdir(cur_work_dir)

    job1 = Job(command=[u"touch", test_filepath],
               name="epac_job_test",
               working_directory=tmp_work_dir_path)
    job2 = Job(command=["%s/readfile" % cur_file_dir, test_bash_script],
               name="epac_job_test",
               working_directory=tmp_work_dir_path)

    soma_workflow = Workflow(jobs=[job1, job2])

    resource_id = socket.gethostname()
    controller = WorkflowController(resource_id, "", "")
    ## run soma-workflow
    ## =================
    wf_id = controller.submit_workflow(workflow=soma_workflow,
                                       name="epac workflow")
    Helper.wait_workflow(wf_id, controller)
    nb_failed_jobs = len(Helper.list_failed_jobs(wf_id, controller))
    if nb_failed_jobs > 0:
        raise ValueError("Soma-workflow error, cannot use working directory")

    if not os.path.isfile(os.path.join(tmp_work_dir_path, test_filepath)):
        raise ValueError("Soma-workflow cannot define working directory")
    else:
        print("OK for creating new file in working directory")
