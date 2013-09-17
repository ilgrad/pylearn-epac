# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:07:47 2013

@author: jinpeng.li@cea.fr
"""

import tempfile
import os
from soma_workflow.client import Job, Workflow
from soma_workflow.client import Helper, FileTransfer
from soma_workflow.client import WorkflowController
import socket
import os.path


if __name__ == '__main__':
    tmp_work_dir_path = tempfile.mkdtemp()
    cur_work_dir = os.getcwd()
    test_filepath = u"./onlytest.txt"

    job = Job(command=[u"touch", test_filepath],
                    name="epac_job_test",
                    working_directory=tmp_work_dir_path)
    soma_workflow = Workflow(jobs=[job])

    resource_id = socket.gethostname()
    controller = WorkflowController(resource_id, "", "")
    ## run soma-workflow
    ## =================
    wf_id = controller.submit_workflow(workflow=soma_workflow,
                                       name="epac workflow")
    Helper.wait_workflow(wf_id, controller)
    if not os.path.isfile(os.path.join(tmp_work_dir_path, test_filepath)):
        raise ValueError("Soma-workflow cannot define working directory")
    else:
        print "OK"
