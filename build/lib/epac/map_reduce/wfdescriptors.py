# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:42:47 2013

@author: jinpeng.li@cea.fr
"""

import os
import sys
from epac import StoreFs
from epac.map_reduce.exports import save_job_list
from epac.map_reduce.split_input import SplitNodesInput
from epac.map_reduce.inputs import NodesInput
from epac.utils import save_dataset_path
from epac.stores import save_tree
from epac.stores import load_tree
from epac.errors import NoSomaWFError


def export_jobs(tree_root, num_processes, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    node_input = NodesInput(tree_root.get_key())
    split_node_input = SplitNodesInput(tree_root,
                                       num_processes=num_processes)
    nodesinput_list = split_node_input.split(node_input)
    return save_job_list(dir_path, nodesinput_list)


def export_bash_jobs(filename, map_cmds, reduce_cmds):
    fileout = open(filename, "w+")
    for map_cmd in map_cmds:
        cmd_str = ""
        for cmd in map_cmd:
            cmd_str = cmd_str + cmd + " "
        fileout.write(cmd_str + "\n")
    for reduce_cmd in reduce_cmds:
        cmd_str = ""
        for cmd in reduce_cmd:
            cmd_str = cmd_str + cmd + " "
        fileout.write(cmd_str + "\n")
    fileout.close()


class BashWorkflowDescriptor(object):
    '''
    Parameters
    ----------
    dataset_dir_path: string
        The path which the saved dataset located. You can use
        epac.utils.save_dataset_path or epac.utils.save_dataset_path
        to save dictionary. Some examples are shown in
        epac.utils.save_dataset_path and epac.utils.save_dataset.
    epac_tree_dir_path: string
        The path where the epac tree is located.
    out_dir_path: string
        The path where the results have been saved.

    Example
    -------
    # =================================================================
    # Build dataset dir
    # =================================================================
    from sklearn import datasets
    from epac.utils import save_dataset_path
    import numpy as np
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=500,
                                        n_informative=2,
                                        random_state=1)
    working_dir_root = "/tmp"
    path_X = os.path.join(working_dir_root, "data_X.npy") #It could be any path
    path_y = os.path.join(working_dir_root, "data_y.npy") #It could be any path
    np.save(path_X, X)
    np.save(path_y, y)
    dataset_dir_path = os.path.join(working_dir_root, "dataset")
    path_Xy = {"X":path_X, "y":path_y}
    save_dataset_path(dataset_dir_path, **path_Xy)

    # =================================================================
    # Build epac tree (epac workflow) and save them on disk
    # =================================================================
    import os
    from epac import Methods
    from epac.stores import save_tree
    from sklearn.svm import LinearSVC as SVM
    epac_tree_dir_path = os.path.join(working_dir_root, "tree")
    if not os.path.exists(epac_tree_dir_path):
        os.makedirs(epac_tree_dir_path)
    multi = Methods(SVM(C=1), SVM(C=10))
    save_tree(multi, epac_tree_dir_path)

    # =================================================================
    # Export scripts to workflow directory
    # =================================================================
    from epac.map_reduce.wfdescriptors import BashWorkflowDescriptor
    # to save results in outdir, for example, the results of reducer
    out_dir_path = os.path.join(working_dir_root, "outdir")
    workflow_dir = os.path.join(working_dir_root, "workflow")
    wf_desc = BashWorkflowDescriptor(dataset_dir_path,
                                 epac_tree_dir_path,
                                 out_dir_path)
    wf_desc.export(workflow_dir=workflow_dir, num_processes=2)
    # =================================================================
    # Goto "workflow_dir" to run bash script
    # =================================================================
    '''
    def __init__(self, dataset_dir_path, epac_tree_dir_path, out_dir_path):
        self.dataset_dir_path = dataset_dir_path
        self.epac_tree_dir_path = epac_tree_dir_path
        self.out_dir_path = out_dir_path

    def export(self, workflow_dir, num_processes):
        '''
        Parameters
        ----------
        workflow_dir: string
            the directory to export workflow
        num_processes: integer
            the number of processes you want to run
        '''
        self.workflow_dir = workflow_dir
        if not os.path.exists(self.workflow_dir):
            os.makedirs(self.workflow_dir)
        tree_root = load_tree(self.epac_tree_dir_path)
        keysfile_list = export_jobs(tree_root,
                                    num_processes,
                                    workflow_dir)
        map_cmds = []
        reduce_cmds = []
        for i in range(len(keysfile_list)):
            key_path = os.path.join(workflow_dir, keysfile_list[i])
            map_cmd = []
            map_cmd.append("epac_mapper")
            map_cmd.append("--datasets")
            map_cmd.append(self.dataset_dir_path)
            map_cmd.append("--keysfile")
            map_cmd.append(key_path)
            map_cmd.append("--treedir")
            map_cmd.append(self.epac_tree_dir_path)
            map_cmds.append(map_cmd)
        reduce_cmd = []
        reduce_cmd.append("epac_reducer")
        reduce_cmd.append("--treedir")
        reduce_cmd.append(self.epac_tree_dir_path)
        reduce_cmd.append("--outdir")
        reduce_cmd.append(self.out_dir_path)
        reduce_cmds.append(reduce_cmd)
        filename_bash_jobs = os.path.join(workflow_dir, "bash_jobs.sh")
        export_bash_jobs(filename_bash_jobs, map_cmds, reduce_cmds)


class SomaWorkflowDescriptor(object):
    '''
    Parameters
    ----------
    dataset_dir_path: string
        The path which the saved dataset located. You can use
        epac.utils.save_dataset_path or epac.utils.save_dataset_path
        to save dictionary. Some examples are shown in
        epac.utils.save_dataset_path and epac.utils.save_dataset.
    epac_tree_dir_path: string
        The path where the epac tree is located.
    out_dir_path: string
        The path where the results have been saved.

    Example
    -------
    # =================================================================
    # Build dataset dir on computing resource
    #   (which means cluster, or distributed resource management system DRMS)
    # =================================================================
    from sklearn import datasets
    from epac.utils import save_dataset_path
    import numpy as np
    import os
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=500,
                                        n_informative=2,
                                        random_state=1)
    working_dir_root = "$HOME"
    working_dir_root = os.path.expandvars(working_dir_root)
    path_X = os.path.join(working_dir_root, "data_X.npy") #It could be any path
    path_y = os.path.join(working_dir_root, "data_y.npy") #It could be any path
    np.save(path_X, X)
    np.save(path_y, y)
    dataset_dir_path = os.path.join(working_dir_root, "dataset")
    path_Xy = {"X":path_X, "y":path_y}
    save_dataset_path(dataset_dir_path, **path_Xy)

    # =================================================================
    # Build epac tree (epac workflow) and save them on computing resource
    # =================================================================
    import os
    from epac import Methods
    from epac.stores import save_tree
    from sklearn.svm import LinearSVC as SVM
    epac_tree_dir_path = os.path.join(working_dir_root, "tree")
    if not os.path.exists(epac_tree_dir_path):
        os.makedirs(epac_tree_dir_path)
    multi = Methods(SVM(C=1), SVM(C=10))
    save_tree(multi, epac_tree_dir_path)

    # =================================================================
    # Export soma-workflow to workflow directory on computing resource
    # =================================================================
    from epac.map_reduce.wfdescriptors import SomaWorkflowDescriptor
    # to save results in outdir, for example, the results of reducer
    out_dir_path = os.path.join(working_dir_root, "outdir")
    workflow_dir = os.path.join(working_dir_root, "workflow")
    wf_desc = SomaWorkflowDescriptor(dataset_dir_path,
                                     epac_tree_dir_path,
                                     out_dir_path)
    wf_desc.export(workflow_dir=workflow_dir, num_processes=2)
    # =================================================================
    # 1. Goto "workflow_dir" and copy soma_workflow to your local machine
    # 2. Run soma_workflow using soma_workflow_gui
    # =================================================================
    '''
    def __init__(self, dataset_dir_path, epac_tree_dir_path, out_dir_path):
        self.dataset_dir_path = dataset_dir_path
        self.epac_tree_dir_path = epac_tree_dir_path
        self.out_dir_path = out_dir_path

    def export(self, workflow_dir, num_processes):
        '''
        Parameters
        ----------
        workflow_dir: string
            the directory to export workflow
        num_processes: integer
            the number of processes you want to run
        '''
        try:
            from soma_workflow.client import Job
            from soma_workflow.client import Group
            from soma_workflow.client import Workflow
            from soma_workflow.client import Helper
        except ImportError:
            errmsg = "No soma-workflow is found. "\
                "Please verify your soma-worklow"\
                "on your computer (e.g. PYTHONPATH) \n"
            sys.stderr.write(errmsg)
            sys.stdout.write(errmsg)
            raise NoSomaWFError

        self.workflow_dir = workflow_dir
        soma_workflow_file = os.path.join(self.workflow_dir, "soma_workflow")
        if not os.path.exists(self.workflow_dir):
            os.makedirs(self.workflow_dir)
        tree_root = load_tree(self.epac_tree_dir_path)
        keysfile_list = export_jobs(tree_root,
                                    num_processes,
                                    workflow_dir)
        # Building mapper task
        dependencies = []
        map_jobs = []
        for i in range(len(keysfile_list)):
            key_path = os.path.join(workflow_dir, keysfile_list[i])
            map_cmd = []
            map_cmd.append("epac_mapper")
            map_cmd.append("--datasets")
            map_cmd.append(self.dataset_dir_path)
            map_cmd.append("--keysfile")
            map_cmd.append(key_path)
            map_cmd.append("--treedir")
            map_cmd.append(self.epac_tree_dir_path)
            map_job = Job(command=map_cmd,
                          name="map_step",
                          referenced_input_files=[],
                          referenced_output_files=[])
            map_jobs.append(map_job)
        group_map_jobs = Group(elements=map_jobs,
                               name="all map jobs")
        # Building reduce task
        reduce_cmd = []
        reduce_cmd.append("epac_reducer")
        reduce_cmd.append("--treedir")
        reduce_cmd.append(self.epac_tree_dir_path)
        reduce_cmd.append("--outdir")
        reduce_cmd.append(self.out_dir_path)
        reduce_job = Job(command=reduce_cmd,
                         name="reduce_step",
                         referenced_input_files=[],
                         referenced_output_files=[])
        for map_job in map_jobs:
            dependencies.append((map_job, reduce_job))
        jobs = map_jobs + [reduce_job]
        # Build workflow and save into disk
        workflow = Workflow(jobs=jobs,
                            dependencies=dependencies,
                            root_group=[group_map_jobs,
                                        reduce_job])
        Helper.serialize(soma_workflow_file, workflow)


class SharePathSomaWorkflowDescriptor:
    '''
    Example
    -------
    # =======================================================================
    # Configuration on computing resource
    #   (which means cluster, or distributed resource management system DRMS)
    #   ! Replace $HOME as the working directory on computing resource
    #     1 Define a translation file, for example, in $HOME/translation_example as
    #         ```
    #         tr_soma_workflow_shared_dir $HOME/soma_workflow_shared_dir
    #         ```
    #     2 Add an option in soma-workflow configuration
    #         for example, $HOME/.soma-workflow.cfg
    #         ```
    #         PATH_TRANSLATION_FILES = EPAC{$HOME/translation_example}
    #         ```
    # =======================================================================

    # =======================================================================
    # Define path variables
    # =======================================================================
    import os
    root = "$HOME/soma_workflow_shared_dir"
    root = os.path.expandvars(root)
    dataset_relative_path = "dataset"
    tree_relative_path = "tree"
    jobs_relative_path = "jobs"
    output_relative_path = "output"
    script_relative_path = "soma_workflow"
    num_processes = 2
    namespace="EPAC"
    uuid="tr_soma_workflow_shared_dir"
    # =======================================================================
    # Build dataset on computing resource
    # =======================================================================
    from sklearn import datasets
    from epac.utils import save_dataset_path
    import numpy as np
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=500,
                                        n_informative=2,
                                        random_state=1)
    # These pathes can be arbitrary, not mandatory in $HOME/soma_workflow_shared_dir
    path_X = os.path.join(root, "data_X.npy")
    path_y = os.path.join(root, "data_y.npy")
    np.save(path_X, X)
    np.save(path_y, y)
    # Save paths in "$HOME/soma_workflow_shared_dir/dataset"
    # dataset_dir_path should be in "$HOME/soma_workflow_shared_dir"
    dataset_dir_path = os.path.join(root, dataset_relative_path)
    path_Xy = {"X":path_X, "y":path_y}
    save_dataset_path(dataset_dir_path, **path_Xy)
    # ======================================================================
    # Build epac tree (epac workflow) on computing resource
    # ======================================================================
    import os
    from epac import Methods
    from epac.map_reduce.wfdescriptors import export_jobs
    from epac.map_reduce.wfdescriptors import save_tree
    from sklearn.svm import LinearSVC as SVM
    epac_tree_dir_path = os.path.join(root, tree_relative_path)
    jobs_dir_path = os.path.join(root, jobs_relative_path)
    multi = Methods(SVM(C=1), SVM(C=10))
    save_tree(multi, epac_tree_dir_path)
    res = export_jobs(multi, num_processes, jobs_dir_path)
    # =====================================================================
    # Export scripts to workflow directory on computing resource
    # =====================================================================
    from epac.map_reduce.wfdescriptors import SharePathSomaWorkflowDescriptor
    # to save results in outdir, for example, the results of reducer
    swf_desc = SharePathSomaWorkflowDescriptor(namespace=namespace,
                                      uuid=uuid,
                                      root=root,
                                      dataset_relative_path=dataset_relative_path,
                                      tree_relative_path=tree_relative_path,
                                      jobs_relative_path=jobs_relative_path,
                                      output_relative_path=output_relative_path)
    script_path = os.path.join(root, script_relative_path)
    swf_desc.export(script_path=script_path)
    # =======================================================================
    # Now, you can copy $HOME/soma_workflow_shared_dir/soma_workflow
    # to your local computer (client) and run with soma_workflow_gui
    # =======================================================================
    '''
    def __init__(self,
                 namespace,
                 uuid,
                 root,
                 dataset_relative_path,
                 tree_relative_path,
                 jobs_relative_path,
                 output_relative_path):
        self.namespace = namespace
        self.uuid = uuid
        self.root = root
        self.dataset_relative_path = dataset_relative_path
        self.tree_relative_path = tree_relative_path
        self.jobs_relative_path = jobs_relative_path
        self.output_relative_path = output_relative_path

    def export(self, script_path):
        try:
            from soma_workflow.client import Job
            from soma_workflow.client import Group
            from soma_workflow.client import Workflow
            from soma_workflow.client import SharedResourcePath
            from soma_workflow.client import Helper
        except ImportError:
            errmsg = "No soma-workflow is found. "\
                "Please verify your soma-worklow"\
                "on your computer (e.g. PYTHONPATH) \n"
            sys.stderr.write(errmsg)
            sys.stdout.write(errmsg)
            raise NoSomaWFError

        # dataset on remote machine
        dataset_dir = SharedResourcePath(
                        relative_path=self.dataset_relative_path,
                        namespace=self.namespace,
                        uuid=self.uuid)
        # Tree on remote machine
        epac_tree_dir = SharedResourcePath(
                        relative_path=self.tree_relative_path,
                        namespace=self.namespace,
                        uuid=self.uuid)
        # Reduce output on remote machine
        out_dir = SharedResourcePath(relative_path=self.output_relative_path,
                                     namespace=self.namespace,
                                     uuid=self.uuid)
        # workflow file for soma-workflow
        soma_workflow_file = script_path
        # iterate all key jobs
        job_paths = []
        for root, _, files in os.walk(os.path.join(self.root,
                                                   self.jobs_relative_path)):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext == ".job":
                    job_paths.append(f)
        # Building mapper task
        dependencies = []
        map_jobs = []
        for i in range(len(job_paths)):
            job_relative_path = os.path.join(self.jobs_relative_path,
                                             job_paths[i])
            key_path = SharedResourcePath(relative_path=job_relative_path,
                                          namespace=self.namespace,
                                          uuid=self.uuid)
            map_cmd = []
            map_cmd.append("epac_mapper")
            map_cmd.append("--datasets")
            map_cmd.append(dataset_dir)
            map_cmd.append("--keysfile")
            map_cmd.append(key_path)
            map_cmd.append("--treedir")
            map_cmd.append(epac_tree_dir)
            map_job = Job(command=map_cmd,
                          name="map_step",
                          referenced_input_files=[],
                          referenced_output_files=[])
            map_jobs.append(map_job)
        group_map_jobs = Group(elements=map_jobs, name="all map jobs")
        # Building reduce step
        reduce_cmd = []
        reduce_cmd.append("epac_reducer")
        reduce_cmd.append("--treedir")
        reduce_cmd.append(epac_tree_dir)
        reduce_cmd.append("--outdir")
        reduce_cmd.append(out_dir)
        reduce_job = Job(command=reduce_cmd,
                         name="reduce_step",
                         referenced_input_files=[],
                         referenced_output_files=[])
        for map_job in map_jobs:
            dependencies.append((map_job, reduce_job))
        jobs = map_jobs + [reduce_job]
        # Build workflow and save into disk
        workflow = Workflow(jobs=jobs,
                            dependencies=dependencies,
                            root_group=[group_map_jobs,
                                        reduce_job])
        Helper.serialize(soma_workflow_file, workflow)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
