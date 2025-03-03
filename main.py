"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging


import sys
import os

# Get the root directory of your project (WebMall)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add all src directories to the Python path
src_dirs = [
    os.path.join(project_root, 'BrowserGym', 'browsergym', 'core', 'src'),
    os.path.join(project_root, 'BrowserGym', 'browsergym', 'webmall', 'src'),
    os.path.join(project_root, 'BrowserGym', 'browsergym', 'experiments', 'src'),
]

for src_dir in src_dirs:
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

# Optionally, print sys.path to ensure everything is included
print(sys.path)

import browsergym.webmall


from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
)
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
agent_args = [AGENT_4o_MINI]
# agent_args = [AGENT_4o]


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
# benchmark = "webarena"
benchmark = "webmall"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 1  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
