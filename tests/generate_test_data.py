import os
import shutil

import ioh
import numpy as np


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "data"))


def run(problem, algorithm_name):
    logger = ioh.logger.Analyzer(
        root=DATA_DIR, 
        folder_name=algorithm_name, 
        algorithm_name=algorithm_name
    )
    problem.attach_logger(logger)
    
    for _ in range(5):
        for _ in range(100):
            x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
            problem(x)
        problem.reset()
    

if __name__ == '__main__':
    p1 = ioh.get_problem(1, 1, 2)    
    p2 = ioh.get_problem(2, 1, 2)    
    
    run(p1, "algorithm_A")
    run(p1, "algorithm_A")
    run(p1, "algorithm_B")
    
    run(p2, "algorithm_A")
    run(p2, "algorithm_B")
    

    
    