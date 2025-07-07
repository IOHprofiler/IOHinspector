from typing import Any
import atexit

import ioh
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA



class Container:
    @property
    def names(self):
        return [x for x in dir(self) if x[0] in "FHG"]
    
    def __repr__(self):
        values = ' '.join([f'{name}: {getattr(self, name):.2e}' for name in self.names])
        return f"<Container: {values}>"


class WrapperProblem:
    __protected_attributes__ = (
        "pymoo_problem",
        "logger",
        "container",
        "num_evaluations",
        "evaluate",
        "create_info",
        "update_container",
        "reset",
        "meta_data",
    )

    def __init__(
        self,
        pymoo_problem,
        algorithm_name: str = "pymoo_algorithm",
        folder_name: str = "pymoo_folder",
        root: str = "MO_Data",
        fid: int = 1,
        exp_attributes: dict = None,
        **kwargs
    ):
        self.pymoo_problem = pymoo_problem

        # TODO: This should be done a bit nicer (also not sure if this f's up the website)
        ioh.logger.property.RAWY = "F1"
        self.logger = ioh.logger.Analyzer(
            triggers=[ioh.logger.trigger.ALWAYS],
            algorithm_name=algorithm_name,
            folder_name=folder_name,
            root=root,
            **kwargs
        )
        for k,v in exp_attributes.items():
            self.logger.add_experiment_attribute(k, v)
        self.meta_data = ioh.MetaData(
            fid,
            1,
            f"pymoo_{pymoo_problem.name()}",
            pymoo_problem.n_var,
            ioh.OptimizationType.MIN,
        )
        self.container = Container()
        self.update_container(
            [None] * pymoo_problem.n_obj,
            [None] * pymoo_problem.n_eq_constr,
            [None] * pymoo_problem.n_ieq_constr,
        )
        self.logger.watch(self.container, self.container.names)
        self.logger.attach_problem(self.meta_data)
        self.num_evaluations = 0
        
        # logger.close should be called in order to create the json info file 
        # I now added an exit handler, we could also add a handle after the
        # call of minimize or something. 
        atexit.register(self.logger.close)

    def create_info(self, f1, x):
        return ioh.LogInfo(
            self.num_evaluations,
            # I replaced raw_y with f1, we could also put a Hv indicator or smth here
            f1,
            0,
            0,
            0,
            0,
            0,
            x,
            [],
            [],
            ioh.iohcpp.RealSolution([], 0),
            True,
        )

    def reset(self):
        self.logger.reset()
        self.logger.attach_problem(self.meta_data)
        self.num_evaluations = 0
        self.update_container(
            [None] * self.pymoo_problem.n_obj,
            [None] * self.pymoo_problem.n_eq_constr,
            [None] * self.pymoo_problem.n_ieq_constr,
        )

    def update_container(
        self,
        objectives: list[float],
        eq_constraints: list[float] = None,
        ieq_constraints: list[float] = None,
    ):
        for i, obj in enumerate(objectives[1:], 2):
            setattr(self.container, f"F{i}", obj )

        for i, con in enumerate(eq_constraints):
            setattr(self.container, f"H{i}", con)

        for i, con in enumerate(ieq_constraints):
            setattr(self.container, f"G{i}", con)

    def evaluate(self, X, *args, **kwargs):
        pymoo_evaluate_result = self.pymoo_problem.evaluate(X, *args, **kwargs)
        for i, x in enumerate(X):
            self.num_evaluations += 1
            self.update_container(
               pymoo_evaluate_result["F"][i] + self.ideal_point(),
               pymoo_evaluate_result["G"][i],
               pymoo_evaluate_result["H"][i]
            )
            print(i,x,pymoo_evaluate_result["F"][i], pymoo_evaluate_result["G"][i], pymoo_evaluate_result["H"][i])
            self.logger.call(self.create_info(pymoo_evaluate_result["F"][i][0] + self.ideal_point()[0], x))

        return pymoo_evaluate_result

    def __getattribute__(self, name: str) -> Any:
        if name in WrapperProblem.__protected_attributes__:
            return object.__getattribute__(self, name)
        return getattr(self.pymoo_problem, name)
    
    def __del__(self, *args, **kwargs):
        self.logger.close()

def run_experiment():
    problems = ["dtlz1"]#,'zdt1', 'zdt2']
    popsize = 10

    for idx, problem_name in enumerate(problems):
        problem = get_problem(problem_name)
        exp_attrs = {'popsize' : f"{popsize}"}
        wrapper_problem = WrapperProblem(problem, algorithm_name=f"NSGA2", algorithm_info=f"{popsize}", exp_attributes=exp_attrs, fid=idx, folder_name=f"NSGA_{popsize}")
        algorithm = NSGA2(pop_size=popsize)
        for i in range(5):
            res = minimize(wrapper_problem, algorithm, ("n_eval", 2000), seed=i, verbose=True)
            print(wrapper_problem.num_evaluations, res.X.shape, popsize)
            #After the run is finished, we evaluate the points returned by the algorithm to be able to look at the final algorithm recommendation as well
            wrapper_problem.evaluate(res.X, return_values_of=['F', 'G', 'H'], return_as_dictionary = True)
            print(wrapper_problem.num_evaluations)
            wrapper_problem.reset()
        wrapper_problem = WrapperProblem(problem, algorithm_name=f"SMS-EMOA", algorithm_info=f"{popsize}", exp_attributes=exp_attrs, fid=idx, folder_name=f"SMSEMOA_{popsize}")
        algorithm = SMSEMOA(pop_size=popsize)
        for i in range(5):
            res = minimize(wrapper_problem, algorithm, ("n_eval", 2000), seed=i, verbose=True)
            print(wrapper_problem.num_evaluations, res.X.shape, popsize)
            #After the run is finished, we evaluate the points returned by the algorithm to be able to look at the final algorithm recommendation as well
            wrapper_problem.evaluate(res.X, return_values_of=['F', 'G', 'H'], return_as_dictionary = True)
            print(wrapper_problem.num_evaluations)
            wrapper_problem.reset()

if __name__ == "__main__":
    import shutil
    shutil.rmtree("MO_Data", True)
    run_experiment()

