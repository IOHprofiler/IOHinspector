from typing import Any
import atexit

import ioh
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize


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
    )

    def __init__(
        self,
        pymoo_problem,
        algorithm_name: str = "pymoo_algorithm",
        folder_name: str = "pymoo_folder",
        root: str = "pymoo.Experiment",
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
        meta_data = ioh.MetaData(
            1,
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
        self.logger.attach_problem(meta_data)
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

    def update_container(
        self,
        objectives: list[float],
        eq_constraints: list[float] = None,
        ieq_constraints: list[float] = None,
    ):
        for i, obj in enumerate(objectives[1:], 2):
            setattr(self.container, f"F{i}", obj)

        for i, con in enumerate(eq_constraints):
            setattr(self.container, f"H{i}", con)

        for i, con in enumerate(ieq_constraints):
            setattr(self.container, f"G{i}", con)

    def evaluate(self, X, *args, **kwargs):
        pymoo_evaluate_result = self.pymoo_problem.evaluate(X, *args, **kwargs)

        for i, x in enumerate(X):
            self.num_evaluations += 1
            self.update_container(
               pymoo_evaluate_result["F"][i],
               pymoo_evaluate_result["G"][i],
               pymoo_evaluate_result["H"][i]
            )
            self.logger.call(self.create_info(pymoo_evaluate_result["F"][i][0], x))

        return pymoo_evaluate_result

    def __getattribute__(self, name: str) -> Any:
        if name in WrapperProblem.__protected_attributes__:
            return object.__getattribute__(self, name)
        return getattr(self.pymoo_problem, name)
    
    def __del__(self, *args, **kwargs):
        self.logger.close()
        return super().__del__(*args, **kwargs)



if __name__ == "__main__":
    import shutil
    shutil.rmtree("pymoo.Experiment", True)
    problem = get_problem("zdt1")
    wrapper_problem = WrapperProblem(problem, algorithm_name="NSGA2")

    algorithm = NSGA2(pop_size=100)
    res = minimize(wrapper_problem, algorithm, ("n_gen", 20), seed=1, verbose=True)