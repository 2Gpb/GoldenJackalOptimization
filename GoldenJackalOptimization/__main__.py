from GJO import *
import Benchmarks
import inspect
from time import time


def test():
    functions = sorted(list(
        filter(lambda x: x[0].startswith("f"), inspect.getmembers(
            Benchmarks, inspect.isfunction))
    ), key=lambda x: int(x[0][1:]))

    for func in functions:
        func = func[1]
        function_name, lb, up, dim = Benchmarks.get_function_param(func.__name__)
        start_time = time()
        gjo = GoldenJackalOptimization(fitness_function=func)
        best_solution, best_score = gjo.optimize()
        time_s = time() - start_time
        print(f'function_name = {function_name}\n'
              f'time_s = {time_s}\n'
              f'best_score = {best_score}\n'
              f'best_solution = {best_solution}')
        print("__________________________________________________________________________")


def main():
    test()


if __name__ == "__main__":
    main()
