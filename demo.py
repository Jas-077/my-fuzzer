from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf

from typing import List, Set, Any, Tuple, Dict, Union

import traceback
import numpy as np
import time

from bug import entrypoint
from bug import get_initial_corpus

## You can re-implement the coverage class to change how
## the fuzzer tracks new behavior in the SUT

class MyCoverage(cv.Coverage):
    def coverage(self) -> Set[cv.Location]:
        """The set of executed lines, as (function_name, line_number) pairs"""
        # print(self.trace())
        return self.trace()


## You can re-implement the runner class to change how
## the fuzzer tracks new behavior in the SUT

class MyFunctionCoverageRunner(mf.FunctionRunner):
    def run_function(self, inp: str) -> Any:
        with MyCoverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self) -> Set[cv.Location]:
        return self._coverage


# class MyRunner(mf.FunctionRunner):
#
#     def run_function(self, inp):
#           <your implementation here>
#
#     def coverage(self):
#           <your implementation here>
#
#     etc...


## You can re-implement the fuzzer class to change your
## fuzzer's overall structure

# class MyFuzzer(gbf.GreyboxFuzzer):
#
#     def reset(self):
#           <your implementation here>
#
#     def run(self, runner: gbf.FunctionCoverageRunner):
#           <your implementation here>
#   etc...

## The Mutator and Schedule classes can also be extended or
## replaced by you to create your own fuzzer!


    
# When executed, this program should run your fuzzer for a very 
# large number of iterations. The benchmarking framework will cut 
# off the run after a maximum amount of time
#
# The `get_initial_corpus` and `entrypoint` functions will be provided
# by the benchmarking framework in a file called `bug.py` for each 
# benchmarking run. The framework will track whether or not the bug was
# found by your fuzzer -- no need to keep track of crashing inputs
if __name__ == "__main__":
    seed_inputs = get_initial_corpus()
    fast_schedule = gbf.AFLFastSchedule(5)
    line_runner = MyFunctionCoverageRunner(entrypoint)

    fast_fuzzer = gbf.CountingGreyboxFuzzer(seed_inputs, gbf.Mutator(), fast_schedule)
    fast_fuzzer.runs(line_runner, trials=1000)
    print(fast_fuzzer.population)
    pop_data = [inp.data for inp in fast_fuzzer.inputs[1:]]
    print(pop_data)
    all_coverage, cumu_coverage = cv.population_coverage(pop_data, entrypoint)
    #print(cum_coverage)
    print(max(cumu_coverage))
    print(line_runner.coverage())
    
    import matplotlib.pyplot as plt
    plt.plot(cumu_coverage, label="Greybox")
    plt.title('Coverage over time')
    plt.xlabel('# of inputs')
    plt.ylabel('lines covered')
    plt.show()
    
    fast_energy = fast_schedule.normalizedEnergy(fast_fuzzer.population)

    for (seed, norm_energy) in zip(fast_fuzzer.population, fast_energy):
        #print(seed.coverage)
        print("%0.5f, %s" % (
                                norm_energy, repr(seed.data)))
    
    print("             path id 'p'           : path frequency 'f(p)'")
    print(fast_fuzzer.schedule.path_frequency)
    
    x_axis = np.arange(len(fast_fuzzer.schedule.path_frequency))
    y_axis = list(fast_fuzzer.schedule.path_frequency.values())

    plt.bar(x_axis, y_axis)
    plt.xticks(x_axis)
    plt.ylim(0, 10000)
    # plt.yscale("log")
    # plt.yticks([10,100,1000,10000])
    plt.show()
