from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf
from fuzzingbook.Fuzzer import Fuzzer,Runner
import traceback
import numpy as np
import time
from typing import List, Set, Any, Tuple, Dict, Union, Callable
import random
from bug import entrypoint
from bug import get_initial_corpus
import pickle
import hashlib

#random.seed(10)

def getPathID(coverage: Any) -> str:
    """Returns a unique hash for the covered statements"""
    pickled = pickle.dumps(sorted(coverage))
    #print("Inside getPath()")
    #print(coverage)
    #print(hashlib.md5(pickled).hexdigest())
    return hashlib.md5(pickled).hexdigest()

class AFLFastSchedule(gbf.PowerSchedule):
    """Exponential power schedule as implemented in AFL"""

    def __init__(self, exponent: float) -> None:
        self.exponent = exponent

    def assignEnergy(self, population: list) -> None:
        """Assign exponential energy inversely proportional to path frequency"""
        #print(population)
        for seed in population:
            #print("Inside assignEnergy()")
            #print(self.path_frequency)
            #print(f"Seed: ",seed)
            seed.energy = 1 / (self.path_frequency[getPathID(seed.coverage)] ** self.exponent)
            #print(f"Seed energe: {seed.energy}")

class Seed:
    """Represent an input with additional attributes"""

    def __init__(self, data: str) -> None:
        """Initialize from seed data"""
        self.data = data

        # These will be needed for advanced power schedules
        self.coverage: Set[cv.Location] = set()
        self.distance: Union[int, float] = -1
        self.energy = 0.0
        self.parent = ""
        self.mut_index = 0
        self.m_prob = {"delete":0.33,"insert":0.33,"flip":0.33}

    def __str__(self) -> str:
        """Returns data as string representation of the seed"""
        return self.data

    __repr__ = __str__


class Mutator:
    """Mutate strings"""

    def __init__(self) -> None:
        """Constructor"""
        self.mutators = [
            self.delete_random_character,
            self.insert_random_character,
            self.flip_random_character
        ]

class Mutator(Mutator):
    def insert_random_character(self, s: str) -> str:
        """Returns s with a random character inserted"""
        pos = random.randint(0, len(s))
        random_character = chr(random.randrange(32, 127))
        return s[:pos] + random_character + s[pos:]

    def delete_random_character(self, s: str) -> str:
        """Returns s with a random character deleted"""
        if s == "":
            return self.insert_random_character(s)

        pos = random.randint(0, len(s) - 1)
        return s[:pos] + s[pos + 1:]

    def flip_random_character(self, s: str) -> str:
        """Returns s with a random bit flipped in a random position"""
        if s == "":
            return self.insert_random_character(s)

        pos = random.randint(0, len(s) - 1)
        c = s[pos]
        bit = 1 << random.randint(0, 6)
        new_c = chr(ord(c) ^ bit)
        return s[:pos] + new_c + s[pos + 1:]

    def mutate(self, inp: Seed) -> Any:  # can be str or Seed (see below)
        """Return s with a random mutation applied. Can be overloaded in subclasses."""
        #print(f"data to mutate: {inp}")
        #print(type(inp))
        prob_list = [i for i in inp.m_prob.values()]
        #print(prob_list)
        mutator = random.choices(self.mutators,weights=prob_list,k=1)[0]
        #mutator = random.choice(self.mutators)
        #mutator = mutator[0]
        #print(mutator)
        #print(mutator)
        #print(self.mutators.index(mutator))
        m=mutator(inp.data)
        #print(m)
        #print(type(m))
        s = Seed(m)
        s.parent = inp.data
        s.mut_index = self.mutators.index(mutator)
        s.m_prob = inp.m_prob
        return s




class FunctionRunner(Runner):
    def __init__(self, function: Callable) -> None:
        """Initialize.  `function` is a function to be executed"""
        self.function = function

    def run_function(self, inp: str) -> Any:
        return self.function(inp)

    def run(self, inp: str) -> Tuple[Any, str]:
        try:
            result = self.run_function(inp)
            outcome = self.PASS
        except Exception:
            result = None
            outcome = self.FAIL

        return result, outcome

class MyCoverageRunner(FunctionRunner):
    def run_function(self, inp: Any) -> Any:
        #print("HIIIIIIIIIIIIIIIIIII")
        #print("inside MyCoverageRunner")
        #print("Inp: ",inp)
        with cv.Coverage() as cov:
            try:
                #print(f"Runner: {type(inp)}")
                if isinstance(inp,str):
                    result = super().run_function(inp)
                else: 
                    #print(type(inp.data))
                    result = super().run_function(str(inp.data))
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self) -> Set[cv.Location]:
        return self._coverage

class AdvancedMutationFuzzer(Fuzzer):
    """Base class for mutation-based fuzzing."""

    def __init__(self, seeds: List[str],
                 mutator: Mutator,
                 schedule: AFLFastSchedule) -> None:
        """Constructor.
        `seeds` - a list of (input) strings to mutate.
        `mutator` - the mutator to apply.
        `schedule` - the power schedule to apply.
        """
        self.seeds = seeds
        self.mutator = mutator
        self.schedule = schedule
        self.inputs: List[str] = []
        self.alpha =0.1
        self.reset()

    def reset(self) -> None:
        """Reset the initial population and seed index"""
        self.population = list(map(lambda x: Seed(x), self.seeds))
        self.seed_index = 0

    def create_candidate(self) -> str:
        """Returns an input generated by fuzzing a seed in the population"""
        seed = self.schedule.choose(self.population)
        #print(seed)
        #print(seed.m_prob)
        #print(all_e)
        # if sum(all_e)==0.0:
        #     seed = np.random.choice(self.population)
        # else:
        #     seed = np.random.choice(self.population,p=all_e)
        #print(seed)
        # Stacking: Apply multiple mutations to generate the candidate
        #print(type(seed))
        c_parent = seed
        candidate = seed
        trials = min(len(candidate.data), 1 << random.randint(1, 5))
        for i in range(trials):
            candidate = self.mutator.mutate(candidate)
        candidate.parent= c_parent
        return candidate

    def fuzz(self) -> str:
        """Returns first each seed once and then generates new inputs"""
        #print(len(self.seeds))
        if self.seed_index < len(self.seeds):
            # Still seeding
            #print(self.population)
            self.inp = self.seeds[self.seed_index]
            self.seed_index += 1
            self.inputs.append(self.inp)
        else:
            # Mutating
            self.inp = self.create_candidate()
            self.inputs.append(self.inp)
        #print(self.inputs)
        return self.inp
    

class GreyboxFuzzer(AdvancedMutationFuzzer):
    """Coverage-guided mutational fuzzing."""

    def reset(self):
        """Reset the initial population, seed index, coverage information"""
        super().reset()
        #self.schedule.path_frequency = {}
        self.coverages_seen = set()
        self.population = []  # population is filled during greybox fuzzing

    def run(self, runner: MyCoverageRunner) -> Tuple[Any, str]:
        """Run function(inp) while tracking coverage.
           If we reach new coverage,
           add inp to population and its coverage to population_coverage
        """
        result, outcome = super().run(runner)
        new_coverage = frozenset(runner.coverage())
        should_add=0
        if new_coverage not in self.coverages_seen:
            #print("We have new coverage")
            #print(self.inp)
            #print(type(self.inp))
            if isinstance(self.inp, str):
                seed = Seed(self.inp) 
                # print("Its a string")
                # print(self.inp)
                for s in self.population: 
                    #print(s.data)
                    if s.data == self.inp:
                        s.coverage = runner.coverage()
                        should_add=1
            else:
                #print("Its a Seed")
                seed = self.inp
                seed.m_prob =  {"delete":0.33,"insert":0.33,"flip":0.33}
                p_s = seed.parent 
                #print(f"Parent: {p_s}")
                m_index = seed.mut_index 
                #print(f"Population: {self.population}")
                #print(type(self.population))
                for s in self.population: 
                    #print(s.data)
                    if str(s.data) == str(p_s):
                        #print("Got a new seed")
                        #print(f"Parent: {s.data}")
                        op = m_op_list[m_index]
                        
                        s.m_prob[op]+=self.alpha 
                        seed.m_prob[op]+=(self.alpha/2)
                        norm = sum(list(s.m_prob.values()))
                        norm2 = sum(list(seed.m_prob.values()))
                        for k in range(3):
                            op_ = m_op_list[k]
                            s.m_prob[op_]/=norm
                            seed.m_prob[op_]/=norm2
                        #print(s.m_prob.values())
                        #s.m_prob.values() =s.m_prob.values()/ sum(s.m_prob.values())
                        #seed.m_prob = s.m_prob
            seed.coverage = runner.coverage()
            self.coverages_seen.add(new_coverage)
            if should_add==0:
                self.population.append(seed)
        return(result, outcome)
class CountingGreyboxFuzzer(GreyboxFuzzer):
    """Count how often individual paths are exercised."""

    def reset(self):
        """Reset path frequency"""
        super().reset()
        self.schedule.path_frequency = {}

    def run(self, runner: MyCoverageRunner) -> Tuple[Any, str]:
        """Inform scheduler about path frequency"""
        result, outcome = super().run(runner)

        path_id = getPathID(runner.coverage())
        if path_id not in self.schedule.path_frequency:
            self.schedule.path_frequency[path_id] = 1
        else:
            self.schedule.path_frequency[path_id] += 1

        return(result, outcome)


if __name__ == "__main__":
    seed_inputs = get_initial_corpus()
    m_op_list = ["delete","insert","flip"]
    n=15000
    #seed_inputs = [Seed(inp) for inp in seed_inputs]
    #print(seed_inputs[0].m_prob)
    fast_schedule = AFLFastSchedule(5)
    my_fuzzer = CountingGreyboxFuzzer(seed_inputs, Mutator(), fast_schedule)
    start = time.time()
    runner = MyCoverageRunner(entrypoint)
    my_fuzzer.runs(runner, trials=n)
    end = time.time()

    print(my_fuzzer.population)
    pop_data = []
    for inp in my_fuzzer.inputs[1:]:
        if isinstance(inp,str):
            pop_data.append(inp)
        else: 
            pop_data.append(inp.data)
    #print(pop_data)
    all_coverage, cumu_coverage = cv.population_coverage(pop_data, entrypoint)
    print(max(cumu_coverage))
    #print(runner.coverage())

    import matplotlib.pyplot as plt
    #np.save("too_big/myfuz_cmu10.npy",cumu_coverage)
    plt.plot(cumu_coverage, label="Greybox")
    plt.title('Coverage over time')
    plt.xlabel('# of inputs')
    plt.ylabel('lines covered')
    plt.show()

    fast_energy = fast_schedule.normalizedEnergy(my_fuzzer.population)

    for (seed, norm_energy) in zip(my_fuzzer.population, fast_energy):
        #print(seed.coverage)
        print("%0.5f, %s" % (norm_energy, repr(seed.data)))

    print("             path id 'p'           : path frequency 'f(p)'")
    print(my_fuzzer.schedule.path_frequency)

    x_axis = np.arange(len(my_fuzzer.schedule.path_frequency))
    y_axis = list(my_fuzzer.schedule.path_frequency.values())

    plt.bar(x_axis, y_axis)
    plt.xticks(x_axis)
    plt.ylim(0, n)
    plt.show()