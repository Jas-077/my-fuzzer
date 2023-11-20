# Student Fuzzer
Template repository for CS5219

## Setup
Install all dependencies needed by the Fuzzing Book baseline fuzzer with:

```
pip install -r requirements.txt
```

You may want to do this in a Python **virtual environment** to avoid global dependency conflicts.

## Usage

The fuzzer expects a file named `bug.py` to be *in the same directory as the fuzzer file* (`student-fuzzer.py`).
This `bug.py` file should have two functions: an `entrypoint` that is fuzzed by the fuzzer and `get_initial_corpus` function which returns a list of initial inputs for the fuzzer.
To execute the fuzzer on the bug in `bug.py`, just run:

```
python student_fuzzer.py
```
Further, the 
```
buggy.py
```
program can be used to reproduce the plots given in the report. The ```example_bug.py``` bug file is the file shown in the report that is used for comparison of results obtained from Boosted Greybox and My Fuzzer.

The ```different_trials``` folder is for storing the array values for producing the plots

