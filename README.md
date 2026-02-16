# Assignment One

## Overview
Implements a genetic algorithm to build an exam timetable that satisfies hard constraints (no student clashes) and minimizes soft constraints (consecutive exams).

## Files
- `Assignment_One_AI.py` — main implementation
- `tinyexample.txt`, `test_case1.txt`, `small-2.txt` — input text files

## Code Structure
Functions in `Assignment_One_AI.py`:
- `read_file` — parse the input text file into `N`, `K`, `M`, and an enrollment matrix
- `initialize_population` — create random schedules
- `count_hard_violations` — counts exam clashes per student
- `count_soft_penalty` — counts consecutive exams per student
- `evaluate_fitness` — `1000 * hard + soft` this calculates the fitness function
- `tournament_select` — selection operator
- `one_point_crossover` — crossover operator
- `mutate` — mutation operator
- `evolve_population_once` — evolve a single island one generation
- `migrate_islands_ring` — ring migration between islands
- `run_island_ga` — island-model GA loop
- `main` — sets parameters, runs GA, prints results, plots fitness

## Configuration
Edit `main()` in `Assignment_One_AI.py` to change the following:
- input file (e.g. `test_case1.txt`)
- population size, generations, crossover, mutation rates
- number of islands and migration settings

## Output
The script prints:
- best schedule found
- best fitness
- hard violations
- soft penaltys
- graph of fitness over generations
- average fitness from 5 runs
