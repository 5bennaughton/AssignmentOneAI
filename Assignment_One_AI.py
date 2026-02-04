import random
from typing import List, Tuple

'''
    Reads the timetable file and puts it into the correct readable format
    First line: N (number of exams), K (number of slots), M (number of students)
'''
def read_file(filename: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(filename, "r") as timeTable:
        lines = [];

        for line in timeTable:
            cleanedTimetable = line.strip();
            if cleanedTimetable:
                lines.append(cleanedTimetable);

    # First line of the timetable: N K M
    first = lines[0].split();
    N = int(first[0]);
    K = int(first[1]);
    M = int(first[2]);

    # Next part of the timetable (after the first line) which is the erollment matrix
    matrix = []
    for i in range(1, 1 + M):
        row = []
        for x in lines[i].split():
            row.append(int(x))
        matrix.append(row)

    return N, K, M, matrix

'''
    Function that is creating the random schedules, it is the GA
    Taking in N, K and the population size as a parameter
    Population size is the amount of random schedules needed to create
'''
def initialize_population(population_size: int, N: int, K: int) -> List[List[int]]:
    population = [];

    for x in range(population_size):
        schedule = []

        for y in range(N):
            schedule.append(random.randint(1, K));
        population.append(schedule);

    return population;


'''
    Function that will count how many hard violations are in a schedual
    A hard violation happens if a has two exams in the same time slot
'''
def count_hard_violations(schedule: List[int], enrollment: List[List[int]]) -> int:
    violations = 0;

    # For each student
    for student_row in enrollment:

        # Collect the slots for exams this student takes
        slots = [];

        for exam_index, takes_exam in enumerate(student_row):
            if takes_exam == 1:
                slots.append(schedule[exam_index]);

        # If any slot repeats, that's a conflict
        # Count how many times each slot appears
        duplicates = {}
        for s in slots:
            duplicates[s] = duplicates.get(s, 0) + 1

        # If a slot appears 2 times, that's 1 conflict
        # If it appears 3 times, that's 2 conflicts, etc
        for count in duplicates.values():
            if count > 1:
                violations += count - 1

    return violations


'''
    Function to count the soft constraints, a constraint happens if

    For each student
        - collect their exam slots
        - sort them
        - count how many consecutive pairs exist (+1)
'''
def count_soft_penalty(schedule: List[int], enrollment: List[List[int]]) -> int:
    penalty = 0

    for student_row in enrollment:
        slots = []

        for exam_index, takes_exam in enumerate(student_row):
            if takes_exam == 1:
                slots.append(schedule[exam_index])

        slots.sort()

        for i in range(len(slots) - 1):            
            if slots[i + 1] == slots[i] + 1:
                penalty += 1

    return penalty


"""
    Fitness score:
    - Hard violations are heavily penalized
    - Soft penalty is added

    Returning the just fitness score
"""
def evaluate_fitness(schedule: List[int], enrollment: List[List[int]]) -> int:

    hard = count_hard_violations(schedule, enrollment)
    soft = count_soft_penalty(schedule, enrollment)

    # Big penalty for hard violations 
    fitness = 1000 * hard + soft

    return fitness


"""
    Tournament selection:
    - pick 'size' random schedules
    - return the best one (lowest fitness)
"""
def tournament_select(population: List[List[int]], enrollment: List[List[int]], size: int =3) -> List[int]:
    candidates = random.sample(population, size)

    best = candidates[0]
    best_fit = evaluate_fitness(best, enrollment)

    for c in candidates:
        fit = evaluate_fitness(c, enrollment)
        if best is None or fit < best_fit:
            best = c
            best_fit = fit

    # This is returning a copy of the best, instead of the orginal list
    return best[:]  

"""
    Function that creates a new schedual by mixing two
    parent scheduals
"""
def one_point_crossover(parent1: List[int], parent2: List[int]) -> List[int]:

    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")

    index = random.randint(1, len(parent1) - 1)
    
    #add parent1 From start up to but not including index
    #to parent2 from index to the end
    child = parent1[:index] + parent2[index:]
    return child

"""
    This function mutates the schedual, assiging a random slot with x porbability
    This can help stop the algo from getting stuck
"""
def mutate(schedule: List[int], K: int, mutation_rate: float) -> List[int]:
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            schedule[i] = random.randint(1, K)

    return schedule


"""
    Run the genetic algorithm.

    Returns:
    best_schedule, best_fitness, best_hard, best_soft, best_fitness_over_time
"""
def run_ga(
    N: int,
    K: int,
    enrollment: List[List[int]],
    pop_size: int = 20,
    generations: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    tournament_size: int = 3,
) -> Tuple[List[int], int, List[int]]:
 
    # Create initial population
    population = initialize_population(pop_size, N, K)

    # Track best fitness each generation 
    best_fitness_over_time = []
    best_schedule = None
    best_fitness = None

    # Evolution loop
    for x in range(generations):
        new_population = []

        # Evaluate current population best
        for s in population:
            fitness = evaluate_fitness(s, enrollment)
            if best_schedule is None or fitness < best_fitness:
                best_schedule = s[:]
                best_fitness = fitness
   
        # Populate this array with all of the new best_fitnesses
        best_fitness_over_time.append(best_fitness)

        # Print out the best fitness and schedual with it for each iteration
        print(f"Generation {x + 1}: Best fitness = {best_fitness} | Best schedule = {best_schedule}")

        # Make next generation
        while len(new_population) < pop_size:
            # parents
            p1 = tournament_select(population, enrollment, tournament_size)
            p2 = tournament_select(population, enrollment, tournament_size)

            # Crossover
            if random.random() < crossover_rate:
                child = one_point_crossover(p1, p2)
            else:
                child = p1[:]  

            # Mutation
            mutate(child, K, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_schedule, best_fitness, best_fitness_over_time

"""
    Basic main function
"""
def main():
    
    filename = "tinyexample.txt"

    try:
        N, K, M, enrollment = read_file(filename)
    except FileNotFoundError:
        # if trys fail, this will be the timetable example
        N = 4
        K = 3
        M = 5
        enrollment = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ]
        print("File not found")

    # Run GA with very small parameters
    best_schedule, best_fitness, history = run_ga(
        N=N,
        K=K,
        enrollment=enrollment,
        pop_size=20,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
    )

    print("\nBest schedule:", best_schedule)
    print("Best fitness:", best_fitness)
    print("Hard violations:", count_hard_violations(best_schedule, enrollment))
    print("Soft penalty:", count_soft_penalty(best_schedule, enrollment))


if __name__ == "__main__":
    main()
