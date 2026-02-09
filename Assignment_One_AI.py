import random
from typing import List, Tuple

'''
    Reads the timetable file and puts it into the correct readable format
    First line: N (number of exams), K (number of slots), M (number of students)
'''
def read_file(fileName: str) -> Tuple[int, int, int, List[List[int]]]:
    with open(fileName, "r") as timeTable:
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
def initialize_population(populationSize: int, N: int, K: int) -> List[List[int]]:
    population = [];

    for x in range(populationSize):
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
    for studentRow in enrollment:

        # Collect the slots for exams this student takes
        slots = [];

        for examIndex, takesExam in enumerate(studentRow):
            if takesExam == 1:
                slots.append(schedule[examIndex]);

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

    for studentRow in enrollment:
        slots = []

        for examIndex, takesExam in enumerate(studentRow):
            if takesExam == 1:
                slots.append(schedule[examIndex])

        slots.sort()

        for i in range(len(slots) - 1):            
            if slots[i + 1] == slots[i] + 1:
                penalty += 1

    return penalty


"""
    Fitness score:
    - Goal is minimal cost score (lower = better)
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
    - pick 'tournament size' random schedules
    - return the best one (lowest fitness)
"""
def tournament_select(population: List[List[int]], enrollment: List[List[int]], tournamentSize: int =3) -> List[int]:
    candidates = random.sample(population, tournamentSize)

    best = candidates[0]
    bestFitness = evaluate_fitness(best, enrollment)

    for candidate in candidates:
        fitness = evaluate_fitness(candidate, enrollment)
        if best is None or fitness < bestFitness:
            best = candidate
            bestFitness = fitness

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
    # should we not return two children i.e. both mutations of the parents*
    child1 = parent1[:index] + parent2[index:]
   # child2 = parent1[index:] + parent2[:index]
    return child1  #, child2

"""
    This function mutates the schedual, assiging a random slot with x porbability
    This can help stop the algo from getting stuck
"""
def mutate(schedule: List[int], K: int, mutationRate: float) -> List[int]:
    for i in range(len(schedule)):
        if random.random() < mutationRate:
            schedule[i] = random.randint(1, K)

    return schedule


"""
    Run one GA generation for a single population.
"""
def evolve_population_once(
    population: List[List[int]],
    enrollment: List[List[int]],
    K: int,
    crossoverRate: float,
    mutationRate: float,
    tournamentSize: int,
) -> List[List[int]]:
    
    newPopulation = []

    while len(newPopulation) < len(population):
        if len(population) == 1:
            child = population[0][:]

        else:
            localTournamentSize = min(tournamentSize, len(population))
            p1 = tournament_select(population, enrollment, localTournamentSize)
            p2 = tournament_select(population, enrollment, localTournamentSize)

            if random.random() < crossoverRate:
                child = one_point_crossover(p1, p2)
            else:
                child = p1[:]

        mutate(child, K, mutationRate)
        newPopulation.append(child)

    return newPopulation


"""
    Island migration:
    - picks a random migrant from each island
    - picks a random schudle from each island to replace with the migrant
"""
def migrate_islands_ring(
    islands: List[List[List[int]]],
    migrantsPerIsland: int,
) -> None:
    outgoing = []

    # Building migrants for each island
    for sourcePopulation in islands:
        # Send no more migrants that island has
        moves = min(migrantsPerIsland, len(sourcePopulation))

        migrants = []
        for x in range(moves):
            # Picks a random schedule from the island
            pickedSchedule = random.choice(sourcePopulation)
            migrants.append(pickedSchedule[:])

        outgoing.append(migrants)

    # Here we send a migrant to the next island
    for islandNumber, migrants in enumerate(outgoing):
        # This sends a migrant from each island to the next island, and the last island wraps back to the first island
        nextIsland = (islandNumber + 1) % len(islands)
        targetPopulation = islands[nextIsland]

        moves = min(len(migrants), len(targetPopulation))

        
        for migrant in migrants[:moves]:
            # Picks random schduale in an island and replaces it with the migrant
            replaceSchedule = random.randrange(len(targetPopulation))
            targetPopulation[replaceSchedule] = migrant


"""
     island-model GA:
    - split all the population into multiple islands
    - evolve each island by its own using the @evolve_population_once function
    - Use the @migrate_islands_ring function to migrate random schedules to the next island from each island
"""
def run_island_ga(
    N: int,
    K: int,
    enrollment: List[List[int]],
    totalPopulationSize: int = 40,
    numberOfIslands: int = 4,
    generations: int = 50,
    crossoverRate: float = 0.8,
    mutationRate: float = 0.05,
    tournamentSize: int = 3,
    migrationInterval: int = 10,
    migrantsPerIsland: int = 1,
) -> Tuple[List[int], int, List[int]]:

    # Split total population as evenly as possible across islands
    islandSize = totalPopulationSize // numberOfIslands
    islands = []

    for x in range(numberOfIslands):
        islands.append(initialize_population(islandSize, N, K))

    bestSchedule = None
    bestFitness = None


    for generation in range(generations):
        # Evolve each island one generation
        for islandIndex in range(numberOfIslands):
            islands[islandIndex] = evolve_population_once(
                population=islands[islandIndex],
                enrollment=enrollment,
                K=K,
                crossoverRate=crossoverRate,
                mutationRate=mutationRate,
                tournamentSize=tournamentSize,
            )

            # Migrate the islands
            migrate_islands_ring(
                islands=islands,
                migrantsPerIsland=migrantsPerIsland,
            )

        # Track the best
        for population in islands:
            for schedule in population:
                fitness = evaluate_fitness(schedule, enrollment)

                if bestSchedule is None or fitness < bestFitness:
                    bestSchedule = schedule[:]
                    bestFitness = fitness

        print(
            f"Generation {generation + 1}: "
            f"Best fitness = {bestFitness} | Best schedule = {bestSchedule}"
        )

    return bestSchedule, bestFitness

"""
    Main function
"""
def main():
    fileName = "test_case1.txt"
    N, K, _, enrollment = read_file(fileName)

    # Run island model GA
    bestSchedule, bestFitness = run_island_ga(
        N=N,
        K=K,
        enrollment=enrollment,
        totalPopulationSize=40,
        numberOfIslands=4,
        generations=50,
        crossoverRate=0.9,
        mutationRate=0.10,
        tournamentSize=3,
        migrationInterval=10,
        migrantsPerIsland=1,
    )

    print("\nBest schedule:", bestSchedule)
    print("Best fitness:", bestFitness)
    print("Hard violations:", count_hard_violations(bestSchedule, enrollment))
    print("Soft penalty:", count_soft_penalty(bestSchedule, enrollment))


if __name__ == "__main__":
    main()
