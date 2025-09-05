import logging
from random import randrange, sample, uniform
import random
#random.seed(1)




# Configure the logger
logging.basicConfig(filename='#genetic_algorithm_sandwhich_logfile.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()
# Redirect print statements to logging.info
print_original = print
def print(*args, **kwargs):
    print_original(*args, **kwargs)
    logger.info(' '.join(map(str, args)))


# Ingredients with fictional health and popularity scores
BREAD_TYPES = [("Whole Wheat", 3, 1), ("Toast", -2, 2), ("Baguette", 0, 3)]
PROTEINS = [("Turkey", 0, 1), ("Ham", -3, 3), ("Veggie Sausage", 3, -1), ("Grilled Chicken", 2, 3)]
TOPPINGS = [("Lettuce", 3, 3), ("Tomato", 3, 2), ("Bacon", -3, 2), ("Cheddar", 0, 2)]
SAUCES = [("Mayonnaise", -3, 3), ("Butter", -2, 1), ("Mustard", 2, 0), ("Hummus", 3, 1)]

NUM_BREAD_BITS = 3
NUM_PROTEIN_BITS = len(PROTEINS)
NUM_TOPPING_BITS = len(TOPPINGS)
NUM_SAUCE_BITS = len(SAUCES)
CHROMOSOME_LENGTH = NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS + NUM_SAUCE_BITS

# Fitness function
def fitness(chromosome):
    print(f"Evaluating fitness for chromosome: {chromosome}")
    
    # Extract bits for each component
    bread_bits = chromosome[:NUM_BREAD_BITS]
    protein_bits = chromosome[NUM_BREAD_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS]
    topping_bits = chromosome[NUM_BREAD_BITS + NUM_PROTEIN_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS]
    sauce_bits = chromosome[NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS:]
    
    print(f"Bread bits: {bread_bits}")
    print(f"Protein bits: {protein_bits}")
    print(f"Topping bits: {topping_bits}")
    print(f"Sauce bits: {sauce_bits}")
    
    # Decode the bread type
    bread_index = int("".join(map(str, bread_bits)), 2) % len(BREAD_TYPES)
    bread_health, bread_popularity = BREAD_TYPES[bread_index][1], BREAD_TYPES[bread_index][2]
    
    print(f"Selected bread index: {bread_index} -> {BREAD_TYPES[bread_index]}")
    print(f"Bread health score: {bread_health}, Bread popularity score: {bread_popularity}")

    # Calculate health and popularity scores
    health_score = bread_health
    popularity_score = bread_popularity
    ingredient_count = 1  # Start with bread
    
    for i, bit in enumerate(protein_bits):
        if bit == 1:
            health_score += PROTEINS[i][1]
            popularity_score += PROTEINS[i][2]
            ingredient_count += 1
            print(f"Added protein: {PROTEINS[i]}")

    for i, bit in enumerate(topping_bits):
        if bit == 1:
            health_score += TOPPINGS[i][1]
            popularity_score += TOPPINGS[i][2]
            ingredient_count += 1
            print(f"Added topping: {TOPPINGS[i]}")

    for i, bit in enumerate(sauce_bits):
        if bit == 1:
            health_score += SAUCES[i][1]
            popularity_score += SAUCES[i][2]
            ingredient_count += 1
            print(f"Added sauce: {SAUCES[i]}")
    
    print(f"Total health score: {health_score}")
    print(f"Total popularity score: {popularity_score}")
    print(f"Total ingredient count: {ingredient_count}")
    
    # Penalize sandwiches with more than 6 ingredients
    if ingredient_count > 6:
        print("Too many ingredients! Penalizing chromosome.")
        return 0  # A hard penalty, invalidating the chromosome

    # Combine health and popularity scores (example: equal weighting)
    combined_score = (health_score + popularity_score) / 2
    print(f"Combined score: {combined_score}")
    return combined_score


# Check if a chromosome is valid
def is_valid(chromosome):
    print(f"Checking validity for chromosome: {chromosome}")
    
    bread_bits = chromosome[:NUM_BREAD_BITS]
    protein_bits = chromosome[NUM_BREAD_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS]
    topping_bits = chromosome[NUM_BREAD_BITS + NUM_PROTEIN_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS]
    sauce_bits = chromosome[NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS:]
    
    print(f"Bread bits: {bread_bits}")
    print(f"Protein bits: {protein_bits}")
    print(f"Topping bits: {topping_bits}")
    print(f"Sauce bits: {sauce_bits}")
    
    # Ensure at least one ingredient from each list
    if sum(protein_bits) < 1:
        print("Invalid: No protein selected")
        return False
    if sum(topping_bits) < 1:
        print("Invalid: No topping selected")
        return False
    if sum(sauce_bits) < 1:
        print("Invalid: No sauce selected")
        return False

    # Ensure no more than 6 ingredients in total
    ingredient_count = 1 + sum(protein_bits) + sum(topping_bits) + sum(sauce_bits)
    print(f"Total ingredient count: {ingredient_count}")
    
    if ingredient_count > 6:
        print("Invalid: More than 6 ingredients selected")
        return False
    
    print("Chromosome is valid")
    return True

# Initialize population
def initialize_population(pop_size):
    print(f"Initializing population with size: {pop_size}")
    population = []
    while len(population) < pop_size:
        chromosome = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
        print(f"Generated chromosome: {chromosome}")
        if is_valid(chromosome):
            population.append(chromosome)
            print(f"Chromosome added to population: {chromosome}")
        else:
            print(f"Chromosome is not valid and not added: {chromosome}")
    return population

# Selection
def select(population, fitnesses, num_parents):
    print(f"Selecting {num_parents} parents from population")
    selected_indices = random.choices(range(len(population)), weights=fitnesses, k=num_parents)
    selected_parents = [population[i] for i in selected_indices]
    print(f"Selected parents: {selected_parents}")
    return selected_parents

# Crossover
def crossover(parent1, parent2):
    print(f"Crossover between parent1: {parent1} and parent2: {parent2}")
    crossover_point = random.randint(1, CHROMOSOME_LENGTH - 1)
    print(f"Crossover point: {crossover_point}")
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    print(f"Generated offspring1: {offspring1}")
    print(f"Generated offspring2: {offspring2}")
    return offspring1, offspring2

# Mutation
def mutate(chromosome, mutation_rate=0.01):
    print(f"Mutating chromosome: {chromosome} with mutation rate: {mutation_rate}")
    mutated_chromosome = [bit if random.random() > mutation_rate else 1 - bit for bit in chromosome]
    print(f"Mutated chromosome: {mutated_chromosome}")
    return mutated_chromosome

# Genetic algorithm
def genetic_algorithm(pop_size, num_generations, mutation_rate=0.01):
    print(f"Running genetic algorithm with population size: {pop_size}, generations: {num_generations}, mutation rate: {mutation_rate}")
    population = initialize_population(pop_size)

    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        fitnesses = [fitness(chromosome) for chromosome in population]
        print(f"Fitnesses: {fitnesses}")
        
        # Selection
        parents = select(population, fitnesses, pop_size // 2)
        
        # Crossover
        next_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                offspring1, offspring2 = crossover(parents[i], parents[i + 1])
                if is_valid(offspring1):
                    next_population.append(offspring1)
                if is_valid(offspring2):
                    next_population.append(offspring2)
            else:
                if is_valid(parents[i]):
                    next_population.append(parents[i])
        
        # Mutation
        next_population = [mutate(chromosome, mutation_rate) for chromosome in next_population]
        
        # Replace old population with new population, ensuring all are valid
        population = [chromosome for chromosome in next_population if is_valid(chromosome)]
        
        # If the population size shrinks, add more valid chromosomes
        while len(population) < pop_size:
            chromosome = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
            if is_valid(chromosome):
                population.append(chromosome)
    
    # Return the best solution found
    fitnesses = [fitness(chromosome) for chromosome in population]
    best_index = fitnesses.index(max(fitnesses))
    best_solution = population[best_index]
    best_solution_fitness = fitness(population[best_index])
    print(f"Best solution found: {best_solution} with fitness: {best_solution_fitness}")
    return best_solution, best_solution_fitness

# Run the genetic algorithm
best_sandwich, best_fitness = genetic_algorithm(pop_size=10, num_generations=500)
print("Best Sandwich (binary):", best_sandwich)
print("Best Fitness:", best_fitness)

# Decode the best sandwich
bread_index = int("".join(map(str, best_sandwich[:NUM_BREAD_BITS])), 2) % len(BREAD_TYPES)
protein_bits = best_sandwich[NUM_BREAD_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS]
topping_bits = best_sandwich[NUM_BREAD_BITS + NUM_PROTEIN_BITS:NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS]
sauce_bits = best_sandwich[NUM_BREAD_BITS + NUM_PROTEIN_BITS + NUM_TOPPING_BITS:]

best_bread = BREAD_TYPES[bread_index][0]
best_proteins = [PROTEINS[i][0] for i, bit in enumerate(protein_bits) if bit == 1]
best_toppings = [TOPPINGS[i][0] for i, bit in enumerate(topping_bits) if bit == 1]
best_sauces = [SAUCES[i][0] for i, bit in enumerate(sauce_bits) if bit == 1]

print("Best Sandwich (decoded):")
print("Bread:", best_bread)
print("Proteins:", best_proteins)
print("Toppings:", best_toppings)
print("Sauces:", best_sauces)
