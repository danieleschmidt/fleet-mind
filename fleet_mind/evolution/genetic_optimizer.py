"""Genetic Optimizer for Swarm Evolution - Generation 5."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
import time


class MutationType(Enum):
    """Types of genetic mutations."""
    POINT_MUTATION = "point_mutation"
    INSERTION = "insertion"
    DELETION = "deletion"
    INVERSION = "inversion"
    TRANSLOCATION = "translocation"
    DUPLICATION = "duplication"


@dataclass
class Chromosome:
    """Represents a chromosome in the genetic algorithm."""
    genes: List[float]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    mutation_history: List[MutationType] = None
    
    def __post_init__(self):
        if self.mutation_history is None:
            self.mutation_history = []


class GeneticOptimizer:
    """Advanced genetic optimizer for swarm characteristics."""
    
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 50,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        self.population: List[Chromosome] = []
        self.generation = 0
        self.fitness_function: Optional[Callable] = None
        self.optimization_history = []
        
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for i in range(self.population_size):
            genes = [random.uniform(0.0, 1.0) for _ in range(self.chromosome_length)]
            chromosome = Chromosome(
                genes=genes,
                generation=self.generation
            )
            self.population.append(chromosome)
            
    def set_fitness_function(self, fitness_func: Callable[[List[float]], float]):
        """Set the fitness evaluation function."""
        self.fitness_function = fitness_func
        
    async def evaluate_population(self):
        """Evaluate fitness of entire population."""
        if not self.fitness_function:
            # Default fitness function (maximize gene values)
            for chromosome in self.population:
                chromosome.fitness = sum(chromosome.genes) / len(chromosome.genes)
        else:
            for chromosome in self.population:
                chromosome.fitness = self.fitness_function(chromosome.genes)
                
    async def selection(self) -> List[Chromosome]:
        """Select parents for reproduction."""
        # Sort by fitness (descending)
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Elite selection
        num_elite = int(self.population_size * self.elitism_rate)
        selected = sorted_population[:num_elite].copy()
        
        # Tournament selection for remaining slots
        remaining_slots = self.population_size - num_elite
        for _ in range(remaining_slots):
            tournament_size = 3
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
            
        return selected
        
    async def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            # No crossover, return copy of parent1
            return Chromosome(
                genes=parent1.genes.copy(),
                generation=self.generation + 1
            )
            
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        child_genes = (parent1.genes[:crossover_point] + 
                      parent2.genes[crossover_point:])
                      
        return Chromosome(
            genes=child_genes,
            generation=self.generation + 1
        )
        
    async def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation to chromosome."""
        mutated_genes = chromosome.genes.copy()
        mutations_applied = []
        
        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(list(MutationType))
                
                if mutation_type == MutationType.POINT_MUTATION:
                    # Point mutation - change single gene
                    mutated_genes[i] += random.gauss(0, 0.1)
                    mutated_genes[i] = max(0.0, min(1.0, mutated_genes[i]))
                    mutations_applied.append(mutation_type)
                    
                elif mutation_type == MutationType.INSERTION:
                    # Insert random gene
                    new_gene = random.uniform(0.0, 1.0)
                    mutated_genes.insert(i, new_gene)
                    if len(mutated_genes) > self.chromosome_length:
                        mutated_genes.pop()  # Remove last gene to maintain length
                    mutations_applied.append(mutation_type)
                    
                elif mutation_type == MutationType.DELETION:
                    # Delete gene (replace with random)
                    if len(mutated_genes) > 1:
                        mutated_genes[i] = random.uniform(0.0, 1.0)
                        mutations_applied.append(mutation_type)
                        
                elif mutation_type == MutationType.INVERSION:
                    # Invert gene value
                    mutated_genes[i] = 1.0 - mutated_genes[i]
                    mutations_applied.append(mutation_type)
                    
        chromosome.genes = mutated_genes
        chromosome.mutation_history.extend(mutations_applied)
        return chromosome
        
    async def evolve_generation(self):
        """Evolve one generation."""
        # Evaluate current population
        await self.evaluate_population()
        
        # Record generation statistics
        fitness_values = [c.fitness for c in self.population]
        generation_stats = {
            'generation': self.generation,
            'max_fitness': max(fitness_values),
            'avg_fitness': sum(fitness_values) / len(fitness_values),
            'min_fitness': min(fitness_values),
            'timestamp': time.time()
        }
        self.optimization_history.append(generation_stats)
        
        # Select parents
        parents = await self.selection()
        
        # Create new population
        new_population = []
        
        # Add elite individuals
        num_elite = int(self.population_size * self.elitism_rate)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:num_elite]
        for individual in elite:
            individual.age += 1
            new_population.append(individual)
            
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            child = await self.crossover(parent1, parent2)
            child = await self.mutate(child)
            
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
    async def optimize(self, num_generations: int) -> Chromosome:
        """Run optimization for specified generations."""
        if not self.population:
            self.initialize_population()
            
        for _ in range(num_generations):
            await self.evolve_generation()
            
        # Return best individual
        await self.evaluate_population()
        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {}
            
        latest_stats = self.optimization_history[-1]
        initial_stats = self.optimization_history[0]
        
        fitness_improvement = latest_stats['max_fitness'] - initial_stats['max_fitness']
        
        return {
            'current_generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': latest_stats['max_fitness'],
            'average_fitness': latest_stats['avg_fitness'],
            'fitness_improvement': fitness_improvement,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'optimization_history_length': len(self.optimization_history)
        }
        
    def get_best_chromosome(self) -> Optional[Chromosome]:
        """Get the best chromosome from current population."""
        if not self.population:
            return None
            
        return max(self.population, key=lambda x: x.fitness)
        
    def analyze_genetic_diversity(self) -> Dict[str, float]:
        """Analyze genetic diversity of population."""
        if not self.population:
            return {}
            
        # Calculate gene variance across population
        gene_variances = []
        for gene_idx in range(self.chromosome_length):
            gene_values = [chromo.genes[gene_idx] for chromo in self.population]
            if len(gene_values) > 1:
                import statistics
                variance = statistics.variance(gene_values)
                gene_variances.append(variance)
                
        avg_variance = sum(gene_variances) / len(gene_variances) if gene_variances else 0.0
        
        # Calculate fitness diversity
        fitness_values = [c.fitness for c in self.population]
        fitness_variance = statistics.variance(fitness_values) if len(fitness_values) > 1 else 0.0
        
        return {
            'genetic_variance': avg_variance,
            'fitness_variance': fitness_variance,
            'diversity_index': min(1.0, avg_variance * 10.0),  # Normalized diversity score
            'population_age_avg': sum(c.age for c in self.population) / len(self.population)
        }