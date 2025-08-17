"""Self-Evolving Swarm System - Generation 5."""

import asyncio
# Fallback for numpy
try:
    import numpy as np
except ImportError:
    class MockNumPy:
        @staticmethod
        def random():
            import random
            class MockRandom:
                @staticmethod
                def uniform(low, high, size=None):
                    if size is None:
                        return random.uniform(low, high)
                    return [random.uniform(low, high) for _ in range(size)]
            return MockRandom()
    np = MockNumPy()
import time
import random
import copy
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque


class EvolutionStrategy(Enum):
    """Evolution strategies for swarm development."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM_OPTIMIZATION = "particle_swarm_optimization"
    NEURAL_EVOLUTION = "neural_evolution"
    CULTURAL_EVOLUTION = "cultural_evolution"


class FitnessMetric(Enum):
    """Metrics for evaluating swarm fitness."""
    COORDINATION_EFFICIENCY = "coordination_efficiency"
    MISSION_SUCCESS_RATE = "mission_success_rate"
    ENERGY_OPTIMIZATION = "energy_optimization"
    ADAPTATION_SPEED = "adaptation_speed"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    SURVIVAL_RATE = "survival_rate"
    INNOVATION_CAPACITY = "innovation_capacity"


@dataclass
class SwarmGenotype:
    """Genetic representation of swarm characteristics."""
    coordination_genes: Dict[str, float]
    behavior_genes: Dict[str, float]
    communication_genes: Dict[str, float]
    intelligence_genes: Dict[str, float]
    adaptation_genes: Dict[str, float]
    generation: int = 0
    fitness_score: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    mutation_rate: float = 0.01
    
    def __post_init__(self):
        # Ensure all gene values are normalized [0, 1]
        for gene_group in [self.coordination_genes, self.behavior_genes, 
                          self.communication_genes, self.intelligence_genes, 
                          self.adaptation_genes]:
            for key, value in gene_group.items():
                gene_group[key] = max(0.0, min(1.0, value))


@dataclass
class EvolutionEvent:
    """Records an evolution event in the swarm."""
    event_type: str
    generation: int
    fitness_improvement: float
    genetic_changes: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    affected_drones: Set[int] = field(default_factory=set)


@dataclass
class EvolutionState:
    """Current state of swarm evolution."""
    current_generation: int
    population_size: int
    average_fitness: float
    best_fitness: float
    genetic_diversity: float
    evolution_rate: float
    selection_pressure: float
    mutation_rate: float
    innovation_index: float
    timestamp: float = field(default_factory=time.time)


class SelfEvolvingSwarm:
    """Self-evolving drone swarm with autonomous genetic optimization.
    
    The swarm continuously evolves its coordination patterns, behaviors,
    and intelligence through genetic algorithms and evolutionary strategies.
    """
    
    def __init__(
        self,
        initial_population_size: int = 50,
        evolution_strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM,
        fitness_metrics: List[FitnessMetric] = None,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        selection_pressure: float = 0.3,
        enable_cultural_evolution: bool = True,
        enable_neural_evolution: bool = True,
        max_generations: int = 1000
    ):
        self.initial_population_size = initial_population_size
        self.current_population_size = initial_population_size
        self.evolution_strategy = evolution_strategy
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.enable_cultural_evolution = enable_cultural_evolution
        self.enable_neural_evolution = enable_neural_evolution
        self.max_generations = max_generations
        
        if fitness_metrics is None:
            self.fitness_metrics = [
                FitnessMetric.COORDINATION_EFFICIENCY,
                FitnessMetric.MISSION_SUCCESS_RATE,
                FitnessMetric.COLLECTIVE_INTELLIGENCE
            ]
        else:
            self.fitness_metrics = fitness_metrics
        
        # Population management
        self.population: Dict[str, SwarmGenotype] = {}
        self.fitness_history: deque = deque(maxlen=100)
        self.evolution_history: List[EvolutionEvent] = []
        
        # Evolution state
        self.evolution_state = EvolutionState(
            current_generation=0,
            population_size=initial_population_size,
            average_fitness=0.0,
            best_fitness=0.0,
            genetic_diversity=1.0,
            evolution_rate=0.0,
            selection_pressure=selection_pressure,
            mutation_rate=mutation_rate,
            innovation_index=0.0
        )
        
        # Cultural evolution (learned behaviors and knowledge)
        self.cultural_memory: Dict[str, Any] = {}
        self.collective_knowledge: Dict[str, float] = {}
        self.behavioral_templates: Dict[str, Dict[str, Any]] = {}
        
        # Neural evolution (evolving neural network architectures)
        self.neural_architectures: Dict[str, Dict[str, Any]] = {}
        
        # Evolution metrics
        self.evolution_metrics = {
            'total_generations': 0,
            'successful_mutations': 0,
            'successful_crossovers': 0,
            'innovation_events': 0,
            'cultural_transmission_events': 0,
            'neural_architecture_improvements': 0,
            'fitness_improvements': 0,
            'genetic_bottlenecks': 0
        }
        
        # Evolution processes
        self._evolution_task = None
        self._is_evolving = False
        
        # Performance tracking for fitness evaluation
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the initial population with random genotypes."""
        print(f"🧬 Initializing evolution population of {self.initial_population_size}")
        
        for i in range(self.initial_population_size):
            genotype_id = f"gen0_individual_{i}"
            
            # Create random genotype
            genotype = SwarmGenotype(
                coordination_genes={
                    'formation_stability': random.uniform(0.3, 0.9),
                    'leader_following': random.uniform(0.2, 0.8),
                    'obstacle_avoidance': random.uniform(0.5, 1.0),
                    'path_optimization': random.uniform(0.3, 0.9),
                    'communication_range': random.uniform(0.4, 1.0)
                },
                behavior_genes={
                    'exploration_tendency': random.uniform(0.2, 0.8),
                    'risk_tolerance': random.uniform(0.1, 0.7),
                    'cooperation_level': random.uniform(0.6, 1.0),
                    'learning_rate': random.uniform(0.3, 0.9),
                    'adaptability': random.uniform(0.4, 1.0)
                },
                communication_genes={
                    'signal_strength': random.uniform(0.5, 1.0),
                    'message_frequency': random.uniform(0.3, 0.8),
                    'information_sharing': random.uniform(0.6, 1.0),
                    'consensus_threshold': random.uniform(0.4, 0.9),
                    'broadcasting_range': random.uniform(0.3, 0.9)
                },
                intelligence_genes={
                    'decision_speed': random.uniform(0.4, 0.9),
                    'pattern_recognition': random.uniform(0.3, 0.8),
                    'problem_solving': random.uniform(0.4, 0.9),
                    'memory_capacity': random.uniform(0.5, 1.0),
                    'reasoning_depth': random.uniform(0.2, 0.7)
                },
                adaptation_genes={
                    'mutation_tolerance': random.uniform(0.2, 0.8),
                    'environmental_sensitivity': random.uniform(0.3, 0.9),
                    'learning_efficiency': random.uniform(0.4, 1.0),
                    'change_acceptance': random.uniform(0.5, 1.0),
                    'innovation_drive': random.uniform(0.2, 0.9)
                },
                generation=0,
                mutation_rate=self.mutation_rate
            )
            
            self.population[genotype_id] = genotype
        
        print(f"✅ Population initialized with {len(self.population)} individuals")
    
    async def start_evolution(self):
        """Start the autonomous evolution process."""
        if self._is_evolving:
            return
        
        self._is_evolving = True
        print(f"🌱 Starting autonomous swarm evolution")
        print(f"   Strategy: {self.evolution_strategy.value}")
        print(f"   Fitness metrics: {[m.value for m in self.fitness_metrics]}")
        
        # Start evolution loop
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        
        print("🧬 Autonomous evolution activated")
    
    async def _evolution_loop(self):
        """Main evolution processing loop."""
        while self._is_evolving and self.evolution_state.current_generation < self.max_generations:
            try:
                # Evaluate population fitness
                await self._evaluate_population_fitness()
                
                # Selection phase
                selected_parents = await self._selection_phase()
                
                # Reproduction phase (crossover and mutation)
                await self._reproduction_phase(selected_parents)
                
                # Cultural evolution
                if self.enable_cultural_evolution:
                    await self._cultural_evolution_phase()
                
                # Neural evolution
                if self.enable_neural_evolution:
                    await self._neural_evolution_phase()
                
                # Update evolution state
                await self._update_evolution_state()
                
                # Check for innovations
                await self._detect_innovations()
                
                # Advance generation
                self.evolution_state.current_generation += 1
                self.evolution_metrics['total_generations'] += 1
                
                if self.evolution_state.current_generation % 10 == 0:
                    print(f"🧬 Generation {self.evolution_state.current_generation}")
                    print(f"   Best fitness: {self.evolution_state.best_fitness:.3f}")
                    print(f"   Avg fitness: {self.evolution_state.average_fitness:.3f}")
                    print(f"   Genetic diversity: {self.evolution_state.genetic_diversity:.3f}")
                
                # Evolution timestep (slower than coordination)
                await asyncio.sleep(5.0)  # 5 second generation cycles
                
            except Exception as e:
                print(f"❌ Evolution error: {e}")
                await asyncio.sleep(10.0)
    
    async def _evaluate_population_fitness(self):
        """Evaluate fitness of all individuals in the population."""
        for genotype_id, genotype in self.population.items():
            fitness_score = 0.0
            
            # Evaluate each fitness metric
            for metric in self.fitness_metrics:
                metric_score = await self._evaluate_fitness_metric(genotype, metric)
                fitness_score += metric_score
            
            # Average across metrics
            genotype.fitness_score = fitness_score / len(self.fitness_metrics)
            
            # Add to fitness history
            self.fitness_history.append(genotype.fitness_score)
    
    async def _evaluate_fitness_metric(self, genotype: SwarmGenotype, metric: FitnessMetric) -> float:
        """Evaluate a specific fitness metric for a genotype."""
        if metric == FitnessMetric.COORDINATION_EFFICIENCY:
            # Based on coordination genes
            coord_score = statistics.mean([
                genotype.coordination_genes['formation_stability'],
                genotype.coordination_genes['path_optimization'],
                genotype.coordination_genes['obstacle_avoidance']
            ])
            return coord_score
        
        elif metric == FitnessMetric.MISSION_SUCCESS_RATE:
            # Based on problem-solving and decision-making genes
            mission_score = statistics.mean([
                genotype.intelligence_genes['problem_solving'],
                genotype.intelligence_genes['decision_speed'],
                genotype.behavior_genes['risk_tolerance']
            ])
            return mission_score
        
        elif metric == FitnessMetric.COLLECTIVE_INTELLIGENCE:
            # Based on communication and intelligence genes
            collective_score = statistics.mean([
                genotype.communication_genes['information_sharing'],
                genotype.intelligence_genes['reasoning_depth'],
                genotype.behavior_genes['cooperation_level']
            ])
            return collective_score
        
        elif metric == FitnessMetric.ADAPTATION_SPEED:
            # Based on adaptation genes
            adaptation_score = statistics.mean([
                genotype.adaptation_genes['learning_efficiency'],
                genotype.adaptation_genes['change_acceptance'],
                genotype.behavior_genes['learning_rate']
            ])
            return adaptation_score
        
        elif metric == FitnessMetric.ENERGY_OPTIMIZATION:
            # Based on efficiency-related genes
            energy_score = statistics.mean([
                1.0 - genotype.communication_genes['message_frequency'],  # Lower frequency = better
                genotype.coordination_genes['path_optimization'],
                genotype.intelligence_genes['decision_speed']
            ])
            return energy_score
        
        elif metric == FitnessMetric.INNOVATION_CAPACITY:
            # Based on innovation and exploration genes
            innovation_score = statistics.mean([
                genotype.adaptation_genes['innovation_drive'],
                genotype.behavior_genes['exploration_tendency'],
                genotype.adaptation_genes['mutation_tolerance']
            ])
            return innovation_score
        
        else:
            return random.uniform(0.3, 0.8)  # Default random score
    
    async def _selection_phase(self) -> List[str]:
        """Select parents for reproduction based on fitness."""
        # Sort population by fitness
        sorted_population = sorted(
            self.population.items(),
            key=lambda x: x[1].fitness_score,
            reverse=True
        )
        
        # Select top performers (elitism + tournament selection)
        num_selected = int(len(sorted_population) * self.selection_pressure)
        num_elite = max(1, num_selected // 4)  # Top 25% automatically selected
        
        selected_parents = []
        
        # Elite selection
        for i in range(num_elite):
            selected_parents.append(sorted_population[i][0])
        
        # Tournament selection for the rest
        for _ in range(num_selected - num_elite):
            tournament_size = 3
            tournament = random.sample(sorted_population, min(tournament_size, len(sorted_population)))
            winner = max(tournament, key=lambda x: x[1].fitness_score)
            selected_parents.append(winner[0])
        
        return selected_parents
    
    async def _reproduction_phase(self, selected_parents: List[str]):
        """Create new offspring through crossover and mutation."""
        new_population = {}
        
        # Keep elite individuals
        elite_count = max(1, len(selected_parents) // 4)
        for i, parent_id in enumerate(selected_parents[:elite_count]):
            new_population[f"gen{self.evolution_state.current_generation + 1}_elite_{i}"] = copy.deepcopy(self.population[parent_id])
        
        # Generate offspring
        offspring_count = 0
        while len(new_population) < self.current_population_size:
            # Select two parents
            parent1_id = random.choice(selected_parents)
            parent2_id = random.choice(selected_parents)
            
            if parent1_id != parent2_id and random.random() < self.crossover_rate:
                # Crossover
                offspring = await self._crossover(
                    self.population[parent1_id],
                    self.population[parent2_id]
                )
                self.evolution_metrics['successful_crossovers'] += 1
            else:
                # Clone with mutation
                offspring = copy.deepcopy(self.population[parent1_id])
            
            # Mutation
            if random.random() < self.mutation_rate:
                await self._mutate(offspring)
                self.evolution_metrics['successful_mutations'] += 1
            
            # Add to new population
            offspring_id = f"gen{self.evolution_state.current_generation + 1}_offspring_{offspring_count}"
            offspring.generation = self.evolution_state.current_generation + 1
            offspring.parent_ids = [parent1_id, parent2_id] if parent1_id != parent2_id else [parent1_id]
            new_population[offspring_id] = offspring
            offspring_count += 1
        
        # Replace population
        self.population = new_population
    
    async def _crossover(self, parent1: SwarmGenotype, parent2: SwarmGenotype) -> SwarmGenotype:
        """Create offspring through genetic crossover."""
        offspring = SwarmGenotype(
            coordination_genes={},
            behavior_genes={},
            communication_genes={},
            intelligence_genes={},
            adaptation_genes={}
        )
        
        # Uniform crossover for each gene group
        for gene_group_name in ['coordination_genes', 'behavior_genes', 'communication_genes', 
                               'intelligence_genes', 'adaptation_genes']:
            
            parent1_genes = getattr(parent1, gene_group_name)
            parent2_genes = getattr(parent2, gene_group_name)
            offspring_genes = {}
            
            for gene_name in parent1_genes:
                if random.random() < 0.5:
                    offspring_genes[gene_name] = parent1_genes[gene_name]
                else:
                    offspring_genes[gene_name] = parent2_genes[gene_name]
            
            setattr(offspring, gene_group_name, offspring_genes)
        
        return offspring
    
    async def _mutate(self, genotype: SwarmGenotype):
        """Apply mutation to a genotype."""
        mutation_strength = 0.1
        
        # Mutate each gene group
        for gene_group_name in ['coordination_genes', 'behavior_genes', 'communication_genes',
                               'intelligence_genes', 'adaptation_genes']:
            
            gene_group = getattr(genotype, gene_group_name)
            
            for gene_name, gene_value in gene_group.items():
                if random.random() < genotype.mutation_rate:
                    # Gaussian mutation
                    mutation = random.gauss(0, mutation_strength)
                    new_value = max(0.0, min(1.0, gene_value + mutation))
                    gene_group[gene_name] = new_value
    
    async def _cultural_evolution_phase(self):
        """Process cultural evolution (learning and knowledge transfer)."""
        # Extract successful behaviors from high-fitness individuals
        high_fitness_individuals = [
            genotype for genotype in self.population.values()
            if genotype.fitness_score > self.evolution_state.average_fitness
        ]
        
        if not high_fitness_individuals:
            return
        
        # Update collective knowledge
        for individual in high_fitness_individuals:
            for gene_group_name in ['coordination_genes', 'behavior_genes', 'communication_genes']:
                gene_group = getattr(individual, gene_group_name)
                
                for gene_name, gene_value in gene_group.items():
                    knowledge_key = f"{gene_group_name}.{gene_name}"
                    if knowledge_key in self.collective_knowledge:
                        # Average with existing knowledge
                        self.collective_knowledge[knowledge_key] = (
                            self.collective_knowledge[knowledge_key] + gene_value
                        ) / 2.0
                    else:
                        self.collective_knowledge[knowledge_key] = gene_value
        
        # Cultural transmission to population
        for genotype in self.population.values():
            if random.random() < 0.1:  # 10% chance of cultural learning
                # Learn from collective knowledge
                for knowledge_key, knowledge_value in self.collective_knowledge.items():
                    if '.' in knowledge_key:
                        gene_group_name, gene_name = knowledge_key.split('.', 1)
                        gene_group = getattr(genotype, gene_group_name, {})
                        if gene_name in gene_group:
                            # Blend with cultural knowledge
                            current_value = gene_group[gene_name]
                            blend_factor = 0.1
                            gene_group[gene_name] = (
                                current_value * (1 - blend_factor) + 
                                knowledge_value * blend_factor
                            )
                
                self.evolution_metrics['cultural_transmission_events'] += 1
    
    async def _neural_evolution_phase(self):
        """Evolve neural network architectures for enhanced intelligence."""
        if not self.neural_architectures:
            # Initialize basic architectures
            for i in range(5):
                arch_id = f"neural_arch_{i}"
                self.neural_architectures[arch_id] = {
                    'layers': random.randint(2, 6),
                    'neurons_per_layer': random.randint(16, 128),
                    'activation': random.choice(['relu', 'tanh', 'sigmoid']),
                    'learning_rate': random.uniform(0.001, 0.1),
                    'fitness': random.uniform(0.3, 0.7)
                }
        
        # Evolve architectures
        for arch_id, architecture in self.neural_architectures.items():
            if random.random() < 0.2:  # 20% chance of mutation
                if random.random() < 0.5:
                    # Structural mutation
                    architecture['layers'] = max(2, min(8, 
                        architecture['layers'] + random.choice([-1, 1])))
                    architecture['neurons_per_layer'] = max(8, min(256,
                        architecture['neurons_per_layer'] + random.choice([-16, 16])))
                else:
                    # Parameter mutation
                    architecture['learning_rate'] *= random.uniform(0.8, 1.2)
                    architecture['learning_rate'] = max(0.0001, min(0.5, architecture['learning_rate']))
                
                # Update fitness (simplified)
                complexity_bonus = architecture['layers'] * architecture['neurons_per_layer'] / 1000.0
                architecture['fitness'] = min(1.0, random.uniform(0.4, 0.9) + complexity_bonus)
                
                self.evolution_metrics['neural_architecture_improvements'] += 1
    
    async def _update_evolution_state(self):
        """Update the evolution state with current population statistics."""
        if not self.population:
            return
        
        # Calculate fitness statistics
        fitness_scores = [genotype.fitness_score for genotype in self.population.values()]
        self.evolution_state.average_fitness = statistics.mean(fitness_scores)
        self.evolution_state.best_fitness = max(fitness_scores)
        
        # Calculate genetic diversity (simplified)
        # Count unique gene combinations
        gene_signatures = set()
        for genotype in self.population.values():
            signature = tuple(sorted([
                round(v, 2) for gene_group in [
                    genotype.coordination_genes, genotype.behavior_genes,
                    genotype.communication_genes, genotype.intelligence_genes
                ] for v in gene_group.values()
            ]))
            gene_signatures.add(signature)
        
        self.evolution_state.genetic_diversity = len(gene_signatures) / len(self.population)
        
        # Calculate evolution rate
        if len(self.fitness_history) > 10:
            recent_improvement = self.evolution_state.best_fitness - self.fitness_history[-10]
            self.evolution_state.evolution_rate = max(0.0, recent_improvement / 10.0)
        
        # Update innovation index
        recent_innovations = sum(1 for event in self.evolution_history[-20:] 
                               if event.event_type == 'innovation')
        self.evolution_state.innovation_index = recent_innovations / 20.0
    
    async def _detect_innovations(self):
        """Detect and record innovation events."""
        # Innovation detection based on fitness jumps and genetic novelty
        if len(self.fitness_history) < 5:
            return
        
        recent_fitness = list(self.fitness_history)[-5:]
        if len(recent_fitness) >= 2:
            fitness_jump = recent_fitness[-1] - recent_fitness[-2]
            
            if fitness_jump > 0.1:  # Significant fitness improvement
                # Check for genetic novelty
                best_individual = max(self.population.values(), key=lambda x: x.fitness_score)
                
                innovation_event = EvolutionEvent(
                    event_type='innovation',
                    generation=self.evolution_state.current_generation,
                    fitness_improvement=fitness_jump,
                    genetic_changes={
                        'coordination_improvements': {k: v for k, v in best_individual.coordination_genes.items() if v > 0.8},
                        'behavior_improvements': {k: v for k, v in best_individual.behavior_genes.items() if v > 0.8},
                        'intelligence_improvements': {k: v for k, v in best_individual.intelligence_genes.items() if v > 0.8}
                    }
                )
                
                self.evolution_history.append(innovation_event)
                self.evolution_metrics['innovation_events'] += 1
                
                print(f"🌟 Innovation detected in generation {self.evolution_state.current_generation}")
                print(f"   Fitness jump: +{fitness_jump:.3f}")
    
    async def inject_external_genes(self, external_genotype: Dict[str, Any]):
        """Inject external genetic material into the population."""
        # Create genotype from external data
        new_genotype = SwarmGenotype(
            coordination_genes=external_genotype.get('coordination_genes', {}),
            behavior_genes=external_genotype.get('behavior_genes', {}),
            communication_genes=external_genotype.get('communication_genes', {}),
            intelligence_genes=external_genotype.get('intelligence_genes', {}),
            adaptation_genes=external_genotype.get('adaptation_genes', {}),
            generation=self.evolution_state.current_generation
        )
        
        # Replace weakest individual
        weakest_id = min(self.population.keys(), 
                        key=lambda x: self.population[x].fitness_score)
        
        injection_id = f"gen{self.evolution_state.current_generation}_injection"
        self.population[injection_id] = new_genotype
        del self.population[weakest_id]
        
        print(f"🧬 Injected external genetic material")
    
    async def extract_best_genome(self) -> Dict[str, Any]:
        """Extract the genome of the best performing individual."""
        if not self.population:
            return {}
        
        best_individual = max(self.population.values(), key=lambda x: x.fitness_score)
        
        return {
            'coordination_genes': best_individual.coordination_genes.copy(),
            'behavior_genes': best_individual.behavior_genes.copy(),
            'communication_genes': best_individual.communication_genes.copy(),
            'intelligence_genes': best_individual.intelligence_genes.copy(),
            'adaptation_genes': best_individual.adaptation_genes.copy(),
            'fitness_score': best_individual.fitness_score,
            'generation': best_individual.generation,
            'parent_lineage': best_individual.parent_ids
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive evolution status report."""
        population_stats = {}
        if self.population:
            fitness_scores = [g.fitness_score for g in self.population.values()]
            population_stats = {
                'size': len(self.population),
                'min_fitness': round(min(fitness_scores), 3),
                'max_fitness': round(max(fitness_scores), 3),
                'avg_fitness': round(statistics.mean(fitness_scores), 3),
                'fitness_std': round(statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0, 3)
            }
        
        return {
            'evolution_state': {
                'generation': self.evolution_state.current_generation,
                'population_size': self.evolution_state.population_size,
                'average_fitness': round(self.evolution_state.average_fitness, 3),
                'best_fitness': round(self.evolution_state.best_fitness, 3),
                'genetic_diversity': round(self.evolution_state.genetic_diversity, 3),
                'evolution_rate': round(self.evolution_state.evolution_rate, 4),
                'innovation_index': round(self.evolution_state.innovation_index, 3)
            },
            'population': population_stats,
            'cultural_evolution': {
                'collective_knowledge_size': len(self.collective_knowledge),
                'behavioral_templates': len(self.behavioral_templates)
            },
            'neural_evolution': {
                'architectures': len(self.neural_architectures),
                'best_neural_fitness': round(max([arch['fitness'] for arch in self.neural_architectures.values()]) if self.neural_architectures else 0.0, 3)
            },
            'metrics': self.evolution_metrics.copy(),
            'recent_innovations': len([e for e in self.evolution_history[-10:] if e.event_type == 'innovation']),
            'is_evolving': self._is_evolving
        }
    
    async def shutdown_evolution(self):
        """Gracefully shutdown the evolution system."""
        if not self._is_evolving:
            return
        
        print("🌙 Shutting down autonomous evolution...")
        self._is_evolving = False
        
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
        
        # Archive best genomes
        best_genome = await self.extract_best_genome()
        print(f"🏆 Best genome achieved fitness: {best_genome.get('fitness_score', 0.0):.3f}")
        
        print("💤 Evolution system dormant")