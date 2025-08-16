"""Adaptive Memory System for Neuromorphic Swarm Coordination.

Bio-inspired memory formation, consolidation, and retrieval for
long-term learning and adaptation in drone swarms.
"""

import asyncio
import math
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

class MemoryType(Enum):
    EPISODIC = "episodic"          # Specific events and experiences
    SEMANTIC = "semantic"          # General knowledge and facts
    PROCEDURAL = "procedural"      # Skills and motor patterns
    WORKING = "working"            # Temporary storage
    SPATIAL = "spatial"            # Location and navigation

class ConsolidationState(Enum):
    ENCODING = "encoding"          # Initial formation
    LABILE = "labile"             # Unstable, can be modified
    CONSOLIDATING = "consolidating" # Becoming stable
    CONSOLIDATED = "consolidated"   # Stable long-term
    RECONSOLIDATING = "reconsolidating" # Reactivated for updating

@dataclass
class PlasticityRule:
    """Synaptic plasticity rule for memory formation."""
    rule_type: str
    learning_rate: float
    decay_rate: float
    threshold: float
    metaplasticity: bool = False
    
    def apply_rule(self, pre_activity: float, post_activity: float, 
                   current_weight: float, time_delta: float) -> float:
        """Apply plasticity rule to update synaptic weight."""
        
        if self.rule_type == "hebbian":
            # Hebbian: "cells that fire together, wire together"
            weight_change = self.learning_rate * pre_activity * post_activity
            
        elif self.rule_type == "stdp":
            # Spike-timing dependent plasticity
            if abs(time_delta) < 0.02:  # 20ms window
                if time_delta > 0:  # Post after pre (LTP)
                    weight_change = self.learning_rate * math.exp(-time_delta / 0.01)
                else:  # Pre after post (LTD)
                    weight_change = -self.learning_rate * math.exp(time_delta / 0.01)
            else:
                weight_change = 0.0
                
        elif self.rule_type == "bcm":
            # BCM rule with sliding threshold
            theta = self.threshold * (post_activity ** 2)
            weight_change = self.learning_rate * pre_activity * post_activity * (post_activity - theta)
            
        elif self.rule_type == "anti_hebbian":
            # Anti-Hebbian for competition
            weight_change = -self.learning_rate * pre_activity * post_activity
            
        else:
            weight_change = 0.0
        
        # Apply weight decay
        decay = -self.decay_rate * current_weight
        
        # Update weight
        new_weight = current_weight + weight_change + decay
        
        # Apply metaplasticity if enabled
        if self.metaplasticity:
            # Adjust learning rate based on recent activity
            activity_factor = (pre_activity + post_activity) / 2.0
            if activity_factor > 0.8:
                self.learning_rate *= 0.99  # Reduce learning rate if too active
            elif activity_factor < 0.2:
                self.learning_rate *= 1.01  # Increase learning rate if inactive
        
        return max(-1.0, min(1.0, new_weight))  # Clip to bounds

@dataclass
class MemoryTrace:
    """Individual memory trace storing learned information."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    strength: float
    consolidation_state: ConsolidationState
    created_at: float
    last_accessed: float
    access_count: int
    associated_traces: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    
    def update_strength(self, reinforcement: float, decay_rate: float = 0.001):
        """Update memory strength based on usage and time."""
        
        # Time-based decay
        time_passed = time.time() - self.last_accessed
        decay = math.exp(-decay_rate * time_passed)
        
        # Apply reinforcement and decay
        self.strength = min(1.0, self.strength * decay + reinforcement)
        
        # Update access information
        self.last_accessed = time.time()
        self.access_count += 1

@dataclass
class MemoryConsolidation:
    """Memory consolidation process configuration."""
    consolidation_time: float = 3600.0  # 1 hour for full consolidation
    rehearsal_probability: float = 0.1   # Probability of spontaneous rehearsal
    interference_threshold: float = 0.8   # Threshold for memory interference
    consolidation_rate: float = 0.001    # Rate of consolidation process
    
    def get_consolidation_progress(self, memory_trace: MemoryTrace) -> float:
        """Calculate consolidation progress for memory trace."""
        
        time_elapsed = time.time() - memory_trace.created_at
        progress = min(1.0, time_elapsed / self.consolidation_time)
        
        # Factor in access frequency and importance
        access_factor = min(1.0, memory_trace.access_count / 10.0)
        importance_factor = memory_trace.importance_score
        
        # Combined progress
        consolidated_progress = progress * 0.6 + access_factor * 0.3 + importance_factor * 0.1
        
        return min(1.0, consolidated_progress)

class AdaptiveMemory:
    """Adaptive memory system for neuromorphic swarm coordination."""
    
    def __init__(self, 
                 memory_capacity: int = 10000,
                 consolidation_config: MemoryConsolidation = None):
        self.memory_capacity = memory_capacity
        self.consolidation_config = consolidation_config or MemoryConsolidation()
        
        # Memory storage
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.memory_associations: Dict[str, List[str]] = defaultdict(list)
        self.working_memory: deque = deque(maxlen=50)  # Limited capacity
        
        # Plasticity rules for different memory types
        self.plasticity_rules = {
            MemoryType.EPISODIC: PlasticityRule("hebbian", 0.01, 0.001, 0.5),
            MemoryType.SEMANTIC: PlasticityRule("bcm", 0.005, 0.0005, 0.7),
            MemoryType.PROCEDURAL: PlasticityRule("stdp", 0.02, 0.002, 0.3),
            MemoryType.WORKING: PlasticityRule("anti_hebbian", 0.05, 0.1, 0.8),
            MemoryType.SPATIAL: PlasticityRule("hebbian", 0.015, 0.001, 0.4, True)
        }
        
        # Memory statistics
        self.memory_stats = {
            'total_memories': 0,
            'consolidated_memories': 0,
            'forgotten_memories': 0,
            'retrieval_successes': 0,
            'retrieval_failures': 0,
            'consolidation_events': 0
        }
        
        # Active processes
        self.consolidation_active = True
        asyncio.create_task(self._background_consolidation())
    
    async def encode_memory(self, 
                          content: Dict[str, Any],
                          memory_type: MemoryType,
                          importance: float = 0.5,
                          emotional_valence: float = 0.0) -> str:
        """Encode new memory into the system."""
        
        memory_id = f"{memory_type.value}_{int(time.time() * 1000000)}"
        
        # Create memory trace
        memory_trace = MemoryTrace(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            strength=1.0,
            consolidation_state=ConsolidationState.ENCODING,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            importance_score=importance,
            emotional_valence=emotional_valence
        )
        
        # Check for memory capacity
        if len(self.memory_traces) >= self.memory_capacity:
            await self._forget_weak_memories()
        
        # Store memory
        self.memory_traces[memory_id] = memory_trace
        
        # Add to working memory if recent
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(memory_id)
        
        # Create associations with existing memories
        await self._create_associations(memory_trace)
        
        # Update statistics
        self.memory_stats['total_memories'] += 1
        
        return memory_id
    
    async def retrieve_memory(self, 
                            query: Dict[str, Any],
                            memory_type: Optional[MemoryType] = None,
                            threshold: float = 0.3) -> List[MemoryTrace]:
        """Retrieve memories matching the query."""
        
        retrieved_memories = []
        
        for memory_trace in self.memory_traces.values():
            # Filter by type if specified
            if memory_type and memory_trace.memory_type != memory_type:
                continue
            
            # Calculate similarity
            similarity = await self._calculate_similarity(query, memory_trace.content)
            
            if similarity >= threshold:
                # Update memory strength (retrieval strengthens memory)
                memory_trace.update_strength(similarity * 0.1)
                
                # Check for reconsolidation
                if memory_trace.consolidation_state == ConsolidationState.CONSOLIDATED:
                    memory_trace.consolidation_state = ConsolidationState.RECONSOLIDATING
                
                retrieved_memories.append((memory_trace, similarity))
        
        # Sort by similarity and memory strength
        retrieved_memories.sort(key=lambda x: x[1] * x[0].strength, reverse=True)
        
        # Update statistics
        if retrieved_memories:
            self.memory_stats['retrieval_successes'] += 1
        else:
            self.memory_stats['retrieval_failures'] += 1
        
        return [trace for trace, _ in retrieved_memories]
    
    async def consolidate_memory(self, memory_id: str) -> bool:
        """Explicitly consolidate a specific memory."""
        
        if memory_id not in self.memory_traces:
            return False
        
        memory_trace = self.memory_traces[memory_id]
        
        # Update consolidation state based on progress
        progress = self.consolidation_config.get_consolidation_progress(memory_trace)
        
        if progress >= 1.0:
            memory_trace.consolidation_state = ConsolidationState.CONSOLIDATED
            self.memory_stats['consolidation_events'] += 1
            self.memory_stats['consolidated_memories'] += 1
            return True
        elif progress >= 0.5:
            memory_trace.consolidation_state = ConsolidationState.CONSOLIDATING
        else:
            memory_trace.consolidation_state = ConsolidationState.LABILE
        
        return False
    
    async def reinforce_memory(self, 
                             memory_id: str, 
                             reinforcement_strength: float = 0.1) -> bool:
        """Reinforce existing memory to strengthen it."""
        
        if memory_id not in self.memory_traces:
            return False
        
        memory_trace = self.memory_traces[memory_id]
        
        # Apply reinforcement
        memory_trace.update_strength(reinforcement_strength)
        
        # Apply plasticity rule
        plasticity_rule = self.plasticity_rules[memory_trace.memory_type]
        
        # Simulate pre and post synaptic activity
        pre_activity = memory_trace.strength
        post_activity = reinforcement_strength
        time_delta = 0.0  # Immediate reinforcement
        
        # Update memory strength using plasticity rule
        new_strength = plasticity_rule.apply_rule(
            pre_activity, post_activity, memory_trace.strength, time_delta
        )
        
        memory_trace.strength = new_strength
        
        return True
    
    async def forget_memory(self, memory_id: str) -> bool:
        """Forget (remove) a specific memory."""
        
        if memory_id not in self.memory_traces:
            return False
        
        memory_trace = self.memory_traces[memory_id]
        
        # Remove associations
        for associated_id in memory_trace.associated_traces:
            if associated_id in self.memory_associations:
                if memory_id in self.memory_associations[associated_id]:
                    self.memory_associations[associated_id].remove(memory_id)
        
        # Remove from working memory if present
        if memory_id in self.working_memory:
            working_memory_list = list(self.working_memory)
            working_memory_list.remove(memory_id)
            self.working_memory = deque(working_memory_list, maxlen=50)
        
        # Remove memory trace
        del self.memory_traces[memory_id]
        
        # Clean up associations
        if memory_id in self.memory_associations:
            del self.memory_associations[memory_id]
        
        # Update statistics
        self.memory_stats['forgotten_memories'] += 1
        
        return True
    
    async def _background_consolidation(self):
        """Background process for memory consolidation."""
        
        while self.consolidation_active:
            try:
                # Process a batch of memories for consolidation
                consolidation_batch = list(self.memory_traces.keys())[:100]
                
                for memory_id in consolidation_batch:
                    if memory_id in self.memory_traces:
                        memory_trace = self.memory_traces[memory_id]
                        
                        # Skip already consolidated memories
                        if memory_trace.consolidation_state == ConsolidationState.CONSOLIDATED:
                            continue
                        
                        # Apply consolidation
                        await self.consolidate_memory(memory_id)
                        
                        # Random rehearsal
                        if random.random() < self.consolidation_config.rehearsal_probability:
                            await self._rehearse_memory(memory_trace)
                
                # Sleep between consolidation cycles
                await asyncio.sleep(1.0)
                
            except Exception as e:
                # Continue consolidation even if individual memories fail
                await asyncio.sleep(0.1)
    
    async def _create_associations(self, new_memory: MemoryTrace):
        """Create associations between new memory and existing memories."""
        
        max_associations = 10
        associations_created = 0
        
        for existing_id, existing_memory in self.memory_traces.items():
            if associations_created >= max_associations:
                break
            
            if existing_id == new_memory.memory_id:
                continue
            
            # Calculate association strength
            similarity = await self._calculate_similarity(
                new_memory.content, existing_memory.content
            )
            
            # Create association if similar enough
            if similarity > 0.5:
                new_memory.associated_traces.append(existing_id)
                existing_memory.associated_traces.append(new_memory.memory_id)
                
                # Add to association graph
                self.memory_associations[new_memory.memory_id].append(existing_id)
                self.memory_associations[existing_id].append(new_memory.memory_id)
                
                associations_created += 1
    
    async def _calculate_similarity(self, 
                                  content1: Dict[str, Any], 
                                  content2: Dict[str, Any]) -> float:
        """Calculate similarity between two memory contents."""
        
        if not content1 or not content2:
            return 0.0
        
        # Get common keys
        common_keys = set(content1.keys()) & set(content2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            val1 = content1[key]
            val2 = content2[key]
            
            # Calculate similarity based on value type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
                
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simplified)
                if val1 == val2:
                    similarity = 1.0
                else:
                    # Simple character-based similarity
                    common_chars = sum(1 for c1, c2 in zip(val1, val2) if c1 == c2)
                    max_len = max(len(val1), len(val2), 1)
                    similarity = common_chars / max_len
                    
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                # Sequence similarity
                common_elements = len(set(val1) & set(val2))
                total_elements = len(set(val1) | set(val2))
                similarity = common_elements / max(total_elements, 1)
                
            else:
                # Default similarity for other types
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarity_scores.append(similarity)
        
        # Return average similarity
        return sum(similarity_scores) / len(similarity_scores)
    
    async def _rehearse_memory(self, memory_trace: MemoryTrace):
        """Rehearse memory to strengthen consolidation."""
        
        # Simulate memory replay/rehearsal
        memory_trace.update_strength(0.05)  # Small strengthening
        
        # Create new associations during rehearsal
        if len(memory_trace.associated_traces) < 5:
            # Find similar memories for new associations
            similar_memories = await self.retrieve_memory(
                memory_trace.content,
                threshold=0.4
            )
            
            for similar_memory in similar_memories[:2]:  # Limit new associations
                if similar_memory.memory_id not in memory_trace.associated_traces:
                    memory_trace.associated_traces.append(similar_memory.memory_id)
                    similar_memory.associated_traces.append(memory_trace.memory_id)
    
    async def _forget_weak_memories(self):
        """Remove weak memories to make space for new ones."""
        
        # Find memories with low strength that are not important
        weak_memories = []
        
        for memory_id, memory_trace in self.memory_traces.items():
            # Calculate forgetting score (higher = more likely to forget)
            time_unused = time.time() - memory_trace.last_accessed
            strength_penalty = 1.0 - memory_trace.strength
            importance_penalty = 1.0 - memory_trace.importance_score
            
            forgetting_score = (
                time_unused / 3600.0 +  # Hours unused
                strength_penalty * 2.0 +
                importance_penalty * 1.5
            )
            
            # Don't forget highly important or recently consolidated memories
            if (memory_trace.importance_score < 0.3 and 
                memory_trace.consolidation_state != ConsolidationState.CONSOLIDATED):
                weak_memories.append((memory_id, forgetting_score))
        
        # Sort by forgetting score and remove weakest memories
        weak_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Remove up to 10% of memories or at least 100 to make space
        removal_count = max(100, len(self.memory_traces) // 10)
        
        for memory_id, _ in weak_memories[:removal_count]:
            await self.forget_memory(memory_id)
    
    async def get_memory_network(self) -> Dict[str, List[str]]:
        """Get the associative network of memories."""
        
        network = {}
        
        for memory_id, memory_trace in self.memory_traces.items():
            network[memory_id] = {
                'type': memory_trace.memory_type.value,
                'strength': memory_trace.strength,
                'consolidation': memory_trace.consolidation_state.value,
                'associations': memory_trace.associated_traces.copy(),
                'importance': memory_trace.importance_score
            }
        
        return network
    
    async def search_associative_memories(self, 
                                        start_memory_id: str,
                                        max_depth: int = 3) -> List[MemoryTrace]:
        """Search for memories through associative connections."""
        
        if start_memory_id not in self.memory_traces:
            return []
        
        visited = set()
        result_memories = []
        to_visit = [(start_memory_id, 0)]  # (memory_id, depth)
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self.memory_traces:
                memory_trace = self.memory_traces[current_id]
                result_memories.append(memory_trace)
                
                # Add associated memories to search
                for associated_id in memory_trace.associated_traces:
                    if associated_id not in visited:
                        to_visit.append((associated_id, depth + 1))
        
        return result_memories
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        
        # Calculate type distribution
        type_distribution = defaultdict(int)
        consolidation_distribution = defaultdict(int)
        
        total_strength = 0.0
        avg_importance = 0.0
        
        for memory_trace in self.memory_traces.values():
            type_distribution[memory_trace.memory_type.value] += 1
            consolidation_distribution[memory_trace.consolidation_state.value] += 1
            total_strength += memory_trace.strength
            avg_importance += memory_trace.importance_score
        
        num_memories = len(self.memory_traces)
        
        return {
            'memory_counts': {
                'total_memories': num_memories,
                'working_memory_size': len(self.working_memory),
                'memory_capacity': self.memory_capacity,
                'capacity_utilization': num_memories / self.memory_capacity
            },
            'memory_distribution': {
                'by_type': dict(type_distribution),
                'by_consolidation': dict(consolidation_distribution)
            },
            'memory_quality': {
                'average_strength': total_strength / max(num_memories, 1),
                'average_importance': avg_importance / max(num_memories, 1),
                'total_associations': sum(len(trace.associated_traces) 
                                        for trace in self.memory_traces.values())
            },
            'processing_statistics': self.memory_stats.copy(),
            'consolidation_config': {
                'consolidation_time': self.consolidation_config.consolidation_time,
                'rehearsal_probability': self.consolidation_config.rehearsal_probability,
                'interference_threshold': self.consolidation_config.interference_threshold,
                'consolidation_rate': self.consolidation_config.consolidation_rate
            }
        }
    
    async def optimize_memory_system(self, target: str = "balanced"):
        """Optimize memory system for specific performance target."""
        
        if target == "capacity":
            await self._optimize_for_capacity()
        elif target == "speed":
            await self._optimize_for_speed()
        elif target == "retention":
            await self._optimize_for_retention()
        else:
            await self._optimize_balanced()
    
    async def _optimize_for_capacity(self):
        """Optimize for maximum memory capacity."""
        
        # Increase forgetting of weak memories
        await self._forget_weak_memories()
        
        # Reduce association overhead
        for memory_trace in self.memory_traces.values():
            if len(memory_trace.associated_traces) > 5:
                # Keep only strongest associations
                memory_trace.associated_traces = memory_trace.associated_traces[:5]
    
    async def _optimize_for_speed(self):
        """Optimize for faster memory retrieval."""
        
        # Increase consolidation rate
        self.consolidation_config.consolidation_rate *= 1.5
        
        # Reduce working memory size for faster access
        self.working_memory = deque(list(self.working_memory)[:25], maxlen=25)
    
    async def _optimize_for_retention(self):
        """Optimize for better memory retention."""
        
        # Increase rehearsal probability
        self.consolidation_config.rehearsal_probability *= 1.5
        
        # Strengthen all important memories
        for memory_trace in self.memory_traces.values():
            if memory_trace.importance_score > 0.7:
                memory_trace.update_strength(0.1)
    
    async def _optimize_balanced(self):
        """Apply balanced optimization."""
        
        # Moderate improvements across all aspects
        await self._forget_weak_memories()
        
        self.consolidation_config.consolidation_rate *= 1.2
        self.consolidation_config.rehearsal_probability *= 1.2
        
        # Strengthen moderately important memories
        for memory_trace in self.memory_traces.values():
            if memory_trace.importance_score > 0.5:
                memory_trace.update_strength(0.05)
    
    def stop_consolidation(self):
        """Stop background consolidation process."""
        self.consolidation_active = False