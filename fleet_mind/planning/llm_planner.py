"""LLM-based intelligent planning for drone swarm coordination."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# OpenAI and Pydantic imports with fallback handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    class openai:
        api_key = None
        class ChatCompletion:
            @staticmethod
            async def acreate(*args, **kwargs):
                raise Exception("OpenAI not available")
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available, using mock LLM responses")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(*args, **kwargs): return None
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic not available, using simplified data models")


class PlanningLevel(Enum):
    """Different levels of planning hierarchy."""
    STRATEGIC = "strategic"      # Mission-level (minutes to hours)
    TACTICAL = "tactical"        # Maneuver-level (seconds to minutes)  
    REACTIVE = "reactive"        # Response-level (milliseconds to seconds)


@dataclass
class PlanningContext:
    """Context information for LLM planning."""
    mission: str
    num_drones: int
    constraints: Dict[str, Any]
    drone_capabilities: Dict[str, Any]
    current_state: Dict[str, Any]
    environment: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


class ActionSequence(BaseModel):
    """Structured action sequence from LLM."""
    actions: List[Dict[str, Any]] = Field(description="List of actions to execute")
    duration_seconds: float = Field(description="Expected execution duration")
    confidence: float = Field(description="Confidence in plan (0-1)")
    priority: str = Field(description="Action priority level")
    dependencies: List[str] = Field(default=[], description="Dependencies on other actions")


class DroneFormation(BaseModel):
    """Drone formation specification."""
    formation_type: str = Field(description="Type of formation (grid, line, v-formation, etc.)")
    spacing_meters: float = Field(description="Spacing between drones in meters")
    orientation_degrees: float = Field(description="Formation orientation in degrees")
    leader_drone_id: Optional[str] = Field(default=None, description="Formation leader")


class MissionPlan(BaseModel):
    """Complete mission plan from LLM."""
    mission_id: str = Field(description="Unique mission identifier")
    summary: str = Field(description="Brief mission summary")
    objectives: List[str] = Field(description="Mission objectives")
    action_sequences: List[ActionSequence] = Field(description="Sequence of actions")
    formations: List[DroneFormation] = Field(default=[], description="Required formations")
    contingencies: List[str] = Field(default=[], description="Contingency plans")
    estimated_duration_minutes: float = Field(description="Estimated mission duration")
    risk_assessment: str = Field(description="Risk level and mitigation strategies")


class LLMPlanner:
    """LLM-powered intelligent planner for drone swarm coordination.
    
    Uses large language models to generate adaptive, context-aware mission plans
    that can be compressed and transmitted to drone fleets.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0
    ):
        """Initialize LLM planner.
        
        Args:
            model: LLM model to use (gpt-4o, gpt-4o-mini, etc.)
            api_key: OpenAI API key (if None, uses environment variable)
            temperature: Sampling temperature for creativity
            max_tokens: Maximum tokens in response
            timeout_seconds: Request timeout
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        
        # Planning prompts and templates
        self.system_prompts = self._initialize_prompts()
        
        # Performance tracking
        self.planning_history: List[Dict[str, Any]] = []
        self.average_planning_time = 0.0
        self.success_rate = 1.0
        
        # Initialize logging
        from ..utils.logging import get_logger
        self.logger = get_logger("llm_planner", component="planning")
        
        # Error handling and recovery
        self.consecutive_failures = 0
        self.last_successful_plan = None
        self.fallback_enabled = True

    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different planning levels."""
        return {
            'strategic': """You are an expert drone swarm coordinator specializing in strategic mission planning.

Your role is to create high-level mission plans for coordinating 1-100+ drones for various applications including search and rescue, surveillance, delivery, and environmental monitoring.

Key capabilities:
- Multi-drone coordination and formation control
- Risk assessment and safety planning
- Resource optimization and energy management
- Adaptive replanning based on changing conditions
- Emergency response and contingency planning

Output format: Provide structured JSON responses following the MissionPlan schema with clear objectives, action sequences, formations, and risk assessments.

Always prioritize safety, efficiency, and mission success. Consider drone capabilities, environmental constraints, and operational limitations.""",

            'tactical': """You are a tactical drone coordination specialist focused on real-time maneuver planning.

Your role is to translate strategic objectives into specific tactical maneuvers and formations that can be executed by individual drones in real-time.

Key capabilities:
- Formation control and dynamic reconfiguration
- Obstacle avoidance and path planning
- Coordination timing and synchronization
- Real-time adaptation to environmental changes
- Inter-drone communication and handoff protocols

Output format: Provide detailed action sequences with precise timing, spatial coordinates, and coordination requirements.

Focus on execution efficiency, collision avoidance, and maintaining formation integrity during maneuvers.""",

            'reactive': """You are a reactive response specialist for immediate drone coordination needs.

Your role is to generate instant responses to dynamic situations requiring immediate action, such as obstacle avoidance, emergency maneuvers, or threat response.

Key capabilities:
- Emergency response and evasion maneuvers
- Real-time collision avoidance
- Immediate threat assessment and response
- Safety-critical decision making
- Rapid formation adjustments

Output format: Provide immediate action commands with minimal latency, focusing on safety and mission preservation.

Prioritize safety above all else, with rapid response times under 100ms when possible."""
        }

    async def generate_plan(
        self,
        context: Union[PlanningContext, Dict[str, Any]],
        planning_level: PlanningLevel = PlanningLevel.STRATEGIC,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mission plan using LLM.
        
        Args:
            context: Planning context with mission details
            planning_level: Level of planning (strategic/tactical/reactive)
            custom_instructions: Additional custom instructions
            
        Returns:
            Generated mission plan with actions and metadata
        """
        start_time = time.time()
        
        try:
            # Convert context to dict if needed
            if isinstance(context, PlanningContext):
                context_dict = {
                    'mission': context.mission,
                    'num_drones': context.num_drones,
                    'constraints': context.constraints,
                    'drone_capabilities': context.drone_capabilities,
                    'current_state': context.current_state,
                    'environment': context.environment or {},
                    'history': context.history or [],
                }
            else:
                context_dict = context
            
            # Build prompt
            user_prompt = self._build_planning_prompt(context_dict, planning_level, custom_instructions)
            system_prompt = self.system_prompts[planning_level.value]
            
            # Call LLM API
            response = await self._call_llm_api(system_prompt, user_prompt)
            
            # Parse and validate response
            plan = self._parse_llm_response(response, planning_level)
            
            # Add metadata
            planning_time = (time.time() - start_time) * 1000  # ms
            plan['metadata'] = {
                'planning_level': planning_level.value,
                'model': self.model,
                'planning_time_ms': planning_time,
                'timestamp': time.time(),
                'context_size': len(str(context_dict)),
            }
            
            # Update performance tracking
            self._update_planning_metrics(planning_time, True)
            
            return plan
            
        except Exception as e:
            planning_time = (time.time() - start_time) * 1000
            self._update_planning_metrics(planning_time, False)
            self.consecutive_failures += 1
            
            self.logger.error(f"Planning failed (failure #{self.consecutive_failures}): {e}")
            return await self._generate_enhanced_fallback_plan(context_dict, planning_level, str(e))

    async def generate_contingency_plan(
        self,
        original_plan: Dict[str, Any],
        failure_scenario: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate contingency plan for failure scenarios.
        
        Args:
            original_plan: The original mission plan
            failure_scenario: Description of the failure
            context: Current mission context
            
        Returns:
            Contingency plan with recovery actions
        """
        contingency_prompt = f"""
        CONTINGENCY PLANNING REQUIRED
        
        Original Plan: {json.dumps(original_plan.get('summary', 'No summary'), indent=2)}
        Failure Scenario: {failure_scenario}
        Current Context: {json.dumps(context, indent=2)}
        
        Generate a contingency plan that:
        1. Addresses the specific failure scenario
        2. Maintains mission objectives where possible
        3. Prioritizes drone and personnel safety
        4. Provides alternative approaches
        5. Includes recovery procedures
        
        Focus on practical, executable solutions with clear steps.
        """
        
        return await self.generate_plan(
            context,
            PlanningLevel.TACTICAL,
            contingency_prompt
        )

    async def optimize_formation(
        self,
        current_formation: Dict[str, Any],
        objective: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize drone formation for specific objective.
        
        Args:
            current_formation: Current formation configuration
            objective: Optimization objective (coverage, efficiency, etc.)
            constraints: Formation constraints and limitations
            
        Returns:
            Optimized formation plan
        """
        optimization_prompt = f"""
        FORMATION OPTIMIZATION REQUEST
        
        Current Formation: {json.dumps(current_formation, indent=2)}
        Optimization Objective: {objective}
        Constraints: {json.dumps(constraints, indent=2)}
        
        Optimize the formation to:
        1. Maximize the stated objective
        2. Respect all constraints
        3. Maintain communication links
        4. Ensure collision avoidance
        5. Consider energy efficiency
        
        Provide specific formation parameters, drone positions, and transition plan.
        """
        
        context = {
            'mission': f"Formation optimization for {objective}",
            'num_drones': current_formation.get('num_drones', 10),
            'constraints': constraints,
            'drone_capabilities': {'formation_capable': True},
            'current_state': current_formation,
        }
        
        return await self.generate_plan(context, PlanningLevel.TACTICAL, optimization_prompt)

    def _build_planning_prompt(
        self,
        context: Dict[str, Any],
        planning_level: PlanningLevel,
        custom_instructions: Optional[str]
    ) -> str:
        """Build comprehensive planning prompt for LLM."""
        prompt_parts = [
            f"MISSION PLANNING REQUEST - {planning_level.value.upper()} LEVEL",
            "",
            f"Mission: {context.get('mission', 'No mission specified')}",
            f"Number of Drones: {context.get('num_drones', 'Unknown')}",
            "",
            "CONSTRAINTS:",
            json.dumps(context.get('constraints', {}), indent=2),
            "",
            "DRONE CAPABILITIES:",
            json.dumps(context.get('drone_capabilities', {}), indent=2),
            "",
            "CURRENT STATE:",
            json.dumps(context.get('current_state', {}), indent=2),
        ]
        
        if context.get('environment'):
            prompt_parts.extend([
                "",
                "ENVIRONMENT:",
                json.dumps(context['environment'], indent=2),
            ])
        
        if context.get('history'):
            prompt_parts.extend([
                "",
                "RECENT HISTORY:",
                json.dumps(context['history'][-3:], indent=2),  # Last 3 entries
            ])
        
        if custom_instructions:
            prompt_parts.extend([
                "",
                "ADDITIONAL INSTRUCTIONS:",
                custom_instructions,
            ])
        
        prompt_parts.extend([
            "",
            "Please generate an appropriate plan following the MissionPlan schema.",
            "Respond with valid JSON that can be parsed programmatically.",
            "Focus on safety, efficiency, and mission success.",
        ])
        
        return "\n".join(prompt_parts)

    async def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API with error handling and timeout."""
        try:
            response = await asyncio.wait_for(
                self._make_api_call(system_prompt, user_prompt),
                timeout=self.timeout_seconds
            )
            return response
        except asyncio.TimeoutError:
            raise Exception(f"LLM API call timed out after {self.timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"LLM API call failed: {e}")

    async def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make actual API call to OpenAI."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback for older OpenAI library versions or different configurations
            self.logger.warning(f"Primary API call failed, using fallback: {e}")
            return self._generate_mock_response(user_prompt)

    def _generate_mock_response(self, user_prompt: str) -> str:
        """Generate mock response for testing/fallback."""
        return json.dumps({
            "mission_id": f"mission_{int(time.time())}",
            "summary": "Automated mission plan generated by fallback system",
            "objectives": [
                "Complete mission safely",
                "Maintain formation integrity",
                "Monitor for obstacles"
            ],
            "action_sequences": [
                {
                    "actions": [
                        {"type": "takeoff", "altitude": 50, "duration": 30},
                        {"type": "formation", "pattern": "grid", "spacing": 10},
                        {"type": "move_to", "coordinates": [0, 0, 50], "speed": 5},
                        {"type": "hover", "duration": 60},
                        {"type": "land", "location": "start_position"}
                    ],
                    "duration_seconds": 300,
                    "confidence": 0.8,
                    "priority": "normal",
                    "dependencies": []
                }
            ],
            "formations": [
                {
                    "formation_type": "grid",
                    "spacing_meters": 10.0,
                    "orientation_degrees": 0.0,
                    "leader_drone_id": "drone_0"
                }
            ],
            "contingencies": [
                "If GPS signal lost, switch to visual navigation",
                "If communication lost, return to home position",
                "If battery low, land at nearest safe location"
            ],
            "estimated_duration_minutes": 5.0,
            "risk_assessment": "Low risk mission with standard safety protocols"
        })

    def _parse_llm_response(
        self,
        response: str,
        planning_level: PlanningLevel
    ) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Remove markdown code blocks
                lines = response_clean.split('\n')
                json_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                
                response_clean = '\n'.join(json_lines)
            
            # Parse JSON
            plan_data = json.loads(response_clean)
            
            # Validate basic structure
            required_fields = ['mission_id', 'summary', 'objectives', 'action_sequences']
            for field in required_fields:
                if field not in plan_data:
                    plan_data[field] = self._get_default_value(field)
            
            # Store successful plan for future fallback
            self.last_successful_plan = plan_data.copy()
            self.consecutive_failures = 0  # Reset failure counter
            
            return plan_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response was: {response[:500]}...")
            return self._generate_fallback_plan({}, planning_level)
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return self._generate_fallback_plan({}, planning_level)

    def _generate_fallback_plan(
        self,
        context: Dict[str, Any],
        planning_level: PlanningLevel
    ) -> Dict[str, Any]:
        """Generate safe fallback plan when LLM fails."""
        mission_id = f"fallback_{int(time.time())}"
        
        fallback_plan = {
            'mission_id': mission_id,
            'summary': 'Fallback safety plan due to planning system failure',
            'objectives': ['Maintain safety', 'Return to safe state'],
            'action_sequences': [
                {
                    'actions': [
                        {'type': 'hold_position', 'duration': 10},
                        {'type': 'return_home', 'speed': 2}
                    ],
                    'duration_seconds': 60,
                    'confidence': 1.0,
                    'priority': 'critical',
                    'dependencies': []
                }
            ],
            'formations': [],
            'contingencies': ['Emergency landing if any issues'],
            'estimated_duration_minutes': 1.0,
            'risk_assessment': 'Safety-first fallback plan',
            'is_fallback': True,
        }
        
        return fallback_plan

    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'mission_id': f"default_{int(time.time())}",
            'summary': 'Generated mission plan',
            'objectives': ['Complete mission safely'],
            'action_sequences': [],
            'formations': [],
            'contingencies': [],
            'estimated_duration_minutes': 10.0,
            'risk_assessment': 'Standard risk level',
        }
        return defaults.get(field, '')

    def _update_planning_metrics(self, planning_time: float, success: bool) -> None:
        """Update performance tracking metrics."""
        # Update average planning time
        if self.planning_history:
            self.average_planning_time = (
                self.average_planning_time * len(self.planning_history) + planning_time
            ) / (len(self.planning_history) + 1)
        else:
            self.average_planning_time = planning_time
        
        # Update success rate
        total_attempts = len(self.planning_history) + 1
        previous_successes = sum(1 for entry in self.planning_history if entry.get('success'))
        current_successes = previous_successes + (1 if success else 0)
        self.success_rate = current_successes / total_attempts
        
        # Add to history
        self.planning_history.append({
            'timestamp': time.time(),
            'planning_time_ms': planning_time,
            'success': success,
        })
        
        # Keep only last 100 entries
        if len(self.planning_history) > 100:
            self.planning_history = self.planning_history[-100:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get planner performance statistics."""
        return {
            'model': self.model,
            'average_planning_time_ms': self.average_planning_time,
            'success_rate': self.success_rate,
            'total_plans_generated': len(self.planning_history),
            'recent_performance': self.planning_history[-10:] if self.planning_history else [],
            'consecutive_failures': self.consecutive_failures,
        }
    
    async def _generate_enhanced_fallback_plan(
        self,
        context: Dict[str, Any],
        planning_level: PlanningLevel,
        error_reason: str
    ) -> Dict[str, Any]:
        """Generate enhanced fallback plan with multiple strategies."""
        self.logger.info(f"Generating enhanced fallback plan (level: {planning_level.value})")
        
        # Strategy 1: Use last successful plan if available and appropriate
        if (self.last_successful_plan is not None and 
            self.consecutive_failures <= 2 and
            planning_level == PlanningLevel.STRATEGIC):
            
            self.logger.info("Using adapted version of last successful plan")
            adapted_plan = self.last_successful_plan.copy()
            adapted_plan['mission_id'] = f"adapted_{int(time.time())}"
            adapted_plan['summary'] = f"Adapted plan: {adapted_plan.get('summary', 'Unknown')}"
            adapted_plan['is_fallback'] = True
            adapted_plan['adaptation_reason'] = error_reason
            return adapted_plan
        
        # Strategy 2: Generate context-aware fallback
        if context and context.get('mission'):
            return await self._generate_contextual_fallback(context, planning_level)
        
        # Strategy 3: Use the original simple fallback
        return self._generate_fallback_plan(context, planning_level)
    
    async def _generate_contextual_fallback(
        self,
        context: Dict[str, Any],
        planning_level: PlanningLevel
    ) -> Dict[str, Any]:
        """Generate contextual fallback based on mission context."""
        mission = context.get('mission', '')
        num_drones = context.get('num_drones', 1)
        
        # Analyze mission for key actions
        mission_lower = mission.lower()
        actions = []
        
        # Basic action recognition
        if 'search' in mission_lower or 'survey' in mission_lower:
            actions.extend(['takeoff', 'form_search_pattern', 'systematic_sweep', 'report_findings', 'land'])
        elif 'delivery' in mission_lower:
            actions.extend(['takeoff', 'navigate_to_target', 'deliver_payload', 'return_home', 'land'])
        elif 'patrol' in mission_lower or 'monitor' in mission_lower:
            actions.extend(['takeoff', 'patrol_pattern', 'continuous_monitoring', 'report_status', 'land'])
        else:
            # Generic mission
            actions.extend(['takeoff', 'formation_flight', 'execute_mission', 'return_base', 'land'])
        
        # Determine formation based on drone count
        if num_drones >= 8:
            formation_type = "grid"
            spacing = 15.0
        elif num_drones >= 4:
            formation_type = "line"
            spacing = 12.0
        elif num_drones >= 2:
            formation_type = "pair"
            spacing = 10.0
        else:
            formation_type = "single"
            spacing = 0.0
        
        fallback_plan = {
            'mission_id': f"contextual_fallback_{int(time.time())}",
            'summary': f'Contextual fallback plan for {mission[:50]}...',
            'objectives': [
                f'Execute {mission_lower.split()[0] if mission_lower.split() else "mission"} safely',
                'Maintain formation integrity',
                'Ensure safe return to base',
                'Report mission status'
            ],
            'action_sequences': [
                {
                    'phase': 'initialization',
                    'duration': 30,
                    'actions': [
                        {'type': 'system_check', 'duration': 10},
                        {'type': 'takeoff', 'altitude': 30, 'duration': 20}
                    ],
                    'confidence': 0.9,
                    'priority': 'high',
                    'dependencies': []
                },
                {
                    'phase': 'formation',
                    'duration': 45,
                    'actions': [
                        {'type': 'form_formation', 'pattern': formation_type, 'spacing': spacing},
                        {'type': 'formation_check', 'duration': 15}
                    ],
                    'confidence': 0.8,
                    'priority': 'high',
                    'dependencies': ['initialization']
                },
                {
                    'phase': 'execution',
                    'duration': 300,
                    'actions': [{'type': action, 'duration': 300 // len(actions)} for action in actions],
                    'confidence': 0.7,
                    'priority': 'normal',
                    'dependencies': ['formation']
                },
                {
                    'phase': 'return',
                    'duration': 60,
                    'actions': [
                        {'type': 'return_formation', 'duration': 20},
                        {'type': 'return_to_base', 'duration': 30},
                        {'type': 'land_sequence', 'duration': 10}
                    ],
                    'confidence': 0.9,
                    'priority': 'high',
                    'dependencies': ['execution']
                }
            ],
            'formations': [
                {
                    'formation_type': formation_type,
                    'spacing_meters': spacing,
                    'orientation_degrees': 0.0,
                    'leader_drone_id': 'drone_0' if num_drones > 1 else None
                }
            ],
            'contingencies': [
                'If communication lost, execute return-to-base protocol',
                'If weather deteriorates, reduce altitude and return',
                'If drone failure detected, adjust formation and continue',
                'If mission area inaccessible, report and return'
            ],
            'estimated_duration_minutes': 7.25,
            'risk_assessment': 'Conservative fallback plan with emphasis on safety',
            'is_fallback': True,
            'fallback_type': 'contextual',
            'planning_level': planning_level.value,
        }
        
        self.logger.info(f"Generated contextual fallback plan with {len(actions)} primary actions")
        return fallback_plan