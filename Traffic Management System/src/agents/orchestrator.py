# src/agents/orchestrator.py
"""
Orchestrator for coordinating all traffic management agents.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

# Type protocols for agent validation
@runtime_checkable
class TrafficAgentProtocol(Protocol):
    def reason_and_act(self, origin: str, destination: str) -> Any: ...

@runtime_checkable
class EventAgentProtocol(Protocol):
    def fetch_event_data(self, city: str) -> Optional[dict]: ...
    def analyze_event_impact(self, data: dict) -> Any: ...

@runtime_checkable
class WeatherAgentProtocol(Protocol):
    def fetch_weather_data(self, city: str) -> Optional[dict]: ...
    def analyze_weather_impact(self, data: dict) -> Any: ...

@runtime_checkable  
class RoutingAgentProtocol(Protocol):
    def get_alternative_routes(self, origin: str, destination: str) -> Optional[dict]: ...
    def analyze_routes(self, data: dict) -> Any: ...


class AgentStatus(Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class OrchestratorError(Exception):
    """Base exception for Orchestrator errors."""
    pass


class ConfigurationError(OrchestratorError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(OrchestratorError):
    """Raised when input validation fails."""
    pass


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent_name: str
    status: AgentStatus
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }


@dataclass
class OrchestrationResult:
    """Complete orchestration result with all agent outputs."""
    traffic: AgentResult
    events: AgentResult
    weather: AgentResult
    routing: AgentResult
    origin: str = ""
    destination: str = ""
    city: str = ""
    overall_status: str = "success"
    total_execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def all_results(self) -> List[AgentResult]:
        return [self.traffic, self.events, self.weather, self.routing]
    
    @property
    def successful_agents(self) -> int:
        return sum(1 for r in self.all_results if r.status == AgentStatus.SUCCESS)
    
    @property
    def failed_agents(self) -> List[str]:
        return [r.agent_name for r in self.all_results if r.status == AgentStatus.FAILED]
    
    @property
    def overall_severity(self) -> str:
        """Determine overall severity based on all agents."""
        severities = []
        
        if self.traffic.status == AgentStatus.SUCCESS and self.traffic.data:
            if self.traffic.data.get("is_congested"):
                severities.append("high")
            elif self.traffic.data.get("congestion_level") in ["moderate", "high"]:
                severities.append("medium")
        
        if self.weather.status == AgentStatus.SUCCESS and self.weather.data:
            if self.weather.data.get("is_hazardous"):
                severities.append("high")
            elif self.weather.data.get("severity") in ["moderate", "severe"]:
                severities.append("medium")
        
        if self.events.status == AgentStatus.SUCCESS and self.events.data:
            level = self.events.data.get("level", "")
            if level in ["high", "critical"]:
                severities.append("high")
            elif level == "medium":
                severities.append("medium")
        
        if "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        return "low"
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        parts = []
        
        if self.traffic.status == AgentStatus.SUCCESS and self.traffic.data:
            action = self.traffic.data.get("recommended_action", "")
            if action:
                parts.append(f"Traffic: {action[:50]}...")
        
        if self.weather.status == AgentStatus.SUCCESS and self.weather.data:
            condition = self.weather.data.get("condition", "")
            temp = self.weather.data.get("temperature", "")
            if condition:
                parts.append(f"Weather: {condition} ({temp}°C)")
        
        if self.events.status == AgentStatus.SUCCESS and self.events.data:
            msg = self.events.data.get("message", "")
            if msg:
                parts.append(f"Events: {msg[:40]}...")
        
        return " | ".join(parts) if parts else "Analysis complete"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "city": self.city,
            "overall_status": self.overall_status,
            "overall_severity": self.overall_severity,
            "summary": self.get_summary(),
            "agents": {
                "traffic": self.traffic.to_dict(),
                "events": self.events.to_dict(),
                "weather": self.weather.to_dict(),
                "routing": self.routing.to_dict()
            },
            "successful_agents": self.successful_agents,
            "failed_agents": self.failed_agents,
            "recommendations": self.recommendations,
            "execution_time_ms": round(self.total_execution_time_ms, 2),
            "timestamp": self.timestamp
        }


class Orchestrator:
    """
    Coordinates interaction between all traffic management agents.
    
    Runs agents in parallel for optimal performance and provides
    unified results with combined recommendations.
    
    Attributes:
        DEFAULT_TIMEOUT: Maximum time to wait for all agents
        MAX_WORKERS: Number of parallel agent executions
        CACHE_TTL: How long to cache results
    
    Example:
        >>> orchestrator = Orchestrator.create_default()
        >>> result = orchestrator.run("New York", "Boston", "New York")
        >>> print(result.overall_severity)
        >>> print(result.recommendations)
    """
    
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_WORKERS = 4
    CACHE_TTL = timedelta(minutes=5)
    
    def __init__(
        self,
        traffic_agent: TrafficAgentProtocol,
        event_agent: EventAgentProtocol,
        weather_agent: WeatherAgentProtocol,
        routing_agent: RoutingAgentProtocol,
        timeout: Optional[float] = None,
        enable_cache: bool = True
    ):
        """
        Initialize Orchestrator with all required agents.
        
        Args:
            traffic_agent: Agent for traffic prediction
            event_agent: Agent for event data
            weather_agent: Agent for weather data
            routing_agent: Agent for routing
            timeout: Override default timeout
            enable_cache: Whether to cache results
            
        Raises:
            ValueError: If any agent is None
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate agents
        if traffic_agent is None:
            raise ValueError("traffic_agent cannot be None")
        if event_agent is None:
            raise ValueError("event_agent cannot be None")
        if weather_agent is None:
            raise ValueError("weather_agent cannot be None")
        if routing_agent is None:
            raise ValueError("routing_agent cannot be None")
        
        self.traffic_agent = traffic_agent
        self.event_agent = event_agent
        self.weather_agent = weather_agent
        self.routing_agent = routing_agent
        
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.enable_cache = enable_cache
        self._cache: Dict[str, tuple] = {}
    
    @classmethod
    def create_default(cls) -> 'Orchestrator':
        """
        Create orchestrator with default agent configuration.
        
        Returns:
            Configured Orchestrator instance
        """
        # Import here to avoid circular imports
        from .traffic_agent import TrafficAgent
        from .event_agent import EventAgent
        from .weather_agent import WeatherAgent
        from .routing_agent import RoutingAgent
        
        return cls(
            traffic_agent=TrafficAgent(),
            event_agent=EventAgent(),
            weather_agent=WeatherAgent(),
            routing_agent=RoutingAgent()
        )
    
    @classmethod
    def create_with_config(cls, config: Dict[str, Any]) -> 'Orchestrator':
        """
        Create orchestrator from configuration dictionary.
        
        Args:
            config: Configuration with API keys and settings
            
        Returns:
            Configured Orchestrator instance
        """
        from .traffic_agent import TrafficAgent
        from .event_agent import EventAgent
        from .weather_agent import WeatherAgent
        from .routing_agent import RoutingAgent
        
        return cls(
            traffic_agent=TrafficAgent(
                model_path=config.get("traffic_model_path"),
                threshold=config.get("congestion_threshold", 0.7)
            ),
            event_agent=EventAgent(token=config.get("eventbrite_api_key")),
            weather_agent=WeatherAgent(api_key=config.get("openweather_api_key")),
            routing_agent=RoutingAgent(api_key=config.get("google_maps_api_key")),
            timeout=config.get("timeout"),
            enable_cache=config.get("enable_cache", True)
        )
    
    def _validate_location(self, location: str, field_name: str) -> str:
        """Validate location input."""
        if not location or not isinstance(location, str):
            raise ValidationError(f"{field_name} must be a non-empty string")
        location = location.strip()
        if len(location) < 2:
            raise ValidationError(f"{field_name} must be at least 2 characters")
        if len(location) > 500:
            raise ValidationError(f"{field_name} too long (max 500 characters)")
        return location
    
    def _validate_city(self, city: str) -> str:
        """Validate city input."""
        if not city or not isinstance(city, str):
            raise ValidationError("city must be a non-empty string")
        city = city.strip()
        if len(city) < 2:
            raise ValidationError("city must be at least 2 characters")
        return city
    
    def _get_cache_key(self, origin: str, destination: str, city: str) -> str:
        """Generate cache key for query."""
        key_data = f"{origin}|{destination}|{city}".lower()
        return md5(key_data.encode()).hexdigest()
    
    def _run_agent(
        self,
        name: str,
        func: Callable[[], Any]
    ) -> AgentResult:
        """
        Execute a single agent with timing and error handling.
        
        Args:
            name: Agent name for logging
            func: Function to execute
            
        Returns:
            AgentResult with status and data
        """
        start = time.time()
        try:
            data = func()
            
            # Convert to dict if has to_dict method
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            
            return AgentResult(
                agent_name=name,
                status=AgentStatus.SUCCESS,
                data=data,
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.logger.error(f"{name} agent error: {e}", exc_info=True)
            return AgentResult(
                agent_name=name,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
    
    def _run_traffic_agent(self, origin: str, destination: str) -> AgentResult:
        """Execute traffic agent."""
        return self._run_agent(
            "traffic",
            lambda: self.traffic_agent.reason_and_act(origin, destination)
        )
    
    def _run_event_agent(self, city: str) -> AgentResult:
        """Execute event agent."""
        def execute():
            data = self.event_agent.fetch_event_data(city)
            if data:
                return self.event_agent.analyze_event_impact(data)
            return {"message": "No event data available", "level": "low"}
        
        return self._run_agent("events", execute)
    
    def _run_weather_agent(self, city: str) -> AgentResult:
        """Execute weather agent."""
        def execute():
            data = self.weather_agent.fetch_weather_data(city)
            if data:
                return self.weather_agent.analyze_weather_impact(data)
            return {"message": "No weather data available", "severity": "unknown"}
        
        return self._run_agent("weather", execute)
    
    def _run_routing_agent(self, origin: str, destination: str) -> AgentResult:
        """Execute routing agent."""
        def execute():
            data = self.routing_agent.get_alternative_routes(origin, destination)
            if data:
                return self.routing_agent.analyze_routes(data)
            return {"message": "No route data available", "routes": []}
        
        return self._run_agent("routing", execute)
    
    def _generate_recommendations(
        self,
        traffic: AgentResult,
        events: AgentResult,
        weather: AgentResult,
        routing: AgentResult
    ) -> List[str]:
        """Generate combined recommendations from all agents."""
        recommendations = []
        priority = []
        
        # Traffic
        if traffic.status == AgentStatus.SUCCESS and traffic.data:
            if traffic.data.get("is_congested"):
                priority.append(
                    f"⚠️ {traffic.data.get('recommended_action', 'Heavy traffic detected')}"
                )
            if traffic.data.get("signal_timing_adjustment"):
                recommendations.append(
                    f"🚦 {traffic.data['signal_timing_adjustment']}"
                )
        
        # Weather
        if weather.status == AgentStatus.SUCCESS and weather.data:
            if weather.data.get("is_hazardous"):
                priority.append(
                    f"🌧️ Hazardous: {weather.data.get('condition', 'Bad weather')}"
                )
            for rec in weather.data.get("recommendations", [])[:2]:
                recommendations.append(f"🌤️ {rec}")
        
        # Events
        if events.status == AgentStatus.SUCCESS and events.data:
            level = events.data.get("level", "")
            if level in ["high", "critical"]:
                priority.append(f"📅 {events.data.get('message', 'Major events')}")
            for rec in events.data.get("recommendations", [])[:2]:
                recommendations.append(f"🎉 {rec}")
        
        # Routing
        if routing.status == AgentStatus.SUCCESS and routing.data:
            routes = routing.data if isinstance(routing.data, list) else []
            if routes:
                best = routes[0]
                recommendations.append(
                    f"🛣️ Best route: {best.get('summary', 'N/A')} ({best.get('duration', 'N/A')})"
                )
        
        return priority + recommendations
    
    def run(
        self,
        origin: str,
        destination: str,
        city: str,
        use_cache: bool = True,
        include_agents: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Run full analysis and recommendation cycle.
        
        Executes all agents in parallel and combines their results
        into a unified response with recommendations.
        
        Args:
            origin: Starting location (address or coordinates)
            destination: End location (address or coordinates)
            city: City name for event/weather data
            use_cache: Use cached results if available
            include_agents: Only run specific agents (default: all)
            timeout: Override default timeout
            
        Returns:
            OrchestrationResult with all agent outputs
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        origin = self._validate_location(origin, "origin")
        destination = self._validate_location(destination, "destination")
        city = self._validate_city(city)
        
        timeout = timeout or self.timeout
        
        self.logger.info(
            f"Starting orchestration: {origin} -> {destination}, city={city}"
        )
        
        # Check cache
        if self.enable_cache and use_cache:
            cache_key = self._get_cache_key(origin, destination, city)
            if cache_key in self._cache:
                result, cached_time = self._cache[cache_key]
                if datetime.now() - cached_time < self.CACHE_TTL:
                    self.logger.debug("Returning cached result")
                    return result
        
        start_time = time.time()
        
        # Determine which agents to run
        all_agents = {"traffic", "events", "weather", "routing"}
        agents_to_run = set(include_agents) & all_agents if include_agents else all_agents
        
        # Prepare tasks
        tasks: Dict[str, Callable] = {}
        if "traffic" in agents_to_run:
            tasks["traffic"] = lambda: self._run_traffic_agent(origin, destination)
        if "events" in agents_to_run:
            tasks["events"] = lambda: self._run_event_agent(city)
        if "weather" in agents_to_run:
            tasks["weather"] = lambda: self._run_weather_agent(city)
        if "routing" in agents_to_run:
            tasks["routing"] = lambda: self._run_routing_agent(origin, destination)
        
        # Execute in parallel
        results: Dict[str, AgentResult] = {}
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_name = {
                executor.submit(task): name
                for name, task in tasks.items()
            }
            
            try:
                for future in as_completed(future_to_name, timeout=timeout):
                    name = future_to_name[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        self.logger.error(f"Agent {name} failed: {e}")
                        results[name] = AgentResult(
                            agent_name=name,
                            status=AgentStatus.FAILED,
                            error=str(e)
                        )
            except TimeoutError:
                self.logger.warning("Orchestration timeout - some agents did not complete")
        
        # Fill in skipped agents
        for name in all_agents:
            if name not in results:
                results[name] = AgentResult(
                    agent_name=name,
                    status=AgentStatus.SKIPPED
                )
        
        total_time = (time.time() - start_time) * 1000
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            results["traffic"],
            results["events"],
            results["weather"],
            results["routing"]
        )
        
        # Build result
        orchestration_result = OrchestrationResult(
            traffic=results["traffic"],
            events=results["events"],
            weather=results["weather"],
            routing=results["routing"],
            origin=origin,
            destination=destination,
            city=city,
            overall_status="success" if any(
                r.status == AgentStatus.SUCCESS for r in results.values()
            ) else "failed",
            total_execution_time_ms=total_time,
            recommendations=recommendations
        )
        
        # Update cache
        if self.enable_cache:
            cache_key = self._get_cache_key(origin, destination, city)
            self._cache[cache_key] = (orchestration_result, datetime.now())
        
        self.logger.info(
            f"Orchestration complete in {total_time:.0f}ms. "
            f"Success: {orchestration_result.successful_agents}/4 agents"
        )
        
        if orchestration_result.failed_agents:
            self.logger.warning(f"Failed agents: {orchestration_result.failed_agents}")
        
        return orchestration_result
    
    def run_quick(
        self,
        origin: str,
        destination: str,
        city: str
    ) -> Dict[str, Any]:
        """
        Quick run returning simple dictionary (backward compatible).
        
        Args:
            origin: Starting location
            destination: End location
            city: City name
            
        Returns:
            Simple dictionary with results
        """
        result = self.run(origin, destination, city)
        
        return {
            "traffic": result.traffic.data if result.traffic.status == AgentStatus.SUCCESS else "Traffic data unavailable",
            "event": result.events.data if result.events.status == AgentStatus.SUCCESS else "Event data unavailable",
            "weather": result.weather.data if result.weather.status == AgentStatus.SUCCESS else "Weather data unavailable",
            "routes": result.routing.data if result.routing.status == AgentStatus.SUCCESS else "Route data unavailable"
        }
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        self.logger.debug("Orchestrator cache cleared")
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all agents.
        
        Returns:
            Dictionary with agent health status
        """
        return {
            "traffic_agent": self.traffic_agent is not None,
            "event_agent": self.event_agent is not None,
            "weather_agent": self.weather_agent is not None,
            "routing_agent": self.routing_agent is not None,
            "cache_enabled": self.enable_cache
        }
