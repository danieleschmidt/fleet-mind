"""Command-line interface for Fleet-Mind drone swarm coordination."""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from .coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints
from .fleet.drone_fleet import DroneFleet, DroneCapability
from .communication.webrtc_streamer import WebRTCStreamer
from .planning.llm_planner import LLMPlanner


app = typer.Typer(
    name="fleet-mind",
    help="Fleet-Mind - Realtime Swarm LLM Coordination Platform",
    no_args_is_help=True,
)

console = Console()


class FleetMindCLI:
    """CLI interface for Fleet-Mind operations."""
    
    def __init__(self):
        self.coordinator: Optional[SwarmCoordinator] = None
        self.fleet: Optional[DroneFleet] = None
        self.config_path = Path.home() / ".fleet-mind" / "config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load CLI configuration."""
        default_config = {
            "llm_model": "gpt-4o",
            "num_drones": 10,
            "update_rate": 10.0,
            "latent_dim": 512,
            "max_altitude": 120.0,
            "safety_distance": 5.0,
            "communication_protocol": "webrtc",
            "topology": "mesh",
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        
        return default_config

    def _save_config(self) -> None:
        """Save current configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    async def _initialize_fleet(self, num_drones: int) -> None:
        """Initialize Fleet-Mind components."""
        constraints = MissionConstraints(
            max_altitude=self.config["max_altitude"],
            safety_distance=self.config["safety_distance"],
        )
        
        # Initialize coordinator
        self.coordinator = SwarmCoordinator(
            llm_model=self.config["llm_model"],
            latent_dim=self.config["latent_dim"],
            max_drones=num_drones,
            update_rate=self.config["update_rate"],
            safety_constraints=constraints,
        )
        
        # Initialize fleet
        drone_ids = [f"drone_{i}" for i in range(num_drones)]
        self.fleet = DroneFleet(
            drone_ids=drone_ids,
            communication_protocol=self.config["communication_protocol"],
            topology=self.config["topology"],
        )
        
        # Connect components
        await self.coordinator.connect_fleet(self.fleet)
        await self.fleet.start_monitoring()

    def _create_status_layout(self) -> Layout:
        """Create live status display layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        return layout

    def _update_status_layout(self, layout: Layout) -> None:
        """Update live status display."""
        if not self.coordinator or not self.fleet:
            return
        
        # Header
        layout["header"].update(Panel(
            Text("Fleet-Mind Live Status", style="bold magenta", justify="center"),
            border_style="bright_blue"
        ))
        
        # Fleet status table
        fleet_status = self.fleet.get_fleet_status()
        
        fleet_table = Table(title="Fleet Status")
        fleet_table.add_column("Metric", style="cyan")
        fleet_table.add_column("Value", style="green")
        
        fleet_table.add_row("Total Drones", str(fleet_status["total_drones"]))
        fleet_table.add_row("Active Drones", str(fleet_status["active_drones"]))
        fleet_table.add_row("Failed Drones", str(fleet_status["failed_drones"]))
        fleet_table.add_row("Average Battery", f"{fleet_status['average_battery']:.1f}%")
        fleet_table.add_row("Average Health", f"{fleet_status['average_health']:.2f}")
        fleet_table.add_row("Missions Completed", str(fleet_status["missions_completed"]))
        
        layout["left"].update(fleet_table)
        
        # Drone details table
        drone_table = Table(title="Drone Details")
        drone_table.add_column("ID", style="cyan")
        drone_table.add_column("Status", style="green")
        drone_table.add_column("Battery", style="yellow")
        drone_table.add_column("Health", style="magenta")
        
        for drone_id in self.fleet.get_active_drones()[:10]:  # Show first 10
            state = self.fleet.get_drone_state(drone_id)
            if state:
                drone_table.add_row(
                    drone_id,
                    state.status.value,
                    f"{state.battery_percent:.1f}%",
                    f"{state.health_score:.2f}"
                )
        
        layout["right"].update(drone_table)
        
        # Footer
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        layout["footer"].update(Panel(
            f"Last Updated: {current_time} | Press Ctrl+C to exit",
            border_style="dim"
        ))


# Initialize CLI instance
cli = FleetMindCLI()


@app.command()
def init(
    drones: int = typer.Option(10, "--drones", "-d", help="Number of drones"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model to use"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
):
    """Initialize Fleet-Mind configuration and components."""
    
    console.print("[bold blue]Initializing Fleet-Mind...[/bold blue]")
    
    # Update configuration
    cli.config.update({
        "num_drones": drones,
        "llm_model": model,
    })
    cli._save_config()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        
        task = progress.add_task("Setting up components...", total=None)
        
        async def init_async():
            await cli._initialize_fleet(drones)
        
        asyncio.run(init_async())
    
    console.print(f"[green]✓ Fleet-Mind initialized with {drones} drones[/green]")
    console.print(f"[green]✓ Using LLM model: {model}[/green]")
    console.print(f"[green]✓ Configuration saved to: {cli.config_path}[/green]")


@app.command()
def status():
    """Display current fleet status."""
    
    if not cli.coordinator or not cli.fleet:
        console.print("[red]Fleet not initialized. Run 'fleet-mind init' first.[/red]")
        raise typer.Exit(1)
    
    async def get_status():
        return await cli.coordinator.get_swarm_status()
    
    status_data = asyncio.run(get_status())
    
    # Display status in formatted tables
    console.print("\n[bold magenta]Fleet-Mind Status Report[/bold magenta]\n")
    
    # Main status
    status_table = Table(title="Swarm Status")
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", style="green")
    
    status_table.add_row("Mission Status", status_data["mission_status"])
    status_table.add_row("Current Mission", status_data.get("current_mission", "None"))
    status_table.add_row("Uptime", f"{status_data['uptime_seconds']:.1f}s")
    status_table.add_row("Recent Latency", f"{status_data['recent_latency_ms']:.1f}ms")
    
    console.print(status_table)
    
    # Fleet health
    fleet_status = cli.fleet.get_fleet_status()
    health_table = Table(title="Fleet Health")
    health_table.add_column("Metric", style="cyan")
    health_table.add_column("Value", style="green")
    
    health_table.add_row("Active Drones", str(fleet_status["active_drones"]))
    health_table.add_row("Failed Drones", str(fleet_status["failed_drones"]))
    health_table.add_row("Average Battery", f"{fleet_status['average_battery']:.1f}%")
    health_table.add_row("Average Health", f"{fleet_status['average_health']:.2f}")
    
    console.print(health_table)


@app.command()
def live():
    """Display live fleet status (updates in real-time)."""
    
    if not cli.coordinator or not cli.fleet:
        console.print("[red]Fleet not initialized. Run 'fleet-mind init' first.[/red]")
        raise typer.Exit(1)
    
    console.print("[bold blue]Starting live status display...[/bold blue]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")
    
    layout = cli._create_status_layout()
    
    try:
        with Live(layout, refresh_per_second=2, screen=True):
            while True:
                cli._update_status_layout(layout)
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Live status display stopped.[/yellow]")


@app.command()
def mission(
    description: str = typer.Argument(..., help="Mission description in natural language"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate plan without execution"),
):
    """Execute a mission using natural language description."""
    
    if not cli.coordinator or not cli.fleet:
        console.print("[red]Fleet not initialized. Run 'fleet-mind init' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Planning mission:[/bold blue] {description}")
    
    async def execute_mission():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            
            # Generate plan
            task = progress.add_task("Generating mission plan...", total=None)
            plan = await cli.coordinator.generate_plan(description)
            progress.update(task, description="✓ Mission plan generated")
            
            # Display plan
            console.print("\n[bold green]Mission Plan Generated:[/bold green]")
            
            plan_table = Table(title="Mission Details")
            plan_table.add_column("Field", style="cyan")
            plan_table.add_column("Value", style="green")
            
            plan_table.add_row("Mission ID", plan["mission_id"])
            plan_table.add_row("Planning Time", f"{plan['planning_latency_ms']:.1f}ms")
            plan_table.add_row("Summary", plan["raw_plan"].get("summary", ""))
            plan_table.add_row("Estimated Duration", f"{plan['raw_plan'].get('estimated_duration_minutes', 0):.1f} min")
            
            console.print(plan_table)
            
            if not dry_run:
                # Execute mission
                task = progress.add_task("Executing mission...", total=None)
                success = await cli.coordinator.execute_mission(plan)
                
                if success:
                    progress.update(task, description="✓ Mission completed successfully")
                    console.print("[bold green]Mission completed successfully![/bold green]")
                else:
                    progress.update(task, description="✗ Mission failed")
                    console.print("[bold red]Mission execution failed![/bold red]")
            else:
                console.print("[yellow]Dry run completed - mission not executed[/yellow]")
    
    asyncio.run(execute_mission())


@app.command()
def emergency():
    """Execute emergency stop for all drones."""
    
    if not cli.coordinator:
        console.print("[red]Fleet not initialized. Run 'fleet-mind init' first.[/red]")
        raise typer.Exit(1)
    
    confirm = typer.confirm("Are you sure you want to execute emergency stop?")
    if not confirm:
        console.print("[yellow]Emergency stop cancelled.[/yellow]")
        return
    
    console.print("[bold red]EXECUTING EMERGENCY STOP[/bold red]")
    
    async def emergency_stop():
        await cli.coordinator.emergency_stop()
    
    asyncio.run(emergency_stop())
    console.print("[bold green]Emergency stop executed successfully.[/bold green]")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key"),
    value: Optional[str] = typer.Option(None, "--value", help="Configuration value"),
):
    """Manage Fleet-Mind configuration."""
    
    if show:
        console.print("[bold blue]Current Configuration:[/bold blue]\n")
        
        config_table = Table(title="Fleet-Mind Configuration")
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="green")
        
        for key, val in cli.config.items():
            config_table.add_row(key, str(val))
        
        console.print(config_table)
        console.print(f"\n[dim]Config file: {cli.config_path}[/dim]")
    
    elif set_key and value:
        # Type conversion for known numeric fields
        if set_key in ["num_drones", "latent_dim"]:
            value = int(value)
        elif set_key in ["update_rate", "max_altitude", "safety_distance"]:
            value = float(value)
        
        cli.config[set_key] = value
        cli._save_config()
        
        console.print(f"[green]✓ Set {set_key} = {value}[/green]")
    
    else:
        console.print("[yellow]Use --show to view config or --set KEY --value VALUE to update[/yellow]")


@app.command()
def drones(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all drones"),
    drone_id: Optional[str] = typer.Option(None, "--drone", "-d", help="Show specific drone"),
    capabilities: bool = typer.Option(False, "--capabilities", "-c", help="Show drone capabilities"),
):
    """Manage and inspect individual drones."""
    
    if not cli.fleet:
        console.print("[red]Fleet not initialized. Run 'fleet-mind init' first.[/red]")
        raise typer.Exit(1)
    
    if list_all:
        # List all drones
        console.print("[bold blue]Drone Fleet Overview:[/bold blue]\n")
        
        drone_table = Table(title="All Drones")
        drone_table.add_column("ID", style="cyan")
        drone_table.add_column("Status", style="green")
        drone_table.add_column("Battery", style="yellow")
        drone_table.add_column("Health", style="magenta")
        drone_table.add_column("Position", style="blue")
        
        for drone_id in cli.fleet.drone_ids:
            state = cli.fleet.get_drone_state(drone_id)
            if state:
                pos_str = f"({state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f})"
                drone_table.add_row(
                    drone_id,
                    state.status.value,
                    f"{state.battery_percent:.1f}%",
                    f"{state.health_score:.2f}",
                    pos_str
                )
        
        console.print(drone_table)
    
    elif drone_id:
        # Show specific drone
        state = cli.fleet.get_drone_state(drone_id)
        if not state:
            console.print(f"[red]Drone {drone_id} not found.[/red]")
            return
        
        console.print(f"[bold blue]Drone {drone_id} Details:[/bold blue]\n")
        
        details_table = Table(title=f"Drone {drone_id}")
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="green")
        
        details_table.add_row("Status", state.status.value)
        details_table.add_row("Position", f"({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})")
        details_table.add_row("Velocity", f"({state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f})")
        details_table.add_row("Battery", f"{state.battery_percent:.1f}%")
        details_table.add_row("Health Score", f"{state.health_score:.2f}")
        details_table.add_row("Mission Progress", f"{state.mission_progress:.1f}%")
        details_table.add_row("Communication Quality", f"{state.communication_quality:.2f}")
        details_table.add_row("Last Update", time.strftime("%H:%M:%S", time.localtime(state.last_update)))
        
        console.print(details_table)
        
        if capabilities:
            caps_table = Table(title=f"Drone {drone_id} Capabilities")
            caps_table.add_column("Capability", style="cyan")
            
            for cap in state.capabilities:
                caps_table.add_row(cap.value)
            
            console.print(caps_table)
    
    elif capabilities:
        # Show capabilities overview
        console.print("[bold blue]Fleet Capabilities Overview:[/bold blue]\n")
        
        caps = cli.fleet.get_capabilities()
        
        caps_table = Table(title="Drone Capabilities")
        caps_table.add_column("Drone ID", style="cyan")
        caps_table.add_column("Capabilities", style="green")
        
        for drone_id, drone_caps in caps.items():
            caps_str = ", ".join(drone_caps)
            caps_table.add_row(drone_id, caps_str)
        
        console.print(caps_table)
    
    else:
        console.print("[yellow]Use --list to show all drones, --drone ID for details, or --capabilities for capabilities[/yellow]")


@app.command()
def version():
    """Show Fleet-Mind version information."""
    from . import __version__, __author__
    
    console.print(f"[bold blue]Fleet-Mind[/bold blue] version [green]{__version__}[/green]")
    console.print(f"Author: [cyan]{__author__}[/cyan]")
    console.print("Real-time Swarm LLM Coordination Platform")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()