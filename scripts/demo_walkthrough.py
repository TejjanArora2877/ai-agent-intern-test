"""Automated demonstration script showcasing the 5 key agent capabilities."""

import sys
import time
from rich.console import Console
from rich.panel import Panel

from pathlib import Path

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.cli.main import print_response
from evaluation.runner import run_evaluation

console = Console(safe_box=True)


def simulate_turn(agent: SupportAgent, session_manager: SessionManager, session_id: str, prompt: str) -> None:
    console.print(f"\n[bold blue]User:[/] {prompt}")
    time.sleep(0.3)
    response = agent.respond(user_message=prompt, session_id=session_id, session_manager=session_manager)
    print_response(response, show_debug=True)
    time.sleep(0.5)


def main():
    agent = SupportAgent(force_mock_mode=True)
    session_manager = SessionManager()

    console.print(Panel.fit(
        "[bold cyan]Aster & Row AI Support Agent — Live Demonstration Walkthrough[/]\n"
        "Demonstrating Knowledge RAG with Citations, Privacy-Safe Order Tool, Multi-Turn, Conflict/Abstention, and Evaluations.",
        border_style="cyan"
    ))

    # Demo 1: Knowledge Base Question with Citations
    console.rule("[bold green]Demo 1: Knowledge-Base Question with Heading Citations[/]")
    simulate_turn(agent, session_manager, "demo_session_1", "How long does a regular customer have to return an unused backpack?")

    # Demo 2: Order Lookup with Privacy & Status Safety
    console.rule("[bold green]Demo 2: Order Status Lookup (Sanitized Tool & Stale Field Protection)[/]")
    simulate_turn(agent, session_manager, "demo_session_2", "Where is ORD-1007 and when should it arrive?")

    # Demo 3: Multi-turn Conversation
    console.rule("[bold green]Demo 3: Multi-Turn Conversation with Context Memory[/]")
    multi_session = "demo_session_3"
    simulate_turn(agent, session_manager, multi_session, "Do you ship internationally?")
    simulate_turn(agent, session_manager, multi_session, "What about Canada, and how long does it take?")

    # Demo 4: Safe Abstention / Active Source Conflict
    console.rule("[bold green]Demo 4: Active Official Source Conflict & Human Escalation[/]")
    simulate_turn(agent, session_manager, "demo_session_4", "Can I put the entire Breeze Tumbler in the dishwasher?")

    # Demo 5: Full Evaluation Suite
    console.rule("[bold green]Demo 5: Comprehensive Evaluation Suite (20 Test Cases)[/]")
    run_evaluation(cases_type="all", offline_mode=True, verbose=False)


if __name__ == "__main__":
    main()
