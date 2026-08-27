"""Interactive CLI interface for Aster & Row Support Agent with debug observability."""

import argparse
import sys
import uuid
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.models.schemas import AgentResponse

console = Console(safe_box=True)


def print_response(response: AgentResponse, show_debug: bool = False) -> None:
    """Pretty-print the agent response with sources, handoff badge, and debug trace."""
    # Main Answer Panel
    title = "[bold green]Aster & Row Support Agent[/]"
    if response.handoff:
        title += " [bold yellow][Human Specialist Handoff Recommended][/]"

    console.print(Panel(response.answer, title=title, border_style="green" if not response.handoff else "yellow"))

    # Sources
    if response.sources:
        sources_text = Text("Sources: ", style="bold cyan")
        for idx, src in enumerate(response.sources):
            if idx > 0:
                sources_text.append(" | ")
            sources_text.append(f"{src.file} > {src.heading}", style="dim underline")
        console.print(sources_text)
        console.print()

    # Tool Calls
    if response.tool_calls:
        for tc in response.tool_calls:
            console.print(f"[dim cyan]Executed Tool:[/] [bold]{tc.tool_name}[/](args={tc.arguments})")

    # Debug Trace
    if show_debug and response.debug_trace:
        trace = response.debug_trace
        tree = Tree(f"[bold magenta]Debug Observability Trace ({trace.latency_ms:.1f}ms)[/]")
        tree.add(f"[dim]Model Mode:[/] {trace.model_mode}")
        tree.add(f"[dim]Order Query Detected:[/] {trace.order_query_detected} (ID: {trace.order_id_extracted})")
        
        if trace.order_tool_result:
            tool_branch = tree.add(f"[dim]Sanitized Tool View:[/] status={trace.order_tool_result.get('status')}")
            for k, v in trace.order_tool_result.items():
                if v is not None and k not in ('order_id', 'status'):
                    tool_branch.add(f"[dim]{k}:[/] {v}")

        if trace.retrieved_chunks:
            rag_branch = tree.add(f"[dim]Retrieved Chunks ({len(trace.retrieved_chunks)}):[/]")
            for c in trace.retrieved_chunks:
                rag_branch.add(f"[bold]{c.get('file_name')}[/] > [italic]{c.get('heading')}[/] (Score: {c.get('score', 0):.2f})")

        tree.add(f"[dim]Conflict Detected:[/] {trace.conflict_detected}")
        console.print(Panel(tree, border_style="magenta"))


def run_interactive(agent: SupportAgent, session_manager: SessionManager, show_debug: bool = False) -> None:
    """Run an interactive CLI chat session."""
    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    console.print(Panel.fit(
        "[bold cyan]Aster & Row AI Customer Support Agent[/]\n"
        "Ask about return policies, shipping, warranty, or order status.\n"
        "Type [bold red]'exit'[/] or [bold red]'quit'[/] to end session.",
        border_style="cyan"
    ))

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/]")
                break

            response = agent.respond(
                user_message=user_input,
                session_id=session_id,
                session_manager=session_manager,
            )
            print_response(response, show_debug=show_debug)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/]")
            break


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent CLI")
    parser.add_argument("query", nargs="?", type=str, default=None, help="Single query to process")
    parser.add_argument("--debug", "-d", action="store_true", help="Display debug observability trace")
    parser.add_argument("--live", action="store_true", help="Use live Gemini LLM mode (requires GEMINI_API_KEY)")
    args = parser.parse_args()

    agent = SupportAgent(force_mock_mode=not args.live)
    session_manager = SessionManager()

    if args.query:
        # Single query mode
        response = agent.respond(user_message=args.query, session_id="single_query", session_manager=session_manager)
        print_response(response, show_debug=args.debug)
    else:
        # Interactive chat mode
        run_interactive(agent=agent, session_manager=session_manager, show_debug=args.debug)


if __name__ == "__main__":
    main()
