"""Primary web-server launcher for Aster & Row Support Agent Web GUI."""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the Uvicorn web server."""
    import uvicorn
    from src.config import settings

    console.print(Panel.fit(
        f"[bold cyan]Aster & Row AI Support Agent — Web Server[/]\n\n"
        f"Server URL: [bold green]http://{host}:{port}[/]\n"
        f"Default Mode: [bold]Offline (Deterministic RAG)[/]\n"
        f"Live Mode Configured: [bold]{'Yes (Gemini: ' + settings.gemini_model + ')' if settings.is_live_llm_enabled else 'No (API key not set)'}[/]\n\n"
        f"[dim]Press Ctrl+C to stop the server.[/]",
        border_style="cyan",
    ))

    uvicorn.run("src.web.app:app", host=host, port=port, reload=reload)


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent Web GUI Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
