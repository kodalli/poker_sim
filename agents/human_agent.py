"""Human player agent for interactive terminal poker."""

from rich.console import Console
from rich.panel import Panel

from agents.base import BaseAgent
from poker.player import ActionType, PlayerAction
from poker.table import TableState
from ui.display import (
    print_divider,
    render_action_menu,
    render_community_cards,
    render_game_header,
    render_hole_cards,
    render_table_info,
)


class HumanAgent(BaseAgent):
    """Agent that prompts a human player for decisions via terminal."""

    def __init__(
        self,
        name: str = "You",
        console: Console | None = None,
    ) -> None:
        super().__init__(name)
        self.console = console or Console()
        # Context set by game loop for display
        self.hand_number: int = 1
        self.last_ai_decision: Panel | None = None
        self.use_alternate_screen: bool = True
        # Callback to fetch latest AI decision
        self._get_ai_decision: callable = lambda: None

    def set_ai_decision_getter(self, getter: callable) -> None:
        """Set callback to fetch latest AI decision panel."""
        self._get_ai_decision = getter

    def decide(self, table_state: TableState) -> PlayerAction:
        """Display game state and prompt human for action."""
        # Fetch latest AI decision before displaying
        self.last_ai_decision = self._get_ai_decision()
        if self.use_alternate_screen:
            self.console.clear()
        self._display_game_state(table_state)
        return self._get_player_action(table_state)

    def _display_game_state(self, table_state: TableState) -> None:
        """Render the current game state."""
        # Title with hand number
        title = f"[bold blue]POKER - Hand #{self.hand_number}[/bold blue]"
        self.console.print(title, justify="center")

        # Round header
        self.console.print(render_game_header(table_state.round))
        self.console.print()

        # Hole cards
        self.console.print(f"[bold]Your Cards:[/bold]  {render_hole_cards(table_state.my_hole_cards)}")
        self.console.print()

        # Community cards
        self.console.print(f"[bold]Community:[/bold]   {render_community_cards(table_state.community_cards, table_state.round)}")
        self.console.print()

        print_divider(self.console)

        # Table info (pot, stacks)
        opponent_id = None
        for p in table_state.players:
            if p.id != table_state.my_player_id:
                opponent_id = p.id
                break

        info_table = render_table_info(
            table_state,
            my_id=table_state.my_player_id or 0,
            opponent_id=opponent_id or 1,
        )
        self.console.print(info_table)

        print_divider(self.console)

        # Show last AI decision if available
        if self.last_ai_decision is not None:
            self.console.print(self.last_ai_decision)
            print_divider(self.console)

        self.console.print()

    def _get_player_action(self, table_state: TableState) -> PlayerAction:
        """Prompt for and validate player action."""
        options, menu_str = render_action_menu(table_state)

        if not options:
            # No valid actions (shouldn't happen but handle gracefully)
            return PlayerAction(ActionType.FOLD)

        self.console.print(menu_str)
        self.console.print()

        while True:
            try:
                raw = self.console.input("[bold]> [/bold]").strip()

                # Allow 'q' to quit
                if raw.lower() in ("q", "quit", "exit"):
                    self.console.print("[yellow]Folding and exiting...[/yellow]")
                    raise KeyboardInterrupt

                choice = int(raw)

                for num, _, action_type, amount in options:
                    if num == choice:
                        return PlayerAction(action_type, amount)

                self.console.print(f"[red]Invalid choice. Enter 1-{len(options)}.[/red]")

            except ValueError:
                self.console.print("[red]Please enter a number.[/red]")
            except EOFError:
                # Handle Ctrl+D
                self.console.print("\n[yellow]Exiting...[/yellow]")
                raise KeyboardInterrupt
