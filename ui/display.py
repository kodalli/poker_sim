"""Display utilities for terminal poker UI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from poker.cards import Card, Suit
from poker.player import ActionType, PlayerAction
from poker.table import GameRound, TableState


SUIT_COLORS = {
    Suit.HEARTS: "red",
    Suit.DIAMONDS: "red",
    Suit.CLUBS: "white",
    Suit.SPADES: "white",
}


def render_card(card: Card) -> str:
    """Render a single card with color (red for hearts/diamonds)."""
    color = SUIT_COLORS[card.suit]
    return f"[{color}][{card}][/{color}]"


def render_hole_cards(cards: tuple[Card, Card] | None) -> str:
    """Render hole cards."""
    if cards is None:
        return "[dim][ ? ][ ? ][/dim]"
    return f"{render_card(cards[0])} {render_card(cards[1])}"


def render_community_cards(cards: list[Card], game_round: GameRound) -> str:
    """Render community cards with placeholders for undealt cards."""
    rendered = []
    for i in range(5):
        if i < len(cards):
            rendered.append(render_card(cards[i]))
        else:
            rendered.append("[dim][ - ][/dim]")
    return " ".join(rendered)


def render_game_header(game_round: GameRound) -> Panel:
    """Render the game round header."""
    return Panel(
        Text(str(game_round), justify="center", style="bold yellow"),
        border_style="blue",
    )


def render_table_info(table_state: TableState, my_id: int, opponent_id: int) -> Table:
    """Render pot and player stacks."""
    info = Table(show_header=False, box=None, padding=(0, 1))
    info.add_column("Label", style="dim")
    info.add_column("Value", style="bold")

    info.add_row("Pot", f"[yellow]{table_state.pot_total}[/yellow] chips")

    if table_state.call_amount > 0:
        info.add_row("To Call", f"[cyan]{table_state.call_amount}[/cyan] chips")

    info.add_row("", "")  # Spacer

    # Show both players
    for player in table_state.players:
        position_name = _get_position_label(player, table_state.num_players)
        if player.id == my_id:
            name = f"[green]You[/green]"
        else:
            name = f"[red]{player.name}[/red]"
        info.add_row(name, f"{player.chips} chips ({position_name})")

    return info


def _get_position_label(player_state, num_players: int) -> str:
    """Get position label for a player."""
    labels = []
    if player_state.is_dealer:
        labels.append("BTN")
    if player_state.is_small_blind:
        labels.append("SB")
    if player_state.is_big_blind:
        labels.append("BB")
    return "/".join(labels) if labels else f"P{player_state.position}"


def calculate_raise_presets(table_state: TableState) -> list[tuple[str, int]]:
    """Calculate raise preset amounts based on pot and stack."""
    presets = []
    min_raise = table_state.min_raise
    max_raise = table_state.max_raise
    call_amount = table_state.call_amount
    pot = table_state.pot_total

    if min_raise > max_raise:
        return presets

    # Min raise
    presets.append(("Min", min_raise))

    # Half pot (current pot + call to see it)
    effective_pot = pot + call_amount
    half_pot = table_state.current_bet + effective_pot // 2
    if min_raise < half_pot < max_raise:
        presets.append(("1/2 Pot", half_pot))

    # Pot size raise
    pot_raise = table_state.current_bet + effective_pot
    if min_raise < pot_raise < max_raise:
        presets.append(("Pot", pot_raise))

    # 2x pot
    two_pot = table_state.current_bet + 2 * effective_pot
    if min_raise < two_pot < max_raise:
        presets.append(("2x Pot", two_pot))

    return presets


def render_action_menu(
    table_state: TableState,
) -> tuple[list[tuple[int, str, ActionType, int]], str]:
    """Build action menu options and return as formatted string.

    Returns:
        Tuple of (options list, formatted menu string)
        Options list: [(number, display_text, action_type, amount), ...]
    """
    valid = table_state.valid_actions
    options: list[tuple[int, str, ActionType, int]] = []
    idx = 1

    if ActionType.FOLD in valid:
        options.append((idx, "Fold", ActionType.FOLD, 0))
        idx += 1

    if ActionType.CHECK in valid:
        options.append((idx, "Check", ActionType.CHECK, 0))
        idx += 1

    if ActionType.CALL in valid:
        options.append((idx, f"Call ({table_state.call_amount})", ActionType.CALL, table_state.call_amount))
        idx += 1

    if ActionType.RAISE in valid:
        presets = calculate_raise_presets(table_state)
        for name, amount in presets:
            options.append((idx, f"Raise - {name} ({amount})", ActionType.RAISE, amount))
            idx += 1

    if ActionType.ALL_IN in valid:
        options.append((idx, f"All-In ({table_state.my_chips})", ActionType.ALL_IN, table_state.my_chips))
        idx += 1

    # Build menu string
    lines = ["[bold]Choose action:[/bold]"]
    for num, text, _, _ in options:
        lines.append(f"  [cyan]{num}[/cyan]. {text}")

    return options, "\n".join(lines)


def render_ai_decision(
    action: PlayerAction,
    probabilities: dict[ActionType, float],
    bet_fraction: float | None = None,
) -> Panel:
    """Render AI decision with probabilities (debug mode)."""
    lines = []

    # Show chosen action
    action_str = action.action_type.name
    if action.amount > 0:
        action_str += f" ({action.amount})"
    lines.append(f"[bold yellow]AI chose: {action_str}[/bold yellow]")
    lines.append("")

    # Show probability distribution
    lines.append("[dim]Probabilities:[/dim]")
    for action_type in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN]:
        prob = probabilities.get(action_type, 0.0)
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        marker = " [yellow]←[/yellow]" if action_type == action.action_type else ""

        prob_pct = f"{prob * 100:5.1f}%"
        lines.append(f"  {action_type.name:7} {prob_pct} {bar}{marker}")

    # Show bet fraction if applicable
    if bet_fraction is not None and action.action_type == ActionType.RAISE:
        lines.append("")
        lines.append(f"[dim]Bet sizing: {bet_fraction * 100:.0f}% of raise range[/dim]")

    return Panel("\n".join(lines), title="[red]AI Decision[/red]", border_style="red")


def render_hand_result(
    won: bool,
    chip_delta: int,
    my_chips: int,
    opponent_chips: int,
    showdown: bool = False,
    winning_hand: str | None = None,
) -> Panel:
    """Render the result of a hand."""
    lines = []

    if won:
        lines.append("[bold green]You won![/bold green]")
    else:
        lines.append("[bold red]You lost.[/bold red]")

    if showdown and winning_hand:
        lines.append(f"Winning hand: {winning_hand}")

    delta_str = f"+{chip_delta}" if chip_delta > 0 else str(chip_delta)
    delta_style = "green" if chip_delta > 0 else "red"
    lines.append(f"Chips: [{delta_style}]{delta_str}[/{delta_style}]")

    lines.append("")
    lines.append(f"Your stack: [bold]{my_chips}[/bold]")
    lines.append(f"Opponent stack: {opponent_chips}")

    border = "green" if won else "red"
    return Panel("\n".join(lines), title="Hand Result", border_style=border)


def clear_screen(console: Console) -> None:
    """Clear the terminal screen."""
    console.clear()


def print_divider(console: Console, char: str = "─", width: int = 50) -> None:
    """Print a horizontal divider."""
    console.print(f"[dim]{char * width}[/dim]")
