import json
import re

def extract_winner_moves_with_state(game_data: str, winning_player: str) -> list:
    """
    Extracts all move lines by the specified winning player along with the last game state
    tag immediately before each move.

    :param game_data: The full game transcript.
    :param winning_player: The label of the winning player, e.g., 'Player 0'.
    :return: A list of tuples (last_game_state, move_line).
    """
    # Pattern to find moves by the winning player
    move_pattern = rf"\[{re.escape(winning_player)}\][^\[]*"
    move_matches = list(re.finditer(move_pattern, game_data))
    # Pattern to find all [GAME] lines
    state_pattern = r"\[GAME\][^\n]*"

    samples = []
    for m in move_matches:
        move_line = m.group().strip()
        # Extract all [GAME] lines before this move
        prior_text = game_data[:m.start()]
        state_lines = re.findall(state_pattern, prior_text)
        last_state = state_lines[-1] if state_lines else ''
        samples.append((last_state, move_line))
    return samples


def format_game_json_samples(game_data: str, winning_player: str) -> str:
    """
    Formats each winning-player move into a separate JSON sample,
    where the 'user' content is the game state before the move,
    and the 'assistant' content is the move itself.

    :param game_data: The full game transcript.
    :param winning_player: The label of the winning player.
    :return: A multiline string; each line is a JSON string as specified.
    """
    samples = extract_winner_moves_with_state(game_data, winning_player)
    json_lines = []
    for state, move in samples:
        state_text = re.sub(r"^\[[^\]]+\]\s*", "", state)
        move_text = re.sub(r"^\[[^\]]+\]\s*", "", move)
        messages = [
            {"role": "user", "content": state_text},
            {"role": "assistant", "content": move_text}
        ]
        json_lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    return "\n".join(json_lines)


if __name__ == "__main__":
    # Example usage:
    sample_game = (
        "[GAME] You are Player 0 in an N-player Liar's Dice game. You have 5 dice: 3, 1, 5, 2, 6. Player 1 has 5 dice. Player 2 has 5 dice. Rules: On your turn, you may either make a new bid with a higher quantity or higher face (or both), or call the last bid by typing '[Call]'. If you call: if the actual count of that face value among all dice is less than the bid, the last bidder loses one die; otherwise, the caller loses one die. A player who reaches 0 dice is eliminated. The last remaining player wins. Current bid: Quantity = 0, Face Value = 0. Your action? (e.g. '[Bid: 3, 4]' or '[Call]')\n"
        "[Player 1] I'll bid 3 dice with face value 4. [Bid: 3, 4]\n"
        "[GAME] Player 1 bids 3 of face 4.\n"
        "[Player 2] That seems like a reasonable bid, but I want to push it further. [Bid: 4, 4]\n"
        "[GAME] Player 2 bids 4 of face 4.\n"
        "[Player 0] I think that's too high. [Call]\n"
        "[GAME] Player 0 calls! The actual count of face 4 is 3, which is LESS than 4.\n"
        "[Player 1] I'll bid 3 dice with face value 4. [Bid: 3, 4]\n"
    )
    # Suppose Player 0 wins:
    print(format_game_json_samples(sample_game, "Player 0"))
    # Suppose Player 2 wins:
    print(format_game_json_samples(sample_game, "Player 2"))

    print(format_game_json_samples(sample_game, "Player 1"))
