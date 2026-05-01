"""
play_chess_ai.py
────────────────
Load a saved chess AI checkpoint and play against it in the terminal.

Requirements:
    pip install python-chess torch numpy

Usage:
    python play_chess_ai.py                        # You play White
    python play_chess_ai.py --color black          # You play Black
    python play_chess_ai.py --sims 200             # Stronger AI (more MCTS sims)
    python play_chess_ai.py --checkpoint my.pt     # Custom checkpoint path
"""
import subprocess
import argparse
import math
import random
import os
import time
from copy import deepcopy
import cairosvg
from IPython.display import SVG, display
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.svg

RESET  = "\033[0m"
BOLD   = "\033[1m"
BG_LIGHT = "\033[48;5;180m"   # tan
BG_DARK  = "\033[48;5;94m"    # brown
FG_WHITE = "\033[97m"
FG_BLACK = "\033[30m"
HIGHLIGHT = "\033[48;5;226m"  # yellow — last move


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural Network  (must match the architecture used during training)
# ─────────────────────────────────────────────────────────────────────────────

NUM_ACTIONS = 4096   # 64 * 64  (from_sq * 64 + to_sq)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class ChessNet(nn.Module):
    """
    AlphaZero-style network for international chess.
    Input  : (batch, 18, 8, 8)
    Outputs: value ∈ (-1, 1),  policy logits over 4096 actions
    """
    def __init__(self, num_res_blocks: int = 5, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.value_head(x), self.policy_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Board Encoding
# ─────────────────────────────────────────────────────────────────────────────

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board as a float32 array of shape (18, 8, 8).

    Channels:
      0-5   White pieces  (P, N, B, R, Q, K)
      6-11  Black pieces  (P, N, B, R, Q, K)
      12    Turn          (1 = white, 0 = black)
      13    White kingside castling
      14    White queenside castling
      15    Black kingside castling
      16    Black queenside castling
      17    En-passant square
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK,  chess.QUEEN,  chess.KING]

    for i, pt in enumerate(piece_types):
        for sq in board.pieces(pt, chess.WHITE):
            r, c = divmod(sq, 8)
            tensor[i, r, c] = 1.0
        for sq in board.pieces(pt, chess.BLACK):
            r, c = divmod(sq, 8)
            tensor[i + 6, r, c] = 1.0

    tensor[12] = float(board.turn)
    tensor[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[17, r, c] = 1.0

    return tensor


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Policy Helper
# ─────────────────────────────────────────────────────────────────────────────

def get_policy_probs(model: ChessNet,
                     board: chess.Board,
                     device: torch.device):
    """
    Returns {move: probability} for all legal moves, and a position value float.
    """
    model.eval()
    with torch.no_grad():
        t = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(device)
        value, policy_logits = model(t)

    logits = policy_logits.squeeze(0).cpu().numpy()
    legal  = list(board.legal_moves)

    if not legal:
        return {}, value.item()

    indices      = [move_to_index(m) for m in legal]
    legal_logits = logits[indices]
    legal_logits -= legal_logits.max()          # numerical stability
    probs  = np.exp(legal_logits)
    probs /= probs.sum()

    return {m: float(p) for m, p in zip(legal, probs)}, value.item()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MCTS
# ─────────────────────────────────────────────────────────────────────────────

C_PUCT = 1.5


class MCTSNode:
    __slots__ = ("board", "parent", "prior", "children", "N", "W", "Q")

    def __init__(self, board: chess.Board, parent=None, prior: float = 0.0):
        self.board    = board
        self.parent   = parent
        self.prior    = prior
        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    @property
    def is_leaf(self):
        return not self.children

    def ucb(self, parent_N: int) -> float:
        return self.Q + C_PUCT * self.prior * math.sqrt(parent_N) / (1 + self.N)


class MCTS:
    def __init__(self, model: ChessNet, device: torch.device):
        self.model  = model
        self.device = device

    def search(self, board: chess.Board, num_simulations: int = 100) -> dict:
        root = MCTSNode(deepcopy(board))

        for _ in range(num_simulations):
            node = root

            # ── SELECT ───────────────────────────────────────────
            while not node.is_leaf and not node.board.is_game_over():
                _, node = max(
                    node.children.items(),
                    key=lambda kv: kv[1].ucb(node.N),
                )

            # ── EXPAND ───────────────────────────────────────────
            if node.board.is_game_over():
                result = node.board.result()
                value  = (1.0  if result == "1-0" else
                          -1.0 if result == "0-1" else 0.0)
                if not node.board.turn:     # flip for current player
                    value = -value
            else:
                probs, value = get_policy_probs(self.model, node.board, self.device)
                for move, prob in probs.items():
                    child_board = deepcopy(node.board)
                    child_board.push(move)
                    node.children[move] = MCTSNode(child_board, parent=node, prior=prob)

            # ── BACKUP ───────────────────────────────────────────
            while node is not None:
                node.N += 1
                node.W += value
                node.Q  = node.W / node.N
                value   = -value
                node    = node.parent

        total = sum(c.N for c in root.children.values())
        if total == 0:
            n = len(root.children)
            return {m: 1.0 / n for m in root.children}
        return {m: c.N / total for m, c in root.children.items()}

    def best_move(self, board: chess.Board, num_simulations: int = 100) -> chess.Move:
        probs = self.search(board, num_simulations)
        return max(probs, key=probs.get)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Checkpoint Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str,
               num_res_blocks: int = 5,
               channels: int = 64,
               device: torch.device = None) -> ChessNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet(num_res_blocks=num_res_blocks, channels=channels).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both raw state-dict and wrapped checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Model loaded on {device}  ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main Game Loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_human_move(user_input: str, board: chess.Board):
    """Accept UCI (e2e4) or SAN (e4, Nf3, O-O) input."""
    user_input = user_input.strip()
    # Try UCI
    try:
        move = chess.Move.from_uci(user_input)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    # Try SAN
    try:
        move = board.parse_san(user_input)
        if move in board.legal_moves:
            return move
    except:
        pass
    return None


def play_game(model: ChessNet,
              device: torch.device,
              human_color: chess.Color = chess.WHITE,
              num_simulations: int = 100):

    mcts  = MCTS(model, device)
    board = chess.Board()
    last_move = None

    color_name = "White ♙" if human_color == chess.WHITE else "Black ♟"
    ai_name    = "Black ♟" if human_color == chess.WHITE else "White ♙"

    print("\n" + "═" * 50)
    print(f"  ♟  Chess AI — Human vs AI")
    print(f"  You : {color_name}")
    print(f"  AI  : {ai_name}  ({num_simulations} MCTS simulations)")
    print("═" * 50)
    print("  Enter moves as UCI (e2e4) or SAN (e4, Nf3)")
    print("  Commands: 'quit' | 'resign' | 'undo' | 'hint'")
    print("═" * 50)

    while not board.is_game_over():
        # print_board(board, last_move)
        # print(chess.svg.board(board, size=400))
        # print(board.unicode())
        os.system('clear')
        svg_data = chess.svg.board(board, size=400)
        image_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        process = subprocess.Popen(['kitty', '+kitten', 'icat'], stdin=subprocess.PIPE)
        process.communicate(input=image_data)

        # Position evaluation by the neural net
        _, pos_value = get_policy_probs(model, board, device)
        eval_str = f"{pos_value:+.3f}"
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        check_str = " ⚠ CHECK!" if board.is_check() else ""
        print(f"  Turn: {turn_str}  |  Net eval: {eval_str}{check_str}")
        print(f"  Fullmove: {board.fullmove_number}  |  Half-clock: {board.halfmove_clock}")
        print()

        # ── Human move ────────────────────────────────────────────────────
        if board.turn == human_color:
            while True:
                try:
                    user_input = input("  Your move ➜ ").strip()
                except EOFError:
                    user_input = "quit"

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit"):
                    print("\n  Goodbye! 👋")
                    return

                if user_input.lower() == "resign":
                    winner = "AI" if human_color == chess.WHITE else "You"
                    print(f"\n  You resigned. {winner} wins.")
                    return

                if user_input.lower() == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop(); board.pop()
                        last_move = board.peek() if board.move_stack else None
                        print("  Last two moves undone.\n")
                        break
                    else:
                        print("  Nothing to undo.")
                        continue

                if user_input.lower() == "hint":
                    print("  Thinking for a hint...", end=" ", flush=True)
                    hint = mcts.best_move(board, num_simulations // 2)
                    print(f"Suggested: {board.san(hint)}")
                    continue

                move = parse_human_move(user_input, board)
                if move:
                    last_move = move
                    board.push(move)
                    break
                else:
                    legal_sample = [board.san(m) for m in list(board.legal_moves)[:8]]
                    print(f"  ❌ Invalid move. Examples: {', '.join(legal_sample)}")

        # ── AI move ───────────────────────────────────────────────────────
        else:
            print(f"  🤖 AI thinking ({num_simulations} simulations)...", end=" ", flush=True)
            ai_move = mcts.best_move(board, num_simulations)
            print(f"AI plays: {BOLD}{board.san(ai_move)}{RESET}")
            last_move = ai_move
            board.push(ai_move)
        # time.sleep(1)

    # ── Game Over ─────────────────────────────────────────────────────────
    svg_data = chess.svg.board(board, size=400)
    image_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    process = subprocess.Popen(['kitty', '+kitten', 'icat'], stdin=subprocess.PIPE)
    process.communicate(input=image_data)


    result = board.result()

    print("═" * 50)
    if board.is_checkmate():
        loser = "White" if board.turn == chess.WHITE else "Black"
        print(f"  🏆 CHECKMATE! {loser} is mated. Result: {result}")
    elif board.is_stalemate():
        print(f"  🤝 STALEMATE — Draw!")
    elif board.is_insufficient_material():
        print(f"  🤝 Insufficient material — Draw!")
    elif board.is_seventyfive_moves():
        print(f"  🤝 75-move rule — Draw!")
    elif board.is_fivefold_repetition():
        print(f"  🤝 Fivefold repetition — Draw!")
    else:
        print(f"  Game over. Result: {result}")
    print("═" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play chess against a saved AI checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="chess_ai_checkpoint.pt",
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"],
                        help="Your color (default: white)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move — higher = stronger AI (default: 100)")
    parser.add_argument("--res-blocks", type=int, default=5,
                        help="Residual blocks in the network (must match training, default: 5)")
    parser.add_argument("--channels", type=int, default=64,
                        help="Network channels (must match training, default: 64)")
    args = parser.parse_args()

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    human_color  = chess.WHITE if args.color == "white" else chess.BLACK

    model = load_model(
        args.checkpoint,
        num_res_blocks=args.res_blocks,
        channels=args.channels,
        device=device,
    )

    while True:
        play_game(model, device, human_color=human_color, num_simulations=args.sims)
        try:
            again = input("\n  Play again? (y/n): ").strip().lower()
        except EOFError:
            break
        if again != "y":
            print("  Thanks for playing! 👋")
            break


if __name__ == "__main__":
    main()
