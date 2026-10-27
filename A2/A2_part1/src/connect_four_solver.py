import math
import random
import numpy as np
from scipy.signal import convolve2d
import copy


# Board dimensions for Connect Four
ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    """
    Creates an empty Connect Four board (numpy 2D array).
 
    Returns:
    np.ndarray:
        A 2D numpy array of shape (ROW_COUNT, COLUMN_COUNT) filled with zeros (float).
    """
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=float)
    return board


def drop_piece(board, row, col, piece):
    """
    Places a piece (1 or 2) at the specified (row, col) position on the board.

    Parameters:
    board (np.ndarray): The current board, shape (ROW_COUNT, COLUMN_COUNT).
    row (int): The row index where the piece should be placed.
    col (int): The column index where the piece should be placed.
    piece (int): The piece identifier (1 or 2).
    
    Returns:
    None. The 'board' is modified in-place. Do NOT return a new board!
    """
    board[row, col] = piece


def is_valid_location(board: np.ndarray, col: int):
    """
    Checks if dropping a piece in 'col' is valid (column not full).

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to check.

    Returns:
    bool: True if it's valid to drop a piece in this column, False otherwise.
    """
    return board[ROW_COUNT - 1, col] == 0


def get_next_open_row(board, col):
    """
    Gets the next open row in the given column.

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to search.

    Returns:
    int: The row index of the lowest empty cell in this column.
    """
    for row in range(board.shape[0]):
        if board[row][col] == 0:
            return row
    return None # if column is completely full

def winning_move(board, piece: int):
    """
    Checks if the board state is a winning state for the given piece.

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    bool: True if 'piece' has a winning 4 in a row, False otherwise.
    This requires checking horizontally, vertically, and diagonally.
    """
    # Use mask w kernels to check horizontal, vertical, and diagonal connections
    mask = (board == piece)

    kernels = [
        np.ones((1,4)),
        np.ones((4,1)),
        np.eye(4),
        np.fliplr(np.eye(4))
    ]

    return any (np.any(convolve2d(mask, k, mode='valid') == 4) for k in kernels)


def get_valid_locations(board):
    """
    Returns a list of columns that are valid to drop a piece.

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    list of int: The list of column indices that are not full.
    """

    valid_locations = []
    for col in range(board.shape[1]):
        if board[ROW_COUNT - 1][col] == 0:
            valid_locations.append(col)
    return valid_locations


def is_terminal_node(board):
    """
    Checks if the board is in a terminal state:
      - Either player has won
      - Or no valid moves remain

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    win_1 = winning_move(board, 1)
    win_2 = winning_move(board, 2)
    no_valid_moves = get_valid_locations(board)
    if win_1 or win_2 or not no_valid_moves:
        return True
    
    return False

def score_position(board, piece):
    """
    Evaluates the board for the given piece.
    (Already implemented to highlight center-column preference.)

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    int: Score favoring the center column. 
         (This can be extended with more advanced heuristics.)
    """
    # This is already done for you; no need to modify
    # The heuristic here scores the center column higher, which means
    # it prefers to play in the center column.
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Performs minimax with alpha-beta pruning to choose the best column.

    Parameters:
    board (np.ndarray): The current board.
    depth (int): Depth to which the minimax tree should be explored.
    alpha (float): Alpha for alpha-beta pruning.
    beta (float): Beta for alpha-beta pruning.
    maximizingPlayer (bool): Whether it's the maximizing player's turn.

    Returns:
    tuple:
        - (column (int or None), score (float)):
          column: The chosen column index (None if no moves).
          score: The heuristic score of the board state.
    """

    # Used Medium's "Connect Four" article by Simon Quellen Field for help with this function
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 1): # player win
                return (None, -1000000)
            elif winning_move(board, 2): # AI win
                return (None, 1000000)
            else: # no more valid moves
                return (None, 0) 
        else: # depth 0
            return (None, score_position(board, 2))

    if maximizingPlayer: # MAX turn (2)
        best_score = -float("inf")
        best_col = valid_locations[0]

        for col in valid_locations:
            b_copy = copy.deepcopy(board)
            row = get_next_open_row(b_copy, col)
            if row is None:
                continue
            drop_piece(b_copy, row, col, 2)

            score = minimax(b_copy, depth - 1, alpha, beta, False)[1]

            if score > best_score:
                best_score = score
                best_col = col

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return (best_col, best_score)
        
    else: # MIN tyrn (1)
        best_score = float("inf")
        best_col = valid_locations[0]

        for col in valid_locations:
            b_copy = copy.deepcopy(board)
            row = get_next_open_row(b_copy, col)
            if row is None:
                continue
            drop_piece(b_copy, row, col, 1)

            score = minimax(b_copy, depth - 1, alpha, beta, True)[1]

            if score < best_score:
                best_score = score
                best_col = col

            beta = min(beta, best_score)
            if alpha >= beta:
                break

        return (best_col, best_score)

if __name__ == "__main__":
    # Simple debug scenario
    # Example usage: create a board, drop some pieces, then call minimax
    example_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    print("Debug: Created an empty Connect Four board.\n")
    print(example_board)

    # TODO: Students can test their own logic here once implemented, e.g.:
    # drop_piece(example_board, 2, 3, 1)
    # print("Debug: Drop piece 1 in board\n")
    # print(example_board)

    # drop_piece(example_board, 5, 4, 2)
    # print("Debug: Drop piece 2 in board\n")
    # print(example_board)

    drop_piece(example_board, 5, 3, 1)
    
    print("Debug: Drop piece 4 pieces in a row in board\n")
    print(example_board)

    print(f"winning move: {winning_move(example_board, 2)}")
    print(get_valid_locations(example_board))
    print(f"terminal node: {is_terminal_node(example_board)}")
    print(f"score position 1: {score_position(example_board, 1)}")
    print(f"score position 2: {score_position(example_board, 2)}")
    
    col, score = minimax(example_board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    print("Chosen column:", col, "Score:", score)
