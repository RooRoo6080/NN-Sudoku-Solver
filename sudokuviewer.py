import csv

def print_sudoku(string):
    board = []
    n = 0
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(int(string[n]))
            n += 1
        board.append(row)
    print("+" + "---+"*9)
    for i, row in enumerate(board):
        print(("|" + " {}   {}   {} |" * 3).format(*[x if x != 0 else " " for x in row]))
        if i % 3 == 2:
            print("+" + "---+" * 9)
        else:
            print("+" + "   +" * 9)

option = 56

with open('sudoku.csv', mode ='r')as file:
  reader = csv.reader(file)
  rows = list(reader)
  print("Unsolved:")
  print_sudoku((rows[option][0]))
  print("Solution:")
  print_sudoku((rows[option][1]))
