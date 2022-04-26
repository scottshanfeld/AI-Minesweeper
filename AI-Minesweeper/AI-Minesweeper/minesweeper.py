from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #normalize and scale feature data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

import itertools
import random
import numpy as ny
import pandas as pd
import seaborn as sns



class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines

    def getSurrounding(self,cell):
        queue = []
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    queue.append((i, j))

        return queue

    def scan_section(self,start, size, revealed):
        board_data = []
        for i in range(size):
            x = i + start[0]
            for j in range(size):
                y = j + start[1]
                if(x >= self.height or y >= self.width): #marks area outside board as -2
                    board_data.append(-2)
                elif (self.is_mine((x, y))):  # marks bomb tiles as -1
                    board_data.append(-1)
                elif((x,y) in revealed):
                    board_data.append(self.nearby_mines((x,y))) #marks tile with number of bombs surround otherwise
                else:
                    board_data.append(-3) #markes tile with -3 if unknown
        return(board_data)

    def scan_all_sections(self, size, revealed):
        sections = []
        for i in range(0,self.height, size):
            for j in range(0, self.width, size):
                sections.append(self.scan_section((i,j),size, revealed))
        return sections

    def scan_all_sections_with_safes(self, size, revealed):
        sections = []
        section = []
        safes_section = []
        for i in range(0,self.height, size):
            for j in range(0, self.width, size):
                section = self.scan_section((i,j),size, revealed)
                safes_section = []
                for k in range(len(section)):
                    if(section[k] == -1):
                        section[k] = -3
                        safes_section.append(0)
                    else:
                        safes_section.append(1)
                sections.append(section + safes_section)
        return sections



class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        known_mines = set()
        if len(self.cells)==self.count:
            for cell in self.cells:
                #cell = True
                known_mines.add(cell)
            #print(f'known_mines is {known_mines}')
            return known_mines
        else:
            return set()


    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        known_safes = set()
        if self.count==0:
            for cell in self.cells:
                #cell = False
                known_safes.add(cell)
            #print(f'known_safes is {known_safes}')
            return known_safes
        else:
            return set()


    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1


    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)

class MinesweeperAI(): #This is the main function to edit
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []
        self.bomb_chance = {}
        self.bomb_chance_surrounding = {}
        for i in range(height):
            for j in range(width):
                coord = str(i)+ ", "+ str(j)
                self.bomb_chance.update({coord : 0})
        for k in range(height):
            for m in range(width):
                coord = str(k)+ ", "+ str(m)
                self.bomb_chance_surrounding.update({coord : 0})
        print(self.bomb_chance)

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)       #1.
        #print("1 executed")
        
        self.mark_safe(cell)            #2.
        #print("2 executed")
                
        cells = set()                   #3.
        
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if 0 <= i < self.height and 0 <= j < self.width:
                    if (i,j)!=cell and (i,j) not in (self.safes or self.mines):
                        cells.add((i,j))
        if len(cells)!= 0 and Sentence(cells,count) not in self.knowledge:
            self.knowledge.append(Sentence(cells,count))
        #print("3 executed")
    
        for sent in self.knowledge:     #4.
            self.mines = self.mines.union(sent.known_mines())
            self.safes = self.safes.union(sent.known_safes())
        #print("4 executed")
        

        knowledge_subset = []
        for i in self.knowledge:        #5.
            if len(i.cells)==0:
                self.knowledge.remove(i)
                continue
            for j in self.knowledge:
                if i.cells != j.cells and i.cells.issubset(j.cells) and Sentence(j.cells.difference(i.cells),j.count-i.count) not in self.knowledge and j.count!=0 :
                    self.knowledge.append(Sentence(j.cells.difference(i.cells),j.count-i.count))
                
                
        print(len(self.moves_made.union(self.mines)))
        #print("5 executed")
               
        #return self.knowledge
        


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        move = None
        if len(self.safes.difference(self.moves_made))!=0:
            move = random.sample(self.safes.difference(self.moves_made),1)[0]
            #print(f"make safe move running {move}")
            return move
        else:
            return None
        


    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        while len(self.moves_made.union(self.mines))< 64-len(self.mines):
            i,j = random.randrange(0,self.width),random.randrange(0,self.height)
            if (i,j) in self.moves_made.union(self.mines):
                self.make_random_move()
            else:
                #print(f"make random move running {(i,j)}")
                return (i,j)
        return None

    def make_nn_move(self,board_section,i,j):
        availMoves = 0
        board_section = [-3 if item == -1 else item for item in board_section] #replaces bomb tiles with unknown
        print(board_section)
        if board_section == [-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3]: #board with no info
            print("EMPTY")
            return
        import tensorflow as tf
        from tensorflow import keras
        model = tf.keras.models.load_model('my_minesweeper_model.h5')

        board_state = pd.DataFrame([board_section], columns=['(0;0)', '(1;0)', '(2;0)', '(3;0)', '(0;1)', '(1;1)', '(2;1)', '(3;1)', '(0;2)', '(1;2)', '(2;2)', '(3;2)', '(0;3)', '(1;3)', '(2;3)', '(3;3)'])
        probs = model.predict(board_state).tolist()
        print(probs)
        maxProb = -9999
        maxCoord = (0,0)
        for ndx in range(len(probs[0])):
            y = int(ndx / 4)
            x = ndx % 4
            coord = (x, y)
            if (y+j, x+i) not in self.moves_made:
                availMoves += 1
            if(((y+j, x+i) not in self.moves_made) and (probs[0][ndx] > maxProb)):
                print(probs[0][ndx], ": ", coord)
                maxProb = probs[0][ndx]
                maxCoord = coord
        print("Move Within Section: ", (maxCoord))

        if maxProb != -9999 and availMoves > 1:
            print("index:", probs[0].index(maxProb))
            return maxCoord
        else:
            print("No moves in this section.")





