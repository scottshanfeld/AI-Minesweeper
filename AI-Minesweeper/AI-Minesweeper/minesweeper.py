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

    #def make_nn_move(self,board_section):
        #model.load_model('my_minesweeper_model.h5')
        #board_state = pd.DataFrame([board_section], columns=['(0;0)', '(1;0)', '(2;0)', '(3;0)', '(0;1)', '(1;1)', '(2;1)', '(3;1)', '(0;2)', '(1;2)', '(2;2)', '(3;2)', '(0;3)', '(1;3)', '(2;3)', '(3;3)'])
        #probs = model.predict(board_state).tolist()
        #return max(probs)

    '''def NN(self, gridVector):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler #normalize and scale feature data
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as ny
        import pandas as pd
        import seaborn as sns
        df=pd.read_csv('board_data.csv')
        df.head()
        x=df[['(0;0)', '(1;0)', '(2;0)', '(3;0)', '(0;1)', '(1;1)', '(2;1)', '(3;1)', '(0;2)', '(1;2)', '(2;2)', '(3;2)', '(0;3)', '(1;3)', '(2;3)', '(3;3)']].values
        y=df[['s(0;0)', 's(1;0)', 's(2;0)', 's(3;0)', 's(0;1)', 's(1;1)', 's(2;1)', 's(3;1)', 's(0;2)', 's(1;2)', 's(2;2)', 's(3;2)', 's(0;3)', 's(1;3)', 's(2;3)', 's(3;3)']].values
        n = x.shape[1] # n is amount of features being processed
        x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=102)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        X_test = scaler.transform(X_test)
        model = Sequential()
        model.add(Dense((10 * n), activation='relu'))  # 1st layer -- input layer
        model.add(Dense((20 * n), activation='relu'))  # 2nd layer
        model.add(Dense((10 * n), activation='relu'))  # 3rd layer
        model.add(Dense((5 * n), activation='sigmoid'))   # 4th layer
        model.add(Dense(n))                            # output (may change n to 1 if we decide to only have 1 output. Currently set up to output a percent chance of a bomb at a given space)
        model.compile(optimizer='rmsprop', loss='mse') # 'mse'-> mean square error
        model.fit(x=x_train, y=y_train, epochs=500)    #epochs -> number of pass over the entired dataset ->this is where neural network is run
        loss_df = pd.DataFrame(model.history.history)
        #Evaluation of Data -- will show us how often ai fails game (clicked a bomb)
        model.evaluate(X_test, y_test, verbose=0) # returns mean square error
        model.evaluate(x_train, y_train, verbose=0)
        test_predictions = model.predict(X_test)
        test_predictions = pd.Series(test_predictions.flatten()) #converting to dataframe
        pred_df = pd.DataFrame(y_test, columns=['orig s(0;0)', 'orig s(1;0)', 'orig s(2;0)', 'orig s(3;0)', 'orig s(0;1)', 'orig s(1;1)', 'orig s(2;1)', 'orig s(3;1)', 'orig s(0;2)', 'orig s(1;2)', 'orig s(2;2)', 'orig s(3;2)', 'orig s(0;3)', 'orig s(1;3)', 'orig s(2;3)', 'orig s(3;3)'])
        pred_df = pd.concat([pred_df, test_predictions], axis=1)
        pred_df.columns = ['orig s(0;0)', 'orig s(1;0)', 'orig s(2;0)', 'orig s(3;0)', 'orig s(0;1)', 'orig s(1;1)', 'orig s(2;1)', 'orig s(3;1)', 'orig s(0;2)', 'orig s(1;2)', 'orig s(2;2)', 'orig s(3;2)', 'orig s(0;3)', 'orig s(1;3)', 'orig s(2;3)', 'orig s(3;3)', 'Predictions']
        #Saves model
        from tensorflow.keras.models import load_model
        model.save('my_minesweeper_model.h5')
        #loads model
        my_model = load_model('my_minesweeper_model.h5')  
        board_state = pd.DataFrame ([[-3,-3,-3,-3,-3,1,-3,4,-3,-3,-3,-3,-3,-3,-3,-3]], columns = ['(0;0)', '(1;0)', '(2;0)', '(3;0)', '(0;1)', '(1;1)', '(2;1)', '(3;1)', '(0;2)', '(1;2)', '(2;2)', '(3;2)', '(0;3)', '(1;3)', '(2;3)', '(3;3)'])
        model.predict(board_state)        
'''
