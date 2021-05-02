# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()
        "*** YOUR CODE HERE ***"
        newFood_list = newFood.asList()
        closestFood = 1000000
        for i in newGhostPositions:
            manDist = manhattanDistance(newPos, i)
            if (manDist < 3):
                return -1000000
        for j in newFood_list:
            manDist = manhattanDistance(newPos, j)
            closestFood = min(closestFood, manDist)
        return score + pow(closestFood,-1)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(currentAgent_index, deepness, gameState):
            if deepness == self.depth or gameState.isLose() or gameState.isWin():
                score = self.evaluationFunction(gameState)
                return score
            else:
                pass
            if currentAgent_index != 0:
                newAgent_index = currentAgent_index + 1
                if gameState.getNumAgents() == newAgent_index:
                    newAgent_index = 0
                else:
                    pass
                if newAgent_index == 0:
                    deepness = deepness + 1
                else:
                    pass
                score = min(minimax(newAgent_index, deepness, gameState.generateSuccessor(currentAgent_index, newState)) for newState in gameState.getLegalActions(currentAgent_index))
                return score
            else:
                score = max(minimax(1, deepness, gameState.generateSuccessor(currentAgent_index, newState)) for newState in gameState.getLegalActions(currentAgent_index))
                return score
        max_score = -1000000
        best_action = 0
        for agentState in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if max_score == -1000000 or score > max_score:
                max_score = score
                best_action = agentState
        return best_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def min_score(deepness, currentAgent_index, game_state, x, y):
            unrelaxedValue = 1000000
            newAgent_index = currentAgent_index + 1
            if game_state.getNumAgents() == newAgent_index:
                newAgent_index = 0
            else:
                pass
            if newAgent_index == 0:
                deepness = deepness + 1
            else:
                pass
            for newState in game_state.getLegalActions(currentAgent_index):
                unrelaxedValue = min(unrelaxedValue, abPrune(newAgent_index, deepness, game_state.generateSuccessor(currentAgent_index, newState), x, y))
                if unrelaxedValue < x:
                    return unrelaxedValue
                else:
                    pass
                y = min(y, unrelaxedValue)
            return unrelaxedValue
        def max_score(deepness, currentAgent_index, game_state, x, y):
            unrelaxedValue = -1000000
            for newState in game_state.getLegalActions(currentAgent_index):
                unrelaxedValue = max(unrelaxedValue, abPrune(1, deepness, game_state.generateSuccessor(currentAgent_index, newState),x ,y))
                if unrelaxedValue > y:
                    return unrelaxedValue
                else:
                    pass
                x = max(x, unrelaxedValue)
            return unrelaxedValue

        def abPrune(currentAgent_index, deepness, game_state, x, y):
            if deepness == self.depth or game_state.isLose() or game_state.isWin():
                return self.evaluationFunction(game_state)
            else:
                pass
            if currentAgent_index != 0:
                return min_score(deepness, currentAgent_index, game_state, x, y)
            else:
                return max_score(deepness, currentAgent_index, game_state, x, y)
        alpha = -1000000
        beta = 1000000
        score = -1000000
        best_action = 0
        for i in gameState.getLegalActions(0):
            ghostScore = abPrune(1, 0, gameState.generateSuccessor(0, i), alpha, beta)
            if ghostScore > score:
                best_action = i
                score = ghostScore
            else:
                pass
            if score > beta:
                return score
            else:
                pass
            alpha = max(alpha, score)
        return best_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(currentAgent_index, deepness, gameState):
            if deepness == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            if currentAgent_index == 0:
                score = max(expectiMax(1, deepness, gameState.generateSuccessor(currentAgent_index, newState)) for newState in gameState.getLegalActions(currentAgent_index))
                return score
            else:
                newAgent_index = currentAgent_index + 1
                if gameState.getNumAgents() == newAgent_index:
                    newAgent_index = 0
                else:
                    pass
                if newAgent_index == 0:
                    deepness = deepness + 1
                else:
                    pass
                probability = sum(expectiMax(newAgent_index, deepness, gameState.generateSuccessor(currentAgent_index, newState)) for newState in gameState.getLegalActions(currentAgent_index)) / float(len(gameState.getLegalActions(currentAgent_index)))
                return probability
        max_score = -1000000
        best_action = 0
        for i in gameState.getLegalActions(0):
            score = expectiMax(1, 0, gameState.generateSuccessor(0, i))
            if score > max_score or max_score == -1000000:
                max_score = score
                best_action = i
            else:
                pass
        return best_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPositions = currentGameState.getGhostPositions()
    score = currentGameState.getScore()
    newFood_list = newFood.asList()
    closestFood = 1000000
    for i in newGhostPositions:
        manDist = manhattanDistance(newPos, i)
        if (manDist < 3):
            return -1000000
    for j in newFood_list:
        manDist = manhattanDistance(newPos, j)
        closestFood = min(closestFood, manDist)
    return score + pow(closestFood, -1)
    util.raiseNotDefined()
# Abbreviation
better = betterEvaluationFunction

