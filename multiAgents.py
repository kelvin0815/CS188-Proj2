# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        weights = [0.2, 0.2, 0.6]
        score = weights[0]*successorGameState.getScore()

        newGhostPos = successorGameState.getGhostPositions()
        newFoodPos = newFood.asList()
        capsulePos = currentGameState.getCapsules()

        if newFoodPos:
            minDistToFood = min([manhattanDistance(newPos, food) for food in newFoodPos])
            score += weights[1]/minDistToFood
        if newGhostPos:
            minDistToGhost = min([manhattanDistance(newPos, ghost) for ghost in newGhostPos])
            if minDistToGhost < 3 and max(newScaredTimes) == 0:
                score += weights[2]*minDistToGhost
        if capsulePos and newPos in capsulePos:
            score += 500

        return score

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

    def isTerminal(self, gameState):
        return gameState.isWin() or gameState.isLose()

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maxValue(self, gameState, depth):
        """
        return a tuple of the maximum value of successors and corresponding action
        """
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        v = [-float("inf"), None]
        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, act) for act in legalActions]
        minValsOfSucc = [self.minValue(succ, depth, 1)[0] for succ in successors]
        v[0] = max(minValsOfSucc)
        v[1] = legalActions[minValsOfSucc.index(v[0])]
        return tuple(v)

    def minValue(self, gameState, depth, agentIndex):
        """
        return a tuple of the mimimum value of successors and corresponding action
        """
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        v = [float("inf"), None]
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, act) for act in legalActions]
        numOfAgents = gameState.getNumAgents()
        # ghost2
        if agentIndex == numOfAgents-1:
            # case 1:   if not reach terminal states, next action is under pacman's control
            if depth != self.depth:
                valsOfSucc = [self.maxValue(succ, depth+1)[0] for succ in successors]
            # case 2:   if reach terminal states
            else:
                valsOfSucc = [self.evaluationFunction(succ) for succ in successors]
        # ghost1, next action is under ghost2's control
        else:
            valsOfSucc = [self.minValue(succ, depth, agentIndex+1)[0] for succ in successors]
        v[0] = min(valsOfSucc)
        v[1] = legalActions[valsOfSucc.index(v[0])]
        return tuple(v)

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
        # returns a action that leads to minimax value
        return self.maxValue(gameState, 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, depth, alpha, beta):
        """
        return a tuple of the maximum value of successors and corresponding action,
        with alpha-beta pruning
        """
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        v = [-float("inf"), None]
        legalActions = gameState.getLegalActions(0)
        for act in legalActions:
            succ = gameState.generateSuccessor(0, act)
            valOfSucc = self.minValue(succ, depth, 1, alpha, beta)[0]
            if valOfSucc > v[0]:
                v[1] = act
            v[0] = max(v[0], valOfSucc)
            # pruning
            if v[0] > beta:
                return v
            # refreshing alpha
            alpha = max(alpha, v[0])
        return tuple(v)

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        """
        return a tuple of the mimimum value of successors and corresponding action,
        with alpha-beta pruning
        """
        numAgents = gameState.getNumAgents()
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        v = [float("inf"), None]
        legalActions = gameState.getLegalActions(agentIndex)
        for act in legalActions:
            succ = gameState.generateSuccessor(agentIndex, act)
            # ghost 2
            if agentIndex == numAgents-1:
                # case 1:   if not reach terminal states, pacman is the successor
                if depth != self.depth:
                    valOfSucc = self.maxValue(succ, depth+1, alpha, beta)[0]
                # case 2:   if reach terminal states
                else:
                    valOfSucc = self.evaluationFunction(succ)
            # ghost 1
            else:
                valOfSucc = self.minValue(succ, depth, agentIndex+1, alpha, beta)[0]
            if valOfSucc < v[0]:
                v[1] = act
            v[0] = min(v[0], valOfSucc)
            # pruning
            if v[0] < alpha:
                return v
            # refreshing beta
            beta = min(beta, v[0])
        return tuple(v)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1, -float("inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, gameState, depth):
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        legalActions = gameState.getLegalActions(0)
        v = [-float("inf"), None]
        successors = [gameState.generateSuccessor(0, act) for act in legalActions]
        valsOfSucc = [self.expValue(succ, depth, 1)[0] for succ in successors]
        v[0] = max(valsOfSucc)
        v[1] = legalActions[valsOfSucc.index(v[0])]
        return tuple(v)

    def expValue(self, gameState, depth, agentIndex):
        if self.isTerminal(gameState):
            return (self.evaluationFunction(gameState), None)
        numOfAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, act) for act in legalActions]
        prob = 1.0/len(legalActions)
        v = [0, None]
        if agentIndex == numOfAgents-1:
            if depth != self.depth:
                valsOfSucc = [self.maxValue(succ, depth+1)[0] for succ in successors]
            else:
                valsOfSucc = [self.evaluationFunction(succ) for succ in successors]
        else:
            valsOfSucc = [self.expValue(succ, depth, agentIndex+1)[0] for succ in successors]
        v[0] = sum(valsOfSucc)*prob
        # Not necessary to record the action to take on expect-node
        return tuple(v)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      Important features:
      (a)   current score
      (b)   distance to nearest food
      (c)   distance to nearest ghost
      (d)   distance to nearest capsule
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    # By experiment
    weights = [0.5, 0.2, 0.2, 0.1]
    val = weights[0]*currentGameState.getScore()

    ghostPos = currentGameState.getGhostPositions()
    foodPos = currentGameState.getFood().asList()
    capsulePos = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    distToFoods = [float(manhattanDistance(currPos, food)) for food in foodPos]
    distToGhosts = [manhattanDistance(currPos, ghost) for ghost in ghostPos]
    if distToFoods:
        val += weights[1]/min(distToFoods)
    if distToGhosts:
        if max(scaredTimes) < 3:
            # Heavy penalty if the pacman is too close to ghost
            val -= weights[2]*10.0/(1+min(distToGhosts))
        else:
            val += weights[2]/(1+min(distToGhosts))
    if capsulePos:
        distToCapsules = [manhattanDistance(currPos, cap) for cap in capsulePos]
        val += weights[3]/min(distToCapsules)
    return val

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
