# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0



        "*** YOUR CODE HERE ***"
        iterationCount = 0
        while iterationCount <= iterations:
          valuesCopy = self.values.copy()
          delta = 0
          for s in mdp.getStates():
            if self.mdp.isTerminal(s):
              self.values[s] = self.mdp.getReward(s, None, None)
            else:            
              possibleActions = self.mdp.getPossibleActions(s)
              actionEVs = []
              for a in possibleActions:
                resultValue = 0
                for r in self.mdp.getTransitionStatesAndProbs(s,a):
                  resultValue += valuesCopy[r[0]]*r[1]
                actionEVs.append(resultValue)
              self.values[s] = self.mdp.getReward(s, None, None) + discount*(max(actionEVs))
          iterationCount += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #print("TAKING A PEEK AT VALUES")
        #print("value of state is: ", self.values[state])
        if (self.mdp.isTerminal(state)):
          return None
        else:
          actionValue = 0
          for (nextState, p) in self.mdp.getTransitionStatesAndProbs(state, action):
            #print("value of nextState is: ", self.values[nextState])
            actionValue += self.values[nextState]*p
          return actionValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if (self.mdp.isTerminal(state)):
          return None
        else:
          bestAction = [None, -99999]
          for a in self.mdp.getPossibleActions(state):
            currAction = [None, 0]
            for (nextState, p) in self.mdp.getTransitionStatesAndProbs(state, a):
              currAction[1] += self.values[nextState]*p
            if currAction[1] > bestAction[1]:
              bestAction[0] = a
              bestAction[1] = currAction[1]
          return bestAction[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
