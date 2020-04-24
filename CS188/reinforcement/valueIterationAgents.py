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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for state in self.mdp.getStates():
          self.values[state] = 0
        for i in range(self.iterations):
          newValues = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              newValues[state] = self.mdp.getReward(state, 'pass', state)
              continue
            qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
            maxQ = max(qValues)
            newValues[state] = maxQ
          self.values = newValues


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
        qValue = 0
        TRs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in TRs:
          qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None
        dic = util.Counter()
        for action in self.mdp.getPossibleActions(state):
          dic[action] = self.computeQValueFromValues(state, action)
        return dic.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for state in self.mdp.getStates():
          self.values[state] = 0
        i = 0
        while True:
          for state in self.mdp.getStates():
            if (i >= self.iterations):
              return
            i+=1
            if self.mdp.isTerminal(state):
              continue
            qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
            maxQ = max(qValues)
            self.values[state] = maxQ

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessor = util.Counter()
        for s in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(s):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(s, action):
                    if nextState in predecessor:
                        predecessor[nextState].add(s)
                    else:
                        predecessor[nextState] = set([s])
        priorityQueue = util.PriorityQueue()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                qValues = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
                diff = abs(self.getValue(s) - max(qValues))
                priorityQueue.push(s, -diff)
        for iteration in range(self.iterations):
            if priorityQueue.isEmpty():
                return
            s = priorityQueue.pop()
            if not self.mdp.isTerminal(s):
                qValues = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
                self.values[s] = max(qValues)
            for p in predecessor[s]:
                if not self.mdp.isTerminal(p):
                    qValues = [self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]
                    diff = abs(self.getValue(p) - max(qValues))
                    if diff > self.theta:
                        priorityQueue.update(p, -diff)



