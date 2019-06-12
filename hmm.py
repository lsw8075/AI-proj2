# -*- coding: utf-8 -*-
# Copyright (c) 2012, Chi-En Wu

# Use HMM code (BSD licence) by jason2506
# from https://github.com/jason2506/PythonHMM/blob/master/hmm.py

import pickle # for dump
from math import log

def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
        else:
            for item in item_set:
                result[item] = 0

    return result

def _normalize_prob_two_dim(prob, item_set1, item_set2):
    result = {}
    if prob is None:
        for item in item_set1:
            result[item] = _normalize_prob(None, item_set2)
    else:
        for item in item_set1:
            result[item] = _normalize_prob(prob.get(item), item_set2)

    return result

def _count(item, count):
    if item not in count:
        count[item] = 0
    count[item] += 1
    return count

def _count_two_dim(item1, item2, count):
    if item1 not in count:
        count[item1] = {}
    count[item1] = _count(item2, count[item1])
    return count

def _get_init_model(sequences):
    symbol_count = {}
    state_count = {}
    state_symbol_count = {}
    state_start_count = {}
    state_trans_count = {}

    for state_list, symbol_list in sequences:
        pre_state = None
        for state, symbol in zip(state_list, symbol_list):
            state_count = _count(state, state_count)
            symbol_count = _count(symbol, symbol_count)
            state_symbol_count = _count_two_dim(state, symbol, state_symbol_count)
            if pre_state is None:
                state_start_count = _count(state, state_start_count)
            else:
                state_trans_count = _count_two_dim(pre_state, state, state_trans_count)
            pre_state = state
    return Model(state_count.keys(), symbol_count.keys(),
        state_start_count, state_trans_count, state_symbol_count)

def train(sequences, delta=0.0001, smoothing=0):
    """
    Use the given sequences to train a HMM model.
    This method is an implementation of the `EM algorithm
    <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_.

    The `delta` argument (which is defaults to 0.0001) specifies that the
    learning algorithm will stop when the difference of the log-likelihood
    between two consecutive iterations is less than delta.

    The `smoothing` argument is used to avoid zero probability,
    see :py:meth:`~hmm.Model.learn`.
    """

    model = _get_init_model(sequences)
    length = len(sequences)

    old_likelihood = 0
    for _, symbol_list in sequences:
        eval_result = model.evaluate(symbol_list)
        if eval_result == 0.0:
            old_likelihood += -720
        else:
            old_likelihood += log(eval_result)

    old_likelihood /= length
    print("old_likelihood", old_likelihood)
    return model

    itercount = 1
    while True:
        new_likelihood = 0
        for _, symbol_list in sequences:
            model.learn(symbol_list, smoothing)

            eval_result = model.evaluate(symbol_list)
            if eval_result == 0.0:
                new_likelihood += -720
            else:
                new_likelihood += log(eval_result)

        new_likelihood /= length

        val = abs(new_likelihood - old_likelihood)
        
        print("Iteration ", itercount, "ends, value", val)
        if val < delta:
            break

        old_likelihood = new_likelihood
        itercount += 1

    return model

class Model(object):
    """
    This class is an implementation of the Hidden Markov Model.

    The instance of this class can be created by passing the given states,
    symbols and optional probability matrices.

    If any of the probability matrices are not given, the missing matrics
    will be set to the initial uniform probability.
    """
    
    def __init__(self, states=None, symbols=None, start_prob=None, trans_prob=None, emit_prob=None):
        if (states is None) or (symbols is None):
            return
        self._states = set(states)
        self._symbols = set(symbols)
        self._start_prob = _normalize_prob(start_prob, self._states)
        self._trans_prob = _normalize_prob_two_dim(trans_prob, self._states, self._states)
        self._emit_prob = _normalize_prob_two_dim(emit_prob, self._states, self._symbols)

    def __repr__(self):
        return '{name}({_states}, {_symbols}, {_start_prob}, {_trans_prob}, {_emit_prob})' \
            .format(name=self.__class__.__name__, **self.__dict__)

    def load(self, filename):
        data = pickle.load(open(filename, 'rb'))
        self._states = data[0]
        self._symbols = data[1]
        self._start_prob = data[2]
        self._trans_prob = data[3]
        self._emit_prob = data[4]

    def dump(self, filename):
        mydata = (self._states, self._symbols, self._start_prob, self._trans_prob, self._emit_prob)
        pickle.dump(mydata, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    def states(self):
        """ Return the state set of this model. """
        return set(self._states)

    def states_number(self):
        """ Return the number of states. """
        return len(self._states)

    def symbols(self):
        """ Return the symbol set of this model. """
        return set(self._symbols)

    def symbols_number(self):
        """ Return the number of symbols. """
        return len(self._symbols)

    def start_prob(self, state):
        """
        Return the start probability of the given state.

        If `state` is not contained in the state set of this model, 0 is returned.
        """
        if state not in self._states:
            return 0
        return self._start_prob[state]

    def trans_prob(self, state_from, state_to):
        """
        Return the probability that transition from state `state_from` to
        state `state_to`.

        If either the `state_from` or the `state_to` are not contained in the
        state set of this model, 0 is returned.
        """
        if state_from not in self._states or state_to not in self._states:
            return 0
        return self._trans_prob[state_from][state_to]

    def emit_prob(self, state, symbol):
        """
        Return the emission probability for `symbol` associated with the `state`.

        If either the `state` or the `symbol` are not contained in this model,
        0 is returned.
        """
        if state not in self._states or symbol not in self._symbols:
            return 0
        return self._emit_prob[state][symbol]

    def _forward(self, sequence):
        
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        alpha = [{}]
        for state in self._states:
            alpha[0][state] = self.start_prob(state) * self.emit_prob(state, sequence[0])
        
        for index in range(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                prob = 0
                for state_from in self._states:
                    prob += alpha[index - 1][state_from] * \
                        self.trans_prob(state_from, state_to)
                alpha[index][state_to] = prob * self.emit_prob(state_to, sequence[index])
        return alpha

    def _backward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        beta = [{}]
        for state in self._states:
            beta[0][state] = 1

        for index in range(sequence_length - 1, 0, -1):
            beta.insert(0, {})
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    prob += beta[1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index])
                beta[0][state_from] = prob

        return beta

    def evaluate(self, sequence):
        """
        Use the `forward algorithm
        <http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm>`_
        to evaluate the given sequence.
        """
        length = len(sequence)
        if length == 0:
            return 0

        prob = 0
        alpha = self._forward(sequence)
        for state in alpha[length - 1]:
            prob += alpha[length - 1][state]
        return prob

    def decode(self, sequence):
        """
        Decode the given sequence.

        This method is an implementation of the
        `Viterbi algorithm <http://en.wikipedia.org/wiki/Viterbi_algorithm>`_.
        """
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        delta = {}
        for state in self._states:
            delta[state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

        pre = []
        for index in range(1, sequence_length):
            delta_bar = {}
            pre_state = {}
            for state_to in self._states:
                max_prob = 0
                max_state = None
                for state_from in self._states:
                    prob = delta[state_from] * self.trans_prob(state_from, state_to)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from
                delta_bar[state_to] = max_prob * self.emit_prob(state_to, sequence[index])
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0
        for state in self._states:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state

        if max_state is None:
            return []

        result = [max_state]
        for index in range(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)

        return result

    def learn(self, sequence, smoothing=0):
        """
        Use the given `sequence` to find the best state transition and
        emission probabilities.

        The optional `smoothing` argument (which is defaults to 0) is the
        smoothing parameter of the
        `additive smoothing <http://en.wikipedia.org/wiki/Additive_smoothing>`_
        to avoid zero probability.
        """
        length = len(sequence)
        alpha = self._forward(sequence)
        beta = self._backward(sequence)

        #print("alpha", alpha)
        #print("beta", beta)
        gamma = []
        for index in range(length):
            prob_sum = 0
            gamma.append({})
            for state in self._states:
                prob = alpha[index][state] * beta[index][state]
                gamma[index][state] = prob
                prob_sum += prob

            if prob_sum == 0:
                continue

            for state in self._states:
                gamma[index][state] /= prob_sum

        xi = []
        for index in range(length - 1):
            prob_sum = 0
            xi.append({})
            for state_from in self._states:
                xi[index][state_from] = {}
                for state_to in self._states:
                    prob = alpha[index][state_from] * beta[index + 1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index + 1])
                    xi[index][state_from][state_to] = prob
                    prob_sum += prob

            if prob_sum == 0:
                continue

            for state_from in self._states:
                for state_to in self._states:
                    xi[index][state_from][state_to] /= prob_sum

        states_number = len(self._states)
        symbols_number = len(self._symbols)
        for state in self._states:
            # update start probability
            self._start_prob[state] = \
                (smoothing + gamma[0][state]) / (1 + states_number * smoothing)

            # update transition probability
            gamma_sum = 0
            for index in range(length - 1):
                gamma_sum += gamma[index][state]

            if gamma_sum > 0:
                denominator = gamma_sum + states_number * smoothing
                for state_to in self._states:
                    xi_sum = 0
                    for index in range(length - 1):
                        xi_sum += xi[index][state][state_to]
                    self._trans_prob[state][state_to] = (smoothing + xi_sum) / denominator
            else:
                for state_to in self._states:
                    self._trans_prob[state][state_to] = 0

            # update emission probability
            gamma_sum += gamma[length - 1][state]
            emit_gamma_sum = {}
            for symbol in self._symbols:
                emit_gamma_sum[symbol] = 0

            for index in range(length):
                emit_gamma_sum[sequence[index]] += gamma[index][state]

            if gamma_sum > 0:
                denominator = gamma_sum + symbols_number * smoothing
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = \
                        (smoothing + emit_gamma_sum[symbol]) / denominator
            else:
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = 0

