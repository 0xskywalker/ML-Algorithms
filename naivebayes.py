# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:56 2018

@author: ALOJOECEE
"""

from collections import Counter

class NaiveBayes():
    def __init__(self, documents):
        self.documents = documents
    def posterior(self, prob_likelihood, prior_prob, prob_all):
        try:
            posterior = (prob_likelihood * prior_prob)/prob_all
            return posterior
        except ZeroDivisionError:
            return 0
    def Vectorizer(self):
        self.agents = len(self.documents)
        self.total = 0
        for doc in self.documents:
            count = Counter(doc)
            for token in count:
                self.total += count[token]
    def predict(self, newdoc):
        self.Vectorizer()
        self.newdoc = newdoc
        self.probability = list()
        for index in range(self.agents):
            prob = 0
            for newtoken in self.newdoc:
                agent_count = Counter(self.documents[index])
                agent_tot = 0
                for distribution in agent_count:
                    agent_tot += agent_count[distribution]
                prob_likelihood = 0
                frequency = agent_count[newtoken]
                prob_likelihood = frequency/agent_tot
                prior_prob = agent_tot/self.total
                prob += self.posterior(prob_likelihood, prior_prob, self.prob_all(newtoken))
            self.probability.append(prob)
        index_of_max = 0
        for f_index in range(self.agents):
            if self.probability[index] == max(self.probability):
                index_of_max = f_index
        print("doc index", index_of_max)
    def prob_all(self, token):
        prob_all = 0
        for ndoc in self.documents:
            p_count = Counter(ndoc)
            if token in p_count:
                prob_all += p_count[token]
        return prob_all/self.total
        
