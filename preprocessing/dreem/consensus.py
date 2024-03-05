"""
This module provides a class to build a consensus hypnogram from multiple scorers.
This file was mostly copied from https://github.com/Dreem-Organization/dreem-learning-evaluation/blob/master/evaluation.py
and slightly adapted to fit the project structure.
"""
import json
import os

import numpy as np
from tabulate import tabulate


def compute_soft_agreement(hypnogram, hypnograms_consensus):
    epochs = range(len(hypnogram))
    probabilistic_consensus = np.zeros((6, len(hypnogram)))
    for hypnogram_consensus in hypnograms_consensus:
        probabilistic_consensus[np.array(hypnogram_consensus) + 1, range(len(hypnogram))] += 1
    probabilistic_consensus_normalized = probabilistic_consensus / probabilistic_consensus.max(0)
    soft_agreement = probabilistic_consensus_normalized[np.array(hypnogram) + 1, epochs].mean()
    return soft_agreement


def build_consensus_hypnogram(ranked_hypnograms):
    """In this function, order matters, first hypnogram is the reference in case of ties"""
    number_of_epochs = len(ranked_hypnograms[0])
    probabilistic_consensus = np.zeros((6, number_of_epochs))
    ranked_hypnograms_consensus_array = np.array(ranked_hypnograms) + 1
    for hypnogram_consensus in ranked_hypnograms_consensus_array:
        probabilistic_consensus[np.array(hypnogram_consensus), range(number_of_epochs)] += 1

    consensus_hypnogram = np.argmax(probabilistic_consensus, 0)
    ties = (
                   probabilistic_consensus ==
                   probabilistic_consensus[consensus_hypnogram, range(number_of_epochs)]
           ).sum(0) > 1
    # consensus_hypnogram[ties] = np.array(ranked_hypnograms_consensus_array[0])[ties]
    # FIX: if we just use the score of the first scorer (the scorer with the highest soft agreement) as the consensus
    # when we have ties, we might use a score that was NOT part of the tie (e.g., if the scores of an epoch are
    # [0, 1, 1, 2, 2], then we have a tie and would use the first score (0) as the resolution, although the tie is
    # between the scores 1 and 2).
    for tie_epoch in np.where(ties)[0]:
        classes = np.where(probabilistic_consensus[:, tie_epoch] == np.max(probabilistic_consensus[:, tie_epoch]))[0]
        consensus_hypnogram[tie_epoch] = [score for score in ranked_hypnograms_consensus_array[:, tie_epoch] if
                                          score in classes][0]
    # END FIX

    consensus_probability = (probabilistic_consensus[consensus_hypnogram, range(number_of_epochs)] /
                             len(ranked_hypnograms_consensus_array))
    consensus_hypnogram = consensus_hypnogram - 1
    return consensus_hypnogram, consensus_probability


class ConsensusBuilder:
    def __init__(self,
                 scorers_folder,
                 record_blacklist=[],
                 lights_off={},
                 lights_on={},
                 start_times=None):
        # Retrieve scorers
        self.scorers = os.listdir(scorers_folder)
        self.index = {}
        self.scorers_folder = {
            scorer: f'{scorers_folder}{scorer}/'
            for scorer in self.scorers
        }
        # Intersection of all available scored records
        self.records = sorted(list(set.intersection(*(
            {record.split(".json")[0] for record in os.listdir(self.scorers_folder[scorer]) if
             record.split(".json")[0] not in record_blacklist}
            for scorer in self.scorers
        ))))
        print(f"Found {len(self.records)} records and {len(self.scorers)} scorers.")

        # Retrieve scorer hypnograms
        self.scorer_hypnograms = {
            scorer: {
                record: np.array(
                    json.load(open(f"{self.scorers_folder[scorer]}/{record}.json", "r")))
                for record in self.records
            }
            for scorer in self.scorers
        }
        self.hypnogram_sizes = {}
        for record in self.records:
            hypnogram_size = set([
                len(self.scorer_hypnograms[scorer][record]) for scorer in self.scorers
            ])
            assert len(hypnogram_size) == 1
            self.hypnogram_sizes[record] = hypnogram_size.pop()

        # Cut hypnograms to light on and off
        for record in self.records:
            # if results hypnogram length is the same as scorer hypno, we have to truncate it

            hypnograms = [self.scorer_hypnograms[scorer][record] for scorer in self.scorers]
            index_min = max([np.where(np.array(hypnogram) >= 0)[0][0]
                             for hypnogram in hypnograms])  # NG: remove continuous artifacts at the start of recording
            index_min = max(index_min, lights_off.get(record, 0))
            # NG: remove continuous artifacts at the end of recording, artifacts in middle epochs (not continuously
            # connected to start or end of the recording) are not removed
            index_max = (len(hypnograms[0]) - max([np.where(np.array(hypnogram)[::-1] >= 0)[0][0]
                                                   for hypnogram in hypnograms]))
            index_max = min(index_max, lights_on.get(record, np.inf))
            self.index[record] = index_min, index_max
            for scorer in self.scorers:
                # removes light on, off and update hypnogram size accordingly
                self.scorer_hypnograms[scorer][record] = self.scorer_hypnograms[scorer][record][
                                                         index_min:index_max]
                self.hypnogram_sizes[record] += index_max - self.hypnogram_sizes[record] - index_min

        # Build up scorer ranking
        self.scorers_ranking = {
            record: sorted(
                self.scorers,
                key=lambda scorer: -compute_soft_agreement(
                    self.scorer_hypnograms[scorer][record],
                    [self.scorer_hypnograms[other_scorer][record]
                     for other_scorer in self.scorers if other_scorer != scorer],
                )
            )
            for record in self.records
        }
        self.scorers_soft_agreement = [
            (scorer,
             np.mean([
                 compute_soft_agreement(
                     self.scorer_hypnograms[scorer][record],
                     [self.scorer_hypnograms[other_scorer][record]
                      for other_scorer in self.scorers if other_scorer != scorer]
                 ) for record in self.records
             ]))
            for scorer in sorted(self.scorers)
        ]

        # Build consensus hypnogram for scorers
        self.scorer_hypnograms_consensus = {
            scorer: {
                record: build_consensus_hypnogram(
                    [self.scorer_hypnograms[other_scorer][record] for other_scorer in
                     self.scorers_ranking[record]
                     if other_scorer != scorer]
                )
                for record in self.records
            }
            for scorer in self.scorers
        }

        # Build consensus hypnogram for all scorers
        self.result_hypnograms_consensus = {
            record: build_consensus_hypnogram(
                [self.scorer_hypnograms[scorer][record]
                 for scorer in self.scorers_ranking[record][:]]  # all scorings
            )
            for record in self.records
        }

        # Build consensus hypnogram for all scorers but the weakest one
        self.result_hypnograms_consensus_n_minus_one = {
            record: build_consensus_hypnogram(
                [self.scorer_hypnograms[scorer][record]
                 for scorer in self.scorers_ranking[record][:-1]]  # N - 1 scorings
            )
            for record in self.records
        }

    def print_soft_agreements(self):
        print(
            tabulate(
                self.scorers_soft_agreement,
                headers=["Scorer", "SoftAgreement"],
                tablefmt="fancy_grid"
            )
        )
