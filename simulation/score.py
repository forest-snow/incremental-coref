import torch
import numpy as np
import time
import json
import logging
from tqdm import tqdm
from incremental import Incremental
from inference import Predictor
from cluster import ClusterList
import sys
from metrics import CorefEvaluator
from collections import defaultdict
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
import pathlib
import util

class Scorer():
    """
    The Scorer class scores spans in [data] using acquisition [model].
    Functions implement different ways to score spans.
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.reset()

    def reset(self):
        self.model.reset_metrics()
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

    def random(self, **kwargs):
        """
        Random sampling but with more spans retained.
        """
        # dont prune spans
        self.model.span_scorer.top_span_ratio = 20
        span_scores = self.mention_detection()
        for span_score in span_scores:
            span_score['score'] = np.random.uniform()
        return span_scores

    def random_ment(self, **kwargs):
        """Random sampling of retained spans"""
        # call mention_detection to prune spans
        span_scores = self.mention_detection()
        for span_score in span_scores:
            span_score['score'] = np.random.uniform()
        return span_scores

    def loop_clusters(self, segment, document, start_idx):
        doc_embs = self.model.encoder(segment, f"{document['doc_key']}_{start_idx}")
        doc_embs = doc_embs.detach()
        num_words = doc_embs.shape[1]
        top_spans = self.model.get_top_spans(document["sentence_map"], doc_embs,
           start_idx,
           given_spans=None)
        sentences = document["sentences"][start_idx: start_idx+num_words]
        segment_map = document["sentence_map"][start_idx:start_idx+num_words]
        sent_gen = util.get_sentence_iter(sentences, segment_map, top_spans,
                start_idx, None, self.model.cluster)
        for sent_idx, genre, cluster_list in sent_gen:
            for cluster in cluster_list:
                yield cluster


    def mention_detection(self, **kwargs):
        """Score spans by mention detection"""
        self.reset()
        self.model = self.model.eval()
        # keep track of mention scores for all spans
        span_scores = []
        eval_iterator = tqdm(enumerate(self.data))
        for doc_id, document in eval_iterator:
            start_idx = 0
            segment_iter = util.get_segment_iter(document)
            for seg_id, (segment, mask, seglen) in segment_iter:
                all_clusters = self.loop_clusters(segment, document, start_idx)
                for cluster in all_clusters:
                    span_score = {
                            "doc_key": document['doc_key'],
                            "span": (cluster.first.start, cluster.first.end),
                            "score": cluster.score.item()
                            }
                    span_scores.append(span_score)
                start_idx += seglen
        return span_scores


    def mention_entropy(self, **kwargs):
        """Score spans by entropy in mention detection."""
        self.reset()
        self.model = self.model.eval()
        # keep track of mention scores for all spans
        span_scores = []
        eval_iterator = tqdm(enumerate(self.data))
        for doc_id, document in eval_iterator:
            start_idx = 0
            segment_iter = util.get_segment_iter(document)
            for seg_id, (segment, mask, seglen) in segment_iter:
                all_clusters = self.loop_clusters(segment, document, start_idx)
                for cluster in all_clusters:
                    mention_scores = torch.cat([self.model.threshold, cluster.score.view(1)])
                    mention_dist = Categorical(logits = mention_scores)
                    mention_entropy = mention_dist.entropy()
                    span_score = {
                        "doc_key": document['doc_key'],
                        "span": (cluster.first.start, cluster.first.end),
                        "score": mention_entropy.item()
                        }
                    span_scores.append(span_score)
                start_idx += seglen
        return span_scores


    def conditional_entropy(self, joint=False, fixed=False, individual=False, **kwargs):
        """
        Score spans by entropy in clustering.
        - Default is conditional entropy.
        - If [fixed] is True, then cluster entropy (no mention detection) is
        computed.
        - If [joint] is True, then joint entropy is computed.
        - If [individual] is True, then cluster entropy from Li et al (2020)
          is computed.
        """
        if joint and fixed:
            raise ValueError('Cant have both joint and fixed settings')
        self.reset()
        self.model = self.model.eval()
        eval_iterator = tqdm(enumerate(self.data))

        entropy_scores = []
        for doc_id, document in eval_iterator:
            doc_entropy_scores = \
                self.incremental_clustering(document, joint, fixed, individual)
            entropy_scores += doc_entropy_scores
        return entropy_scores


    def incremental_clustering(self, document, joint, fixed, individual):
        doc_entropy_scores = []
        total_clusters = ClusterList()
        total_clusters.reset()
        pred_clusters = [] # only for Li's entropy
        start_idx = 0
        segment_iter = util.get_segment_iter(document)
        for seg_id, (segment, mask, seglen) in segment_iter:
            doc_embs = self.model.encoder(segment, f"{document['doc_key']}_{start_idx}")
            doc_embs = doc_embs.detach()
            num_words = doc_embs.shape[1]
            gold_spans = None
            top_spans = self.model.get_top_spans(document["sentence_map"], doc_embs,
                       start_idx,
                       given_spans=gold_spans)

            if len(top_spans) == 0:
                continue

            sentences = document["sentences"][start_idx: start_idx+num_words]
            segment_map = document["sentence_map"][start_idx:start_idx+num_words]
            genre_emb = self.model.genre_embedder(document["doc_key"][:2])
            sent_gen = util.get_sentence_iter(
                sentences, segment_map, top_spans, start_idx, genre_emb, self.model.cluster)

            if individual:
                self.li_resolve_local(
                    total_clusters, sent_gen, document["antecedent_map"],
                    document["doc_key"], doc_entropy_scores, pred_clusters
                )
            else:
                self.resolve_local(
                    total_clusters, sent_gen, document["antecedent_map"],
                    document["doc_key"], doc_entropy_scores, joint, fixed
                )
            start_idx += seglen

        return doc_entropy_scores


    def resolve_local(self, clusters, sent_gen, gold_map, doc_key, entropy_scores, joint, fixed):
        """
        Use ICoref algorithm to incrementally process spans in document and
        compute entropy scores for each span
        """
        for sent_idx, genre, cluster_list in sent_gen:
            # [cluster_list] refers to spans in the sentence
            for cluster in cluster_list:
                # [cluster] refers to individual spans

                if self.model.mention_classifier:
                    p_mention = torch.softmax(
                        torch.cat([self.model.threshold, cluster.score.view(1)]), dim=0
                    )
                else:
                    # everything is a mention
                    p_mention = torch.tensor([0.,1.])

                if joint:
                    mention_dist = Categorical(probs = p_mention)
                    mention_entropy = mention_dist.entropy().item()
                    ent_score = mention_entropy
                else:
                    ent_score = 0

                # Edge case for first mention
                if len(clusters) == 0:
                    if not self.model.singleton or cluster.score > self.model.threshold:
                        clusters.append(cluster)
                else:
                    # iterate over embeddings for each mention cluster
                    cluster_embs = [c.emb for c in clusters]
                    antecedent_starts = torch.tensor([(c.start/2.0 + c.end/2.0)
                        for c in clusters], device=self.model.device)
                    computed_offsets = (cluster.start + cluster.end)/2.0 - antecedent_starts
                    pairwise_scores = self.model.pair_scorer(cluster.emb,
                            cluster_embs, computed_offsets, genre)
                    if self.model.mention_classifier:
                        scores = pairwise_scores[0]
                    else:
                        scores = pairwise_scores[0] + cluster.score

                    cluster_scores = torch.cat([self.model.threshold, scores])
                    cluster_dist = Categorical(logits = cluster_scores)
                    # span_entropy = H(C|X=entity)
                    span_entropy = cluster_dist.entropy().item()
                    if not fixed:
                        # span_entropy = P(X=entity) H(C|X=entity)
                        span_entropy = p_mention[1] * span_entropy
                    ent_score += span_entropy

                    span_score = {
                        "doc_key": doc_key,
                        "span": (cluster.start, cluster.end),
                        "score": ent_score
                    }
                    entropy_scores.append(span_score)

                    p_cluster = torch.log_softmax(cluster_scores, dim=0)
                    p_score, best_cluster_idx = torch.min(-1 * p_cluster, dim=0)
                    self.model.update_clusterlist(
                        clusters, cluster, p_score, best_cluster_idx,
                        gold_cluster_id=0, train=False
                    )

        return None

    def joint_entropy(self, **kwargs):
        """Score spans by joint entropy"""
        scores = self.conditional_entropy(joint=True)
        return scores

    def cluster_entropy(self, **kwargs):
        """Score spans by cluster entropy"""
        scores = self.conditional_entropy(fixed=True)
        return scores

    def li_entropy(self, **kwargs):
        """Score spans by cluster entropy from Li et al (2020)"""
        scores = self.conditional_entropy(individual=True)
        return scores

    def li_resolve_local(self, clusters, sent_gen, gold_map, doc_key,
            entropy_scores, pred_clusters):
        """
        Variant of resolve_local where each span is placed in its own cluster.
        Entropy computation will look at confidence of span-span probabilities
        rather than span-cluster probabilities. This enables computation of
        cluster entropy from Li et al (2020).
        """
        next_cluster_idx = 1
        for sent_idx, genre, cluster_list in sent_gen:
            # [cluster_list] refers to spans in the sentence
            for cluster in cluster_list:
                # [cluster] refers to individual spans
                # Edge case for first mention
                if len(clusters) == 0:
                    if not self.model.singleton or cluster.score > self.model.threshold:
                        clusters.append(cluster)
                        pred_clusters.append(next_cluster_idx)
                        next_cluster_idx += 1
                else:
                    # iterate over embeddings for each mention cluster
                    cluster_embs = [c.emb for c in clusters]
                    antecedent_starts = torch.tensor([(c.start/2.0 + c.end/2.0)
                        for c in clusters], device=self.model.device)
                    computed_offsets = (cluster.start + cluster.end)/2.0 - antecedent_starts
                    pairwise_scores = self.model.pair_scorer(cluster.emb,
                            cluster_embs, computed_offsets, genre)
                    if self.model.mention_classifier:
                        scores = pairwise_scores[0]
                    else:
                        scores = pairwise_scores[0] + cluster.score

                    span_scores = torch.cat([self.model.threshold, scores])

                    # compute entropy
                    p_span = torch.softmax(span_scores, dim=0)
                    ent_score = 0
                    for i in range(1, next_cluster_idx):
                        prob_cluster = p_span[0].item() # prob. dummy cluster
                        for j in range(1, len(pred_clusters)):
                            if pred_clusters[j] == i:
                                prob_cluster += p_span[j+1].item()
                        prob_cluster = min(1, prob_cluster)
                        if prob_cluster > 0:
                            ent_cluster = -prob_cluster * np.log(prob_cluster)
                            ent_score += ent_cluster

                    span_score = {
                        "doc_key": doc_key,
                        "span": (cluster.start, cluster.end),
                        "score": ent_score
                    }
                    entropy_scores.append(span_score)



                    # always create new cluster for entity mentions
                    if not self.model.singleton or cluster.score > self.model.threshold:
                        clusters.append(cluster)

                        p_score, best_cluster_idx = torch.min(-1 * p_span, dim=0)
                        if best_cluster_idx == 0:
                            pred_clusters.append(next_cluster_idx)
                            next_cluster_idx += 1
                        else:
                            ante_cluster = pred_clusters[best_cluster_idx-1]
                            pred_clusters.append(ante_cluster)

        return None
