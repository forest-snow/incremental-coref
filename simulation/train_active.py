import torch
import pathlib
import numpy as np
from incremental import Incremental
from inference import Predictor
import util
from cluster import ClusterList
from tqdm import tqdm
import json
import logging
import random
from eval_all import ALL

from trainer import Trainer

class TrainerActive(Trainer):
    """
    Extends Trainer class for active learning.
    This class mainly just modifies I/O files and saves mention detection
    accuracy.
    """
    def __init__(self, config, model, data, model_dir):
        self.model_dir = model_dir
        Trainer.__init__(self, config, model, data)

    def train(self, evaluator):
        best_f1 = -1.0
        best_epoch = -1
        mentions = -1
        living = {name for name, param in self.model.state_dict().items() if not self.is_unused_layer(name)}
        for epoch in range(self.num_epochs):
            if epoch - best_epoch >= self.patience:
                logging.info(f"Ran out of patience, stopping on epoch {epoch} " +
                                         f"(saved {best_epoch} with {best_f1:.3f})")
                scores = {'f1': best_f1, 'mentions': mentions}
                return scores
            random.shuffle(self.data)
            self.data = self.data
            loss = self.train_epoch()
            logging.info(f"average training loss: {loss:.3f}")
            with torch.no_grad():
                f1 = evaluator.evaluate()
            if f1 > best_f1:
                save_dict = {
                    'optimizer': self.model_optimizer,
                    'model': self.get_filtered_state_dict(self.model)}
                if self.encoder_optimizer is not None:
                    save_dict["encoder_optimizer"] = self.encoder_optimizer
                torch.save(save_dict, self.model_dir / 'checkpoint.bin')
                logging.info(f"Saved model with {f1:.3f} dev F1 on epoch {epoch}")
                best_f1 = f1
                best_epoch = epoch
                mentions = evaluator.evaluators[ALL].evaluators[3].get_f1()
        scores = {'f1': best_f1, 'mentions': mentions}
        return scores



class IncrementalActive(Incremental):
    """
    Extends Incremental class for active learning.
    Major difference is that model only backprops if the span was queried and
    labeled.
    """

    def resolve_local(self, clusters, sent_gen, gold_map, train=False,
            metrics=True):
        total_loss = None
        cpu_loss = 0
        num_spans = 0
        for sent_idx, genre, cluster_list in sent_gen:
            # print (f"{sent_idx}: {len(cluster_list)} spans (so far {num_spans})")
            for cluster in cluster_list:
                curr_span = (cluster.first.start, cluster.first.end)
                gold_span = gold_map.get(curr_span, None)
                gold_cluster_ids = clusters.get_cluster_ids(gold_span, original=curr_span) # in range(0, num_clusters + 1)
                gold_cluster_id = max(gold_cluster_ids) # pick most recent gold cluster
                # Compute singleton loss here (it runs over clusters now)
                if self.mention_classifier:
                    p_mention = torch.log_softmax(torch.cat([self.threshold, cluster.score.view(1)]), dim=0)
                    gold_span_label = 0 if gold_span is None else 1
                    span_loss = -1 * p_mention[gold_span_label]

                    # Shortcircuit: If we know/predict it has no antecedent at training/inference, stop now.
                    do_cluster_scoring = ((train and gold_span_label == 1) or
                        (not train and cluster.score > self.threshold))
                else:
                    span_loss = None
                    do_cluster_scoring = True
                    # Might want to skip if there are too many negative spans
                    if gold_span is not None:
                        self.loss_count += 1
                    elif np.random.uniform() < self.negative_sample_rate:
                        self.sampled_loss_count += 1
                    else:
                        do_cluster_scoring = False # Just skip everything

                # Compute cluster/linking loss. All spans making it to this step will be written to predictions file.
                if do_cluster_scoring:
                    num_spans += cluster.size
                    # Edge case for first mention
                    if len(clusters) == 0:
                        if not self.singleton or cluster.score > self.threshold:
                            clusters.append(cluster)
                        cluster_loss = None
                    else:
                        cluster_embs = [c.emb for c in clusters]
                        antecedent_starts = torch.tensor([(c.start/2.0 + c.end/2.0) for c in clusters], device=self.device)
                        computed_offsets = (cluster.start + cluster.end)/2.0 - antecedent_starts  # type: int - tensor -> tensor
                        pairwise_scores = self.pair_scorer(cluster.emb, cluster_embs, computed_offsets, genre)  # [1,1] score

                        # With a mention classifier, we don't add the cluster.score (which is already used).
                        # Without it, we can interpret cluster.score as an additional feature of the pairwise scorer.
                        if self.mention_classifier:
                            scores = pairwise_scores[0]
                        else:
                            scores = pairwise_scores[0] + cluster.score

                        cluster_scores = torch.cat([self.threshold, scores])
                        p_cluster = torch.log_softmax(cluster_scores, dim=0) # [num_clusters + 1]
                        p_score, best_cluster_idx = torch.min(-1 * p_cluster, dim=0)
                        cluster_loss = -1 * torch.logsumexp(p_cluster[gold_cluster_ids], dim=0) # log gold probabilities

                        self.update_clusterlist(clusters, cluster, p_score, best_cluster_idx, gold_cluster_id, train)
                        # Stats on updates
                        if (not train) and metrics:
                            self.compute_attach_stats(best_cluster_idx, gold_cluster_id)
                else:
                    cluster_loss = None

                # Update total_loss to a single final loss
                total_loss = util.safe_add(total_loss,
                                           util.safe_add(span_loss, cluster_loss))

                # Check if we need to run backward() for memory reasons, only a concern in train
                if train and curr_span in gold_map and total_loss is not None:
                    total_loss.backward(retain_graph=True)
                cpu_loss = float(util.safe_add(cpu_loss, total_loss)) # float() needed to detach total_loss
                total_loss = None
        return cpu_loss




    def forward(self, segment, model_data, clusters, start_idx, mask, train=False, metrics=True, consolidate=True):
        if consolidate:
            genre_emb = self.genre_embedder(model_data["doc_key"][:2])
            # Consolidate given clusters
            new_clusters = ClusterList()
            cons_loss = self.resolve_local(
                new_clusters, [(None, genre_emb, clusters)],
                model_data["antecedent_map"],
                train=train, metrics=False
            )
            clusters.update(new_clusters)
        else:
            cons_loss = 0.0
        doc_embs = self.encoder(segment, f"{model_data['doc_key']}_{start_idx}")
        if mask is not None:
            doc_embs = torch.index_select(doc_embs, 1, torch.tensor(mask).to(self.device))
        if not self.finetune or not train:  # not detaching is memory expenisve
            doc_embs = doc_embs.detach()

        if self.use_gold_spans:
            if "gold_spans" in model_data:
                gold_spans = model_data["gold_spans"]
            else:
                # Assume gold spans are just the cluster spans
                # also dedup
                gold_spans = list(set([tuple(x) for x in util.flatten(model_data["clusters"])]))
        else:
            gold_spans = None
        top_spans = self.get_top_spans(model_data["sentence_map"], doc_embs,
                                                                     start_idx, given_spans=gold_spans)
        if len(top_spans) == 0:
            # Nothing to update
            return 0.0
        num_words = doc_embs.shape[1]
        if mask is None:
            sentences = segment
        else:
            sentences = model_data["sentences"][start_idx: start_idx+num_words]
        segment_map = model_data["sentence_map"][start_idx:start_idx+num_words]
        genre_emb = self.genre_embedder(model_data["doc_key"][:2])
        sent_gen = util.get_sentence_iter(sentences, segment_map, top_spans, start_idx, genre_emb, self.cluster)
        spans_loss = self.resolve_local(
            clusters, sent_gen, model_data["antecedent_map"],
            train=train, metrics=metrics
        )
        return cons_loss + spans_loss



def finetune_on_queries(config, train_data, model_dir, src_path):
    """
    Finetune source model on spans sampled from active learning
    - [config]: active learning hyperparameters
    - [train_data]: data to train model
    - [model_dir]: directory to save checkpoints and results
    - [src_path]: path to load source model
    """
    util.set_seed(config)
    model = IncrementalActive(config)
    trainer = TrainerActive(config, model, train_data, model_dir)
    data_dev = util.load_data(config["dev_path"])
    evaluator = Predictor(model, data_dev, config["singleton_eval"])
    logging.info(f'Loading model and optimizer from {src_path}')
    util.load_params(model, src_path , "model")
    util.load_params(trainer.model_optimizer, src_path, "optimizer")
    logging.info(f"Updating threshold to {config['threshold']}")
    model.set_threshold(config["threshold"])

    # Train
    scores_dev = trainer.train(evaluator)
    # Write preds
    preds_file = model_dir / 'preds_dev.json'
    evaluator.write_preds(preds_file)
    logging.info(f"Wrote preds to {preds_file}")
    return scores_dev, model


