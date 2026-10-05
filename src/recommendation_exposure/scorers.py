from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context
from pathlib import Path

from .types import PreparedDataset, ScoredCandidate, ScoredUserEvalCase, UserEvalCase
from .utils import minmax_normalize, resolve_path, stable_int, tokenize


class BaseScorer:
    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        raise NotImplementedError


class PopularityScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset) -> None:
        self.max_popularity = max(dataset.train_popularity.values(), default=1)
        self.popularity = dataset.train_popularity

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        return [self.popularity.get(candidate.item_id, 0) / self.max_popularity for candidate in case.candidates]


class RandomScorer(BaseScorer):
    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        values: list[float] = []
        for candidate in case.candidates:
            raw = stable_int((case.split, case.user_id, candidate.item_id)) & 0xFFFF
            values.append(raw / 0xFFFF)
        return values


def _torch_device(scorer_cfg: dict) -> str:
    import torch

    requested = str(scorer_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _train_pairs(dataset: PreparedDataset, max_pairs: int, seed: int) -> tuple[list[str], list[str]]:
    pairs: list[tuple[str, str]] = []
    for user_id, history in dataset.train_histories.items():
        pairs.extend((user_id, item_id) for item_id in history if item_id in dataset.items)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return [user_id for user_id, _ in pairs], [item_id for _, item_id in pairs]


class BprMfScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset, scorer_cfg: dict) -> None:
        import torch
        import torch.nn.functional as F

        seed = int(scorer_cfg.get("seed", 17))
        torch.manual_seed(seed)
        random.seed(seed)
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(sorted(dataset.train_histories))}
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(sorted(dataset.items))}
        self.device = _torch_device(scorer_cfg)
        dim = int(scorer_cfg.get("embedding_dim", 64))
        epochs = int(scorer_cfg.get("epochs", 4))
        batch_size = int(scorer_cfg.get("batch_size", 4096))
        max_pairs = int(scorer_cfg.get("max_train_pairs", 300_000))
        lr = float(scorer_cfg.get("learning_rate", 0.01))

        user_ids, item_ids = _train_pairs(dataset, max_pairs, seed)
        user_idx = torch.tensor([self.user_to_index[user_id] for user_id in user_ids], dtype=torch.long, device=self.device)
        pos_idx = torch.tensor([self.item_to_index[item_id] for item_id in item_ids], dtype=torch.long, device=self.device)
        user_emb = torch.nn.Embedding(len(self.user_to_index), dim).to(self.device)
        item_emb = torch.nn.Embedding(len(self.item_to_index), dim).to(self.device)
        torch.nn.init.normal_(user_emb.weight, std=0.05)
        torch.nn.init.normal_(item_emb.weight, std=0.05)
        optimizer = torch.optim.Adam(list(user_emb.parameters()) + list(item_emb.parameters()), lr=lr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        print(
            f"[rec:score:bpr_mf] users={len(self.user_to_index)} items={len(self.item_to_index)} "
            f"pairs={len(user_idx)} device={self.device}",
            flush=True,
        )
        for epoch in range(epochs):
            order = torch.randperm(len(user_idx), generator=generator, device=self.device)
            total_loss = 0.0
            steps = 0
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                users = user_idx[batch]
                positives = pos_idx[batch]
                negatives = torch.randint(
                    len(self.item_to_index),
                    (len(batch),),
                    generator=generator,
                    device=self.device,
                )
                pos_scores = (user_emb(users) * item_emb(positives)).sum(dim=1)
                neg_scores = (user_emb(users) * item_emb(negatives)).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
            print(f"[rec:score:bpr_mf] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, steps):.4f}", flush=True)
        self.user_weights = user_emb.weight.detach().cpu()
        self.item_weights = item_emb.weight.detach().cpu()

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        user_idx = self.user_to_index.get(case.user_id)
        if user_idx is None:
            return [0.0 for _ in case.candidates]
        user_vec = self.user_weights[user_idx]
        scores = []
        for candidate in case.candidates:
            item_idx = self.item_to_index.get(candidate.item_id)
            scores.append(float(torch_dot(user_vec, self.item_weights[item_idx])) if item_idx is not None else 0.0)
        return scores


def torch_dot(left, right) -> float:
    return float((left * right).sum().item())


class NeuMfScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset, scorer_cfg: dict) -> None:
        import torch
        import torch.nn.functional as F

        class NeuMfModel(torch.nn.Module):
            def __init__(self, user_count: int, item_count: int, dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.user_gmf = torch.nn.Embedding(user_count, dim)
                self.item_gmf = torch.nn.Embedding(item_count, dim)
                self.user_mlp = torch.nn.Embedding(user_count, dim)
                self.item_mlp = torch.nn.Embedding(item_count, dim)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(dim * 2, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim // 2),
                    torch.nn.ReLU(),
                )
                self.out = torch.nn.Linear(dim + hidden_dim // 2, 1)

            def forward(self, users, items):
                gmf = self.user_gmf(users) * self.item_gmf(items)
                mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=1))
                return self.out(torch.cat([gmf, mlp], dim=1)).squeeze(1)

        seed = int(scorer_cfg.get("seed", 17))
        torch.manual_seed(seed)
        random.seed(seed)
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(sorted(dataset.train_histories))}
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(sorted(dataset.items))}
        self.device = _torch_device(scorer_cfg)
        dim = int(scorer_cfg.get("embedding_dim", 48))
        hidden_dim = int(scorer_cfg.get("hidden_dim", 96))
        epochs = int(scorer_cfg.get("epochs", 4))
        batch_size = int(scorer_cfg.get("batch_size", 4096))
        max_pairs = int(scorer_cfg.get("max_train_pairs", 250_000))
        neg_per_pos = int(scorer_cfg.get("negatives_per_positive", 1))
        lr = float(scorer_cfg.get("learning_rate", 0.001))
        user_ids, item_ids = _train_pairs(dataset, max_pairs, seed)
        user_idx = torch.tensor([self.user_to_index[user_id] for user_id in user_ids], dtype=torch.long, device=self.device)
        pos_idx = torch.tensor([self.item_to_index[item_id] for item_id in item_ids], dtype=torch.long, device=self.device)
        model = NeuMfModel(len(self.user_to_index), len(self.item_to_index), dim, hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        print(
            f"[rec:score:neumf] users={len(self.user_to_index)} items={len(self.item_to_index)} "
            f"pairs={len(user_idx)} device={self.device}",
            flush=True,
        )
        for epoch in range(epochs):
            order = torch.randperm(len(user_idx), generator=generator, device=self.device)
            total_loss = 0.0
            steps = 0
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                users = user_idx[batch]
                positives = pos_idx[batch]
                neg_users = users.repeat_interleave(neg_per_pos)
                negatives = torch.randint(
                    len(self.item_to_index),
                    (len(neg_users),),
                    generator=generator,
                    device=self.device,
                )
                all_users = torch.cat([users, neg_users])
                all_items = torch.cat([positives, negatives])
                labels = torch.cat(
                    [
                        torch.ones(len(users), device=self.device),
                        torch.zeros(len(neg_users), device=self.device),
                    ]
                )
                logits = model(all_users, all_items)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
            print(f"[rec:score:neumf] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, steps):.4f}", flush=True)
        self.model = model.eval().cpu()

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        import torch

        user_idx = self.user_to_index.get(case.user_id)
        if user_idx is None:
            return [0.0 for _ in case.candidates]
        item_indices = [self.item_to_index.get(candidate.item_id) for candidate in case.candidates]
        with torch.no_grad():
            users = torch.tensor([user_idx] * len(item_indices), dtype=torch.long)
            items = torch.tensor([idx if idx is not None else 0 for idx in item_indices], dtype=torch.long)
            logits = self.model(users, items).tolist()
        return [float(score) if idx is not None else 0.0 for score, idx in zip(logits, item_indices)]


class SasRecLiteScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset, scorer_cfg: dict) -> None:
        import torch
        import torch.nn.functional as F

        class SasRecLite(torch.nn.Module):
            def __init__(self, item_count: int, dim: int, max_len: int, heads: int, layers: int, dropout: float) -> None:
                super().__init__()
                self.item_emb = torch.nn.Embedding(item_count + 1, dim, padding_idx=0)
                self.pos_emb = torch.nn.Embedding(max_len, dim)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.max_len = max_len

            def encode(self, sequences):
                positions = torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
                hidden = self.item_emb(sequences) + self.pos_emb(positions)
                padding_mask = sequences.eq(0)
                encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
                lengths = sequences.ne(0).sum(dim=1).clamp(min=1) - 1
                return encoded[torch.arange(sequences.shape[0], device=sequences.device), lengths]

            def score(self, user_reps, items):
                return (user_reps * self.item_emb(items)).sum(dim=1)

        seed = int(scorer_cfg.get("seed", 17))
        torch.manual_seed(seed)
        random.seed(seed)
        self.item_to_index = {item_id: idx + 1 for idx, item_id in enumerate(sorted(dataset.items))}
        self.device = _torch_device(scorer_cfg)
        dim = int(scorer_cfg.get("embedding_dim", 64))
        max_len = int(scorer_cfg.get("max_sequence_length", 50))
        heads = int(scorer_cfg.get("heads", 2))
        layers = int(scorer_cfg.get("layers", 2))
        dropout = float(scorer_cfg.get("dropout", 0.1))
        epochs = int(scorer_cfg.get("epochs", 4))
        batch_size = int(scorer_cfg.get("batch_size", 512))
        max_sequences = int(scorer_cfg.get("max_train_sequences", 120_000))
        lr = float(scorer_cfg.get("learning_rate", 0.001))

        examples: list[tuple[list[int], int]] = []
        for history in dataset.train_histories.values():
            indexed = [self.item_to_index[item_id] for item_id in history if item_id in self.item_to_index]
            for end in range(2, len(indexed) + 1):
                examples.append((indexed[max(0, end - max_len - 1) : end - 1], indexed[end - 1]))
        rng = random.Random(seed)
        rng.shuffle(examples)
        if max_sequences > 0:
            examples = examples[:max_sequences]
        model = SasRecLite(len(self.item_to_index), dim, max_len, heads, layers, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        print(
            f"[rec:score:sasrec_lite] items={len(self.item_to_index)} sequences={len(examples)} device={self.device}",
            flush=True,
        )
        for epoch in range(epochs):
            rng.shuffle(examples)
            total_loss = 0.0
            steps = 0
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                seq = torch.zeros((len(batch), max_len), dtype=torch.long, device=self.device)
                pos = torch.empty(len(batch), dtype=torch.long, device=self.device)
                for row, (sequence, target) in enumerate(batch):
                    clipped = sequence[-max_len:]
                    seq[row, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=self.device)
                    pos[row] = target
                neg = torch.randint(1, len(self.item_to_index) + 1, (len(batch),), generator=generator, device=self.device)
                reps = model.encode(seq)
                pos_scores = model.score(reps, pos)
                neg_scores = model.score(reps, neg)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
            print(f"[rec:score:sasrec_lite] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, steps):.4f}", flush=True)
        model.eval()
        self.item_weights = model.item_emb.weight.detach().cpu()
        self.user_reps: dict[str, object] = {}
        with torch.no_grad():
            for user_id, history in dataset.train_histories.items():
                indexed = [self.item_to_index[item_id] for item_id in history[-max_len:] if item_id in self.item_to_index]
                if not indexed:
                    continue
                seq = torch.zeros((1, max_len), dtype=torch.long, device=self.device)
                seq[0, : len(indexed)] = torch.tensor(indexed, dtype=torch.long, device=self.device)
                self.user_reps[user_id] = model.encode(seq).squeeze(0).detach().cpu()

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        rep = self.user_reps.get(case.user_id)
        if rep is None:
            return [0.0 for _ in case.candidates]
        scores = []
        for candidate in case.candidates:
            item_idx = self.item_to_index.get(candidate.item_id)
            scores.append(float(torch_dot(rep, self.item_weights[item_idx])) if item_idx is not None else 0.0)
        return scores


class TokenOverlapScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset) -> None:
        self.item_tokens = {item_id: tokenize(record.text) for item_id, record in dataset.items.items()}

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        return _score_token_overlap_case(case, self.item_tokens)


def _score_token_overlap_case(
    case: UserEvalCase,
    item_tokens: dict[str, set[str]],
) -> list[float]:
    history_counter: Counter[str] = Counter()
    for item_id in case.history_item_ids:
        history_counter.update(item_tokens.get(item_id, set()))
    history_norm = math.sqrt(sum(value * value for value in history_counter.values())) or 1.0

    scores: list[float] = []
    for candidate in case.candidates:
        candidate_tokens = item_tokens.get(candidate.item_id, set())
        if not candidate_tokens:
            scores.append(0.0)
            continue
        overlap = sum(history_counter.get(token, 0) for token in candidate_tokens)
        candidate_norm = math.sqrt(len(candidate_tokens))
        scores.append(overlap / (history_norm * candidate_norm))
    return scores


_TOKEN_OVERLAP_ITEM_TOKENS: dict[str, set[str]] | None = None


def _init_token_overlap_worker(item_tokens: dict[str, set[str]]) -> None:
    global _TOKEN_OVERLAP_ITEM_TOKENS
    _TOKEN_OVERLAP_ITEM_TOKENS = item_tokens


def _score_token_overlap_case_worker(case: UserEvalCase) -> list[float]:
    if _TOKEN_OVERLAP_ITEM_TOKENS is None:
        raise RuntimeError("Token-overlap worker state was not initialized.")
    return _score_token_overlap_case(case, _TOKEN_OVERLAP_ITEM_TOKENS)


def _resolved_score_workers(total_cases: int) -> int:
    raw = os.environ.get("REC_SCORE_WORKERS")
    if raw is not None:
        return max(1, int(raw))
    if total_cases < 2_000:
        return 1
    return max(1, min(8, cpu_count()))


class PrecomputedJsonlScorer(BaseScorer):
    def __init__(self, dataset: PreparedDataset, scores_path: Path) -> None:
        self.scores: dict[tuple[str, str, str], float] = {}
        with scores_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                key = (payload["split"], payload["user_id"], payload["item_id"])
                self.scores[key] = float(payload["score"])

    def score_case(self, case: UserEvalCase, dataset: PreparedDataset) -> list[float]:
        values: list[float] = []
        for candidate in case.candidates:
            key = (case.split, case.user_id, candidate.item_id)
            if key not in self.scores:
                raise KeyError(f"Missing precomputed score for {key}.")
            values.append(self.scores[key])
        return values


def build_scorer(config: dict, dataset: PreparedDataset) -> BaseScorer:
    scorer_cfg = config["scorer"]
    kind = scorer_cfg["kind"]
    if kind == "popularity":
        return PopularityScorer(dataset)
    if kind == "random":
        return RandomScorer()
    if kind == "bpr_mf":
        return BprMfScorer(dataset, scorer_cfg)
    if kind == "neumf":
        return NeuMfScorer(dataset, scorer_cfg)
    if kind == "sasrec_lite":
        return SasRecLiteScorer(dataset, scorer_cfg)
    if kind == "token_overlap":
        return TokenOverlapScorer(dataset)
    if kind == "precomputed_jsonl":
        path = resolve_path(dataset.config_path, scorer_cfg["scores_path"])
        return PrecomputedJsonlScorer(dataset, path)
    raise ValueError(f"Unsupported scorer kind: {kind}")


def score_cases(
    cases: tuple[UserEvalCase, ...],
    dataset: PreparedDataset,
    scorer: BaseScorer,
    normalize: bool = True,
) -> tuple[ScoredUserEvalCase, ...]:
    raw_cases: list[ScoredUserEvalCase] = []
    flat_scores: list[float] = []
    score_lengths: list[int] = []
    total_cases = len(cases)
    workers = _resolved_score_workers(total_cases)
    def consume(score_iter) -> None:
        for index, (case, scores) in enumerate(zip(cases, score_iter), start=1):
            if len(scores) != len(case.candidates):
                raise ValueError(f"Scorer returned {len(scores)} scores for {len(case.candidates)} candidates.")
            flat_scores.extend(scores)
            score_lengths.append(len(scores))
            raw_cases.append(
                ScoredUserEvalCase(
                    split=case.split,
                    user_id=case.user_id,
                    group=case.group,
                    history_item_ids=case.history_item_ids,
                    candidates=tuple(
                        ScoredCandidate(
                            item_id=candidate.item_id,
                            label=candidate.label,
                            score=score,
                            item_group=(
                                dataset.items[candidate.item_id].category
                                if candidate.item_id in dataset.items
                                else None
                            ),
                        )
                        for candidate, score in zip(case.candidates, scores)
                    ),
                )
            )
            if index % 2_000 == 0 or index == total_cases:
                print(f"[rec:score] cases={index}/{total_cases}", flush=True)

    if isinstance(scorer, TokenOverlapScorer) and workers > 1:
        print(f"[rec:score] token_overlap_workers={workers}", flush=True)
        mp_context = get_context("fork") if os.name != "nt" else get_context()
        chunksize = max(1, total_cases // max(1, workers * 16))
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=_init_token_overlap_worker,
                initargs=(scorer.item_tokens,),
            ) as executor:
                consume(executor.map(_score_token_overlap_case_worker, cases, chunksize=chunksize))
        except (OSError, PermissionError) as exc:
            print(f"[rec:score] process_pool_unavailable fallback=serial reason={exc}", flush=True)
            consume(scorer.score_case(case, dataset) for case in cases)
    else:
        consume(scorer.score_case(case, dataset) for case in cases)

    if not normalize:
        return tuple(raw_cases)

    normalized_scores = minmax_normalize(flat_scores)
    print(f"[rec:score] normalizing_scores count={len(normalized_scores)}", flush=True)
    normalized_cases: list[ScoredUserEvalCase] = []
    index = 0
    for case, length in zip(raw_cases, score_lengths):
        slice_scores = normalized_scores[index : index + length]
        index += length
        normalized_cases.append(
            ScoredUserEvalCase(
                split=case.split,
                user_id=case.user_id,
                group=case.group,
                history_item_ids=case.history_item_ids,
                candidates=tuple(
                    ScoredCandidate(
                        item_id=candidate.item_id,
                        label=candidate.label,
                        score=score,
                        item_group=candidate.item_group,
                    )
                    for candidate, score in zip(case.candidates, slice_scores)
                ),
            )
        )
    return tuple(normalized_cases)
