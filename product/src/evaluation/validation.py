import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from data_io.centralised_logger import CentralisedLogger
from evaluation.metrics import aggregate_fold_metrics, calculate_metrics
from gcn.masked_transformer import MaskedTransformer
from utils.paths import display_path


# Utilities
def log_worker(message: str, identifier: str = None):
    """
    Logs a timestamped message to the console with optional worker context.

    Args:
        message: The status message to print.
        identifier: Optional string (e.g., fold ID) to color-code the log source.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    id_tag = f"\033[92m[{identifier}]\033[0m " if identifier else ""
    print(f"\033[94m[{timestamp}]\033[0m {id_tag}{message}", flush=True)


def get_optimal_n_jobs(is_gcn: bool = False) -> int:
    """Return n_jobs based on hardware and model type.

    GPU training (GCN/SDAE) uses a single job to avoid VRAM OOM errors.
    CPU-only tasks use all available cores.
    """
    if torch.cuda.is_available() or (torch.backends.mps.is_available() and is_gcn):
        return 1
    return max(1, os.cpu_count())


def get_optuna_callback(identifier: str, patience: int = 25, log_interval: int = 5):
    """
    Generates a callback for Optuna studies to handle logging and early stopping.

    This closure tracks the best value seen across trials within a specific 
    worker process and stops the study if performance plateaus.

    Args:
        identifier: Context label for logging.
        patience: Number of trials to wait for improvement before stopping.
        log_interval: Frequency of progress updates (in trials).

    Returns:
        A callback function compatible with optuna.study.optimize.
    """
    best_seen = [float('-inf')]

    def callback(study, trial):
        current_best = study.best_value

        if current_best > best_seen[0]:
            best_seen[0] = current_best
            label = "Initial best" if trial.number == 0 else "New best"
            log_worker(f"{label}: {current_best:.4f} (Trial {study.best_trial.number})", identifier)

        if trial.number > 0 and trial.number % log_interval == 0:
            log_worker(
                f"Progress: Trial {trial.number} complete. Current Best AUC: {current_best:.4f}",
                identifier
            )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            return

        best_trial_number = study.best_trial.number
        trials_since_best = [t for t in completed_trials if t.number > best_trial_number]

        if len(trials_since_best) >= patience:
            log_worker(f"Stopping Study: No improvement for {patience} trials.", identifier)
            study.stop()

    return callback


# Hyperparameter sampling
def _sample_params(trial, param_grid: dict) -> dict:
    """
    Samples a hyperparameter configuration from a defined search space.

    Args:
        trial: The current Optuna trial object.
        param_grid: Dictionary defining 'float', 'int', or 'categorical' spaces.

    Returns:
        A dictionary of sampled parameter values.
    """
    sampled = {}
    for name, space in param_grid.items():
        t = space[0]
        if t == 'float':
            sampled[name] = trial.suggest_float(name, space[1], space[2], log=space[3] if len(space) > 3 else False)
        elif t == 'int':
            sampled[name] = trial.suggest_int(name, space[1], space[2], step=space[3] if len(space) > 3 else 1)
        elif t == 'categorical':
            sampled[name] = trial.suggest_categorical(name, space[1])
    return sampled


# GCN helpers
def _gcn_fit_params(N: int, tr_idx, val_idx, device: str, metadata: pd.DataFrame,
                    model: Pipeline = None, hpo_mode: bool = False) -> dict:
    """
    Constructs the fit_params dictionary required for transductive GCN training.

    This function identifies any MaskedTransformer or SDAE steps 
    within the pipeline and injects the 'train_mask' to ensure that 
    feature-level transformations are only 
    fitted on training nodes, avoiding look-ahead bias.

    Args:
        N: Total number of nodes in the graph.
        tr_idx: Indices of training nodes.
        val_idx: Indices of validation nodes.
        device: Torch device for mask tensors.
        metadata: Phenotypic data for graph construction.
        model: The pipeline instance to inspect for masked steps.
        hpo_mode: Flag passed to SDAE components to accelerate HPO trials.

    Returns:
        A dictionary of parameters passed to model.fit().
    """
    t_mask = torch.zeros(N, dtype=torch.bool, device=device)
    v_mask = torch.zeros(N, dtype=torch.bool, device=device)
    t_mask[tr_idx] = True
    v_mask[val_idx] = True

    params = {
        'gcn__train_mask': t_mask,
        'gcn__val_mask': v_mask,
        'population__metadata': metadata,
        'population__train_mask': t_mask,
    }

    if model is not None:
        # Auto-detect MaskedTransformer steps so they fit on training nodes only,
        # preventing test-label leakage
        for step_name, step in model.named_steps.items():
            if isinstance(step, MaskedTransformer):
                params[f'{step_name}__train_mask'] = t_mask
        if 'sdae' in model.named_steps:
            params['sdae__train_mask'] = t_mask
            params['sdae__hpo_mode'] = hpo_mode

    return params


# Inference and scoring
def _predict(model, X: np.ndarray, test_idx, is_gcn: bool):
    """
    Handles inference logic for both GCN (transductive) and standard (inductive) models.

    For GCNs, the model predicts for all nodes in the graph, and this helper 
    slices out only the requested test indices.

    Args:
        model: The fitted pipeline.
        X: Feature matrix.
        test_idx: Indices of the samples to predict.
        is_gcn: Boolean indicating if the model is a GCN.

    Returns:
        A tuple of (probabilities, class_predictions).
    """
    if is_gcn:
        return model.predict_proba(X)[test_idx, 1], model.predict(X)[test_idx]
    return model.predict_proba(X[test_idx])[:, 1], model.predict(X[test_idx])


def _score_val(model, X: np.ndarray, y: np.ndarray, val_idx, is_gcn: bool) -> float:
    """
    Calculates the internal optimisation score.

    Args:
        model: The fitted pipeline.
        X: Feature matrix.
        y: Ground truth labels.
        val_idx: Indices used for scoring.
        is_gcn: Boolean indicating if the model is a GCN.

    Returns:
        A scalar score representing model performance.
    """
    if is_gcn:
        return roc_auc_score(y[val_idx], model.predict_proba(X)[val_idx, 1])
    return model.score(X[val_idx], y[val_idx])


# Fold workers
def _run_single_eval_fold(identifier, train_idx, test_idx, model, X, y, metadata, is_gcn, device):
    """
    Worker function to execute one split of a standard CV or LOSO experiment.
    """
    log_worker(f"Starting {identifier}...", identifier)

    fold_model = clone(model)
    N = X.shape[0]

    if is_gcn:
        tr_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, stratify=y[train_idx], random_state=42
        )
        fit_params = _gcn_fit_params(N, tr_idx, val_idx, device, metadata, fold_model)
        fold_model.fit(X, y, **fit_params)
    else:
        fold_model.fit(X[train_idx], y[train_idx])

    probs, preds = _predict(fold_model, X, test_idx, is_gcn)
    scalars, artefacts = calculate_metrics(preds, probs, y[test_idx])

    log_worker(f"Complete.", identifier)

    return {
        "id": identifier,
        "scalars": scalars,
        "artefacts": artefacts,
    }


def _inner_objective(trial, model, X, y, train_idx_out, param_grid, is_gcn, device, metadata, identifier):
    """
    Objective function for the inner loop of a Nested Cross-Validation.

    This function performs a train/validation split within the outer training 
    set to evaluate a single hyperparameter configuration.
    """
    inner_model = clone(model)
    inner_model.set_params(**_sample_params(trial, param_grid))

    i_tr, i_val = train_test_split(
        np.arange(len(train_idx_out)),
        test_size=0.2,
        stratify=y[train_idx_out],
        random_state=42,  # Fixed for fair comparison across trials
    )

    global_tr = train_idx_out[i_tr]
    global_val = train_idx_out[i_val]

    if is_gcn:
        fit_params = _gcn_fit_params(X.shape[0], global_tr, global_val, device, metadata, model, hpo_mode=True)
        inner_model.fit(X, y, **fit_params)
    else:
        inner_model.fit(X[global_tr], y[global_tr])

    return _score_val(inner_model, X, y, global_val, is_gcn)


def _run_generic_fold(identifier, train_idx, test_idx, model, X, y, metadata,
                      param_grid, n_trials, is_gcn, device, storage=None):
    """
    The full Nested-CV worker: performs HPO, refits with best params, and evaluates.
    """
    log_worker(f"Starting {identifier}...", identifier)

    callback = get_optuna_callback(identifier)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        storage=storage,
        study_name=str(identifier),
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _inner_objective(t, model, X, y, train_idx, param_grid, is_gcn, device, metadata, identifier),
        n_trials=n_trials,
        n_jobs=1,  # Set to 1 because inner parallelism slows down outer parallelism
        callbacks=[callback],
        show_progress_bar=False,
    )

    final_model = clone(model)
    final_model.set_params(**study.best_params)

    if is_gcn:
        tr_f, val_f = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx], random_state=42)
        fit_params = _gcn_fit_params(X.shape[0], tr_f, val_f, device, metadata, model)
        final_model.fit(X, y, **fit_params)
    else:
        final_model.fit(X[train_idx], y[train_idx])

    probs, preds = _predict(final_model, X, test_idx, is_gcn)
    scalars, artefacts = calculate_metrics(preds, probs, y[test_idx])

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    log_worker(f"Complete.", identifier)

    return {
        "id": identifier,
        "scalars": scalars,
        "artefacts": artefacts,
        "best_params": study.best_params,
    }


# Results aggregation and logging
def _process_results(results, metadata, y, model, param_grid, logger, strategy_name, verbose):
    """
    Aggregates metrics across folds and logs the final summary to the filesystem.
    """
    metrics = []
    best_params_map = {}
    has_hpo = False

    for res in results:
        id_ = res["id"]
        metrics.append(res["scalars"])

        if "best_params" in res:
            best_params_map[id_] = res["best_params"]
            has_hpo = True

        if logger:
            logger.save_artefacts(res["artefacts"], prefix=str(id_), verbose=False)

    aggregated = aggregate_fold_metrics(metrics)

    if logger:
        meta_info = logger.prepare_meta(
            metadata, y,
            eval_strategy=strategy_name,
            n_folds=len(results),
        )

        log_kwargs = {
            "meta": meta_info,
            "params": model.get_params(),
            "metrics": aggregated,
        }

        if has_hpo:
            log_kwargs["search_space"] = param_grid
            log_kwargs["best_params"] = best_params_map

        print(f"Detailed artefacts saved to: {display_path(logger.current_run_dir)}")
        logger.log_run(**log_kwargs)


def stratified_k_fold(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    model: Pipeline,
    logger: Optional[CentralisedLogger] = None,
    n_folds: int = 5,
    verbose: bool = True,
    device: str = "cpu",
    n_jobs: int = -1,
):
    """
    Performs standard Stratified K-Fold Cross-Validation.
    """
    step_flow = " -> ".join(name for name, _ in model.steps)
    log_worker(f"Starting experiment on {step_flow}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    is_gcn = 'gcn' in model.named_steps

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_eval_fold)(
            f"fold_{idx+1}", tr, te, model, X, y, metadata, is_gcn, device
        ) for idx, (tr, te) in enumerate(skf.split(X, y))
    )

    log_worker(f"Experiment complete.")

    return _process_results(results, metadata, y, model, {}, logger, "SKF", verbose)


def leave_one_site_out(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    model: Pipeline,
    logger: Optional[CentralisedLogger] = None,
    verbose: bool = True,
    device: str = "cpu",
    n_jobs: int = -1,
):
    """
    Performs Leave-One-Site-Out (LOSO) Cross-Validation.
    
    This strategy evaluates model generalisability by training on N-1 sites 
    and testing on the held-out site.
    """
    step_flow = " -> ".join(name for name, _ in model.steps)
    log_worker(f"Starting experiment on {step_flow}")

    sites = sorted(metadata['SITE_ID'].unique())
    is_gcn = 'gcn' in model.named_steps

    splits = [
        (site, np.where(~(metadata['SITE_ID'] == site).values)[0], np.where((metadata['SITE_ID'] == site).values)[0])
        for site in sites
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_eval_fold)(
            sid, tr, te, model, X, y, metadata, is_gcn, device
        ) for sid, tr, te in splits
    )

    log_worker(f"Experiment complete.")

    return _process_results(results, metadata, y, model, {}, logger, "LOSO", verbose)


def nested_stratified_k_fold(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    model: Pipeline,
    param_grid: dict,
    logger: Optional[CentralisedLogger] = None,
    n_outer_folds: int = 5,
    n_trials: int = 20,
    verbose: bool = True,
    device: str = 'cpu',
    n_jobs: int = -1,
    storage: Optional[str] = None,
):
    """
    Performs Nested Stratified K-Fold Cross-Validation with hyperparameter optimisation.

    This strategy ensures that hyperparameter selection is performed within 
    each outer fold, providing an unbiased estimate of model performance.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    splits = list(outer_cv.split(X, y))

    if storage is not None:
        log_worker(f"Storage initialised: {display_path(storage)}")
        for idx in range(len(splits)):
            optuna.create_study(
                direction="maximize",
                study_name=f"fold_{idx+1}",
                storage=storage,
                load_if_exists=True,
            )
    else:
        log_worker(f"Storage not initialised. Running in-memory.", "WARNING")

    step_flow = " -> ".join(name for name, _ in model.steps)
    log_worker(f"Starting experiment on {step_flow}")

    is_gcn = 'gcn' in model.named_steps
    is_colab = 'google.colab' in sys.modules
    backend = "threading" if is_colab else "loky"

    results_gen = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(
        delayed(_run_generic_fold)(
            f"fold_{idx+1}", tr, te, model, X, y, metadata, param_grid, n_trials, is_gcn, device, storage
        ) for idx, (tr, te) in enumerate(splits)
    )

    results = list(results_gen)
    log_worker(f"Experiment complete.")

    return _process_results(results, metadata, y, model, param_grid, logger, "Nested SKF", verbose)


def nested_leave_one_site_out(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    model: Pipeline,
    param_grid: dict,
    logger: Optional[CentralisedLogger] = None,
    n_trials: int = 20,
    verbose: bool = True,
    device: str = "cpu",
    n_jobs: int = -1,
    storage: Optional[str] = None,
):
    """
    Performs Nested Leave-One-Site-Out CV with hyperparameter optimisation.
    """
    sites = sorted(metadata['SITE_ID'].unique())
    splits = [
        (site, np.where(~(metadata['SITE_ID'] == site).values)[0], np.where((metadata['SITE_ID'] == site).values)[0])
        for site in sites
    ]

    if storage is not None:
        log_worker(f"Storage initialised: {display_path(storage)}")
        for site, _, _ in splits:
            optuna.create_study(
                direction="maximize",
                study_name=f"{site}",
                storage=storage,
                load_if_exists=True,
            )
    else:
        log_worker(f"Storage not initialised. Running in-memory.", "WARNING")

    step_flow = " -> ".join(name for name, _ in model.steps)
    log_worker(f"Starting experiment on {step_flow}")

    is_gcn = 'gcn' in model.named_steps
    is_colab = 'google.colab' in sys.modules
    backend = "threading" if is_colab else "loky"

    results_gen = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(
        delayed(_run_generic_fold)(
            sid, tr, te, model, X, y, metadata, param_grid, n_trials, is_gcn, device, storage
        ) for sid, tr, te in splits
    )

    results = list(results_gen)
    log_worker(f"Experiment complete.")

    return _process_results(results, metadata, y, model, param_grid, logger, "Nested LOSO", verbose)


# Final model training
def _final_objective(trial, model, X, y, param_grid, is_gcn, device, metadata, identifier):
    """
    Objective function for the final HPO study performed on the entire dataset.
    """
    inner_model = clone(model)
    inner_model.set_params(**_sample_params(trial, param_grid))
    N = X.shape[0]

    train_idx, val_idx = train_test_split(
        np.arange(N),
        test_size=0.2,
        stratify=y,
        random_state=42,  # Fixed for trial consistency
    )

    if is_gcn:
        fit_params = _gcn_fit_params(N, train_idx, val_idx, device, metadata, model, hpo_mode=True)
        inner_model.fit(X, y, **fit_params)
    else:
        inner_model.fit(X[train_idx], y[train_idx])

    return _score_val(inner_model, X, y, val_idx, is_gcn)


def final_study(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    model: Pipeline,
    param_grid: dict,
    logger: Optional[CentralisedLogger] = None,
    n_trials: int = 50,
    n_jobs: int = -1,
    device: str = "cpu",
):
    """
    Conducts a final hyperparameter study using the full available dataset.
    
    This is intended to select the absolute best parameters before producing 
    a production-ready model for external validation.
    """
    is_gcn = 'gcn' in model.named_steps
    N = X.shape[0]
    identifier = "Final_Study"

    callback = get_optuna_callback(identifier)
    log_worker(f"Starting {identifier}...", identifier)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda t: _final_objective(t, model, X, y, param_grid, is_gcn, device, metadata, identifier),
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback],
        show_progress_bar=False,
    )
    log_worker(f"Complete.", identifier)

    final_model = clone(model)
    final_model.set_params(**study.best_params)

    if is_gcn:
        tr_idx, val_idx = train_test_split(np.arange(N), test_size=0.1, stratify=y, random_state=42)
        final_fit_params = _gcn_fit_params(N, tr_idx, val_idx, device, metadata, model)
        final_model.fit(X, y, **final_fit_params)
    else:
        final_model.fit(X, y)

    print("Experiment complete.")

    if logger:
        meta_info = logger.prepare_meta(metadata, y, eval_strategy="Final Hyperparameter Selection", n_folds=1)
        logger.log_run(
            meta=meta_info,
            params=final_model.get_params(),
            metrics={},  # No test metrics: full dataset used for training
            search_space=param_grid,
            best_params=study.best_params,
        )