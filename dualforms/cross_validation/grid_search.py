import numpy as np
from sklearn.model_selection import KFold
from itertools import product
from joblib import Parallel, delayed


def _fit_and_score(build_model, params, X, y, aux, train_idx, val_idx, train_kwargs, grid_id):
    """Fit one model for one (params, fold) pair and return its validation score."""
    model = build_model(**params)
    data_train_args = (X[train_idx],)
    if y is not None:
        data_train_args += (y[train_idx],)
        if aux is not None:
            data_train_args += (aux[train_idx],)
    data_val_args = (X[val_idx],)
    if y is not None:
        data_val_args += (y[val_idx],)
        if aux is not None:
            data_val_args += (aux[val_idx],)
    try:
        model.train(*data_train_args, **train_kwargs)
    except Exception as _:
        return grid_id, None

    if not model.solver_success:
        return grid_id, None

    score = model.cv_score(*data_val_args)
    return grid_id, score


class GridSearch:
    """
    Grid search of hyperparameters for a specific dual formulation.

    Instantiate a dual formulation for each hyperparameter combination then
    train and validate each combination on every fold.
    Select the hyperparameter combination with the lowest cross-validation score.

    Parameters
    ----------
    build_model : callable
        Function to construct a dual model from a set of hyperparameters.
    param_grid: dict
        Dictionary with keys corresponding to each hyperparameter accepted by build_model and values the list of possible values for that hyperparameter.
    n_folds: int
        Number of folds into which to divide the training data. Common values are 5, 7, 10.
    n_jobs : int
        Number of parallel computations. Integer greater than or equal to 1.
    """

    def __init__(self, build_model, param_grid, n_folds, n_jobs=1):
        self.build_model = build_model
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = None
        self.results = []

    def train(self, X, y=None, aux=None, **train_kwargs):
        """
        GridSearch cross-validation training for both supervised and unsupervised learning problems.
        
        Divide the training data into folds, create the list of hyperparameter combinations based on param_grid and
        launch the computation of all (folds x hyperparameter) trainings and validations.
        Then, retrieve the set of hyperparameters associated with the best score and 
        train the final model on the full training data with the identified best set of hyperparameter.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature vectors in training data.
        y : np.ndarray of shape (n_samples, 1) or None
            Target values in training data. None for unsupervised learning problems.
        aux : object
            Additional information needed for training, such as mean or median prediction on training data. None if not applicable.
        """
        # Assemble folds
        kf = KFold(n_splits=self.n_folds, shuffle=False)
        folds = list(kf.split(X))

        # Create list of hyperparameter combinations
        keys = list(self.param_grid.keys())
        param_combos = [dict(zip(keys, values))
                        for values in product(*self.param_grid.values())]
        
        # Launch training and validation for all folds and combinations
        jobs = [
            delayed(_fit_and_score)(
                self.build_model, params, X, y, aux, train_idx, val_idx, train_kwargs, grid_id,
            )
            for grid_id, params in enumerate(param_combos)
            for train_idx, val_idx in folds
        ]
        flat_results = Parallel(n_jobs=self.n_jobs)(jobs)

        # Postprocess scores
        fold_scores = {grid_id: [] for grid_id in range(len(param_combos))}
        for grid_id, score in flat_results:
            fold_scores[grid_id].append(score)

        for grid_id, params in enumerate(param_combos):
            all_scores = [score for score in fold_scores[grid_id] if score is not None]
            if all_scores:
                mean_score = np.mean(all_scores)
            else:
                mean_score = np.inf

            self.results.append({"params": params, "score": mean_score})
            if self.best_score is None or mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = params

        # Train final model with the best hyperparameter combination
        self.best_model = self.build_model(**self.best_params)
        data_final_args = (X,)
        if y is not None:
            data_final_args += (y,)
            if aux is not None:
                data_final_args += (aux,)
        try:
            self.best_model.train(*data_final_args, **train_kwargs)
            final_message = "Optimization of final model successful."
        except Exception as _:
            final_message = "Error encountered during optimization of final model, try using a different solver or increasing lambda2."

        if not self.best_model.solver_success:
            final_message = "Optimization of final model did not converge, try using a different solver or max_iters."

        print(f"{final_message} Solver={train_kwargs['solver']} | Iterations={self.best_model.solver_iter}")

    def to_latex_table(self, save_path=None, caption="Cross-validation results.", label="tab:cv_results", top_k=None, float_fmt="{:.4f}", param_labels=None, exclude_params=None):
        if not self.results:
            raise RuntimeError("No CV results available: call train() first.")

        sorted_results = sorted(self.results, key=lambda r: r["score"])
        if top_k is not None:
            sorted_results = sorted_results[:top_k]

        exclude_params = set(exclude_params) if exclude_params else set()
        param_names = [name for name in sorted_results[0]["params"].keys() if name not in exclude_params]
        labels = dict(param_labels) if param_labels else {}
        for name in param_names:
            labels.setdefault(name, name)

        col_spec = "l" * len(param_names) + "c"
        header = " & ".join([labels[name] for name in param_names] + ["CV score"])

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
        for i, r in enumerate(sorted_results):
            values = [str(r["params"][name]) for name in param_names]
            score_str = float_fmt.format(r["score"]) if np.isfinite(r["score"]) else "--"
            row = " & ".join(values + [score_str])
            if i == 0:
                row = r"\textbf{" + row.replace(" & ", "} & \\textbf{") + r"}"
            lines.append(row + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
        table_str = "\n".join(lines)

        if save_path is not None:
            with open(save_path, "w") as f:
                f.write(table_str)

        return table_str
