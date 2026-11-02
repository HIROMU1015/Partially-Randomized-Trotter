"""Predeclared DF holdout with ONE transferable intercept; no data access.

Use the same NumPy OLS as the saved64 study, without its cohort intercepts or
within-cohort folds. Training must be frozen before any third-DF price access.
"""
TRAIN = ('M4_Proposed', 'tail_P4')
HOLDOUT = 'StageA_gamma0'
MODELS = {'mean_only':(), 'A_switch':('basis_switches',), 'B_overlap':('overlap_path',)}
VERSION = 'two_DF_global_intercept_third_DF_holdout_v1'


def _rows(rows):
    if len(rows) != 32 or [x['sample'] for x in rows] != list(range(32)):
        raise ValueError('FIXED32_SAMPLE_ORDER')


def _design(rows, features):
    import numpy as np
    x = np.asarray([[1.] + [r['features'][k] for k in features] for r in rows], dtype=float)
    if not np.isfinite(x).all(): raise ValueError('NONFINITE_FEATURE')
    return x


def freeze_training(cohorts, metric):
    """No heldout argument; no scale/intercept/feature tuning on third prices."""
    import numpy as np
    if set(cohorts) != set(TRAIN): raise ValueError('TRAIN_DF_IDS_NO_HELDOUT')
    for rows in cohorts.values(): _rows(rows)
    rows = [x for key in TRAIN for x in cohorts[key]]
    y = np.asarray([r['metrics'][metric] for r in rows], dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0): raise ValueError('FINITE_NONNEGATIVE_PRICES')
    models = {}
    for name, features in MODELS.items():
        X = _design(rows, features)
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=1e-12)
        full = rank == X.shape[1]
        models[name] = dict(features=list(features),rank=int(rank),
            status='FROZEN_TRAINING' if full else 'UNRESOLVED_RANK_DEFICIENT',
            coefficients=beta.tolist() if full else None)
    return dict(version=VERSION,train_DF_ids=list(TRAIN),heldout_DF_id=HOLDOUT,
                training_rows=64,training_DF_count=2,metric=metric,models=models,
                common_intercept=True,centering=False,scaling=False,
                heldout_price_accessed=False,optimizer_weights_adopted=False)


def evaluate(snapshot, rows, *, cohort):
    import numpy as np
    if (snapshot['version'] != VERSION or cohort != HOLDOUT
            or snapshot['train_DF_ids'] != list(TRAIN) or snapshot['heldout_DF_id'] != cohort
            or not snapshot['common_intercept'] or snapshot['heldout_price_accessed']
            or set(snapshot['models']) != set(MODELS)):
        raise ValueError('DF_HOLDOUT_IDENTITY_OR_LEAKAGE')
    _rows(rows);metric = snapshot['metric']
    y = np.asarray([r['metrics'][metric] for r in rows], dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0): raise ValueError('FINITE_NONNEGATIVE_PRICES')
    result = {}
    for name, features in MODELS.items():
        m = snapshot['models'][name]
        if m['features'] != list(features): raise ValueError('FROZEN_FEATURES_CHANGED')
        if m['status'] != 'FROZEN_TRAINING':
            result[name] = dict(status=m['status'],MAE=None,RMSE=None,negative_predictions=None)
            continue
        beta = np.asarray(m['coefficients'], dtype=float)
        if beta.shape != (len(features)+1,) or not np.isfinite(beta).all():
            raise ValueError('FROZEN_PARAMETERS')
        pred = _design(rows,features) @ beta
        result[name] = dict(status='DF_HELDOUT_NUMERICAL_PILOT',MAE=float(np.mean(abs(pred-y))),
            RMSE=float(np.sqrt(np.mean((pred-y)**2))),
            signed_mean_prediction_error=float(np.mean(pred-y)),
            negative_predictions=int(np.count_nonzero(pred < 0)),predictions=pred.tolist(),
            observed_sample_mean=float(np.mean(y)),predicted_sample_mean=float(np.mean(pred)))
    return dict(cohort=cohort,metric=metric,models=result,DF_count=3,
                new_heldout_DF_candidates=1,independent_population_sampling=False,
                generalization_established=False,
                causal_effect=None,head_effect_separated=False)


def qualification(rz):
    m = rz['models'];a=m['A_switch'];others=[m['mean_only'],m['B_overlap']]
    if any(x['MAE'] is None for x in [a]+others): status='UNRESOLVED'
    elif a['negative_predictions'] or not all(a['MAE'] < x['MAE'] for x in others):
        status='SWITCH_TRANSFER_NOT_SUPPORTED'
    else: status='SWITCH_TRANSFER_DIAGNOSTIC_PASS'
    return dict(status=status,criterion='A RZ MAE strictly below global mean and B;no negative A prediction',
                supports_new_fit_automatically=False,new_objective_adopted=False,
                independent_novelty=None,causal_effect=None)
