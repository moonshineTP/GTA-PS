import numpy as np

from .base import PrioritizedSweepingModule, BellmanMixin, TdMixin, PlanningModel
from .oracle_model import OracleModel
from .standard_ps import StandardPS
from .batching import BatchedPriorityModule
from .gps_h2 import GpsH2
from .gta_ps import GtaPS
from .vast import GtaPsFa, LinearStateProjection, LinearTabularModel, VaST
from .dp_host import DpHost
from .dyna_host import DynaHost
from .empirical_model import EmpiricalModel, CountModel, SampleModel

# ---------------------------------------------------------------------------
# Variant → adaptive flag map
# ---------------------------------------------------------------------------
_ADAPTIVE_FLAGS: dict = {
    "standard_ps": {},
    "gta_base":    {},
    "gta_beta":    {"adaptive_beta":  True},
    "forward_gta": {"adaptive_beta":  True},
    "full_gta":    {"adaptive_beta":  True},
}

_CANONICAL_VARIANT: dict[str, str] = {
    "standard_ps": "standard_ps",
    "gta_base": "gta_base",
    "gta_beta": "gta_beta",
    "forward_gta": "gta_beta",
    "full_gta": "gta_beta",
}

_FIXED_MODULE_KWARGS: dict[str, dict] = {
    "forward_gta": {"alpha": 0.0},
}

VARIANTS = list(_ADAPTIVE_FLAGS.keys())


def make_agent(
    variant: str,
    host: str = "dp",
    *,
    # DP host args (oracle model)
    P: "np.ndarray | None" = None,
    r: "np.ndarray | None" = None,
    gamma: float = 0.95,
    q: int = 20,
    # Dyna host args (empirical model)
    n_states: int | None = None,
    n_actions: int | None = None,
    eta_td: float = 0.1,
    model_type: str = "count",
    rng: "np.random.Generator | None" = None,
    **module_kwargs,
) -> "DpHost | DynaHost":
    """
    Unified agent factory.

    Parameters
    ----------
    variant : str
        One of ``VARIANTS``.
    host : {'dp', 'dyna'}
        ``'dp'``   — oracle model (DpHost + OracleModel). Requires ``P`` and ``r``.
        ``'dyna'`` — empirical model (DynaHost + CountModel / SampleModel).
                     Requires ``n_states`` and ``n_actions``; ``P`` / ``r`` are
                     used to warm-start the GTA-PS topology if provided.
    P, r : ndarray
        Oracle transition kernel and reward vector.  Required for ``host='dp'``.
        Optional warm-start for ``host='dyna'``.
    gamma : float
        Discount factor.
    q : int
        Top-q frontier size (GTA-PS modules only).
    n_states, n_actions : int
        State / action space sizes.  Required for ``host='dyna'`` when ``P`` is
        not supplied; inferred from ``P.shape`` otherwise.
    eta_td : float
        TD learning rate (Dyna host only).
    model_type : {'count', 'sample'}
        Empirical transition model used by ``host='dyna'``.
    rng : np.random.Generator | None
        Optional RNG used by Dyna planning rollouts.
    **module_kwargs
        Forwarded to GtaPS (e.g. ``beta1``, ``beta2``, ``alpha``, ``eta``, ``b``).

    Returns
    -------
    DpHost or DynaHost, depending on ``host``.
    """
    import numpy as np

    if host not in ("dp", "dyna"):
        raise ValueError(f"host must be 'dp' or 'dyna', got {host!r}")
    if variant not in _ADAPTIVE_FLAGS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANTS}.")

    # --- infer dimensions ---
    if P is not None:
        _n_actions, _n_states, _ = P.shape
    else:
        _n_states = n_states
        _n_actions = n_actions

    if _n_states is None or _n_actions is None:
        raise ValueError("Provide P (shape (|A|,|S|,|S|)) or explicit n_states+n_actions.")

    # --- build PS module ---
    canonical_variant = _CANONICAL_VARIANT[variant]

    if canonical_variant == "standard_ps":
        if P is None or r is None:
            raise ValueError("standard_ps requires P and r.")
        module = StandardPS(P, r, gamma=gamma)
    else:
        flags = _ADAPTIVE_FLAGS[variant].copy()
        # _ADAPTIVE_FLAGS is authoritative for adaptive_beta
        module_kwargs = {k: v for k, v in module_kwargs.items()
                         if k not in ("adaptive_beta",)}
        module_kwargs.update(_FIXED_MODULE_KWARGS.get(variant, {}))
        if P is None or r is None:
            raise ValueError(f"variant '{variant}' requires P and r for topology initialisation.")
        module = GtaPS(P, r, gamma=gamma, q=q, **flags, **module_kwargs)

    # --- wrap in host ---
    if host == "dp":
        if P is None or r is None:
            raise ValueError("host='dp' requires P and r.")
        model = OracleModel(P, r)
        return DpHost(model, r, gamma, module)
    else:
        return DynaHost(
            _n_states,
            _n_actions,
            gamma,
            eta_td,
            module,
            model_type=model_type,
            r_init=r,
            rng=rng,
        )





__all__ = [
    "PrioritizedSweepingModule",
    "BellmanMixin",
    "TdMixin",
    "PlanningModel",
    "OracleModel",
    "StandardPS",
    "BatchedPriorityModule",
    "GpsH2",
    "GtaPsFa",
    "LinearStateProjection",
    "LinearTabularModel",
    "VaST",
    "DynaPS",
    "GtaPS",
    "DpHost",
    "DynaHost",
    "EmpiricalModel",
    "CountModel",
    "SampleModel",
    "make_agent",
    "VARIANTS",
]
