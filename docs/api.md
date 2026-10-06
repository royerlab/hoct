# API reference

These entries are generated from the source docstrings. Start with
[tracking](tracking.md) for a complete workflow or
[incremental correction](correction.md) for a UI integration example.

## Models and prediction

::: hoct.load_model

::: hoct.available_models

::: hoct.predict

## Candidate graph

::: hoct.features.create_graph

## Solver configuration

Use `ILPSolverConfig.default()` to obtain the package's current tracking defaults.
When constructing a configuration directly, supply all required weight fields
and `tracklet_solver`.

::: hoct.tracking.ILPSolverConfig
    options:
      members: [default]

## Correction

::: hoct.correction.label_edge

::: hoct.correction.fit_from_labels

::: hoct.correction.ProbedModel
