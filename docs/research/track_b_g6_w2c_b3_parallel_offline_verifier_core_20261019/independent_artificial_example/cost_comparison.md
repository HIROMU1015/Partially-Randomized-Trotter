# Parallel cost boundaries

Status: **VERIFIED_PARALLEL_COST_BOUNDARIES**.

Parent CPU + all-worker CPU is total CPU at the same declared host boundary. Pair CPU, launch, IPC and finalization are already contained and are not added again. Global wall is the outer wall, not the sum of overlapping walls. Coarse/fine are alternative partitions. Failure prefixes are labeled and excluded from normal scientific comparisons. Unknown costs stay null. Parallel concurrent and serial execution are different conditions; no scientific ranking is adopted.
