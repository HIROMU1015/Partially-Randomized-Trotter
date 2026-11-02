"""Continue B3 stages with immutable occupation number matrices shared by fork.

Only eager cache preparation is new. Each matrix is exactly the expression in
OccupationOracle.component; spectra, events, circuits, seeds and prices agree.
The already-running exploration pool is left unchanged.
"""
import h46_b3 as b
import h46_b3_finish as finish

original_context=b.native_context


def cached_context(system,prefixes):
    import numpy as np
    ctx=original_context(system,prefixes)
    for index,rotation in ctx.oracle.rotations.items():
        for mode in range(ctx.oracle.n):
            column=rotation[:,mode]
            ctx.oracle.numbers[index,mode]=ctx.oracle.one_body(np.outer(column,column.conj()))
    b.save(b.PRIVATE/(system+'_confirmation_oracle_cache.json'),dict(
        prepared_rotated_number_operators=len(ctx.oracle.numbers),
        shared_immutable_before_fork=True,scientific_expression_unchanged=True))
    return ctx


if __name__=='__main__':
    b.native_context=cached_context
    finish.pipeline()
