"""Stdlib-only case-ID plan. Ordinals are recipes, never event payloads.

Planning may run in PRE; resolving ordinals into events is execution-only.
The unchanged I authorization gate validates the expanded ID inventory.
"""
from __future__ import annotations

ARMS = ('SS', 'SP', 'PS', 'PP')
LENGTHS = dict(SS=14, SP=58, PS=26, PP=51)
GROUPS = dict(H_ID='F1.H', REFERENCE='F1.REFERENCE', LAW='F1.LAW',
    COMPONENT='F1.COMPONENT', MOTIF='F1.MOTIF', WRAPPER='F1.WRAPPER',
    COMPILE='F1.COMPILE', MEAN='F1.MEAN', CATALOG_EVENT='F1.CATALOG.EVENT',
    CATALOG_MEAN='F1.CATALOG.MEAN', PRICE_DIRECT='F1.PRICE.ENUM',
    PRICE_DP='F1.PRICE.DP', PRICE_COMPARE='F1.PRICE.COMPARE',
    ARM_PRICE='F1.PRICE.ARMS', SELECTOR='F1.SELECTOR', EXHAUSTIVE='F1.EXHAUSTIVE')


def iter_cases():
    def case(cid, handler, requires=(), **fields):
        return dict(id=cid, handler=handler, call_group=GROUPS[handler],
                    requires=list(requires), **fields)

    def references(base, arm, kind, **fields):
        low = base+'_90'; high = base+'_128'
        yield case(low, 'REFERENCE', arm=arm, reference_kind=kind, digits=90, **fields)
        yield case(high, 'REFERENCE', [low], arm=arm, reference_kind=kind,
                   digits=128, paired_reference=low, **fields)

    for arm in ARMS:
        target = arm+'_target_128'
        yield from references(arm+'_target', arm, 'target')
        yield case(arm+'_H', 'H_ID', [target], arm=arm, reference=target)
        # Stable ordinal in child-first lexicographic n0 then n2 order.
        for ordinal in range(LENGTHS[arm]+LENGTHS[arm]**3):
            yield case(f'{arm}_law_{ordinal:06d}', 'LAW', [target],
                       arm=arm, reference=target, event_ordinal=ordinal)
        components=[]
        for atom in range(LENGTHS[arm]):
            base=f'{arm}_atom_{atom:03d}'
            yield from references(base, arm, 'atom', atom_index=atom)
            for suffix, role, order in [('P','product',0),('R0','rotation',0),('R2','rotation',2)]:
                cid=base+'_'+suffix; components.append(cid)
                yield case(cid, 'COMPONENT', [base+'_128'], arm=arm,
                           atom_index=atom, role=role, order=order, reference=base+'_128')
        yield case(arm+'_mean', 'MEAN', [target,*components], arm=arm,
                   reference=target, components=components)
        for motif in range(8):
            base=f'{arm}_motif_{motif}'
            yield from references(base, arm, 'motif', motif_index=motif)
            for variant in ('raw','optimized','adjoint'):
                yield case(base+'_'+variant, 'MOTIF', [base+'_128'], arm=arm,
                           motif_index=motif, variant=variant, reference=base+'_128')
            for axis in ('X','Y'):
                wrapper=base+'_'+axis
                yield case(wrapper, 'WRAPPER', [base+'_optimized',base+'_128'], arm=arm,
                           axis=axis, evolution=base+'_optimized', reference=base+'_128')
                yield case(wrapper+'_compile', 'COMPILE', [wrapper,base+'_128'], arm=arm,
                           wrapper=wrapper, reference=base+'_128')
        yield case(arm+'_price', 'ARM_PRICE', arm=arm)

    events={variant:[] for variant in ('raw','optimized')}
    prices=[]; event_by_price={}
    for ordinal in range(68):
        base=f'catalog_{ordinal:02d}'
        yield from references(base, 'catalog', 'catalog', event_ordinal=ordinal)
        for variant in events:
            cid=base+'_'+variant; events[variant].append(cid)
            yield case(cid, 'CATALOG_EVENT', [base+'_128'], arm='catalog',
                       event_ordinal=ordinal, variant=variant, reference=base+'_128')
        cid=base+'_price'; prices.append(cid); event_by_price[cid]=base+'_optimized'
        yield case(cid, 'PRICE_DIRECT', [base+'_optimized'], arm='catalog', event=base+'_optimized')
    for variant, ids in events.items():
        yield case('catalog_mean_'+variant, 'CATALOG_MEAN', [*ids,'catalog_00_128'],
                   arm='catalog', events=ids, reference='catalog_00_128')
    yield case('catalog_dp', 'PRICE_DP', arm='catalog')
    yield case('catalog_price_compare', 'PRICE_COMPARE',
               [*prices,*events['optimized'],'catalog_dp'], arm='catalog',
               direct_prices=prices, event_by_price=event_by_price, dp='catalog_dp')
    price_ids={a:a+'_price' for a in ARMS}
    for cid, handler in [('selector','SELECTOR'),('exhaustive','EXHAUSTIVE')]:
        yield case(cid, handler, price_ids.values(), prices=price_ids)
