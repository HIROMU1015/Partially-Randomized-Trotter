"""Complete the explicit delta=.2 panel, reusing all already acquired costs."""
import json
import h46_long_time_signal as study


def run():
    import h46_long_time_native as native
    import h46_long_time_report as accounting
    existing=accounting.costs();ep=accounting.endpoints()
    eligible={e['candidate']['id'] for e in ep if e['status']=='ELIGIBLE'}
    targets=[]
    for row in study.candidates():
        anchor=3 if row['system']=='H4' else 5
        if row['q']!=round(row['T']/.2):
            continue
        if row['method']=='FULL' and row['formula']=='2nd':
            targets.append(row)
        elif row['method']=='PR' and row['prefix']==anchor:
            if (row['R_outer']==32 and row['K']==4) or (row['formula']=='2nd' and row['R_outer']==4 and row['K']==2):
                targets.append(row)
    missing=[r for r in targets if r['id'] not in existing and r['id'] in eligible]
    plan=dict(candidates=missing,all_fixed_delta_targets=targets,delta=.2,
        screening_trajectories_per_new_PR=9,already_priced=sum(r['id'] in existing for r in targets),
        no_eligible_endpoint=[r['id'] for r in targets if r['id'] not in eligible],
        PR_prefixes=dict(H4=3,H6=5),reason='Separate fixed delta/time effects from per-T parameter selection')
    study.save(study.PUBLIC/'fixed_delta_selection.json',plan)
    # Reuse the contexts and prefix preparations left by the q32 stage.
    old_context=native.native_context
    def cached(system,prefixes):
        ctx=native.CONTEXT.get(system)
        if ctx is None:
            return old_context(system,prefixes)
        from h46_new2_native import prepare
        for p in prefixes:
            if p not in ctx.preparations:
                ctx.preparations[p]=prepare(ctx.ham,p)
        return ctx
    native.native_context=cached
    return native.run_cost('FIXED_DELTA_EXPLORATION',plan)


if __name__=='__main__':
    study.install();print(json.dumps(run()))
