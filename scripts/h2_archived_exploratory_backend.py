"""Narrow adapter: call existing numerical methods without their circuit setup."""
import importlib
from types import SimpleNamespace
import h2_archived_exploratory as c
from h2_accepted_compare_backend import Backend as Existing


class Backend:
    def __init__(self,session):
        c.require(type(session) is c.Session and session.used['bootstrap']==1,'AUTHORIZED_ARCHIVED_SESSION_REQUIRED')
        self.s=session
        self.accounting_module=importlib.import_module('trottertracks.resource_applicability.h6_matched_accounting_v1')
        self.shot_module=importlib.import_module('trottertracks.resource_applicability.ax2b_numerical_accounting')
        self.original_metric=self.accounting_module.metric_summary
        self.original_headroom=self.shot_module.axis_headroom
        self.accounting_module.metric_summary=lambda *a,**kw:session.call('metric_summary',self.original_metric,*a,**kw)
        self.shot_module.axis_headroom=lambda *a,**kw:session.call('axis_headroom',self.original_headroom,*a,**kw)
        self.legacy=Existing.__new__(Existing)  # Never call the constructor/setup/native methods.
        (session.root/'delegated_receipts/results').mkdir()
        self.legacy.s=SimpleNamespace(root=session.root/'delegated_receipts')
        empirical=self.accounting_module.empirical_allowance
        shots=self.shot_module.u_aware_shots
        self.legacy.accounting=SimpleNamespace(empirical_allowance=lambda **kw:session.call('empirical_allowance',empirical,**kw))
        self.legacy.shots=lambda *a,**kw:session.call('u_aware_shots',shots,*a,**kw)

    def close(self):
        self.accounting_module.metric_summary=self.original_metric
        self.shot_module.axis_headroom=self.original_headroom

    def load(self):
        c.require(self.s.active=='bundle_load','AUTHORIZED_SAVED_INPUT_LOAD')
        data=c.saved_inputs(self.s.cfg);self.legacy.reference_id=data['reference_id'];return data

    def allowance(self,candidate,diagnostic):
        return Existing.allowance(self.legacy,candidate,diagnostic)

    def account(self,candidate,diagnostic,allowance,costs):
        return Existing.account(self.legacy,candidate,diagnostic,allowance,costs)

    def statistics(self,candidate,row,costs):
        # Shape adapters only; the original resource_row retains paired G variability.
        cost_rows=[dict(replica=t['task'],axis=ax,control='symmetric_directional',metrics=dict(RZ=t[ax]))
                   for t in costs for ax in ('cosine','sine')]
        shot_input=dict(row['shots'],cost_acquisition_eligible=row['eligible'])
        signal=dict(cell=dict(id=candidate.id,method=candidate.method),
                    precision_rows={str(.0005):{'1.0':shot_input}})
        stats=self.s.call('resource_row',self.accounting_module.resource_row,signal,cost_rows,.0005)
        if stats is not None:
            axis_statistics=stats['axes']
        else:
            # C-only always uses the same two metric_summary calls. No fake N/G.
            axis_statistics={a:self.accounting_module.metric_summary([x for x in cost_rows if x['axis']==a],'RZ')
                             for a in ('cosine','sine')}
        # Expose the two terms already used by Existing.account; preserve its operation order.
        G_axis=({a:row['shots']['axes'][ch]['shots']*sum(t[a] for t in costs)/len(costs)
                 for a,ch in (('cosine','real'),('sine','imag'))} if row['eligible'] else None)
        if G_axis is not None:c.require(sum(G_axis.values())==row['G'],'DELEGATED_ACCOUNTING_AXIS_CLOSURE')
        result=dict(candidate=candidate.id,series=c.SERIES,
            numeric_implementation_series=row['series'],saved_cost_sample_tasks=[t['task'] for t in costs],
            compiled_RZ_statistics=axis_statistics,analytic_shots=row['shots'],allowance=row['allowance'],
            G_axis=G_axis,G_total=row['G'],eligible=row['eligible'],
            paired_resource_statistics=stats,
            point_arithmetic='Primary G/axis terms use original COMPARE multiply-then-divide order; resource_row G_point uses original mean-then-multiply order',
            nominal_interval=None,nominal_interval_reason='Reused metric_summary/resource_row define SD/SE; no new interval rule adopted',
            population_mean_certified=False,unobserved_events_assigned_zero=False,
            normalization_recomputed=False,vector_payload_decoded=False,
            status='EXPLORATORY_CONDITIONAL_ACCOUNTING' if row['eligible'] else 'C_ONLY_N_G_NOT_ELIGIBLE')
        c.save(self.s.root/'results'/(candidate.id+'_exploratory.json'),result);return result

    def rank(self,rows):
        return Existing.rank(self.legacy,rows)
