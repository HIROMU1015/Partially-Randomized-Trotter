"""Reuse the successful H2 numerical function bodies with H4 metadata bindings."""
import importlib
from types import FunctionType, SimpleNamespace
import h4_archived_exploratory as c
from h2_accepted_compare_backend import Backend as Existing
from h2_archived_exploratory_backend import Backend as H2


def rebound(fn):
    """Retarget metadata constants only; retain the original numerical code object."""
    return FunctionType(fn.__code__,dict(fn.__globals__,c=c),fn.__name__,fn.__defaults__,fn.__closure__)


class Backend:
    def __init__(self,session):
        c.require(type(session) is c.Session and session.used['bootstrap']==1,'AUTHORIZED_H4_SESSION_REQUIRED')
        self.s=session
        self.accounting_module=importlib.import_module('trottertracks.resource_applicability.h6_matched_accounting_v1')
        self.shot_module=importlib.import_module('trottertracks.resource_applicability.ax2b_numerical_accounting')
        self.original_metric=self.accounting_module.metric_summary
        self.original_headroom=self.shot_module.axis_headroom
        self.accounting_module.metric_summary=lambda *a,**kw:session.call('metric_summary',self.original_metric,*a,**kw)
        self.shot_module.axis_headroom=lambda *a,**kw:session.call('axis_headroom',self.original_headroom,*a,**kw)
        self.legacy=Existing.__new__(Existing)  # Never invoke circuit setup/native constructors.
        (session.root/'delegated_receipts/results').mkdir()
        self.legacy.s=SimpleNamespace(root=session.root/'delegated_receipts')
        self.legacy.shots=lambda *a,**kw:session.call('u_aware_shots',self.shot_module.u_aware_shots,*a,**kw)

    close=H2.close

    def load(self):
        c.require(self.s.active=='bundle_load','AUTHORIZED_BUNDLE_LOAD')
        return c.saved_inputs(self.s.cfg)

    def allowance(self,candidate,diagnostic):
        # H4's accepted floor/reference policy is inherited; H2 diagnostics are not substituted.
        return dict(real=diagnostic['u'],imag=diagnostic['u'],log_B_margin=diagnostic['log_B_margin'],
            kind='EMPIRICAL',certified=False,scope=c.SERIES,recomputed=False,
            reference_absolute_error=None,reference_assumption='CONDITIONAL_SAVED_H4_REFERENCE')

    def account(self,candidate,diagnostic,allowance,costs):
        d=dict(bias=diagnostic['bias'],primary=dict(log_B=diagnostic['log_B']))
        result=rebound(Existing.account)(self.legacy,candidate,d,allowance,costs)
        # Reproduce the accepted shot integers, without recalculating any reference/bias.
        for axis in ('real','imag'):
            c.require(result['shots']['axes'][axis]['shots']==diagnostic['saved_shots'][axis],
                      'H4_SAVED_SHOT_REPRODUCTION:'+axis)
        return result

    statistics=rebound(H2.statistics)

    def rank(self,rows):
        result=rebound(Existing.rank)(self.legacy,rows)
        # Both questions are recorded explicitly; the H2 directional reversal flag is preserved.
        if all(r['eligible'] for r in rows):
            result['four_method_point_order']=[r['candidate'] for r in sorted(rows,key=lambda r:r['G'])]
            result['opposite_order_directions']=(rows[1]['G']-rows[0]['G'])*(rows[3]['G']-rows[2]['G'])<0
        else:
            result['four_method_point_order']=None;result['opposite_order_directions']=None
        result['causal_attribution']=None
        return result
