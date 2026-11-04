"""Opt-in exact rewrites and bounded-support internal block certificates.

Structural matching is a partition heuristic, not proof of compiler history or
of closeness. Every unmatched block is numerically enclosed before any bound is
issued. Disjoint components can be factored exactly; no sector projection or
full-register column fallback is used.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import difflib

from . import c_compiled_raw_local as local
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z
from .c_per_word_population_range import _digest

VERSION='exact_Clifford_support_components_Gram_telescope_Arb192_v1'


def exact_rewrite(record):
    """Wire-adjacent C4 Clifford multiplication and existing exact inverses.

SX^4=S^4=I with their declared gate phases. RZ(binary64 pi) is never
classified as a Clifford or dropped on the basis of numerical closeness.
"""
    local.validate(record);nodes=[];last=[None]*record['qubits'];counts=Counter()
    groups=[('id','sx','x','sxdg'),('id','s','z','sdg')]
    for index, original in enumerate(record['gates']):
        g=deepcopy(original);origins=[index]
        while g is not None:
            if g['name']=='id':counts['identity_removed']+=1;break
            previous=[last[q] for q in g['qubits']];older=previous[0]
            a=nodes[older] if older is not None and all(x==older for x in previous) else None
            inverse=a is not None and local._inverse(a['gate'],g)
            group=next((x for x in groups if a is not None and len(g['qubits'])==1
                and len(a['gate']['qubits'])==1 and a['gate']['name'] in x and g['name'] in x),None)
            if inverse or group is not None:
                a['active']=False;origins=a['origins']+origins
                for q,p in zip(g['qubits'],a['previous']):last[q]=p
                if inverse:
                    counts['CX_inverse_pairs' if g['name']=='cx' else 'other_inverse_pairs']+=1;g=None
                else:
                    name=group[(group.index(a['gate']['name'])+group.index(g['name']))%4]
                    counts['C4_Clifford_products']+=1
                    g=dict(name=name,qubits=g['qubits'],params_hex=[])
                continue
            nodes.append(dict(gate=g,previous=previous,origins=origins,active=True))
            for q in g['qubits']:last[q]=len(nodes)-1
            break
    result=dict(record,gates=[x['gate'] for x in nodes if x['active']])
    return result,dict(counts= dict(counts),global_phase_unchanged=True,
        theorem='exact C4 Clifford multiplication and inverses;intervening supports disjoint',
        output_origins=[x['origins'] for x in nodes if x['active']])


def verify_saved_plan(p,left,right):
    """Exact descriptor replay only;no Qiskit definitions or Arb evaluation."""
    local.validate(left);local.validate(right)
    if (p['version']!=local.VERSION or p['commute_CX'] is not False
            or p['left_id']!=_digest(left) or p['right_id']!=_digest(right)
            or p['plan_id']!=_digest({k:v for k,v in p.items() if k!='plan_id'})):
        raise ValueError('SAVED_PRIMITIVE_PLAN_SOURCE_IDENTITY')
    local._validate_plan(p)
    for source,e in zip((left,right),p['expansions']):
        gates=[];phase=F(float.fromhex(source['phase_hex']));multiplicity=Counter()
        for g in source['gates']:
            if len(g['qubits'])==1 or g['name']=='cx':gates.append(g);continue
            key=_digest(local._primitive(g));definition=e['definitions'][key]
            if definition['primitive']!=local._primitive(g):raise ValueError('PRIMITIVE_TARGET_IDENTITY')
            phase+=F(definition['phase_exact_rad']);multiplicity[key]+=1
            gates.extend(dict(item,qubits=[g['qubits'][q] for q in item['qubits']])
                         for item in definition['candidate']['gates'])
        if (gates!=e['record']['gates'] or phase!=F(e['phase_exact_rad'])
                or dict(multiplicity)!={k:v['multiplicity'] for k,v in e['definitions'].items()}):
            raise ValueError('COMPLETE_PRIMITIVE_EXPANSION_WITNESS')
    return True


def _items(skeleton):
    items=[dict(kind='CX',gate=e['gate'],blocks=e['blocks']) for e in skeleton['events']]
    items.extend(dict(kind='terminal',qubit=q,gates=g) for q,g in enumerate(skeleton['terminal']))
    return items


def _token(item):
    return ('CX',tuple(item['gate']['qubits'])) if item['kind']=='CX' else ('terminal',item['qubit'])


def _gates(item):
    if item['kind']=='terminal':return [dict(g,qubits=[item['qubit']]) for g in item['gates']]
    out=[]
    for q,block in zip(item['gate']['qubits'],item['blocks']):out.extend(dict(g,qubits=[q]) for g in block)
    return out+[item['gate']]


def _components(gates):
    parent={q:q for g in gates for q in g['qubits']}
    def root(q):
        while parent[q]!=q:q=parent[q]
        return q
    for g in gates:
        for q in g['qubits'][1:]:parent[root(q)]=root(g['qubits'][0])
    values={}
    for q in parent:values.setdefault(root(q),[]).append(q)
    return sorted((sorted(x) for x in values.values()),key=lambda x:x[0])


def partition(saved,*,max_support=4,guard=1,exact_clifford=True):
    """Partition all canonical input items;optionally pool adjacent anchors.

Each window is factored into disjoint support components, preserving order
within each component. SequenceMatcher only chooses boundaries;closeness is
not assumed. max_support bounds local matrix size,not machine resources.
"""
    if type(max_support) is not int or not 1<=max_support<=4:
        raise ValueError('LOCAL_SUPPORT_DIMENSION')
    if type(guard) is not int or guard<0 or type(exact_clifford) is not bool:
        raise ValueError('PARTITION_POLICY')
    sk=[];rewrites=[]
    for e in saved['expansions']:
        record,proof=exact_rewrite(e['record']) if exact_clifford else (e['record'],dict(counts={},global_phase_unchanged=True))
        sk.append(local.skeleton(dict(e,record=record),commute_CX=False));rewrites.append(proof)
    items=[_items(s) for s in sk];a,b=items
    matcher=difflib.SequenceMatcher(None,[_token(x) for x in a],[_token(x) for x in b],autojunk=False)
    opcodes=matcher.get_opcodes();pieces=[]
    for tag,i,k,j,l in opcodes:
        if tag=='equal':pieces.extend(dict(kind='anchor',left=[x,x+1],right=[y,y+1]) for x,y in zip(range(i,k),range(j,l)))
        else:pieces.append(dict(kind='gap',left=[i,k],right=[j,l]))
    windows=[]
    for i,p in enumerate(pieces):
        if p['kind']=='gap':
            lo=max(0,i-guard);hi=min(len(pieces),i+guard+1)
            # Terminal wire anchors are disjoint factors of one common boundary.
            # Pool all of them if the window reaches that boundary; this avoids
            # splitting a closing Clifford conjugation across separate factors.
            if any(a[x['left'][0]]['kind']=='terminal' for x in pieces[lo:hi]
                   if x['kind']=='anchor'):hi=len(pieces)
            if windows and lo<=windows[-1][1]:windows[-1][1]=max(hi,windows[-1][1])
            else:windows.append([lo,hi])
    by_start=dict(windows);factors=[];index=0;unsupported=[]
    while index<len(pieces):
        if index not in by_start:
            p=pieces[index]
            if p['kind']!='anchor':raise ValueError('UNCOVERED_ALIGNMENT_GAP')
            x,y=a[p['left'][0]],b[p['right'][0]]
            if x['kind']=='CX':
                for q,ga,gb in zip(x['gate']['qubits'],x['blocks'],y['blocks']):
                    factors.append(dict(kind='oneq',support=[q],left=ga,right=gb))
                # Same exact CX multiplies both arms after these one-qubit factors.
                factors.append(dict(kind='shared_CX',gate=x['gate']))
            else:factors.append(dict(kind='oneq',support=[x['qubit']],left=x['gates'],right=y['gates']))
            index+=1;continue
        end=by_start[index];window=pieces[index:end]
        spans=[[_[arm][0] for _ in window] for arm in ('left','right')]
        bounds=[[min(v),max(p[arm][1] for p in window)] for v,arm in zip(spans,('left','right'))]
        ga=[g for item in a[bounds[0][0]:bounds[0][1]] for g in _gates(item)]
        gb=[g for item in b[bounds[1][0]:bounds[1][1]] for g in _gates(item)]
        components=_components(ga+gb)
        for qs in components:
            mapq={q:j for j,q in enumerate(qs)}
            side=lambda seq:[dict(g,qubits=[mapq[q] for q in g['qubits']]) for g in seq if g['qubits'][0] in mapq]
            factors.append(dict(kind='block',support=qs,left=side(ga),right=side(gb),
                window_pieces=[index,end],source_item_spans=bounds,
                partition_proof='all input items covered;different support components commute exactly'))
            if len(qs)>max_support:unsupported.append(dict(support=qs,window_pieces=[index,end]))
        index=end
    value=dict(version=VERSION,source_primitive_plan_id=saved['plan_id'],left_id=saved['left_id'],
        right_id=saved['right_id'],qubits=saved['qubits'],max_support=max_support,guard=guard,
        exact_clifford=exact_clifford,rewrites=rewrites,normalized_CX_counts=[len(s['events']) for s in sk],
        input_item_counts=[len(x) for x in items],opcodes=[list(x) for x in opcodes],
        windows=windows,factors=factors,unsupported=unsupported,complete_partition=True,
        applicable=not unsupported)
    return dict(value,plan_id=_digest(value))


def make_plan(left,right,*,saved=None,max_support=4,guard=1,exact_clifford=True):
    saved=local.plan(left,right) if saved is None else saved
    verify_saved_plan(saved,left,right)
    return saved,partition(saved,max_support=max_support,guard=guard,exact_clifford=exact_clifford)


def _matrix(gates,n,cache):
    """Full local products with recenter radii and declared determinant phase."""
    from flint import acb_mat
    size=1<<n;v=acb_mat([[int(i==j) for j in range(size)] for i in range(size)])
    error=F();pi=F();rad=F()
    phases=dict(id=F(),x=F(1,2),z=F(1,2),h=F(1,2),sx=F(1,4),sxdg=F(-1,4),
                s=F(1,4),sdg=F(-1,4),rz=F(),ry=F(),p=F(),cx=F(1,4))
    for g in gates:
        key=(g['name'],tuple(g['params_hex']))
        if g['name'] not in phases:raise ValueError('LOCAL_ONEQ_CX_ONLY')
        if key not in cache:cache[key]=fixed._local(key[0],[float.fromhex(x) for x in key[1]])
        B=acb_mat(size,size)
        for i,j,a,b in local._embedding_layout(n,tuple(g['qubits'])):B[i,j]=cache[key][a,b]
        v,delta=local._midpoint_matrix(B*v);error+=delta;pi+=phases[g['name']]
        if g['name']=='p':rad+=F(float.fromhex(g['params_hex'][0]))/2
    v,delta=local._midpoint_matrix(z._scale(v,local._phase(-pi,-rad)))
    return v,error+delta,pi,rad


def _operator_upper(M):
    """Gershgorin on M* M and Frobenius, both outward Arb bounds."""
    from flint import acb_mat
    adj=acb_mat([[M[j,i].conjugate() for j in range(M.nrows())] for i in range(M.ncols())])
    H=adj*M;rows=[]
    for i in range(H.nrows()):
        value=z._fraction(H[i,i].real.upper())
        value+=sum((z._fraction(H[i,j].abs_upper()) for j in range(H.ncols()) if i!=j),F())
        rows.append(value)
    gram=z._fraction(z._real(max(rows)).sqrt().upper())
    return min(gram,F(z._norm(M)['upper_exact']))


def _internal_job(pair):
    from flint import ctx
    n=len(pair['support']);size=1<<n
    if not 1<=n<=4:raise ValueError('LOCAL_MATRIX_SUPPORT_ONLY_NO_DENSE_FALLBACK')
    with z._BALL_LOCK,ctx.workprec(192):
        cache={};a,ea,pa,ra=_matrix(pair['left'],n,cache);b,eb,pb,rb=_matrix(pair['right'],n,cache)
        options=[]
        for j in range(size):
            gamma=F(2*j,size)
            norm=_operator_upper(z._scale(a,local._phase(gamma))-b)+ea+eb
            options.append((norm,j,gamma))
        norm,j,gamma=min(options)
        return dict(upper_exact=str(min(F(2),norm)),delta_pi_exact=str(pa-pb-gamma),
            delta_rad_exact=str(ra-rb),root_index=j,root_order=size,support_dimension=size,
            phase_compensation_retained=True,phase_fit=False,rounding_enclosed=True,
            method='min(Gram Gershgorin,Frobenius)+outward recenter radii;finite central root gauge')


def _factor_job(pair):
    return local._block_job(pair) if pair['kind']=='oneq' else _internal_job(pair)


def evaluate(saved,p,*,parallel=True):
    """No original record/decomposition generation; caller binds saved inputs first."""
    from flint import ctx
    z.runtime_model()
    if p!=partition(saved,max_support=p['max_support'],guard=p['guard'],exact_clifford=p['exact_clifford']):
        raise ValueError('COMPLETE_BLOCK_PARTITION_IDENTITY')
    if (saved['plan_id']!=_digest({k:v for k,v in saved.items() if k!='plan_id'})
            or saved['version']!=local.VERSION):raise ValueError('SAVED_PLAN_IDENTITY')
    local._validate_plan(saved)
    dimension=1<<p['qubits']
    if not p['applicable']:
        return dict(status='NOT_APPLICABLE_BLOCK_SUPPORT',upper_exact=None,full_operator_coverage=False,
                    dimension=dimension,dense_fallback_used=False,unsupported=p['unsupported'])
    definitions={};dc=Counter()
    for e in saved['expansions']:
        for key,value in e['definitions'].items():
            if key in definitions and {k:v for k,v in value.items() if k!='multiplicity'}!={k:v for k,v in definitions[key].items() if k!='multiplicity'}:
                raise ValueError('CONFLICTING_PRIMITIVE_DEFINITION')
            definitions[key]=value;dc[key]+=value['multiplicity']
    dw,dr=local._map(local._definition_job,definitions.values(),parallel);dw=dict(zip(definitions,dw))
    dr['method']='independent unique local primitive matrices;process-local gate cache'
    blocks={};bc=Counter()
    for f in p['factors']:
        if f['kind']=='shared_CX':continue
        # Global labels do not affect a local operator norm, but support size does.
        body=dict(kind=f['kind'],support=list(range(len(f['support']))),left=f['left'],right=f['right'])
        key=_digest(body);blocks[key]=body;bc[key]+=1
    bw,br=local._map(_factor_job,blocks.values(),parallel);bw=dict(zip(blocks,bw))
    br['method']='independent unique SU2/internal Gram blocks;process-local gate cache'
    with z._BALL_LOCK,ctx.workprec(192):
        translation=sum((F(dw[k]['upper_exact'])*dc[k] for k in dw),F())
        blocksum=sum((F(bw[k]['upper_exact'])*bc[k] for k in bw),F())
        pi=sum((F(bw[k]['delta_pi_exact'])*bc[k] for k in bw),F())
        rad=F(saved['expansions'][0]['phase_exact_rad'])-F(saved['expansions'][1]['phase_exact_rad'])
        rad+=sum((F(bw[k]['delta_rad_exact'])*bc[k] for k in bw),F())
        phase=z._fraction((local._phase(pi,rad)-1).abs_upper())
    return dict(status='VALIDATED_INTERNAL_BLOCK_NATIVE_PAIR',version=VERSION,kind='rigorous_enclosure',
        norm='full_operator_2',dimension=dimension,full_operator_coverage=True,columns_evaluated=0,
        sector_projection=False,precision_bits=192,rounding_enclosed=True,phase_fit=False,
        coverage_proof='complete exact rewrite/partition;disjoint support tensor identity;unitary telescope',
        left_id=p['left_id'],right_id=p['right_id'],plan_id=p['plan_id'],
        upper_exact=str(min(F(2),translation+blocksum+phase)),lower_exact='0',
        translation_upper_exact=str(translation),blocks_upper_exact=str(blocksum),global_phase_upper_exact=str(phase),
        global_delta_pi_exact=str(pi),global_delta_rad_exact=str(rad),
        unique_definitions=len(dw),unique_blocks=len(bw),block_occurrences=sum(bc.values()),
        internal_block_occurrences=sum(f['kind']=='block' for f in p['factors']),
        maximum_support=max((len(f['support']) for f in p['factors'] if f['kind']!='shared_CX'),default=0),
        definition_witnesses=dw,block_witnesses=bw,definition_multiplicities=dict(dc),block_multiplicities=dict(bc),
        definition_resources=dr,block_resources=br,dense_fallback_used=False,
        nonunitary_literal_target_supported=False,all_word_uniform=False,
        all_q_words_compile_success_qualified=False,calibration_ready=False,actual_CA_achieved=None,Stage_C_ready=False)


def compare(left,right,*,parallel=True,**policy):
    saved,p=make_plan(left,right,**policy)
    result=evaluate(saved,p,parallel=parallel)
    if result['full_operator_coverage']:check_binding(result,left,right)
    return result


def check_binding(cert,left,right):
    """Scope and composition check; not a replay of the numerical proof."""
    local.validate(left);local.validate(right)
    if (cert.get('status')!='VALIDATED_INTERNAL_BLOCK_NATIVE_PAIR' or cert.get('version')!=VERSION
            or cert.get('left_id')!=_digest(left) or cert.get('right_id')!=_digest(right)
            or left['qubits']!=right['qubits'] or cert.get('dimension')!=1<<left['qubits']
            or cert.get('precision_bits')!=192 or cert.get('rounding_enclosed') is not True
            or cert.get('phase_fit') is not False or cert.get('full_operator_coverage') is not True
            or cert.get('columns_evaluated')!=0 or cert.get('sector_projection') is not False
            or cert.get('dense_fallback_used') is not False
            or any(cert.get(k) is not False for k in ('nonunitary_literal_target_supported','all_word_uniform',
                'all_q_words_compile_success_qualified','calibration_ready','Stage_C_ready'))
            or cert.get('actual_CA_achieved') is not None):raise ValueError('BLOCK_CERTIFICATE_TARGET_SCOPE')
    terms=[F(cert[k]) for k in ('translation_upper_exact','blocks_upper_exact','global_phase_upper_exact')]
    if min(terms)<0 or F(cert['upper_exact'])!=min(F(2),sum(terms,F())) or F(cert['lower_exact'])!=0:
        raise ValueError('BLOCK_TELESCOPE_COMPOSITION')
    return True
