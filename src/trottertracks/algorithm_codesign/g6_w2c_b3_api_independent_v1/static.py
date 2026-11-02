"""Independent AST inventory; never imports or evaluates a scientific source."""
import ast
import hashlib
from pathlib import Path
from ..g6_w2c_b3_offline_result import require

MODULES=('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g6_w2c_b3_driver','g6_w2c_cost_preparation')
SOURCE='src/trottertracks/algorithm_codesign/'

def sha(b):return hashlib.sha256(b).hexdigest()

class Inventory(ast.NodeVisitor):
    def __init__(self,text):self.text=text;self.path=[];self.entries=[]
    def enter(self,node):
        name='.'.join(self.path+[node.name])
        # Owned syntax excludes nested function/class bodies; producer return_syntax
        # includes nested bodies and is retained separately as a syntactic comparison.
        def owned(n):
            for x in ast.iter_child_nodes(n):
                if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
                yield x
                yield from owned(x)
        nodes=list(owned(node))
        row=dict(name=name,kind=type(node).__name__,line=node.lineno,end_line=node.end_lineno,
                 source_sha256=sha(ast.get_source_segment(self.text,node).encode()),doc=ast.get_docstring(node),
                 owned_returns=sorted({ast.unparse(x.value) if x.value else 'None' for x in nodes if isinstance(x,ast.Return)}),
                 owned_raises=sorted({ast.unparse(x.exc) if x.exc else 'reraise' for x in nodes if isinstance(x,ast.Raise)}),
                 calls=sorted({ast.unparse(x.func) for x in nodes if isinstance(x,ast.Call)}),
                 assignments=sorted({ast.unparse(y) for x in nodes if isinstance(x,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for y in (x.targets if isinstance(x,ast.Assign) else [x.target])}),
                 input_parameters=ast.unparse(node.args) if hasattr(node,'args') else None,
                 runtime_schema='NOT_ESTABLISHED_STATIC_ONLY',dynamic_execution=False)
        if hasattr(node,'args'):
            row.update(signature=ast.unparse(node.args),returns_annotation=ast.unparse(node.returns) if node.returns else None,
                return_syntax=sorted({ast.unparse(x.value) for x in ast.walk(node) if isinstance(x,ast.Return) and x.value}),
                raise_syntax=sorted({ast.unparse(x.exc) for x in ast.walk(node) if isinstance(x,ast.Raise) and x.exc}))
        self.entries.append(row);self.path.append(node.name);self.generic_visit(node);self.path.pop()
    visit_FunctionDef=enter
    visit_AsyncFunctionDef=enter
    visit_ClassDef=enter

def inventory(root):
    result={}
    for name in MODULES:
        p=SOURCE+name+'.py';b=(Path(root)/p).read_bytes();v=Inventory(b.decode());v.visit(ast.parse(b,filename=p))
        result[p]=dict(sha256=sha(b),bytes=len(b),entries=v.entries,analysis='INDEPENDENT_AST_ONLY')
    return dict(schema='b3_API_independent_static_inventory_v1',modules=result,entries=sum(len(x['entries']) for x in result.values()),dynamic_execution=False)

def compare(actual,published):
    require(set(actual['modules'])==set(published['modules']),'STATIC_MODULE_SET')
    renderer_differences=[]
    for p,m in actual['modules'].items():
        ref=published['modules'][p];require(m['sha256']==ref['sha256'] and m['bytes']==ref['bytes'],'STATIC_SOURCE:'+p)
        rows={r['name']:r for r in m['entries']};rrows={r['name']:r for r in ref['entries']}
        require(rows.keys()==rrows.keys(),'STATIC_ENTRY_SET:'+p)
        for name,row in rows.items():
            for k,v in rrows[name].items():
                observed=row.get(k)
                if k in ('return_syntax','raise_syntax'):
                    # Python 3.10/3.12 unparse add different parentheses; compare AST.
                    try:
                        observed=sorted(ast.dump(ast.parse(x,mode='eval'),include_attributes=False) for x in observed)
                        v=sorted(ast.dump(ast.parse(x,mode='eval'),include_attributes=False) for x in v)
                    except SyntaxError:
                        renderer_differences.append(dict(entry=name,field=k,reason='Python3.12 PEP701 unparse is not Python3.10 syntax; exact segment SHA matches'))
                        continue
                require(observed==v,'STATIC_ENTRY:'+name+':'+k)
    return dict(status='PASS',modules=len(actual['modules']),entries=actual['entries'],runtime_schema_proven=False,renderer_differences=renderer_differences)

def node(root,module,name):
    b=(Path(root)/(SOURCE+module+'.py')).read_text();v=Inventory(b);v.visit(ast.parse(b));r=next(x for x in v.entries if x['name']==name)
    return b.splitlines()[r['line']-1:r['end_line']],r

def verify_mapping(root,mapping):
    arms=('ordinary','partial_return_tail','closed_P3_tail','full_return')
    require(mapping['dispatch_implemented'] is False and set(mapping['arms'])==set(arms),'MAPPING_DISPATCH_OR_ARMS')
    for arm in arms:
        r=mapping['arms'][arm];require(r['axes']==['X','Y'],'MAPPING_AXES')
        require(r['layer_A']=='g7_generator.make_generator -> '+('FullReturnGenerator.sample' if arm=='full_return' else 'CanonicalGenerator.sample'),'MAPPING_A')
        expected=('g6_f3_generator.prepare -> GeneratorSession.sample/angle_for_event' if arm=='full_return' else
          'g6_w2c_b3_driver.LegacyEndpoint: canonical_snapshot/AuditParity/event_witness; CanonicalGenerator.event(group_index, raw_word, child) requery; g6_f3_precision.enclose_atan')
        require(r['layer_B']==expected,'MAPPING_B')
    checks=[]
    specs=[('g7_generator','make_generator',['FullReturnGenerator','CanonicalGenerator']),
           ('g7_generator','FullReturnGenerator.sample',['self.packet','rational_event_weight','_event']),
           ('g6_f3_generator','prepare',['make_request','legacy.FullReturnGenerator','bind_law']),
           ('g6_f3_generator','GeneratorSession.sample',['self._generator.sample','self._stop','self._event_checks']),
           ('g6_f3_generator','GeneratorSession.angle_for_event',['self._generator.event','enclose_atan','fingerprint']),
           ('g6_w2c_b3_driver','LegacyEndpoint.__init__',['modules[\'f3\'].prepare','modules[\'g7\'].make_generator']),
           ('g6_w2c_b3_driver','LegacyEndpoint._angle_for.operation',['self.g.event',"self.modules['precision'].enclose_atan"])]
    for mod,name,calls in specs:
        lines,row=node(root,mod,name)
        require(set(calls)<=set(row['calls']),'MAPPING_CALLS:'+name)
        checks.append(dict(module=mod,entry=name,line=row['line'],calls=calls,status='PASS_STATIC_CALL_SITES'))
    text='\n'.join(node(root,'g6_f3_generator','prepare')[0]);require("if req.arm != 'full_return':" in text and "CANONICAL_PREFLIGHT_NOT_IMPLEMENTED" in text,'CANONICAL_F3_REJECTION')
    # x=0 identity is before positive canonical rejection, not evidence for positive input.
    require(text.index('if req.x == 0:')<text.index("if req.arm != 'full_return':")<text.index('assessment = assess_precision'),'F3_POSITIVE_REJECTION_ORDER')
    return dict(status='PASS_STATIC_MAPPING_ONLY',checks=checks,executed_science=False,codec_conversion='PENDING',bundle_injection='UNIMPLEMENTED')
