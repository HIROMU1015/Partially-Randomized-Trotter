#!/usr/bin/env python3
"""C7-4 saved fit diagnostics and reports; no new fit or circuit simulation."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_d_c7_4_models/2026-11-07/run02'
DOC=ROOT/'docs/research/track_d_c7_4_model_stability_20261107'
def read(name):return json.loads((OUT/name).read_text())
def write(name,value):
    with (OUT/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def num(v):return format(v,'.8g') if isinstance(v,float) else str(v)
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(map(num,r))+'|' for r in rows])
def save(name,text):
    with (DOC/name).open('x') as f:f.write(text)


def main():
    result=read('result.json')
    if result['status']!='C7_4_ANALYSIS_COMPLETE_MANDATORY_STOP':raise ValueError('COMPLETE_C7_4_REQUIRED')
    stats=read('integrated_statistics.json');features=read('event_features.json');prices=read('saved_prices_scalar.json')
    models={k:read('model_'+k+'.json') for k in ('frozen_C6a','affine','sqrt_lane','basis_runs','lane_runs')}
    labels=stats['quantity_labels'];q=np.array([r['q'] for r in features]);qs=sorted(set(q))
    lookup={(r['sample_id'],r['method'],r['axis']):r for r in prices}
    def value(f,label):
        m,a,g=label;sid=f['sample_id']
        if m=='paired_delta':return lookup[sid,'reference',a]['metrics'][g]-lookup[sid,'symmetric_directional',a]['metrics'][g]
        return lookup[sid,m,a]['metrics'][g]
    y=np.array([[value(f,l) for l in labels] for f in features],float)
    scores=[]
    for name,model in models.items():
        for mode in ['residuals']+(['leave_one_q_out'] if name!='frozen_C6a' else []):
            for label in labels:
                rows=[r for r in model[mode] if r['quantity']==label]
                scores.append(dict(model=name,evaluation=mode,quantity=label,
                    equal_q_mean_error_RMSE=float(np.sqrt(np.mean([r['mean_error']**2 for r in rows]))),
                    equal_q_path_RMSE=float(np.sqrt(np.mean([r['RMSE']**2 for r in rows]))),
                    max_abs_q_mean_error=max(abs(r['mean_error']) for r in rows)))
    write('model_scores.json',dict(rows=scores,model_families_tried=5,new_fit_families=4,
        CV_folds=5,source='saved fitted/CV residuals only; no refit',confirmatory=False))
    def center(v):
        v=np.array(v,float).copy()
        for a in qs:v[q==a]-=v[q==a].mean()
        return v
    correlations=[]
    for key in ['basis_runs','lane_max_runs','lane_imbalance','support_work']:
        x=center([f[key] for f in features])
        for j,label in enumerate(labels):
            z=center(y[:,j]);den=np.linalg.norm(x)*np.linalg.norm(z)
            correlations.append(dict(feature=key,quantity=label,within_q_centered_Pearson=None if den<1e-12 else float(x@z/den)))
    maxgap=[]
    for a in qs:
        f=[f for f in features if f['q']==a];ea=np.mean([r['lane_A_runs'] for r in f]);eb=np.mean([r['lane_B_runs'] for r in f])
        em=np.mean([r['lane_max_runs'] for r in f]);maxgap.append(dict(q=int(a),mean_max_lane_runs=float(em),max_mean_lane_runs=float(max(ea,eb)),gap=float(em-max(ea,eb))))
    q32=ROOT/'artifacts/track_d_c7_3_q32/2026-11-07/run01_accelerated_completion/paths'
    dag=[]
    for f in [f for f in features if f['q']==32]:
        for m in ('symmetric_directional','reference'):
            for axis in ('X','Y'):
                p=q32/f['sample_id']/(m+'_'+axis+'_features.json');d=json.loads(p.read_text())
                for metric in ('rz_depth','cx_depth','total_depth'):
                    end=d['weighted_wire_finish'][metric]
                    dag.append(dict(sample_id=f['sample_id'],method=m,axis=axis,metric=metric,
                        wire_max=max(end['qubit_finish']),price=d['metrics'][metric],exact=end['depth']==d['metrics'][metric]))
    rank=int(np.linalg.matrix_rank(np.column_stack((np.ones(len(q)),q,q-1,q-2,np.ones(len(q))))))
    write('feature_diagnostics.json',dict(correlations=correlations,correlations_causal=False,
        empirical_Emax_gap=maxgap,step_boundary_cut_design_columns=['1','q','q-1','q-2','cut_loss=1'],
        step_boundary_cut_design_rank=rank,separate_causal_coefficients_identifiable=False,
        q32_compiled_DAG_records=dag,compiled_DAG_feature_is_future_predictor=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    basis=models['basis_runs'];old=models['frozen_C6a'];aff=models['affine'];sqrt=models['sqrt_lane']
    fitted=[];comparison=[];coefs=[]
    for g in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size'):
        label=['paired_delta','X',g]
        score=lambda n:next(r for r in scores if r['model']==n and r['evaluation']=='leave_one_q_out' and r['quantity']==label)
        comparison.append([g,*[score(n)['equal_q_mean_error_RMSE'] for n in ['affine','sqrt_lane','basis_runs','lane_runs']]])
        j=labels.index(label);coefs.append([g,*[r[j] for r in aff['coefficients']],*[r[j] for r in basis['coefficients']]])
        for a in qs:
            r=lambda mod:next(r for r in mod['residuals'] if r['q']==a and r['quantity']==label)
            fitted.append([int(a),g,r(old)['observed_mean'],r(old)['mean_error'],r(aff)['mean_error'],r(sqrt)['mean_error'],r(basis)['mean_error']])
    absolute=[]
    for m in ('symmetric_directional','reference'):
        for g in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size'):
            label=[m,'X',g]
            absolute.append([m,g,*[next(r for r in scores if r['model']==n and r['evaluation']=='leave_one_q_out' and r['quantity']==label)['equal_q_mean_error_RMSE'] for n in ['affine','sqrt_lane','basis_runs','lane_runs']]])
    future=read('frozen_C6a_future.json');fp=[]
    for i,a in enumerate(future['q']):
        for g in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size'):
            j=labels.index(['paired_delta','X',g]);fp.append([a,g,future['predictions'][i][j],aff['future']['predictions'][i][j],sqrt['future']['predictions'][i][j],sqrt['future']['nominal_bootstrap_SD'][i][j]])
    # Plot saved means and within-q SD: these bars are not mean confidence intervals.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,g in zip(axes.flat,('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')):
        points=[next(r for r in group['quantities'] if r['quantity']==['paired_delta','X',g]) for group in stats['by_q']]
        ax.errorbar(qs,[p['mean'] for p in points],yerr=[p['sample_SD'] for p in points],fmt='ko',label='saved mean +/- path SD')
        for n in ['frozen_C6a','affine','basis_runs']:
            rs=[r for r in models[n]['residuals'] if r['quantity']==['paired_delta','X',g]]
            ax.plot([r['q'] for r in rs],[r['prediction_mean'] for r in rs],label=n)
        ax.set_xscale('log',base=2);ax.set_title(g);ax.set_xlabel('q');ax.set_ylabel('Reference - strong')
    axes.flat[0].legend(fontsize=7);fig.savefig(OUT/'paired_price_models.svg',metadata={'Date':None});plt.close(fig)
    source=result['source_commit'];link='../../../artifacts/track_d_c7_4_models/2026-11-07/run02/'
    save('MODEL_REPORT.md',f'''# C7-4：保存88 whole-pathの価格モデル再解析

2026-11-07 JST。実行SOURCE `{source}`。[PLAN](PLAN.md)／[初回schema停止と修正](PREPARATION_NOTE.md)／[保存統計]({link}integrated_statistics.json)／[全候補score]({link}model_scores.json)／[図]({link}paired_price_models.svg)。本書は許可された技術調査・探索的再解析で、詳細科学レビューではない。

## データ・条件・証拠の範囲

Linear H4/1Å/STO-3G/Nα=Nβ=2、number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground仮定・準備費除外。q2/4/8/16/32のn=16/16/32/16/8、計88 path・352 price records。single_hadamard_interrogation_without_state_preparation、strongのbasis共有/head相対融合/directional/step境界融合・scalar集約と一つのwhole-path Reference gadgetを維持。同P0(Qiskit1.3,opt1,seed17,rz/sx/x/cx,approx1,initially_zero=false,topology-free/宣言位相保持)。旧C6a/frozen係数・共分散と各旧診断・SOURCE/STOPは不変。[入力hash]({link}input_bindings.json)／[統合scalar]({link}saved_prices_scalar.json)。packet/endpoint/lawの完全identityも価格recordから検査した。

C7-3最終acea439、sampling/raw35bfd、compile/post94e94を区別。C6a e9d4/C6b fed5/C7-1 86c5の固定価格bytesと照合した。新H4 sampling/build/compile・追加標本・N/ground/DF fit・全RPE費用は0。X/Yを倍の独立nとしない。各q内ddof1共分散は方式/資源/axisの依存を保持。q間のideal joint-IIDや有限bit law誤差、population support/tailは未資格。

## CX：成立した部分と全P0 bridge

全88保存pathのX/Yで整数残差ΔCX−(208−224q)=0。有限観測の支持であり、母分散0/全path公式ではない。C7-2の局所非零RZ/RZZ control premium、単体CSWAP8CXと、無条件加法性への反例を保持する。

一般会計は ΔCX=F_gadget+F_ends+q*d_step+(q−2)*b_R−(q−1)*b_S+T(q,ω)+ε_P0(q,ω)。別の狭い候補は2*N*f+2*G0−2*q*(Pdet+r)+R_CX。N8/r32、onebody8＋二つの各8RZ/28RZZをprune後も非零で保てばPdet80から傾き−224を説明できる。これは条件付きの中心数で、任意support/角度での閉包を証明したものではない。f8の孤立価格をshell固定費へ無条件加算できず、scalar-only shellにもC7-2反例がある。G0=40を208から逆算して確定しない。

必要条件は、(i)prune/group後の非零primitive数、(ii)signed/zero/special-angleとsame/different basis transition、(iii)ordinary block0をdirectionalと混同しないこと、(iv)31/30等のstrong/reference境界会計とglobal cut損失を一回だけ含むこと、(v)実Gaussian閉包と端部/CSWAP/χの全P0界面残差を全supportで覆うこと。保存88本では零角等を下記featureへ記録したが未観測例外は排除しない。全P0残差0の十分条件は、上記の局所展開規約が任意合法列で閉じ、R_CX=0と一様に示せること。現在は**primitive/boundaryの部分説明＋有限支持、P0_GLOBAL_BRIDGE_MISSING**。[C7-2](../track_d_c7_2_cx_structure_20261107/REPORT.md)。次はGaussian/CSWAP/cut界面をそのSOURCEで限定分類し、既存人工反例を満たす条件付きbridgeを作る最小技術検証であり、大量の追加分子compileで代用しない。

## モデル仕様・識別性・過学習

比較は旧凍結C6a、全5q fixed+bulk affine(2係数)、1/q/sqrt(q)(3係数)、1/q/total basis runs(3係数)、1/q/max(A_runs,B_runs)(3係数)の5 family。新fitは4 family、36 response（方式別24＋paired12）を同時に扱う。各qの総重みを1とするWLSでn差を保持した。step/boundary/cut列[1,q,q−1,q−2,1]のrankは{rank}。したがってfixed+bulkと別々のboundary/cut因果係数をデータから識別できず、追加fitをしない。

run特徴は保存eventのdescriptorを数えただけ、O(qr)。Gaussian/control/head cost全体の完全な価格ではない。左右laneのmaxはpathごとに先に取り、その後標本平均にする。[feature診断]({link}feature_diagnostics.json)。compiled DAGのwire endpointのmaxはq32の全96 metric記録で既存価格と一致したが、これは実compile後のoracle的再記述。未来qの入力にすると情報漏洩となるため採用しない。単純shell+max(lane)はC7-2で反例があり、定理にしない。

各候補5-fold leave-one-q-outはモデル選択にも使った探索的CV。q32 foldは短qのみからの外挿で、他のfoldは補間も含む。q2/4を元から使う凍結モデルの残差はCVと呼ばない。全88本をモデル設計に使用したため、q32を新モデルの独立holdoutとして再使用しない。256 bootstrapはq内whole-pathの**統計的再標本化**で新trajectory生成ではない。全method/axis/metricに同じ再標本indexを用い、係数共分散と未来round間共分散を保存。pseudo-IID条件付きの校正変動であり、bias/tail/law/coverageの保証を与えない。

## 保存qでのpaired残差（X、Yは同じ価格）

符号は観測−予測。各qのpath変動・方式別平均/SD・全共分散はJSONにある。

{table(['q','metric','observed mean','frozen error','new affine error','sqrt error','basis-run error'],fitted)}

LOQのq平均誤差RMSE（5qを同じ重み）:

{table(['metric/paired X','affine(2)','sqrt(3)','basis-run(3)','lane-run(3)'],comparison)}

方式別絶対価格のLOQ q平均誤差RMSE:

{table(['method','metric/X','affine','sqrt','basis-run','lane-run'],absolute)}

係数（paired X、Y同値、b0+bq*qまたはc0+cq*q+cr*runs）:

{table(['metric','affine b0','affine bq','basis c0','basis cq','basis cr'],coefs)}

countsでは全q affineが小さな残差で十分であり、basis/laneを増やす利益を一律に認定しない。depthではbasis-runがq平均残差とpath間変動を説明する有力な探索候補となった。3係数の一部相関/小q群数/feature分布の外挿により係数を物理的普遍定数と解釈しない。sqrt(q)は二laneの最大と揺らぎを動機とする仮説で、任意長qの極限定理ではない。全responseの結果を保存し、有利な資源だけの採択はしない。

追加の探索的観測として、全88本でΔsize−ΔRZ−ΔCX=192。旧保存q平均から示唆された関係の再解析で、事前独立仮説検定ではない。P0 basisのsizeはRZ/CX以外のSX/X/measure等も含み、その残余差192を観測したもの。全pathの恒等式とは認定しない。

## q64〜4096への探索的予測

以下はpaired Xのモデル値とsqrt版の名目的bootstrap SDのみ。実compile価格・保証された期待価格・予測区間ではない。

{table(['q','metric','frozen','all-q affine','sqrt','sqrt bootstrap SD'],fp)}

元凍結版は[元係数共分散付き]({link}frozen_C6a_future.json)、新各版はmodel_*.jsonへ別保存。round間の同じ係数を独立と扱わず、paired全7q×12量の交差共分散を保持した。bias/lawの未取得boundはnullであり0ではない。run/lane版の未来数値予測は、未使用qのjoint feature law・E[max]が不足するため生成していない。数値精度/P0/入力law/全support橋渡しも独立の資格で、RZ→T、depth→実機時間、価格×N/全13-round費用へ換算しない。

**長qへ登録可能な保証付き価格モデルは未取得。** 次の独立domainで探索版仕様と係数をsampling前固定して検証する。[実行条件案](NEXT_EXECUTION_DRAFT.md)。
''')
    numerical=read('numerical_residual_trends.json');saved=read('saved_numerical_recheck.json');art=read('artificial_stability.json')
    nr=[]
    for a in qs:
        rs=[r for r in numerical['rows'] if r['q']==a]
        obs=[r['max_frobenius'] for r in rs if r['label'] in ['compiled_vs_CAR_observable','compiled_vs_raw_observable']]
        raw=[r['max_entry'] for r in rs if r['method']=='raw']
        compiled=[r['max_entry'] for r in rs if r['method']!='raw']
        nr.append([int(a),max(raw,default=0),max(compiled,default=0),max(obs,default=0)])
    tr=[]
    for r in art['cases']:
        g=r['generic_vs_extended_shortblock_power'];n=r['native_vs_extended_shortblock_power']
        tr.append([r['width'],r['block_repetitions'],r['operations'],g['max_entry'],g['frobenius'],n['frobenius'],r['generic_vs_native']['max_entry']])
    sr=[[r['label'],r['max_entry'],r['frobenius'],r['status']] for r in saved['checks']]
    save('NUMERICAL_STABILITY.md',f'''# C7-4：数値安定性の技術検証

実行SOURCE `{source}`。C7-3 sampling/raw35bfdとcompile/post94e94を区別し、旧判定/精度は変更していない。[保存残差]({link}numerical_residual_trends.json)／[保存原本の限定再照合]({link}saved_numerical_recheck.json)／[人工比較]({link}artificial_stability.json)。全H4再シミュレーション/新H4 compile/旧run resumeは0。

## 保存q2〜32の残差

{table(['q','raw max entry','compiled max entry','observable max Frobenius'],nr)}

component1e−10、observable2e−10、rtol0、phase fitなし。full-wrapper全行列のFrobeniusはcomponent基準と別で、その値をobservable閾値で再判定しない。q32の最大observable残差1.197156474600893e−10は閾値の約60%。q64で同じ閾値に収まるとは言えない。q32は小演算子評価が公式Aer double/unitaryへ切り替わっており、qだけの因果効果と断定しない。

raw CAR積/AB/actual inverse/vacuum rayに比べ、compiled観測の残差が大きいことは、長いnative gate列の評価・compiler変換/位相処理・observable集約が追加寄与する可能性と整合する。各処理を独立に再propagateしていないため、最大誤差を特定のpass/cache/global phaseへ定量配賦できていない。CAR ordered積の演算順序、Gaussian double行列、宣言χ/fsum/MPFR160、unitary/statevectorの反復丸め、256列のGram/observable積を区別する。phase精度160bitは全回路の演算精度を160bitにするものではない。

## 最大残差を持つ保存1 pathの再照合

最大値（同値は辞書順）で選んだq32 `{saved['selected']['sample_id']}`、{saved['selected']['method']}/{saved['selected']['axis']}。原本hashを旧公開索引へ照合した上で、保存A/B/actual adjoint/Vと保存4 observableだけを読んだ。全256×256要素を保持し、extended精度の積・差・normで再照合した。行列や参照状態の再伝播、phase fit、renormalizationはない。

{table(['check','max entry','Frobenius','status'],sr)}

結果はPASSであり、旧残差は単なる保存normの丸めで消えるものではなかった。extended subtractionは以前のbinary64 simulationが生んだ誤差を回復しない。厳密区間oracleでもなく、元H ground vectorも使っていない。全88標本を再シミュレーションしていない。

## 人工経路比較と技術改善の成否

既存exact cache/実class/control state/binary64角度、actual inverse/global phase、Reference state-action batchの回帰を再利用した。新C7-4定義/native比較7件PASSと、旧cache6＋handoff5の不変回帰11件PASS、distinct18件。旧guard/17qubit全Operator禁止を維持。

固定3/5qubit blockの1024/8192/32768反復を、generic局所行列変換とnative rz/sx/x/cxを保持する別revisionで比較した。参照は短いQiskit/binary64 blockをlongdoubleで累乗したもので、短block誤差は反復される。厳密な高精度解/誤差上界とは扱わない。

{table(['width','repeats','operations','generic max','generic Frob','native Frob','generic-native max'],tr)}

6件ともcomponentと記録したfull-matrix Frobenius診断は現行許容値内。nativeとgenericは今回すべて同じ値で、**精度改善は確認できなかった**。native版を「改善済み長q verifier」として採択しない。cacheの共用は同じ公式局所行列を再利用し、この人工比較と既存回帰で適合したが、全support誤差boundではない。3/5qubit短blockのP0 compileは計2本だけで、raw/compiled全行列は許容値内。人工gate数が多いこと自体をH4/17qubit/256入力列の長q資格へ転用しない。

## q≥64前のpreflightと失敗時の対処

1. 新SOURCEでP0/入力law/ordered規約/actual inverse/χと固定精度を維持して人工回帰を通す。演算順序/phase扱い/extended集約を変える場合は旧結果を保持し別revisionへ固定する。
2. 次の承認済みdomainの先頭固定pathをcohortの一部としてraw→4 wrapper→postcheckし、CAR/raw/paired X/Y/全256列/reference leakageを照合。失敗時に除外/再抽選しない。
3. threshold近傍・超過なら依存する分子実行をSTOP。normの再計算だけでPASS化せず、誤差をcompiler位相・短gate列/小演算子・state-action/Gramへ分ける限定診断を次のSOURCEに設計する。必要ならより高精度のgate/Gram/ordered積経路を検証するが、真の相違をpolar補正/renormalization/phase fitで消さない。
4. 任意入力の丸め誤差包絡と、17qubit Reference経路の精度改善は未取得。q32→64の線形誤差スケールも証明されていないため、**現行閾値を長qで維持できるとの事前GOは出せない**。

本C7-4の数値比較は完了。q≥64の実H4計算と、旧失敗を上書きする再計算は実行していない。次の科学実行は[条件案](NEXT_EXECUTION_DRAFT.md)をユーザー/GPTが採択・承認してから行う。
''')
    print(json.dumps(dict(status='REPORTS_WRITTEN',models=5,source=source,rank=rank,compiled_DAG_records=len(dag))))


if __name__=='__main__':main()
