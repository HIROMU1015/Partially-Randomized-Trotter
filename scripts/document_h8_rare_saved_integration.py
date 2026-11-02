#!/usr/bin/env python3
"""Post-run rendering/structural audit only; NEVER calls scientific arithmetic.

Decimal is used solely to round stored display strings. No new numerical results.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('artifacts/resource_applicability/h8_rare_rte_saved_integration/2026-10-21')
REPORT=Path('docs/research/h8_rare_rte_saved_integration_results_20261021.md')
BASE='f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
         'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def read(path):return json.loads((ROOT/path).read_text())
def write(path,value):(ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def fmt(x):return format(Decimal(x),'.10E')
def fixed(x,n=3):return format(Decimal(x),f'.{n}f')


def render():
    result=read(OUT/'result.json');rs=result['conditions'];source=result['SOURCE']
    lines=['# H8希少RTE：保存値統合・残余費用の感度解析結果','',
      '2026-10-21。固定H8の2条件について、段階Bのcanonical層確率・S0独立確認各48本と、C-1のS2_SINGLE条件付きpilot各43本を、層別寄与として統合した。**承認された保存値解析1回を完了し、追加sampling/compileなしでmandatory STOP**。', '',
      '新S2_SINGLEの平均はS0平均に近く、既知層を経験平均へ固定した残余費用感度では、PR低費用となる領域が大きく残ることが分かった。一方、S2_MULTI/S4_PRESENTは支持内で平均・分散とも未知であり、全層の母平均順位は保証していない。有限標本の点比較と条件付き感度を論文の主張範囲とするなら、追加希少eventの実測は必須ではなく、現状で停止できる。','',
      '## 固定資料・会計・実行範囲','',
      '- 段階B：`3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb`。[B結果](pr_rare_rte_stage_b_results_20261021.md)。',
      '- C-1結果：`f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c`、C-1 SOURCE：`383dd9068504b8f578944984fc45a60dd4ed18b2`。[pilot結果](h8_rare_rte_c1_pilot_repair_20261021.md)。',
      f'- 今回の科学SOURCE：`{source}`。実行前契約commit：`64b684f2`。[固定契約](h8_rare_rte_saved_integration_contract_20261021.md)、[全数値・来歴・監査](../../{OUT.as_posix()}/README.md)。','',
      'Linear H8、1 Å、STO-3G、tol-only 1e-8 DF28、T=.8、P=0、二次partial PF、通常signed bias、allowance1。splitはp8/p7、q=2/16、r=4、K=4、時間幅δ=T/qを固定。元のHamiltonian/state/preparation/analytic shots/native DF/scalar/control/full measured Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17の保存compiled RZを使用した。今回は分子・compiler runtimeをimportせず、標準ライブラリの整数・有理数演算だけを実施した。','',
      '|ε|PR|cos/sin shots|固定PF|G_PF|', '|---|---|---|---|---:|']
    for r in rs:
        lines.append(f"|{r['condition']}|{r['candidate']}|{r['shots']['cosine']} / {r['shots']['sine']}|{r['fixed_PF']}|{r['fixed_PF_G_exact']}|")
    lines+=['', 'Gは同じtrajectoryのpaired費用 `Ncos*RZcos+Nsin*RZsin`。PFは段階Bの固定値であり、後続qモデルや別4次PF研究の新しい最良候補へ置換していない。各条件のS0とsingleを48+43本の無条件cohortへpoolせず、別々の条件付き平均へ既存の層確率を掛けた。元Pで48本すべてS0だった事実は、希少層の確率0を意味しない。H8粗精度Pのうち元から再利用された1 trajectoryは同一P cohortの来歴を維持した。','',
      '## 層確率・条件付き費用・期待費用への寄与','',
      '以下は今回保存結果の表示丸め。確率は段階Bの100桁canonical文字列をそのまま入力し、新しいRTE法則・層確率を計算/正規化していない。','',
      '|条件|層|p_s|n|条件付きG平均|標本SE|p_s*平均|p_s*SE|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rs:
        for s in r['known_layers']:
            lines.append('|'+ '|'.join([r['condition'],s['stratum'],fmt(s['probability']),str(s['n']),
                          fmt(s['mean_decimal']),fmt(s['SE']),fmt(s['contribution_G']),fmt(s['contribution_SE_G'])])+'|')
        for layer in ('S2_MULTI','S4_PRESENT'):
            lines.append(f"|{r['condition']}|{layer}|{fmt(r['probabilities'][layer])}|0|未知|未知|未知|未知|")
    lines+=['', 'S0平均・分散は段階B、single平均・分散はC-1とexact一致。各層のGはRZ×固定shotで、軸を独立な標本として二重に数えない。未知層の平均・分散・寄与はnullのままで、標本最大値も全support費用上界へ格上げしない。','',
      '## 既知層の推定誤差','',
      '`K_known_hat=p_S0*mean_hat_S0+p_SINGLE*mean_hat_SINGLE`。真のcanonical期待費用は','',
      '`E[G]=K_known_hat+p_MULTI*mu_MULTI+p_S4*mu_S4+delta_K`','',
      'である。`delta_K=p_S0*(mu_S0_canonical-mean_hat_S0)+p_SINGLE*(mu_SINGLE_canonical-mean_hat_SINGLE)`は標本誤差とsampling lawのtarget差を含み得る。後者の費用bias上界はない。','',
      '層内iid・有限分散、元Pの観測S0本数48へ条件付けた抽出と新single pilotの独立性をモデル仮定とすると、`Varhat(K_known)=p_S0²*s_S0²/48+p_SINGLE²*s_SINGLE²/43`。仮定を外すと`2*p_S0*p_SINGLE*Cov(mean_hat_S0,mean_hat_SINGLE)`が必要で、その共分散は未評価。seed非衝突は独立性の数学的証明ではない。実費用分布の正規性も確認・仮定していない。','',
      '|条件|K_known_hat|SE_known|固定PFとの差D|D/SE_known|K_known_hat/G_PF|', '|---|---:|---:|---:|---:|---:|']
    for r in rs:
        lines.append('|'+ '|'.join([r['condition'],fmt(r['K_known_G']),fmt(r['SE_K_known_G']),fmt(r['remaining_budget_D_G']),
                                 fixed(r['headroom_in_known_SE']),fixed(r['K_known_over_PF'],6)])+'|')
    lines+=['','D/SEは既知層の推定誤差尺度との比較であり、p値・母平均優位の信頼水準ではない。全層のSE、未知層の分散、law差の期待費用biasは含まれない。重み付きsingleのSEは粗精度約2.69e4、細精度約1.30e9で、既知層SEはS0側が支配的。singleの取得を増やすことが現在の主要な既知層誤差を直接解消するとはいえない。','',
      '## S2_SINGLE取得によって解消した部分','',
      '|条件|single平均 / S0平均|従来未知だった質量のうち今回推定した割合|未知質量の減少factor|残余質量|旧混合境界 / S0|新残余混合境界 / S0|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rs:
        b=r['stage_B_comparison']
        lines.append('|'+ '|'.join([r['condition'],fixed(b['single_mean_over_S0'],6),fixed(b['fraction_of_previous_unknown_mass_measured'],8),
                                   fixed(b['unknown_mass_reduction_factor']),fmt(r['residual_mass']),fixed(b['previous_threshold_over_S0']),
                                   fixed(r['residual_mixture_threshold_over_S0'])])+'|')
    lines+=['', '取得されたsingle平均はS0の約1.0293倍/1.0022倍。従来の未知質量の約99.67%/99.56%へ費用推定を付与し、未知質量は約304分の1/229分の1となった。single寄与は、段階B時点のPFまでの余裕の約19.64%/20.46%を使うため、未知費用を0とした楽観値を維持したわけではない。','',
      '旧境界はsingleも含む3層混合、新境界はmulti/S4だけの2層混合で、同じ母数の精度改善率ではない。未観測だった分散の上限がなかったため、全期待費用の不確かさや分散が「何%減った」とは計算できない。減ったのは未知項の数と、その確率係数である。','',
      '## 残余2層を区別した逆転境界','',
      '`D=G_PF-K_known_hat` とすると、経験的な既知層平均を固定した境界は','',
      '`p_MULTI*mu_MULTI+p_S4*mu_S4=D`。','',
      '非負費用領域で残余加重和がD未満/等号/D超なら、条件付きPR低費用/同費用/PF低費用。これは`delta_K=0`へ固定した感度であり、真の母平均の判定ではない。`mu_R=(p_MULTI*mu_MULTI+p_S4*mu_S4)/(p_MULTI+p_S4)`は残余混合平均の定義で、2層同費用の仮定ではない。','',
      '|条件|mu_R境界G|mu_MULTI軸切片G|mu_S4軸切片G|multi切片 / S0|S4切片 / S0|p_MULTI / p_S4|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rs:
        b=r['two_layer_boundary']
        lines.append('|'+ '|'.join([r['condition'],fmt(r['residual_mixture_mean_threshold_G']),fmt(b['multi_intercept_G']),
                                   fmt(b['S4_intercept_G']),fixed(b['multi_intercept_over_S0']),fixed(b['S4_intercept_over_S0']),
                                   fixed(b['p_MULTI_over_p_S4'])])+'|')
    lines+=['', '各軸切片は他方の座標を仮に0へ置いた幾何学的交点である。未知の実費用を0補完した結果ではなく、他方の寄与が正なら許容される一方の費用はこの切片より小さい。残余混合平均がS0の約1,276倍/888倍へ達すると固定平均下の方向は逆転し得るが、現在の証拠はそのような値を数学的に排除していない。','',
      '|条件|残余budgetのmulti配分t|mu_MULTI / S0|mu_S4 / S0|', '|---|---:|---:|---:|']
    for r in rs:
        for p in r['two_layer_boundary']['points']:
            lines.append('|'+ '|'.join([r['condition'],p['multi_budget_share'],fixed(p['mu_MULTI_over_S0']),fixed(p['mu_S4_over_S0'])])+'|')
    lines+=['', 'これら5座標は同じ境界上の仮想配分。multiの確率係数はS4より約20.9倍/187.9倍大きい。同じ平均費用ならmultiが寄与しやすいが、平均費用が同程度という仮定・観測はないので、S4が支配しないとは保証しない。','',
      '## 既知層誤差を動かした境界感度','',
      '`K_known_hat±k*SE_known` を置いたときの残余budget範囲。下端は既知層費用が高い側。kは表示座標で、信頼区間・90%/95%/99.7% coverageを表さない。','',
      '|条件|k|残余budget Dの下端|上端|残余混合平均境界の下端G|上端G|', '|---|---:|---:|---:|---:|---:|']
    for r in rs:
        for s in r['known_error_sensitivity']:
            lines.append('|'+ '|'.join([r['condition'],str(s['k']),fmt(s['residual_budget_low']),fmt(s['residual_budget_high']),
                                      fmt(s['mixture_threshold_low']),fmt(s['mixture_threshold_high'])])+'|')
    lines+=['', '両条件とも+3SEの既知層費用でも残余budgetが正である。ただし、これを使って未知層費用やlaw差を消したり、母平均の優位を保証したりはできない。各層別の軸切片の移動もCSV/JSONへ保存した。','',
      '## sampling lawとcanonical targetの差','',
      'C-1は`EXACT_INTEGER_CONDITIONING_OF_STORED_BINARY64_V1_PROPOSED`を使用。保存binary64重みの厳密整数比、条件付き位置/count、元component正規化・元event構成を保持した。一方、段階Bの層確率は保存binary64 τで評価したcanonical有限lawの高精度値である。C-1実load時のq/r/K/T/τ/normalization・元law hash一致と、正規化後次数確率との差1e-12以内という既存receiptを今回転記した。実H8のlawを再形成・再照合していない。','',
      'このPASSはengineering整合確認で、旧NumPy choiceとのbit列同一性、conditional component lawの厳密同一性、compiled費用で重み付けたbiasの上界を保証しない。確率の小差へ未知の大費用が掛かる影響を無条件に無視できず、今回の数値は既存整合証拠に条件付けた統合である。全母平均保証を求めるなら残余費用だけでなく、この法則差と既知層の標本誤差の扱いも必要となる。','',
      '## 追加compileの要否・最小候補（未実行）','',
      '|求める主張|現在評価できること|不足と追加作業の必要性|', '|---|---|---|',
      '|有限標本の観測・層別条件付き感度|S0 n48とsingle n43の費用、既知寄与/SE、残余2層による逆転線|**追加compileは必須ではない。両条件とも現状で停止できる。** 未観測平均・法則差・iid仮定を明記する。|',
      '|特定の残余費用範囲でPRが低費用という条件付き主張|上記境界の内側という必要十分な会計条件、±kSEでの境界移動|範囲を仮説・条件と明記すれば新compile不要。範囲が実際に成立するという実証を求める場合は、その根拠が不足。|',
      '|無条件期待費用のPR優位・母平均順位の保証|現時点では不可|multi/S4の費用上界または妥当なtail推論、既知層誤差、sampling law差が不足。有限pilotの追加だけでは厳密保証にならない。新しい保証の採択は今回行わない。|','',
      '|条件|残る未知|追加実測を選ぶ場合の最小対象・優先判断|', '|---|---|---|',
      '|p8/q2|multi/S4の平均・分散・全support上界、S0推定誤差、法則差|同じ候補のS2_MULTIを最初のpilot候補とし、S4を別層として残す。S4が不要との保証はない。S0誤差も相対的に大きいため、single増量だけへ進まない。|',
      '|p7/q16|同上。残余混合境界/S0が粗精度より低い|同じ候補のS2_MULTIが、確率係数と相対境界の観点では優先候補。ただし実行負荷はC-1で粗精度より大きく、必須順位・標本数は未決定。S4を最後まで未評価にする保証はない。|','',
      '今後はまずユーザーが論文の主張を有限標本・条件付き結果に限定するか判断する。限定するなら追加の科学計算は不要。強める場合は、同一candidate/cohort/law/compilerでのmultiまたはS4費用を、境界までの余裕に対して評価する契約を別途設計する。旧探索・legacy・別q・別pのpilotを混合しない。未観測費用が観測最大以下という仮定、pretranspile countがcompiled RZ上界という仮定を採用しない。追加実験のpilot本数/production本数は未設計で、追加実験用grantの発行・compile実行は0。','',
      '## 数値照合・保全・終了監査','',
      '科学callはcampaign1、primary2条件、別算術2条件、表出力1を各呼出前に非返還予約し、予定=予約=実使用で完了。旧C-1/段階Bのgrantを使わず、新SOURCE・契約・現在ユーザー承認に結合した新grant1、科学load前のO_EXCL marker1を消費した。retry/resumeなし。実行前の封印技術処理で、取得metadata追記後の全worktree clean要求が1回停止したが、科学call/grant0の段階で原因を記録し、使用SOURCE各byteのcommit/SHA照合へ修正した。旧SOURCEを保全し、科学実行は修正SOURCEで1回だけである。','',
      '人工15件PASS。主経路のcentered Fraction統計と別CSVのraw整数moments/Decimal140・integer sqrtを同じone-shot内で照合し、両条件で相対/絶対1e-85 PASS。これは担当Codex自身の別算術で、別reviewerや数学的certificateではない。','']
    native=read(OUT/'native_exit.json');audit=read(OUT/'completion_audit.json')
    lines+=[f"数値childはnative exit0、親がwait4回収、残owned worker0。wall {native['elapsed_seconds']}秒、CPU {native['CPU_seconds']}秒、peak RSS {native['peak_RSS_bytes']} bytes。2条件の小規模な整数演算を1 processにまとめ、恣意的なCPU/RAM/worker capを設けていない。実空きCPU/メモリを記録し、共有環境や他worktree/他processの設定を変更していない。",'',
      f"固定Git blob64件と旧SOURCEを取得・照合。旧raw/結果/STOP/grant/ledger/markerを含む保全対象{audit['protected_file_SHA_matches']:,}件と、使用SOURCE/input/runtimeを前後SHA照合した。SOURCE全体や他worktree HEADを一括凍結せず、使用範囲を限定した。新結果のSHA目録・CSV/JSON対応・文書リンク/限定private-pattern検査はpacketの終了記録を参照。内部locator/seed/private rawを新公開候補へ転記していない。",'',
      '**mandatory STOP**。新sampling/simulation/signal/回路/compile・候補再選択・production・詳細GPT/Claude科学review・final freeze・pushは0。次担当は**ユーザー**：上記主張範囲と公開範囲を判断する。**Codex**：追加実験は提案に留め、停止を維持する。','']
    (ROOT/REPORT).write_text('\n'.join(lines))
    packet=['# H8保存値統合：成果物入口','',f'科学SOURCE `{source}`。2条件の固定保存値解析を1回完了、mandatory STOP。','',
      f'[結果・主張境界](../../../../{REPORT.as_posix()})、[契約](../../../../docs/research/h8_rare_rte_saved_integration_contract_20261021.md)。','',
      '|内容|正本|','|---|---|',
      '|固定計算・数値規約|[analysis_definition.json](analysis_definition.json)、[contract.json](contract.json)|',
      '|SOURCE・入力SHA|[SOURCE_manifest.json](SOURCE_manifest.json)、[input_acquisition.json](input_acquisition.json)|',
      '|現在の承認・新grant|[execution_approval_receipt.json](execution_approval_receipt.json)、[grant_issue_receipt.json](grant_issue_receipt.json)、[grant_consumption_receipt.json](grant_consumption_receipt.json)|',
      '|人工・実行前gate|[synthetic_validation.json](synthetic_validation.json)、[preflight.json](preflight.json)、[technical_preflight_log.json](technical_preflight_log.json)|',
      '|全数値|[result.json](result.json)|',
      '|既知層4行・条件概要2行|[known_layers.csv](known_layers.csv)、[condition_summary.csv](condition_summary.csv)|',
      '|2層境界10行・SE感度8行|[two_layer_boundary.csv](two_layer_boundary.csv)、[known_error_sensitivity.csv](known_error_sensitivity.csv)|',
      '|別算術検証・非返還会計|[independent_numeric_audit.json](independent_numeric_audit.json)、[calls.json](calls.json)|',
      '|native/保全/構造/最終SHA|[native_exit.json](native_exit.json)、[completion_audit.json](completion_audit.json)、[documentation_audit.json](documentation_audit.json)、[SHA256_manifest.json](SHA256_manifest.json)|',
      '|状態・停止|[status_manifest.json](status_manifest.json)、[STOP.json](STOP.json)|','',
      '未知費用/分散はnull。軸切片の0は仮想座標で、unknown zero-fillではない。known SEはiid・独立性モデル下の標本推定で、全層保証ではない。法則差1e-12 gateを期待費用biasの保証へ格上げしない。有限標本・条件付き感度の報告に追加compileは必須でなく、母平均順位は未保証。','',
      '非公開grant原本・marker・予約ledger・入力cache・保全locatorは新worktree内のignored private directoryへ保存し、公開候補はSHA/receiptのみ。次担当ユーザー：主張/公開範囲判断。Codexは停止。','']
    (ROOT/OUT/'README.md').write_text('\n'.join(packet))
    note=Path('docs/research/研究ノート/2026-10-21_h8_rare_rte_saved_integration.md')
    (ROOT/note).write_text('# H8保存値統合：2026-10-21\n\n[結果と限界](../h8_rare_rte_saved_integration_results_20261021.md)。\n\n'
      'ユーザーがH8 2条件の保存値解析1回を明示承認。S0 n48/single n43をpoolせずcanonical層別寄与へ結合。未観測質量の約99.67%/99.56%へ費用推定を付与し、残余混合境界/S0は約1276/888。未知2層・law差・母平均順位は未保証。有限標本/条件付き感度を主張範囲とするなら追加compileは必須でなく、現状停止可能。\n\n'
      '封印metadataの自己変更gateを科学0/grant0で修正して新SOURCEを保全。科学は修正SOURCEで1回・別算術PASS・native exit0/reaped。追加science/production/review/push0、mandatory STOP。次担当ユーザー：主張と公開範囲判断。Codex停止。\n')
    summary=('H8のS0各48とS2_SINGLE各43をpoolせず、B canonical確率の層別寄与へ統合。'
      '未知質量の約99.67%/99.56%に費用推定を付与、残余混合境界/S0約1276/888。'
      '未知multi/S4・law差・母平均順位は未保証。有限標本/条件付き感度なら追加compileは必須でなく現状停止可能。'
      '人工15・別CSV数値2条件・SHA/予算/native exit0/reap PASS。'
      '保存値one-shot1、追加sampling/compile/production/review/push0、mandatory STOP。'
      '次担当ユーザー：主張/公開範囲判断。Codex停止。以下は旧bytesを保全。')
    import os
    for path in INDEXES:
        p=Path(path);old=git('show',BASE+':'+path)
        if (ROOT/p).read_bytes()!=old:raise ValueError('INDEX_PREPEND_ONLY')
        rlink=os.path.relpath(REPORT,p.parent);alink=os.path.relpath(OUT/'README.md',p.parent)
        text=f'## 2026-10-21 H8希少RTE：保存値統合・残余2層感度\n\n[結果・主張境界]({rlink})、[数値・SOURCE・監査]({alink})。\n\n{summary}\n\n'
        (ROOT/p).write_bytes(text.encode()+old)
    write(OUT/'status_manifest.json',dict(status='COMPLETE_SAVED_STRATIFIED_INTEGRATION_MANDATORY_STOP',
        SOURCE=source,conditions=['H8_0.05','H8_0.0005'],S0_n_each=48,SINGLE_n_each=43,
        scientific_saved_analysis_runs=1,new_sampling_compile=0,cohorts_pooled=False,
        residual_layers_unknown=['S2_MULTI','S4_PRESENT'],full_population_ranking_guaranteed=False,
        finite_sample_conditional_claim_requires_more_compile=False,push_performed=False,next_science_authorized=False))


def audit(record):
    if record:
        write(OUT/'documentation_audit.json',dict(status='PENDING_STRUCTURE_CHECK'))
        write(OUT/'SHA256_manifest.json',dict(status='PENDING_MANIFEST_CREATION'))
    rs=read(OUT/'result.json')['conditions']
    ledger=read(Path('.rare_integration_private/budget/ledger.json'))
    call=read(OUT/'calls.json');assert ledger['limits']==ledger['used']==call['planned']==call['reserved']==call['actual_entered']
    assert call['all_entered_calls_completed'] and ledger['nonrefundable']
    private=ROOT/'.rare_integration_private'
    grant=json.loads((private/'grant.json').read_text());marker=json.loads((private/'ONE_SHOT.json').read_text())
    sealed=json.loads((private/'sealed.json').read_text());contract=read(OUT/'contract.json');approval=read(OUT/'execution_approval_receipt.json')
    assert sha((ROOT/OUT/'contract.json').read_bytes())==grant['contract_sha256']==sealed['contract_sha256']
    assert sha((ROOT/OUT/'execution_approval_receipt.json').read_bytes())==grant['approval_sha256']==sealed['approval_sha256']
    assert marker['grant_sha256']==sha((private/'grant.json').read_bytes())
    assert grant['SOURCE']==contract['SOURCE']==approval['SOURCE']==sealed['SOURCE']
    reserved={k:0 for k in ledger['limits']}
    for file in (private/'budget').glob('*.reserved.json'):
        row=json.loads(file.read_text());assert row['nonrefundable'];reserved[row['kind']]+=row['count']
    assert reserved==ledger['used']
    csv_expected={
      'known_layers.csv':[dict(condition=r['condition'],**s) for r in rs for s in r['known_layers']],
      'two_layer_boundary.csv':[dict(condition=r['condition'],**s) for r in rs for s in r['two_layer_boundary']['points']],
      'known_error_sensitivity.csv':[dict(condition=r['condition'],**s) for r in rs for s in r['known_error_sensitivity']],
      'condition_summary.csv':[{k:r[k] for k in ('condition','candidate','K_known_G','SE_K_known_G','remaining_budget_D_G',
                  'residual_mass','residual_mixture_mean_threshold_G','residual_mixture_threshold_over_S0','headroom_in_known_SE')} for r in rs]}
    for name,expected in csv_expected.items():
        with (ROOT/OUT/name).open() as f:actual=list(csv.DictReader(f))
        assert actual==[{k:'' if v is None else str(v) for k,v in r.items()} for r in expected]
    for r in rs:
        assert all(x is None for x in r['residual_means'].values())
        assert all(x is None for x in r['residual_variances'].values())
        assert r['full_population_mean_G'] is None and r['law_transfer_bias_bound'] is None
    for path in INDEXES:assert (ROOT/path).read_bytes().endswith(git('show',BASE+':'+path))
    changed=[p for p in git('diff','--name-only','-z',BASE).decode().split('\0') if p]
    untracked=[p for p in git('ls-files','--others','--exclude-standard','-z').decode().split('\0') if p]
    paths=sorted(set(changed+untracked)-{(OUT/'SHA256_manifest.json').as_posix()})
    for path in paths:
        assert '.rare_integration_private' not in path
        b=(ROOT/path).read_bytes()
        if path in INDEXES:
            old=git('show',BASE+':'+path)
            assert b.endswith(old)
            b=b[:-len(old)]  # Privacy scope is newly authored bytes, not old history.
        assert not re.search(rb'/(?:home|tmp|root)/|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}',b),path
    links=0
    for path in [REPORT,OUT/'README.md',Path('docs/research/h8_rare_rte_saved_integration_contract_20261021.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',(ROOT/path).read_text()):
            assert (ROOT/path.parent/target).resolve().exists(),(str(path),target)
            links+=1
    report=dict(status='PASS_DOCUMENT_STRUCTURE_SHA_ACCOUNTING_PRIVACY_SCOPE',
                numerical_analysis_reexecuted=False,separate_numeric_receipt_reused=True,
                CSV_JSON_correspondence=True,old_index_bytes_retained=True,
                private_pattern_scope='changed/new published candidates only; not a security certificate',
                local_links_checked=links,grant_approval_marker_SOURCE_binding=True,
                nonrefundable_reservation_ledger_matches=True,
                scientific_source_and_old_evidence_verified_by='h8_rare_saved_run.py audit')
    if record:write(OUT/'documentation_audit.json',report)
    if record:
        if (OUT/'documentation_audit.json').as_posix() not in paths:paths.append((OUT/'documentation_audit.json').as_posix())
        write(OUT/'SHA256_manifest.json',dict(baseline=BASE,self_excluded=True,
             files=[dict(path=p,sha256=sha((ROOT/p).read_bytes()),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    else:
        for r in read(OUT/'SHA256_manifest.json')['files']:
            data=(ROOT/r['path']).read_bytes();assert sha(data)==r['sha256'] and len(data)==r['bytes']
            assert git('show','HEAD:'+r['path'])==data
        assert not git('status','--porcelain')
    print('PASS_POSTRUN_STRUCTURE_NO_SCIENTIFIC_REEXECUTION',len(paths),links)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['render','record-audit','verify-committed'])
    action=parser.parse_args().action
    if action=='render':render()
    else:audit(action=='record-audit')
