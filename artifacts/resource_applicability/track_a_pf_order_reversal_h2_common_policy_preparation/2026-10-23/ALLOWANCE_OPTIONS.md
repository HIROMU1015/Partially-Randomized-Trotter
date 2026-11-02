# 共通allowance政策：SOURCE定義と未採択の選択肢

本書は政策案である。採択・数値適用・shot再計算は行っていない。推奨案の数値パラメータも、ユーザーの採択と入力充足確認が済むまで実行条件にならない。
対象はH2直鎖1.00 Å/STO-3G/tol-only1e-8・DF3/T=.8/ε=.0005/P=0、Full PF prefix3のS2 q4/Yoshida4 q2、PR prefix1/K4/ABS_TIME_PER_OUTER32の同じ2公式・q。δはそれぞれ1/5と2/5、追加δ窓の評価はない。

## 1. A：旧Full PFの実際の定義

取得SOURCE `cf99e4452df96291d7e9d37d76d2e91be5cbc413` の `h2_panels_science.setup` は `h4_native_science.accounting_plan` を経由する。実provider `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` の `h48_preflight_contract_v1.h4_plan` → `h6_matched_contract_v1.plan` の numerical を継承し、H2ではsensitivity factorを1に固定する。

旧設定は `floor=1e-12, safety_factor=10, roundoff_multiplier=64`。`h6_matched_accounting_v1.empirical_allowance` の生成式は以下である。ここでeps_machはbinary64 machine epsilon、Wは両経路のaction count合計、Mは両経路の全記録stageで観測した最大normである。

```
rho = 64 * eps_mach * (1 + W) * max(1, M)
a   = e_reference + max(e_path) + e_raw_closure
u_real = u_imag = 10 * max(1e-12, rho, a)
margin_logB = 10 * max(1e-12, rho, e_logB)
```

`h6_matched_port_v1.correctness` の `e_path` はnative/Horner経路対独立forward経路のsignal差とstatevector差の両方。`e_raw_closure` は `abs(exp(logB)*raw_signal-corrected_signal)`、`e_logB` は両経路のlogB差である。Mは中間stageも確認し、Wはdeterministic actionとtail matvecの両経路合計を数える。

**H2のreference定義は汎用H6 setupをそのまま使わない。** `h2_panels_science.setup` のFP段階でsector行列対独立occupation行列のnorm差とexpm/eighのstate/signal差の最大を `reference.error` とし、Dではその保存値を継承する。汎用H6 setupの「列差＋signal差」と取り違えない。保存H2 referenceと2候補の診断入力は [saved_input_excerpts.json](saved_input_excerpts.json) に無計算で転記した。

この規則の技術目的は、数値経路・referenceの不一致とwork依存の丸めproxyに余裕を置くこと。SOURCE自体が `kind=EMPIRICAL, certified=false, local_probe_discrepancy_is_global_bound=false` と宣言する。安全係数10・floor・roundoff係数64の妥当性を厳密保証する証明は取得資料にない。経験的整合確認を厳密bias証明として採用した理由は存在しない。

## 2. B：新outer32 PRの生成規則と保存値

取得SOURCE `4ec26921c6872ed6e8667f167074e1e405cf1a3f` の `h2_outer32_runtime.signal_stage` は `h46_m5_signal_contract.eligibility`（132–148行）を呼ぶ。**保存u=1e-9、margin=1e-12は定数代入ではなく、下限付き生成規則の保存結果である。** 当該4候補の比較で新PRにこの保存値が使われた事実は変わらない。

```
s = q * (2*(prefix+1) + [PRなら1]) * [S2なら1、Y4なら3]
rho_B = 64 * 2**-52 * (1 + M_B) * (1 + s)
u_B = max(1e-9, 10*e_raw_vector_closure + rho_B)
margin_B = max(1e-12, 64*2**-52*(1+abs(logB))*(1+s))
```

`M_B` はspectral meanのpiece終了時に記録した最大norm。`e_raw_vector_closure` は `norm(raw_vector-exp(-logB)*corrected_vector)` で、Aの補正済みsignal単位のclosureとは正規化の向き・vector/signalの意味が異なる。sもAの実action count Wと同じ量ではない。floorに安全係数10を掛けるAと、closureに係数を掛けてfloorとmaxを取るBは、同じラベルのfloorだけを交換しても同一規則にならない。

同SOURCEのH2 signal段階は別に、exact-tail/corrected/rawの独立forward signal差とlogB差が `<1e-9` という技術gateを行う。しかしこの独立forward差は `eligibility` のu生成に入力されない。新referenceは同じ保存Hamiltonian/stateからfull-H expmで作り、旧FP保存scalarとの一致を `<1e-9` で確認する。`reference_absolute_error=null`、`CONDITIONAL_ON_SAVED_REFERENCE_NOT_RIGOROUS` が保存される。gateを通ったことからcombined candidate/reference uncertaintyが1e-9以下だとは導けない。

原signal束は今回未取得。closure/M/forward差の実値を、保存uやPASS labelから逆算・0補完しない。

会計SOURCE `9a77d2b0a0b87e246e85f085b37f5923c18e27bb` の `shot_record` は保存u/marginをfactor1で読み、元のshot kernelへ渡す。保存Eの `policy_adoption.json` は `H246_HIGHER_PF_OUTER32_EXPLORATION` 限定の経験的・reference条件付き政策を採択済みと記録する。前段 `policy_adoption_required.json` も厳密上界を主張せず、旧契約を変更しない。**この既存seriesの採択は、今回のA/B共通政策の採択ではない。** 1e-9というfloorを全方式・全referenceの誤差を覆う量として導いた証明や校正は未取得であり、数値gateと同値だとは説明しない。

## 3. 含める誤差と別会計

| 成分 | A旧Full PF | B新outer32 PR | 共通化の条件 |
|---|---|---|---|
| reference経路差 | 保存H2 reference.errorをuへ含む | reference精度に条件付き。絶対誤差はnull | 同じreference基準と差の定義・採用範囲を明記。nullを0としない |
| 独立candidate経路差 | signal/state差のmaxをuへ含む | 別gate、uには未入力 | 同じ意味の診断入力を集める。signal差とstate差を混同しない |
| raw/corrected closure | 補正済signal単位のclosure | raw側vector単位のclosure | B因子とnormの向きを揃える。ラベルだけで代入しない |
| 丸めproxy | 両経路workと全観測stage最大norm | 公式由来stage proxyとpiece終端norm | work/stage/maxnormの測定範囲を仕様化。未観測値を推測しない |
| logB誤差 | 独立logB差＋work proxyのmax | logB・stage依存proxy、floor | normalization marginを別フィールドに維持 |
| PF/finite-RTE近似bias | PF signed biasの各軸絶対値 | outer PF＋finite RTEのsigned totalの各軸絶対値 | uへ再加算しない。相殺を禁止する政策は別の科学判断 |
| native/compiled vector gate | native controlsとcompiler fingerprintを保存。compiled vector gateは取得旧経路に見当たらない | 各taskでnative/occupation/wrapper/compiled gate | gate許容差をshot allowanceへ流用しない |
| MC回路単価の不確実性 | 決定論的2軸の実測単価 | exploration n33/n34のpaired標本 | MC SEを信号uや量子shot失敗確率へ混入しない |

共通の評価目的は固定時間Tの複素signal。軸精度ε/√2、各軸α=.025（両軸の和=.05）、通常signed bias、P=0を維持する。これは1候補の信号精度会計であり、4候補の選択後家族誤差や母平均順位保証を自動的に与えない。

設計上の会計式は既存SOURCEの `h_axis=ε/√2-|bias_axis|-u_axis`、`L=logB+margin`、`N_axis=max(1,ceil(2*exp(2L)/h_axis²*log(2/alpha_axis)))`、`G=N_cos*C_cos+N_sin*C_sin`。本書では式を記すだけで評価していない。Full PFのlogB=0とPRの非零logBは方式固有の違いとして残す。負時間はnormalizationを相殺しない。数値marginはuへ重複加算せず、biasもsigned totalとPF/RTE成分を同時に重複控除しない。

## 4. 3案の比較

| 観点 | 案1 共通適応的規則（推奨・未採択） | 案2 共通固定政策 | 案3 方式固有allowance＋共通原則 |
|---|---|---|---|
| 科学的意味 | 同じ誤差成分・同じ生成式を方式別の測定値に適用。uの値自体は異なってよい | 全4候補に同じ数値u/marginを使う条件付き評価 | 各数値経路に必要な生成器を許し、保証対象・校正基準を共通化 |
| 公平性の条件 | reference、signal単位、path/norm/work範囲、floor/係数、欠測処理を事前固定 | 固定値が全4候補の同じ誤差範囲を覆う根拠、又は明示した仮定の下だけの比較 | 同じ被覆対象、証拠水準、採否条件・欠測基準。片側だけreferenceを除外しない |
| SOURCE整合 | Aの既存empirical_allowanceを出発点にできる。Bの入力adapterは現状未実装 | 凍結shot kernelは再利用可能。両SOURCEのallowance生成を置き換える別政策になる | 既存A/B生成器を保持しやすいが、現状の成分差をそのまま公正と呼べない |
| 必要変更 | 共通入力schema、単位・work観測仕様、B診断adapter、独立した再会計計画。元SOURCEは不変 | uとmarginを選ぶ独立根拠、採用文書、新しい4候補再会計 | 成分対応表、方式別校正/受入基準、reference差の扱いを補う |
| 保存入力の利用 | Aのreference/path/vector/counts/allowanceは保存済み。Bのprivate signalが取得できれば一部を充足 | 保存bias/logB/Cから仮定下の再会計が可能。全誤差を覆う証拠は別に必要 | A/B保存診断を可能な限り再利用。欠測を既存uの大小だけで正当化しない |
| 新規計算の可能性 | Bの未保存vector比較・全stage norm/work等だけ。まず原保存資料を確認 | 被覆根拠を新しく求めるなら診断が必要。固定値の感度探索は今回の最小案に含めない | 方式別診断が不足する部分。厳密保証を求めるなら新科学設計が必要 |
| 利点 | どの成分の差がu差になるか追跡でき、将来のCを説明しやすい | 再会計は軽い。政策依存の比較を明快に定義できる | 異なる数値手法の必要な余裕を認め、実装変更が少ない可能性 |
| 問題点 | 同じ関数を使うだけでは保証にならない。B入力欠測により現時点では実行不可 | 1e-9/1e-12の横展開を正当化する共通根拠は未取得。恣意的な値選択になり得る | 証拠水準が不均衡だと政策差が残り、順位の解釈が難しい |

**推奨は案1を次の採択候補とすること。** Aの定義とBの未充足成分が明確で、誤差成分の対応を追跡できるためである。初期候補はA由来の経験的式・floor1e-12/係数10/roundoff64であり、共通schemaでの被覆を確認した上でユーザーが決定する。これらの値を厳密上界と認定する提案ではない。Bのraw vector closureをそのままAのsignal closureへ入れない。

独立誤差診断に不足があれば、案1を「実行可能」と偽らず保留する。案2を利用するなら「同じ仮定値を置いた条件付き固定点比較」としてユーザーが明示的に選ぶ必要がある。案3は各方式の成分を共通の被覆基準へ対応付けられた場合の代替案である。現行政策の単純併用で問題が解消したとはしない。

いずれもGの大小・逆転の有無を見る前に定義・採否条件を固定する。逆転が残るようにuを選ぶ、複数案を評価して好ましい順位の案を採用する、失敗後にfloorや許容差を変える、という手順は含まない。今回、案ごとの新N/G・感度・順位はすべて未評価である。

定義の固定commit・実取得copy・SHAは [SOURCE_INDEX.md](SOURCE_INDEX.md)。元政策・原本には変更を加えていない。
