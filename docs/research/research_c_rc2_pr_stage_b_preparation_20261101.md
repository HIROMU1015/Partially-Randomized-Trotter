# 研究C：保存RC2／PR-tailのStage B最小検証準備

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：f6bb336b41cddface5647c6b1acf23c330f32637。

**直近の共通収束実験で保存した2候補を、同じM4政策のPF／補正RTE／attenuationへ接続する専用entryを実装した。人工・回帰88件PASS、SOURCE74／保存入力13件のpassive preflight PASS。実H4の新規科学call・配列decode・fit・ground／PF／RTE・回路build／compile・shotsは0。未承認143-call契約を固定し、実行直前で停止する。**

[未承認科学契約](research_c_rc2_pr_stage_b_contract_unapproved_20261101.json)／[SOURCE](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/SOURCE.json)／[人工log](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/tests.txt)／[passive preflight](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/passive_preflight.json)。実装commitは50a490b20a12a5387a84b0bccc42b7e2ca45d70b。最終公開commitは本書・契約・技術証拠を含むbranchの先端とする。

共通運用指示2026-10-25版はworkspace原本を実際に確認し、§8.1/8.2/8.4/10/10.1/10.3を適用した。チャット上の許可を証明するファイル、承認原文、grantは作成しない。本書はCodexの技術準備報告で、重要なGPT科学レビューではない。

## 1. 比較対象と資格

対象は保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number CDF rank8、一般実対称J／実直交U、元frame順prefix LD2。同じ元x0、初期penalty寄与を合わせたα=γ=.1、同じL-BFGS-B options／関数評価上限の2終点である。異なる目的関数の有限予算終点を比較し、共通収束達成後の最良点比較とはしない。

| 保存済みの結果からの引用 | RC2 | PR-tail |
|---|---:|---:|
| Exact tail λ [Ha] | 8.545687128230103 | .9545432846581517 |
| DF予算 | PASS | PASS |
| Solver success | false | false |
| 共通一次収束資格 | 未達 | 未達 |
| Optimizer停止 | nfev上限 | nfev上限 |

正式なJSON入力は[修復後の有効result](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/result.json)。[修復報告](research_c_common_convergence_export_repair_20261101.md)、[元の科学実行](research_c_common_convergence_execution_20261101.md)、[共通政策](research_c_common_convergence_policy_design_20261101.md)を引き継ぐ。旧partial result.jsonは不正JSONのまま保全し、読み込まない。旧failed STOPも成功へ変更しない。

DF物理予算は保存sector norm＋既存engineering guardがCA/8内という数値pilotの資格。Optimizer収束とは独立である。有限J/Uが定義するHamiltonianの下流挙動は評価できるが、十分に最適化されたRC-DF一般、最適強度、大域最適性、CA達成、総資源優位、新規性は認定しない。

## 2. 固定identity

| 固定物 | SHA256／SOURCE |
|---|---|
| 元2-fit科学SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| 元2-fit実行・raw公開commit | 5ff11540262dfe74446be9ee089b4b0a8e136c67 |
| JSON復旧公開commit／今回起点 | f6bb336b41cddface5647c6b1acf23c330f32637 |
| RC2 final.npz | d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798 |
| PR final.npz | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 共通h/g/constant snapshot | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 元P-H4入力の来歴 | 8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50 |
| 新評価SOURCE manifest | d55fb55b1a6c9db609f2fd9667ba937a9fbdadeb1d9ec6aa14f602de6760b9ce |
| 新未承認契約 | 851f43acc65be42dfcc21cef30f8c019bab7085566a93f9d42453917c4db048b |

J/Uの公開パスはそれぞれ `artifacts/research_c_common_convergence/2026-11-01/run01/{rc2,pr}/final.npz`。Snapshotは `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`。契約に全13入力hashとrole別identityを固定した。

NPZはbyte hashとNPY headerのみ検査した。J/U各[8,4,4]、x/jac各[128]、float64・非Fortranというschemaを確認し、payloadをdecodeしていない。元fit契約のfresh_output／role／final.npz、元postflightのraw hash、科学SOURCEを結び、role metadataだけで候補を取り違えないようにした。

旧RC2 fit SOURCE150f904f6975868abe09966dda8ab41722e2a7cc、旧PR fit SOURCEe1085c95f5c01b68566e17d543ace5832cb94061、その旧Stage A／B1／M4の数値は別の候補である。旧γ0 Baselineも今回のRC2ではない。既存APIの内部dispatchではrc2→baseline、pr→proposedを使うが、返却Arm.role、model.label、source/input hash、出力ファイル・比較結果には新しい科学identityを保持する。

## 3. 再利用したコードと必要な差分

新規実装は[専用module](../../src/trottertracks/c_rc2_pr_stage_b.py)、[CLI](../../scripts/run_research_c_rc2_pr_stage_b.py)、[人工テスト](../../tests/management/test_research_c_rc2_pr_stage_b.py)。旧runnerを再起動せず、今回のSOURCE／契約／fresh outputへ接続する。

| 既存実装 | 今回の利用 |
|---|---|
| c_pr_aware_df_stage_b.prepare_arm | 保存J/Uをnumber一般K／元g補正κ／scalar一回・元順LD2へ接続。NativeModelはmetadataであり実QuantumCircuitではない |
| candidate_factor、SectorCAR.raw_frame／one／physical | 一般Kと独立CARのbase＋8 frame、両sum、物理sectorの構築。新CARやHamiltonian再構成器は作らない |
| atom_operators、tail_gate | signed Z/ZZ、density projector、identity phase、tail sum・λの独立接続 |
| configure、c1_stage3_connection | 各arm固有のevent確率・normalizationを再生成。旧lawの値は転用しない |
| kernels、reference_kernels／reference_round | PFと補正有限RTE kernel、coherent反復、operator norm、A、raw complex signal |
| B1 ground／ground_gate | physicalと両own-DFのeigensolve／残差／alias。旧groundを別J/Uへ流用しない |
| feasibility.classify_round／summarize | 十分条件・不適格理由・証拠不足の分類 |
| integrated.Calls／resource_plan | 科学call、最初のcorrectness FAIL、動的並列度 |
| 修復済みsave_json | NumPy scalar変換、serialize-before-open、atomic/exclusive出力。旧partialの再発を避ける |

新adapterが必要だった理由は、旧B1の入場条件が旧Stage Aのsuccessと固有identityに固定され、M4も旧Proposedの保存PF／groundの同一性を要求するため。その旧条件を緩和せず、今回の有限予算終点の資格を独立に受け取る。Fitter、CAR、PF／RTE engine、ground solver、RPE estimatorは再実装していない。

旧B1のmanifestと今回起点には**既存の4ファイル差分**がある：stage_b.atom_operators、SectorCAR、reference._ordered_exponentials、rte.event_unitaryのoptional `arithmetic=None` 対応。今回その4ファイルを編集していない。既定binary64分岐の演算を静的に確認し、[履歴差分](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/historical_B1_default_path_diff.txt)と新SOURCEに記録した。B1時点のmanifestと現行library全体がbyte一致するとは主張しない。旧manifest・Git履歴・rawは保全する。

SOURCE74はB1の依存集合とJSON修復の依存集合、新3ファイルのunion。旧依存集合を選び直して省略するより、この小さな既存集合をそのまま固定した。Importされるmoduleに旧fit関数の定義があっても、このentryにfit呼出しはない。評価環境にffsimを追加しない。

## 4. 採用する最小政策と公平性

M4の政策をそのまま選定する：元順LD2、partial ordered symmetric S2、δ=h=.1、s1／r128／K0、m0..12、n=2^m、t=.1×2^m。既存B1の接続と[M4](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md)の有限RTE／時間／phase規則を再利用でき、複数r/s/Kの探索を追加する必要がない。

共通CA=.00159 Ha、DF reserve=.00019875 Ha、βtotal=π/6、βstat=π/12、failure total=.01／axis=.01/26、alias window[-10,10] Haを維持する。Own-DF sector ground-state oracleを仮定し、状態準備費用は除外。参照lawはideal finite-RTE IID。今回shotsやRPE estimatorは呼ばない。

同一物理snapshot／sector／rank8／frame順／LD2／PF-RTE政策／time grid／精度目標／十分条件を揃える。各候補のJ/Uが異なるので、**ground、tail係数・符号、λ、確率、normalization、PF/RTE誤差、Aは独立に取得する**。旧M3／M4に保存された数値kernelやPFセルは今回の新J/Uと異なるため再利用できない。旧報告のPASSも引き継がない。

RC2のλが以前の候補より大きくても、今回の適格性は実行前に決めない。必要な結果が得られなければそのまま不適格／未資格として保存し、rやKを変更しない。

## 5. 誤差と13roundの評価

各armのH_DFに対し、既存referenceで独立に次を測定する。

- εPF=||PF(h)^n−exp(−iH_DF t)||₂。
- εRTE=||corrected(h)^n−PF(h)^n||₂。補正有限RTEの誤差であり、PF誤差と区別する。
- A：既存normalizationによるattenuation。K0ではA=(1+τ²)^(−128n/2)、τ=λ_R h/128。
- Raw complex signal=A〈ψ_DF|corrected(h)^n|ψ_DF〉。Aはこのsignalの絶対値そのものではない。

Operator errorは無次元、DF reserveはHa。両者を単純加算しない。既知のraw-signal誤差はA(εPF＋εRTE)へ接続する。登録十分条件はεPF＋εRTE≤sin(βtotal−βstat)かつA>0で、**全law synthesis／implemented-law誤差は未知のまま**である。

各rowにarm identity、m、物理時刻、反復数、PF／RTE／A、raw signal実虚部、phase-room基準、分類と理由を保存する。PFが増えRTEが減る等のtradeoffを別々に比較し、λだけで下流性能を推定しない。両armの同じmをmatched_roundsにも対応付ける。

| 状況 | 処理 |
|---|---|
| 必要値がfiniteで登録十分条件内 | POTENTIALLY_FEASIBLE。CA達成や収束資格PASSではない |
| 登録十分条件を満たさない | INFEASIBLE。物理的不可能性を意味せず、他の登録セルは続行 |
| 必要値が欠測／未資格 | UNRESOLVED。未知誤差を0で補わない |
| Aがbinary64で0 | INFEASIBLE。PFだけ独立測定し、不安定な補正RTE power／raw signalはnull |
| SOURCE／入力／表現／law／ground／alias不一致、非finite等 | correctness FAIL。依存する後続を止め、開始済みtaskを回収、partial evidenceとfailed STOPを保存 |

Component atol1e-10・rtol0、ground残差1e-10、normalization1e-12、mass rel1e-12／abs1e-14を旧B1から維持する。元DF guardもそのまま保存結果から受け取る。Binary64数値pilotであり厳密enclosureへ昇格しない。

## 6. 科学call契約と実行順序

今回必要な処理から積み上げ、最大**143 logical units**とした。旧141／98／25-callの実行枠は再利用しない。

| 実行単位 | 最大units |
|---|---:|
| Input、保存候補資格、sector context、physical operator | 4 |
| 2 export load＋2 model metadata | 4 |
| 18 raw factors＋18一般K整合 | 36 |
| 2 sum＋2 law＋2 tail gate＋2 mass gate＋2 law distribution gate | 10 |
| Physical／両DFの3 ground＋3 alias gate | 6 |
| 2 PF／corrected RTE kernel reference | 2 |
| 26 exact evolution＋26 round reference＋26分類 | 78 |
| 2候補summary＋1 matched比較 | 3 |
| 合計 | **143** |

準備時preflightは0科学call。科学実行後もfit／optimizer／DF norm再計算／sampling／native build／compile／shots／総費用は0に固定。全correctness gateが通ってから26セルを並列実行する。候補のphase不適格は探索や全体correctness FAILと混同しない。Retry／resume0、fresh outputは `artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01`。

既定CLIはpassiveのみ。以下の科学実行オプションは**別途明示されたチャット許可後だけ**使う。契約のunapprovedファイル名やmetadataを承認証明へ変換しない。

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_rc2_pr_stage_b.py \
 --contract docs/research/research_c_rc2_pr_stage_b_contract_unapproved_20261101.json
# 科学許可後のみ、上記に --execute --output artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01 を追加
~~~

## 7. 人工・回帰と環境

新24＋既存B1／M4／Stage3 feasibility／JSON修復の64＝**88 PASS、12.45秒**。新テストの成功E2Eでは人工4軌道・rank8、実際の既存CAR／general-K／ground／finite-RTEを通し、26セル・143単位・journal・STOPを確認した。これは実H4の科学callではない。

Role取り違え、exportとmetadataを同時に交換するケース、rank／hash／SOURCE／policy／旧STOP／DF資格不一致、一般K／mass／alias／law FAIL、非finite時のfirst_failureとpartial STOPを検証した。INFEASIBLE／UNRESOLVEDでは残りの26セルが完了し、DF予算とoptimizer未資格、synthesis／law／CA／shots／total=nullを保持することも確認した。Tiny attenuationではPFを欠測と誤認しない。旧原本hashとfresh-output再起動拒否も人工E2Eで確認した。

実環境は[dependency pins](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/evaluation-pins.txt)：Python3.11.1、NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0等の既存evaluation venvを読み取り専用で使用。実fit用ffsim0.0.83は候補の来歴であり今回runtimeでは不要。Python symlinkをresolveせず既存venvから起動し、package metadata検査がPASSした。

Passive preflightではnumpy／scipy／qiskit／ffsim／jaxをimportせず、SOURCE74／入力13／header／environmentを照合した。Fresh scientific run01は未作成。実J/U・g配列のdecode、新operator・ground・PF/RTE・Aの評価は行っていない。保存DF予算の整合確認は既存JSONの技術照合に限定した。

資源確認時はaffinity32 CPU、available RAM56,234,827,776 bytes、load約1.96。既存planは26独立round taskに26 workers、BLAS各1を提案した。これはtask並列との調整で、CPU/RAM/worker/wallの人工上限ではない。実行時に空き資源／quota／他利用を再確認する。速度向上を実測したとは主張しない。共有venv、cache、設定、他jobは変更しない。

## 8. Q1〜Q10と次の判断

| 問い | 回答 |
|---|---|
| Q1：何の比較か | 同じ固定予算政策で保存されたRC2／PR-tailの下流挙動。収束済み最良RC-DFとの比較ではない |
| Q2：未収束でも意味があるか | DF予算内の有限候補が定義するHamiltonian／lawは評価できる。物理資格と最適化資格を分離する |
| Q3：M4再利用範囲 | LD2、s1/r128/K0、δ=.1、13round、own-ground、phase規則、既存有限RTE／reference。旧数値は転用しない |
| Q4：λの効果をどう分けるか | Aをlawから取得し、εPFとεRTEを別々に測る。J/U変更全体の比較でありλだけの因果効果ではない |
| Q5：何を独立取得するか | 両DFのHamiltonian／ground、tailとnormalization、kernel、13roundのexact/PF/RTE/A/raw signalと位相判定 |
| Q6：再利用はどこまでか | CAR、general-K、law、kernel、ground、feasibility、Calls、resource plan、JSON出力。旧libraryとrunnerは今回無変更 |
| Q7：なぜ新adapterか | 新role／保存終点／修復JSON／未収束資格を旧固定入場条件と混同せず、新fresh実行へ結ぶため |
| Q8：人工検証の範囲 | 正常143単位、identity／係数／phase／rank／SOURCE、FAIL・未資格・欠測・STOP・原本保全を88 testsで検証。科学GOではない |
| Q9：最小科学実行 | 2候補、同一1政策、26セル、最大143 units、新fit/native/shots0 |
| Q10：何の判断が変わるか | 下記A〜Dの整理からStage C取得の情報価値を判断できる。自動実行しない |

A：PR-tailのA／PF-RTE／位相tradeoffが有利なら、Stage Cのshots／native校正にどれだけ情報価値があるかをユーザー／研究Cが判断する。B：λ低下がPF悪化等で相殺されるなら、その理由を取得範囲で整理する。C：片方または両方が十分条件外なら、固定政策の不適格として保持し、方法全般の否定にしない。D：証拠不足なら不足する最小項目を示す。どの結果でも両optimizerの共通収束未達という制限は残る。

残る技術blockerは今回の人工範囲では検出されていない。実H4でのcorrectness／phase適格性は未測定で、全law synthesis／implemented-law／cost supportも未資格。Stage Bを通ってもCAや保証付き総資源の認定にはならない。

[技術postflight](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/postflight.json)でSOURCE74、入力13、元common SOURCE44、JSON修復SOURCE47、旧fit raw37／修復raw7を再照合した。起点の既存trackedファイルは今回の4索引への追記以外に差分がなく、旧failed STOP／partial JSONも不変である。科学outputは作成していない。

**次の担当：ユーザー／研究Cが、公開SOURCE・入力・143-call契約について限定科学実行を明示許可するか判断する。Codexはその許可後、固定fresh runを一度だけ実行する。** 今回は公開SHA/tree・本文・SOURCE・旧原本の照合後mandatory STOPし、実H4評価、Stage C、追加fit、重要科学レビューへ自動進行しない。
