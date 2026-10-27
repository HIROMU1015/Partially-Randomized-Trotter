# H8 q16既存結果の再利用確認（Phase A）

2026-10-27。保存資料・SOURCE・metadataの確認のみ。新しいsignal、N、G、SE、順位は計算していない。終了済みH8希少RTE研究のSTOPを維持する。

**ε=.001のq16会計は既に存在する。ただしEXPLORATION n8であり、ε=.0005の独立確認n48とは別の結果である。n48価格をε=.001へ結び直すには、正確な保存endpointの回収と、別承認による保存値会計が必要。現段階で重複compileを提案する理由はない。**

## 判定を適用する範囲

|対象|分類|今回確認できたこと／残る条件|
|---|---|---|
|q16・ε=.001・通常signed・allowance倍率1・P=0の既存探索点|`REUSE_SUPPORTED`|既存の有限標本費用として引用可能。n8の保存G/SEをそのまま使用し、q8との新しい順位は作らない。|
|q16・ε=.0005の独立n48結果と両軸価格記録|`REUSE_SUPPORTED`|元の採択範囲の保存結果として利用可能。実行SOURCE、固定shot、96軸の候補・replica・cohort対応がある。外部再現や母平均保証を意味しない。|
|独立n48価格を元mapのε=.001比較へ用いること|`REUSE_CONDITIONAL`|epsilonは回路ではなくsignal/shot会計を変える。後述の入力・law・controlは対応するが、q16のε=.001における正確なNとfull signal recordの回収・対応検査が未完了。回収後の再会計も今回未承認。|
|q16のε=.001 full signal、signed bias、u、logB/margin、headroom、両軸Nの完全な原記録|`SOURCE_OR_DATA_MISSING`|公開Git資料の確認範囲ではfull recordを取得できなかった。保存GからNを逆算しない。元private A_signalの所在・内容の回収が最小の次作業。|
|ε=.0005のN/Gを倍率でε=.001へ移すこと、または異なるcohortをpoolすること|`REUSE_NOT_SUPPORTED`|signalのbias/headroomや標本の意味が変わるため、その転用は採用しない。|

これは新しい科学的順位・保証の認定ではない。前回境界調査 `db1b5bdf96c05aa9ff23e9f843c2d3101496c8ed` の「q16・ε=.001 endpoint未発見」は、旧120行の会計表等に限った探索結果だった。今回は後続960行の会計表から該当行を発見したため、**その資料不足の記述を本報告で補正する**。旧調査ファイルは変更しない。

## 保存数値と出所

以下は原表からの転記であり、再計算・相互比較・再ランキングはしていない。

|候補・精度|cohort / n|保存G|保存paired SE|出所・row selector|
|---|---|---|---|---|
|`B2_p7_q8_2nd_r4_K4` / .001|旧独立確認 / 32|15360006659421.375|45479861536.20314|resource-map結果 `681ec23c93fdd1090d325908f4b155a79115afdf:artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/selected_confirmation_rows.csv`：cell_id=当該ID, epsilon=0.001, allowance_factor=1.0, metric=RZ|
|`B2_p7_q16_2nd_r4_K4` / .001|`EXPLORATION_N8` / 8|`40802612537883/2`|50209504581.336464|下記A1：system=H8, cell_id=当該ID, bias_mode=USUAL_SIGNED, epsilon=0.001, factor=1.0|
|同q16 / .0005|`H8_CONFIRM_P7_Q16` / 48|`546493203592571/6`|144211830223.71762|下記A2：candidate=当該ID, epsilon=0.0005, mode=primary|

- **A1**：`862d732fba0a5f353c522a9219dec78ae2f21a4b:artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/new_pr_cost_points.csv`。960保存行。上記rowの会計SOURCEは `8601b921342fc1cc83780a830e1f65bf64fe9e93`、status=`FINITE_SAMPLE_POINT`、independent_confirmation=False、population_mean_certified=False。
- **A2**：同commitの `artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv`。保存N_cos=79,863,816、N_sin=99,933,856。対照は `B0I_p15_q4_4th`、保存G=`94193016652860`。これは.0005の固定比較である。
- **A3**：同ディレクトリの `paired_metrics.csv`。q16はreplica 0..47×cosine/sineの96行。candidate、replica、axis、cohort、reused_complete_trajectory、RZ/CX/depth/size等を保存。今回、平均・共分散・SEをこの表から作り直していない。
- **A4**：[n48公開結果説明](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)。完了済み両軸の再利用、旧compiler outputとの完全同一性は未検証、ローカル検証という限界を明記している。

## 科学条件と実装の照合

線形H8・隣接1 Å・STO-3G・tol-only 1e-8 DF rank28、16 spin modes、Na=Nb=4、sector4900、T=.8、P=0。元mapはprefix7/q8/r4/K4、後続はprefix7/q16/r4/K4、いずれもS2。δはそれぞれ.1/.05、trajectoryのevent数q×rは32/64であり、同じ物理候補の価格ではない。**q8価格をq16価格に置換するのではなく、同じ物理入力の別候補として比較する。**

mapの入力manifestと後続確認designは次のbindingを共有する。

|binding|SHA256|
|---|---|
|Hamiltonian|`1a5ebefa18fc1c9f01ad644b81d13f49c7bf944bccfe3837d459c35d3325dbe2`|
|state|`50a8fcb4cd29d60349db8e8fe728cb609eaaa903d9531833ef63e2711fa363fb`|
|snapshot|`d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`|
|compiler|`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`|
|profile|`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`|

mapの科学providerは `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、後続確認SOURCEは `745a674546f2cb6258526fbfde1d9a00601689f8`。後者の `artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json` とSOURCEを照合した。**PROPOSED自体は実行receiptではなく、実行経路の意図・照合対象を示す資料**である。実行済み結果はA2/A3/A4に基づく。

|項目|SOURCEから確認した対応|証拠の限界|
|---|---|---|
|H/state/fragment順|`scripts/pr_cost_native.py:load_context` はham/state/snapshot、元native config、preparation Hamiltonian、deterministic fragment indices=range(prefix)を検査。元mapもgeneration順を維持。|後続private provider/preparationの全実行receiptは今回未回収。hash文字列だけを実入力再監査と呼ばない。|
|reference|両研究とも同じ保存full-H/state・T=.8の複素signalを参照し、reference非回転。q16の価格取得はreference solverを呼ばない。|q16 full A_signal未回収のため、元mapとのreference複素値の直接照合は未実施。|
|finite-RTE law / B|`h8_pr_confirmation.py:compile_one` は元request法則のdict/digest、preparation hash、各step r=4、R=64、R log(b)対保存logB/marginを検査。`pr_cost_native.steps` は元finite cutoff、raw concatenation、q回のδ=.8/qを維持。|その検査の公開SOURCEと保存結果の対応を確認したもので、新しくlaw/Bを数値評価したものではない。|
|scalar・signed evolution|full-Fock scalarを一度適用し、元のsigned eventとtimeを維持。S2なのでm5 signed-piece allocationとは別契約。|ε変更でlawを変えてよいという意味ではない。|
|control|map `h8_fast_cost_port_v1.py` も確認 `h8_pr_confirmation.py:224` も実loweringへ **`control_policy='symmetric_directional'`** を指定する。|ordinaryという文章表現だけから異なるcontrol方式とは認定しない。|
|ordinary ancilla|`pr_cost_native.control` のlow/high branch、漏れ、X/Y、event oracle検査を `ordinary_ancilla_branch=True` と記録。確認受入処理はreference_rotated=Falseも検査する。|ordinaryはこのbranch検査のフラグであり、実lowererへのordinary policy指定ではない。|
|actual adjoint|map providerの `ax2b_native_df_v5._append_block` はDF basisをinverse=True/Falseで実adjointにより挟む。後続dispatcherは同名providerの元preparation/loweringを呼ぶ。|同providerモジュールの実行bytes一式は確認SOURCEのGit tree内にないため、完全byte照合は未完。明らかな意味論の不一致は発見していない。|
|wrapper|確認SOURCEは同一trajectoryのcosine/sineへ `build_native_hadamard_wrapper(...,include_measurement=True)`、入力不変fingerprint、元compiler設定を検査。|stateprep P=0、測定込みcompiled RZ×analytic shotの有限時間signal費用。QPE/RPE全体費用ではない。|
|compiler/runtime|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。compiler hash以外にも上記law/preparation/control/wrapperを検査するSOURCEである。|runtime/provider完全receipt・旧compiled outputのbitwise同一性は未検証。これは既存有限標本価格の引用を否定するものではなく、再現性と転用範囲の限界。|

追加の静的照合：`src/trotterlib/df_partial_s2_repeated.py` と `df_rte_qiskit.py` はc31efと745aのGit blob bytesが一致した（SHA256それぞれ `65b5ebd842a972735c7aa1e7ba4d11d8265ab586119f29bf5acc8631700c7d6d`、`c710de23b7bfef525199ab28b897c21b2a78723a01b29da6067ff64354dcd436`）。ただし、これだけでprivate provider全体の一致とはしない。

q16 designが参照する保存record等の識別子は以下。回収時に同定するための値であり、内容を今回読めたという意味ではない。

- A_signal record：`3d3e0acd6354183202d4f1c6bc2d48240d5741c5446489514953900db30456eb`
- .0005 primary accounting：`62252351a064b6e8c91c02242ae1d065fc232d9eb0975724a83ae5610cef16d6`
- prefix7 preparation：`f794c14366f6081237cafd80a91b6659c98450f3eb720c90073ab5ed2263abc4`
- q16 original law：`2caf4ea83b53cbedccc77fdd8a022adc695717c4423c60b2b5f294a6e17519b7`

## 最小の追加作業とSTOP

1. 元A_signal export/private保存資料を**読み取り回収**し、q16の.001通常signed/倍率1/P0についてreference、corrected/raw、bias、u、B/margin、headroom、両軸Nとeligibilityを対応付ける。旧152候補signal、旧1900行phase-C会計、旧120行PR会計にはq16がなかった。後続960行会計にGはあるが、N/bias/uの列はない。探索A候補index・測定軸表・確認designからもfull recordは回収できなかった。
2. 回収後、既存SOURCEの検査に沿ってlaw/preparation/axis/cohortを確認する。後続private runtime receiptは、科学的に重要なcontrol/adjoint/orderingを補強するために確認する。単なる環境差を理由に旧結果を自動的に破棄・再compileしない。
3. **別承認後のみ**、既存paired統計と正確な.001 Nでn48のG/SEを保存値会計する。既存paired関数を再利用し、.0005のN/Gをスケールしない。full signalが回収不能なら、必要なsignal/shot復元はさらに別の科学実行範囲として提示する。現状は新しいsignal・sampling・compileを要求しない。
4. 旧q8 C32、q16 E8、q16 C48、S2_SINGLE条件付きpilotをpoolしない。q16 C48ではTaylor0=3072、Taylor2/4=0という保存観測があり、無条件母平均の保証はない。採択済みの層別研究も元の.05/.0005固定範囲のまま保全する。

[正式終了方針](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)は維持する。q16重複実測、prefix6/r8/K6、未観測層の追加取得、H8の自動拡張は開始しない。**Phase A完了、mandatory STOP。**
