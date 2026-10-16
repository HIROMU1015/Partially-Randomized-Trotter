# M1-S V1 保存信号shot feasibility：実験契約候補

2026-10-17 JST。V0技術準備。`CANDIDATE_NOT_ADOPTED_NOT_EXECUTABLE / MANDATORY_STOP`。

今回ユーザーが採択したのは、Task A先行・Task Bを最終目標に保持・無補正raw直接route・保存全cellと二つの内部対照、という研究方針である。本候補の数値条件や統計policyの採択、V1数値解析の実施承認ではない。

## 対象と証拠の固定

基点 `870af3f65e3e31e19d2b002c556216c7340c06f0`。受領した[科学設計レビュー原本](../../research/m1s_rpe_applicability_scientific_design_review_20261017.md)をbyte一致で新規保存した。原本SHA256は`3be81af2707ac60f9259f828d4428f691f0a7d45ca9ed17d5f44ab4f3f63a10b`。原本内の未採択という過去状態は書き換えず、今回の方向採択を別のintake記録に保存する。

共通指示は`research_project_instructions_unified_2026-10-16.md`（SHA256 `8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa`）。以前の[Decision Packet](m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)と候補は履歴として保持する。

Hamiltonianはlinear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、保存stateの元norm²による商。既存δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1、shared_pm／ordinary／independent_pmを変更しない。24非零cell×3方式=72予測行、36保存zero control、計108行と48対応比較を必ず報告する。G9を落とさず、最良cellの事後選択はしない。

入力はcommit `025c5a90888b02b73d2815fbb10d6bd1d2823602`の保存raw結果だけである。

- path：`artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json`
- bytes：454422
- SHA256：`ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`

CSVはidentityを参照するだけで二重の数値入力にしない。Hamiltonian、NPZ、oracle、Pauli、W/v/ρ、有限step信号を再計算する経路は新SOURCEに含めない。

## Task Aの保証の意味

対象は保存raw-H理想coherent平均信号の円周位相である。`+iTH`の規約を保持する。エネルギーへの変換、alias解除、多round decoder、T=2の補完は含まない。Task Bの対象固有state・prior・全round・実装費用は未取得である。

将来の予測は常に`IDEAL_SIGNAL_HYPOTHETICAL_SUFFICIENT_SHOT_FORECAST`、証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。理想平均信号と同じmeanを持つ、fresh IIDな±1のX/Y測定を仮定する。実機のSPAM、状態準備、HD近似、合成誤差を含むshot保証ではない。旧oracleを厳密固有stateと扱わない。

## 事前固定する計算法の候補

保存された数値半径のcoverageとprincipal phaseの条件成立時に、

\[
 b^+=\min\{\pi,|\widehat b|+r_b\},\qquad
 A^-=\max\{0,\widehat A-r_A\},\qquad q=\gamma-b^+
\]

を使用する。保存`phase_bias_rad`は`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`である。モデル差`eta=|T|B`は既に総biasに含まれ、**二重加算しない**。signal精度2^-24／phase精度2^-18 radは旧数値解像度であり、新たなγの既定値ではない。

今回実装した候補routeは共通の`0<gamma<pi/2`、`q>0`、`A^->0`を要求する。これは単一時刻の円周位相に対する十分条件routeの定義であり、将来RPE decoderのπ/3条件と混同しない。γの値・このroute自体は未採択。

各軸のfailure予算をα_axisとし、

\[
 t=A^-\sin(q)/\sqrt2,\qquad
 N_X=N_Y=\left\lceil\frac{4\log(2/\alpha_{axis})}
 {(A^-)^2\sin^2(q)}\right\rceil_{\!upper}.
\]

±1変数のHoeffding境界`2 exp(-N t²/2)`とX/Y union bound、半径`A^- sin(q)`のdisk幾何から得る**十分設計**であり、最小必要shot数ではない。`atan2`で得る位相と真の位相の円周距離を評価する。branchをまたぐ実数差の保証やenergy/RMSE保証は主張しない。

αのscopeは未選択で、SOURCEは次の二候補だけを受理する。

- `PER_METHOD_ROW`：各方式行にα、各軸にα/2。全72行同時のconfidenceとは呼ばない。
- `FAMILY_ALL_72_METHOD_ROWS`：全72方式行のunion boundとして各軸にα/144。比較が48個でも同一72行を共用する。方式間の独立性はunion boundに不要。

乱択labelは各shotでfresh IID、両軸のconfidenceは等配分、整数化規則は共通とする候補。回路再利用、分散適応、方式ごとの特別な配分は扱わない。classical trajectory数をquantum shotsへ掛けない。

SOURCEは有理数のsin/log/π区間から式の上界を切り上げる。採択する共通`arithmetic_bits`はnullであり、結果を見て精度を上げる機構はない。小さなqや巨大なshot整数がcapを超えたらSTOP、区間が適格性を確定できなければUNDETERMINED。詳細は[算術・保証範囲](m1s_v1_arithmetic_coverage_and_resource_design_20261017.md)。

## 比較・費用・分類

全方式に同一γ・confidence scope・半径会計・算術解像度を適用する。比較は**同一の十分設計の整数shot数**であって真の最適shot数ではない。式の区間と選択した保守的整数を併記する。出力の上下界比較を、実際の資源の頑健優越へ読み替えない。

両方式で設計が得られた場合のみ、整数設計比を記録できる。軸別価格が等しく固定費を無視する限定proxyとして、`(C_sh+P_sh)/(C_cmp+P_cmp)<N_cmp/N_sh`を記号式と比率で示せる。C/Pのnative価格、状態準備の品質・実費、固定費Fは計算しない。一般の総額は引き続き`F+sum N_axis*(C_axis+P_axis)`であり、未評価。

|分類|意味|
|---|---|
|`CONDITIONAL_SUFFICIENT_DESIGN_AVAILABLE`|採用した条件付きモデル内の十分shot設計を構成できた|
|`PHASE_BUDGET_NOT_CERTIFIED_UNDER_SELECTED_ROUTE`|選んだ上界routeで正の位相余裕を示せない。一般的no-goや実際の推定失敗ではない|
|`TASK_OR_EVIDENCE_UNRESOLVED`|必要なphase status／半径／正の振幅下界が欠ける|
|`ARITHMETIC_RESOLUTION_UNDETERMINED`|採択解像度の区間で判別できない。自動precision retryしない|
|`SAVED_ZERO_CONTROL_NOT_FORECAST`|保存controlの来歴を保持。数値を再検証せずshotを付与しない|
|`TECHNICAL_STOP`|構造不一致、invalid値、resource cap、例外等。既得の全行prefixと未評価placeholderを保存して停止|

旧G9の`UNRESOLVED_ABSOLUTE_CONTRAST`をそのままcarryする。新予測の整数差・相対比で有意な改善へ格上げしない。materialityは未選択で、実装候補は`DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO`のみ。μ=0等を暗黙に採らない。`scientific_GO`、native費用、総資源は常にnull。

## Resource・実行guard

技術候補はCPU1 worker、GPUなし、BLAS等1 thread、メモリ2 GiB、入力1 MiB、result/log各16 MiB。有理数65536 bits、shot整数4096 bits、各series4096項、bound step総数100万、解像度fieldの実装範囲8–256 bits。wall/CPU time capなし。実測runtime・数値correctnessは未検証。

machine候補：[execution_contract_candidate.json](../../../artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/execution_contract_candidate.json)。全scientific fieldはnull、`adopted=false`、`coverage_model_adopted=false`、`contract_grants_execution=false`。方向採択flagだけでは開かない。

将来は(1)科学fieldを明示した新しい正式契約C、(2)さらに別途の明示実行指示を保存したauthority-only子commit Aが必要。SOURCE SはCより前に固定し、S<C<A、全SOURCE hash、入力hash、HEAD=A、clean worktree、契約Cのbytes、専用出力未存在を確認する。科学engineはguard通過後だけimportする。今回authorityテンプレート実体、grant、markerを発行しない。

将来のone-shotは専用出力dirのexclusive作成で消費する。初回解析／算術処理の失敗もretryなし。全108行・48対照をplaceholder付きで先に保存し、逐次checkpointする。例外はfailure付きで保全し、完了・失敗いずれもmandatory STOP。OSによる強制終了では直前checkpointを保持する。

## 採択前にユーザーへ返す事項

共通γ、α、confidence scope、誤差意味論・fresh IID/Hoeffding/等配分policy、共通算術解像度、保存半径coverageの条件付き受理、十分設計だけを比べる比較policy、科学GOを行わないmateriality policy、resource候補の採否。具体的値は推測していない。[未決一覧](../../../artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/scientific_open_items.json)を参照。

このSOURCEは未実行であり、数値correctness・性能を静的検査PASSから推定しない。未取得のTask B/native条件をV1のために独断で埋めない。

**V0完了後mandatory STOP。正式契約採択後も、V1数値解析には別途明示的な実行承認が必要。**

## V0固定・静的証拠

SOURCE S：`de622bd77ebd61a6d2cf5b69e29c4e620c8563b3`。machine候補はrunner・5つのnamespace file・結果schemaの7 fileをbytes/SHA256で結合する。

V0静的監査128 PASS／0 FAIL、metadata・guard・AST tests12 PASS／0 FAIL、結果schemaのDraft 2020-12 metaschema検査PASS。旧非index7,606 fileのbytes不変、旧manifest全entry不変、親1284 tracked file・index・HEADと全既存worktree HEADを保全。これらはローカル静的証拠であり、CI独立再現性や数値correctnessの検証を意味しない。

証拠は`artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/`の`static_validation.json`、`non_numerical_tests.json`、`preservation_audit.json`、`evidence_index.json`、`publication_receipt.json`を参照。receiptは候補payload公開後に別commitへ追加し、最終remote HEADを照合する。
