# GPTへのPR-2 M2 transfer 最終実行前レビュー依頼

## 依頼と停止条件

M2の実行authorizationとactual science sourceをレビューしてください。
今回はsource/contract/schema/plan/authorizationと保存済みdevelopment JSONだけを確認し、
held-out NPZのresolve/stat/hash/load、signal、trajectory sampling、circuit build/compile、transfer実行は行わないでください。
必要なtestsも分子NPZを読まない下記対象だけに限定し、全repository testsは実行しないでください。

現在は `M2_AUTHORIZATION_FROZEN_AWAITING_FINAL_REVIEW` で停止中です。研究方針に追加修正を入れず、
固定5構成のheld-out transferを一度だけ行う条件が整ったかを判断してください。
本reviewが承認されても、利用者の実行指示まではlaunchせず、M2後の全statusでmandatory STOPへ戻します。

## 固定identity

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- Branch：`pr2-v4-s2-parallelization-20260928`。
- M1-B1 evidence：`8e0814e70c14ecf526444fac8a2142799610dc96`。
- 修正版contract source：`a529e9434d2e62fe752fdab5bd4c9a63fb15e830`。
- 正式contract plan commit：`40888b84b680ca726a603b677296cc97ec3534c4`。
- actual execution source：`2978e2fea672b7a1ff20cac74269ec9a610159dc`。
- execution plan commit：`b79c9c1bc392635898254a04a5d5239b55bcce6b`。
- authorization commit：`90a9f24707ec439cd3618cc4ec2616a8caaf1148`。
- 本依頼と監査資料を含むreview bundleの正確な40文字commitはhandoffに記載します。自己参照hashは埋め込みません。

| identity | SHA-256 |
|---|---|
| M1-B1 result | `71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4` |
| M1-B1 validation | `c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f` |
| 正式contract plan v2 | `d2bb5c5e57002fac5e8045f89a048913f4dadd5177d6f6d1465cc40a8755af7c` |
| execution plan v1 | `2aa09a927e5ac58ebe417397802ace0e70c0097e8d0c53c05457075a41e85527` |
| authorization JSON v1 | `dbc8b66fad5316004ef404fe8253d3f6cb0f29bf7065502ea995240e5fbd7ff1` |
| result schema v2 | `8612a9ba3ebb1a281214ad687b0ca766a51783172bda6cf5a40fca6847c06a36` |
| science module | `3b0d1e76ca488a28cf81165df1b36bdd671d07b17c4c155476ce38162c30c35b` |
| science runner | `1783da8cbdb22b60e491d073c1bc3e77dbb41b470998fb9f19cf8d1cf045224c` |
| science test | `e28a58f80d61b5b0f621aa8fb25b8f416fc272ac5abd5b660780b12574cd1234` |
| contract module | `394f528b157ebb76a1a12ae0ab7d7336f81339319d235655a7ef5fa592c39534` |

execution plan fingerprintは `ff7ed3d74bf4a0316adecdbac6b633978d71ca87bc1856d03db6fff9a2153276`、
正式contract plan fingerprintは `7880c8fed57a02f30a07ff7463eb65e423098ad9420e38410f365fad2e24cc4f`。
128 source hashはplan/authorizationに完全一致で保存しています。contract/science sourceは上記commit blobと同一です。

## 確認する資料

1. [契約v1](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)と
   [usable B2 amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)。
2. [science実装資料](pr2_matched_accuracy_m2_transfer_execution_implementation.md)と
   [実行authorization](pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)。
3. `src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`。
4. `src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py`。
5. `scripts/run_pr2_matched_accuracy_m2_transfer.py`。
6. `tests/test_pr2_matched_accuracy_m2_transfer_execution.py`と契約test。
7. `artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`の正式v2 planとresult schema。
8. `artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/`の
   `execution_plan_v1.json / execution_authorization_v1.json / authorization_audit_v1.json`。

旧v1と明示的draftは監査履歴だけです。正式v2 planを使い、M1-B1 result/validationも変更しません。

## 前回reviewの修正と科学的範囲

前回指摘された重大underestimateとratio supportの不整合は、
`usable B2 := method=B2 AND accuracy_eligible=true AND primary major_cost_underestimate=false`
で統一しました。6指標point Paretoは全accuracy-eligible構成から作りますが、
Pareto supportの証人とratio分子はusable B2だけに限定します。

優先順位は、gate failure、usable B2空ならNOT_SUPPORTED、usable B2ありendpoint空ならINCONCLUSIVE、
usable B2のpoint Paretoまたはupper2SE≤1.10ならSUPPORTED、それ以外でlower2SE>1.10ならNOT_SUPPORTED、
残りはINCONCLUSIVEです。P≥0感度はsecondaryです。

対象はH4 linear 1.30 Å、STO-3G、DF rank12、8 system qubits、T=0.8への次の5構成だけです。

| candidate | full wrappers |
|---|---:|
| B2-rank3-q1-r4-K2 | 64 |
| B2-rank3-q1-r8-K2 | 64 |
| B0-rank6-q1-r0-K0 | 2 |
| B1-rank12-q1-r0-K0 | 2 |
| B3-rank0-q8-r32-K4 | 64 |

held-outでaccuracy/bias/normalization/analytic shotsは同じM1式で再計算しますが、
候補、rank/q/r/K/T、thresholdを再探索しません。ineligibleな構成を置換しません。
primary RZと6指標matched-work Paretoを保存し、major primary underestimateはactual>1.10×development-cost予測です。
paired Re/Imのcovarianceを保持したSEとindependent-candidate delta-method ratio±2SEはengineering intervalでありformal CIではありません。

SUPPORTEDでも「developmentで固定した5構成のtransfer」を支持するだけで、
held-outにおけるmethod最適性、一般的rank3/q1最適性、r4/r8厳密winnerは主張しません。

## 資源identityと一回制限

random3構成の32 trajectoryはplanの96 seedsを使い、各trajectoryはRe/Imで同じevolutionを共有します。
axisごとにwrapper identityを分け、unique wrapper keysは196件です。
最大5 spawned CPU workers、各BLAS1、snapshot load一回、追加trajectory0、GPU操作0に固定します。

Qiskit1.3.0、basis `rz,sx,x,cx`、opt1、seed17、backend/coupling/layout/routingなし。
既存Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1をplanと一致させ、環境変更をしません。
固定root・output・commandはauthorization文書にあります。今回output/registryは作成していません。

source commit/candidate/axis/trajectory seed/index/compiler/wrapper semanticsまで一致したcheckpointだけを照合し、
cross-cell流用、runtime移送、欠けた結果の推定、未解決compile予約のretryを禁止します。
resume commandはなく、failure/interruption後はSTOPしてreviewへ戻します。
exclusive registryはfixed root内の制限であり、全checkout横断のglobal lockではありません。
別root/server/outputへ移して同じauthorizationを追加実行しない運用規則も固定しています。

machine status `M2_EXECUTION_AUTHORIZED_ONCE` は結果前の固定条件です。
最終review待ちは運用barrierであり、現runnerに独立review承認artifactの機械検査があるとは主張しません。

## 科学データを読まない検査

commit済みauthorizationのzero-science positive gateはPASSしました。
128 source blobs、plan SHA/fingerprint、環境、5候補、96 unique seeds、196 unique wrapper keysを照合し、
status/source/plan/permission/budget/run-limit/outputの7改変を拒否しました。
held-out resolve/stat/hash/load、H4 science、production run invocation、GPUは全て0です。

local focused84 passed、helper回帰134 passed、fail/skip0。正確なcommand、Python identity、
開始・終了UTC時刻、stdout/stderrはauthorization auditに保存しています。
synthetic小型回路の実Qiskit testsを含みますが、H4科学compileではなくimmutable CIでもありません。

許可するtest対象は次に限定します。

- focused：M2 execution/contract、M1-B1 contract/execution/result-validationの5 test files。
- helper：DF partial S2 repeated/repeated cost、RPE Hadamard interrogation、
  DF RPE Hadamard compiled cost、RPE Hadamard compiled-cost benchmarkの5 test files。

既存science runnerの `run` を試しに呼ばないでください。negative gateを試す場合もprivate data boundaryをmockで禁止してください。

## Reviewで判定してほしい点

1. actual science source→plan→別authorizationの順序と128 hash/blob identityに欠落がないか。
2. fixed候補・signal/cost fingerprint・q依存delta・step/occurrence独立sampling・paired axisが整合するか。
3. usable B2が両support経路へ適用され、empty-set優先順位、10%厳密境界、SE/covarianceの解釈に不整合がないか。
4. 196 wrapper、最大5 workers、32 trajectory、一回限り、no resume/retry、fixed outputがbounded transferに十分か。
5. implementation failureのpartial/null ledger監査を、科学的NOT_SUPPORTEDや0計算と混同しないか。
6. 終了4 statusすべてmandatory STOP、next-stage=falseを維持し、追加96・新分子等の認可へ進まないか。
7. 最終レビュー後に利用者の明示指示を受けて、一回だけM2を実行する段階へ進めるか。

## 回答形式

次のいずれか一つを先頭に示し、重大な問題と必要な最小修正だけを列挙してください。

- `APPROVE_M2_EXECUTION`
- `REVISE_M2_EXECUTION_BEFORE_HELD_OUT`
- `STOP_OR_NARROW_BEFORE_HELD_OUT`

reviewだけでheld-outを開かず、追加96、retuning、H5/H6/H12、別geometry/分子/PF、S3、
長RPE、最終総cost、研究判断の自動実行へ進まないでください。
