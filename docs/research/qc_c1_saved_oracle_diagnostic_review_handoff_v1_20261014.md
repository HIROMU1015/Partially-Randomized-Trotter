# C1保存oracle・selector診断 v1：GPT判断用引継ぎ（2026-10-14 JST）

## 到達点

別途明示承認された保存posthocを一回実行し、独立checker32context PASS。通常16contextすべてがA_NO_ORACLE_5PCT_ADVANTAGE、bad-state16contextはeligible不足によりUNDETERMINED。oracle優位・selector missの分類Bは0。新eigensolver/PF/compile0、retry0。GPT科学レビューは開始していない。

[詳細結果・全scope・明細・限界](qc_c1_saved_oracle_diagnostic_v1_results_20261014.md)、[一次診断](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/diagnostic.json)、[独立audit](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/independent_audit.json)、[新規認可](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/authorization.json)、[report summary](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/result_summary.json)、[保全検査](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/static_integrity.json)。

## 今回の情報価値と既存C1結果との関係

元pilotの「三selectorで登録5%改善なし」を変更せず、未選択macro24の保存eligible点も含めたoracle費用を取得した。通常16件では強い対照を5%以上上回るmacroがない。method regretは存在するが、同じcatalogのより安いmethodへの選択差であり、強い既存対照に勝つmacroの見逃しではない。
method内hは既にoracle最適化されており、このregretは運用precision選択の指標ではない。jointの保存precision-valid結果（元6/32）は今回再評価しない。

## 科学判断として残る事項

- 今回の固定catalog内の否定的診断と、既存の同符号制約・known composition同値性・取得費用未確定を合わせ、現C1研究を継続・限定・終了するか。
- 再設計に科学的理由がある場合、今回のscope外にどの具体的な差分・改善機構が必要か。Codexの同型探索は再開していない。
- C4など別候補への移行を判断するか。今回の資料公開や診断承認は、その採択・開始承認を含まない。

この問いは判断依頼の準備であり、継続/終了/新規性の結論ではない。今回新たな数学探索・一般先行調査は行わず、同じC1で繰り返さない。

## 範囲と限界

非分子2qubit Pauli4入力、24macro/6h、h=1/(8Λ2^s),s0…5,Λ=fragment係数L1、通常|00>16/bad|11>16、energy target10^-5/10^-6 model units、controlled価格3/6。geometry/basis/DF rank-policy/L_D/固定粒子sector非該当。保存binary64の厳密有理算術と1ULP丸め感度で、費用モデル・物理誤差の不確かさは保証しない。bad controlsは未判定として保持する。
分子移送・全QPE/RPE総費用・近似state取得費用・真値なしのmethod/precision選択・独立新規性は未実証。catalog/grid外やC1全般の不能性を意味しない。

## 公開とSTOP

Branch `qc-c1-saved-oracle-execution-20261014`、固定source `657513ad06661aed84950d5907f3304dcdd5f1f7`、準備base `e82a5202098ad95f2eb7280279d2c97cc06190bc`。結果payload・remote SHA/byte照合・リンクは[公開receipt](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/publication/receipt.json)で取得確認する。
旧source/結果/認可/STOP/global FAIL、N1 STOP・N2 HOLD・N3 CLOSED・THRIFT現template終了を維持。今回認可は消費済み、保存posthoc再実行・新pilot・C4・新科学レビューは開始しない。Mandatory STOP、次のGPT科学判断にはユーザーの別明示開始承認が必要。

## 公開確認・最終STOP

[GitHub再取得receipt](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/publication/receipt.json)：結果payload `899289dacf3c55e00917c8f3e89a27edf8071cbc` の31変更blobs、338保護files、固定source16/input25をremoteからbyte/SHA照合した。公開は独立branchのみ、原worktree・旧source/result/認可/STOP/global FAILを保全。receipt自体へのリンクは後続公開metadataで解決する。数値診断や独立metric auditは再実行せず、Mandatory STOP。
