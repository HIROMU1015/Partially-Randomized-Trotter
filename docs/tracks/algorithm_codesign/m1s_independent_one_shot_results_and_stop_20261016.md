# M1-S Gate 0 v5：独立one-shot照合結果・mandatory STOP（2026-10-16）

**照合可能範囲PASS、全scope UNDETERMINED。承認された独立科学計算1回、retry0、mandatory STOP。**
H4 linear1.00 Å／STO-3G／8 spin orbitals／保存DF rank12、保存DF順HD prefix G3/G6/G9・残りHR、保存数値sector oracle。delta window・finite-step評価なし。
旧v5の保存結果に対する事後独立再現性確認であり、held-out実験や独立hardware認証ではない。

## 固定条件と一回の実施

- [別途のユーザー実施承認と直前文脈](../../../artifacts/track_b_m1s_independent_execution_preparation/2026-10-16/v1/user_scientific_execution_approval.json)：最新発言「次の作業に進んで大丈夫」が、直前に特定した独立科学one-shotを承認したものとして記録。
- [正式採択契約](m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md)：C=`3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8`、machine SHA256=`aaf9ac98f0f2ca27e3038834623a0724bbab3f01c050725357dd934c56957712`。
- 独立SOURCE S=`4787cf90b053d251bf6392988af35516bdbbe663`。旧v5 SOURCE=`69de022dc8142f703423b1ce0bd42f05e123954e`、証明=`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`、比較結果=`cdcb7f6f151df217960611f502eb1bac8be63f58`。
- [新専用authorization](../../../artifacts/track_b_m1s_independent_execution_authorization/2026-10-16/v1/execution_authorization.json)：A=`2a55ba0103c68a074b7be7bc43b90b383bb63233`。S < C < A、clean HEAD=A、input/SOURCE/proof/contract/comparator/runtime hashを確認して一回起動した。
- raw NPZ SHA256=`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、旧v5比較結果SHA256=`8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5`。

SOURCE・比較条件・精度閾値・resourceは採択契約から変更していない。旧v5は再実行していない。
[外部排他launch marker](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/exclusive_launch_marker.json)とSOURCEの排他output予約により一回を消費した。
[実行完了記録](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/launch_completion.json)、[実行context](../../../artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot/execution_context.json)、[停止記録](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/scientific_execution_stop_record.json)を保存する。

## 照合結果

全12 DF blockのoccupation/CAR整合、full-HおよびHD/HRの全65536 Pauli語trace/tensor再構成、identity移動、基底・state整合を固定SOURCE内で通過し、全3分割のexact referenceを完成した。
係数・lambda・ideal law・W/v/rhoは保存binary64入力が定義する対象を整数/dyadic/Fractionで評価した。
全referenceのcanonical hashを固定した後でだけ旧v5結果をparseし、照合後のhash不変も確認した。
reference SHA256=`f957093029feba4cfbf475fccdc65d32348bac323dc56ba86639eece5122f6e0`。

|分割|照合可能範囲|abs差 W|abs差 v|abs差 rho|全scope|
|---|---|---|---|---|---|
|G3|PASS|3.8647e-17|3.0066e-17|1.7203e-17|UNDETERMINED|
|G6|PASS|5.9335e-17|4.5105e-17|9.1407e-17|UNDETERMINED|
|G9|PASS|6.8064e-17|6.8509e-17|1.1499e-17|UNDETERMINED|

差は保存されたexact rational誤差の表示値を丸めたもの。W/v/rhoの固定絶対許容値は2^-24／2^-24／2^-12で全行PASS。
全3分割でsupport missing/spuriousと符号反転はそれぞれ0。係数L1、HR行列、identity、lambda、sampling lawは採択済みallowance内でPASS。
exact oracle絶対残差品質もPASS。ただしfull-Fock ground、固有状態距離、非縮退を証明したものではない。
[保存値表示summary](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/saved_result_display_summary.json)はrawからの引用だけで、新しいreference算定・閾値調整を行っていない。

full-H／HDの旧v5行列は未保存なので、今回もそれらへの直接一致を確認できない。
独立referenceの内部代数整合と、旧v5の未保存行列への直接一致は別であり、この部分は`UNDETERMINED_NOT_SAVED_BY_V5`を維持する。
全scope最終判定はUNDETERMINED、available_comparator_statusはPASS。全scope PASSへの格上げはしない。

## 保存・resource・監査

[raw result](../../../artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot/result.json)：541714 bytes、SHA256=`32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac`。
context 8841 bytes、checksum65 bytes、runner.log0 bytes。reference係数/行列については全語のsupport bitmap/digestと、各最大64件の差分witnessを保存する契約であり、独立行列/係数の全entry出力を主張しない。

CPython stdlib exact、CPU1 worker、各thread1、GPUなし。実行時間2.3633293355815113秒、Linux maxRSS56668 KiB、primitive-event10095034回。
2 GiB address-space、result/log各128 MiB、5億primitive-eventなど全採択capsを維持。scientific wall/CPU-time上限なし。
[静的性能計画](../../../artifacts/track_b_m1s_independent_execution_preparation/2026-10-16/v1/runtime_and_performance_preflight.json)どおり元SOURCEのcache/整数演算を利用した。比較benchmarkはなく、speedupを主張しない。

[保存専用auditor](../../../scripts/tracks/algorithm_codesign/audit_m1s_independent_saved_result.py)の[99 checks](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/saved_only_audit.json)はPASS。
監査はJSON/schema/hash/既存誤差台帳の整数比不等式の照合のみ。NPZを開かず、reference/係数/W/v/rho/allowance式を再評価していない。
SOURCEの既存143 static PASS・採択115 static PASSは過去の固定証拠として保持し、この実行後に同じsuiteを再起動したものではない。
[evidence index](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/evidence_index.json)と[publication receipt](../../../artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/publication_receipt.json)で公開bytes/SHA256・remote SHA・既存成果保全を記録する。

## 主張の範囲と停止

これは同じ保存入力・oracle・3分割に対するローカル独立計算経路の照合可能範囲PASSである。
SOURCE/CPython整数・有理数実装の正しさという前提は残り、既採択gamma比較モデルの新しい独立証明やplatform数値certificateではない。
finite-step信号、RPE shots、native circuit、総資源削減・研究としての科学的GO/no-goは今回評価/判断していない。
結果の科学的な意味の判断は次段階とし、GPT詳細レビューは開始していない。

旧v4 TECHNICAL_STOP/null・v5 one-shot raw結果・旧SOURCE/契約/authorization/消費済marker、元Codex/親worktreeを保全。
一回の独立実行後に新しい科学処理を追加せず、retry0、scope拡張0。**mandatory STOP。**
