# M1-S Gate 0 v5：独立correctness検証契約の正式採択（2026-10-16）

**VALIDATION_CONTRACT_ADOPTED_BY_USER / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED。mandatory STOP。**
ユーザーがcommit `2ba0858c3c92b61581cfbf6cdfb5039d25a174c9`の候補を正式採択した。今回の承認は採択記録・契約固定・静的検査・公開まで。
新規科学計算、独立Pauli生成、実入力/模擬入力の数値correctness test、W/v/rho評価、旧v5再実行は行っていない。

## 採択根拠と固定契約

- [今回のユーザー指示原文](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/user_adoption_instruction.md)：「科学計算は承認しません」を含む。
- [採択文脈](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/adoption_context.json)は、直前の推奨が採択ではなかったことと、今回の明示採択を区別する。新しいGPT詳細科学レビューは行っていない。
- [正式machine契約](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/independent_validation_contract_adopted.json) SHA256：`aaf9ac98f0f2ca27e3038834623a0724bbab3f01c050725357dd934c56957712`。
- [ユーザー採択記録](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/user_contract_adoption_record.json)／[入力・SOURCE・proof・契約のidentity台帳](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/identity_ledger.json)。
- [旧候補と仕様説明](m1s_gate0_v5_independent_correctness_contract_candidate_20261016.md)および候補JSONはbytes不変で保存する。

採択済みmachine契約の差分はstatus=ADOPTED、adopted=true、adoption_recordのpath/SHA256のみ。
算術経路・対象・精度条件・resource・失敗分類に変更はない。record→candidate、contract→record、別identity ledger→contractのhash関係で循環参照を避ける。
adopted=trueは検証契約の採用を表し、数値correctness、SOURCEの形式検証、platform認証、科学的GOを意味しない。
contract_grants_execution=falseを保持する。

## 採択した対象・比較・限定

linear H4 1.00 Å／STO-3G／8 spin orbitals／保存DF rank12、保存DF順prefix HD G3/G6/G9、残りHR、保存数値sector oracle。
delta window・finite-step・simulation・native回路費用・総資源の評価は含まない。既存結果公開後の事後独立再現性確認である。
SOURCE S=`4787cf90b053d251bf6392988af35516bdbbe663`、v5結果=`cdcb7f6f151df217960611f502eb1bac8be63f58`。

occupation-bit/CAR行列、全Pauli traceと独立tensor再構成を用い、保存binary64をexact dyadicとして扱う。
stateは元のnorm²で割り、integer moments/FractionでW/v/rhoを求める設計。reference丸めallowance=0はSOURCEとCPythonの実装が正しいという前提付きであり、厳密なplatform certificateではない。
旧v5計算helperは利用せず、全3分割のreferenceを固定するまで比較結果JSONをparseしない。採択済みgamma比較モデルは別実装で共有し、そのモデルの新しい独立証明を主張しない。

係数L1 allowance=gamma_(713+r)S_R [Ha]、HR行列F差²<=256 alpha_R² [Ha²]、identity、lambda、nearest-even probability照合、state norm²=2^-40、probability mass=2^-46を維持。
W/v/rho絶対比較は2^-24／2^-24／2^-12、oracle residual²品質は(1e-8 Ha)²。identity scalar加算は既存gamma_34 allowanceで、nearest-even関係は診断のみ。
許容範囲内のsupport/sign差はUNDETERMINED、許容超過はFAIL。結果に合わせた閾値変更は禁止する。

**今回明示的に受け入れた限定**：旧v5のfull-H行列とHD非identity係数/行列は未保存。
独立targetは再構成・内部整合検証できるが、未保存の旧行列への直接一致は確認できない。
照合可能なHR・identity・lambda・sampling law・state・W/v/rhoの`available_comparator_status`はPASS／FAIL／UNDETERMINEDとして報告する。
一方、全scopeはFAILがあればFAIL、それ以外はUNDETERMINEDであり、現比較データでは全scope PASSを出さない。
欠落範囲だけを理由に照合可能範囲を省略しない。full-H/HDを回収するための旧v5再実行は承認されていない。

## Resource・one-shot・承認境界

候補どおりCPU1 worker、各thread1、GPUなし、2 GiB address space、result/log各128 MiB、context4 MiB。
integer bit cap65536、primitive-event cap500000000、coefficient/matrix cells65536、transition cache4352、Pauli cache512、差分witness各64。
wall/CPU-time上限なし。固定Python3.11.0rc1の実行ファイル・stdlib5hashを維持する。性能実測・数値成功は未検証。
全3分割の状態報告、排他的output予約時のone-shot消費、上書き/automatic retry禁止、failure時の完了ledger保存とmandatory STOPを採択した。

実行順はv5結果 < 独立SOURCE S < 候補 < 採択契約C < 将来の別実施authorization A。
採択契約のmetadata検査完備とactual_execution_permitted=falseを明確に区別する。
将来Aでは別の明示的ユーザー科学計算承認、独立検証専用namespace、clean HEAD、全hash/runtime/thread照合が必要。
今回A・grant・marker・科学outputを作成していない。旧v5のauthorityで独立検証は実行できない。

## 静的検証・保全・停止

[採択static checker](../../../scripts/tracks/algorithm_codesign/check_m1s_independent_contract_adoption_static.py)／[監査artifact](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1)／[未決条件](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/remaining_conditions.json)。
原SOURCEの143項目suiteは候補JSONで再実行し、採択契約は追加の非数値checkerで型・metadata差分・hash・guard・coverage分類を確認する。
NPZはraw bytesの同一性だけ確認し、配列decode・reference kernel・数値testは呼ばない。静的検査は実数値correctnessの検証ではない。
旧候補・SOURCE・v4 TECHNICAL_STOP/null・v5 one-shot結果・旧authorization・消費済みmarker、親worktree、他Codex成果を保全する。
GitHub remote SHAと公開bytes/SHA256確認は[publication receipt](../../../artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/publication_receipt.json)へ保存する。

**新規科学計算0件、新しい科学実施authorization/grant/marker0。採択記録固定後mandatory STOP。**
次の独立数値one-shotにはユーザーの別途明示的な実施承認が必要。
