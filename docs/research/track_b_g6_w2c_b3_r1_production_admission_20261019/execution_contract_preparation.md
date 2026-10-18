# 将来の正式実行契約とSOURCE freeze準備

[JSON](execution_contract_preparation.json)は未認可の**準備metadata**で、`b3_formal_admission_v1`形式ではない。
R1採択を新しい版へ記録し、既存の未採択candidate・science contractは履歴として保存した。
今回admission SOURCE/testsとschemaは完全SHAで固定できるが、**実行可能な最終bundleのfreezeは未完了**。

| 項目 | 今回の状態 |
|---|---|
| R1/A6/input/policy/seed/setup/cache/X-Y/保存/二pass | 採択・正本照合済み |
| 旧science SOURCEと保存N | bytes/SHA固定、不変、再解析なし |
| 新admission SOURCE/人工tests | 実装・検査・公開固定 |
| guard・旧driver | SHA固定・不変。旧driverはF01ありでscienceに使えない |
| F01修正済driver SOURCE | 今回は未取得、未来SHAはnull |
| F01/admission/host統合、failure-provenance regression | 未統合・未実施 |
| 最終release40桁commit、全SOURCE pin、正式contract digest | 未固定、null |
| 独立実承認参照/hash・trusted bootstrap・実scope/claim/store | 未発行・未配置、null |

16cell（4arm×A/B×X/Y）、既存Nを維持。correctness256、baseline15,855,296、instrumented15,855,296、**総31,710,848**。
constructor24、Full prepare3（constructor込み）、canonical preflight9、budget load1、再計算0。
B accepted毎に数学的angle取得とevent再query、既存law/phase/weight/precision/confidence/identity guardを保つ。
source/stateと既存compact/初期全文/stream/failure-prefix保存規則を省略しない。
retry0、resumeなし、一束one-shot、失敗・完了ともmandatory STOP。
新B1/B2、C1/C2、P2/P3、新angle探索、native compile、科学レビューは含まない。

最終承認前preflight：F01公開固定blob確認 → 差分・接続interface照合 → 人工failure回帰 → 最終hostのcontext STOP接続 → 全pin/実import path照合 →
formal contract作成・完全digest固定 → 公開bytes/remote/preservation確認 → **ユーザーの別途科学実行承認**。
releaseと契約を先に固定し、その実値をユーザーに提示する。自己commit/hash循環はpublication receiptで避ける。
trusted authority/store provisioningはその実承認の対象へ含め、候補metadataから自動発行しない。
