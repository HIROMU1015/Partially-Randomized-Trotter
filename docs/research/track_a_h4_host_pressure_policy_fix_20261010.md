# H4 host-only pressure STOP：未承認の判定修正案

run06は約856.585241秒でmemory_pressure STOP。host full PSI avg10=0.18%、全visible nonroot cgroupは0%、
effective/host available478155976704B（約445GiB）、OOMイベント増加なし、max worker AS5.951GiB/RSS5.604GiBで8GiB以内。
4 actual予約/完了0/signal0、全6owned identityを2回ABSENT確認、driver exit143。正確な終了時刻とpressure発生者は未確定。
run05でもhost0.18/nonroot0でSTOPし、worker12→4への減数では解消しなかった。
キャッシュ原因はこのSTOPでは記録されず、今回実証された原因は既存policyによるhost PSI非zero停止である。
他ユーザーのjob停止・調査・共有host/cgroup/sysctl/venv/GPU変更は行わない。

## 元契約と変更が必要な一点

元[固定contract D4](../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/zero_compute_plan_v2.json)のpressure_stopには
「memory PSI full avg10>0 ... => stop own run」と明記されている（行3208）。
利用者は通常の必要修正とfresh再実行を任せ、失敗費用の次attemptへの加算不要も明示承認している。
この条件は既存の明示契約であるため、通常の実装判断だけで緩めた扱いにはしない。
必要な追加承認は以下のpressure判定変更だけで、環境/CPU/observer/17GiB/74805等の再承認は不要。
承認前は本計算を起動しない。旧contract/認可/source/入力/STOP証拠を保持する。

## 具体的なinactive実装案

host full.avg10が1.0%以上ならSTOP。全visible nonroot cgroupのどれかが非zeroでも従来どおりSTOP。
0 < host full.avg10 < 1.0%の例外は、initial cgroup namespaceと完全なv2 hierarchy/scopes、全nonroot PSI0、
OOM baseline一致、effective available120.25GiB以上、観測fresh<=5秒を同時に満たす場合だけ継続する案。
host PSI0の通常経路は従来のeffective headroom16GiBを保持。欠測・NaN/Inf・negative/boolean・scope/namespace/hierarchy不整合・OOM増分はSTOP。
role AS/RSS8GiB、observer AS256MiB/RSS64MiB、interval/duration/staleness5秒、所有確認、first STOPのfsyncとown-run限定cleanupは保持する。
1.0%は候補値で、最適値やあらゆる共有負荷での安全性が検証済みとは扱わない。pressureの発生元を他jobと断定しない。

pure module pressure_proposal.pyは判定を試算するだけで、production codeに接続していない。
全結果とprofileはapproved=false/runtime_authorization=false/production_wiring_present=false。
旧55 sourceはbyte-identicalで、現在のobserverはhost0.18でSTOPする旧動作のままであることも検査済み。
新SOURCE `41d2b61f640937255c5fa6cab9549ae67b83c462`、closure58、branch `track-a-h4-host-pressure-policy-fix-20261010`。
将来の採用にはproductionへの接続と新SOURCE固定、pressure amendmentのauthority/profile、seed、plan/auth/reviewの再結合と新未使用runが必要。
本案のSOURCEを既存run06 authorizationへ使い回さない。

## 限定検証と現在の状態

16pure tests PASS、fail/error/skip0、wall0.002024秒、peak RSS24,641,536B。
通常1 process/thread1、AS2GiB/RSS128MiB/wall30s/output1MiB。科学array/回路build/compile/transpile/実child/affinity/GPU/本体0。
recorded run06のhost0.18・nonroot0・available/OOM値が提案ではPSI理由でSTOPしないことをfixtureで確認した。
この再生はnamespace/hierarchyのsynthetic metadataを明示的に補ったもので、過去のnamespaceを復元した証明や本計算成功ではない。
threshold境界、nonroot各scope、OOM増分、headroom、欠測/改変、旧8GiB/5秒/first-cause/frame8192Bも検査。
旧benchmark128/synthetic28/64・旧campaignは再実行しない。

本計算は停止中。carry0、4workers CPU[2,4,5,6]/driver16/observer18、既存venv不変、17GiB/74805/72hの既承認を保持する。
科学条件・compiler options・凍結6入力とgeneration freezeを変更・再生成しない。旧partial/cache/random結果を混合しない。
[資料bundle・契約変更草案](../../artifacts/resource_applicability/track_a_h4_host_pressure_policy_fix/2026-10-10/README.md)へ検証・source・旧STOP証拠をまとめる。

独立review：`PASS_INACTIVE_PROPOSAL_UNAPPROVED_NO_PRODUCTION_LAUNCH`、blocking findingsなし。SHA256 `d00fd7d80f1feb78152fb5245be980cba486c0d3c0033e6a36133d8f7c8b3c93`。inactive実装/元production不変/契約変更必要/限定fixtureの証拠範囲を照合。runtime採用は未承認、接続/新SOURCE/seed/binding/freshは別途。
