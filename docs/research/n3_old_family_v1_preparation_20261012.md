# N3旧条件族到達可能性／分離 v1：実装・検証準備

2026-10-12 JST。完了済み[科学レビュー](../../artifacts/n3_old_family_v1/2026-10-12/preparation/review_intake.md)を受領し、利用者の「Codex内で進められる地点まで」に従って準備した。
**S1二例の新しい診断は未実行、別の明示実行指示待ち、Mandatory STOP。**
今回のsource準備で同じ科学レビューをやり直していない。中心テーマは未採択、N1停止・N2保留を保持する。

対象はstar型S1の二つの既存方向、実対称one-body＋非負density、一外部mode、full10→active9、ν4、元δgoal=.005。
化学geometry/basis/DF rank/L_D/PF窓なし。posthoc developmentであり、元primary NO_ADDITIONAL_ACCEPTANCE_OBSERVED/2-base未達を変更しない。
δは保存OCCの正の有理上端をそのまま採用する。同じ係数/v/U/cと勝者を固定し、新しいκ/μを許すのは旧族診断の認可後だけ。

[数学・比較・resource契約](n3_old_family_v1_mathematical_contract_20261012.md)に一変数等価性、primal/dual、NORMの別条件、FIVE包含、STOPをまとめた。
有限探索の失敗を不可行性にしない。generator-selected κの再利用だけに制限せず、旧族全体をseparatorで排除するか、新しい厳密primalで回収するかを判定する。
新規性・研究継続・中心テーマ採択の判断はCodexでは行わない。

新moduleはcontract/schema、producer、独立checker、lossless inputsの4本。
producerはexact analytic ray/混合→固定grid→数値golden候補→有理検査。checkerは元係数からの旧Audit LDL再構成と厳密trace。
scriptはprepare/run/verify、testsはtest_n3_old_family_v1.py。元sourceは不変で、旧run()を呼ばない。

23 synthetic tests PASS：primal/dual/未判定、数値候補からの厳密証拠、singular/indefinite PSD、異常入力・偽証拠・不正認可の拒否、translation/scaling、dense9 reference、2 spawn workers。
[test log](../../artifacts/n3_old_family_v1/2026-10-12/preparation/synthetic_tests.log)、[集計](../../artifacts/n3_old_family_v1/2026-10-12/preparation/synthetic_summary.json)。
非dyadicの一点可行κ=1/3を有限候補が取り逃すfixtureは、数学的不可行へ誤分類せず未判定となった。
dense9 fixtureで一度、恣意的なchecker条件数>100というtest期待だけが失敗した（実際92、PSD検証はPASS）。意味のあるexact certificate PASSへ修正し、raw failure logを保持した。producerの数学や対象を変更していない。

S1ではlossless selectionとschemaだけを確認。[入力索引](../../artifacts/n3_old_family_v1/2026-10-12/inputs/index.json)。
新診断/PSD/eigh/証拠探索/U,c再計算はすべて0。レビュー付録codeや新S1解析を実行していない。
入力small shardsは8192bytes以内で、元pack/ledger/幅JSONのSHAと原pointerを保存する。

2 worker・thread各1・cached trace/abscissa、AS2GiB/worker・sampled aggregate4GiB、wall/CPU capなし。
[resource確認](../../artifacts/n3_old_family_v1/2026-10-12/preparation/resource_configuration.json)。速度倍率・量子総費用の主張はない。
新sourceを固定後、source/input/planの新しい一回認可を必要とするplanを作成し、資料をHIROMU1015へ公開・remoteで再取得確認する。
消費済みのsaved-width認可は流用しない。全結果・失敗後STOP、retry0、次段false、中心テーマnull。

次に必要なのは固定S1二例の一回診断の明示実行指示。新しい証拠・未判定結果が得られた後、その科学的意味をGPTへ戻す。

## 固定source・plan

source13 `1aa12ab0b777b28f1428d9bb8417af93c4376aae`、fingerprint `46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488`。
[execution plan](n3_old_family_v1_execution_plan_20261012.json)、SHA `ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5`。
input fingerprint `81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d`。新scopeの[authorization template（false）](../../artifacts/n3_old_family_v1/2026-10-12/preparation/authorization_template.json)。
未認可templateを用いてlaunchせず、明示実行指示後に別execution branchで有効な認可を作成する。
既存global manifestのdescription欠落FAILは保持し、今回entry schema/prefix保全を別に確認する。

## 実行前監査・保全

[source-bound preflight](../../artifacts/n3_old_family_v1/2026-10-12/preparation/source_bound_preflight.json) PASS、新S1診断/PSD/eigh/証拠提案0。
[未認可・消費済み認可の拒否](../../artifacts/n3_old_family_v1/2026-10-12/preparation/authorization_rejection_audit.json)はoutput作成前にPASS。
作業中rootでは外部の資料配置整理が入り、51文書は同じSHAの移動先を確認、5索引等の外部編集はそのまま保持した。
同一path/statusを維持したとは主張しない。専用branchの旧source/result/STOP・保護tree2882entries・旧manifest98entriesは不変。
[保全監査](../../artifacts/n3_old_family_v1/2026-10-12/preparation/preservation_audit.json)、[外部配置変更](../../artifacts/n3_old_family_v1/2026-10-12/preparation/root_external_relocation.json)。
