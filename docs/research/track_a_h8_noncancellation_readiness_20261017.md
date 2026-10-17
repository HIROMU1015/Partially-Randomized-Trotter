# H8 非相殺bias診断：runner・証拠照合・人工検証の準備

**全184決定論候補（旧B0+I復元12候補を含む）とPR確認union6を対象に、per-axis非相殺診断runner・入力/SOURCE結合・無効grant案・人工54件を準備し、PR確認集合の保存証拠をmetadataだけで照合した。実H8保存値による非相殺bias・shot・費用・ランキングは未実行（0）。**
B2 recovery attempt02は準備中の16:15:03 JSTに `MEMORY_PRESSURE` でSTOPしたため、B2全4軸PASSの受領証拠は存在しない。runnerの費用比較stageはB2全PASS結合なしでは起動できない設計で、今回も実行していない。

- SOURCE：`d567e6e1f3b1713b455a80da9b5c5f9e7848f2de`（runner・契約・監査器・tests・入力manifest）。レビュー指摘2点の修正後SOURCEは `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`、未承認plan v2 digest `002f9edb…`（[修正追補](#修正追補レビュー指摘2点2026-10-17)）
- 起点：signed成分検証結果 `8e54303bfe46bbebbf4d37148cbe26ef4013ae38`、証拠監査 `b75039c1d5ead475d8c0ff8b62dba04a9978a6fa`、B2準備 `3a450af0fb431269ac94152807992883c1490b26`
- branch：`track-a-h8-noncancellation-readiness-20261017`
- [準備bundle（plan・無効grant・監査・検証・保全）](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/README.md)
- [数値module](../../src/trotterlib/h8_noncancellation.py)／[契約・入力adapter](../../scripts/h8_noncancellation_contract.py)／[runner](../../scripts/run_h8_noncancellation_diagnostic.py)／[metadata監査器](../../scripts/audit_h8_noncancellation_evidence.py)／[人工tests](../../tests/test_h8_noncancellation_diagnostic.py)／[入力manifest](../../references/h8_noncancellation_inputs_v1.json)

H8直線鎖1.0 Å/STO-3G、保存tol-only DF28、16 modes/sector4900、T=.8、q1/2/4/8。
旧32決定論候補（B0/B0+I prefix7/14/21 S2、full PF prefix28 S2/S4）と追加152候補（B0/B0+I S2 prefix8〜20（14を除く）、S4 prefix15〜21）。
PRは事前固定の確認union6（`B2_p7_q{2,4,8}_2nd_r4_K{2,4}`）各32 paired trajectory。探索n8は混合しない。
データ内method `B1` はfull PFであり、費用fitのPhase B1とは別。

## 1. 三つのbias量の区別

|区分|各real/imag軸の量|runner内の扱い|
|---|---|---|
|通常signed bias|`abs(total_signed_axis)`|保存済み通常会計1,900行を凍結ruleで完全再現するgateにだけ使う。新しい通常費用比較は行わない|
|経験的非相殺診断|B0/B0+I `abs(discard)+abs(PF)`、full PF `abs(PF)`、PR `abs(outer_PF)+abs(finite_RTE)`|今回準備した新診断。分解粒度に依存する経験的感度診断|
|厳密bias保証|真の対象signalとの差の証明付き上界|全出力で `null`。非相殺量・経験的uを足しても昇格しない|

成分は再計算せず、signed検証run01（SOURCE `64ec6cbb…`、plan/grant digestとraw結果SHA `b9539323…` を固定）の検証済み368軸を再利用する。旧B0+I12候補は同run01で保存端点から復元された24軸である。
各軸で成分和と保存totalのclosure（`math.isclose`、rel/abs `1e-12`、signed検証と同値）を結合検査し、軸平均・complex norm・許容差緩和で救済しない。

## 2. shot算術の維持

shotは凍結D0 accounting（`scripts/d0_saved_value_reanalysis_v1.py` SHA `1ded6445…`、規則本文 `d0_frozen_shot_rule_v1.py` SHA `ed3f5793…`、起源 `c31ef424…`）を**そのまま呼び出し**、`abs(total)` を宣言した非負bias量に置き換える点だけを変える。

- ε=.05/.01/.005/.001/.0005、軸ごとにε/√2、α_axis=.025、allowance倍率は**1と10の2条件**（0.1ではない）。
- 保存した経験的combined u×倍率を軸ごとに一度だけ使う。成分ごとの反復加算・二乗和縮小をしない。
- normalizationは保存log_B＋倍率×保存log_B_margin。PRは `log_B == R*log(b)` の保存binding完全一致を要求する。
- headroom三状態（ELIGIBLE/INELIGIBLE/UNDETERMINED）とu/headroom≤.01のguardを維持。guard不成立・UNDETERMINEDは「未解決」で、不適格や費用0にしない。

実行前gateとして、保存済み通常会計1,900行（旧B0/full PF200・旧B0+I120・Phase A1,520・PR60）を同じ凍結ruleで再計算しdict完全一致を要求する。これは費用比較ではなく、入力と算術の同一性確認である。
旧B0+Iの保存会計も来歴codeが `g.d0.accounting(signal,e,f)` を使っていることをcode読取で確認した（数値照合は未実行）。
保存PR paired費用（確認n32・通常shot）の凍結 `weighted_metric` による完全再現は、**費用比較stage（`COST_COMPARISON`）だけのgate**である。shot診断だけのstageでは実行せず、結果にも未実行と明記する（修正追補参照）。SE浮動小数の同一性のため、実行環境をD0契約と同じPython3.12.3・`statistics.py`（hash一致を確認）へ固定した。

## 3. STOPと未解決の区別

|種類|例|扱い|
|---|---|---|
|STOP（fail closed、両stage）|入力bytes/SHA・来歴不一致、通常会計の非再現、非finite・overflow、PR normalization不一致、自processへのSIGINT/SIGTERM、読取後のinput/source変更|最初の理由をterminalに保存、部分結果を成功扱いにしない、自動retryなし|
|STOP（費用比較stageだけ）|B2 gate不合格、n32 pairing/seed不一致、n8混入、保存PR paired費用の非再現、eligible決定論候補の単価欠測|同上。shot診断だけのstageでは評価しない|
|未解決（候補を残してnull）|検証済み成分の欠測・非PASS、PR端点/closure/forward/referenceの不一致、UNDETERMINED、u/headroom guard、PR paired費用の一貫した欠測|候補を除外せず `UNRESOLVED_*` を記録。その候補を含む集合の最良・PR比・削減率はnull|

部分集合の最良を「全候補最良」と呼ぶ出力は作らない。旧32集合と強化184集合は別々に判定し、例えば追加候補だけが未解決なら旧32の比較は残り184側だけnullになる。

## 4. 費用比較stage（今回未実行）

費用は `G(P=0) = N_real*RZ_cosine + N_imag*RZ_sine`（state preparation除外、full measured wrapper）を正確な有理数で比較し、同値はcell IDの辞書順。単価の優先順位と区分は次の通り。

1. **B2実測**：B0 p15/q1 S4・B0+I p21/q8 S4の登録4軸。B2 recovery SOURCE `2e604e4a…` の完了terminal・全4軸PASS・native exit0・全reap・監視gap≤5秒・最終gate PASS・witness native wait0を全て要求する。
2. **B1登録anchor実測**：B0 p14/21×q1/8、B0+I p14/21×q1/2/4/8の24軸（来歴・profile・compiler結合）。
3. **full PF旧N=1実測**：保存wrapperの探索replica0。元runtime profileはunknownのまま後付けしない。旧B0のwrapper単価は来歴未解決のため使わない。
4. **B1固定model推定**：S2 `a+b*q+d*prefix*q`、S4 `a+3*b*q+3*d*prefix*q`（model SHA `8420b9f2…`）。prefix14〜21外は外挿flag。S4はB2全PASSの結合時だけ使い、その場合だけ `S4_model_status` を登録4軸限りの検証済みとする（全grid精度保証にしない）。B2未結合では未検証のまま使わない。再fitなし。

PRは確認n32の32 paired trajectoryを非相殺shotで重み直し、`G`は正確な標本平均、SEはpaired標本SE（信頼区間・母平均優位ではない）。通常biasのSEを流用しない。
PR削減率は `100*(1-G_PR/G_det)`、旧32と184の差をpercentage pointsで出す。
B2 gateは最初の会計・model予測・候補選択より前に評価する。B2が途中STOP・部分PASS・欠測なら未検証S4 modelで補わずSTOPし、費用比較を一切出さない。

## 5. PR確認union6の証拠照合（実保存値のmetadataのみ）

[監査結果JSON](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/evidence_audit_v1.json)／[候補別CSV](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/PR_confirmation_evidence_v1.csv)。
監査器は和・差・絶対値・対数・shot・費用・SE・順位を保存科学値に適用せず、科学値を書き出さない（出力中のfloat0を確認）。wall約0.30秒、peak RSS約73 MiB。

|項目|6候補の結果|
|---|---|
|membership|selection・resource-map completion・D0確認scope・wrapper確認行の6候補が一致|
|signed成分|outer_PF・finite_RTE・total_signedが両軸とも保存（有限値型）。complex modulusの `absolute_parts` は使わない|
|端点・forward|signals/independent_forward_signalsのraw・corrected・exact_tail、referenceを保存|
|normalization・u|b・log_B・normalization_log_discrepancy、EMPIRICAL/uncertifiedのu real/imag・log_B_margin、`numerical_allowance_certified=false`|
|登録・seed|各32登録（replica0〜31）、seedは固定H8 seed式から再導出一致、expected_signalは保存correctedと同一|
|compiled費用|各64行＝32 paired trajectory、symmetric-directional、compiler `1239304c…`、source `c31ef424…`（許可集合内）、seed一致、6 metric非負整数、fingerprintあり|
|探索n8|各16 symmetric行（K2/q2はordinary control2行も）を別に保持し、runnerは使わない|
|保存D0会計|各10行（5ε×倍率1/10）、確認phase/n32、DERIVED行のpaired_samples32|
|判定|6候補とも非相殺shot・paired総費用・SEを将来算出するデータは揃っている|

来歴joinは16項目すべて一致：D0契約の入力receipt4件、selection digest・signal digest、確認前固定/非pooling、resource-map完了のSOURCE/seed/input/compiler/union、D0 terminal完了と会計receipt、D0 provenance、旧B0+I会計・signal receipt、B1 terminal/model digest/native exit、B2保存B0登録2軸、signed run01のraw SHA・共有入力8 role、Phase A gzip原bytes。
入力22 roleの7,099,660 bytesは全てSHA一致。ここでの「一致」は来歴・schema・同一性の確認であり、PR成分の数値closure・端点差・forward一致・`log_B=R*log(b)`・保存費用の再現は**未実行**（runnerの承認後gate）。

希少事象：保存Taylor次数histogramは確認0次3,580/2次4、**4次は0観測（UNOBSERVED）**、探索は0/2/4次＝5,024/15/1。未観測層の費用はnull、母平均上界は未証明のまま維持し、費用0とみなさない。

## 6. 決定論184候補・費用入力の照合

- 184候補の保存recordあり、signed run01の368軸PASS（復元24軸）、成分名はB0/B0+I＝discard/PF、full PF＝PFのみ。
- 184候補すべてにu real/imag・log_B_margin・log_B（EMPIRICAL/uncertified）。保存通常会計は1,900行ちょうど（欠測・余剰0）。
- 単価入力：B1 anchor24軸、model4層（S2 holdout/training PASS、S4は未検証仮説）、full PF8候補とも探索N=1 pair、B2保存B0 2軸。
- B2：attempt02は登録B0+I cosineの受入fileを残したが、driver terminal `STOP`（`DRIVER_STOP_SIGNAL_OR_DEADLINE`）、監視 `MEMORY_PRESSURE`、driver native exit −9、witness native exit1、最終gate fileなし。B2 processは不在。よって**費用比較の入力は未完備**。

## 7. 人工検証

[限定検証記録](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/limited_validation_v1.json)。新54件を固定Python3.12.3で、移植性確認として既存Python3.11.0rc1でも実行し両方PASS。既存回帰113件（signed38・証拠監査30・凍結D0 33・H8 D0 adapter12）もPASS。
人工の端点・shot・wrapper・receiptだけを使い（候補IDだけ登録fileから読む）、相殺と非相殺、軸別適格性、shot整数式、倍率1/10、guard、通常会計再現gate、欠測・非PASS・非finite、旧32/184の全集合null、候補選択とexact tie、PR n32 pairing/seed/n8非混入/SE、paired費用欠測をnull（0ではない）とすること、B2の未完了・部分・2%・exit・gap・SOURCE・refitでの拒否、adapter来歴改竄13種、draft/grant/stage/環境/source/manifest/出力/fresh gate拒否、runner end-to-end（shotのみ・B2結合費用stage）、one-shot再使用拒否、STOP時lock保持・結果非出力、自process SIGTERMを確認した。
公開した実際の未承認planと無効draft grantを固定runner CLIへ渡し、入力読取・出力作成の前に `EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED`（exit2）で拒否されることも確認した。
これは主担当のローカル検証であり、独立review・CI・外部再現ではない。GPT/Claudeの追加レビューとsubagentは0。

## 8. 実行契約・資源

[未承認plan](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/execution_plan_UNAPPROVED_v1.json)（digest `c9bc5205…`）と[無効grant](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/grant_INVALID_DRAFT_v1.json)。
draftは `approved=false`、`runtime_authorization=false`、stage空、入出力・B2 receipt root未指定、CPU未予約、command null。B2依存は `binding_complete=false`（受領予定6 receiptのpathだけ固定、bytes/SHAはnull）。
将来の実行には、SOURCE9 fileのbytes/SHA/Git blob、manifest一致、D0 interpreter fingerprint、排他output・one-shot lock、300秒以内のfresh gate（source・環境・資源・B2非干渉）、明示承認文言を結合した別grantが必要。費用stageはさらにB2全PASS receipt6件のbytes/SHA結合を要求する。B2が別SOURCE・別試行で完了した場合は自動差替えせず、planを再結合する。

処理は約1,900行の凍結rule評価と60件のpaired metricで、入力を一度parse/index化し成分・単価をcacheする。1 processで十分なためworker poolを作らない。CPU/RAM/worker/wallの独自上限は設けず、出力64 MiB・4 inodeは空き容量の受付見積もりに過ぎない。CPU affinity・共有環境・他jobは変更しない。

## 9. 保全・非干渉

[保全receipt](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/preservation_receipt_v1.json)。B2 attempt02の保護対象692 files・127 worktreeのHEAD/statusを作業前後に読取専用（Git optional lock無効）で照合し全て不変。B2保護外の既存4 worktree・signed run01/準備・bias監査・Phase C準備の240 filesも不変。
B2のSTOPは `/sys/fs/cgroup/user.slice` のfull PSI avg10=0.18（全利用者の上位slice）で発生した。同じ記録で本作業も属するuser-30038.slice・session scopeのfull PSIはavg10=0、累積total（1,792/1,793 µs）は作業前16:06の読取値から増えていない。新worktree作成はSTOP後の16:15:18 JSTである。本作業がB2 STOPの原因になった証拠はないが、因果を完全に否定する測定ではない。
B2の領域には書込・signal・lock・rename・affinity変更を行っていない（読取は状態field確認のみ）。signed検証結果・B2 STOP/SOURCE/raw/grant/ledger/marker・M1-S・既存契約は変更していない。

準備の終盤（16:43:36 JST）に別sessionが新しいB2 recoveryの保全snapshotを作成し、このworktreeのHEAD `d567e6e1…` とstatus hash `7ae421d5…` も保護対象に記録した。
その後のB2保全gateを妨げないよう、このworktreeのHEAD/statusは固定したまま、資料commitは一時indexで作成してrefとして公開し、local branchとworktreeを動かしていない。
そのため、このworktreeでは資料filesが未commit差分として残るが、内容は公開した資料commitと同一である。別sessionのfile・worktreeは変更していない。

## 10. 未解決事項と次の担当

- PR成分の数値照合、通常会計・paired費用の完全再現、非相殺shotは、承認後runnerのgateとして初めて実行される。
- 費用比較にはB2登録4軸の完了PASSが必要。現在の受領可能なB2証拠は旧run01の保存B0 2軸だけ。
- 共有reference/scalar、旧native projection8/12、prefix19元runtime SHA欠測16候補、full PF旧profile unknown、PR希少次数未観測、経験的u非証明の限定を保持する。

**ユーザー**：(1) B2をどう扱うか（同契約での新試行・資源条件の見直し・費用比較を保留したままの診断のみ）を判断する。(2) 保存値での非相殺shot診断（`NONCANCELLATION_SHOTS` stageのみ）の一回実行を、このSOURCE・plan digestに対して別途明示承認するか判断する。
**Codex**：B2 attempt02のSTOP記録と原因調査を、既存B2の担当範囲で行う（本作業はB2領域を変更していない）。
**Claude Code**：承認があればfresh gateとgrantを結合して指定stageを一回だけ実行し、結果/欠測/STOP/native exitを保存してmandatory STOP。費用stageはB2全PASSの結合とその明示承認の後に限る。
今回の準備でmandatory STOP。

## 修正追補：レビュー指摘2点（2026-10-17）

利用者の確認依頼に対するCodexの読取レビュー（16:51 JST）が、SOURCE `d567e6e1…`・plan `c9bc5205…` について次の2点を指摘した。科学条件・固定契約は変えず、表示と出力の範囲を直した。
[修正receipt](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/review_fix_receipt_v1.json)、[動作差](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/behaviour_diff_v1.json)、[整合性](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/consistency_check_v2.json)。

|指摘|該当（修正前）|問題|修正|
|---|---|---|---|
|1. PR保存paired費用の再現gateの説明|`src/trotterlib/h8_noncancellation.py` `costs()`（l.476付近、`COST_COMPARISON` 内だけで実行）|本文とplanが、この再現（とn32 pairing/seed検査）をshot診断だけでも実行するgateのように記載し、shot診断だけの結果にも未実行の記録がなかった|結果の `gates` に `PR_saved_paired_metric_reproduction`・`PR_confirmation_n32_pairing` を常に出し、`stage=COST_COMPARISON`、shot診断だけでは `executed=false`・`NOT_EXECUTED_IN_NONCANCELLATION_SHOTS_STAGE`。plan v2はgateをstage別に列挙。本文の該当2箇所を訂正|
|2. S4の検証済み表示|`scripts/h8_noncancellation_contract.py` `assemble()`（l.306）|`S4_model_validated_on` がB2未PASSでも無条件に設定された|来歴欄は `S4_model_validation_scope=PLANNED_FOUR_REGISTERED_B2_AXES_VALIDATED_ONLY_WHEN_B2_FULL_PASS_BOUND`。実行時の `S4_model_status` はB2全PASS結合時だけ検証済み、それ以外は `UNVALIDATED_B2_FULL_PASS_NOT_BOUND_S4_MODEL_NOT_USED`|

両指摘を再現する人工回帰testを2件追加し、修正前のcodeでは失敗（KeyError 1、assert失敗 1）、修正後はPASSした。
人工データで修正前後を比較すると、1,900会計行・380軸envelope・費用stageの値は完全に同一で、追加されたのは上記2つのgate記録と `S4_model_status` だけである。
既存54件＋新2件＝56件をPython3.12.3と3.11.0rc1で、既存回帰113件をPASS。plan v2・無効grant v2・source closure v2を新SOURCEへ結合し、science・accounting・bias区分・B2依存・入力manifest・凍結D0 fileがv1と同一であることを確認した。v2 draftは固定runnerで入力読取前に拒否（exit2）。v1のplan・grant・receiptは履歴として上書きしていない。

保護対象のworktreeを動かさないため、修正はGit refを共有repositoryに作らない独立clone上で行った。元worktreeのHEAD `d567e6e1…`・status hash `7ae421d5…` は変更していない。
この作業中、レビュー原文を探す広域grep（17:00:55 JST頃開始）がsession cgroupのmemory stallを起こし、並行していたB2 concurrent-preservation attempt01が17:01:35にMEMORY_PRESSUREでSTOPした。また準備段階の16:38〜16:40には、この準備worktreeへのcommit・file追加が別sessionのB2 sine-spool準備attempt01〜03のworktree保全gateに触れ、STOPさせていた（grant・dispatch・compileは0）。詳細と未確定の関与は[保全receipt v2](../../artifacts/resource_applicability/track_a_h8_noncancellation_readiness/2026-10-17/preservation_receipt_v2.json)に記録した。

**ユーザー**：修正後のSOURCE `cbfd7cac…`・plan v2 `002f9edb…` に対する `NONCANCELLATION_SHOTS` 一回実行の承認を判断する。B2側のSTOPの扱いもB2担当とともに判断する。
**Codex**：B2担当として、上記のB2 STOPを既存の承認範囲で扱う。
**Claude Code**：承認があれば、その範囲だけを一回実行してmandatory STOPする。今回の修正はここでmandatory STOP。
