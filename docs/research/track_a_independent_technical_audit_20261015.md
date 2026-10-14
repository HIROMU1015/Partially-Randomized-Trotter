# Track A：H8 D0・H6 native B0+I 独立技術監査（2026-10-15）

## 判定と入口

**固定実装に6件の制御不具合を人工再現し、独立branchで最小修正した。新22件を含む限定75件はPASS。今回の静的・人工範囲で、再現済み不具合の未修正残件は0。科学レビューへ戻し、mandatory STOPする。**

H8は旧値完全一致の先行gate・標本分離・paired covarianceを人工検証した。H6は保存binary64モデルのfull-Fock identity・scalar/phase規約を静的・人工検証した。これは実H8旧値一致、実H6 native回路・費用、厳密なnumerical certificate、実行認可の判定ではない。

- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 監査branch：`track-a-independent-technical-audit-20261015`
- 起点HEAD：`2ec66aae5e17c9329fcea8ad962be27130980d31`
- 監査対象SOURCE：`60eeb51c148007359cf6e8bcf6288e6837a39701`
- **修正SOURCE：`20f89bc32b161ee22e4a1ed709f9e3414c19633a`**
- 旧D0科学SOURCE：`fa75c7997e135da7b7a0c4e23949e47252bd7a2e`、H8結果：`681ec23c93fdd1090d325908f4b155a79115afdf`（ともに不変）
- [監査bundle入口](../../artifacts/resource_applicability/track_a_independent_technical_audit_v1/2026-10-15/README.md)：finding、独立担当所見、red/green logs、source closure、旧/new blob・bytes/SHA、未認可binding更新索引。

監査担当はrootによる実行制御監査と、著者会話・設計結論を引き継がない別Codex担当 `/root/audit_independent_h8_h6`。別担当は数値/bridge・契約を独立読取し、新13testsを作成、監視diff/red証拠も確認した。修正authorはroot、修正評価は別担当である。テストprocessは親が一括管理した。CI・外部再現・GPT詳細科学レビューではない。

## 固定対象と公開来歴

originの起点branch HEADは指定40文字SHAと一致した。固定GitHubから主要コード・契約・資料、旧D0コード・H8 completion metadataの計16fileを実取得し、local Git objectとのbytes/SHA一致を確認した。[取得receipt](../../artifacts/resource_applicability/track_a_independent_technical_audit_v1/2026-10-15/github_reference_receipt_v1.json)に固定URLとhashを保存した。

対象は `h8_d0_saved_execution_v1.py`、`final_pre_review_execution_v1.py`、`launch_h8_d0_saved_once_v1.sh`、`pre_review_h8_d0_adapter_v1.py`、`h6_native_b0i_bridge_v1.py` とそのfrozen D0/math依存、既存H8/H6 draft契約。共通指示原本2026-10-12はSHA `709524f856a3117c85042309c390a3697d9b2c8c8d79af6b6c290e9d20814bcf` の全文を参照した。

旧契約・旧SOURCE・既存worktreeは保全した。旧draftのSOURCE/hashは当時の履歴として残し、今回sourceとの差を[候補binding更新索引](../../artifacts/resource_applicability/track_a_independent_technical_audit_v1/2026-10-15/candidate_binding_update_v1.json)へ記録した。新grant・seal・起動commandは作らない。将来の認可には修正SOURCEからcontractを再結合する必要がある。

## CONFIRMED_DEFECT：再現・最小修正

| ID | 再現した問題 | 最小修正と検証 |
| --- | --- | --- |
| F01 | H8 cosine/sineでseed/event/compilerが同じでも、allowlist内の別SOURCEを一方に与えると受理された。 | adapterで `(phase,cell,replica,control)` ごとのSOURCE一致を要求。探索/確認の片軸変更は拒否、両軸が同じallowlist内SOURCEなら受理。旧D0は変更しない。 |
| F02 | H6 prepare済requestをcopy、または`built=False`へ戻すと、同grantでnative builderをもう一度呼べた。 | grant側にcaseごとのbuild予約を持ち、provider呼出前に消費。copy/resetとprovider失敗後の再試行を拒否。 |
| F03 | H6 production分岐はproviderのmodule名だけを検査し、未検証の同名objectでもbuilderへ到達した。 | source/Git/origin照合を通った`native_provider()`のobjectをgrantへ記憶し、buildで同一性を要求。検証成功/同名impostorを全mockで確認。 |
| F04 | cgroupの必須OOM counter欠測が`.get(...,0)`で0扱いされた。NaN PSIも比較を抜けた。 | 新guardで必須`oom`/`oom_kill`・非負整数counter・有限0〜100 PSIを確認。任意`oom_group_kill`の有無はbaselineと整合させる。frozen D0 policyは不変。 |
| F05 | quotaのavailable bytes/inodesがNaN/∞でもcheckerを通過した。 | quota availabilityを非負整数、観測時刻を有限値に限定。receipt欠測/staleは従来どおりSTOP。productionのdefault quota経路はreceipt接続がないためSTOPを維持。 |
| F06 | 観測IO4秒の後にcheckpointが4秒欠けても、end-to-startの計測は4秒として許容した。 | start-to-startで8秒を記録し、5秒超でSTOP。観測所要時間は別欄で保持する。 |

F01/F02/F03/F04/F06は契約・再実行・監視の重大な保護経路、F05は将来のquota receipt接続時に現れる条件検査不良である。今回の再現は人工変異であり、既存実結果がこれらの不具合で汚染されたという認定ではない。

変更codeは3fileだけ。旧D0/shot関数、math prototype、native scientific provider、科学条件・comparison・uの意味・資源上限・carryは変更していない。H6の新予約はprocess内のhelper防御であり、将来runnerに必要な永続one-shot lockを代替しない。

F01は別SOURCEを科学的に同等と認定するaliasではない。将来の実入力にresumeをまたぐ異SOURCE pairがあれば旧値解析前にSTOPする。今回、その有無は実データで調べていない。

## NO_MATERIAL_ISSUE_FOUND：新たな独立証拠

H8について、確認mapの**最後の行**の値改変、行欠落、ineligible候補の旧precision改変を別々に与え、ε=.0005のaccountingへ一度も到達しないことを検証した。944 precision/3576 costの実一致は未実行だが、旧値検査の対象漏れや部分PASSによるPhase B遷移は今回の人工範囲で認めなかった。

探索92と確認union6＋det20は別scope、B2 sample8/32は別phase・別seedである。人工の反相関したcosine/sine費用からpaired totalのSE=0を独立算術で確認し、axis SEを独立に足してcovarianceを落とす経路ではないと確認した。旧12 adapter testsと既存14 driver testsも標本混合・seed変異・missing cost・metric選択・固定RZ pairと全候補envelope・union外表示・UNOBSERVED/nullを確認した。費用未取得を0と置く変更はない。

H6では2mode全occupation basisを直接構成し、signed complex Hermitian Gについて λ/4[(Tr G)²+Tr G²] とfull-Fock traceが一致した。固定native providerを静的に確認し、final ancilla `p(-scalar*T)`一度、sine wrapperのSdg、T=.8/S2/q規約は旧H4意味と整合する。元B0のconstant/one-body/retained blocksを保持し、discarded tailだけを足す。保存corrected/exact-truncatedは位相回転し、full-H referenceは不回転である。

これらは実12mode native buildやcompiled fingerprintの確認ではない。H4のcompiled Δ=0は、H6のΔ=0へ昇格しない。

## 人工検証と資源実績

- 修正前：新9監視testsの4methodsで11subtest failures、独立数値10testsの4methodsで5subtest failuresを再現。これらは意図したred証拠で、実計算の失敗ではない。
- 修正後：**新22＝監視9＋独立数値13、既存41、変更adapter旧12、合計75/75 PASS**。既存23 prototype testsのうち無変更math側11は再実行しない。
- 最終一括回帰wall **16.08984176442027秒**、peak RSS **81,489,920 bytes**、process1/thread1。
- AS512MiB/CPU120秒/wall180秒、fixture file64MiB/log1MiB/scratch512MiB/5,000inodeの限定計画。すべてhome内exclusive scratch、teardownで削除、既存PIDへのsignalなし。
- NumPy/Qiskit/CuPy/PySCF/trotterlib/trottertracksの実importを禁止するfinderを付けて実行。provider/source/Git/ioの成功経路はmockのみ。`bash -n` と `git diff --check` PASS。
- 新SCF/DF/state/signal/trajectory・実再会計・実native build/compile・新実grant・affinity操作：**すべて0**。

test名・全log・actual資源とhashはbundle内に保存する。独立testsは [数値/bridge](../../tests/test_independent_track_a_numeric_audit_v1.py) と [監視/authorization/exit](../../tests/test_independent_track_a_execution_audit_v1.py)。既存fixtureは足場として使い、新しい破壊的変異・oracle・assertionを別担当が追加した。

## 残る制約と分類

| 分類 | 残る事項 | 判定への影響 |
| --- | --- | --- |
| SCIENTIFIC_ASSUMPTION | `64ε`/floorによるbinary64/libm phase rotation allowanceと、旧empirical uを各diagnostic contributionへ使う仮定は真の誤差上界の証明ではない。 | `EMPIRICAL`/`certified=false`を保持。科学的採否はGPTへ戻す。 |
| SCIENTIFIC_ASSUMPTION | `stored_model_coefficient_error_upper=0`は保存binary64係数を対象モデルとした入力係数誤差0。 | 元の化学HamiltonianやDF近似誤差0を意味しない。Fractionで算出した保存係数算術誤差は別に保持。 |
| RUNTIME_ONLY | H8実入力での全旧precision/map一致、新精度の実結果、sample SOURCE pair、actual資源・quota/profile。 | 今回は実科学データを数値解析しない。別承認されたone-shotで一致不良ならSTOP。 |
| RUNTIME_ONLY | H6 snapshot→prepared→native providerの実接続、relative phase/numeric fingerprint、実費用Δ・wall/RSS。 | tiny mockだけで実6回路成功を認定しない。 |
| TEST_GAP / RUNTIME_ONLY | 長いC処理やblocking IO中、Python timerは5秒以内の非同期停止を保証しない。 | 戻った時点のstalenessは拒否する。SIGKILL/exitでは同親shell wait receiptが必要。即時5秒停止が必須なら、この実装だけでは満たせない。 |
| TEST_GAP | 実SIGKILL/CPU hard-limit退出・実長C stressは未実施。wait137・terminal欠測はmock検証。 | terminal欠測を完了証拠として扱わない。native wait/terminal/result receiptsを揃えて初めて最終成功監査する。正確なexit時刻は補完せずnull。 |
| TEST_GAP / future binding | quota-enabledのdefault実経路はfresh receipt接続未実装。H6にはまだbounded runner/永続one-shot/実cleanup接続がない。 | 既存draftは未seal・未認可。今回実装の完成度と、将来launch-readyを区別する。 |

欠測quotaでは入力mappingを一度も呼ばずFAIL_CLOSED STOPし、markerは消費されて再試行できない。partial artifactは完成receiptへ登録されず、first STOP理由は後続cleanup理由で上書きされない。native exitの生PIDだけを証拠とせず、start_ticks/boot/UID/同親を確認する。監査時点の資源観測や人工成功を将来fresh launchの証拠へ流用しない。

## 固定・公開・STOP

SOURCE commitは最小code3＋新tests2。REVIEW_BUNDLE側には監査資料・lightweight証拠・索引だけを含め、sourceを変更しない。[公開receipt](../../artifacts/resource_applicability/track_a_independent_technical_audit_v1/2026-10-15/publication_receipt_v1.json)でpush結果を区別する。全closure/主要fileのbytes/SHAと旧/new blob、旧73worktree HEAD/status保全を確認した。

候補状態は **approved=false、runtime_authorization=false、allowed_cpus=[]、execution_plan_sealed=false、absolute_launch_command=null**。既存grant・ledger・raw/runtime・科学SOURCE・他job・venv・共有設定を変更していない。科学採択・新実行承認・launchなしに次stageへ進まない。これ以上の準備campaignを自動開始せず、GPT科学レビューに必要な制約を渡して **mandatory STOP**する。
