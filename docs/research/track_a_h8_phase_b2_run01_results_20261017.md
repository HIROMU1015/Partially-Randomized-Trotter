# H8 Phase B2 run01：契約STOP・自動再実行なし

**H8/1Å/STO-3G/DF28/T=.8、S4の固定4軸を直列・一回実行。実compile entry3、保存判定2軸（PASS2）。SOURCE244c45e7…、model8420b9f2…を固定。STOP理由は結果報告と原terminalに保存（MEMORY_PRESSURE）。B0I cosineは中断、sine未起動、未settle予約1を保持し、後続compile・retry/refit0。driver native exit−9、launcher報告exit1（外側native wait欠測）、全owned process回収。旧684保護files/123worktree・B0原53/B1原34 rawを保持。新fit/signal/再会計0、元B0 STOP/V2 NO_GOを維持。C未承認、公開後mandatory STOP。**

利用者の明示承認を受け、公開済み準備commit `3a450af0fb431269ac94152807992883c1490b26` の契約に従い、一つの新scope grant・一つの未使用実行領域で一回起動した。準備時の無効grant、既存grant/ledger/markerは変更・再使用していない。結果用branchは `track-a-h8-phase-b2-run01-results-20261017`。計算終了後に別worktreeを作成し、実行元readiness worktreeを保全した。

## 条件と実測結果

H8直線鎖、隣接1Å、STO-3G、16 modes、Nα=Nβ=4/sector4900。保存tol-only DF rank28、tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。T=.8、q1/8、δ=.8/.1、L_D=p15/21、S4は負時間中段を含む非mergeの3 S2段。state preparationを除くnative full measured Hadamard wrapper、symmetric-directional controlを維持した。

| 順序 | S4候補 | 軸 | 固定予測RZ/軸 | 実測RZ/軸 | 誤差 % | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | B0_p15_q1_4th | cosine | 1175933/7 | 168353 | 0.2153638062 | PASS |
| 2 | B0_p15_q1_4th | sine | 1175933/7 | 168353 | 0.2153638062 | PASS |
| 3 | B0I_p21_q8_4th | cosine | 1852640 | 未完了 | — | 未判定 |
| 4 | B0I_p21_q8_4th | sine | 1852640 | 未起動 | — | 未判定 |

予測はB1正本 `83a3c4a6688e66576f85471bf41a45e29cbfb705` の凍結modelから保存された値であり、実測列と区別する。model canonical SHA：`8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。今回再fit、H6係数移送、軸のpool、shotを掛けた総費用計算は行っていない。

各軸は `abs(predicted-measured)/measured < 1/50` のときだけPASS。丸めた予測や軸平均を判定に使わず、2%以上は即STOPとした。1軸のnative exit0・回収・原結果SHA・wrapper fingerprint・compilerを確認してから判定を保存し、PASSした場合だけ次軸を投入する。同時compileは1、候補の追加・retry/resumeは0。

B0はoriginal constant一回/no tail identity。B0+Iはoriginal constantと保存p21 full-Fock Cを各一回使用し、Cを新計算していない。保存full preparationからp15/p21 prefix viewを復元し、live basis操作・順序・diagonal・constantをmetadataへ照合。p15の未保存root hashを捏造せずview recipeに結合し、p21は既存discard rootにも照合した。

native controlは左branch identity、右branchと保存corrected/independent-forward signal、X/Y wrapperの一致を1e-8未満で判定する。保存control receiptはbundle内に収録した。これは承認されたB2の限定正しさ検証であり、新signal探索・新reference・PR samplingではない。

## 起動gate・実行と保全

実行SOURCE：`244c45e752fa703e286d40ece9ad641430ea9c6e`。21 SOURCE、265 provider、34,273 runtime、15入力を照合。providerは `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・backend/topologyなし。compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12` を固定した。

起動直前とdriverの完全binding照合後のfresh gateで、SOURCE/入力/profile、別3物理core・SMT siblingを含むraw busy5%以内、CPU quota、実効空きRAM80.25 GiB以上、disk16 GiB/inode40,000以上、PSI/OOM、保全を確認した。driver/worker/observerは各thread1、compile worker最大1、AS/RSSはdriver64/32 GiB・worker64/32 GiB・observer512/256 MiB、aggregate RSS64.25 GiB＋reserve16 GiB。wall12時間/累積CPU24時間の上限を維持し、自動CPU再割当はしていない。

外部監視wallは9345.687614秒、観測peak aggregate RSSは44275134464 bytes（約41.234 GiB）、観測最大gapは4.665690秒。これは観測区間内の値であり、連続peakの完全捕捉や実装・監査・公開を含む総作業時間ではない。driverの実native exitは-9、中断workerはexit2、全owned process回収・cleanup errorなし。外側launcherのnative wait receiptは欠測し、launcher admissionが報告したexit1と区別する。現在のPID不存在から外側exit値を復元せず、欠測原因も特定していない。

終了状態：`STOP`。STOP理由：`RuntimeError:MEMORY_PRESSURE`。raw terminal・reservations・worker native receiptをprivate原本のまま保持し、公開projectionではlocal絶対pathを伏せた。公開SHA inventoryは原本のbytesに対するSHAであり、redacted projectionのhashとは別である。有効grant本体・private入力・cacheは公開しない。

旧684 grant/ledger/carry/marker等、123旧worktreeのHEAD/status、旧V2 untracked、元B0 run02の53 raw、B1の34 raw、今回config/承認grantのbytesを照合した。元B0 run02のSTOPをPASSに変更せず、B1を再実行せず、別B1実装・M1-S・V1・V2 CLOSED_NO_GO・H5/H7を変更していない。

## STOP証拠と欠測

本日04:13:48 JSTに終了terminalが保存された。B0+Iのcontrol最大誤差は1.3360773021112734e-13でPASSし、03:56:02 JSTにcosine compileへ突入したが、完了結果を保存する前にMEMORY_PRESSUREでSTOPした。sineのworker・compileは未起動。2%不一致によるSTOPではなく、未完了cosineのRZやモデル誤差は不明である。

監視sourceのMEMORY_PRESSUREはhost full PSI avg10が1%以上、またはcgroup full PSI avg10が0を超えたときに返される。失敗時のPSI raw sample・対象group・数値は保存されていないため、どちらが成立したかや他jobとの因果を特定しない。観測peak aggregate RSS約41.234 GiBは64.25 GiBの契約上限内であり、RSS上限超過によるSTOPとは扱わない。

原driver terminalのcompile_entry_markers=0は、sealed sourceが旧wave_* globを参照する報告上の不具合。原本を変更せず、別auditで実axis_* marker3、完了/settle2、未settle予約1を確認した。compile予約や原call上限を返却・書き換えず、欠測1を残す。

外側launcherのnative wait receiptは存在せず、保存されたadmission reported exit1を実native waitと同一視しない。toolの旧sessionも不在、外側parentとlauncher/driver/3 workerのidentityは退出確認済みだが、外側exit値・receipt欠測原因は不明のまま残す。driverのnative exit−9、worker03 exit2、監視のall_reaped=true/cleanup_errors=[]はそれぞれ保存receiptの値である。

## 結論の範囲と次の境界

今回の実測判定は登録したS4最大4軸に限る。S2の独立holdoutとB0+I training残差の区別、保存未回転full-H reference、経験的allowance factor1/10、厳密bias/PR母平均保証の未確立事項を維持する。全prefix/qのモデル誤差保証、PR費用優位、5精度の最良対照を新たに確定した結果ではない。

Phase Cの費用再会計は今回未承認のため実行していない。新fit/再会計/SCF/DF/state/new signal/full-H reference/PR sampling/law/tail scalar/unregistered compileは0。成功・失敗によらず、このPhaseでmandatory STOPする。追加実行は別承認が必要。

既存準備の119人工testsと2 qubit S4接続11項目は準備commitの証拠として参照し、今回反復していない。公開checkerは保存bytes・metadata・終了/保全状態を確認するだけで、科学計算や実測誤差の再計算を起動しない。主エージェントによるlocal照合であり、独立レビュー・CI・外部再現とは呼ばない。

[保存結果・checker](../../artifacts/resource_applicability/track_a_h8_phase_b2_run01/2026-10-17/README.md)をcommit・pushし、remote SHAを確認後mandatory STOP。
