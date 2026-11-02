# H8 Phase B1 run01：全4層PASS・費用model固定

**B1は正常終了。新B0保存8軸と既存B0+I16軸を再利用し、B0/B0+I × cosine/sineの4層を独立にfitした。B0の事前固定holdoutは両軸0%でPASS、B0+Iのtraining anchor残差も全件0%。新規compileは0。B2へ進まずmandatory STOPする。**

利用者の「その方針で進めて」を直前のB1実行方針への承認として受け、新scope grantを一件発行して一回実行した。元B0 run02の取得契約STOPと未記録thread違反processは、そのまま保持する。今回正常終了したのは、読取専用auditを通過した保存8軸を入力として明示承認された別のB1契約である。

## 科学条件と入力

H8直線鎖、隣接1Å、STO-3G、16 modes、Nα=Nβ=4/sector4900。保存tol-only DF rank28、tol1e-8、cutoff0、signed generation order、weighted Hermitian projection。T=.8、q1/2/4/8、δ=.8/.4/.2/.1、S2 anchorのL_D=p14/21。B0新取得はq1/8の4構成・8軸、B0+Iはq1/2/4/8の8構成・16軸。

controlはsymmetric-directional、native full measured Hadamard wrapper、state preparation費用除外。B0はoriginal constant一回/no tail identity、B0+Iはoriginal＋full-Fock tail identity一回で、層をpoolしない。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/topologyなし。compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12` を維持した。

元のfull-H未回転reference、経験的allowance factor1/10、厳密bias/PR母平均保証の未確立事項を保持。signal・bias・shot・PR sampling・SCF/DF/state・V2の新計算はしていない。

B0は[保存8軸のaudit](track_a_h8_b0_saved_anchor_audit_20261017.md)を再照合して採用。B0+Iは旧B1台帳と公開成果・16原cost receiptのSHA、実測RZ、fingerprint before/after、compiler、worker exit0を結合した。旧B0の来歴未確立16軸は使用していない。旧B1 guard/台帳を緩めず、新しい24軸の契約を実装した。82入力のbytesを実行前・driver・終了後に照合した。

## 固定したfitと検証結果

S2は `a+b*q+d*prefix*q`。標準libraryのFractionによる有理数最小二乗で、各層の3列設計rank3を確認した。H6係数・既存modelの移送はない。

| 層 | fitに使用 | 独立holdout | (a,b,d) | 検証 |
| --- | --- | --- | --- | --- |
| B0/cosine | p14q1/p21q1/p14q8 | p21q8 | (1592,1314,25271/7) | holdout誤差0%、PASS |
| B0/sine | 同上の別軸実測 | p21q8 | (1592,1314,25271/7) | holdout誤差0%、PASS |
| B0I/cosine | p14/21×q1/2/4/8の8点 | 新設なし | (1592,1314,25271/7) | 全training anchor残差0%、PASS |
| B0I/sine | 同上の別軸実測 | 新設なし | (1592,1314,25271/7) | 全training anchor残差0%、PASS |

B0のholdoutは各軸で予測618,608 RZ/軸、保存実測618,608 RZ/軸で一致した。holdoutはfitに入れていない。B0+Iの0%はtraining anchor残差であり、新しい独立holdout検証とは呼ばない。4層は独立fitの結果として同じ係数になったもので、共通係数を前提にpoolした結果ではない。

各診断は `abs(predicted-measured)/measured < 1/50` をPASS、2%以上・非正予測・rank不足・資源/来歴不整合は次軸へ進まずSTOP。今回はすべてPASSし、refit/retry/再分割/追加候補を行っていない。独立holdout一条件は全prefix/qやS4の費用誤差保証を与えない。

## S4仮説とB2への固定出力

S4は未検証のschedule-composition仮説 `a+3*b*q+3*d*prefix*q`。scalar/wrapperを一回だけ残す元の非merge native条件を維持するが、compiled費用の加法性はまだ実測していない。

| 事前登録B2候補 | 軸 | 固定予測RZ/軸 | 区分 |
| --- | --- | --- | --- |
| B0 p15/q1 S4 | cosine/sine各独立 | 1175933/7 | model推定・未実測 |
| B0I p21/q8 S4 | cosine/sine各独立 | 1852640 | model推定・未実測 |

model canonical SHAは `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。予測4行も同じSHAへ固定した。B2のcompile authorizationは全行falseであり、B2は開始していない。5精度の最良候補・PR費用削減率・総shot費用の再会計も行っていない。

## 実行・資源・人工検証・保全

実行SOURCE：`6fb8a0a71986fda571d82b5658686f5e74a32f8b`、19 SOURCE filesを固定して公開・remote照合後に起動。grant scope：`H8_B1_REUSED_B0_RUN02_B0I16_FIT_V1`。古い取得grantは変更・再使用していない。

新13＋旧readiness8＋保存audit14の計35人工testsが固定Python3.11.0rc1でPASS。人工の正常系とholdout不一致STOPを実OSのdriver/observerで確認し、不合格時の次軸抑止・全回収・one-shot再使用拒否、厳密2%境界、欠測/来歴拒否を検証した。主エージェントのローカル検証であり、独立レビュー・CI・外部再現ではない。

計算本体は小さい4層の有理数演算を1 process/thread1で実施し、外部observer1 process/thread1を配置。fitのAS512 MiB/RSS256 MiB/CPU240秒/wall600秒は旧案を維持。observerはAS512 MiB/RSS256 MiB/CPU120秒、合計RSS512 MiB＋reserve512 MiBを新契約に明示した。CPU0/10の別物理coreを固定し、raw 5%以内のfresh gate、quota・disk/inode・PSI/OOM・完全runtime34,273 filesと保全gateを通過した。自動CPU再割当なし。

保存receiptによるdriver内fit wallは0.09213536605238914秒、外部監視区間は1.0120194554328918秒。これらはSOURCE/input監査・実装・公開作業を含む総作業時間ではない。driver自己観測peak RSSは61,411,328 bytes、外部observerの記録peak aggregate RSSは193,925,120 bytes。短い処理なので自己peakと外部観測値を区別し、外部の連続完全捕捉を主張しない。driver/launcherとも実native exit0、全owned process回収・cleanup errorなし、最終SOURCE/input/runtime/保全gate PASS。

実callはfit4、B0 trainingチェック6＋holdout2、B0I anchorチェック16、B2予測出力4。新compile/signal/再会計/SCF/DF/state/PR/S4/B0I compileは0。旧681保護files、119既存worktreeのHEAD/status、旧V2 untracked、元B0 run02 raw/STOP/grantと今回config/grantを保持した。

## 次の境界

次は別承認のB2：事前登録S4最大4軸を実compileし、各軸2%以上なら即STOP、自動refit/追加compile/retryなし、既定のcompile並列1を維持する。その後のC再会計も独立承認。今回のB1 PASSからS4の適合やPR利益を先取りしない。V2 CLOSED_NO_GO・V1/H5/H7を維持する。

[公開evidence/checker](../../artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/README.md)をcommit・pushし、remote SHA確認後mandatory STOP。
