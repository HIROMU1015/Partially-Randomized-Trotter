# H8 B0：CPU gateの選択・診断・起動前STOP記録の修正

**CPU負荷の判定条件は緩めず、承認前のCPU候補選択と失敗時の証拠保存を修正した。新規compile・取得の再起動は行っていない。run01の失敗時CPU番号は未確定のまま保持する。**

branch：`track-a-h8-b0-anchor-cpu-gate-fix-20261016`。
起点：[run01 STOP commit `a6f76beccd009649a8a0ff932f36a8cc00dc83e9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a6f76beccd009649a8a0ff932f36a8cc00dc83e9)。
修正後の実行SOURCE：`db559a518214fd27fa96e2f1451911df05c4f4a8`。
[公開契約・人工検証・新しい無効grant案・保全証拠](../../artifacts/resource_applicability/track_a_h8_b0_anchor_cpu_gate_fix/2026-10-16/README.md)。

## 確認できたことと原因の限界

[run01](track_a_h8_b0_anchor_run01_stop_20261016.md)はfresh CPU idle checkで `PHYSICAL_CORE_BUSY`。原コードはどのCPUが何tick busyだったかを保存せず、科学driver起動前のterminal receiptも作成していなかった。このため、当時の失敗が外部job・割込・launcher自身の実行のどれによるかを確定できない。

今回の**現在時点の読取専用診断**では、旧割当のCPU0は1秒間にbusy17/99 ticks（約17.17%）、ほかの旧割当coreは0だった。診断processは前後ともCPU43、診断自身のCPU時間差は0 ticks。この観測はCPU0が常に空いているわけではないことを示す。前回もCPU0だったとの証明や、外部jobの特定ではなく、run01へ遡及補完しない。

## 修正内容

1. [CPU候補probe](../../scripts/probe_h8_b0_anchor_cpus.py)を追加した。grant発行前に固定3回×1秒のcounterを読み、全windowで5%以下だった物理coreから、最悪busy率・CPU IDの固定順で6coreの**候補**を出す。SMT siblingも含み、同一物理coreを二つ選ばない。最後のwindowだけを見て一時的負荷を隠さない。観測回数を自動追加して通るまで待つ処理はない。
2. [runtime](../../scripts/h8_b0_anchor_runtime.py)は選択core全件のbefore/after、8 counter差分、total/busy ticks、busy率、失敗CPUを保存可能なreceiptへ変更した。欠測・counter巻戻り・total0はUNKNOWNでSTOP。判定は整数演算で `20*busy <= total` をPASSとし、従来の5%境界を維持する。
3. [launcher](../../scripts/launch_h8_b0_anchors.py)は初回とdriver refreshのCPU receiptを、例外を送出する**前**に保存する。launcher自身のuser/system ticks・実行CPU・affinityも記録するが、自身の負荷をbusy値から差し引かない。科学import・driver forkより前でも失敗terminalを保存し、停止点、path状態、driver/予約journalの有無を区別する。
4. initial fresh観測前にlauncher自身だけを契約上のobserver CPUへ固定する。初回は従来どおり1秒、driver refreshも1秒に統一し、旧.25秒windowの量子化影響を減らす。raw5%判定と全6core/SMTの検査は維持する。他processや他jobのaffinityは変更しない。
5. 既存のscience lockより前に、新しい `launcher_admission01` を作り、一度fresh gateで失敗した実行域も再利用を拒否する。config/grant・既存receiptの上書きはしない。STOP後のCPU自動再割当・retry/resumeは追加しない。

新しいterminalの `reported_exit_code` はlauncherが報告する戻り値であり、外部からwait4で観測したnative exit codeとは区別する。run01の未知exit codeを今回の値で埋めない。

## 維持した科学契約

linear H8、隣接1 Å、STO-3G、16 modes、Nα=Nβ=4、sector4900、保存tol-only DF28（tol1e-8/cutoff0、signed generation order、weighted Hermitian projection）、T=.8、q1/8、δ=.8/.1。S2 TRAIN p14/q1・p21/q1・p14/q8、HOLDOUT p21/q8、各cosine/sine、計8軸。最大4 worker・2 wave。

候補、TRAIN/HOLDOUT、native adapter、original scalar一回、control、compiler、preparation、call caps、資源上限は変更していない。6物理core・実効空きRAM176.25 GiB・完全profile・input gateが必要である。B0+Iは既存16軸の再利用、旧B0は混合せず、H6係数を移送しない。取得後のB1 fit/holdout、B2 S4、C再会計は別承認で、それぞれmandatory STOP。

runtime完全profile `622ce4a8…` の34,273 files、provider `c31ef424…` の265 filesとinventory digestを維持する。新SOURCE closureはCPU候補probeを含む15 files。新しいconfig・無効grant案へSOURCEを結合し、source15/provider265/runtime34,273/input8をbyte照合した。無効grantでは副作用前に拒否され、fresh gate・admission/control/output/lockは作成されないことも確認した。

## 検証・現在の状態

- 新[CPU gate tests](../../tests/test_h8_b0_cpu_gate.py) 11 methods、既存execution17＋計画10 methodsがPASS。5%ちょうど、5%超過、欠測、self負荷非控除、SMT、非独立core、過去windowのbusy、起動前CPU失敗の保存、同じ試行の再利用拒否、既存資源/候補/call capsの不変を検査した。
- 実adapter/compilerは変更しておらず、新たな人工transpileも行っていない。以前の2 qubit fixture結果と今回のCPU guard検証を区別する。実H8復元・control・compile・fit・残差・再会計はすべて0。
- liveの読取専用probeは3 windowを一回だけ観測し、CPU1/2/3/6/7/13の候補を返した。**新grantのCPU割当ではなく、fresh gate通過の証拠でもない。** これを使ってrun01を再起動していない。将来の実行時には新しい候補観測と、別のfresh gateが必要である。
- 旧677保護ファイル（run01の承認済みgrantを含む）、116旧worktreeのHEAD/status、旧V2未追跡scriptが不変。run01原stderr・STOP receipt・旧SOURCEを改変していない。V1/V2 NO_GO/H5/H7を保持した。

これはprimary agent自身のlocal技術検証であり、独立レビュー・CI・H8実測成功の証拠ではない。空きcoreがその後も空いている保証や、次回必ずfresh gateが通る保証はない。

**修正と新SOURCE seal・無効grant案を公開し、remote SHA/treeを確認後mandatory STOPする。新有効grant発行・取得の再起動は0。再実行は別途明示承認後、新grant・未使用実行域で元のfresh条件を満たした場合に限る。**
