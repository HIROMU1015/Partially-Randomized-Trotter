# H4/H6 m5：基本88条件のsignal専用実装・実行準備

2026-10-20。基本88条件の範囲に沿って、signal専用adapter、prefix別context、並列launcher、one-shot契約、経験的allowance案を実装した。人工14件・既存回帰16件、metadata17件・改変拒否7件がPASSした。**今回の完了は技術準備であり、実分子signal・sampling・compile・費用再会計は未実行。有効grantは発行せず、mandatory STOPする。**

SOURCE：`480ca4507394e35185ec535c9f488bd90538de25`。branch：`track-a-h46-m5-signal-readiness-20261020`。起点：基本範囲計画`7415e79edf365fcfb60750f1872722e4238a1950`。ユーザーの「その範囲でよいので作業を開始して」を、[直前の計画](track_a_h46_m5_minimal_comparison_plan_20261019.md)に対するdomain確定と技術準備の指示として扱った。追加PFの主比較採用AD03、allowanceの正式採択、数値実行は別判断である。

## 対象と再利用

保存済みの線形1 Å/STO-3G、tol-only1e-8・rank自動決定のH4（観測rank10）/H6（19）、T=.8、P=0を使う。通常signed bias・allowance倍率1、5精度ε=.05/.01/.005/.001/.0005、両軸ε/√2を対象とする。両系の入力生成全規則の同一性やサイズの因果効果は認定しない。

| 対象 | prefix | q | 条件数 |
| --- | --- | --- | ---: |
| H4 PR：S2/Yoshida4/m5 | 3 | 1/2/4/8 | 12 |
| H6 PR：同3公式 | 5 | 同上 | 12 |
| H4 m5 B0/B0I | 3/8/9 | 同上 | 24 |
| H4 m5 full | 10 | 同上 | 4 |
| H6 m5 B0/B0I | 5/7/11/15 | 同上 | 32 |
| H6 m5 full | 19 | 同上 | 4 |
| 合計 | | | 88 |

PRはK4・`ABS_TIME_PER_OUTER=32`。固定係数版`PF_M5_BEST_20D64C2_BINARY64_V1`の11段signed S2、既存の絶対時間整数配分を再利用する。局所20条件・K2・q16は含めない。旧piece r2のpilot trajectoryは新outer32 cohortへ混ぜない。

元pilot SOURCE `c7d53f04c3be131078ee91e8bef3c78ca434ed3e`の保存raw signalを、H4/H6それぞれB0/B0I anchorとfullのq2、計6候補で再利用する。元のraw inventoryのSHA/bytes、候補・入力・reference・runtimeを結合した。したがって**新しい平均signalは最大82候補**である。旧runの費用PASSや正常終了を再認定する操作ではなく、H6 fullの旧STOPも維持する。

full-H referenceも保存2件をSHA・Hamiltonian/state・full preparationへ結合して再利用し、新たに作用しない。実行時は各系のsnapshot/referenceを1回ずつ復元し、5精度×2軸の経験的適格性880行を出す予定である。shot数・費用・rankingは出さない。保存単価12軸の再利用や既存強化S2/S4対照との結合は後段であり、今回のsignal契約から費用を認定しない。

## 実装と並列化

- `references/h46_m5_signal_domain_v1.json`：固定PRIMARY88行。optional20を含まない。
- `scripts/h46_m5_signal_contract.py`：SOURCE/input/runtime/保存reference/6raw signalの結合、認可、候補別one-shot台帳。
- `scripts/h46_m5_signal_adapter.py`：保存tol-only loaderと既存`h46_m5_native.signal`を再利用する入口。入力、full preparation、occupation oracle、referenceを系内共有し、prefixごとにcut preparation・tail・identity・λ・spectrumを作る。cache identityはSOURCE・系・入力・prefix・preparationを含み、新prefixへanchor tailを流用しない。
- `scripts/run_h46_m5_signal.py`：H4/H6を別process、各process内の候補をthread poolで並列評価する。共有contextは読取専用。ローカルBLASを1threadにして過剰な入れ子並列を避ける。精度ごとの再作用、同一候補の再計算をしない。
- `scripts/prepare_h46_m5_signal.py`：metadataとbyte hashだけで新契約を封印する。数値配列を復元しない。

最大2系processは対象系数による上限であり、過去のworker制限の流用ではない。fresh idle CPU、affinity、実効cgroup空きRAM、diskで起動可否と候補thread数を決める。RAM admissionには同じ保存入力の旧context RSS proxy（H4約6.35 GB/H6約6.86 GB）を使うが、将来のprefix cache/thread時のpeak保証やruntime上限とは呼ばない。実測peak・速度改善はまだ未検証である。

保存runtimeはPython3.12.3/NumPy1.26.4/Qiskit1.3.0。旧compiler profileはdirectional/full wrapper/rz,sx,x,cx/opt1/seed17の参照bindingとして保持する。今回は回路生成・compile・回路状態作用を行わないため、新条件でのnative検証完了とは扱わない。thread poolの追加stdlibも含め、2,046ファイルを封印した。

## allowance案：経験的・reference条件付き

旧pilotのallowanceはshot用に未採択だったため、新SOURCEで提案を明示した。現時点では正式基準ではない。`M5_SIGNAL_EMPIRICAL_OBSERVATIONS_V1_PROPOSED`として、機械ε=2^-52、最大中間norm M、raw/corrected vector closure c、概数stage数 Nを用い、次を計算する案である。

```
N = q × [2(prefix+1) + 1(PRのみ)] × 公式段数(1/3/11)
roundoff_observation = 64 × 2^-52 × (1+M) × (1+N)
u = max(1e-9, 10c + roundoff_observation)
log_B_margin = max(1e-12, 64 × 2^-52 × (1+abs(log_B)) × (1+N))
h_axis = ε/√2 - abs(signed_bias_axis) - u
```

h≤0は`INELIGIBLE`、h>0でもu/h>.01なら`UNDETERMINED`、それ以外は提案した経験的政策の下で`ELIGIBLE`とする。これは新しい実行で評価する予定の診断であり、今回実保存値へこの算術を適用していない。

**保存referenceの絶対誤差上界は欠測（null）であり、0としない。** このuを厳密な丸め・reference・bias上界と呼ばず、保存referenceの精度に条件付けた経験的判定と表示する。norm/closureが小さいことだけでreferenceの正確さは証明しない。実行承認では、この限定したallowance案の当該runでの使用も明示的に受け入れる必要がある。後段でshot政策へ正式採用したことにはならない。CT01のcompiled vector閾値1e-7は一切uへ流用しない。

## 契約・STOP・保全

private plan SHA：`b9bf938ff4977097b57938a4659afbc3f25d7efd5f63a8aebc83c476f899a04f`。現configは`authorized=false`、grantは`valid=false`、実行承認receiptもfalse。公開artifactは非実行用projectionであり、有効grantではない。実際に`--execute`/`--supervise`の双方へ未承認設定を渡し、科学入力の復元前にexit2で拒否し、出力・marker・台帳・supervisorファイルを作らないことを確認した。booleanだけの変更でも承認receiptが拒否される。

SOURCE・入力・runtime・保存reference・raw signal不一致、非finite値、closure/λ/identity検査不一致、実worker異常、本人の停止signalで該当系をSTOPする。自動retry・再fit・追加候補はない。片方の系がSTOPしても、無関係な他方の正常系は継続し、campaign全体は部分結果を全PASSにしない。host PSIや一時的CPU競合だけでkillせず、他worktreeのHEAD/statusを実行阻止条件にしない。

終了時に結果SHA、候補coverage、台帳88件、SOURCE、worker exit/reapと外部wait4 witnessを照合する設計である。今回は実分子での終了経路は未実行。既存SOURCE/raw/grant/ledger/marker/STOPなど**2,899ファイルは作業前後でSHA一致**し、他worktree・共有環境・他利用者のprocessを変更していない。旧STOPやCT01受入は過去の来歴のまま保持する。

## 技術検証と次の担当

[artifact入口](../../artifacts/resource_applicability/track_a_h46_m5_signal_readiness/2026-10-20/README.md)。人工14件では、3公式×q1/2/4/8×4手法の48 toy matrix条件を独立に組んだdense積と比較し、signed負時間、prefix tail/scalar、full tail空集合、cache共有、6raw再利用、one-shot拒否、2系process成功/片系異常時の独立継続を確認した。既存tol-only回帰16件もPASSした。著者自身による検査であり、独立レビューではない。人工回路のtestは実分子compileに数えない。

metadata17項目・改変7種の拒否、公開資料の構造24項目、全2,046 seal、2,899保護fileを確認した。数値許容条件の現実分子での妥当性・実効高速化・科学的優位性は未確定。新科学signal/compile/sampling/reference生成/保存値再解析/費用再会計/有効grant/pushは全て0。

- **ユーザー**：固定88条件（82新signal＋6raw再利用）、保存reference再利用、上記のreference条件付き経験的allowance案に対する一回のsignal実行を別途承認するか判断する。
- **Codex**：承認された場合だけ、新承認receipt・新grant・fresh gateを結合してsignalを一回実行し、結果・exit/reapを保存してmandatory STOPする。
- **ユーザー**：その結果を受けて後段の費用取得domain、n/seed、compile、会計、独立確認を別々に判断する。n_E/n_Cは未採択、AD03は保留である。

local commit後にmandatory STOP。今回は公開pushも科学起動も行わない。
