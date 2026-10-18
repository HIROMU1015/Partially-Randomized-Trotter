# Track A Codex 2：H4/H6 高次PF B_SAVED_ONLY run01

2026-10-18。**保存値の限定数値解析は正常終了。最良qは未確定のままmandatory STOP。**

利用者の「次の作業に進んで大丈夫」を、直前に提案した最小段階B_SAVED_ONLYへの承認として受領した。Aの新係数適用、native取得、Cの独立sampling、αの変更へ承認を広げていない。

## 1. 結果と意味

H4/H6各12条件、合計24条件のfinite K4 signalを保存値から評価した。**経験的allowanceの下で18適格・6不適格**。費用実測がある6群はいずれも適格で、全ての保存shot数・paired費用標本・標本平均が再現した。SEも保存値と丸め許容内で一致した。

残る18の費用欠測条件のうち6は不適格だったため、**適格な費用欠測は12条件**になった。6層すべてに適格な欠測qが残り、full-grid winnerは全てnull。保存費用があるqを、最適qと読み替えない。

| H4・H6で共通 | 適格q | 保存native費用のq | 適格な費用欠測q |
|---|---|---|---|
| S2 | 4,8 | 4 | 8 |
| Yoshida4 | 2,4,8 | 2 | 4,8 |
| 元8桁new_2 | 1,2,4,8 | 1 | 2,4,8 |

これはT=.8、ε=.001、保存prefix/rank、経験的allowanceに条件付けた判定であり、除いたqが他精度や時間でも不適格という意味ではない。最大qを増やす判断はしていない。

## 2. 保存実測6群の再現

| 系 | PF/q | paired n | Shot重み付きRZ標本平均 | 標本SE |
|---|---|---|---|---|
| H4 | S2/q4 | 8 | 4.976017044e12 | 7.884596393e10 |
| H4 | Yoshida4/q2 | 8 | 2.262585049e12 | 3.451086749e10 |
| H4 | new_2/q1 | 8 | 1.321206962e12 | 1.724510638e10 |
| H6 | S2/q4 | 8 | 7.127100012e13 | 1.408771761e12 |
| H6 | Yoshida4/q2 | 8 | 1.363182105e13 | 7.676736648e10 |
| H6 | new_2/q1 | 8 | 5.993261164e12 | 2.256837221e10 |

値は今回の出力の表示丸め。元の独立n8へsampleを追加していない。n16と数えず、独立確認結果ともしない。各trajectoryでcosine/sineの同一eventを保ち、軸別shotで `N_real*C_cos+N_imag*C_sin` を計算した。CXも出力済み。model費用の混入・欠測の0補完なし。

候補間6点比較の出力は点費用差まで。候補間の独立性は認定せず、差/比の厳密区間や母平均winnerは出していない。保存new_2の係数問題、未知希少event、経験的allowanceの非厳密性は変わらない。PRの3公式間比較であり、B0/B0+Iや全RPEエネルギー費用の評価ではない。

## 3. SOURCE・承認・固定条件

- 準備branch：`track-a-codex2-h46-higher-pf-readiness-20261018`。
- 起動前remote HEAD：`8464c06a333686d2b503ffb26b2014112bc973b2`。利用者のpush後、remote一致と主要公開3fileのbytes一致を確認した。過去のpush認証失敗記録はその時点の履歴として保持する。
- 実行SOURCE：`ed14d951a4761838f25aeb16a653b55aba499c34`。専用のdetached checkoutで実行し、資料を追加する元cloneのHEADを動かしてSOURCEへ偽装しない。
- plan digest：`6d65530aef91f8cb77892ceb48fb7cb5d4e4e16f1d9fff7b050b63b8c9aa11ed`。
- 一回grant：`H46-HIGHER-PF-B-SAVED-20261018-RUN01`。消費済みmarker1件を保持。旧INVALID draft、SOURCE/grant/ledger/STOPは上書きしない。
- native元SOURCE：H4 `687679a4…`、H6 `f0dd2a9a…`、provider `c31ef424…`。元signal commit `7198bbde…`、元native結果 `495496c4…`。

条件はlinear H4/H6、隣接1Å、STO-3G、DF rank12/19、PR prefix3/5、保存state/fragment ordering、T=.8、K4、q1/2/4/8。outer step当たりRTE step32（全体32q）、元signed配分。ε=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、logB margin1e-12、P0。

native費用は保存ordinary control＋explicit XY/number Phase系列、profile hash `1239304c…`、rz/sx/x/cx・optimization1・seed17・backend/couplingなし。同じprofileを参照したが、今回compilerを起動していない。α=.025の変更や、別α研究の検証はしていない。

## 4. 実行・監査・保全

18:19:35–18:19:36 JSTに一回実行。native exit0、wait4で回収、wall **0.7575秒**、最大RSS **19.0 MiB**。Source6file/input24fileを起動前後に照合した。89 imported provider fileと元入力4blobも固定commitへ照合した。CPU affinity64個・実効空きRAM約469GiBを観測したが、本解析は小さい24行でprocess起動overheadが主のため1workerで完了した。将来のnative取得へこのworker数を上限として移送しない。

新SCF/DF/state、新signal、新trajectory sampling、新native build/compile、係数最適化、Phase A/C起動は0。旧結果、grant、ledger、STOP、他worktree、共有環境に書き込んでいない。SOURCE実行checkoutの新markerだけを一回消費した。

監査では6群すべてのshot数値fieldとpaired RZ標本・点平均を厳密照合した。SEの比較は相対許容2e-15。新しいevidence_refはcommit:path#candidate、旧参照はSHA256文字列なので、その参照文字列自体は異なるが、同じ保存fileのSHA256へ結合することを確認した。最初の監査で参照文字列まで含めたdict比較が一致しなかったため、検査側を参照先SHAと数値fieldに分けた。科学計算の修正・再実行はしていない。

新しい実装変更はないため、準備時32人工/mockの保存PASSを再利用した。今回追加したのは実行契約に基づく6群の保存値再現・終了監査であり、独立reviewや新native correctness検証ではない。

[公開bundle](../../artifacts/resource_applicability/track_a_h46_higher_pf_B_saved_run01/2026-10-18/README.md) に未加工result、24行CSV、summary、消費grant/plan、起動gate、native exit、終了監査を保存した。private実行領域は `/home/AbeHiromu/projects/h46-higher-pf-readiness/20261018/runtime/B_saved_run01/`。新grantを発行して同じ解析を重複実行しない。

## 5. 次の作業と承認

**ユーザー**：適格な費用欠測12群について、どの範囲のnative費用を取得するか、探索標本数n_E・control予算を含む契約の採否を判断する。12群を無条件に全取得する承認はない。全格子の最適qを判断するには、この欠測を埋めるか、別承認された有効な費用bound等で候補を除ける証拠が必要。

**Codex**：承認されれば、既存native kernelを再利用してB_ACQUIREのgroup/seed/profile/入力・並列実行契約をsealする。12群全てならcost標本は12*n_E trajectory・24*n_E軸、control/診断compileは別登録。n_Eは未決定。新sampling/compileを、今回のB_saved承認から自動起動しない。

**C独立確認**：探索selectionを事前固定し、目標MCSE/検出差/多重性と必要n_Cを決めた後の別承認。元n8や今回再解析を独立確認へ混入しない。

Aの新しい係数採用、B取得、C確認、α感度、他分子・時間・geometryへ進まず、ここでmandatory STOP。
