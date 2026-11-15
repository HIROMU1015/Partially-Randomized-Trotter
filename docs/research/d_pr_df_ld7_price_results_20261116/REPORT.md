# PR-DF LD7固有価格・独立holdout・候補別Q3（2026-11-16 JST）

**LD7のV0/A_SHAPE/B_PRICE内の最小適合点はA_SHAPE、モデル費用は1,235,340,731.912698 RZ。** X/Yの選択列はいずれも `[48,45,42,39,35,32,29,25,22,19,16,12,9]`。MSEは `[5.56517630181748821477536E-7, 5.56517630181748840998541E-7] Ha²`。採択した2配分はともにQ3適合した。計画f89の分岐規則では、次の候補として**LD9**を提案する。

今回新規はnormal10 path・20 compile・軸別fit2・26参照価格・配分2・Q3一回ずつ計2件。configuration/信号/追加配分/科学retryは0。全20価格receipt有効、欠測・取得失敗・UNKNOWN・保留は0。計画段階3〜5を終え、公開・remote一致確認後mandatory STOP。

## ① 共通V0：各分割自身の信号・価格

X/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,12]` を共有する。

| L_D | 配分 | モデル費用 (RZ) | MSE lower..upper (Ha²) | 判定 | m7..12費用割合 |
|---:|---|---:|---|---|---:|
| 1 | V0 | 261,314,830,475.559718 | 4.72462588934653801951643E-7 .. 4.72462588934653960729534E-7 | 適合 | 99.923854% |
| 2 | V0 | 63,914,076,114.980849 | 4.72409731948051054908399E-7 .. 4.72409731948051194818804E-7 | 適合 | 99.896387% |
| 7 | V0 | 1,668,829,456.071429 | 4.28414160318294721301353E-7 .. 4.28414160318294721365383E-7 | 適合 | 96.017751% |

## ② 候補別V0/A/B内の最小適合点

| L_D | 配分 | モデル費用 (RZ) | MSE lower..upper (Ha²) | 判定 | m7..12費用割合 |
|---:|---|---:|---|---|---:|
| 1 | B_PRICE | 189,232,849,610.910739 | 7.67458737806743819998359E-7 .. 7.67458737806743978776249E-7 | 適合 | 99.894849% |
| 2 | B_PRICE | 46,341,978,336.212733 | 7.67401652027366652784195E-7 .. 7.67401652027366792694600E-7 | 適合 | 99.857098% |
| 7 | A_SHAPE | 1,235,340,731.912698 | 5.56517630181748821477536E-7 .. 5.56517630181748840998541E-7 | 適合 | 95.945359% |

LD1/LD2は `8ec7c882806809460d38a638a891bafcf18fafd1` の保存6点・最小点をそのまま使用し、再fit/再予測/再Q3していない。LD7のV0 Q3は `0defa286c53a1b9acf513be453907bcad83102b3` から再利用した。費用は各候補自身のモデルによる実際の加重額であり、3/4試行予算額を比較値にしていない。表示だけ小数6桁で、集計・比較・最小点選択はexact有理数。

LD1は保存E_I[S]＋有限normal Qの残差モデル、LD2はm0/m3/m6 full-wrapper Cの3点モデル、LD7はm0/m6 full-wrapper Cの2点モデルと、根拠が異なる。この表は採択した探索モデルと有限配分内の比較であり、厳密な対象lawのE[C]、CI、正式G/winnerを認定しない。

## 全9点とLD7の固定予算

| L_D | 配分 | モデル費用 (RZ) | MSE lower..upper (Ha²) | 判定 | m7..12費用割合 |
|---:|---|---:|---|---|---:|
| 1 | V0 | 261,314,830,475.559718 | 4.72462588934653801951643E-7 .. 4.72462588934653960729534E-7 | 適合 | 99.923854% |
| 1 | A_SHAPE | 193,529,481,006.191629 | 6.25468375335249038887699E-7 .. 6.25468375335290388084314E-7 | 適合 | 99.922095% |
| 1 | B_PRICE | 189,232,849,610.910739 | 7.67458737806743819998359E-7 .. 7.67458737806743978776249E-7 | 適合 | 99.894849% |
| 2 | V0 | 63,914,076,114.980849 | 4.72409731948051054908399E-7 .. 4.72409731948051194818804E-7 | 適合 | 99.896387% |
| 2 | A_SHAPE | 47,334,382,243.300078 | 6.25404466306816472494629E-7 .. 6.25404466306840669256650E-7 | 適合 | 99.894147% |
| 2 | B_PRICE | 46,341,978,336.212733 | 7.67401652027366652784195E-7 .. 7.67401652027366792694600E-7 | 適合 | 99.857098% |
| 7 | V0 | 1,668,829,456.071429 | 4.28414160318294721301353E-7 .. 4.28414160318294721365383E-7 | 適合 | 96.017751% |
| 7 | A_SHAPE | 1,235,340,731.912698 | 5.56517630181748821477536E-7 .. 5.56517630181748840998541E-7 | 適合 | 95.945359% |
| 7 | B_PRICE | 1,240,709,696.976190 | 9.19224093342162256494626E-7 .. 9.19224093342162256502003E-7 | 適合 | 94.643635% |

全点 `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。不適合/不確定の結果を捨てた選別ではなく、今回の2 callは両方完了・適合した。

LD7のV0モデル総額は `1668829456.07142857142857...` RZ、採択予算はその3/4の `1251622092.05357142857143...` RZ。

| 配分 | X=Yの整数列 | 実モデル費用 | 予算残額 |
|---|---|---:|---:|
| A_SHAPE | 48,45,42,39,35,32,29,25,22,19,16,12,9 | 1,235,340,731.912698 | 16,281,360.140873 |
| B_PRICE | 64,60,56,52,47,43,38,34,30,25,21,14,5 | 1,240,709,696.976190 | 10,912,395.077381 |

価格は正の有理数、重みはV0またはV0/C_ref、各軸1≤N≤V0。B2の有理数breakpoint generator（e094b2d6）を再利用した。整数列・軸別価格・実費用・残額は科学Q3前の `ca7c016f5b6bc084de82e047ffc30befea73cb4c:ALLOCATION_INPUTS.json` に固定。元V0やA/B間に同一vectorはなく、新規Q3は2件だけだった。[MENU.csv](MENU.csv)、[SUMMARY.json](SUMMARY.json)にexact値と原Q3を結ぶ。

## LD7自身の較正と価格規則

保存H4 linear1.00Å/STO-3G/tol-only DF10、元H/state/snapshot/fragment順。one-bodyは別headでL_Dに含めず、DF0..6をhead、DF7..9をtailとする。S2・Δ=.1・K2・共通κ、configuration `ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1` と、元mapperによる全13 R=1・単一pieceを再利用。

測定付きfull controlled Hadamard wrapperの総compiled RZ/shotがC。P0・symmetric-directional、CONTROLLED_ZZ_CX_CRZ_CX_V1、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17を維持した。c=`2.29310124732` とtail identity s_R=`4.5302064712666724e-05` は別の保存SOURCEで、総scalar `2.293146549384713` を総Tに一度だけ適用。実NativeRuntime preparationはhead/tail・lambda・total scalar hexをrequestと照合した。

較正はm0 physical `2b4a68084b0c88723f93` q1/R1と、m6 `e7194af048772d7160c6` q64/R1の各fresh4 path。cohortは `D_PR_DF_LD7_PRICE_CAL_20261116_V1`。各outer seedを元trial_seed(physical,cohort,index,outer,0)で事前固定し、sampler出力をそのまま使用した。X/Yは同じevent列の別wrapperを個別compileした。

| 役割 | m | path | S_X / S_Y | C_X / C_Y |
|---|---:|---:|---:|---:|
| calibration | 0 | 0 | 5798 / 5798 | 6439 / 6439 |
| calibration | 0 | 1 | 5796 / 5796 | 6589 / 6589 |
| calibration | 0 | 2 | 5800 / 5800 | 6499 / 6499 |
| calibration | 0 | 3 | 5800 / 5800 | 6471 / 6471 |
| calibration | 6 | 0 | 355621 / 355621 | 392817 / 392817 |
| calibration | 6 | 1 | 355607 / 355607 | 394877 / 394877 |
| calibration | 6 | 2 | 355616 / 355616 | 394134 / 394134 |
| calibration | 6 | 3 | 355629 / 355629 | 393521 / 393521 |
| holdout | 3 | 0 | 44673 / 44673 | 49537 / 49537 |
| holdout | 3 | 1 | 44665 / 44665 | 49843 / 49843 |


Cの4 path平均はX/Yそれぞれ、m0=`12999/2`、m6=`1575349/4` RZ。この2点から既存exact solverで各軸を別々に解いた。

`C_ref(m,a)=alpha[a]+beta[a] q[m]`

今回は両軸とも `alpha=88523/252`、`beta=1549351/252`。offset0でfull Cを予測し、Sや残差を再加算しない。全R1ではqR=qなので、betaはhead/tail/basis/optimizerを合わせた探索上の傾きであり、別々のhead/tail係数を同定したものではない。LD2係数の移送は0。較正残差0は2点2係数の解であり、予測性能の検証とはしない。

16 receiptを `ec09e36cd56fa30a776812132c26584f15841af7` で公開後、軸別モデルと26軸の正・有限な参照価格を `4326422a4633ed46968d6e348fe5ffaa24fbb468` で固定・pushしてからholdoutへ進んだ。[FROZEN_MODELS.json](FROZEN_MODELS.json)、[AXIS_PRICES.json](AXIS_PRICES.json)を参照。

## 独立m3：補間域の記述的確認

physical `a6a83aa244eddd8b0956` q8/R1、cohort `D_PR_DF_LD7_M3_PRICE_HOLDOUT_20261116_V1`、固定normal2 path（較正とseed重複なし）。m3を較正へ再投入せず、モデルbytesは凍結後変更していない。

| 軸 | 凍結予測P | 2 path観測平均O | P−O | 絶対差 | (P−O)/O | 絶対相対差 |
|---|---:|---:|---:|---:|---:|---:|
| X | 1783333/36 | 49690 | −5507/36 | 5507/36 | −5507/1788840 | 5507/1788840 |
| Y | 1783333/36 | 49690 | −5507/36 | 5507/36 | −5507/1788840 | 5507/1788840 |

相対差は約−0.307853%、負は過小予測。`DESCRIPTIVE_COMPLETE`、価格モデルの判定は `NOT_ASSESSED_NO_ADOPTED_TOLERANCE`。事後閾値・再fit・追加pathは0。凍結モデルのholdout未開封という欄は凍結時点の状態を保ち、開封後の結果を別sidecar [HOLDOUT_COMPARISON.json](HOLDOUT_COMPARISON.json)（c160ffb2f）へ保存した。

m3は補間域だけの確認で、m7..12はq64より先の未検証外挿。選択LD7 A_SHAPE費用の約95.945359%をその外挿が占める。各最小点のm0..6/m7..12小計と加重26項はSUMMARYにあり、LD7はそれぞれ `3155583784/63` と `149341764653/126` RZ。観測価格やholdout誤差で長round予測を差し替えていない。

## 実行SOURCE・検査・資源

[SOURCE_LOCATORS.json](SOURCE_LOCATORS.json)に40桁commitを集約。価格取得・fit/配分の実行SOURCEは `8d4503e00`、Q3/結果joinは `ec09e36cd`。使用SOURCEとraw公開commitは別。

§10.5はf89/0defで済んだ調査を再利用し、fetch後の同等prefix7 receipt・cohort・モデル・配分・Q3・job・出力先の差分を確認した。受入可能な価格の再利用0、未解決0。PR-Pauli LD3と決定論PF/Y4は別方式として除外し、他担当job/worktreeへ介入していない。[DELTA_CHECK.json](DELTA_CHECK.json)、[Q3_DELTA_CHECK.json](Q3_DELTA_CHECK.json)。

prefix7の実取得経路 `from_b → NativeRuntime.context → 既存acquire_path → Adapter.acquire` を、10個の人工fragment・保存H4入力禁止・供給toy event・fake compilerで検査した。総scalar一回・head/tail・X/Yのmetrics/fingerprint保存、seed不一致のcompile前STOPを確認。最初の人工B fixtureとfixed-grid row欄の違いを修正した経緯も[TESTS.txt](TESTS.txt)に残す。元guardは変更していない。

実価格gateは元P2と同じ入力/preparation/request/lowering/compiler/fingerprintの範囲。numerical action gateとfull-wrapper statevectorは未実施で、新しい保証を加えていない。10 pathで計276 eventを保存し、実観測はorder0のみ、order2/productは未観測。追加samplingで補っていない。旧保存H4 P0の独立数値検査suiteは実行0。

Q3はPython3.11.0rc1、元kernel710aa69b・accelerator9e294778・元AST hashのまま。元decoder/target/lossとexact τ=`15136964828156312001/5960464477539062500000000 Ha²`、ideal finite-RTE raw IIDの契約を維持した。count-only→元科学API内の新しい閉包検査→total DPの順で各1回。API_RETURN/RESULTとvector/call/closure/exitの一致は[CHECKS.json](CHECKS.json)。作成者自身の照合であり、独立レビューではない。

| 段階 | wall(s) | CPU(s) | peak RSS(KiB) |
|---|---:|---:|---:|
| 較正8 path/16 compile | 202.816 | worker合計823.024 | 単一worker最大926,540 |
| holdout2 path/4 compile | 29.992 | worker合計54.672 | 単一worker最大444,088 |
| Q3 A_SHAPE | 149.894 | worker+child242.370 | worker573,896 |
| Q3 B_PRICE | 222.068 | worker+child402.904 | worker830,176 |

Q3二件は並列、supervisor wall222.146s。sampled process-tree peakはA=3,756,672、B=5,426,832 KiB（同時合算peakの測定ではない）。全worker/外側entryのexit0、signalなし。価格runtimeはPython3.12.3/NumPy1.26.4/Qiskit1.3.0、Q3は3.11.0rc1。各PROCESS/RESOURCE_FINALと協調stage/round logを保全。科学SOURCEを実行途中に変更せず、周期faulthandler・無断retryは0。

## 次の判断と残る資格

f89 REPORTの事前分岐「LD7がLD2より低い最小適合モデル費用ならLD9」を満たしたため、次はLD9を提案する。DF0..8 head/tail9に移した領域を調べる案で、今回はconfiguration/schedule/signal/価格を一切生成していない。LD3/LD5が劣るとも、費用がL_Dに単調とも結論しない。小さいtailのprovider normalizer精度gate等は次の固定計画で照合が必要。

価格は実装normal sampler Qの有限cohort由来の探索モデル、Q3は保存ideal finite-RTE lawの信号による条件付きmodel MSE。全law価格・PCG64 joint lawとの移送、有限pathの推定誤差、長round価格誤差、ground/numerical qualificationは未認定。LD1 F0をLD7へ移送していない。[HANDOFF.md](HANDOFF.md)にC入力とロードマップ差分を記載。固定SOURCE/raw、LD1凍結モデル・m4独立holdout、旧STOPは保全し、以後の取得は別指示。
