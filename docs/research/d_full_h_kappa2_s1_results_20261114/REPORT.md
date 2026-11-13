# Full-H pure RTE κ2・K2 S-1：全13roundの解析的平均signal

2026-11-14 JST。利用者が明示的に承認したS-1を**1回だけ**実行し、完了した（`BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED`）。

- 実行したのは主agent（Claude Code）である。値の検証は作者自身によるもので、独立した査読ではない。
- 実施していないこと：P-1、S-2、κ・K・groupingの変更、sampling、回路の作成・compile、N・MSE・DP・price・Gの計算、retry。

| 項目 | 内容 |
|---|---|
| 固定caller | `scripts/run_full_h_kappa2_signals.py`（固定SOURCE `04bead115`）。数値SOURCEはC `5cc7c07e` の2 module、reference readerはR448 caller `5a172629` で、いずれもbyte一致を確認した |
| 実行tree | 新branch `codex/d-full-h-kappa2-s1-results-20261114`（D準備docs `5ccc1dfc3` から分岐。上記のbyteはすべて同一）。Dのbranch・worktree・processは使っていない |
| 環境 | R448と同じvenv（Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1） |
| 原本 | [run01/](run01/)（PLAN、round_0〜12、RESULT、POST_GATES、B_SIGNAL_INPUT、SIGNALS.csv、COVERAGE、SOURCE、INPUT_PROVENANCE、STOP） |
| 実行記録 | [EXECUTION.txt](EXECUTION.txt) |

## 1. 実行の経過と件数

1. 事前確認（科学計算なし）：
   - 固定byteを照合した。
   - callerの人工test 7件がPASSした。
   - metadata preflightがPASSした：configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、全13round。
   - 計画ではDの`KAPPA_CANDIDATES.json`を再生成し、次の一致を確認している：configuration全体、physical ID、q/R/s、reference descriptor。また、R448のphysical IDが混入していないことも確認している。
2. 実行：`python -B scripts/run_full_h_kappa2_signals.py --execute-approved-s1`を1回だけ実行した。終了コードは0。

| 項目 | 件数・値 |
|---|---|
| 保存H4 contextの読込 | 1 |
| 非scalar spectrum | 1 |
| finite-K normalizer | 13 |
| 平均（raw/corrected） | 13 |
| 新しいreference・scalar抽出・sampling・回路 | 0 |
| 保存signalのcopy | 0 |
| retry | なし |
| 時間 | callerの計測3.53 s、wall 4.13 s |
| CPU | user 11.4 s、sys 5.8 s（NumPyの並列で416%） |
| 最大RSS | 593 MB |

入力の扱い：

- 保存snapshot・receiptはSHA-256で照合した（INDEX `7a8bc7f1`）。
- 状態は36次元、行列は11個（one-body＋DF10）。
- scalarは保存Full-H preparation（9b83からの転記）を使い、新たに抽出していない。

## 2. 全13roundの平均signal（原本：[SIGNALS.csv](run01/SIGNALS.csv)、round_m.json）

| m | T | q | R | s | raw（Re, Im） | corrected（Re, Im） | \|corr−ref\| | arg(corr/ref) | log B | B | attenuation |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0.1 | 1 | 9 | 9 | 0.6067576023, 0.1335429331 | 0.9766253970, 0.2149481433 | 6.48e−8 | 1.06e−9 | 0.4759737828 | 1.60958082 | 0.62127977 |
| 1 | 0.2 | 2 | 18 | 36 | 0.5574945528, 0.2578935595 | 0.9075945656, 0.4198476773 | 1.62e−8 | 1.32e−10 | 0.4873450327 | 1.62798822 | 0.61425506 |
| 2 | 0.4 | 4 | 36 | 144 | 0.3965269288, 0.4667412470 | 0.6474558418, 0.7621029621 | 4.05e−9 | 1.65e−11 | 0.4903066383 | 1.63281683 | 0.61243857 |
| 3 | 0.8 | 8 | 71 | 568 | −0.0982162561, 0.5997784062 | −0.1616018587, 0.9868560367 | 1.06e−9 | 2.19e−12 | 0.4979639053 | 1.64536773 | 0.60776687 |
| 4 | 1.6 | 16 | 142 | 2272 | −0.5759103693, −0.1938127028 | −0.9477696781, −0.3189555403 | 2.64e−10 | 2.70e−13 | 0.4981594775 | 1.64568955 | 0.60764802 |
| 5 | 3.2 | 32 | 283 | 9056 | 0.4831378351, 0.3667155204 | 0.7965347265, 0.6045927798 | 6.71e−11 | 3.54e−14 | 0.4999687415 | 1.64866973 | 0.60654962 |
| 6 | 6.4 | 64 | 566 | 36224 | 0.1631204907, 0.5841960706 | 0.2689351412, 0.9631582891 | 1.61e−11 | 4.0e−15 | 0.4999811053 | 1.64869012 | 0.60654212 |
| 7 | 12.8 | 128 | 1132 | 144896 | −0.5188028520, 0.3142207342 | −0.8553477796, 0.5180542209 | 1.16e−11 | 5.3e−15 | 0.4999841964 | 1.64869522 | 0.60654025 |
| 8 | 25.6 | 256 | 2264 | 579584 | 0.2809732727, −0.5375355992 | 0.4632396483, −0.8862330552 | 2.45e−11 | 1.8e−15 | 0.4999849691 | 1.64869649 | 0.60653978 |
| 9 | 51.2 | 512 | 4528 | 2318336 | −0.3462237894, −0.4980155076 | −0.5708180565, −0.8210765777 | 4.07e−11 | −3.8e−13 | 0.4999851625 | 1.64869681 | 0.60653966 |
| 10 | 102.4 | 1024 | 9056 | 9273344 | −0.2112780680, 0.5685524613 | −0.3483334929, 0.9373706728 | 2.96e−10 | 7.2e−13 | 0.4999852102 | 1.64869689 | 0.60653963 |
| 11 | 204.8 | 2048 | 18112 | 37093376 | −0.4593491691, −0.3960917238 | −0.7573275558, −0.6530352011 | 3.03e−10 | −4.2e−13 | 0.4999852244 | 1.64869691 | 0.60653962 |
| 12 | 409.6 | 4096 | 36223 | 148369408 | 0.0892147152, 0.5999340839 | 0.1470900538, 0.9891231112 | 6.57e−9 | 2.0e−14 | 0.4999990155 | 1.64871965 | 0.60653126 |

表の値は丸めた表示で、正本はround_m.json（binary64）である。

**finite-K誤差・referenceとの差の診断**

- `|corrected−reference|`は全roundで6.5e−8以下だった。最大はm0、次がm12（6.6e−9）。
- 位相差は1.1e−9 rad以下。
- これは「保存した独立reference pair」との差であり、paired finite Taylorの打切り・数値誤差の合計である。数学的な誤差上界ではない。
- `raw = corrected·e^{−log B}`：raw/corrected vectorのclosureは最大3.5e−14。
- |raw|の最大成分は0.607で、測定確率の範囲[−1,1]の内側にある。

## 3. 数値gateとSOURCE照合（全項目PASS）

| gate | 規則 | 結果 |
|---|---|---|
| state norm | ≤1e−12（Context作成時） | PASS |
| Hermiticity・spectrum再構成 | ≤1e−9（入力行列と非scalar H、固定C core `e417ffc9`） | PASS |
| 保存reference pair | ≤1e−10 | 最大9.3e−12 |
| raw/corrected closure | ≤1e−9 | 最大3.5e−14 |
| raw X/Yの範囲 | \|·\|≤1、clampなし | 最大0.607 |
| normalizer | 有限かつb>1 | b=1.0543（m0）〜1.0000000034（m12） |
| scalar・identity | 総時間Tで1回だけ適用（`scalar_once`） | PASS |
| logB包絡 | `0<logB≤nextafter((λT)²/s)+normalizer_one_ulp_log_sensitivity` | 13/13 PASS（[POST_GATES](run01/FULL_H_K2_QDYADIC_KAPPA2/POST_GATES.json)） |
| ID照合 | configuration ID、全physical ID、q/R/s | 一致 |
| SOURCE | caller・integration・数値2 module・layout helper・R448 readerについて、HEADとのbyte一致、および固定blobとの一致 | PASS |

**logB包絡の余裕**

- `(λT)²/s − logB`は全roundで正だった（最小2.3e−9、m11）。つまり、1 ULP感度のallowanceを使わなくても包絡内にある。
- `logB/包絡`の比は0.969（m0）〜0.9999999954。
- 1 ULP感度（3.8e−15〜6.6e−8）は丸めによる誤STOPを防ぐための数値allowanceである。厳密な数学的誤差保証ではない。

## 4. Bに渡すsignal packet

- [B_SIGNAL_INPUT.json](run01/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json)：既存の`I.signal_input`形式で、`configuration_id`、`target`、`source`、13 meansを含む。
- 各meanのstatusは`SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED`、qualificationは`UNQUALIFIED`である。
- 各meanは次を持つ：raw、corrected、reference、`normalization`（logB/B/attenuation）、bias、`provider_binding`、`law_binding.kappa`（κ2）。

## 5. 保存R448との比較：使える条件と再利用できない数値

| 区分 | 内容 |
|---|---|
| 共通（同条件として比較できる） | H4/1 Å/STO-3G、保存DF10 snapshot・state・weighted-Hermitian completion、sector(2,2)、target（full DF-Hの基底エネルギー）、**保存reference pair 13組（同じ`bba15083` PHYSICAL_INPUTS selector）**、K=2、DYADIC q=2^m、T_m=0.1·2^m、scalarの一回会計、数値SOURCE、ideal finite-K law |
| 異なる | groupあたりの事象数：R448はR=448（全round）、s=448·2^m。κ2はR=9…36223、s=9…148,369,408。そのため、finite normalizer b、logB、B、attenuation、raw、物理回路は別物になる |
| 比較に使えるもの | 同じreferenceに対する`corrected`のbias（finite-Kの打切り）、および同じ包絡式に対するlogBの大きさ（方式間の診断として） |
| 再利用しないもの | R448のraw・normalizer・logB・B・physical ID・価格・N。今回はR448のsignalを一切copyしていない（`saved=()`、copy 0） |

R448の保存値と今回の値の差を数値で比較する分析は、今回の承認範囲に含めておらず、実施していない。

## 6. 残るUNKNOWN

- **N・MSE・price・G**：十分N、直接モデルMSE、DP・decoder評価、compiled価格・期待価格、Gは評価していない（null）。meanは`UNQUALIFIED`のままである。
- **ground資格・target**：保存solverのtargetのground資格、DF10とDF前の分子Hの差は未取得。
- **sampler law**：実sampler law（PCG64/CDF・経路sampling）との距離は未資格（`sampler_law_distance=null`）。
- **数値認証**：数値は`certified=false`で、区間演算による認証ではない。
- **κ2の位置付け**：κ2は評価候補の一つで、最適値として採択したものではない。κ1/κ4（S-2）、P-1の新ZZ compileは未実行。

## 7. 保全・STOP

- 旧結果（R448を含む）、固定SOURCE・input・raw・grant・ledger・marker・KEEP・STOP、D・B・Cのbranch・worktree・processは変更していない。
- 共有環境と他ユーザーへの影響はない。
- **mandatory STOP**：P-1、S-2、N/MSE/Gへは自動で進まない。
