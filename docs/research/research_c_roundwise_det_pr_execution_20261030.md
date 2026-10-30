# 研究C：Round別D/PR割当・保存189-callの限定科学実行結果

2026-10-30 JST。**固定SOURCEを変更せず189/189 logical units COMPLETE、correctness FAIL・retry/resume0。主RZの両仮定laneで全13roundがDを選び、`MIXED_ROUND_DOMINATED`となった。** 登録した保存価格の点モデル内では混合による全決定論からの改善がない。数学的な混合の不成立、PR一般の不可能性、母平均の保証付き支配性とは区別する。

## 1. 固定対象・SOURCE・許可範囲

保存直鎖H4・1.00Å・STO-3G、Nα=Nβ=2/sector36、同一PR-aware number CDF8 γ=.1 J/Uを用いた。Dは全frame ordered symmetric S2 s1、PRはLD2 partial ordered symmetric S2 s1/r128/K0。δ=h=.1、m0..12、t_m=.1·2^m、own-DF ground-state oracle、状態準備費用除外。元g補正κ・核scalar一回・spin/frame順を保持した。CA=.00159 Ha、DF reserve CA/8、βtotalπ/6、βstatπ/12、αaxis=.01/26、αtotal=.01は固定。

- 科学SOURCE commit：`79fa0107658c92c3b3169d62348498c7dc37fab0`。実行前後に無変更。
- Branch：`codex/research-c-roundwise-det-pr-execution-20261030`。
- [固定契約](research_c_roundwise_det_pr_contract_unapproved_20261030.json) SHA256：`edbeaa1986d98e4005729393fc10d48dd8bbc17bdf074f6432173e70b3f547db`。
- [SOURCE118 manifest](../../artifacts/research_c_roundwise_det_pr/2026-10-30/technical/SOURCE.json) SHA256：`d6a1daa001dea6fa764007a1f17e18bbc9f2c74d093c4619b2aacd0bef848b25`。
- [技術設計・数学的十分条件](research_c_roundwise_det_pr_allocation_design_20261030.md)／[実行前照合](../../artifacts/research_c_roundwise_det_pr/2026-10-30/execution_preflight.json)／[原本hash・postflight](../../artifacts/research_c_roundwise_det_pr/2026-10-30/execution_postflight.json)。

ユーザーの明示的なチャット許可に基づき、独立worktree・fresh run01で準備済みentryを一度だけ実行した。歴史的unapproved名/metadataは変更せず、承認証明・承認原文保存・新grantを作成していない。旧M2/M4のrunnerや残call枠を使わず、新契約189 unitsで実施した。

同じ保存J/U `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、input snapshot `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、元P-H4 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`を照合した。共通energy IDは [common_energy.json](../../artifacts/research_c_roundwise_det_pr/2026-10-30/run01/common_energy.json) に入力/ground/reference/phase等のbindingと保存。旧method IDと証拠出典も各roundに保持した。

## 2. Gate・資格・選択の結果

| Gate/区分 | 結果 | 意味 |
|---|---|---|
| SOURCE118・input23・環境・fresh | PASS | 公開SOURCEと契約・入力を照合 |
| 同H/JU/ground/reference・位相・時間・保存alias/DF来歴 | PASS | 旧target名の置換だけでなく共通energyへbind |
| 保存PR shots/status | PASS | 再接続後の全3lane各13行が旧M4と一致 |
| Evidence D / PR | D 13 PASS、PR 13 UNRESOLVED | Dは保存numerical synthesis pilot。PR全law syn/law未資格 |
| Ideal / Assumed 各D・PR | 各13 PASS | 登録した条件付きphase/radius・shots条件 |
| Full13 energy/failure：ideal/assumedの全D・全PR・選択 | 全6 scheduleがCONDITIONAL_CA_PASS | 実CAや厳密enclosureではない |
| Correctness FAIL | 0 | 科学SOURCE修正・fallbackなし |

| Lane | 割当判定 | 選択 | 主RZの改善対全D |
|---|---|---|---|
| evidence | UNQUALIFIED | null | null |
| ideal_exact_native | MIXED_ROUND_DOMINATED | m0〜12すべてD | 0 |
| assumed_raw_signal_budget | MIXED_ROUND_DOMINATED | m0〜12すべてD | 0 |

Evidenceのmixed energyが`INCOMPLETE_ROUNDS`なのは、未資格のPRを含む比較からscheduleを構成しなかったため。保存13roundの欠測やcorrectness FAILではない。PRの未知誤差・価格supportを0として判定していない。新しいPF/RTE/ground/DF normや回路の正しさ評価も実施していない。

## 3. Round別の条件付きRZ比較

以下は登録済みallocation出力の転記。価格はcomplete kernel＋outer shell、各X/Yの条件付き十分shots込み。Roundごとに同じ方法をX/Yへ用い、RZの最小値を選ぶ。未知の費用を推測していない。

| m | Ideal D | Ideal PR | 選択 | Assumed D | Assumed PR | 選択 |
|---:|---:|---:|---|---:|---:|---|
| 0 | 11,272,149 | 14,417,856 | D | 11,316,267 | 14,474,175.75 | D |
| 1 | 22,584,832 | 28,832,128 | D | 22,628,943 | 28,944,753.5 | D |
| 2 | 45,166,080 | 57,660,672 | D | 45,254,295 | 57,885,909 | D |
| 3 | 90,328,576 | 115,542,990 | D | 90,504,999 | 115,993,450 | D |
| 4 | 180,653,568 | 231,532,842 | D | 181,006,407 | 232,433,748 | D |
| 5 | 361,303,552 | 464,863,884 | D | 362,009,223 | 466,665,682 | D |
| 6 | 724,014,855 | 936,931,320 | D | 725,426,190 | 940,534,902 | D |
| 7 | 1,450,848,782 | 1,899,084,025 | D | 1,453,671,445 | 1,906,291,175 | D |
| 8 | 2,912,984,604 | 3,906,271,506 | D | 2,918,629,923 | 3,920,685,792 | D |
| 9 | 5,882,418,751 | 8,288,210,425 | D | 5,893,709,382 | 8,317,038,983 | D |
| 10 | 11,968,065,150 | 18,623,243,946 | D | 11,990,646,405 | 18,680,901,048 | D |
| 11 | 24,794,214,147 | 47,105,846,615 | D | 24,839,376,650 | 47,336,474,995 | D |
| 12 | 53,382,074,409 | 152,445,349,926 | D | 53,472,399,408 | 153,137,235,024 | D |

[全26比較CSV](../../artifacts/research_c_roundwise_det_pr/2026-10-30/round_comparisons26.csv)／[全78 accuracy/price行](../../artifacts/research_c_roundwise_det_pr/2026-10-30/round_evidence78.csv)。各rawファイルには共通target、旧source_target_id、method、evidence_id、使用したsyn/law誤差、shots、serial per-shot価格を保存した。Evidence laneの未知項は空欄/nullである。

## 4. 全D・全PR・選択scheduleの総資源

| Lane | Schedule | 算定shots（X+Y合計） | 条件付き総RZ |
|---|---|---:|---:|
| evidence | all_D | 13,610 | 101,825,929,455 |
| evidence | all_PR | null | null |
| evidence | mixed | null | null |
| ideal_exact_native | all_D | 13,610 | 101,825,929,455 |
| ideal_exact_native | all_PR | 16,056 | 234,117,788,135 |
| ideal_exact_native | mixed | 13,610 | 101,825,929,455 |
| assumed_raw_signal_budget | all_D | 13,638 | 102,006,579,537 |
| assumed_raw_signal_budget | all_PR | 16,120 | 235,155,559,637.25 |
| assumed_raw_signal_budget | mixed | 13,638 | 102,006,579,537 |

Idealは両方式のexact nativeを仮定する。Assumedは両方式syn≤.001A、PRだけlaw≤.001Aを追加仮定する。Dのassumed shotsは旧M2 numeric synthesisのshotsをそのまま流用した値ではなく、契約通り同じraw-signal予算へ揃えて再評価した。PRの両laneは保存M4と一致した。これらは十分shotsの算定で、量子shots実行は0。

全6指標の総費用（同じ主RZ選択schedule、指標別の再最適化なし）：

### ideal_exact_native

| Metric | 全D | 全PR | 選択schedule |
|---|---:|---:|---:|
| rz_count | 101,825,929,455 | 234,117,788,135 | 101,825,929,455 |
| rz_depth | 51,420,917,395 | 72,220,391,610.5 | 51,420,917,395 |
| cx_count | 64,645,490,000 | 92,796,579,603 | 64,645,490,000 |
| cx_depth | 57,959,299,320 | 56,962,137,414 | 57,959,299,320 |
| total_depth | 111,725,965,325 | 159,242,635,766.5 | 111,725,965,325 |
| circuit_size | 193,188,517,795 | 465,933,091,247 | 193,188,517,795 |

### assumed_raw_signal_budget

| Metric | 全D | 全PR | 選択schedule |
|---|---:|---:|---:|
| rz_count | 102,006,579,537 | 235,155,559,637.25 | 102,006,579,537 |
| rz_depth | 51,512,143,605 | 72,540,522,177.375 | 51,512,143,605 |
| cx_count | 64,760,178,000 | 93,207,918,053.25 | 64,760,178,000 |
| cx_depth | 58,062,125,304 | 57,214,632,898.5 | 58,062,125,304 |
| total_depth | 111,924,179,139 | 159,948,508,924.375 | 111,924,179,139 |
| circuit_size | 193,531,254,873 | 467,998,428,053.25 | 193,531,254,873 |

[全54 metric行](../../artifacts/research_c_roundwise_det_pr/2026-10-30/conditional_metrics54.csv)。CX depthは全PRの点値が低いが、登録主指標RZを事後変更しない。従って「Dが全指標で支配」とは主張しない。Depthはfenced serial scheduled sumで、全interrogationのglobal compile depthではない。

両仮定の全D・全PR・選択scheduleのenergy評価は共通で、登録条件下の誤差評価値0.0012783296188530234 Ha、candidate failure upper .01。同じDF誤差をenergy側で一度だけ計上した。実測energy誤差や実CA保証ではなく、数値pilot/明示仮定/own-DF oracleに条件付けた十分条件である。

## 5. 不確かさとNO-GOの範囲

PRの32標本校正meanは全roundで共通に再利用した。All-PRのRZ weighted descriptive SEはidealで4,243,453,649.4821386、assumedで4,262,263,563.1163826。W_PR×SEを使い、sqrt(13)縮小や32倍の量子shots乗算をしない。選択scheduleはすべてDなのでW_PR=0で、PR校正SEの寄与は0になる。これは合成誤差・DF誤差・oracle仮定・その他の不確かさが0という意味ではない。

選択と同じ校正meanに基づく記述的な点モデルであり、selection bias補正、母平均CI、全law cost support、ideal-law価格移送、PR全law synthesis/implemented-law、厳密CA/保証付き総資源優位は未資格。Finite32の極値を厳密supportへ格上げしない。各candidateのfailure .01を、全比較/全仮定の同時保証とも呼ばない。

**今回のNO-GOは、このH4・同J/U・固定D/PR二方式・固定β/α・主RZの保存点モデル内に限る。** 混合自体の数学的合法性は維持される。結果を見てr/K、split、精度配分、compiler、対象DF、主指標を変える追加探索は行わない。

## 6. Calls・資源・保全・公開

| Kind | COMPLETE |
|---|---:|
| saved_bundle | 1 |
| common_energy_binding | 1 |
| qualification | 3 |
| round_accuracy | 78 |
| round_cost | 78 |
| allocation | 3 |
| schedule_energy | 9 |
| schedule_total | 9 |
| shared_uncertainty | 3 |
| comparison | 3 |
| summary | 1 |
| **合計** | **189** |

Journalはattempt189＋COMPLETE189＝378行。これは378 logical unitsや2実行ではない。[STOP](../../artifacts/research_c_roundwise_det_pr/2026-10-30/run01/STOP.json)のretry/resume0、first_failure=null、failed=falseを保持した。

[原本91件](../../artifacts/research_c_roundwise_det_pr/2026-10-30/run01/)は547,258 bytes。Contract/SOURCE、common energy、78round、3allocation、3lane結果、全結果、call journal、STOP、resourcesを保存した。CSV3件は登録結果の表示用転記だけで、新しい指標・ratio・ランキング・候補の追加ではない。

実行環境で人工・回帰117件を再確認しPASS。[実行時test log](../../artifacts/research_c_roundwise_det_pr/2026-10-30/execution_tests.log)。CPU affinity32、空きRAM約52GiBを確認。小さいscalar chainと既存依存importが主で、operator/compile workloadはない。既存CPU経路で逐次依存を保ち、恣意的なCPU/RAM/worker/wall capや共有設定変更はない。Task-local venv/cache/tmp、既存依存read-only。

Runner wall 1.1675512604415417秒、CPU 9.441716257000001秒、parent peak RSS 356,464 KiB。GNU timeはwall1.43秒、max RSS358,616 KiB、swap0、exit0。[runner resources](../../artifacts/research_c_roundwise_det_pr/2026-10-30/run01/resources.json)／[外部time](../../artifacts/research_c_roundwise_det_pr/2026-10-30/execution_time.txt)。測定対象の異なるRSSを混同せず、速度向上倍率も主張しない。

SOURCE118/input23/旧保護2728件は実行前後でhash一致。旧M2/M3/M4/native/H2/H4/H6のSOURCE・raw・grant・ledger・marker・STOPは変更していない。新fit/optimizer、分子NPZ数値decode、operator/PF-RTE/ground/DF再計算、QPY decode、sampling/build/compile、新候補/β/α最適化は0。Scientific SOURCEとrunnerも変更していない。

## 7. 次の研究判断とmandatory STOP

ユーザー／研究C：固定H4二方式の主RZ round割当を、今回の点モデル内でNO-GOとして区切るか判断する。PR一般の否定や研究C全体の終了判断には拡大しない。

Codex：本結果を公開・remote SHA/tree・SOURCE/input/旧原本/成果物照合後、**mandatory STOP**する。追加sampling/独立価格検証、新r/K/γ/rank/DF、native最適化、全law保証、詳細GPT科学レビューへ自動進行しない。別の具体的依頼と必要な実行許可後だけ次の作業を行う。

次を一つ検討する場合は、前設計Q10の「tail平均を保つ辞書/samplingの自由度が、既存Euler/IS/Parent allocationを超えてnormalizationとnative価格の交換を改善し得るか」の紙上判定を提案に留める。この次課題の調査・実装・科学計算は今回開始しない。
