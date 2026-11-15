# 粗幅p7／標準Yoshida4：Q3復旧完了、この固定配分ではモデル不適合

**互換Pythonで元AST guardをそのまま通し、承認された粗幅32h/5のQ3を1件完了した。MSE区間は `[3.409058131550481301103526e−6, 3.409058131550481301103539e−6] Ha²`（表示）。下界が元exact τを超えるため、この固定配分では `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。取得・数値gateのUNKNOWNとは区別する。** 新しい信号・reference・回路build/compile・価格・fit・別配分は0、粗幅総費用はnull、追加実行せずmandatory STOP。

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、B0I p7、一体項＋fragment0..6保存順、標準Yoshida4、PF要求幅32h/5、h_RPE=.1、m0〜m12、P0除外。元configurationは `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。scalar・signed step・入力・raw信号は保存packetのままで再評価していない。

## 同じ配分で現幅と粗幅を比較

X/Y各軸 **`[59,55,51,47,43,39,35,31,27,23,19,7,2]`**。現幅Yoshida4の保存B_PRICE列をそのまま使った。

| PF要求幅 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 | 取得 |
|---|---|---|---|---|
| 現幅16h/3 | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | モデル適合 | b4296b110の保存結果引用 |
| 粗幅32h/5 | 3.409058131550481301103526e−6 | 3.409058131550481301103539e−6 | **この配分でモデル不適合** | 復旧1 scientific API call |

現幅は `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/run01/B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json` を引用。現幅の再Q3は0。 両結果の評価target・統計law・数値options・exact τは一致し、SOURCE欄だけは各実行の来歴をそのまま保持する。元exact τの表示は `2.539561285063813282041692e−6 Ha²`。判定は表示値ではなく、保存したexact有理数のMSE下界/上界と元τを比較した。

exact両幅の端点と固定result locatorは [SUMMARY.json](SUMMARY.json)、C向けのconfiguration・配分・26軸・判定の結合は [C_RESULT_JOIN.json](C_RESULT_JOIN.json)。旧AST停止結果 `34783cd…` のUNKNOWNは履歴として保全し、今回の有効なモデル不適合結果と混ぜない。

この結果は「現幅で適合した同じ列を、粗幅へそのまま移しても適合するか」という1点の否定である。粗幅全体の不可能性、shot増での救済可否、最小必要N、最適幅・費用順位は判定していない。配分の追加・生成、価格規則の取得へ自動進行しない。

## 保存入力と科学契約の再利用

- 13round信号：`f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53:docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json`。
- 固定Q3入力：`278fb5214f1091efba2a05b305bc05190cd18ff6:docs/research/d_p7_y4_coarse_signal_q3_20261116/Q3_INPUTS.json`。
- 復旧INPUTS wrapperの `inputs` は元Q3_INPUTSと完全一致し、13round・configuration・同一整数列・target・options・契約を再生成していない。新しい出力は本ディレクトリの `run01/`。

元Z1＋T1 total decoder（`consecutive_dyadic_nearest_branch_principal_energy_total_v1`）、保存DF10 solver proxy、IID deterministic rawのexact affine probability、全事象平方energy loss、元exact τ、zero HOLD・tie規則、残余lossとbinary64 engineering allowanceを維持した。元Minkowski/AδEをMSEへ加算せず、十分N・global RMSE・物理ground/sampler移送の新しい保証は出していない。

## 本番入口の互換性確認

実際のexecutableは **`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`**、実体は `/home/AbeHiromu/projects/h6-environments/20261011/python-3.11.0rc1/bin/python3.11`。Python **3.11.0rc1**、NumPy1.26.4、Linux/glibc情報もINPUTS.runtimeとraw EXECUTIONに保存した。D2の8ec7c8828の環境をそのまま使用し、install・共有環境変更は0。

元kernel710aa69b、accelerator9e294778、既存PF workerとAST guardを変更していない。実コンストラクタでactual/expected hashはともに
`3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8`
へ一致した。type_params除去、期待hash置換、guard無効化、数値許容差変更は0。前回のPython3.12停止を回避するための変更は専用runtimeの選択であり、元停止SOURCE・出力を上書きしていない。

科学call前の5人工/metadata testはPASS（[TESTS](TESTS.txt)）。本番と同じPF ProcessPool→worker→Observed/Accelerator→T.evaluate→原API/partials/RESULT/RESOURCE/PERSISTED保存をinventedな2-round信号で実行し、metadata intake以外をmockしなかった。さらに既存exact-DP testを再利用し、全人工law、worker/task分割、scope境界で元kernelとloss・質量・残余・状態出力が完全一致した。人工例を実H4のQ3件数や科学性能として数えていない。primary agent自身の検査であり、独立レビューとは呼ばない。

## 実callのgateと原結果

- 科学API呼出し **1件**、原API/RESULT一致、取得失敗・monitor error **0**。
- 再利用した粗幅自身の13round信号で `VERIFIED_FINITE_BINARY64_SUPPORT`、検査pair **2,551,245**。
- total DP遷移 **85,523,546**、固定上限内・非終端frontier≤8192。全13roundを処理した。
- 残余probabilityは約 **3.490286031618168215020489e−31**、残余loss上界は約 **1.296260797304026829969874e−29**。0に置かず元のexact量と上界会計を保存した。
- API/partials、exact上下界、元判定、資源、PERSISTED/SUPERVISOR receipt、終了code/signalを保存。記録された98 child終了entryはすべてexit0、signalなし。取得・保存正常と、閾値へのモデル不適合を別に扱う。

原API・partialsの固定公開先は **`537df23766e962fad027bf98ba7edf3c6854d927:docs/research/d_p7_y4_coarse_q3_recovery_20261116/run01/`**。判定のexact比較、質量partition、残余loss加算、元入力一致の確認は [CHECKS.json](CHECKS.json)。新しい科学条件での再解析・fitはしていない。

## 資源と並列化

既存の性能検証済み8-worker/batch8 DPを再利用した。小さい末尾roundではtask数に応じてworker数を自動減少させる。恒常的なCPU/RAM制限や共有設定の変更は0。事前観測はCPU affinity64、available RAM約510,187,356 KiBであり、別担当processへ干渉していない。

| 記録範囲 | 実測 |
|---|---:|
| 監督全体wall | 134.798212 s |
| worker API/RESULT/資源保存までwall | 134.755864 s |
| worker CPU | 126.881767 s |
| child CPU合計 | 152.480065 s |
| process-tree CPU合計 | 279.361833 s |
| worker peak RSS | 596,748 KiB |
| 最大単一child peak RSS | 435,780 KiB |
| 定期観測tree RSS最大 | 3,634,356 KiB |

tree RSSは定期観測値で厳密な同時peakではない。periodic faulthandlerや非同期stack読取を使わず、既存のstage/round logと親monitorで進捗を確認した。科学call中のSOURCE変更・停止再起動・retryは0。別interpreterとの速度倍率や科学結果の一致は測定していない。

## SOURCE・Cへの引渡し・STOP

専用branch `codex/d-p7-y4-coarse-signal-q3-20261116` を継続した。

- 実行SOURCE：**`5e2246d1233fbeebc709cc9f20a3f48e2fb3a1ad`**、`scripts/run_p7_y4_coarse_q3_recovery.py` と本ディレクトリINPUTS.json。
- 原API/partials/結果SOURCE：**`537df23766e962fad027bf98ba7edf3c6854d927`**。
- C向け結合表SOURCE：**`8a2f5c46d201b26bdaf2aad7ca556e31083779aa`**、`docs/research/d_p7_y4_coarse_q3_recovery_20261116/C_RESULT_JOIN.json`。
- 結果projection code：**`b961922959cdcea1e22f5779d51f05662a799ba6`**、本ディレクトリsummarize.py。元APIからの値引用とexact検算のみで、新risk callは0。
- 旧signal SOURCE3179680ed、bound packet f23c3f10e、旧UNKNOWN34783cdと旧q3_run01は不変。

§10.5の確認は [DELTA_CHECK](DELTA_CHECK.json)。関連D/B/Aの公表namespaceと同ユーザーの対応jobを差分照合し、同候補・同配分・同契約の受入可能な後続結果/進行jobは確認範囲で0だった。Bの後続S2 one-shot、Pauli LD3準備、AのPauli LD3信号は別候補なので流用しない。未参照branch/host全体の不存在は主張しない。

[HANDOFF](HANDOFF.md) にCが読む固定commit/path、26軸の結合、exact result/assessmentを示す。粗幅総費用は **null**。同公式現幅435,641,568 RZ、計画時点p7最安435,481,090 RZは将来比較の保存基準として残し、この不適合点の費用を算出していない。

統合担当への差分は「粗幅Y4の固定配分Q3はUNKNOWNから、有効なモデル不適合結果へ解決。新信号・価格資格・適合費用は追加なし」。ロードマップ本体は編集しない。原SOURCE/input/raw/STOP、grant/ledger/marker/KEEP、他担当branch/worktreeを保全し、commit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。追加Q3・shot増・価格評価・G/winnerへ進まない。
