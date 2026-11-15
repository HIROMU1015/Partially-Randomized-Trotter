# 粗幅p7／標準Yoshida4：保存A_SHAPE配分のQ3適合・価格待ち

**粗幅32h/5で、現幅から保存整数列を移したA_SHAPEのQ3を1件完了し、`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD` を得た。MSE上界は表示で `2.074377165792685953356238e−6 Ha²`、元exact τ以下。Cへの状態は `Q3_ACCEPTED_PRICE_PENDING`（Q3適合・価格待ち）。粗幅総費用はnull。** 新信号/reference/build/compile/価格/fit/他配分0、追加実行せずmandatory STOP。

対象は保存linear H4／1.00 Å／STO-3G／tol-only DF10、B0I p7（一体項＋fragment0..6の保存順）、標準Yoshida4、PF要求幅32h/5、h_RPE=.1、m0〜m12、P0除外。元configurationは `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。保存粗幅信号のsigned step・scalar総時間一回適用・physical scheduleを変更していない。

## 幅×配分の4点

B_PRICEはX/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,7,2]`、A_SHAPEはX/Y各軸 **`[37,34,32,29,27,24,22,19,17,14,11,9,6]`**。今回評価したのは粗幅A_SHAPEだけで、他3点は固定保存結果の引用である。

| PF要求幅 | 配分 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 | 出所 |
|---|---|---|---|---|---|
| 現幅16h/3 | B_PRICE | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | モデル適合 | b4296b110保存 |
| 現幅16h/3 | A_SHAPE | 1.174900865076655516197599e−6 | 1.174900865076752235939216e−6 | モデル適合 | b4296b110保存 |
| 粗幅32h/5 | B_PRICE | 3.409058131550481301103526e−6 | 3.409058131550481301103539e−6 | この配分でモデル不適合 | 537df2376保存 |
| **粗幅32h/5** | **A_SHAPE** | **2.074377165792602390524467e−6** | **2.074377165792685953356238e−6** | **モデル適合・価格待ち** | **今回1 Q3** |

元exact τは `15136964828156312001 / 5960464477539062500000000 Ha²`（表示 `2.539561285063813282041692e−6`）。判定は丸め前の有理数で、上界≤τ／下界>τ／跨ぐ場合INCONCLUSIVEを区別した。今回のinput・数値・保存gateは正常であり、UNKNOWNではない。4点のexact両端・出所・配分は [SUMMARY.json](SUMMARY.json) に保存した。

現幅2点は `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/run01/` 以下の `B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json` と `B0I_P7_Y4_16h_over_3__A_SHAPE/RESULT.json`。粗幅B_PRICEは `537df23766e962fad027bf98ba7edf3c6854d927:docs/research/d_p7_y4_coarse_q3_recovery_20261116/run01/B0I_P7_Y4_32h_over_5/RESULT.json`。3点の再Q3は0。

## 入力・実装・§10.5

- 保存13round信号：`f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53:docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json`。新mean・reference・step作成0。raw/corrected/reference/normalization/bias/uncertaintyは保存値と13/13一致。
- 整数列出所：`b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json#cases[8].allocation`。labelで一意に選択し、元ID・整数列を照合。現幅のsignal・physical ID・価格を粗幅へ付け替えていない。
- 新配分ID：`B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01`。評価入力は278fb5214の粗幅Q3入力を基点に、case label・配分ID・整数列・count-only finite bound・現幅比較locatorだけを変更した。
- 元Z1＋T1 decoder、DF10 solver proxy、IID deterministic raw、全事象平方energy loss、exact τ、tie/zero HOLD、DP上限と残余loss、binary64 engineering allowanceを維持した。物理target・実sampler lawへの移送やglobal RMSEの保証は追加しない。

[DELTA_CHECK.json](DELTA_CHECK.json) は前回preflightを継承した関連D/B/A公表namespace・旧別名・後続metadata・同ユーザーjobの差分照合。受入可能な同候補・同配分・同契約Q3は確認範囲で0。現幅A_SHAPEは幅/信号が異なり、粗幅B_PRICEは配分が異なる。後続Pauli LD3の保存結果と進行PR DF LD7 jobは別候補で、今回へ流用しない。未参照host/archiveの不存在は主張しない。他担当を停止・変更・一括凍結していない。

成功済みの `run_p7_y4_coarse_q3_recovery.py` の監督・保存部分を同ファイルの `run_case` へ抽出し、新 [caller](../../../scripts/run_p7_y4_coarse_a_shape_q3.py) から再利用した。元PF worker・scientific API・kernel・Accelerator・guard・decoderは無変更。旧entryのinput gate・出力・再入拒否も保持。旧callerの初期化先だけoptional outを追加し、新診断を旧出力へ書かない。新しい価格／信号／推定pipelineは作っていない。

人工・metadata [8 test](TESTS.txt) はPASS（3.288秒）。新3件は保存整数列/候補/axis/order/信号の拒否・固定execute入口の一回dispatch・共有helperから実toy PF workerの保存/再入拒否。旧5件は本番runtime/AST constructor・実toy PF worker保存・exact DPのloss/質量/残余/状態同値・worker/task分割・scope境界を再利用した。実H4データを人工科学callへ代用せず、primary agent自身の検査を独立レビューとは呼ばない。

## Runtime・原API・数値gate

実行SOURCE **`1cf462e3ae1f0bb16cf2f488d830cc8595909c34`** にcaller/helper・INPUTS・preflight・テスト・計画を科学実行前に固定した。実行環境は `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`、実体は `/home/AbeHiromu/projects/h6-environments/20261011/python-3.11.0rc1/bin/python3.11`、Python3.11.0rc1／NumPy1.26.4。SOURCE・runtimeはINPUTSおよびEXECUTIONへ記録した。

元kernel710aa69b、accelerator9e294778、code snapshot b4296b110を再利用。実PF constructorで期待/実AST hashはともに `3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8`。AST正規化・期待hash置換・guard無効化・許容差変更0。Python3.12で停止した旧UNKNOWNも、前回成功/不適合の原結果も保全した。

- 実科学API **1件**、API_RETURN[0]とRESULT.record完全一致。取得/monitorエラー0、102記録child終了entryすべてexit0。外側実行processもreturncode0、signalなし。
- 粗幅自身の保存信号で `VERIFIED_FINITE_BINARY64_SUPPORT`、検査pair **296,579**。
- total DPは全13round・**27,206,693**遷移。残余probability約 **1.938611686445849766331674e−21**、残余loss上界約 **8.356283177116105312770643e−20 Ha²**を0にせず元上下界へ会計した。質量partition・残余loss加算をexact検算した。
- 原API・partials・RESULT・resource/segment/progress・child終了・PERSISTED/SUPERVISOR receiptを [run01](run01/) に保存。取得・数値gateの正常完了とモデル適合を別々に記録した。

## 資源・並列化

既存8-worker/batch8 DPを再利用し、末尾はtask数に応じてworker数を減らした。新しい高速化ruleや恒常的CPU/RAM上限は追加していない。事前観測はaffinity64 CPU、available RAM511,559,128 KiB。別候補の稼働processへ干渉していない。元の協調stage/round logと親monitorを使い、periodic faulthandlerは0。

| 範囲 | 実測 |
|---|---:|
| 外側実行・終了保存までwall | 91.182410 s |
| Q3 supervisor wall | 89.807685 s |
| worker API/RESULT/資源保存までwall | 89.772090 s |
| worker CPU | 92.608368 s |
| child CPU合計 | 48.466461 s |
| process-tree CPU合計 | 141.074830 s |
| worker peak RSS | 409,684 KiB |
| 最大単一child peak RSS | 313,444 KiB |
| 定期観測tree RSS最大 | 2,773,708 KiB |

tree RSSは観測値で厳密な同時peakではない。B_PRICEとは整数列・DP支持が異なるため、所要時間差を高速化倍率とは呼ばない。実行中のSOURCE変更・科学retry・追加配分0。

## Cへの固定引渡し・STOP

- 実行SOURCE：**`1cf462e3ae1f0bb16cf2f488d830cc8595909c34`**、本ディレクトリINPUTS.jsonとcaller/helper。
- 新raw公開SOURCE：**`442a386bc6985652fc6b65548f714829bb19d375`**、本ディレクトリrun01/。
- 4点projection code：**`ade91dcf2570f9240be1dde34c9e40e2bfa30d37`**、本ディレクトリsummarize.py。新risk callは0。
- C結合表の固定locator：**`8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json`**。同commitのSUMMARY.jsonに4点のexact両端・元result locator・判定を保存。

Cへの状態は **Q3適合・価格待ち**。粗幅総費用は **null**。同じ粗幅のB_PRICE不適合という旧結論は配分限定でそのまま保つ。新A_SHAPEの適合は、粗幅全体の最適性・最小N・十分N・物理的global RMSE・G_qualified・winnerを意味しない。粗幅固有の価格規則や総費用は未取得で、現幅係数や細幅S2の規則を流用しない。

ロードマップ差分は「粗幅32h/5に固定配分のQ3適合点を1件追加、価格待ち。B_PRICEのFAIL・旧UNKNOWN/STOP・価格証拠は不変」。本体の更新は統合担当へ渡す。旧SOURCE・input・raw・凍結モデル・grant/ledger/marker/KEEP/STOPを保全し、専用branchへcommit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。新信号・reference・compile・価格fit・別配分Q3・N/G/winnerへ進まない。
