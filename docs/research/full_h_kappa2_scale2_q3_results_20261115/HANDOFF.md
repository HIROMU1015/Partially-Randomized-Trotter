# Full-H κ2・2倍shotのQ3結果：B/C引渡し

正本は[INPUTS.json](INPUTS.json)、`run01/FULL_H_KAPPA2/API_RETURN.json`（原API return）、同directoryのRESULT・3種PARTIAL・RESOURCE_FINAL・PERSISTED。実行SOURCEは**8a8871021b83c139479d0f47d24d8304145e8cf4**、callerは`scripts/run_full_h_kappa2_scale2_q3_fixed.py`。結果公開commitはこれと区別し、最終REPORTと固定URLで示す。

- [SUMMARY.json](SUMMARY.json)：今回のモデル閾値判定、MSE上下界、moments、残余、zero/tie・全13round、数値gate、資源・保存状態。
- [COMPARISON.json](COMPARISON.json)：a82b0f0の保存control行をそのまま添付し、新配分と並べる。旧riskを再評価せず、倍率・寄与率を新規推定しない。
- [C_RESULT_JOIN.json](C_RESULT_JOIN.json)：同じ元/total configuration、binding、physical candidate、round/axis、新allocation、整数N、law selector、科学SOURCEの26軸索引。価格取得やGを実行した証拠ではない。
- [SCOPE_AND_PREFLIGHT.md](SCOPE_AND_PREFLIGHT.md)：計画後の重複・SOURCE照合、整数包含、既存worker/monitorの人工検証。テストは実H4評価の代用ではない。

## 保存signalとdecoderの対応

元configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、total configuration `265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`、binding `579c5de23a2336d402a1d5b195061b34b3c7a1b49e99ea2e1a674c1a086189a5`。

ce6aacfaのFull-H κ2/K2・one-body込み13round packetをSOURCE付きの既存Full-H routeで読み、元grouping・law・scalar・normalizationを維持した。信号再取得、rawのcorrectedへの置換、normalization/scalarの再適用はしていない。Z1＋T1 total decoder（kernel710aa69b）と高速backend9e294778を再利用し、新しいdecoderやrisk kernelを作っていない。R448・PR-DF・Pauli・PFは今回の評価対象ではない。

allocation ID `FULL_H_KAPPA2_TOTAL_Q3_CONTROL_SCALE2_PROPOSAL_01`、X/Y各軸`[128,120,112,104,94,86,76,68,60,50,42,34,24]`。名前にPROPOSALが残るが、この固定配分・新scopeは今回ユーザーが採択した。USER_SUPPLIEDの固定診断vectorであり、必要N・最適N・原論文nominal Nではない。

solver proxy −2.1663874486347634 Ha、保存ideal finite-K raw/IID二項モデル、EXACT_AFFINE_OF_BINARY64_RAW、全事象平方loss、exact十進εE²による直接モデルMSE filterを使う。旧Minkowskiは別参照で、AδEを加算しない。元packetのground認証・資格欄は変更しない。

## 資源記録と保存状態

workerが親monitorへのreturn前に原API return・RESULT・RESOURCE_FINAL・PERSISTEDを保存する。親のCOMPLETIONとSUPERVISOR_RECEIPTはその保存状態を示す。数値判定と保存状態を分け、UNKNOWNや部分成果をPASSへ置換しない。

RESOURCE_FINALのworker CPUは当該worker、child CPUは回収済み子processのCPU合計、process-tree CPUは両者の和。worker peak RSSはworker自身、child peak RSSは最大単一子の値であり、合計ではない。親COMPLETION/RESOURCE_MONITORのtree RSSは同時点で自分のsubtreeを足した観測ピークで、サンプリング間の厳密最大ではなく、共有pageの重複を含み得る。wallにはmetadata・数値計算・原結果/資源保存を含む範囲を明記する。欠測があれば欠測として保持する。

zero/tieは全関数decoderのHOLD/argmin規則で出力を定義し、旧partial規則の未解決確率と区別する。今回m4/5/9/10/11のshot数の偶奇が変化した。zero/tieの変化を測定回数だけの因果効果と断定しない。追跡massを残余込みの厳密全履歴率と同一視せず、m0を含むfrontier pruning・principal guardの残余mass/lossを会計する。

## C側・ロードマップへの利用範囲

物理回路は旧controlと同一なので、価格側は元physical ID・law・compiler/lowering・full-wrapper・SOURCE/cohortの一致を検査して索引を結合できる。ただし今回価格を取得・推定しておらず、Cのguardと資格検査を緩めない。allocation変更だけを理由に価格receiptを無条件に付け替えない。

今回の閾値判定はこの1vectorに条件付くモデル結果。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。ground/target包含・実sampler law移送・厳密数値certificateは未資格であり、モデル適合が得られても物理的RMSE保証や最小Nへ昇格しない。

ロードマップ本体は統合担当が更新する。今回の結果・保存状態・残る不足はREPORTの固定SOURCE付き記述を引き渡す。新vectorの追加、4倍配分、round別探索、frontier/ρ/scope/backend変更、価格/G/十分N認定には進まない。**最大1call終了又は重大異常後mandatory STOP。**
