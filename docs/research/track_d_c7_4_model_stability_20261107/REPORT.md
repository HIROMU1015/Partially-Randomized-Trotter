# Track D C7-4：最終技術報告

2026-11-07開始、2026-11-08 JST最終収録。保存88 pathの再解析・探索的モデルfit・数値安定性比較・次段階条件案を完了した。詳細科学レビューではない。[価格モデル報告](MODEL_REPORT.md)／[数値安定性](NUMERICAL_STABILITY.md)／[未採択の次段階案](NEXT_EXECUTION_DRAFT.md)／[固定PLAN](PLAN.md)。

実行SOURCE `dd31e1e2878479eabe2e7cc27c21b26c8c7dc794`。初回 `deb92995d693361e3c56de816e8c0efc4c618b65` はC6a完了ラベルの実装誤認で再解析前に停止した。run01のSOURCE/failure/result/STOPを保全し、科学条件を変えずschema対応だけを修正してrun02で初めて科学再解析を実行した。[停止と修正](PREPARATION_NOTE.md)。旧C6a〜C7-3 runは再起動していない。

## 実施内容と結果

|処理|実施件数・状態|
|---|---|
|保存whole-path/price統合|88 path、352 price records、全固定入力/P0/旧event hash照合|
|新探索fit|4 family、各36 response、2/3係数|
|leave-one-q-out|各新familyの5 fold、モデル選択に使う探索CV|
|統計bootstrap|各family256反復、q内whole-path再標本化、H4 samplingではない|
|保存数値再照合|q32最大残差の1 path、保存A/B/V/actual adjoint＋4 observable、15項目PASS|
|人工長系列比較|3/5qubit×3長、6 case、2経路、全行列比較は既存許容値内|
|人工P0 compile|短block2本、raw/compiled全行列照合PASS|
|人工・回帰テスト|distinct18 PASS（修正版7＋不変cache6＋handoff5）|
|新H4 sampling/build/compile、追加標本、旧run retry/resume|0|
|ground solve/DFfit/N/actual RPE/全13-round費用/routing/T/noise|0|

**CX:** 全88保存pathでΔCX=208−224q、整数残差0。部分primitive/boundary説明と有限観測支持に留まり、任意pathのGaussian/cut/CSWAPを含む全P0 bridgeは未取得。

**RZ/size:** paired差は全q affineの2係数で十分小さい探索残差。追加特徴が一律に改善するわけではない。Δsize−ΔRZ−ΔCX=192が全88本で観測されたが、事後的な保存データ解析で全path定理ではない。

**Depth:** paired XのLOQ q平均誤差RMSEは、全q affine→basis-runでRZ depth91.159→22.771、CX depth40.003→13.611、total depth191.272→47.975。strong絶対depthはtotal-run、Reference絶対depthはlane-max-runが良い探索候補。応答別に異なるdesignを混ぜたjointモデルには追加の共分散整合が必要で、総費用モデルとして無断採択しない。全5 family/全資源/方式別の結果を保存し、モデル選択に用いたqを独立holdoutと呼ばない。

**数値安定性:** q32最大observable残差1.1971564746e−10は保存原本のextended再照合でもPASS。旧誤差をnorm再計算だけで除けなかった。人工native gate保持版は既存generic版と同じ結果で、精度改善は未確認。cache/既存高速化との互換性は人工・回帰範囲で確認したが、q64以降の閾値維持・全256列/全support実装誤差boundは未取得。

## 再利用・新実装・保全

C6a describe/統計規約・元affine係数/共分散、C6b/C7-1/C7-3の保存価格・check・seed/event hash、C7-2局所反例/max-plus、C3/C5/StrongBuilderと元P0/phase compiler、C7-3 SmallUnitary/cacheとC6b native executorを再利用した。新実装は88-path入力adapter/低次元fit・CV・共有bootstrap、人工専用native小演算子adapter、明示runnerと保存結果projection。旧guardを緩めたり別分子simulatorを作ったりしていない。

使用した旧公開入力32ファイルと88 eventの前後hash、最大残差pathのNPZ hash、新SOURCEの前後一致を確認した。旧SOURCE/入力/raw/価格/診断/結果/grant/ledger/marker/STOPを変更しない。非公開event/QPY/NPZ/内部CPU/RAM/pathは公開しない。新成果は `artifacts/track_d_c7_4_models/2026-11-07/run02/`。実空き資源を確認して6独立人工caseをprocess並列化し、小行列のnested BLASだけを抑えた。共有設定/他jobへの変更なし。

## 未資格と次の判断

q64〜4096には元凍結版と新q-only版の探索予測・共有係数共分散を別保存した。run/lane版は未来feature lawが未取得で、保証付き期待価格へ外挿していない。model bias/law誤差のboundはnullであり0ではない。古典価格の名目的bootstrap/normal診断をRPE成功保証へ混ぜない。

次候補は数値preflight付きq64/cut32/fresh8、最大32 full-wrapper（先頭path4本を含む）の探索pilot。q128以上も比較した上で近接外挿と数値不確かさを優先した案であり、自動採択/実行はしない。確認的な母平均標本数はtail/variance/bias根拠が不足し確定できない。ユーザー/GPTが、追加の数値経路技術検証を先にするか、停止条件付きq64 pilotを許可するかを判断する。Codexは新しい明示許可後だけ実施する。

**C7-4の成果を専用branchへ公開し、remote SHA/tree/公開bytes照合後mandatory STOP。** q≥64のH4 sampling/compile、全RPE費用、actual成功保証、詳細科学レビューへは進まない。
