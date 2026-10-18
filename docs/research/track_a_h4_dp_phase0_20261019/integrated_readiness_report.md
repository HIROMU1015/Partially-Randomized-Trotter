# H4 決定論対照完成・PR独立確認 Phase 0統合報告

2026-10-19 JST。`PHASE0_PREPARED_MANDATORY_STOP`。
技術準備・保存値のmetadata照合・人工テストだけを実施した。
新しい実H4回路生成/compile/sampling/signal/SCF/DF/state/reference、
保存raw標本の統計集計、bootstrap/power/nの数値計算、費用再会計・新順位はすべて0。

## 結果と承認境界

|項目|Phase 0の到達点|次に必要な承認|
|---|---|---|
|D登録|打切り96＋full8＝104、直接実測16、未取得88|不要（metadata照合済み）|
|D取得計画|55候補110軸、最大220 control。33件は全10 endpointの保存不適格|D科学実行・決定論費用会計1 pass|
|P代表|保存winner 7候補＋境界K相方2、主10＋診断4 endpoint|P実行前に集合・契約を最終seal|
|P標本数|保存mean/SE/n/shotの転記、式と感度設計を準備。全nはnull|S_STATS：保存統計値からの数値設計1 pass|
|P固定PF|完成D会計の各ε最小・辞書順tie規則|D結果受入後に固定|
|P sampling/compile|予算式と専用namespace/schema/STOP、lazy adapterを準備|最終n/PF/seed/SOURCEへの独立したP承認|
|raw統計再集計|実施なし。この設計には必須と判断していない|必要が生じた場合だけ別S_RAW承認|

33候補の除外は保存された経験的適格集合内の論理であり、厳密biasや物理的劣位の証明ではない。
したがってD完了後も「104候補すべてを直接compileした」とは言わない。
全打切り単価自体の完成を別目的にするなら176軸案になる。本固定案は110軸で、無断拡張しない。

## 資料・入力・既存結果

指定6 commitを実Git blobから取得し、commit:path/SHA/bytesを保存した。
H6確認原本321cd8c3…、H8旧確認の文脈・新n48資料、native compiler profileも取得した。
各資料の読取範囲と原本byte受領を区別する。来歴はsource_inventory.json、
既存SOURCE11 fileとprovider236 fileのSHA照合、native private入力9件のSHA照合を記録した。
全runtime34,273 fileは保存manifestを保持し、Phase 0での全runtime再検証済みとは主張しない。
将来の科学起動前後に実行closureだけをfresh照合する。

Hamiltonian/state/snapshotはscientific_binding.jsonに固定。旧full PF8の軸別metricsと
追加8の完了D結果を結合した。ordinary controlの旧CSV行は混入させていない。
geometry6距離・ordinary controlと、1 Å native symmetric-directionalを再利用同一と認定しない。
follow-upのT=.8保存再利用は測定件数へ二重加算しない。rank10は別感度系列。

旧D/A/Nのgrant/one-shot/native start/exit/result/terminal/final gateと実入力を
private保全baselineへ記録し、終了時にbyte不変を照合する。
新Aの固定spec/payload/result/terminal/one-shotも保全する。
他研究のHEAD/statusや論文統合worktreeの変更は凍結対象にしない。

## 実装と人工検証

`prepare_h4_dp_phase0.py`は固定資料取得・JSON/CSV転記・ID/件数・SHA照合だけを行う。
`h4_dp_contract.py`はDRAFT/INVALID gate、元seed階層、paired schema、将来会計の純粋関数。
`h4_dp_science.py`は承認後だけproviderをloadするD/P adapter。
`run_h4_dp_verification.py`は承認・SOURCE/input/runtime/provider・one-shot・固定task数を検査し、
spawn executorでown workerだけを管理する。科学的に必要なnative接続の実検証は今回行わず、
人工compiler/metrics・人工paired標本でロジックを検証した。

検査結果・件数は同artifact directoryのtechnical_validation.jsonを参照する。
これは作者Codex自身の技術確認であり、別担当による独立科学review・CI・外部再現ではない。
metadata抽出の初回に旧metricsのordinary行も含まれていたため、明示的control filterを修正した。
修正前には抽出が例外で停止し、科学実行はなかった。最終結果はsymmetric行だけを使用する。

## 未確定事項と解決手順

1. D：ユーザーが55候補110軸と限定決定論会計を承認。新しいone-shot/実効資源とSOURCE/runtimeをsealし1回実行。
2. D終了：全行・native回収・最終SHAを受入しlocal保存。許可された公開範囲だけ処理してmandatory STOP。
3. S_STATS：ユーザーが保存済み統計値とD費用からの数値設計と14固定pairの基準方向比較を別承認。
   raw解析・候補再選択なしでn/感度表とP前の方向を1回算出。Phase 0ではいずれも未実施。
4. P最終化：Codexが規則どおりPF/n/予算を固定。候補固有のlaw metadataが不足するvisibility値はnullで明示する。
5. seed：既存Aと旧除外台帳は取得済みだが、geometry等の網羅性と最終seed衝突検査はpending。
   新SOURCE/nが決まった後にmetadata inventoryを完成し、全master/step collision receiptをseal。未完了なら起動拒否。
6. P：ユーザーが最終契約を別承認。固定nだけ取得し、順位の方向によらず保存してmandatory STOP。

通常の技術修正や人工検査について追加科学reviewを繰り返す必要はない。
重要な科学条件・主要仮説の変更だけを研究判断へ戻す。

## 保存場所と公開

本6文書：`docs/research/track_a_h4_dp_phase0_20261019/`。
公開可能なmetadata候補：`artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/`。
private原本copy・path・seed参照・INVALID grant・詳細preservation：専用worktreeのignored directory。
local commit/SOURCE SHAはtechnical_source_binding.jsonおよび最終チャットに記録する。
今回はpushしない。公開承認の有無とremote照合を、local commitの成功と混同しない。

**mandatory STOP。次の担当はユーザー：Dの110軸契約を審査する。CodexはD/Pを自動開始しない。**
