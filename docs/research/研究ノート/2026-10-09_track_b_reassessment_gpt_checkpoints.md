# Track B：GPT再評価方針の受領記録

2026-10-09 JST。文書のみの方針記録。実行authorizationではない。

利用者から共有された[研究再評価とGPTチェックポイント](../track_b_reassessment_and_gpt_checkpoints_20261009.md)
を今後のTrack Bの前提として保存した。ルートcheckoutの該当文書だけを明示して取り込み、
添付GPTレビューは[原文snapshot](../../../artifacts/track_b_reassessment_gpt_checkpoints/2026-10-09/inputs/gpt_review_20261009.md)
として別途保持した。元ファイルと添付のidentity、今回の文書変更は
[manifest](../../../artifacts/track_b_reassessment_gpt_checkpoints/2026-10-09/evidence_manifest_v1.json)に記録する。

基点はbackend v2の`7f9062975d9f2b09f12cda6e83e7de1e830beac4`。
独立branch `track-b-reassessment-gpt-checkpoints-20261009`で文書だけを公開する。
Track Aとルートcheckoutの既存変更には触れない。

## 採用した進行規則

RA-RTEは現在の主候補として限定継続する。degree-local自由度が既存representationの混合に
独立した資源的価値を加えるかを判断し、数値基盤の修正を継続すること自体を目的にしない。

| 判断点 | 到達条件 | GPT側へ戻す判断 |
|---|---|---|
| G0 | 今回の方針共有 | RQ階層、次の判断単位、研究と技術基盤の区別 |
| G1 | 承認された独立構造監査＋最大8人工LP、または重大な反例・技術停止 | 6頂点の適用class、backend採用、source簡素化、次の最小比較と作業価値 |
| G2 | 承認されたsourceとoff-domain tests完成、登録LP前 | 比較class、upper/lower、query、資源上限、materiality、別実行authorization |
| G3 | 最初の承認B2/B3有限batch後 | strict witness、headroom、勝因、強いbaseline比較の必要性 |
| G4 | 承認された強いbaselineと小型transfer後 | 新規性、PRへの適用範囲、論文claimと最小completion |

G1前にproduction source、新authorization、登録optimizationへ進まない。
G1で不十分でも追加修正pilotを自動で重ねない。旧737点、旧call上限を自動継承しない。
過去のSTOPとconsumed one-shotのno-retry規則を維持する。

## 次の作業前に未固定の事項

- 独立数式監査の具体的な作業契約・成果物・停止条件。GPTの6頂点導出は独立未検証として扱う。
- 最大8人工LPのうち追加高精度2 fixtureの入力、固定backend/runtime/guard、呼出し順と資源上限。
- proof取得失敗と、取得済みの不正証明を区別するcontrollerの仕様。
- 新しい一回限りの作業の明示指示。旧v2 runを続行・再実行する形にしない。

構造監査は理想degree matching、頂点完全性、precision復元、v4変数との対応と限界を対象にし、
登録LP・J1/J2/J3の費用採点を行わない。旧B2へ追加3構成を注入しない。
100桁の入力取扱いと10^-99 infeasibility gapの証明取得を分け、旧失敗記録を消さない。

positiveは共通task/capsの認証済み`U3 < L2`。no witnessはnegative theoremではない。
妥当な元B3 lower `L3`とB2 upper `U2`を得られる場合だけ、共通条件でheadroomを評価する。
`B3 inner minimum`を元B3 lowerとして使用しない。新規性・実用materiality・PR全体への利益は別の判断。

## 今回行った操作

指定文書と添付の読み取り、原文snapshot、方針の索引追記、履歴・manifest作成と公開のみ。
独立数式検算、LP solve、source実装、tests実行、backend build、science/synthesis/GPUは0。
旧source・contract・authorization・marker・結果は不変。
既存458対象のうち現状要約一件だけは既存本文を保持して方針節を追記し、残り457件はbyte不変。
今回の資料公開は次の検証の認可を意味しない。
