# 候補1の最小検証案：未承認・非実行仕様

[候補の機構と不確実性](candidate_comparison.md)／[判断・承認](handoff.md)。**これは研究課題選定用の提案であり、実験契約、採択記録、authorization、実行可能planではない。** 入力・精度・confidence・資源・成功閾値を今回固定していない。

## 最初に調べる一点

Koizumiほか2606.30601v2 S8の既知charge/incidence共分散classにDF因子を直接入れた最強対照が、候補と同じ情報・縮約順・操作数・storageに達するか。

ユーザーが別途この限定検討を依頼する場合、まずCodexの紙上方法差確認だけで終えられる形にする。

1. 既知SOURCEの1-/2-/mixed共分散classと元energy係数の対応、scalar・normal-order項、粒子数/holes適用条件を明記する。
2. 候補側と既知側へ、同じDF factors・state情報・precision・cache・streaming・標準tensor contraction最適化を与える。dense全共分散表を強制する対照は置かない。
3. 必要な高次RDMまたはその線形結合と取得費を列挙する。未知state情報を定数time oracle・exact classical ground vectorで無料供給しない。
4. 新しい縮約則があるなら入力class・rank条件・正確性・bit/storage/操作数を紙上で示し、既知routineへ単なる変数代入・通常のfactorized contractionで同じ結果が得られるかを比較する。

最初の成果は新しい数値ではなく、**同等の既知手順が構成できるか、独立した具体的差分が残るか、証拠不足か**の説明である。現在その説明を得たとはしない。新数学保証の構成・証明が必要になったら、その範囲も別途指示を受ける。今回の文献整理から証明作業へ移行しない。

## その後の条件付き段階

| 段階 | 必要になる証拠・公平性 | 不適格またはSTOPとなる場合 | 必要な承認・担当 |
|---|---|---|---|
| 紙上方法差 | 同じ共分散・同じ情報を両法が扱う具体的演算列、rank依存、state情報の取得量 | 既知classの普通の縮約と同等なら独立アルゴリズム候補から外す。engineering比較を希望するなら別目的として扱う | ユーザーが限定技術調査を別途指示、Codexが実施 |
| 科学的価値の判断 | 差分が元energy推定の何を改善し、何を改善しないか。定理/応用/実装の区分 | 古典後処理改善をshots/T数改善へ読み替える必要がある、原Hやstate準備が欠落する | ユーザーが研究課題を選択。重要判断が必要なら開始承認後にGPT |
| 人工検証の準備・実装 | 係数・符号・mixed covariance・定数shift・zero/cancellation例、独立occupation reference。強対照も同じ最適化 | 欠けた高次相関をproxyで補ったまま保証を主張する、表が巨大化する、差分不成立 | SOURCE実装・人工テストの別途承認後にCodex。現段階は未承認 |
| 最小科学pilot | 元H/DF誤差、state access、共通精度・confidence、全取得費、低rank条件と不利なcontrolを事前に選定 | 結果後のrank/cell選択、無料exact-state、遅い対照しか使わない、曖昧なcoverage | ユーザーが契約を別途採択し、科学実行も別途明示承認した場合だけCodex |
| 量子総費用接続 | copy準備、Gaussian測定の合成・routing、全shots/失敗予算、古典費と償却。同じtaskの強い測定法・coherent法 | queryの改善だけ、原H/ground state誤差が未取得、実装費が利益を相殺 | native/状態準備/総費用作業を個別承認。現在は全て未承認 |

未知の相関stateで難しければ、その事実を残す。容易なSlaterだけの一致を一般相関stateへの有望性にしない。新しいuniform分散上界と、与えられたstateの正確な分散評価は別の目標であり、どちらを行うかもユーザー判断事項である。

## 実行計画へ進む前に未決の条件

入力class・n/η/R/rank尺度、元Hへのmodel誤差、copy/state access、相関取得方法、精度とcoverage、confidence scope/失敗確率、比較baseline・費用metric・償却回数、最小実行件数と打切り条件。今回具体値・実行ID・hash付きinput・one-shot条件・resource capを新発行しない。

将来承認された処理では共通指示§8.4に従い、ワークロード・実空き資源に応じてclass/fragment/task並列、factorized contraction・cache等を検討する。CPU/RAM/worker/wallの恣意的上限や実測していないspeedupを提案しない。共有環境や他者jobを変更しない。科学call・出力・ログ・instruction予算と旧契約は別途保持する。

**今回この表の段階は一つも実行していない。新しい検証契約・実験への移行は禁止したままmandatory STOP。**
