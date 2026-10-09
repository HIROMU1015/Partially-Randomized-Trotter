# G4-C：次の独立検証の設計比較 — 実行なし

G4-A/B後のGPT判断用。どの研究経路を選ぶかはGPT側の担当。
旧STOPとG4 STOPを維持し、以下の新条件を取得・採点・実行していない。
Tをprimaryに維持し、1Qは探索的・補助座標。新しい5%/10%を旧結果へ適用しない。

## 現在の証拠が判別したこと

x=1/4では、同一保存J1 lawが今回認証したB2 digital classのT・readout付き1Q下界を下回る。
ただし、I1 Pauli collectionを使用するmatched CTSの一つのlawは、そのJ1 lawをT/CX/1Qで下回り、
workspaceは同じ。x=1/8でも、保存J1 1Q-selected lawについて同様のCTS対照が存在する。
これは有限CTS poolの構成例であり、全CTSの最適性や全J1の劣位の証明ではない。

このtoyではI1 collectionを実際に取得できる。I0限定を実世界の情報制約と読み替えない。
したがって、次の設計で同じtoyのJ1の勝ちをもう一度探すことに情報価値があるとは限らない。
B2限定分離の定理と、I1対照を含むtaskでの採用価値は別の問い。

## 次scope候補の比較

| 設計候補 | 判別する未解決事項 | 最小限の結果前固定事項 | GPT判断への情報価値・限界 |
|---|---|---|---|
| 共通wrapper費用を戻す | 同じlawのshot増と一回路費用低下の交換が実taskで残るか | 一つのtask、同じstate preparation/readout、各resourceの一shot費用をsourceから取得する規則、全arm同一会計、有限candidate集合 | PR最終taskへの接続には必要。ただしx=1/4の保存CTS 1Q lawはJ1 1Q lawよりshotsも全費用も小さい。非負の同一共通費用だけではこの二点の順位は反転しない。J1救済目的のoverhead gridを作らない。 |
| 異なるp・basis構造でselectorを移送 | degree-local構成の価値が一つのtoyの偶然か、指定構造から予測できるか | 新しいstructural contrastを一つ、変更する構造と不変条件、selectorを旧証拠だけで固定、原ordinary/PTSC/A/J1とsame-target CTS、取得上限 | 一般性と強い対照後の価値を同時に判別できる可能性。新pやbasisを試して勝つ条件を選ぶ方式は禁止。構造ごとの意味と独立条件の選択はGPTが決める。 |
| 合成費用の離散性への頑健性 | 特定のcheap angleやsynthesizer seedが交換の符号を支配しているか | synthesizer/tool identity、一つの独立implementation context、同じerror guard/precision、normalizationとnative costを分離する読み方 | primitive費用に依存する結果の限界を調べられる。しかしI1 CTSとの差がある現在、角度を細かく増やすだけでは中心claimを強めにくい。 |
| Pauli/DF情報の取得費用・利用可能性 | I0から得られる構成が、I1 collectionを使う対照に対し意味を持つcontextが存在するか | 入力accessとclassical cost metric、同じtarget、dictionary construction/acquisitionの手順と上限、取得失敗時の扱い | I0主張を続ける場合に最も直接的な不足を調べる設計候補。DF/molecule移送や大型dictionary取得は今回未認可。toyの9+6 Pauli積だけからscale advantageを推論しない。 |

一度に全候補を追加しない。GPTが残す中心claimに応じて一候補を選び、最小stageを別に認可する。
Codexは方法の採択、構造・baseline・成功閾値の科学的変更をここで決めていない。

## 保存lawだけのoverhead診断

x=1/4のA 1Q-selected（n_A=1,022,850）とJ1 1Q-selected（n_J=1,114,220）について、
両axisの各shotに同じresource overhead hを足すと
G_k(J)+2n_J h < G_k(A)+2n_A hの境界は
h=(G_k(A)−G_k(J))/[2(n_J−n_A)]。
保存exact値の再集計でT≈14.93、CX≈0.3124、1Q≈35.35。
これは固定二点のcrossoverであり、再最適化されたfrontierのboundaryではない。
新h gridを実行していない。

一方、x=1/4のCTS 1Q-selectedはn_CTS=970,447であり、J1 1Q-selectedより小さい。
三費用も低いので、各座標の同一非負overheadではこの二点のdominanceをJ1側へ反転できない。
方法ごとに異なる付帯費用を与える場合は、根拠となる実taskが必要。
G4結果を救済するための方法別penaltyを結果後に追加しない。

## 次実行前に必要な契約

GPTが選んだ問い、independent/dev区分、full operator target、情報access、入力取得手順、
candidate/precision/proposal集合、T-primary taskと許すtrade-off、prospective materiality、
tool/source/seed、wall/CPU/RSS/output/call caps、one-shot authorizationを結果前固定する。
今回の小さな差を見てmaterialityを逆算しない。1Q結果をT-primary GOへ格上げしない。

採択せず限定定理・mechanism noteへ区切る選択肢もGPTに残す。
この文書は実装・追加science・次stageを認可しない。
