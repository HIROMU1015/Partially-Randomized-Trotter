# 未実行案：残15セルのsignal screen再開

2026-11-13 JST。**今回は実施しない。ユーザーが数値方式と有限scopeを明示承認した後に開始する。** 基礎は元PLAN256caad34、元4局所record699178305、今回二経路診断ade6438f8＋本結果公開SOURCE。旧STOPを削除しない。

## 推奨する数値方式

Full PFとB0Iには`pf_power_stability.expm_step`の直接expm macro-step＋matrix_powerを使用する。S2/標準Y4/M5 best、各signed weightsのbinary64値、forward/reverse順、負時間、PF幅・q、scalarの旧規則を維持。B0Iだけのtail identity scalarをtotal Tで一回追加する。数値gate1e−9をそのまま用い、失敗値を先に保存してSTOPする。

直接expmは人工48条件と実代表1点で有効だったが、実M5/B0I全体の資格は未取得。固有vectorの再直交化・unitary投影やstate正規化で救済しない。Full-H pure RTEはfinite-Kの理想平均そのものを評価するため、指数関数に置き換えず元polynomial power/normalizer/独立raw/correctedを使用する。

## 保存を再利用する範囲

元保存40セル、run02のFull PF/m6の4セルをそのまま保持し、再計算しない。今回direct F2-COARSE/m12の1点もphysical identityは一致するため、数値qualification付きのmeanとして再利用候補。元spectralの失敗値は診断専用である。

今回direct点には共通screen用の経験的allowance等を生成していない。packetへのfield copyだけで足りない新しい診断を行う場合は、この1点に対する**保存値の限定scalar bridge**を明示的な実行範囲として含める。旧4点・元40点の再解析やN再評価へ拡張しない。

## 残15セル

| 対象 | round | 新complex mean上限 |
|---|---|---:|
| Full PF S2/Δ.1 | m12 | 1 |
| Full PF標準Y4/Δ.1・.4 | 各m12 | 2 |
| Full PF M5 best/Δ.1・.4 | 各m6・m12 | 4 |
| B0I/prefix7/Y4/Δ.4 | m6・m12 | 2 |
| Full-H pure RTE K2・K4 | 各m3・m6・m12 | 6 |
| 合計 | 元PLANと同じ候補 | **15** |

Full-H normalizer最大6、新reference0。保存pairは同じm3/m6/m12を再利用。新SCF/DF/state/ground、sampling、回路compile、N/RMSE/C/Gは0。元run02の空例外Full-H1件は、履歴上のnormalizer call数未確定を保持し、今回の新scopeで勝手に過去のcall数を確定しない。

## 実行前の最小技術差分

承認される場合は、既存screen evaluatorのPF macro-stepをdirect expmへ明示接続し、経験的allowance、reference/binding、scalar一回会計を再利用する。新しい固定SOURCEで15候補だけを列挙し、旧SOURCEを上書きしない。failure recordにnorm・unitarity値、例外型、未開始/取消状態を残す。これらは過去の欠測を遡って埋める修正ではない。

source/input/reference不一致、非有限、closure/norm/unitarity gate失敗時は部分保存しSTOP。開始済み有限workerは保存し、未開始候補を新たに始めず、候補差替え・retry・閾値緩和はしない。物理biasが大きいだけで全RMSE不適格と判定しない。

今回の二経路診断は再開の技術的根拠を1点分与えるが、再開承認そのものではない。**ユーザー**が方式・15mean/6normalizer/reference0と必要なら新direct1点scalar bridgeを承認し、**Codex A**が固定SOURCEで実行、**B**が候補比較の材料へ接続する。全13roundや最適化はさらに別scope。
