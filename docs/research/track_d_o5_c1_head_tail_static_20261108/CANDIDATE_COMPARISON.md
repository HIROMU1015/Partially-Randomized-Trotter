# 接口候補の比較と判断

分類は静的調査の状態を示す。POTENTIAL_GAINは科学的GOでも追加削減の実測でもない。

| 候補 | 機構・既存との差 | Strong / Reference | 増分利益・負担・失敗条件 | 分類 |
|---|---|---|---|---|
| head内部のrelative Gaussian | 既存transitionと同じ | 双方で有効済み | 再実装は利益なし | ALREADY_HANDLED |
| tail同basis sharing | 既存runを再利用 | 双方で有効済み | 中心の順序を変えない。tail異frame全体を新最適化する案へ拡張しない | ALREADY_HANDLED |
| 同block0のstep半sweep融合 | directional＋uncontrolled→ordinary full、phase集約 | Strong q−1、Reference内部q−2 | 既存最適化。global cut損失は維持 | ALREADY_HANDLED |
| 同frameのhead-tail閉包削除 | G→G†を消すだけ | 両方式に適用し得る | P0重複が強く疑われる。現在のH4残存有無は未確認、これだけの新builderは推薦しない | UNKNOWN |
| **head出口→tail先頭runの異frame relative橋** | 閉じたG_H/G_T†の組を既存phase-corrected transitionへ置換。新合成器なし | Strongのbasisは非制御なので中心modeを保ったまま適用。Referenceは各physical segmentで同じ規則を適用後actual inverse | raw横断入口は未接続、数理条件を明示できる。P0増分は不明だが小さな既存人工例で判別可能。full-wrapperで悪化/差0も受容 | **POTENTIAL_GAIN：最初の人工判別候補一種類** |
| tail出口→reverse-head入口 | 上と逆向き、event phaseとの配置も扱う | 双方に適用し得る | 合法性条件は共通だが初回を広げない。今回次案の実装対象に追加しない | POTENTIAL_GAIN（次順位・未採択） |
| modeを無視して中心をまとめる / gaugeを落とす | 対象operatorやbranch phaseが変わる | 公平比較不能 | 全入力coherence/χを壊し得る | INELIGIBLE |
| global cut越しのGaussian相殺 | 別wire/registerとCSWAPを跨ぐ | 現Referenceでは不可 | gadgetやcutの変更が必要。今回C1局所案の範囲外 | INELIGIBLE |
| 一般H4でのP0後の残存/削減 | 実際のcompiled回路の比較が必要 | 双方 | 保存QPY/価格の新解析を許可されていないため未取得 | UNKNOWN |

## 四構成の公平性

| 比較 | 確認する量 | 今回の状態 |
|---|---|---|
| S₀→S₁ | Strong単体での各資源差S₁−S₀ | 未測定 |
| R₀→R₁ | Reference単体での各資源差R₁−R₀ | 未測定 |
| R₀−S₀ | 既存強い対照に対する差 | 既存報告のみ、今回再解析0 |
| R₁−S₁ | 両方式へ同じ合法規則を与えた後の差 | 未測定 |
| (R₁−S₁)−(R₀−S₀) | Referenceの相対的な追加差 | 未測定。増減どちらもあり得る |

RZ count/depth、CX count/depth、total depth、size、qubitsを別々に比較する。Gaussian橋の変更自体に追加registerは不要だが、Referenceの既存reference register、CSWAP、χ補正はfull wrapperに含める。RZ→Tやtopology-free depth→実行時間への換算をしない。主資源や新しい精度・成功率をここで採択しない。

## A/B/C1の現在地

Aは既知round-only mixedを超えるpath選択利益が未確認。Bは保存H4/CDF8/LD2/S2/s1/r32/K0/h=.1/q4のcalibration16本が全て中央で、追加H4 compile保留という既存結果を維持する。C1は既知relative Gaussianを未接続のhead-tail接口へ適用する一種類の人工判別余地が残る。独立新機構や全PR利益の認定ではない。

今回の提案は**C1継続（最小人工判別の採否をユーザー/GPTへ）**。その人工比較でP0が利益を全回収する、両方式とも差0/悪化、または正当性条件が保てない場合、C1を保留/終了候補として追加実装を増やさず、O1/PF、O2/law・return、O3/誤差・総資源との共同最適化へ重点を戻す案が自然である。その共同作業も今回開始しない。
