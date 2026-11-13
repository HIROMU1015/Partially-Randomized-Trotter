# A → B：S3の次に必要なRMSE証拠

2026-11-13 JST。[REPORT](REPORT.md)／[ERROR_GAP_MATRIX](ERROR_GAP_MATRIX.md)／[有限案](NEXT_SCIENCE_PLAN.md)。新しい統計処理・物理評価器は追加していない。Aの追加mean計算をB統計評価の一律前提にしない。

## 1. 固定結果としてそのまま引き継ぐ

- S3結果 `2084791b1b09cff65e07fa8566bb734e1e6bff8e`、計画ee1d456、code ab12b691、caller7486b915、入力bf515a976。
- 固定43,819回の条件付きRMSE **.0006847202619833286 Ha**、未解決0、97.5% CP上端9.999788982664716e−5。
- 主population RMSE CI=null、副MSE envelope=[0,28.963512732533932] Ha²。副評価はloss感度で、全energy推定器の認証ではない。
- canonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、H4 DF10/S2/L_D1/K2/ΔPF=h=.1/M12/元exactκ・N配分を保全。
- A bba150834の全13mean/誤差契約と169d3316の混合SOURCEを保持。m3/m4観測norm欠測、empirical allowance、未知actual/state/ground誤差を0にしない。
- C d72dc4は生成構造のSOURCE対応を確認したが、実H4 P/Q/F/G・PCG64 joint距離はnull。価格S/Cのlaw資格をsignal側の保証に付け替えない。

旧S3で全観測がDECODEDだったことは、branch正解・population未解決率0・十分Nを意味しない。今回旧SIMULATION.json.gzを開き直したり、loss/CIを再集計していない。

## 2. 最初にBが確定する契約

1. **全事象の対象**：主UNDEFINEDを維持したままの条件付きriskを報告するか、energy補完familyの一様risk envelopeを定義するか。特定の未解決energy fallbackを選ぶ場合は新推定器としてユーザー判断が必要。Aは選ばない。
2. **populationの精度**：解決分と未解決lossを含む上側MSE、support、confidence、数値rigor。標本条件付きRMSEだけに成功ラベルを付けない。
3. **主張のlaw**：保存binary64 plug-in、解析的ideal finite-K、実sampler/実backendのどれを扱うかを区別。既定ideal modelを勝手に変更しない。
4. **経路**：固定raw全riskを直接評価するのか、元εqpe/εtrotの個別達成を示すのか。前者の成功は後者の成功とは別。

support・CI法・couplingの人工検証までは技術作業。既存43819 trialへの新CI、保存rawからの確率再評価、解析的count分布への数値適用は新科学再解析なので、別途SOURCEと有限scopeを明示して承認を得る。S3の反復追加・再seedはこの調査から許可されない。

## 3. Aから提供できる物理側の接続

**保存HDFのtarget**：元generateはsector最小固有対を数値的に選んでいる。最小の認証案は、同じ36次元HDFの変分上界と正定値下界によりE0,DFを狭い区間で囲むこと。新SCF/DF/state生成は不要。狭いtarget包含だけで直接riskを移送する経路があり、保存stateの厳密ground同定・overlapを全て先に証明する必要はない。

**rawの適用誤差**：13meanのuはempirical。認証raw座標enclosureを必要とするclaimでは、Aが新しいverified数値経路の有限案を提示する。uをそのまま確率intervalにしない。新enclosureを取得しても元mean/result/m3/m4欠測を書き換えない。

**PF/finite-K**：保存rawにこれらの効果は既に表れている。Bの同raw全riskに同じsignal biasを再加算しない。個別energy budgetを求める場合だけ、基準推定量とcouplingに対応する別L2評価が必要。

## 4. 認定の判定形（数値適用なし）

全事象energy estimator Z又は明示energy補完family、固定target E*、評価law Pとplug-in Qについて、同じ有界loss≤U・TV(P,Q)≤Δ・proxy距離≤dtargetがある場合、十分条件の形は

`sqrt(R_Q^+ + U*Δ) + dtarget ≤ εE`。

ここでRQ⁺はBの有効な**population**上側risk。S3の点推定や主nullを代入しない。confidence付きなら同confidenceに限定し、認証なしの数値観測を数学的boundとしない。実装/law/数値項が未取得なら、その部分の資格は未取得のまま。

εqpe=εtrot=εE/2の個別評価には、同じ全事象出力規約のZ0とZのcouplingが必要：

`||Z0−E0,DF||₂ ≤ εqpe`、`||Z−Z0||₂ ≤ εtrot`。

単一位相・saved-state reference・attenuated referenceでは基準lawが異なるため、A/Bで別々に意味を定義しない。target/law/数値誤差をどの既定budget内へ入れるかも明示する。独立性がないこと自体はMinkowskiの障害ではないが、異なる周辺lawをcoupleせずL2差を宣言できない。

## 5. 次の担当と承認範囲

**ユーザー**：既定targetが保存HDFであることを維持し、上記loss/claim/評価経路を判断する。DF前Hmolまで求める場合は追加誤差契約を別指定する。次の保存再解析・target包含・C F0のうち必要な有限scopeを選び、公開SOURCEとともに別途承認する。

**B**：loss/support/population上側評価法の技術案を仕上げる。最小科学候補は既存43819 trialの1回の新上側評価で、新draw0。ただし有用な精度に届く保証はなく、科学的未達も有効な結果とする。反復追加を自動提案・実行しない。

**A**：同HDF1個のground包含の技術準備。必要な場合だけraw認証を別scopeにする。**C**：既存F0案とjoint-lawの限界を所有し、Aで重複実装しない。**D**：今回のRMSE接続に価格作業は不要。

現N packetの十分N/global RMSEはnullを維持。公開照合後Aはmandatory STOP。調査結果を新科学計算の承認として扱わない。
