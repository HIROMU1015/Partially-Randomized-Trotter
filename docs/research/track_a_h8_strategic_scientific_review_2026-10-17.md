# Track A H8：通常bias・非相殺bias費用比較の戦略的科学レビュー

- 日付：2026-10-17（JST）
- レビュー状態：**完了（Level 1の科学的判断）**
- 開始承認：ユーザーの「レビューを開始して」。
- 対象：`HIROMU1015/Partially-Randomized-Trotter` のPRリソース評価（Track A）。
- 直近の固定結果：通常Phase C `6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086`、非相殺費用比較 `23d46e489af054735d9d328aad0342ecdbddf305`。
- 正式共通運用指示：添付 `research_project_instructions_unified_2026-10-17.md` を実際に全文確認した。
- **この文書は科学的な評価・提案であり、ユーザーによる研究方針の正式採択、追加compile・sampling・保存値再会計の実行承認ではない。**
- 新規科学計算、既済runの再実行、grant発行、repository更新、サーバー操作：すべて未実施。

## 0. 結論と推奨判断

**研究は継続する価値がある。ただし「PRの一貫した優位性を示す研究」としてではなく、強い決定論的対照と誤差会計の選択によって資源順位がどこで変わるかを明確にする、条件付きの数値資源研究として仕上げることを推奨する。**

| 判断事項 | 推奨 | 理由・限界 |
|---|---|---|
| 通常signed biasの位置付け | **主評価として維持** | 固定した対象signalとの差を評価する元の基準。結果が不利だから変更しない。 |
| 非相殺biasの位置付け | **補助的な相殺感度診断として維持** | 誤差分解内の成分間相殺を除く診断。厳密上界や新アルゴリズムの改善ではない。 |
| 現在のPR利益 | **中間の3精度点で、両会計に共通するモデル条件付き利益が残る** | ε=.01/.005/.001。連続した精度区間や一般系の保証へ拡張しない。 |
| ε=.05 | **会計政策で符号が変わる点として扱う** | 通常−7.14%、非相殺+16.62%。非相殺側だけを採用して勝利としない。 |
| ε=.0005 | **両会計でPR点費用が高い結果を保持** | 通常−51.14%、非相殺−29.09%。PR一般の不可能性ではない。 |
| 次の科学計算候補 | **重要な決定論候補の直接compileを少数に限定** | 現在の184側winnerはすべてモデル推定を含む。 |
| 最初の実測パッケージ案 | **3構成・各cosine/sine、計6軸** | `B0I_p15_q4_4th`、`B0I_p11_q1_2nd`、`B0_p15_q4_2nd`。採択・実行は別承認。 |
| PR統計監査 | **主要候補の未観測層に絞る** | q4/r4/K4、q8/r4/K4の確認では2次も4次も未観測。全域sampling増量は必須にしない。 |
| 新しい系・候補・手法の全面追加 | **当面は優先しない** | H8の主要結論に残るモデル・標本依存を先に明確化する。 |

通常・非相殺の新しい費用比較は既に完了している。比較が不利であることを理由に再会計、係数再fit、候補探索を反復してはならない。[S1–S3]

## 1. 実際に確認した資料とレビューの範囲

### 1.1 主な取得資料

GitHub接続で、通常Phase C結果、非相殺費用結果、B2完了報告、B1のfit・検証規約、非相殺cost adapter、凍結shot rule、paired費用・SE算術、H8の元の希少次数記録、2026-10-15の既存科学レビューの必要部分を取得した。H6 V1成果整理も補助的に読んだ。[S1–S10]

外部一次文献は、PR原論文の最新版abstract・書誌、およびSPRINT論文の導入・対象・方法の位置付けを確認した。原論文のfull-text HTML取得は失敗したため、全文を読了したとは扱わない。SPRINTも全文の証明・全数値を独立検証したものではない。網羅的な優先権・新規性調査ではない。[L1,L2]

### 1.2 行っていないこと

- 実H8の1,900行を読み直して新しい費用・順位・統計量を作る処理。
- 既存の数値runner、人工テスト、compiler、sampling、SCF/DF/state solverの実行。
- private入力・全runtime・全grant原本の独立なbyte監査。
- サーバーのlive processや新しいbranchの状態への操作。
- 新たなconditional sampling、bootstrap、CI計算、モデル再fit。

本書の式は、固定SOURCEの意味論と比較の論理を説明するための代数的考察である。数表の数値は保存報告からの引用であり、新しい再ランキングではない。新しい実験候補は提案として明示し、既存結果と混同しない。

### 1.3 技術PASSと科学結論の区別

取得した報告は、SOURCE・入力・B2受入・paired再現・native終了・保全を明確に分離している。静的に確認したcost adapterは完成済みshotの再利用を行い、通常Phase Cの対応単価との一致を検査している。[S7]

今回の資料読解では、通常と非相殺の比較表を直ちに無効化する数式上の矛盾は特定しなかった。ただし、これは全SOURCEの独立監査や全実入力の外部再現が完了したという認定ではない。

## 2. 現在の結果が示しているもの

### 2.1 対象と費用指標

対象はlinear H8、隣接1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes、Nα=Nβ=4、sector dimension 4900、T=.8。比較はq=1/2/4/8、5つのcomplex signal精度、allowance倍率1/10である。費用は状態準備を除くnative full measured Hadamard wrapperのshot-weighted RZで、P=0。[S1,S2]

**εはここではcomplex signal精度であって、Hartree単位のエネルギー誤差ではない。** またRZ総数はRZ depth、T/Toffoli数、実機時間、物理qubit数と同じ指標ではない。

保存DFモデル上のreferenceを用いたbenchmarkであり、元化学Hamiltonian・連続基底極限・実機ノイズまで誤差を保証する評価ではない。既存のtruth-free候補選択やend-to-end QPE費用へ昇格させない。[S1,S2,S8]

### 2.2 主表：allowance倍率1、P=0

削減率は `100 × (1 − PR費用 / 決定論最良費用)`。負値はPR点費用が高いことを表す。[S1,S2]

| ε | 通常・旧32基準 | 通常・184基準 | 非相殺・旧32基準 | 非相殺・184基準 | 非相殺−通常（184、pp） |
|---:|---:|---:|---:|---:|---:|
| .05 | 28.767874% | −7.136909% | 39.480266% | 16.624251% | +23.761161 |
| .01 | 52.717979% | 33.092838% | 52.720260% | 34.684864% | +1.592026 |
| .005 | 52.733612% | 31.306608% | 52.738917% | 35.125323% | +3.818715 |
| .001 | 56.120547% | 22.961345% | 56.147848% | 39.283346% | +16.322001 |
| .0005 | 23.833234% | −51.140301% | 23.833234% | −29.091008% | +22.049293 |

倍率10でも優劣の符号とwinnerは同じだが、すべての費用・削減率がbit単位で同じという意味ではない。例えば最厳格点の184削減率は、通常−51.141772%、非相殺−29.092101%である。倍率1/10を独立な二つの物理実験と数えない。[S1,S2]

### 2.3 支持される限定結論

1. 決定論集合の強化により、旧32候補だけでは見えていたPR利益の一部が消えた。
2. 通常・非相殺の両方で、ε=.01/.005/.001という**実際に調べた3点**にはPR点費用の利益が残る。
3. ε=.05では相殺をどのように会計するかで順位の符号が変わる。
4. ε=.0005では、相殺を除いた診断でもPR点費用は低くない。
5. 184側の最良候補の実測費用は未取得であり、上記の順位はモデル条件付きである。[S1,S2]

「0.001≤ε≤0.01の連続区間で勝つ」「高精度ほどPRが有利」「5精度の80%で普遍的に有利」といった表現は支持されない。5点は同じ系・状態・候補・標本を再利用する関連した評価であり、独立な一般化標本ではない。

## 3. 非相殺診断の科学的な意味

### 3.1 数学的定義

各軸で、保存された端点から

\[
d_{\rm discard}=z_{\rm truncated}-z_{\rm ref},\qquad
 d_{\rm PF}=z_{\rm candidate}-z_{\rm truncated},
\]

\[
d_{\rm total}=d_{\rm discard}+d_{\rm PF}
\]

と分解する。通常biasは \(|d_{\rm total}|\)、非相殺量は \(|d_{\rm discard}|+|d_{\rm PF}|\)。PRでは対応する分解をouter-PFとfinite-RTEで行う。full PFにはdiscardを仮設しない。[S1,S4,S8]

三角不等式から、同じ軸・同じ保存値について

\[
b_{\rm nc}\ge b_{\rm usual}
\]

となる。しかしこの不等式は、保存成分・共有reference・scalarが真の誤差を保証することまで証明しない。PF成分内部の相殺や、分解の取り方への依存も残る。

### 3.2 shot数への作用

凍結SOURCEは、宣言した数値allowanceを用い

\[
h_a=\varepsilon/\sqrt2-b_a-u_a,
\]

\[
N_a=\left\lceil \frac{2B_{\rm upper}^2}{h_a^2}
\log\!\frac{2}{\alpha_a}\right\rceil
\]

を安全な整数域で用いる。適格性の三状態、u/headroom guard、normalizationの保存規約も固定されている。[S5]

同じB・uのもとでは非相殺化でheadroomは増えない。したがって個々の候補のshotは減らず、不適格化も起こり得る。headroomが小さい候補では絶対biasの変化が小さくてもshotへの影響は大きくなる。これは式の性質であり、独立の新しい科学法則ではない。

### 3.3 非相殺でPR削減率が上がった理由

今回、選ばれたPR winnerのshot重み・費用・SEは通常と非相殺で同じだったことが保存報告に記されている。SEをコピーしたのではなく、再重み付けした結果が同じである。[S1]

一方、決定論側はshot増加と候補の切替えによって最小費用が上がった。したがって今回の削減率上昇は、PRの回路や乱択法が改良された効果ではない。

**支持される記述：**「この保存モデル・この誤差分解では、選択PR候補の費用は成分間相殺の除去にほぼ影響されず、決定論的最良候補の選択と費用は影響された。」

**支持されない記述：**「非相殺評価により本当のPR優位を証明した」「通常biasの決定論結果は相殺しているから不正である」。

### 3.4 主評価を通常biasのままにする理由

固定された系・時間・観測量に対して、誤差成分が打ち消し合うこと自体は不正ではない。B0+Iのscalarは、保存Hamiltonianから決まる既知のidentity成分を保持するもので、結果を見て真値へfitした補正とは区別されている。[S8]

相殺が別の時間・距離・状態でも維持されるかは今回検証していない。したがって「相殺がある」から「不安定」「偶然」「一般化不能」と断定しない。同時に、この一点の小さい通常biasを他の条件へ移送する保証もない。

**推奨する提示順序は、通常biasを主表、非相殺を補助表とし、両者を同じ候補集合・単価政策で並べること。** ε=.05の不利な通常結果を非相殺の有利な結果で置換してはならない。

## 4. 費用モデルの証拠と、現在の最大の不足

### 4.1 B1・B2が確認した範囲

S2モデルは

\[
C_{S2}(p,q)=1592+1314q+(25271/7)pq,
\]

S4は固定composition仮説

\[
C_{S4}(p,q)=1592+3\cdot1314q+3(25271/7)pq
\]

である。B0のtrainingはp14/q1・p21/q1・p14/q8、holdoutはp21/q8。B0+Iの0%残差はp14/21のtraining anchorに対するもので、新しい独立holdoutではない。[S6]

B2はB0 p15/q1 S4の両軸168,353 RZ、B0+I p21/q8 S4の両軸1,852,640 RZを実測し、それぞれ約0.2154%、0%のモデル誤差だった。[S3]

これは局所的に有用な検証である。しかし、4軸は2構成のcosine/sineであり、独立な4つのprefix/q構成の検証ではない。全格子に対する誤差上界や、winner選択後の順位保証にはならない。

### 4.2 全winnerがモデル推定を含む

| ε | 通常184 winner | 非相殺184 winner |
|---:|---|---|
| .05 | `B0I_p11_q1_2nd` | `B0I_p15_q1_2nd` |
| .01 | `B0_p15_q4_2nd` | `B0I_p15_q4_2nd` |
| .005 | `B0_p15_q4_2nd` | `B0I_p15_q4_2nd` |
| .001 | `B0_p15_q8_2nd` | `B0I_p15_q4_4th` |
| .0005 | `B0I_p15_q4_4th` | `B0I_p16_q4_4th` |

通常ε=.05のp11はfitしたprefix14〜21の外挿である。他のp15/p16も、fit範囲内にあることと直接実測済みであることは異なる。[S1,S2,S6]

Fractionを使用して比較の算術を正確にしても、入力したモデル予測が実compiled費用に厳密一致するとは限らない。「exact rational cost」はモデル条件下の演算の意味である。

### 4.3 順位の感度を考えるための代数

同じ候補対に限定して、モデル費用を \(\widehat G_D\)、PR標本点費用を \(\widehat G_P\) とし、未知の補正を \(\eta_D,\eta_P\) と置けば

\[
\rho=\frac{\widehat G_P}{\widehat G_D},\qquad
\rho_{\rm corrected}=\rho\frac{1+\eta_P}{1+\eta_D}.
\]

これは新しい誤差上界ではなく、必要な検証を考えるための恒等的な書き換えである。

例えば通常ε=.05の−7.136909%という保存結果は、PR点費用を固定した場合、同じ決定論候補の費用がモデルより約7.14%高ければ、その候補対の優劣が逆転し得ることを意味する。通常ε=.0005の対応する値は約51.14%、非相殺では約29.09%である。[S1,S2]

この数値を全候補への安全marginと呼ばない。モデル補正で別の候補が最良になり得るし、PR母平均の補正も未知である。B2の2%を全gridへ移送してこの問題を解決済みとしない。

### 4.4 勝ちの証明と負けの確認は非対称

一つの実測済み決定論候補が保存PR点費用より安ければ、「そのPR点費用がすべての決定論候補より安い」という主張は否定できる。決定論側の全候補最小値を直接測る必要はない。

逆に、一つの決定論候補を測ってPRより高かっただけでは、未測定の別候補も高いとは分からない。全184候補に対するPR勝利を証明するには、未測定候補に関する追加の根拠が必要である。

したがって、次の限定compileは**主要比較のモデル依存を減らす検証**であって、全候補最適性を証明するcampaignではない。

## 5. 次の直接compile：最初の限定パッケージ案

### 5.1 推奨する3構成・6軸

次の候補は、今回の結果を見て選んだ**結果依存の追試候補**である。過去に事前登録された無作為なvalidation setではない。ユーザーが採用する場合、新しい結果を見る前にSOURCE・入力・候補・判定・再会計範囲を固定する。

| 優先順 | 候補 | 直接確認したいこと | 既存結果の利用 |
|---:|---|---|---|
| 1 | `B0I_p15_q4_4th` cosine/sine | 通常ε=.0005の不利な結果と、非相殺ε=.001の有利な結果に共通するS4単価 | 一つの実測を両会計の保存shotへ利用できる。 |
| 2 | `B0I_p11_q1_2nd` cosine/sine | 通常ε=.05の小さい逆転が、prefix外挿の誤差で消えないか | 非相殺では既に不適格。通常側の限定対照として用いる。 |
| 3 | `B0_p15_q4_2nd` cosine/sine | 通常ε=.01/.005の主な正の利益が、S2単価実測後も残るか | 同じ単価で二つの精度点を確認できる。 |

この3構成は、PRが不利な点だけでも、有利な点だけでもない。S4の重要な共通候補、S2の外挿点、S2の中間精度の有利な点を選んでいる。[S1,S2]

これは数学的に最小・最適と証明した実験設計ではない。結論への影響と既存結果の再利用を重視した、最初の限定案である。

### 5.2 その後に追加するか判断する候補

- `B0I_p16_q4_4th`：非相殺ε=.0005の不利な結果を直接単価で強めて主張する場合。
- `B0I_p15_q1_2nd`：ε=.05の通常／非相殺の符号反転を、両側とも直接単価で示す場合。
- `B0I_p15_q4_2nd`：非相殺ε=.01/.005の数値を直接単価化する場合。
- `B0_p15_q8_2nd`：通常ε=.001の数値を直接単価化する場合。

全7構成・14軸を最初から必須にしない。3構成の結果と最終的な主張範囲を見て、残りの情報価値を判断する。

### 5.3 実行と判定の提案

保存preparation・control・compiler/profileを維持し、H/state/reference/shotの再生成、係数再fit、PR sampling追加は行わない。既存の登録軸ごとの2%基準を新しい契約へ採用するなら、実行前に明記し、不一致時は記録して後続をSTOPする。

モデルと2%以上ずれた場合、それは当該モデルの局所予測が基準を満たさなかった結果である。保存signalやPR法全体が誤りだと即断せず、回路・profileの同一性とモデル適用範囲をCodexで切り分ける。数字を見て自動refitして成功へ変更しない。

実測後の限定費用更新も科学的再解析である。ユーザーが望む場合は、実compileと「同じ候補・保存shotによる限定再会計」を一つの明示承認にまとめられる。準備・技術確認を細分化して何度もGPTへ戻す必要はない。

**本レビューでは、その実行承認を発行していない。**

## 6. PR有限標本と希少イベント

### 6.1 n32が表すもの

確認n32は乱択回路のcompiled費用を評価するためのtrajectory標本数である。量子測定shot数N、32個の独立なsignal誤差保証、32個の分子の検証ではない。固定されたensemble signalから求めたshot数と、有限標本の回路平均費用を組み合わせている。[S8]

静的に確認したD0コードでは、同じtrajectoryのcosine/sineを組にし

\[
D_i=N_c C_{i,c}+N_s C_{i,s},\quad
\widehat G=\frac1n\sum_iD_i,\quad
\widehat{\mathrm{SE}}=s_D/\sqrt n
\]

を計算する。元pairのseed/event/compiler同一性を検査し、exact標本平均と浮動小数の標本SEを別に記録する。[S9]

このpaired処理は、同じtrajectory内の軸相関を落とすより適切である。しかし、未観測層や、union6から点最小を選ぶ選択効果まで自動補正するものではない。α_axis=.025はshot算術の失敗確率割当であり、この費用標本の母平均CIや最良候補の同時保証ではない。[S5,S9]

### 6.2 aggregate histogramより重要な候補別事実

元の`rare_order_observation.json`で確認した内容：[S10]

| PR候補 | R | 確認trajectory | 保存event観測 | 未観測次数 |
|---|---:|---:|---|---|
| `B2_p7_q4_2nd_r4_K4` | 16 | 32 | 0次が512 | 2・4 |
| `B2_p7_q8_2nd_r4_K4` | 32 | 32 | 0次が1024 | 2・4 |

これらは現在のε=.01/.005、およびε=.001/.0005のPR winnerに対応する。[S1,S2]

確認全体の「0次3580・2次4・4次0」という集計だけでは、主要winnerで2次も未観測だったことを見落とす。512や1024を費用平均の独立trajectory標本数と数えてはいけない。

一方、全0次eventでもcomponentやbasisの乱択は残り得るため、32本がすべて同一回路だったとは解釈しない。[S8,S10]

### 6.3 母平均に残る項

保存lawでeventが同じ分布から独立に選ばれる場合、whole-trajectoryをs0（全0）、s2（4なし・2あり）、s4（4あり）に分けると

\[
\pi_0=p_0^R,\quad \pi_2=(p_0+p_2)^R-p_0^R,\quad
\pi_4=1-(p_0+p_2)^R.
\]

費用母平均は

\[
\mathbb E[D]=\pi_0\mu_0+\pi_2\mu_2+\pi_4\mu_4.
\]

pの記録があっても、\(\mu_2,\mu_4\)の費用が未観測なら母平均を閉じたことにはならない。希少確率が小さいことだけでも十分ではない。逆に、未観測だから必ずPR順位が逆転するとも言えない。

既存2026-10-15レビューは、q4/K4・q8/K4の非ゼロtrajectory確率を約3.4684%・1.7530%、32本すべてs0となる確率を約32.32%・56.78%とする近似算術を記載している。ここではその既報を参照し、新しい確率計算は行っていない。元samplerのbit単位replayではない点も維持する。[S8]

### 6.4 次に必要な統計作業

**全面的なn増量ではなく、q4/r4/K4を最初の候補にした限定的な証拠整理を推奨する。** ε=.01/.005の主な正の結果に共通し、主要claimに直接関係する。q8/r4/K4はε=.001の主張を強くする場合の次候補とする。

Codexの読取・静的調査で、既存のlaw、trajectory、component、paired費用、層別記録の有無を確認する。その後、必要なら別承認で条件付きwhole-trajectory pilotを設計する。4次だけを自動優先せず、2次を含む非ゼロ層とその確率質量を区別する。

条件付きpilotは元lawの条件付き分布を保持し、複数の非ゼロeventやcomponent選択を勝手に落とさない。少数pilotは費用寄与の記述的情報を増やすが、それだけで母平均上界を認定しない。

bootstrapは観測済みsupportを再標本化するため、未観測の\(\mu_2,\mu_4\)をそれ自体から復元しない。観測最大値・実行上限・compiler安全guardを、証明なしに母平均上界へ読み替えない。

### 6.5 統計監査をいつまで後回しにできるか

限定的なn32点benchmarkとして結果を示す段階では、未保証事項を明記して成立させられる。母平均の費用優位や高確率の勝利を主要claimにするなら、未観測層と選択効果は解決すべき前提になる。

したがって、直接compileを先に進める方針は維持するが、**最終的な強いPR優位の主張まで、統計問題を未確認のまま棚上げし続けることは推奨しない。**

## 7. 科学的新規性と研究の着地点

### 7.1 既存文献との関係

GüntherほかのPR原論文は、決定論部分と乱択部分を組み合わせたHamiltonian simulationを提案し、single-ancilla phase estimationの量子化学資源評価と水素鎖のサイズ傾向を既に扱う。したがって「PRの初提案」「最初の水素鎖資源評価」は本研究の新規性にできない。[L1]

CasaresほかのSPRINTは、near-integrability、randomization、symmetry protection、QROM等を組み合わせた問題適合的PFを扱う。ここでの旧32／184集合を「既知の最良の決定論的手法すべて」と呼べない。また同論文のToffoli・spectroscopy費用を、本研究のRZ・固定signalへ直接移すこともできない。[L2]

今回の限られた文献確認から、同じ全条件の比較が先行研究に絶対にないという優先権までは認定しない。

### 7.2 推奨する中心的主張

> 保存されたDF Hamiltonianの有限時間signal資源評価では、決定論的対照のprefix・PF次数を強化し、bias成分の相殺への会計依存を明示すると、PRの費用順位は要求精度ごとに変化する。回路費用の削減だけでなく、誤差によるshot負担と比較集合の選択が総資源評価を決める。

これは新アルゴリズムの提案ではなく、比較の公平性と資源評価の信頼性を具体的な実装・保存値・限定実測に結び付ける数値研究としての主張案である。

独立した研究貢献として強くするには、単に多数の監査ファイルを揃えたことではなく、(i)弱い対照での利益がどこで消えるか、(ii)通常／非相殺でどの候補が切り替わるか、(iii)直接compile後にも残る現象は何か、を説明する必要がある。

### 7.3 負の結果の位置付け

ε=.0005の不利な結果は、PR全体を否定する証拠ではない。しかし「より厳しい精度でも常にPRが有利」とする物語を制限する有用な結果である。ε=.05の符号反転は、どの誤差会計を採用したかを隠せないことを示す。

この二つを主表に残すことが、本研究の信頼性を高める。非相殺の好ましい結果だけを主張する選択は、研究として弱くなる。

## 8. H6・他の系との統合と、拡張の優先度

H6 V1成果整理では、通常ε=.05においてB0+I p7/q1 S2の両軸実測単価×保存shotで、保存PR点に対する削減率−13.2901%が報告されている。他の4精度の正の利益はモデル条件付きである。[S11]

このため「強い対照で緩い精度の見かけの利益が減る」という観察はH8だけの記録ではない。ただし、H6はDF19・確認union8、H8はDF28・union6であり、完全に同一条件の系サイズscalingデータとしてpoolしない。

今回H5/H7の全pilot結果やTrack Bの全資料は再評価していない。それらの性能を本レビューで新たに認定しない。既存のH5/H7結果とアルゴリズム改善系列は保全する。

既存のH6固定δのT比較や他の採択済み方針を自動取消しするものではない。ただし、**今この結果を仕上げるための優先順位は、H8の限定直接実測と必要最小限の統計的証拠を先に置く**ことを推奨する。H9以上、新geometry、q16以上、PF全種類、FT全pipelineの追加を必須にしない。[S8]

## 9. 代替案と終了条件

### A. 現在の証拠で限定case studyとしてまとめる

可能な選択肢である。全184側winnerがモデル条件付き、PRがn32点費用であることを明記する必要がある。追加計算を要しないが、最も重要な順位が直接単価で確かめられていないため、実証的な強さには限界が残る。

### B. 限定直接実測と主要候補の統計的証拠で仕上げる（推奨）

本書の3構成パッケージと、主要PR候補の未観測層に絞る。勝敗の方向を先に決めず、負の結果も残す。モデルが概ね合っているなら過剰な再fitや全候補測定へ拡大しない。

### C. 厳密certificateや全域最適性の研究へ転換する

別の研究目標としては考えられる。しかし全support費用、真のbias上界、選択後保証などが必要となり、現在の比較を完成させる作業を越える。現段階の必須路線にはしない。

### 終了の目安

次を満たせば、計算追加を止めて成果整理へ移ることが合理的である。

1. 通常／非相殺、旧32／184の比較が固定結果として揃っている（現在完了）。
2. 主要な負け側・勝ち側の限定対照について、必要な直接単価が得られ、残りのモデル条件を明示できる。
3. PRの主張をn32点benchmarkに留めるか、限定統計検証を加えてどこまで強くするかを明確に決める。
4. 結果を見て候補を増やし続けず、最終scopeと未保証事項が固定される。

直接実測や統計検証で利益が消えた場合も、条件を変えて勝つまで続けるのではなく、主張を「利益の適用範囲・比較条件への依存」に修正して完結させられる。

## 10. 以前の判断から維持・変更する点

| 論点 | 今回の扱い |
|---|---|
| 通常＋非相殺結果を揃えてから判断する | 両方完了したので、次は計算準備の反復ではなく主要実測の選定へ進む。 |
| 希少イベント監査を後回しにする | 直接単価確認を先にする点は維持。ただし主要claim確定前には対象を絞って扱う。 |
| 非相殺で利益が増えること | 「PRの改善」ではなく、主に対照側のshotと選択が変わったと解釈する。 |
| H6／H8に関連結果があること | 共通の問題意識を支えるが、数値の無条件移送・scaling認定はしない。 |
| 追加科学レビュー | 通常の実装・人工テスト・公開・限定測定の進捗だけで繰り返さない。モデルの大きな不一致や主張変更が出た場合に戻す。 |
| 既存SOURCE／STOP／grant | 全て維持。レビューが過去の実行契約を変更することはない。 |

## 11. 次の担当・作業・承認

**直近の推奨作業は、Codexによる「3構成の直接compileと限定費用更新を行うための一括準備」である。**

- ユーザー：本レビューの主評価方針と限定パッケージ案を採用するか判断する。
- Codex：既存実装・保存preparation・compilerを再利用し、3構成6軸のSOURCE/input・科学call・判定・保存・STOPを具体化する。新たな大規模監査基盤は作らない。
- ユーザー：準備済み対象の実compileと、必要な保存shotによる限定再会計を明示承認する。必要なら一つの指示にまとめる。
- Codex：承認範囲だけを実行し、直接実測とモデル推定を別namespace・来歴で保存し、mandatory STOP。
- 主要PR候補の静的な標本証拠照合は独立して進められるが、新しい統計計算・samplingは別承認。

資源は恣意的なCPU/RAM/worker上限を新設せず、実際の空き資源を使って適切に並列化・高速化する。共有環境や他ユーザーのprocess・設定を変更せず、広域grep・大量の無関係I/Oを避ける。既存の固定契約に逐次STOP等の制約がある場合は無断変更しない。

ローカルPCのworktree整理はサーバー側の研究とは分離して扱い、共有GitHub ref・固定結果を壊さない。研究レビューを理由にworktree削除やbranch整理を新たに実行しない。

## 12. 引継ぎ用の結論索引（実行認可ではない）

```text
review_status = COMPLETE
review_date_JST = 2026-10-17
review_scope = TRACK_A_H8_USUAL_AND_NONCANCELLATION_COST_STRATEGY
primary_evaluation_recommendation = USUAL_SIGNED_BIAS
secondary_evaluation_recommendation = EMPIRICAL_NONCANCELLATION_SENSITIVITY
research_direction_recommendation = CONDITIONAL_RESOURCE_BENCHMARK_WITH_STRENGTHENED_CONTROLS
universal_PR_advantage = NOT_SUPPORTED
rigorous_bias_certificate = NOT_ESTABLISHED
PR_population_winner = NOT_ESTABLISHED
all_184_measured_minimum = NOT_ESTABLISHED
proposed_first_measurements = B0I_p15_q4_4th; B0I_p11_q1_2nd; B0_p15_q4_2nd
proposed_axes_per_case = cosine; sine
new_measurement_count_proposal = 3_cases_6_axes
first_targeted_PR_statistical_scope_proposal = B2_p7_q4_2nd_r4_K4
next_PR_scope_if_needed = B2_p7_q8_2nd_r4_K4
automatic_sampling_increase = false
automatic_refit = false
new_scientific_execution_authorized = false
new_grant_issued = false
repository_modified = false
mandatory_STOP = preserved
formal_user_adoption_of_proposals = pending
```

## 参考資料

### GitHubの固定証拠

[S1] 非相殺費用比較の報告（表、winner、paired SE、実行範囲）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md

[S2] 通常Phase C結果（旧32／184、モデル依存、winner）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md

[S3] B2登録S4全4軸の完了報告
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md

[S4] 非相殺shot完了結果（この会話で取得済みの内容を含む）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f902229cef9da26cb6feeff09eb35860b6115a6/docs/research/track_a_h8_noncancellation_shots_result_20261017.md

[S5] 凍結shot rule（実際に静的読解）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac932974a0f250be0df5720e1b6dad06f9ff9535/scripts/d0_frozen_shot_rule_v1.py

[S6] B1費用モデル・training／holdoutの報告（必要部分を取得）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/docs/research/track_a_h8_phase_b1_run01_results_20261017.md

[S7] 非相殺cost-only adapter（1〜180行の静的読解）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac932974a0f250be0df5720e1b6dad06f9ff9535/src/trotterlib/h8_noncancellation_cost.py

[S8] 2026-10-15 H8/D0既存科学レビュー（研究方針・scalar・希少次数・参照資料等の必要部分）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md

[S9] D0 weighted_metric・paired SE実装（110〜225行の静的読解）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac932974a0f250be0df5720e1b6dad06f9ff9535/scripts/d0_saved_value_reanalysis_v1.py

[S10] 元H8の希少次数保存記録（特にconfirmation末尾2700〜2980行を取得）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/rare_order_observation.json

[S11] H6 V1成果整理（branchから取得。記載元の結果commitはf89c40af73289af9f4c2b70ed0b3966d95d47b06。H6全資料の再監査ではない）
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/track-a-v1-h6-p7q1-compile-readiness-20261016/docs/research/track_a_v1_results_synthesis_20261016.md

### 外部一次文献：確認範囲を限定

[L1] Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram Harrow. *Phase estimation with partially randomized time evolution*. arXiv:2503.05647v2（2026-07-10改訂）、PRX Quantum 7, 020332 (2026)。今回確認はabstract・書誌・版履歴。全文HTMLは取得失敗。
https://arxiv.org/abs/2503.05647

[L2] Pablo A. M. Casares et al. *Theory and practice of Trotter product formulas for quantum chemistry*. arXiv:2606.30741（2026-06-29）。SPRINTの導入・対象・方法の位置付けと一部factorization記述を確認。全証明・全数値の再検証ではない。
https://arxiv.org/abs/2606.30741

---

**最終状態：科学レビュー完了。提案の正式採択と新しい科学実行は未承認。既存結果・SOURCE・grant・ledger・STOPを維持する。**
