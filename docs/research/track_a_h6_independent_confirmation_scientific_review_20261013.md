# Track A：H6独立32 trajectory確認後の科学的レビュー
## 具体的なPR資源評価、平均費用の不確かさ、サイズ拡張と研究の着地点

- 作成日：2026年10月13日（日本時間）。主要な実験結果は2026年10月12日。
- 開始承認：利用者の「レビューを開始して」。
- 対象：`HIROMU1015/Partially-Randomized-Trotter`。
- 結果branch：`track-a-h6-hash-reuse-eight-worker-20261012`。
- 確認結果commit：`321cd8c3066194fe98741cbebbcdba1515d1deaa`。
- 探索結果commit：`df99d6ee28942df8d12aac3c4db7cd202df55d77`。
- 科学SOURCE：`5a903aea2492c2d211c1e15e7737df0d50694c12`。
- 確認SOURCE：旧159 groupは`73018cbcc39803890974c7ee51401063e6c8daf6`、新97 groupは`87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。
- レビュー状態：**完了。研究継続を推奨するが、母平均での資源優位の正式認定は保留。**
- 次の担当：**Codex。希少次数の費用寄与を限定評価する準備、保存値の機構整理、H4/H6の比較条件対応を一つの作業単位にまとめる。**
- 実行認可：**本レビューは新しい科学計算・追加sampling・H8実行のgrantではない。** 既存結果、one-shot履歴、mandatory STOPを保全する。

---

## 0. 結論

**Track Aを、既知PR法の具体的な資源量と適用範囲を明らかにする数値研究として継続することを推奨する。** H6の今回の結果を、以前のH4限定noteと同じ評価のまま打ち切る理由は弱くなった。一方、H8や大量の追加trajectoryへそのまま進むより、まず現在の平均費用の不確かさが主要な比較を変え得るかを限定的に調べる方がよい。

現在のH6は、full-rank二次・四次PFを比較に残したまま、事前固定したPR構成の独立標本でも低いRZ-work点推定を再現した。登録範囲、所定のshot規則、状態準備除外という条件下で、**約54–63%の差が一つの探索標本だけに依存する結果ではなかった**ことは、重要な実証的前進である。[R1–R4]

ただし、この独立確認が新しく検証したのは**固定8構成における回路費用標本の再現性**である。同じHamiltonian、保存state、signal bias、normalization、compilerを使っており、独立した分子一般化試験、数値精度certificate、最終エネルギー推定実験ではない。

最も優先度の高い残課題は、**既知の有限RTE確率分布における希少次数を、期待回路費用へどう戻すか**である。32本を機械的に128本・1024本へ増やすことを第一案にはしない。保存確率、既存回路、実装構造から、比較を逆転させ得る未評価寄与の大きさを調べる。必要な場合だけ、希少次数に条件付けたfull-wrapper費用取得などを新しい限定実験として設計する。

H8は依然として、サイズ方向の次の主検証候補である。ただしH6の平均費用評価法、比較する実装、候補生成・高精度coverageの規則を整理してから実行条件を固定する。H8でPRが勝つことを進行条件にはしない。

**本研究の主目的を「厳密certificateの新理論」や「解析モデルの失敗探し」へ変更しない。** 具体的なPR資源見積もりそのものに価値を置き、その信頼性・成立範囲・実装依存性を説明する。解析モデルの検証は、その目的を支える副課題として扱う。

---

## 1. 証拠、確認範囲、未確認部分

### 1.1 証拠の区分

本書では、以下を区別する。

- **保存結果**：固定commitの報告・個別signal・選抜資料・sourceに記録された事実。
- **本レビューの算術**：取得した保存スカラーを転記し、費用比、shot比、確率、境界を計算したもの。
- **解釈・推奨**：上記の根拠から本レビューで判断した研究方針。
- **未検証の設計案**：次の費用上界や条件付きsampling。実装済み・実証済みとはしない。

### 1.2 実際に読んだ主要資料

確認結果報告、希少次数JSON、探索後の候補選定JSON、会計source、科学契約、signal評価source、集計source、前回のMemoryError後科学レビューをGitHubから取得した。個別signalは、`B2_p5_q8_2nd_r2_K2`、`B2_p5_q4_2nd_r4_K2`、`B1_p19_q2_4th`を重点的に確認した。[R1–R10]

H4のM1-B1結果、AX-1bの資源モデル結果、Track Bの完成ノートの関連部分も確認した。[R11–R13] 過去のM2、PM-0/1/2、H4限定原稿から拡張研究へ移った理由は、提供された会話履歴と既存の科学判断を引き継いだ。

**全512 wrapperのrawを再構築・再compileしたわけではなく、全536費用行を本レビューで独立再集計したわけでもない。** 536行の照合完了はCodex側の保存監査の報告であり、本レビュー独自の監査件数と混同しない。

本レビューでは、4精度の代表費用を用いた寄与分解、上記3 signalのε=0.001における軸別整数shot数、希少次数の未観測確率と条件付き逆転感度を、repository moduleをimportしないスカラー演算で確認した。SCF、DF、state solve、Hamiltonian作用、signal再評価、trajectory sampling、circuit build、transpile、repository tests、remote writeは実施していない。

### 1.3 公開証拠の限界

報告・選抜資料・個別signal・関連sourceを取得できたので、今回の研究解釈と次の優先順位は判断できる。一方、報告用commitは外部raw runtime、全実行中間記録、NPZをすべて含むものではない。**このレビューをGitHub単独での計算全体再現や、旧hostとのbit一致認証とは呼ばない。**

今すぐ全rawを再pushさせることは、今回の判断の必須条件にしない。次の未観測費用評価で、個別trajectoryのorder列・event列・費用との対応が必要になる場合は、Codexが元rawから必要な最小限の証拠を抽出し、固定commitで取得できるようにする。ハッシュだけから未取得内容を推測しない。

---

## 2. 研究全体の変遷と、今回の結果の位置付け

| 段階 | 主な問い／結果 | 今回への意味 |
|---|---|---|
| 当初の動機 | PRを利用した時間発展・位相推定の総資源を減らせるか | 最終目的。ただし現時点の評価taskとは区別 |
| H4 M1/M2 | 固定DFでの資源mapと、developmentで選んだ構成の別geometryへの移送 | PRが低費用となる具体例と固定移送の証拠 |
| PM-0 | q自由度、候補集合、proxy選抜、compilerの効果を混同しない | 因果帰属の訂正を維持 |
| PM-1 | 近接discard rank4/5を入れても低費用点が残るか | 粗いdiscard集合だけによる利益という懸念を限定的に解消 |
| PM-2／旧原稿 | 精度依存、統計誤差余裕、準備費用、固定移送の限界 | 旧H4限定noteとして閉じる根拠 |
| 拡張方針・AX-1b | 具体的PR資源とモデルの予測信頼性を別のRQとして整理 | モデル誤差だけを成功条件にしない |
| H6探索 | rank19、S2/S4対照、symmetric_directional control、4精度で92候補 | 旧H4の制約の一部を超えた直接評価 |
| H6独立確認 | 8構成×32のfresh費用標本でも低費用点が維持 | 費用推定の再現性。ただし母平均tailは未評価 |

以前の「H4の限定technical/resource noteとして閉じる」という判断は、当時のscopeについてのものだった。利用者はその後、具体的なPR資源評価の価値を明示し、サイズ・精度・強い対照を含む拡張方針を採用した。**その後のH6結果を無視して旧評価を固定することも、H6が成功しただけで一般的優位へ飛躍することも、適切ではない。**

H6の最小signal精度は0.001である。ただしH4のAX-1bにもε=0.001の保存値解析があるため、「高精度0.001へ初めて到達したのが今回」という説明はしない。[R12] 今回の強化点は、**H6、四次full PF、具体的full-wrapper費用、独立費用標本**を同じ評価体系へ接続したことにある。

---

## 3. 現在の比較を正確に定義する

### 3.1 Taskと候補集合

対象はlinear H6、1.00 Å、STO-3G、12 modes、Nα=Nβ=3の400次元sector、保存state、T=0.8。DFはtol-only 1e-8、rank19、cutoff0、signed generation order、採用済みweighted Hermitian projectionである。[R1,R4]

| Family | 登録範囲 | 候補数 |
|---|---|---:|
| B0 | generation-prefix5/10/15、S2、q1/2/4/8 | 12 |
| B1 | full19、S2/global Yoshida S4、q1/2/4/8 | 8 |
| B2 | prefix5/10/15、q1/2/4/8、r1/2/4、K2/4 | 72 |

B0 prefix5は宣言した全4精度で不適格となり、費用取得対象は適格なdeterministic16候補だった。PR72候補×探索8本は576 groupで、16+576=592 group。確認は探索後に固定した8構成×32本=256 groupである。

B3は84 normalization組だけで、signal・full-wrapper費用は未取得である。**本比較の最良PRが、random-dominantを含む全手法で最良という結論はない。**

### 3.2 量子資源の定義

\[
G_x(0)=N_{x,R}\,\mathbb E[C_{x,R}]+N_{x,I}\,\mathbb E[C_{x,I}].
\]

Cはsymmetric_directionalの測定付きHadamard full wrapperを実際にtranspileしたRZ数であり、状態準備は含まない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。

Nは、保存されたsignal biasと経験的allowanceに条件付けた解析的shot予算である。各軸にε/√2、失敗確率α_axis=.025を配分する。

\[
h_{x,a}=\epsilon/\sqrt2-|b_{x,a}|-u_{x,a},\qquad
N_{x,a}=\left\lceil
\frac{2\exp(2\log B_{x,\mathrm{upper}})\log(2/\alpha_a)}{h_{x,a}^2}
\right\rceil.
\]

h>0などの現行gateを満たす候補について会計する。この十分shot規則は、すべての推定器の最小shot数ではない。実機shotを実行した結果でもない。[R3,R4]

**独立32 trajectoryと量子shot数は別の量である。** 前者は平均回路費用を推定する古典的な標本数、後者は想定する量子推定taskの必要測定回数である。

---

## 4. 主要結果の評価

### 4.1 確認点推定

| ε_sig | 確認点最良PR（すべてprefix5） | G_PR | 最良full PF | G_PF | PR/PF |
|---|---|---:|---|---:|---:|
| .05 | q1 r4 K4 | 4.03423200e8 | S2 q1 | 8.77049350e8 | .459978 |
| .01 | q2 r4 K2 | 1.9549158739e10 | S2 q2 | 4.8918110661e10 | .399630 |
| .005 | q4 r4 K2 | 1.0183633918265625e11 | S2 q4 | 2.736762519e11 | .372105 |
| .001 | q8 r2 K2 | 5.14673171597278125e12 | S4 q2 | 1.3298858970625e13 | .387006 |

点推定は保存結果。[R1] 最良は、PRについて確認済み8構成内、PFについて登録済みS2/S4の適格集合内である。

この結果は、少なくとも「探索時の8本だけで偶然大きく安く見えた」という懸念を弱める。特にε=.001では、PFの点最良が四次へ移ってもPRの低費用点が残った。**二次PFだけを対照にした結果とは区別できる。**

一方、「all full PFより優れる」「全prefixで最適」「K2が一般に優れる」は支持しない。

### 4.2 回路費用とshot数の寄与分解

\[
\overline C_x^{\mathrm{eff}}=G_x/N_x,\qquad N_x=N_{x,R}+N_{x,I},
\]

とおくと、保存点推定についても

\[
\frac{G_{\mathrm{PR}}}{G_{\mathrm{PF}}}
=\frac{N_{\mathrm{PR}}}{N_{\mathrm{PF}}}
\frac{\overline C_{\mathrm{PR}}^{\mathrm{eff}}}{\overline C_{\mathrm{PF}}^{\mathrm{eff}}}
\]

が恒等的に成り立つ。以下は本レビューの保存スカラー算術である。

| ε | PR/PF shot比 | PR/PF有効1-shot RZ比 | PR/PF総RZ比 |
|---|---:|---:|---:|
| .05 | 1.301927 | .353305 | .459978 |
| .01 | 1.142213 | .349874 | .399630 |
| .005 | 1.069154 | .348037 | .372105 |
| .001 | .893562 | .433104 | .387006 |

粗い3精度では、PRはPFよりshot数が多い。それでも1-shot費用をPFの約35%に抑えたことで総費用が低い。最も厳しい.001では、PRは1-shot費用とshot数の両方が小さい。

したがって、利益の説明を「randomizationで測定回数が減った」だけにするのは誤りである。**主にdeterministic backboneの短縮による回路削減が、normalizationとbiasによる測定負担を上回った**という形が現在の結果に合う。なお、この分解自体は恒等式であり、新しい理論成果ではない。

### 4.3 同じRでqを変えた機構

選択されたq4/r4/K2とq8/r2/K2は、どちらもR=qr=16、同じprefix5、T=.8である。保存値は両方とも

\[
\log B=.03368262897922761,\qquad B^2\approx1.0696861182
\]

となる。[R8,R9]

| 保存値 | q4/r4/K2 | q8/r2/K2 |
|---|---:|---:|
| 全signal biasの絶対値 | 6.6616782311e−4 | 1.6610227040e−4 |
| outer PF成分の絶対値 | 6.6616721047e−4 | 1.6610166855e−4 |
| finite RTE成分の絶対値 | 2.4902336090e−8 | 2.4824797245e−8 |
| N_total（ε=.001） | 546,721,412 | 53,687,969 |

qを倍にしたとき、総biasは約4.0106分の1、shot数は約10.1833分の1になった。normalizationは同じなので、**この比較でのshot改善はnormalization削減ではなく、主にPF biasが統計誤差予算を消費しにくくなったことによる。**

q8/K2のfinite RTE成分はouter PF成分の約0.01495%である。この点では、Kを4へ増やしてTaylor残差をさらに減らすことより、qによる外側PF誤差と回路費用の釣り合いが主要な論点である。ただし、K2/K4の費用順位をこの誤差比だけで確定しない。

H6 sourceはB0のexact truncated-H signalも生成している。[R7] H4の旧段階では中間reference欠測だったことを、そのままH6にも当てはめて新しいsignal計算を依頼しない。discard/PFの寄与整理が必要なら、まずH6の保存値を使用する。

---

## 5. 独立確認の意味と、選抜後推定

候補選択資料には、各精度の探索上位2、最大8構成の和集合、探索row digest、新しいconfirmation seed、探索標本をpoolしない方針が保存されている。[R2,R3] これは、探索に使った標本をそのまま「確認」に流用する設計より明確に良い。

ただし、現在の主表は**確認値を見た後の8構成内の点最小**である。探索時の点最良をそのまま固定した推定量だけを報告しているのではない。そのため、確認段階内でも最小値選抜に由来する下方偏りは残り得る。

必要な対応は、必ず第三のholdoutを追加することではない。以下を整理すればよい。

- 探索で事前に選ばれた点最良構成の確認値を、固定構成の再評価として別表示する。
- 確認8構成全体の値を残し、確認後の最小値はそのように明示する。
- 強い確率的優位の主張をする場合は、希少次数と複数比較を含む評価方法を事前に定める。

現行K2/K4の差が2√(SE₂²+SE₄²)未満であることは、「区別しにくい」という記述には使える。しかし同等性の証明、検定の非棄却、K2優位の認証へ言い換えない。**今回の研究の中心は細かなK winnerではない。**

---

## 6. 最優先の未解決事項：希少次数の期待費用

### 6.1 32本で希少次数が未観測になるのは不自然ではない

ε=.001の確認点最良q8/r2/K2では、1 occurrenceの非ゼロ次数確率は

\[
p_2=.001051813380852956.
\]

1 trajectoryにR=16 occurrenceあるので、少なくとも一つ非ゼロ次数を含む確率は

\[
p=1-(1-p_2)^{16}\approx .01669690614.
\]

32 trajectoryで一度も非ゼロ次数を含まない確率は

\[
(1-p)^{32}\approx .5834410472.
\]

**約58.3%であり、今回の「32本とも非ゼロ次数なし」は十分起こり得る。** pは標本から推定した値ではなく、保存finite lawの値を用いた算術である。[R5]

また、同じ候補の解析shot予算は53,687,969回である。各shotでfresh whole trajectoryを引く現行モデルなら、非ゼロ次数を含む量子実行回数の期待値は約896,400回となる。これは実機で測定した数ではない。**費用評価用の32本では未観測でも、想定する量子計算全体で稀な枝を無視してよいとは限らない。**

参考までに128本でも未観測確率は約11.59%である。少なくとも1回を95%の確率で見るには178本が必要になる計算だが、これは「平均費用を十分精密に推定するために178本でよい」という意味ではない。1件観測しただけでは、条件付き費用のばらつきは分からない。K4のorder4ではさらに稀な層がある。

### 6.2 小さな標本SEだけでは解消しない理由

各trajectoryについて

\[
D_x(\omega)=N_{x,R}C_{x,R}(\omega)+N_{x,I}C_{x,I}(\omega)
\]

と定義する。求めたい母平均はμ_x=E[D_x]で、32本平均はその推定量である。

希少eventの確率がpでも、条件付き費用が大きければ期待値への寄与pE[D|rare]は大きくなり得る。観測標本にrareが含まれなければ、その費用の影響は通常の標本SEへ反映されない。これは、標本が不正だったという意味でも、PRの数学的平均作用が誤っているという意味でもない。

現在のsourceは`cost_mean_tail_bound=null`、`instruction_cap_is_population_bound=false`を保持しており、この扱いは適切である。[R3,R5] **実行中に5,000,000 instructionを超えたらSTOPすることは、未生成の全trajectoryの費用がその上限以下だという証明ではない。**

### 6.3 大きな費用差があることも無視しない

尾部未評価だからといって、すべての結果を「何も分からない」に戻す必要はない。保存点推定を基準にすると、PRの平均workがfull PFへ達するには、各精度で現在値から約117%、150%、169%、158%増加する必要がある。これは通常の数%の変動と同じ大きさではない。ただし、未評価尾部がその大きさにならないという保証ではない。

特にq8/K2について、E₀=「全occurrenceがorder0」、E₁=その補集合とすると

\[
\mu=(1-p)\mu_0+p\mu_1.
\]

μ₀を今回の全zero-order標本平均5.146731716e12で置く**仮定付き感度計算**では、PFの1.329885897e13へ達するμ₁は

\[
\mu_{1,\mathrm{break}}
=\frac{G_{\mathrm{PF}}-(1-p)\widehat\mu_0}{p}
\approx4.933885165e14.
\]

これはzero-order標本平均の約95.86倍、同じshot重みで換算した有効RZ費用で約9.19e6/shotに相当する。

**これは尾部上界ではなく、μ₀自体の推定誤差も除いた条件付き逆転目安である。** 意味は、「厳密winner」というラベルを得るため際限なく標本を増やすのではなく、この比較差を覆すほどの費用寄与が実装上あり得るかに焦点を合わせられる、ということである。

### 6.4 推奨：既知確率を使った費用寄与の分離

全trajectoryのorder列やorder histogramなどで排反な層sを事前に定義し、その確率π_sを保存finite lawから計算できるなら、

\[
\mu_x=\sum_s\pi_s\,\mu_{x,s},\qquad
\mu_{x,s}=\mathbb E[D_x\mid s]
\]

と分解できる。これは全期待値の恒等式で、新しいPR法を提案するものではない。

次の技術作業は、保存確率とsource構造により、どの層の費用寄与がまだ拘束できていないかを特定することである。componentやbasisの選択はorder条件の中でもランダムなので、order別の観測数だけで条件付き費用を確定しない。

**第一案は保存値・構造からの評価。** 利用可能な正当な構成上界があれば、未観測層の最大寄与を評価する。ただし、部品費用の単純和を、そのまま現行full-wrapper transpileの実測費用や厳密上界とは呼ばない。全回路融合やcompilerの増減を含む対応が必要である。

**それだけで比較できない場合は限定条件付き取得。** 希少層を意図的に含むtrajectoryを、その層内の正しい条件付き分布で生成し、同じfull wrapperをcompileする。そして本来のπ_sで重み付けして平均費用を評価する案を準備する。全component列の網羅を無条件に要求しない。

この提案は**古典的な資源見積もりの標本設計**を変えるものであり、量子実行時のPR分布、normalization、signal bias、Nを変える提案ではない。既存の探索8・確認32を後から別の検証だったと書き換えず、補足の費用評価として別来歴にする。

層内の推定誤差と未評価層の寄与を合わせた区間を示し、上界として証明した部分、経験的な部分、仮定を分ける。厳密区間が構築できない場合でも、工学的な頑健性評価として正直に報告してよい。**数値資源研究の完成を、全ての誤差の新しい一般certificateを発明することへ依存させない。**

---

## 7. 数値精度、回路再現性、状態準備

### 7.1 経験的allowanceの扱い

sourceは、matrix-freeと独立occupation参照、expm/eigh、native作用と独立spectral/forward作用、raw×Bとcorrectedの整合を用いて経験的uを作っている。[R3,R7] 重点3候補のε=.001について、uはおおむね6.2e−11〜9.6e−11で、主要なheadroomは約1.4e−4〜6.2e−4だった。実際にshot式を別のスカラーコードで計算し、保存整数と一致した。

この桁差とu10倍での順位不変は、指定した診断の範囲では境界がroundoff直近ではないことを示す。しかし、共有した実装上の系統誤差、未検証方向の作用誤差、誤ったHamiltonianを参照している可能性まで排除する証明ではない。

ここから直ちに全592群のinterval演算を要求する必要はない。現在の結論は**宣言された数値診断とallowanceに条件付けた資源比較**と明示し、具体的な意味論上の不整合が見つかったときに限定検査を追加する。

### 7.2 Roundoff bridgeを再承認済みのbit一致と混同しない

旧449結果とserver側の結果には異なる来歴があり、旧host/compilerの出力bit完全一致は未検証である。既に承認されたroundoff bridgeの下で結果を受理したことと、旧数値生成・compiler outputが完全に同一だったことは別である。[R1,R6]

小さな浮動小数点差が小さいoperator差であることと、離散的なgate countが必ず不変であることも別である。このため、H4/H6間やmodel/direct間の比較では、どの数値・回路・compiler系列を使ったかを明記する。

本レビューは既存の採用済みbridgeを撤回しない。次の対応表はまず保存証拠で作り、現行の大きな比較差へ影響する具体的な未確認点がある場合のみ、同一taskの限定paired診断を別途設計する。全面再計算は第一案ではない。

### 7.3 共通準備費用

固定pairでは

\[
G_x(P)=G_x(0)+N_xP
\]

と書ける。粗い3精度ではPRのshot数が多いため、大きなPでPFが逆転し得る。保存pointに対する交点は約67,181、313,654、1,336,599 RZ/shot。ε=.001の当該PR/PF pairはPRのNも小さく、非負Pで交点がない。

これは**固定pairかつ共通Pモデル**の話である。Pごとの全候補再最適化、実際のground-state準備費用、fault-tolerantなstate-preparationプロトコルの評価ではない。「.001では状態準備込みQPEで優位が保証された」とは言えない。

---

## 8. H4・H6・H8をどう比較するか

### 8.1 H4とH6の数値をそのままサイズスケーリングにしない

H4の旧M1/PM系列はDF rank12・旧wrapper系列・二次PF中心であり、H6はtol-only rank19、symmetric_directional、full S4追加、異なるprefix/q/r/K集合、経験的uの会計である。保存stateも各instanceのものである。[R4,R11,R12]

従って「H4で約65%削減、H6で約54%削減だからサイズが増えると優位が減る」といった単純比較はできない。逆に、似たprefix分率3/12と5/19が選ばれたから一般的な25%則が得られた、とも言えない。

共通task、DF生成・切断規則、control、compiler、測定規則、候補生成規則を対応表にし、再利用可能な共通条件の結果から比較する。同一条件へ合わせる新計算が必要かどうかは、その対応表の後に決める。H4の進行中geometry計算を変更・停止する必要はない。

### 8.2 H8の研究価値

H6で費用差が維持された以上、より大きい水素鎖で同じ評価方法を使う検証には引き続き意味がある。H8は「H6が勝ったので成功例をもう一つ追加する」ためではなく、**費用削減と誤差・normalizationの釣り合いがサイズを変えても成立するかを反証可能に調べる**ための候補である。

H6のε=.001の点最良PRはq=8という登録上限にある。現行r上限も4である。このままH8へ同じ整数gridをコピーして全PRが不適格になっても、それは方法の原理的不可能性とは区別しなければならない。逆に、q上限にいるから追加qで必ず改善するとも言えない。

H8を見る前に、サイズに対応したprefix候補、q/R/Kのcoverage規則、finite候補内の比較という限界、費用標本設計、失敗時の扱いを固定する。qとRの自由度を不必要に混同しない。全直積を巨大にすることを必須にはしない。

H8について、本書が勧めるのは**準備候補として維持すること**である。新しいH8入力・signal・compileの実行は未承認。H6の平均費用評価が整理され、共通比較の仕様と実行予算が固定された後に、別途承認を受ける。

### 8.3 H8でPRが負ける場合

それも正常な科学結果である。利益消失をnormalization、PF bias、基底変換、回路・shot費用から説明できれば適用範囲の知見になる。PRが勝つgeometryやseedへ変更して救済しない。H4/H6/H8の3サイズから漸近計算量の指数を認定しない。

---

## 9. 解析的資源モデルの扱い

AX-1bでは、H4の保存costに対してSINGLEからFEWへモデルを変えると、pooled out-of-fold MAREが約72.77%から8.91%へ下がった。しかし、これはH4内部のcross-fitted診断であり、未使用H6/H8への予測精度ではない。B3を学習から外した場合のFEW誤差は大きく、全方式への外挿保証もない。[R12]

重要なのは、資源選択で使ったNとeligibilityが参照bias由来の**CONDITIONAL_ORACLE**だったことである。

\[
G_{\mathrm{conditional}}(x)=\sum_a N_{\mathrm{reference},a}(x)C_{\mathrm{model},a}(x)
\]

で良い構成を選べても、未知系でbias・Nまで安く予測できたとは言えない。回路費用モデルと誤差／shotモデルを区別する。

H6の結果が得られたので、既存モデルを結果に合わせて再fitする前に、入力feature・費用単位・control・prefix/DF条件の対応を確認する価値がある。対応しない部分はモデルの数値誤差ではなく、実装やtaskの差とする。構造モデル、H4校正モデル、H6を見た後に改良したモデルは別々に記録する。

H6を使って変更したモデルにとってH6は開発データとなる。H8を独立予測評価に使うなら、係数、feature、利用可能情報、入力取得費用、選択規則をH8評価前に固定する。H6結果を既に見た現段階で、新しいモデル評価を完全に盲検・結果前予測だったと遡及的に呼ばない。

**モデル検証は副課題でよい。** モデルがよく当たった場合にも具体的資源評価の価値は残る。原論文のモデルを失敗させることや、新しい予測法を発明することを本研究の必須完成条件にはしない。

---

## 10. 先行研究との差分と新規性

### 10.1 外部文献の確認範囲

主要一次文献の書誌・abstractと、PR原論文の関連本文を確認した。原論文の解析式・図の数値を本レビューで再現したわけではない。PDF取得では解析テキストと画像のページ対応に不整合があり、版ごとの図表数値・正確なページ番号の断定には使用していない。特定版の式・figureを数値的に再現する場合には、著者公開版・datasetと版を固定して別途行う。

SPRINT、Kanasugi、resource-optimal importance samplingについては、今回重点的に確認したのはabstractが支持するscopeであり、すべての定理や数値実験を独立検証したものではない。

### 10.2 既知として扱う内容

| 一次文献 | 既知範囲／今回への意味 |
|---|---|
| Güntherら、Phase estimation with partially randomized time evolution [P1] | 部分ランダム化、DFを含む表現、経験的Trotter誤差、量子化学のsingle-ancilla QPE資源、水素鎖のサイズ依存を既に扱う。単に「先行研究はboundだけ」「初めて具体的PR資源を評価した」は不適切 |
| Simon–Love、Halving the Cost of Controlled Time-Evolution [P2] | 対称PFの制御化で任意角回転の追加を減らす既知構成。H6のsymmetric_directionalを無視して「制御化改善が未適用」と扱わず、総RZを一律半減と仮定しない |
| Casaresら、SPRINT/GRADE [P3] | factorization、near-integrability、randomization、symmetry、QROM、誤差評価を含む広い設計。S2/S4有限比較を、最良Hamiltonian simulation一般への勝利とはしない |
| Kanasugiら、early-FT QPE [P4] | PRとsingle-ancilla QPEを化学系のend-to-end物理資源へ接続。現在の単一signal・RZ-workとはtaskと費用単位が違う |
| Cugini–Atif–Subaşı、resource-optimal importance sampling [P5] | 回路費用とvarianceを共同最適化する一般的問題を扱う。ここで提案する古典的費用推定の層化と、量子PR法のsampling law最適化は区別する |

この照合は「同じ評価が世界中のどこにもない」という不存在証明ではない。「初めて」「世界初」をここで認定しない。

### 10.3 今回残すべき寄与

> **固定されたDF-native PR実装について、有限時間signalのbias・normalization・所定の測定規則・実full-wrapper費用を接続し、要求精度と問題サイズに応じた資源量と、見積もりの成立範囲を定量化する。解析会計と直接取得の差、誤差・回路・測定の寄与を分けて報告する。**

この実証的寄与は、新しいPR法や一般定理がなくても研究対象になり得る。特に「一回の回路が短い」だけでなく、必要測定数を戻しても利益があるかを同じtaskで検証する点が中心となる。

ただし、新しいデータ点の数、test数、hash数、runtimeの工夫は、それだけで独立論文の新規性を保証しない。今回のH6は、原論文より大きい系を扱ったという意味でもない。**具体的な選択・改善率・適用範囲について、既存評価に何を追加したか**を成果の一文にする必要がある。

### 10.4 他のPR研究は参考に留める

Track Bのreturn集約ノートは、係数質量・support・量子呼出し数・native T/CXを分け、一般full集約が必ず最良native費用にならないことを示している。[R13] これは、normalizationだけで優劣を決めないという本研究の解釈に参考になる。

一方、その証拠は有限Taylor演算子とsynthetic providerのものであり、H6にそのまま転用できない。今回のH6の課題を解くために、別のRTE law、return集約、Hamiltonian構成法を導入することは勧めない。研究間の目的と証拠を保つ。

---

## 11. 研究の着地点と代替案

### 11.1 推奨する着地点

**既知PR法の、実装に対応した資源量と適用範囲を示す数値資源研究。**

有限時間signalを主taskとして維持し、共通条件で比較可能なH4/H6、可能ならH8、要求精度ごとのPF/PR選択、期待回路費用の不確かさ、bias・normalization・測定負担の分解を統合する。解析モデル照合が成立する範囲は副結果にする。

原稿の仮題例：

*Implementation-resolved resource trade-offs of partially randomized time evolution for molecular coherent signals*

H8を含めれば自動的に論文として十分になるわけではない。逆に、厳密な母平均certificateを全候補で与えられないだけで、誠実な数値資源研究を棄却する必要もない。研究上重要な差を、説明可能な不確かさと明確なscopeで提示できるかが基準である。

### 11.2 最小の着地点

H6の独立標本確認と主要機構を、未評価尾部・数値allowance・固定比較集合を明示した限定resource benchmarkとして完成させる。追加資源評価の負荷が利益に見合わない場合は、この範囲で閉じる選択がある。

ただし、利用者が重視した「サイズと精度を広げた具体的PR資源評価」という拡張目標に対して、今すぐ最小scopeへ戻すことを第一案にはしない。

### 11.3 今回は別taskへ分けるもの

化学精度の基底エネルギーQPE/RPE全体を主claimにするには、保存stateのoverlap、時間列、alias復元、推定器、round別の誤差とshot、状態準備、rotation synthesis・FT実装を接続する必要がある。単一T=.8・ε_sig=.001をその代わりにしない。

本レビューは、その全体接続を現在のH6資源研究完成の必須条件にはしない。原論文のQPE総資源と直接数値比較する主張を採択する場合には、その時点で別の明示的な実験設計が必要である。

### 11.4 代替案の比較

| 案 | 評価 |
|---|---|
| H6結果だけで直ちに終了 | 限定benchmarkとして可能。ただし拡張研究の目的に対しては早い |
| 直ちにH8の大規模gridへ進む | 有用な候補だが、平均費用の未評価寄与と比較仕様を先に整理する方が効率的 |
| 全候補のtrajectoryを一律増量 | 第一案にしない。希少層を効率よく評価できず、細かいK順位へ計算を浪費し得る |
| 希少次数の寄与評価＋共通条件対応、その後H8 | **推奨。主要claimに直接関係し、既存結果を最大限利用できる** |
| Track Bの新lawや別Hamiltonian表現へ置換 | 今回の問いを変えるため不採用。別研究の参考に留める |
| 新しい厳密certificate研究へ主題転換 | 現時点では不採用。必要な費用不確かさ評価と、新しい一般理論の研究を混ぜない |

---

## 12. 次のCodex作業と判断点

### 12.1 一つにまとめる作業単位

**H6の費用寄与・不確かさ・比較条件を、保存値とsourceを中心に整理する。**

1. 固定8構成、探索で事前に選ばれた各精度のwinner、確認後の点最良を区別して表にする。全8構成の値を残し、既存の536行・512 wrapperを変更しない。
2. whole-trajectory order層、既知確率、component/basis選択、full-wrapper費用の対応を確認する。既存rawが必要なら、新規計算ではなく最小の公開可能な抽出を用意する。
3. 保存費用・source構造から、未観測層の寄与をどこまで評価できるか整理する。正当な上界、経験的見通し、未評価を明示する。instruction guardや無検証の部品和を母平均上界へ格上げしない。
4. 保存値だけで不足する場合は、同じPR lawの期待費用を評価するための限定条件付き取得案、誤差の扱い、最大資源、必要な実行承認を提案する。量子PR分布・B・signal・shot規則は変更しない。
5. 同じRでのq依存、finite RTE/PF成分、shot×1-shot費用を保存signalから整理する。H4/H6のDF・control・compiler・candidate・数値会計の対応も確認する。AX-1bモデルを使える箇所と使えない箇所を分ける。
6. 必要な資料を一つの証拠索引にまとめ、対象だけをcommit/pushし、remoteから取得して照合する。旧結果・source・他研究・既存dirtyを保全する。

実装手段・限定test・データ抽出の細部はCodexに任せ、軽微なsource変更ごとにGPTへ戻さない。数理上界の成立に疑義がある、比較対象や量子lawを変更する、重要な予算・受理基準を新たに採択する必要がある場合は、その科学的論点を分けて報告する。

### 12.2 新しい実行の扱い

保存スカラーの整理とsource読取は新しい科学実験ではない。一方、新trajectory、条件付きevent取得、compile、signal再計算、refit、H8入力生成は、このレビューだけでは実行認可されない。必要な計算を特定してから、固定source・予算・STOPに結び付く別の利用者承認を得る。

既定方針内の技術診断・準備が終わったという理由だけで、大きなGPTレビューを繰り返さない。次の重要な判断点は、**期待費用の不確かさを整理した結果、PR/PFの主要比較とH8計画がどこまで支持されるか**である。

### 12.3 終了・変更条件

- 希少層の寄与を含めても主要な資源差が頑健なら、その結果を費用推定の補強として固定し、共通比較仕様のH8へ進む価値を判断する。
- 希少層を戻すと順位や改善率が大きく変わるなら、それ自体を資源見積もりの知見として報告する。別seedや有利なgeometryで救済しない。
- 誤差区間が比較差より大きいままなら「現予算では判定不能」とし、さらに計算する情報価値と限定scopeで閉じる案を比較する。自動増量しない。
- H8や他精度で不適格になったときは、登録範囲不足・数値判定不能・方式の精度限界を分ける。
- 細かなK2/K4順位だけが未確定なら、そのまま残す。主claimを閉じるための必須追加実験にしない。
- H4/H6の比較が実装差で説明される場合、サイズ効果へ帰属せず、実装依存として報告する。

---

## 13. 現時点で採用できる主張と、採用しない主張

### 採用できる主張案

> linear H6・1.00 Å・STO-3G・所定のDF表現と有限時間signal推定に対し、登録したfull S2/S4とpartial-random構成を同じ測定規則・full-wrapper費用scopeで比較した。探索後に固定した8構成の独立32 trajectory確認でも、選択集合内の最良PRのRZ-work点推定は、登録最良full PFより約54–63%低かった。粗い3精度では回路短縮がshot増を上回り、ε_sig=.001では対応するPR構成は1-shot費用とshot予算の両方が小さかった。未観測の希少次数、経験的数値allowance、選択集合・状態準備の範囲を留保する。

### 採用しない主張

「母平均で54–63%の削減を厳密認定」「PRが全Hamiltonian simulation法より最適」「K2がK4より優れる」「H4→H6の漸近スケーリングを確認」「化学精度QPEの総費用削減を実証」「旧compiler出力が全件bit一致」「希少order未観測なので費用0」「32 trajectoryは実機32 shots」は、いずれも現在の証拠では採用しない。

### レビューの最終判断

**H6独立確認は、具体的PR資源研究を継続する価値を高めた。次は新しいアルゴリズム探索ではなく、平均費用の主要な未評価寄与を明らかにし、共通条件でのサイズ拡張へつなげる。**

研究の継続方針と、個別計算の実行認可を分ける。今回のレビューは完了したが、追加sampling・H8のgrantは発行していない。

---

## 付録A. 本レビューの保存スカラー検算

repository moduleをimportせず、報告と個別signalから転記した値だけを使用した。全データ・全byteの独立監査ではない。

### A.1 ε=.001のshot検算

| 候補 | 再計算N_real | 再計算N_imag | 合計 | 保存値 |
|---|---:|---:|---:|---|
| B2 p5 q8 r2 K2 | 24,402,829 | 29,285,140 | 53,687,969 | 一致 |
| B2 p5 q4 r4 K2 | 73,854,805 | 472,866,607 | 546,721,412 | 一致 |
| B1 p19 S4 q2 | 25,798,979 | 34,284,146 | 60,083,125 | 一致 |

log Bの上側marginには保存uと同値の保存fieldを使い、log(2/α)=log80、各軸headroomへuを引いた。binary64の境界認証を追加したわけではない。

### A.2 有効1-shot RZ

| ε | G_PR/N_PR | G_PF/N_PF |
|---|---:|---:|
| .05 | 13,270.5 | 37,561 |
| .01 | 25,994.5625 | 74,297 |
| .005 | 51,234.15625 | 147,209 |
| .001 | 95,863.78125 | 221,341 |

この表はshot重み付き有効平均であり、一般には単純な二軸算術平均とは限らない。

### A.3 計算の分類

本書の式変形、費用比、確率、条件付き逆転目安は、レビュー上の算術と推論である。追加のscientific campaign、resource profile、Monte Carlo、refit、quantum simulationではない。未観測eventの回路を生成したこともない。

---

## 付録B. 主要一次証拠と文献

### Repository証拠

以下の[R番号]は本書内の出典である。コミット固定URLを用いる。

[R1] H6独立確認の結果報告。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md

[R2] 探索後・確認前に固定した候補選択。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12/sealed_v1/candidate_selection_v1.json

[R3] Seed、経験的allowance、整数shot、paired費用、選抜、order診断のsource。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trottertracks/resource_applicability/h6_matched_accounting_v1.py

[R4] H6固定科学plan・候補・評価指標。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trottertracks/resource_applicability/h6_matched_contract_v1.py

[R5] 確認8構成の希少次数確率・観測数。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/rare_order_diagnostics_v1.json

[R6] H6補完契約と前回科学レビュー。文中の未実行statusは各契約作成時点の履歴として扱う。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_matched_supplement_execution_contract_v3.md
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_matched_v2_memory_stop_independent_review_2026-10-11.md

[R7] H6有限平均signal・独立参照・誤差分解のsource。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trottertracks/resource_applicability/h6_matched_port_v1.py

[R8] q8/r2/K2の個別signal。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p5_q8_2nd_r2_K2_signal.json

[R9] q4/r4/K2の個別signal。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p5_q4_2nd_r4_K2_signal.json

[R10] full S4/q2の個別signal。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B1_p19_q2_4th_signal.json

[R11] H4 M1-B1結果。qやselectorについての歴史的な解釈は、後のPM-0訂正を優先する。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e0814e70c14ecf526444fac8a2142799610dc96/docs/pr2_matched_accuracy_m1_b1_result_validation.md

[R12] AX-1bのH4保存値モデル比較とconditional-oracle結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md

[R13] Track B有限Taylor return集約ノート。別研究の参考資料。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md

[R14] 保存集計のsource。外部rawを使うため、そのままGitHub単独再現可能とはしない。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/aggregate_saved_results.py

[R15] 先に取得済みの結果索引、CSV、来歴、536行照合の参照先。照合件数は保存監査の報告値である。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/321cd8c3066194fe98741cbebbcdba1515d1deaa/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12

### 外部一次文献

[P1] J. Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647（abstract上の最新版v2：2026-07-10）、PRX Quantum 7, 020332 (2026), DOI:10.1103/ynxb-p2xq。
https://arxiv.org/abs/2503.05647
https://doi.org/10.1103/ynxb-p2xq

[P2] W. A. Simon and P. J. Love, *Halving the Cost of Controlled Time-Evolution*, arXiv:2511.13855（2025-11-17）。
https://arxiv.org/abs/2511.13855

[P3] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741（2026-06-29）。
https://arxiv.org/abs/2606.30741

[P4] S. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2（2026-03-30）。
https://arxiv.org/abs/2603.22778

[P5] Davide Cugini, Touheed Anwar Atif and Yigit Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495（2026-03-13、v1）。
https://arxiv.org/abs/2603.13495
