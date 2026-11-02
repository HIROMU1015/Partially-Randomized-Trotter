# Track A：H6条件付きs2 pilot後の科学的レビュー
## 追加検証の停止判断、共通サイズ比較、原稿への統合

- 作成日：2026年10月13日（日本時間）。
- 開始承認：利用者の「レビューを開始して」。対象は新しいs2 pilotの科学的解釈と、H6追加検証・H8・研究の着地点の比較。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 対象branch：`track-a-h6-conditional-s2-pilot-preparation-20261013`。
- 結果commit：`586b8242bad6865db9809d8183830e6f1e461938`。
- 取得時のremote HEAD／本レビューの結果固定ref：`e0e6473d7047158a7f0f788d026a7a89a01898b8`。
- Pilot SOURCE：`843265b624eac8fbb9acea47ef21cab1207e1880`。
- 科学SOURCE：`5a903aea2492c2d211c1e15e7737df0d50694c12`。
- 前回レビュー：`track_a_h6_independent_confirmation_scientific_review_20261013.md`。全文を引き継ぎ、本書は新しい証拠による追補判断を記録する。[R0]
- **レビュー状態：完了。H6の結果は留保付きの実装資源benchmarkとして統合する。追加H6 samplingを現時点の必須工程にはせず、共通条件でのサイズ比較設計と原稿整理へ進む。**
- **実行承認：本書は、新しいtrajectory・compile・H4補完・H8入力生成／科学計算のgrantではない。既存mandatory STOPと消費済みgrantを維持する。**

---

## 0. 最終判断

**H6の希少次数に関する追加samplingは、現時点で一旦止めることを推奨する。** 今回のpilotによって、直前まで全く費用標本がなかったq8/r2/K2の非ゼロ次数層について、元の条件付き法則から8本のfull-wrapper費用が得られた。既知確率で戻した点推定は従来値から約0.00734%しか増えず、ε_sig=.001のPR/full PF比は約0.387034、約61.30%の削減点推定を維持した。[R1–R3]

これは「平均費用の上界が証明された」「これ以上調べても結果は変わらない」という結論ではない。**当初の研究目的に照らし、未観測費用を追う追加実験を自動的に続けるより、現在の証拠を統合し、比較可能なサイズ方向の検証へ進む準備の優先度が高くなった**という研究判断である。

前回は、通常32本に非ゼロ次数が一度も含まれないことが最優先の情報欠落だった。今回は、その層を正しく条件付けた実測が追加された。残る問題は、複数order2や、order0／order2のいずれの層にも存在する希少component列の費用上界である。この未保証部分を隠さず、有限標本に基づく数値資源研究として扱う。[R1,R4]

**次の担当はサーバー側Codex。** 既に作成済みのH4/H6/H8対応表をもう一度作るのではなく、採用できる既存データ、比較に不可欠な補完、H8の候補生成・精度coverage・費用標本設計を具体化する。同時に、探索・確認・条件付きpilotを分けた原稿用の証拠表を完成させる。[R9]

H8は次のサイズ方向の検証候補として維持するが、**H8本計算はまだ開始しない**。具体的な対象・source・予算・停止条件を固定し、必要な科学的条件の判断と実行承認を分ける。

---

## 1. レビューの根拠と確認範囲

### 1.1 証拠の区分

本書では次を区別する。

1. **保存結果**：GitHubの固定commitに記録された費用、実行状況、来歴、監査結果。
2. **本レビューの算術**：公開CSVの8組のRZ整数、保存確率、shot数から行った平均・分散・確率・仮定付き感度の再計算。
3. **本レビューの解釈**：検証の情報価値、追加計算の優先順位、原稿での主張範囲。
4. **未確認事項**：全component列の費用、全supportのcompiled費用上界、旧hostとの出力bit一致、H8の実測値。

### 1.2 実際に取得・確認した資料

今回の結果報告、費用寄与JSON、16 wrapperのCSV、条件付きsamplerの凍結source、実行準備契約、pilot metadataの先頭160行、40件の公開算術監査結果、既存の希少次数解析の前半、H4/H6/H8比較対応表、前回レビューの公開部分をGitHubから取得した。branchのremote HEADが上記固定refであることも確認した。[R1–R10]

前回レビューの添付Markdownはローカルの全文を読んで、研究目的と既存判断を引き継いだ。今回、旧レビューの全数値実験や全ソースレビューをやり直してはいない。

16 wrapperのCSVにおける両軸のRZ整数を読み、8組のRZ標本を再集計した。平均、標本SD／SE、既知確率による平均補正、ε=.001の総費用比が報告値と表示精度で一致した。全8本のorder2数が1であることは、報告および公開監査を根拠に扱う。metadata全74,150文字の全recordを本レビューで独立再監査したとは主張しない。

**今回実行した計算は保存スカラーの算術だけである。** 科学repository moduleのimport、repository tests、SCF、DF、state／signal再計算、新trajectory、回路生成、transpile、remote writeは行っていない。Codexの「40検査PASS」は保存された監査結果であり、GPTがその40検査を実行したという意味ではない。[R7]

### 1.3 外部文献

新規性の位置付けを確認するため、PR原論文と対称PF制御化の出版済み一次資料、SPRINT/GRADE論文の書誌・abstractを確認した。[P1–P3] 今回の主要な判断はpilotと既存研究記録に基づき、文献の全定理・全数値比較を再評価したものではない。

### 1.4 資料可用性の限界

元のNPZ、全runtime、全回路の再現に必要なサーバー環境は、この結果bundleだけでは揃わない。本レビューは全計算の外部再現やbinary同一性認証ではない。ただし、今回の限定的な結果解釈と優先順位判断に必要な報告・費用・確率・sampler sourceは取得できたため、全rawの再公開を判断開始の条件にはしなかった。

---

## 2. 研究全体における位置付け

本研究の中心は、既知の部分ランダム化時間発展について、**具体的なDF-native実装の回路費用、signal精度、normalization、必要測定数を接続し、どの条件で資源上の利点があるかを明らかにすること**である。厳密な平均費用certificateを新しく発明することや、PRの乱択法則自体を変更することは、現在の主目的ではない。[R0]

| 段階 | 得られた情報 | 今回の判断への意味 |
|---|---|---|
| H4のM1／M2／PM系列 | 固定条件での資源map、候補・対照・移送に関する知見 | 具体的PR資源研究の出発点。H6と自動的に同一条件にはならない |
| H6探索592 group | 92候補、S2／S4を含むfull PF、複数signal精度 | 強い有限比較集合で低費用点を探索 |
| H6独立確認256 group | 固定8構成×新32 trajectory | 別の回路費用標本でも低費用点が再現 |
| 希少次数の保存解析 | 事前K4候補の確認値、既知確率、逆転感度、未観測層の特定 | 選抜と尾部の問題を切り分けた |
| 今回のs2 pilot | q8/r2/K2の条件付き8本・16 compile | 主要な未観測層の回路費用に実測を追加 |

過去のH4限定noteの判断だけへ戻って研究を縮小する必要はない。一方、今回の8本だけで一般的なPR優位へ広げる根拠もない。**既存の数値資源研究としての目的を維持し、今回のpilotをその補足証拠として統合する**のが適切である。

---

## 3. 今回何を測定したのか

### 3.1 対象と費用

対象はlinear H6／1.00 Å／STO-3G／12 modes／Nα=Nβ=3のsector400／保存state／T=.8、tol-only DF1e-8・rank19・cutoff0である。generation order、weighted Hermitian projection、受理済みroundoff方針は維持されている。

Pilotは`B2_p5_q8_2nd_r2_K2`だけを対象とする。prefix5、q8、delta=.1、r2、R=qr=16、K2。full-wrapperはsymmetric_directional・raw concatenation・両軸測定付きで、state preparationは含まない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17という既存設定を使用した。[R1,R5]

量子費用の対象は

\[
G=N_{\cos}\,\mathbb E[C_{\cos}]+N_{\sin}\,\mathbb E[C_{\sin}]
\]

である。Nは保存biasと経験的allowanceを用いた解析的shot予算であり、今回の8本は量子shotの測定回数ではない。epsilonは有限時間complex signalの精度であり、chemical accuracyやエネルギー推定誤差と同一ではない。

### 3.2 s2とS2を区別する

本書の**s2層**は「whole trajectoryにorder2が1個以上含まれる」という費用標本の層を意味する。**S2**という二次product formulaの記号とは別である。またMはorder2 occurrenceの個数であり、Taylor次数そのものではない。

このK2構成の各occurrenceはorder0またはorder2である。したがって

\[
M\sim\operatorname{Binomial}(16,p_2),\qquad
s_0=\{M=0\},\quad s_2=\{M\ge1\}.
\]

K4には別にorder4があり、今回のK2のpilotをその費用へ流用してはいけない。[R4,R5]

---

## 4. 条件付きsamplingと重み付けの科学的妥当性

### 4.1 意図した確率モデルで条件付けは正しい

凍結sourceは、MをBinomial(16,p2)のM≥1条件付き分布から引き、M個の位置を一様に選び、各eventのk+1個のcomponentを元の確率で引く。元の`_make_event`を使用し、量子eventの角度・sign・phase・weightを新しい条件付き確率で変更しない。[R4]

order2がm個の特定次数列について、

\[
\Pr(\boldsymbol k\mid s_2)
=\frac{\binom{16}{m}p_2^m(1-p_2)^{16-m}}{\pi_2}
\frac1{\binom{16}{m}}
=\frac{p_2^m(1-p_2)^{16-m}}{\pi_2},
\quad
\pi_2=1-(1-p_2)^{16}.
\]

これは元の次数列をs2に条件付けた分布に一致する。**「order2を必ず1個だけ入れた別分布」にはなっていない。** 全8本がM=1だったことは、後述の確率で起こる観測結果である。

この確認は、sourceが実数確率モデルで意図した法則を実装することの静的な確認である。binary64の確率丸め、有限PRNGの全support到達性、厳密なfinite-bit確率certificateまで証明したものではない。これらを旧方針より強い保証へ格上げしない。

### 4.2 平均への戻し方も適切

\[
\mathbb E[C]=\pi_0\,\mathbb E[C\mid s_0]
+\pi_2\,\mathbb E[C\mid s_2],\qquad \pi_0=1-\pi_2.
\]

したがって、s0の32本とs2の8本を単純な40本平均にせず、本来の層確率で戻す方針は正しい。[R2]

単純poolをするとs2の比率を20%として扱うことになり、真の約1.67%と異なる。今回の正しい重みは、標本数32:8ではなくπ0:π2である。

ただし、**期待値分解が正しいことと、有限標本平均が厳密に真の期待値へ近いことは別**である。さらに対象K2は確認後に選ばれた点最良であり、既存s0推定に選抜の影響がないとは言えない。現在の評価を「事前に固定したwinnerの完全な独立再確認」として扱わない。

---

## 5. 数値結果の再計算と解釈

### 5.1 公開RZ標本

両軸は同じtrajectoryを共有するため独立標本数は8であり、16ではない。今回のCSVでは各組のcosine／sine RZが同数だった。[R3]

| replica | cosine RZ | sine RZ |
|---|---:|---:|
| 0 | 96,583 | 96,583 |
| 1 | 96,259 | 96,259 |
| 2 | 96,109 | 96,109 |
| 3 | 95,928 | 95,928 |
| 4 | 98,079 | 98,079 |
| 5 | 95,786 | 95,786 |
| 6 | 94,866 | 94,866 |
| 7 | 96,672 | 96,672 |

この8値から本レビューで再計算した平均は96,285.25、標本SDは916.4115965、SEは324.0004271であり、保存値と一致する。[R2,R3]

### 5.2 平均補正

保存されたs0確認平均は95,863.78125、π2は0.01669690613968732である。

\[
\widehat C
=(1-0.01669690613968732)\,95{,}863.78125
+0.01669690613968732\,96{,}285.25
=95{,}870.8184742.
\]

| 項目 | 保存値からの本レビュー再計算 |
|---|---:|
| s2とs0の平均差 | 421.46875 RZ/shot |
| 条件付き平均の相対差 | 約0.439654% |
| 層確率を戻した平均差 | 約7.037224 RZ/shot |
| 平均費用の相対増分 | 約0.00734086% |
| ε=.001のPR G | 約5.1471095302453×10¹² |
| 同精度のfull PF G | 1.3298858970625×10¹³ |
| PR／PF | 0.387033920851 |
| 削減率の点推定 | 61.2966079% |

N_cos=24,402,829、N_sin=29,285,140、N_total=53,687,969を使用した。代表対照はfull19 S4 q2である。報告値と一致した。[R1,R2]

### 5.3 何が分かり、何は分からないか

**分かること**：今回の条件付き標本では、非ゼロ次数を含む回路はs0標本と同程度のRZ費用だった。本来の出現確率へ戻しても、主要なPR／PFの費用差の点推定はほぼ変わらなかった。

**分からないこと**：真の平均の補正が必ず0.00734%以下であること、s2とs0が統計的に同等であること、全component列を含む平均で61.30%の削減が保証されること。

0.00734%は**新しい標本で更新した点推定の差**であり、総推定誤差の上限ではない。観測されたs2−s0の差は、独立標本を仮定した記述的な差のSE約369.46 RZの約1.14倍に過ぎない。したがって「s2の費用増加が確定した」という検定上の主張も不要である。

標本SEを層確率で戻すと、s2側は約5.410 RZ、s0側は約174.585 RZ、合成SEは約174.669 RZである。s2側が占める観測分散の割合は約0.0959%。これは現在観測された分散だけの記述で、未知tailの寄与は含まない。**同じs2の標本数だけを増やしても、表示されたSEの改善は小さく、未観測tailの保証が自動で得られるわけでもない。**

### 5.4 適用範囲

今回の補正はq8/r2/K2のものだけである。報告の4精度表は、この同じq8構成へ4精度の保存shot数を掛け直した値であり、4精度すべてのwinnerを補正した表ではない。例えばε=.05でこのq8構成のPR／PF比は約1.65048となり、そもそもその精度の最良PRではない。[R1,R2]

q1／q2／q4の各winner、事前K4 winner、K4のs4費用を、このpilotで追加検証したとは扱わない。

---

## 6. 残る未観測部分はどの程度か

### 6.1 複数order2の確率

p2=0.001051813380852956、R=16を使用する。[R5,R6]

\[
\Pr(M=1)=16p_2(1-p_2)^{15}
\approx0.0165654453820,
\]

\[
\Pr(M\ge2)
=1-(1-p_2)^{16}-16p_2(1-p_2)^{15}
\approx0.0001314607577.
\]

| 対象 | 確率（本レビューの算術） |
|---|---:|
| 通常trajectoryにorder2が1個以上 | 1.6696906% |
| 通常trajectoryにorder2がちょうど1個 | 1.6565445% |
| 通常trajectoryにorder2が2個以上 | 0.0131461% |
| s2条件付きでorder2が2個以上 | 0.7873360% |
| s2条件付きでorder2がちょうど1個 | 99.2126640% |
| 条件付き8本すべてでM≥2が未観測 | 93.8721768% |

全8本がM=1だったことは、正しいsamplerと十分両立する。M≥2が出なかったためpilotが実装上失敗した、と判断する根拠はない。

ただし「M≤1の確率が約99.9869%だから、全費用supportを99.9869%測定できた」と解釈してはいけない。M=0／M=1のどちらにも非常に多くのcomponent列があり、今回測定したのはその有限標本だけである。

### 6.2 単純増量の情報効率

同じs2法則から追加で引く場合、32本すべてでM≥2が出ない確率は約77.65%である。少なくとも1回のM≥2を95%の確率で見るための標本数は、

\[
n\ge\frac{\log(0.05)}{\log(1-0.0078733602844)}
\]

より379本となる。通常の無条件trajectoryからなら、同じ目的に22,787本が必要になる計算である。

**これは提案する計算本数ではない。** また1件を観測する確率の計算であり、条件付き平均の精度保証やsample-size justificationではない。今後M≥2の費用が主要論点になった場合、既にあるs2 samplerを機械的に増量するより、M≥2への別条件付けを検討する方が直接的である。

しかし、それを今すぐ次の必須実験にするとは判断しない。理由は次節の逆転感度と、研究目的に対する情報価値である。

### 6.3 逆転感度の更新：保証ではない

全非ゼロ層が未知だった前回は、s0平均を固定したとき、希少層の平均が約9,189,927 RZ/shotならPFへ達する、という目安があった。

今回、観測した8本はすべてM=1だった。この8本平均をM=1平均の代用として固定し、さらにs0平均も固定して、**M≥2の平均だけを未知とする仮定付き感度**を考える。

\[
\mu=w_0\mu_0+w_1\mu_1+w_{\ge2}\mu_{\ge2},
\]

\[
\mu_{\ge2}^{\mathrm{break}}
=\frac{G_{\mathrm{PF}}/N-w_0\widehat\mu_0-w_1\widehat\mu_1}
{w_{\ge2}}
\approx1.155084935\times10^9\ \mathrm{RZ/shot}.
\]

ここでw0=1−π2、w1=Pr(M=1)、w≥2=Pr(M≥2)。M=1標本平均96,285.25の約11,996倍である。

この大きな値は、「通常規模の複数event費用で点推定が反転する状況ではない」という**条件付き感度**を示す。ただし、以下の限定を必ず付ける。

- M=0／M=1の真の平均を保存点推定へ固定した仮定を使っている。
- 希少componentはM=0／M=1にも存在し、その影響はこの式で除かれている。
- 未観測M≥2の平均がこの値未満であるという上界は証明していない。
- 条件付きs2平均をM=1平均へ再ラベルして元結果を書き換えたわけではない。これは観測がすべてM=1だったことを利用する補助的な感度分解である。
- 全supportの費用上界、実行guard、正式confidence boundとは別である。

従って、この計算を「逆転は数学的に不可能」として扱ってはいけない。

### 6.4 実行guardを費用保証にしない

compiled instructionsが上限を超えたらSTOPする実装は、未生成の全trajectoryがその上限以下でcompileできる証明ではない。観測最大、部品費用の和、AX-1bモデル、今回の小さいRSSを全support上界へ格上げしない。失敗枝を費用0として期待値から消してもいけない。[R5,R8]

---

## 7. 追加検証の必要性に関する判断

### 7.1 今回はH6追加samplingを必須にしない

判断理由は、次の三点の組合せである。

**第一に、元の主要な情報欠落へ直接答えた。** 非ゼロ次数が含まれるfull-wrapperの費用を条件付きで取得し、その観測値は従来の逆転目安から大きく離れていた。

**第二に、主要な比較差が大きい。** 点推定はPFの約38.7%であり、今回の補正は相対0.00734%。その差を縮めるための通常の観測範囲の変動は見えていない。ただし未知tailを数学的に排除した意味ではない。

**第三に、研究の問いは全supportのcertificate開発ではない。** 残るすべての希少eventに別pilotを繰り返せば、目的が「未観測をなくすこと」へ変わる。component列は依然多数あるため、標本を有限回追加するだけで必ず保証が得られるわけではない。[R0,R4]

このため、**現在の証拠と限界を明示した経験的benchmarkとしてH6を統合し、次の科学的な問いであるサイズ・精度・実装条件の比較へ作業の重点を移す**ことを推奨する。

これは登録済みの統計的受理基準を新たにPASSしたという宣言ではなく、pilot後の研究上の情報価値判断である。8本が母平均保証に十分な本数だった、と遡って宣言しない。

### 7.2 代替案

| 選択肢 | 得られる情報 | 今回の採否 |
|---|---|---|
| 無条件trajectoryを一律増量 | 通常費用平均の標本精度、希少eventの偶発観測 | 不採用。今回の不足に対して非効率 |
| 同じs2をさらに8／32本 | s2の通常component変動、偶発的なM≥2 | 必須にしない。観測SEへの寄与も小さい |
| M≥2だけへ条件付ける | 複数order2の費用に直接回答 | 将来の限定オプション。現時点では主工程にしない |
| K4のs4／他7構成を追加取得 | 4精度・K4を含む補足範囲の拡大 | 全候補のtail補正を主claimにする場合に検討。今回のK2結果から外挿しない |
| 全supportのcompiler上界を証明 | 高信頼な平均費用保証への材料 | 別課題。現在の研究完了の必須条件にしない |
| H6を統合し、共通条件のサイズ比較を設計 | 具体的な資源量と適用範囲へ回答 | **第一推奨** |
| H6のみで研究全体を終了 | 限定的resource benchmarkとしてまとまる | 最小着地点として残すが、拡張目標を今すぐ取り下げない |

### 7.3 H6を再開する条件

新たなH6取得が必要になるのは、例えば次の場合である。

- 保存sourceや既存結果から、特定component・basis列に桁違いの費用が生じる具体的な兆候が見つかった。
- 未観測層を考慮するとPR／PF差と同程度の寄与があり得るという、単なる未保証以上の証拠が得られた。
- 論文の主張を「有限標本の資源見積もり」から、所定信頼度での母平均優位や全4精度・全候補の保証へ変更する。
- H8や同一条件の再利用検査で、H6結果の意味論・compiler・数値受理に重大な不整合が見つかった。

この場合も、追加実験の対象・母集団・推定量・受理規則を先に定める。未確定を理由に自動増量、seed変更、費用の高い枝の除外は行わない。

---

## 8. H4／H6／H8と研究の着地点

### 8.1 H8へ移る前に必要なのは、比較仕様の具体化

既にH4/H6/H8対応表は作成されている。[R9] 今回、その表の作り直しを依頼する必要はない。次は、表にある差を実際のデータ採用・補完計画へ落とす。

旧H4 M1／PMとH6では、DFのrank方針、PF候補、control、数値会計などが違う。現在のH4 production系列も旧M1とは別系列であり、単に完了済みwrapper数が多いという理由でH6と混合できない。**既存H4が再利用できる範囲を確定し、必要なら同条件の最小補完を設計する**。H4全体の再計算は自動的に要求しない。[R9]

H8については、H6と対応するfinite-time complex signal、T、geometry／basis方針、DF生成・切断・順序、S2/S4とB0/B2、control、compiler、shot会計、state-preparation scopeを明示する。入力の保存方法、roundoff受理、候補の適格・不適格・技術失敗の区別も引き継ぐ。

### 8.2 科学的に固定すべき残事項

H6ではq8が登録上限であり、prefix5が選択された。これをH8へそのままコピーして「固定移送」とするか、サイズに応じた候補集合で資源mapを作るかは、異なる研究課題である。両方を同じ検証として扱わない。

- **固定移送の問い**：H6で選んだ設定／生成規則が未使用サイズでも通用するか。
- **サイズ別資源mapの問い**：同じ結果前の候補生成・探索・確認規則で、各サイズの有利な構成領域がどう変わるか。

本レビューは、現在の主題である具体的資源mapと適用範囲を維持する。固定移送を副結果として扱う場合は、その設定をH8結果より前に固定する。q・R・K・prefixの具体的な数値範囲は、今回のpilot結果だけでは未確定である。

最小精度でPRが不適格になった場合、登録範囲不足、数値判定不能、資源STOP、方法の原理的精度限界を区別する。H8でPRが勝つことを実験の成功条件にしない。[R9]

### 8.3 費用標本設計はH8結果を見る前に定める

H6では無条件探索・独立確認の後に不足が判明し、補足としてs2費用を取得した。この来歴を事前の層別設計だったと書き換えない。

H8では既知の次数確率を使い、探索・候補固定・独立確認・必要な条件付き費用取得をどう組み合わせるかを事前に記録できる。全層を必ず測定するのか、限定pilotとしてどの範囲まで取得するのか、何を未評価として残すのかも固定する。

ただし、新しい上界や高信頼区間を実装済みと仮定してはいけない。今回と同様の経験的評価を採用する場合は、その証拠レベルを明示する。

### 8.4 H8の追加によって何を得るか

H8は「もう一つPRが勝つ点」を追加するためではなく、回路短縮・normalization・PF bias・測定数の釣り合いがサイズを変えてどうなるかを調べるための対象である。

H8で利益が減った場合も、どの費用要因が増えたかを説明できれば研究成果になる。H4/H6/H8の3サイズから一般的な漸近指数や全分子への優位を認定しない。

**H8で新しい情報を得る設計を準備することと、H8の大規模計算を今すぐ開始することは別**である。比較仕様・実装・予算を固定し、別途明示的な実行承認が必要となる。

---

## 9. 新規性と原稿の構成

### 9.1 既知として扱う範囲

PR原論文は既に、部分ランダム化、量子化学系のsingle-ancilla位相推定に対する具体的資源見積もり、水素鎖でのサイズ依存を報告している。[P1] 従って「初めてPRの資源を具体的に計算した」「先行研究はboundだけだった」という位置付けは採用しない。

対称Trotter化を低い回転数で制御化する構成も既知である。[P2] 今回のsymmetric_directionalを新しい改善の発明として扱わない。また、SPRINT/GRADE等はより広い設計と資源評価を扱うため、登録S2/S4との比較を最良のHamiltonian simulation一般に対する優位と呼ばない。[P3]

今回の条件付き費用取得は、既知の全期待値分解に基づく資源見積もりの標本設計である。量子PRアルゴリズムの新法則や新しい一般統計定理ではない。

### 9.2 本研究が目指す寄与

前回の方針を維持する。

> 固定されたDF-native PR実装について、finite-time signal精度・normalization・所定の測定規則・実full-wrapper費用を接続し、要求精度と問題サイズに応じた資源量と見積もりの成立範囲を定量化する。費用差を1-shot回路・shot数・biasの寄与へ分けて説明し、探索・独立確認・希少次数の補足評価を区別して示す。

今回のpilotは、この寄与のうち**有限標本による費用見積もりをどこまで信用するか**を補強する証拠になる。一方、pilotの実行時間、テスト件数、hash照合件数だけで独立論文の新規性が増えたとは言わない。

H8を足せば自動的に論文化可能になるわけでもない。結果から分かる具体的な設計判断と、既存の評価へ追加した知見を原稿で明確にする必要がある。現時点で論文としての十分性を無条件に認定しない。

### 9.3 原稿に統合する証拠の順序

1. 問いとtask：単一finite-time complex signal、精度、state-preparation scope、費用単位。
2. 登録したPF／PRの有限比較集合と、実装／compilerの対応。
3. 探索592 groupと、探索後固定8構成の独立確認。
4. 探索で事前に選んだK4候補の確認値と、確認後に選び直した点最小の分離。
5. q8/r2/K2の条件付きs2 pilotと、元確率へ戻した費用点推定。
6. 未観測component、経験的allowance、roundoff、準備費用、範囲外の方法という限界。
7. 比較可能なH4／H6、必要ならH8による精度・サイズ依存と機構説明。

これは新しい計算を待たずに、現有資料から原稿構成とclaim–evidence対応を作れる部分である。

### 9.4 主張案と不採用の主張

**採用できる記述案**

> 固定されたH6のDF-native PR実装のq8/r2/K2構成について、通常の確認32本で未観測だった非ゼロTaylor次数層を、元の条件付き法則から8本取得した。観測された条件付きRZ費用は通常次数層と同程度で、本来の層確率で重み付けした平均費用の点推定は約0.00734%増加した。ε_sig=.001での登録最良full S2/S4に対するRZ-work比は約0.38703であり、この補足費用評価は従来の大きな費用削減点推定をほぼ変えなかった。有限標本、未観測component、経験的数値allowance、状態準備除外という制約を維持する。

**採用しない記述**

- 「平均費用の誤差は0.00734%以下」
- 「すべての希少次数／componentを評価した」
- 「4精度の全winnerをtail補正して優位を認証した」
- 「61.30%削減を所定信頼度で保証した」
- 「K2がK4より優れると確定した」
- 「chemical accuracyの基底エネルギーQPE／RPE総費用を実証した」
- 「旧hostと新hostのcompiled outputが全件bit一致する」

---

## 10. 次にCodexへまとめて任せる作業

### 10.1 直近の作業単位

**H6結果の原稿統合と、共通条件のH4／H6／H8比較設計を具体化する。** 作業場所は、元rawと固定環境のあるサーバー側Codexを基本とする。ただし今回の次作業の中心は既存データの再利用・文書化・技術準備であり、新しい大規模計算ではない。

- 既存の探索／確認／s2 pilotの値を変更せず、固定K4と確認後K2、通常と条件付き標本を区別した原稿用表を確定する。
- 既にある希少次数確率・逆転感度・比較対応表を再作成せず、本レビューの結果解釈と停止判断を追記する。
- H4のどの保存系列がH6と同条件で比較可能かを、source／DF／control／compiler／shot／state-preparation scopeに沿って確定する。不可欠な差があれば最小補完案を出す。
- H8について、登録候補生成、高精度coverage、失敗分類、費用標本設計、保存する入力・準備値、資源予算を実行準備案へ落とす。q／R／prefix等の未決定値を勝手に承認済みとして扱わない。
- 解析モデルは副課題として、featureと費用単位が対応する範囲だけを扱う。H6を見た後の改良モデルを、H6での独立事前予測と呼ばない。
- 必要なsource／人工テスト／索引／原稿案を一つにまとめ、対象だけをcommit・non-force公開し、remoteから取得性を照合して報告する。

### 10.2 今回認可していないこと

本レビューは研究方向と優先順位を判断したものであり、追加s2／M≥2／K4 sampling、H4の再compile、新H8入力・DF・state・signal・wrapper取得を実行するgrantではない。既存の消費済みgrantを再利用せず、source／task／資源／累積callを固定した別承認が必要である。

### 10.3 GPTへ戻す条件

source変更や人工テストPASSだけで全面レビューを繰り返す必要はない。次に重要な科学的判断が必要なのは、例えば以下のときである。

- H8の候補生成、比較baseline、誤差会計、推定／受理規則に、既存方針では決められない科学的な選択が残る。
- H4との比較でsource差が科学的意味を変え、採用結果や比較方法を変更する必要がある。
- H8などの新しい結果が得られ、適用範囲、研究継続、主要claimを判断する。
- H6の未観測費用または数値・回路意味論に、具体的な重大な反証が現れた。

その際も既存レビューの結論を引き継ぎ、変化した論点に集中する。明示的な開始承認と計算実行承認は別に扱う。

---

## 11. 前回レビューから変更した点・維持した点

| 論点 | 前回 | 今回 |
|---|---|---|
| H6非ゼロ次数の費用 | q8の確認32本では未観測。最優先で限定取得 | 元s2法則から8本取得し補正点推定は微小。追加samplingを主工程から外す |
| 母平均上界 | 未証明 | 未証明のまま。証明済みにしない |
| H6の証拠レベル | 独立費用標本付きの限定benchmark | 条件付き費用診断を加えた限定benchmark |
| K2／K4の厳密winner | 主要課題にしない | 維持 |
| H8 | 比較仕様・平均費用評価の整理後の候補 | 比較仕様の具体化を次の重点にする。本計算は別承認 |
| 研究目的 | 具体的PR資源・適用範囲 | 維持。新certificate研究へ転換しない |
| 最終QPE／energy | 現在のsignal taskと分離 | 維持 |

**判断変更の根拠は新しいpilot費用であり、質問の表現やテストPASSではない。** 残る未保証を無視したわけではなく、有限標本に基づく主張の範囲と、さらに情報を取る優先順位を区別した。

---

## 付録A. 本レビューの算術を再確認する最小コード

公開CSVと保存スカラーを転記したコードであり、研究repositoryをimportしない。新trajectoryを生成しない。得られるSEは記述統計、逆転値は明示した仮定付き感度である。

```python
import math
import statistics

rz = [96583, 96259, 96109, 95928, 98079, 95786, 94866, 96672]
p2, R = 0.001051813380852956, 16
pi2 = 0.01669690613968732
mu0 = 95863.78125
N = 24402829 + 29285140
Gpf = 13298858970625

mu2 = statistics.mean(rz)
se2 = statistics.stdev(rz) / math.sqrt(len(rz))
mu = (1 - pi2) * mu0 + pi2 * mu2
p1 = R * p2 * (1 - p2)**(R - 1)
p_ge2 = pi2 - p1
conditional_multi = p_ge2 / pi2

print('s2 mean / SE:', mu2, se2)
print('stratified mean / G:', mu, N * mu)
print('relative correction %:', 100 * (mu / mu0 - 1))
print('PR/PF:', N * mu / Gpf)
print('P(M>=2):', p_ge2)
print('P(M>=2 | s2):', conditional_multi)
print('P(no M>=2 in 8 s2 samples):', (1 - conditional_multi)**8)

# Sensitivity only: freeze the M=0 and M=1 means at their sample estimates.
mu_ge2_break = (
    Gpf / N - (1 - pi2) * mu0 - p1 * mu2
) / p_ge2
print('conditional break-even, NOT a bound:', mu_ge2_break)

# Encounter probability only, NOT a sample-size recommendation for mean accuracy.
n_encounter = math.ceil(math.log(.05) / math.log1p(-conditional_multi))
print('s2 samples for 95% chance of >=1 multi:', n_encounter)
```

主な出力：s2 mean 96,285.25、SE324.0004271、層別平均95,870.8184742、PR/PF0.387033920851、相対補正0.0073408581%、Pr(M≥2)=0.0001314607577、M≥2条件付き確率0.0078733602844、8本未観測確率0.938721767679、仮定付き逆転費用約1.155084935×10⁹ RZ/shot、遭遇確率だけのn=379。

---

## 付録B. 証拠索引

以下は固定commitの原資料であり、本書の[R番号]に対応する。取得済みの範囲を明記し、報告上の監査と本レビューの再計算を区別する。

- **[R0] 前回の科学レビュー**。添付全文を読み、GitHubでは冒頭1–32行も確認。既存判断と研究目的の根拠。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md
- **[R1] 今回のpilot結果報告**。全文取得。固定条件、結果、限界、STOP。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md
- **[R2] 費用寄与JSON**。全文取得。s0／s2平均、shot、層別費用、SE、未認定field。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/cost_contribution_analysis_v1.json
- **[R3] 16 wrapper CSV**。全文取得。8組のRZ整数から本レビューで平均・SD・SEを再計算。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/pilot_wrapper_metrics_v1.csv
- **[R4] 条件付きsampler SOURCE**。全文取得・静的読取。実行していない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/843265b624eac8fbb9acea47ef21cab1207e1880/src/trottertracks/resource_applicability/h6_conditional_s2_sampling_v1.py
- **[R5] Pilot実行準備契約**。本文を取得。未承認記述は準備時点の履歴で、実行後状況はR1を参照。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/docs/research/track_a_h6_conditional_s2_pilot_execution_contract_v1.md
- **[R6] Pilot metadata**。先頭160行を取得し、1番目のreceipt、finite law、seed、order列、両軸metricを確認。全record再監査ではない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/pilot_metadata_evidence_v1.json
- **[R7] 公開算術40検査の結果**。全文取得。Codexの保存監査であり、本レビューが40検査を実行したのではない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/public_arithmetic_verification_v1.json
- **[R8] 希少次数の保存解析**。今回は前半1–60行を取得。残りは既存報告・前回の進行確認を引き継ぐ。保証と仮定付き感度を区別。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md
- **[R9] H4/H6/H8比較対応表**。全文取得。再利用条件、未確定grid、H8未実行の根拠。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h4_h6_h8_comparison_readiness_20261013.md
- **[R10] Remote branch取得**。取得時HEADはe0e6473d7047158a7f0f788d026a7a89a01898b8。後の更新を自動反映する保証はない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/track-a-h6-conditional-s2-pilot-preparation-20261013

### 外部一次資料

- **[P1]** Jakob Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332, published 19 May 2026. 出版社abstractで化学QPE資源評価と水素鎖サイズ依存の既報範囲を確認。
  https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq
- **[P2]** William A. Simon and Peter J. Love, *Halving the arbitrary rotation cost of controlled, Trotterized time evolution*, Physical Review A 114, 032425, published 9 September 2026. 前回引用のarXiv題名から出版時題名が変わっている。今回確認は出版社abstractの範囲。
  https://journals.aps.org/pra/abstract/10.1103/h9gv-ml7w
- **[P3]** Pablo A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741, submitted 29 June 2026. 今回は書誌・abstractの範囲で確認。全手法・数値を再評価していない。
  https://arxiv.org/abs/2606.30741

---

**レビュー完了。H6の母平均上界は未認定のまま、追加samplingの自動継続は勧めない。次はCodexによる原稿統合と共通サイズ比較の具体化。新しいH4／H8科学計算や追加pilotの実行承認は別途必要である。**
