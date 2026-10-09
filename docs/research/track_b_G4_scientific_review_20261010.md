# Track B G4 科学的研究レビュー
## 条件付きB2分離、matched CTSの優越、固定辞書研究の着地と次の範囲

- **作成日**：2026-10-10（JST）
- **研究**：Partially Randomized Trotter / Track B、RA-RTE
- **レビュー開始承認**：利用者の「レビューを開始して」
- **固定対象commit**：`a221588f42ef3e58f63373915de95607a4ca36be`
- **対象branch**：`track-b-g4-conditional-separation-cts-20261009`
- **前回レビュー**：`track_b_G3_scientific_review_20261009_rereview_v2.md`
- **本書の状態**：GPT科学レビュー完了。以下の進行判断は本レビューの提案であり、GitHubの実行statusを変更したものではない。
- **研究判断**：現行の固定toy・7-prototype/J1を実用的優位法へ育てる実験主線は、ここで一区切りとする。限定理論成果は保持する。新条件の科学実行・全面v4は開始しない。
- **次の担当**：Codex。保存証拠からの固定辞書の終了認証・成果整理・既存sourceの情報アクセス棚卸しまでを、一つの限定作業として行う。その後、新たな中心仮説の採択はGPTへ戻す。

---

## 1. 結論

**G4-Aは残すべき正の成果であり、G4-Bは現在の実用上の主張を縮小させる結果である。両者は矛盾しない。**

G4-Aにより、固定された辞書、内部label分布、合成列、誤差会計、Bernstein予算規則の下では、既存3構成を任意に混合・精度配分・importance samplingしても到達できないT/1Q資源点を、一つのJ1有限lawが与えることが認証された。

しかし、同じfull finite P3 taskに対するmatched CTSには、指定J1 lawよりT・CX・1Qのすべてが小さい有限lawが存在した。CTSはこのtoyで実際に取得できるPauli情報を使っている。したがって、限定B2への改善を、そのまま既知法一般への採用価値へ昇格させることはできない。[R1–R5]

今回のレビューではさらに、G2の6頂点classの補題・保存最小値とG4のCTS lawを組み合わせ、**x=1/4の理想6頂点class全体に対する保守的下界よりも、同じCTS lawの三座標が小さい**という条件付き評価を得た。これはG4で既に認証された結果ではなく、本レビューでの導出・保存値算術である。対応するdigital classへの延長は、独立認証を残す。[R6–R7、§7]

従って次に優先するのは、同じdictionaryでsampling gridを細かくしたり、新しいxや合成seedを試してJ1の勝ちを探したりすることではない。**現在の問いを正確に閉じ、一般involutionを使う新しい研究へ進むなら、その情報・実装・取得費用の根拠を先に作る**ことである。

これは、PR内部のアルゴリズム研究であるTrack B全体を否定する判断ではない。一方、未検証の一般性を理由に現行J1路線を無期限に継続する判断でもない。

---

## 2. 使用資料・実施範囲と証拠の位置付け

### 2.1 固定証拠

主に次を確認した。参照先は末尾にcommit固定URLで記載する。

| 資料 | このレビューで確認した内容 |
|---|---|
| G4 handoff [R1] | 実行scope、A/B結果、原資料への対応、未決事項 |
| G4-A独立証明 [R2] | B2 class、63-profile還元、予算下界、digital bridge |
| G4-A result抜粋 [R3] | 下界・bridge・保存J1の証明書項目 |
| G4-B結果前contract [R4] | CTSの有限specialization、full operator mean、位相、合成・予算規約 |
| 比較CSV・保存監査 [R5] | 同一選択axisでの数値と、同一lawの三座標・shot数 |
| G2一般数学監査 [R6] | 理想6頂点とpure-precision profileの凸還元 |
| G2 resultの指定範囲 [R7] | x=1/4・all-verticesのT/CX/native-1Q最小値の有理区間 |
| G4-C [R8] | wrapper、構造移送、合成頑健性、情報取得の4候補 |
| R0・R0.5 [R9–R10] | 限定classの一般構成と、既知法との差についての既存監査 |
| 前回G3再レビューv2 [R11] | G4に進んだ理由、CTS優位時に予定した判断 |

GitHubのlive connectorから固定commitの資料を読んだ。ローカルにある既存レビューと共通指示も確認した。GitHubへの書込み・commit・pushは行っていない。

### 2.2 外部一次資料

CTSの出版本文Theorem 1、式(5)–(6)、Methodsを確認した。ISのTheorem 1、Sparse Probabilistic Synthesisの一般枠組み、PRおよびPTSCの公式書誌・公開概要を照合した。[L1–L5]

**読めた範囲を超えて確認済みとはしない。** CTSのSupplementary Note 5のPDFは今回の取得では失敗したため、該当補足の全文を独立に照合したとは主張しない。CTS adapterの検討は、出版本文の演算子分解と、リポジトリに明示された有限P3の代数・位相契約を基礎にした。PR/PTSCについて今回の照合は、既存R0.5監査を置き換える新たな全文監査ではない。

### 2.3 今回行った算術

G2に保存された3個の最小値区間の下端と、G4の同一CTS lawの3個のexact resource値を入力に、有理数算術で新しい保守的比較を行った。log下界も有理Taylor上界で自己検算した。再現コードは付録Aに収録した。

これは、G2の全504 profiles、G4の全テスト、全回路、全819 protected pathsを今回再実行・再監査したという意味ではない。保存区間の正しさ・G2のclass還元を前提とするレビュー計算である。

**新規synthesis、LP、Hamiltonian/分子生成、trajectory、量子測定、GPU処理は0。** 既存の実行status、marker、source、科学分類は変更していない。

### 2.4 用語

- **law**：有限のevent抽出確率と、その補正重みからなる推定手順。
- **B2**：この文脈では、ordinary/PTSC-K0/Aから構成する指定された係数class。旧数値K2と同一視しない。
- **理想6頂点class**：G1/G2で監査された、7 prototypesの理想degree matchingを満たすclass。便宜上B3と呼ぶ場合も、旧数値K3全体を意味しない。
- **digital class**：ある理想係数に対するL1差をbiasへ戻す、指定された同辞書実装のclass。
- **予算規則の下界**：固定した十分shot数の計算方式が出力する費用の下界。情報理論的・物理的に必要な最小資源の下界ではない。

---

## 3. 研究目的と、G4によって変わった点

Track Bの目的は、PR内部のランダム化・時間発展アルゴリズムを改善し、科学的に独立した構成や設計原理を得ることだった。現行RA-RTEは、その候補の一つである。

R0では、同じ有限Taylor平均を保つ非負adjacent-degree familyと、そのclassに限ったnormalization最小構成Aが得られた。R1以降、normalization最小がnative資源最小とは限らないことを動機に、degree-local配分へ進んだ。G1は固定P3の6頂点構造、G2は限定緩和目的の完全診断、G3は有限lawとreturn対照を与えた。[R6,R9,R11]

G3再レビューでG4へ進めた理由は、次の二点を閉じる情報価値があったからである。

1. J1が有限poolだけでなく、適切に定義した既存混合classを本当に分離するか。
2. 実際に利用できる強い既知CTSを戻しても、指定J1を採用する理由があるか。

G4はこの二点を、**前者は肯定、後者は指定比較について否定**という形で判別した。前回から進行判断を変えるのは、同じ証拠を違って評価したからではなく、予定した重要な対照の結果が得られたためである。

---

## 4. G4-Aの科学的評価

### 4.1 固定task

\[
R=\frac34 Q_0+\frac14 Q_1,\quad
Q_0=ZI,\quad Q_1=V^\dagger IZV,\quad V=e^{-i\pi XX/16},
\]

\[
M=P_3(-ixR)=I-ixR-\frac{x^2}{2}R^2+i\frac{x^3}{6}R^3.
\]

対象はx={1/8,1/4}の既知development条件、controlled full operator mean、追加workspace1である。axis精度1/200、failure allocation1/5280、shot cap10^9/axisを固定する。最終exponential、QPE全体、分子・DFの性能はこのtaskに含まれない。[R1,R4]

### 4.2 何を認証したか

x=1/4で、同一の保存J1 lawについて次が成立した。[R1–R3]

| 座標 | 限定B2 digital classへの下界 | 同一J1 law | 差 |
|---|---:|---:|---:|
| T | 176,461,958.04 | 174,820,322.72 | 約1,641,635.32 |
| readout込1Q | 465,306,606.84 | 461,712,911.04 | 約3,593,695.80 |

Tと1Qは別々のJ1最適点を合成した値ではない。1Qで選ばれた同じ有限lawが両下界を下回る構成例である。その事後的選択を、独立したT-primary実験へ遡及変更しない。

x=1/8では今回の下界による分離は成立しなかった。これは、その条件における全J1の無利益証明ではない。

### 4.3 証明の論理

event係数c_i≥0、費用C_i≥0、合成bias上界d_i、proposal q_i>0について、

\[
K_C=\sum_i c_i\sqrt{C_i},\quad
s=\epsilon-\sum_i c_i d_i>0,
\]

\[
m_2=\sum_i c_i^2/q_i,\qquad L=\max_i c_i/q_i.
\]

固定policyが

\[
n=\left\lceil\ell\left(2m_2/s^2+4L/(3s)\right)\right\rceil,
\quad\ell=\log10560
\]

なら、T/CXのtwo-axis費用には

\[
G_C=2n\mathbb E_q[C]\ge4\ell(K_C/s)^2
\]

が成立する。これはCauchy–Schwarzと、range・切上げを下側へ落とす操作による。zero-cost eventのIS最適値が未達成であっても不等式は有効である。

1QはC_iをC_{1Q,i}+5/2へ置換すれば、readoutを含む同じ議論ができる。ordinary/PTSC/Aの任意混合・precision配分を63 profilesへ還元するのは、この緩和目的のためであり、finite-shot目的そのものの最適性を主張していない。[R2]

### 4.4 digital bridge

理想cに対して非負デジタル係数\(\tilde c\)と\(e\ge\|\tilde c-c\|_1\)を持ち、

\[
\tilde s=\epsilon-e-d^T\tilde c>0
\]

とする。全eventで\(h_i+r d_i\le r\)が成り立ち、理想classで\(h^Tc\ge r(\epsilon-d^Tc)\)なら、

\[
h^T\tilde c-r\tilde s
=(h^Tc-rs)+(h+rd)^T(\tilde c-c)+re\ge0.
\]

G4-Aはこの条件を対象データ上で確認している。証明の向きは整合しており、理想B2とJ1のデジタル化を非対称に扱う懸念を、指定classの範囲で閉じている。[R2–R3]

ただし、degree residualだけで許される旧K2/K3の任意点を、このL1対応付きclassへ含めてはいない。任意の物理的B2法、state-dependent variance、stratification、別confidence方式にも拡張しない。

### 4.5 成果としての意味

B2は任意の弱い価格を付けた架空対照ではなく、3つの具体的構成と同じnative辞書、任意の混合・精度配分・full-support samplingを含む。したがって、この分離は「有限gridでAより少し安かった」より強い。

しかし、**このclass内の強い結果であることと、class外の既知構成より有用であることは別**である。この区別をG4-Bが具体的に示した。

---

## 5. G4-Bの科学的評価

### 5.1 比較は、何をそろえているか

G4-BはSCUのchannel費用をそのまま流用していない。CTS論文で構成されるunitary ensembleをM=3へ有限化し、G3と同じcoherent weighted Hadamard first-moment taskへ接続している。[R4,L1]

sourceでは、同じfull P3、strict controlled phase、係数の丸め誤差、angle近似誤差、native合成のerror guard、Bernstein予算規則、T/CX/1Q会計、workspaceをそろえている。negative identity correctionも無料で消去せず、rotationのidentityと融合する追加最適化も導入していない。[R4]

従って、**channelとoperatorが異なるという一般論だけで、今回のCTS比較を無効とする根拠はない。** また、CTSの全最適性を証明していなくても、比較対象より安い一つのfeasible lawがあれば、指定J1の採用主張を反証するには十分である。

### 5.2 full operator meanの確認

c=cos(π/8)、s=sin(π/8)とすると、sourceのPauli代数は

\[
Q_1=cIZ+sXY,\qquad R^2=\frac58I+\frac{3c}{8}ZZ,
\]

\[
R^3=\frac{15+3c^2}{32}ZI+\frac{7c}{16}IZ+
\frac{5s}{32}XY+\frac{3cs}{32}YX.
\]

この式はPauli積と、R^3=R R^2から直接照合できる。full P3は実補正二成分と虚成分四つへ分かれる。虚係数をb_kとし、Ls=Σ|b_k|、N=√(1+Ls²)なら、

\[
U_k=\frac{I+i\,\operatorname{sgn}(b_k)L_sP_k}{N}
\]

はunitaryで、

\[
\sum_k N\frac{|b_k|}{L_s}U_k=I+i\sum_k b_kP_k.
\]

実補正を加えるとfull P3が得られる。これは状態|00〉での期待値だけにtargetを縮小したものではない。[R4]

### 5.3 T-primaryの同選択axis比較

| x | J1のT選択law | matched CTSのT選択law | CTS/J1 |
|---|---:|---:|---:|
| 1/8 | 179,160,991.73 | 150,889,450.00 | 約0.84220 |
| 1/4 | 174,720,368.17 | 160,268,880.00 | 約0.91729 |

この有限poolではCTSのT予測は約15.78%／8.27%低い。これは記述的な差であり、過去に存在しなかった成功率やmateriality閾値を追加したものではない。[R5]

### 5.4 一つのlawでの三資源比較

次はx=1/4の1Q-selected law同士である。各行は一つの完成lawであり、座標ごとの最小値を寄せ集めていない。[R5]

| law | n/axis | T | CX | 1Q（readout込） |
|---|---:|---:|---:|---:|
| 保存J1 | 1,114,220 | 174,820,322.72 | 5,661,819.95 | 461,712,911.04 |
| matched CTS | 970,447 | 164,669,575.77 | 3,712,301.36 | 431,125,042.10 |

CTSはこのJ1点に対してT約5.81%、CX約34.43%、1Q約6.62%低く、shotsも約12.90%少ない。workspaceも同じである。

一方、T-selected CTSはCXが増える場合があるため、「CTSはどのlawでも全資源を改善する」とは言わない。G4が示したのは、上表のような同一lawによる具体的dominanceと、別にT-primaryの有限pool最良値での差である。

### 5.5 改善原因をどこまで言えるか

CTSはPauli collectionを使い、rotation-times-wordのeventと異なる辞書を作っている。G4は、係数だけを再配分する検索classから辞書・利用する代数情報を変えると、より安い実装点を作れる具体例である。

ただし、差の何%がcancellation、rotationの安さ、basis-change、bias、shotsに由来するかを分解するcounterfactual実験は今回行われていない。**構成差は明確だが、差の全てを一つの原因に帰属しない。**

---

## 6. 二つの結果は、どう両立するか

G4-Aの主張は、

> 固定B2 digital classより安い、class外のJ1がある。

G4-Bの主張は、

> さらに異なる辞書・Pauli情報を使うCTSに、指定J1より安いlawがある。

である。

後者は前者の反例ではない。CTSが認証B2 classに入るとは主張していないからである。逆に、前者が成立したことも、後者を無視する理由にはならない。

正確な含意は、**「完成済み既存ensembleの混合だけでは十分ではない」という限定設計上の結果は残るが、「degree-local再配分を採用することが現実に最善である」という根拠は、今回のtoyでは失われた**ということである。

原G4報告は全J1の劣位を主張していない。この留保は維持する。次節は、その未確定点を既存の下界でどこまで狭められるかについての、本レビューでの新しい検討である。

---

## 7. 本レビューの追加検討：理想6頂点class全体へのCTS比較

### 7.1 なぜここを見るのか

G4後に「J1のlawを選び直す」「J2/J3へ戻る」「任意混合を大きいsolverで探索する」ことに価値があるかを判断したい。

このために全finite-law最適化を実行する必要があるとは限らない。class全体への有効な下界と、class外の一つの上界を比較できればよい。

### 7.2 理想classへの下界

G2で監査された6頂点classを\(\mathcal C_6\)とし、固定辞書・内部IID法・三precision・保存bias上界を維持する。

\[
\Phi_{6,C}
=\min_{c\in\mathcal C_6,\;\epsilon-d^Tc>0}
\left(\frac{\sum_i c_i\sqrt{C_i}}{\epsilon-d^Tc}\right)^2.
\]

G2の補題により、これは252 pure-precision profilesで最小化できる。finite Bernstein目的全体を252点で解いているのではなく、そこから得られる**緩和の下界**を利用する。[R6]

G4-Aと同じ固定予算規則なら、class内の任意proposalに対して

\[
G_C\ge4\log10560\,\Phi_{6,C}>37\Phi_{6,C}.
\]

1Qについてはnative Cを使った弱い下界で十分である。実際のreadout込1Qはnative1Q以上なので、下界側でreadoutを落とすことは下界を保守的にする。

### 7.3 保存値を使った、特に簡単な下界

G2 resultのx=1/4・same_IS_all_verticesの有理区間下端は、表示すると次である。[R7]

| 目的 | \(\Phi_{6,C}\)の保存下端（丸め表示） | その下端より小さい\(r_C^2\) |
|---|---:|---:|
| T | 4,696,713.681735929 | 2,150² = 4,622,500 |
| CX | 123,618.99516544052 | 350² = 122,500 |
| native1Q | 12,262,155.515725166 | 3,500² = 12,250,000 |

従って保守的に\(G_C\ge37r_C^2\)とできる。これを、G4の同一CTS 1Q-selected lawと比較した。

| x=1/4 | 理想6頂点classへの保守的下界 | 同一CTS lawの費用 | 下界−CTS |
|---|---:|---:|---:|
| T | **171,032,500** | 164,669,575.77 | 約6,362,924.23 |
| CX | **4,532,500** | 3,712,301.36 | 約820,198.64 |
| total1Q | **453,250,000** | 431,125,042.10 | 約22,124,957.90 |

CTSの1Q列にはreadoutが含まれ、左側はnative-onlyから得た弱い下界である。それでも右が小さいため、readoutを落としてCTSを有利に扱った比較ではない。

付録Aの有理数計算で、\(\Phi_{6,C}>r_C^2\)、\(e^{37/4}<10560\)、\(37r_C^2>G_C^{CTS}\)の符号を確認した。floatは表示にだけ使用した。

### 7.4 この導出の強さと、未完了部分

**条件付きで言えること**：G2のclass還元・保存区間、および同じ誤差・予算会計を前提とすると、x=1/4では、理想6頂点class内で任意の係数混合・precision配分・full-support proposalを選んでも、上表の同じCTS lawにT/CX/1Qで追いつけない。

これはG4原報告の「指定J1 pointへのdominance」より広いclassについての検討である。ただし、**G4で全B3に対する独立証明書が取得済みである、とはしない。** 本レビューで過去のG2結果とG4結果を組み合わせたpost-hoc導出である。

特に未完了なのは、これを対応するdigital six-vertex classへ延長するための全event条件の独立確認である。G4-AでB2用rについて条件が成立したからといって、今回の異なるrへ無条件に移せない。

### 7.5 digital classへ延長する最小確認

同じL1課金付きdigital classなら、G4-Aの補題をそのまま使える。今回の各rについて、全eventで

\[
\sqrt{C_i}+r_Cd_i\le r_C
\]

を確認する。平方根を新たに近似せずとも、

\[
d_i\le1,\qquad C_i\le r_C^2(1-d_i)^2
\]

の有理数比較で十分である。1Qはここではnative費用を使い、total1Qへの下界として扱う。

この条件とG2区間の独立再構成が通れば、同辞書の指定digital classに対するclass-wide exclusionを確定できる。通らなければ理想classの条件付き評価に留める。**旧の許容幅だけで定義されたK3全体、別dictionary、別confidence方式へは広げない。**

### 7.6 研究進行への意味

この計算は、新しい方法を提案するための数値探索ではない。現在の検索classの中で細かな最適化を続ける余地があるかを、下界で評価するものである。

結果を踏まえると、同toy・同dictionaryで再最適化を大型化する優先度は低い。独立認証を行うとしても、**J1を救うための次pilotではなく、現行routeを正確に閉じるための保存証拠の確認**と位置付けるのがよい。

---

## 8. 共通wrapper費用は、現行J1を救済する根拠にならない

G4-Cも述べるように、x=1/4の同一law比較では、CTSは三費用もshotsもJ1より小さい。[R5,R8]

各shotに両方式共通の非負費用h_Cを加え、他の条件を固定すると、

\[
G_C^J(h_C)-G_C^{CTS}(h_C)
=\bigl(G_C^J(0)-G_C^{CTS}(0)\bigr)
+2(n_J-n_{CTS})h_C>0.
\]

従って、この二点に関して共通overheadだけで順位はJ1側に戻らない。新しいoverhead gridを計算して確認する必要もない。

これは、多block全体の全候補・再最適化されたfrontierの定理ではない。multi-block化でmean/bias/momentが変わる場合は別問題である。x=1/8では選択CTSのshotsがJ1より多いため、同じ結論をそのまま移さない。

方法別の古典前処理費用・shotごとのweight計算・異なる回路付帯費用を入れる余地はある。しかし、その差は実際のtaskに根拠がある場合だけ計上する。**CTSが勝った後で、測っていないCTS固有の罰則を加えてJ1を勝たせることはしない。**

---

## 9. I0に研究価値は残るか

### 9.1 I0は後付けの研究条件ではない

一般involutionのsampling/implementation accessを使うI0路線は、R0.5時点から明示されていた。従って、CTSに負けた後に初めて作った条件である、と扱うのは不正確である。[R10]

ただし、**そのaccess modelが一般的な数式として定義できることと、今回の実装例がその情報上の優位性を実証したことは別**である。今回のtoyは少数のPauli成分へ展開でき、CTSの取得も実際に完了した。ここでPauli情報を利用不能とすることはできない。

### 9.2 何が未解決か

I0を方法研究として継続するには、少なくとも次の対応を閉じる必要がある。

| 問い | 現在の不足 |
|---|---|
| 入力として何が与えられるか | Qの作用oracle、明示回路、pのsampler、係数表を区別する必要がある |
| controlled-Qと任意角rotationを作れるか | 一般Qの作用を呼べることだけで無料には得られない |
| 設計用の費用・誤差情報をどう得るか | 保存native tableを与えたtoyと、大系での取得手順は別 |
| CTS系でどこまで情報を低費用で利用できるか | full collection以外の手順も含めて比較する必要がある |
| 最後に何を改善するか | 量子gate counts、classical setup、メモリ、task反復回数を分ける必要がある |

R0のO(m)は係数生成の算術量であり、native dictionaryや条件付き\(\mathbb E\sqrt C\)の取得量までO(m)という主張ではない。[R9]

明示的にL種類から長さmのwordを全列挙する取得法なら、候補数がL^m規模になる。この組合せ的負担を避ける仕組みは、係数生成が速いという事実からは出ない。一方で、その取得が常に不可能だとする下界も現資料にはない。

### 9.3 先行研究を弱めない

CTS本文はPauli展開による取得を述べるだけでなく、Markov samplingと別basisの可能性にも言及している。したがって「CTSは常にfull exponential collectionが必要、RA-RTEは常に安い」といった対比は採用しない。[L1]

ordinaryやzero-order PTSCも同I0で構成できるというR0.5の所見を維持する。一般involutionに対応することだけをRA-RTE固有の新規性としない。[R10]

### 9.4 今回の判断

**I0構成の数学的研究価値は否定されない。しかし、I0による実用優位を現時点で採択する根拠は不足している。**

これは「大きいDFへ行けばきっと有利」という期待で埋める問題ではない。再開するなら、実際に与えられる情報、実装できる操作、設計情報の取得法、同accessの強い対照を先に書けることが必要である。

---

## 10. 先行研究と新規性の再評価

本レビューは関連文献全体に対する優先性の不存在証明ではない。確認した主要な部品と残るclaimを分ける。

| 部分 | 評価 |
|---|---|
| 固定protocolのcost×second moment最適IS | 既知。新規性を置かない。[L2] |
| 有限辞書から平均的な量子操作を凸最適化で構成 | 一般原理は既知。channel/operatorのtaskを区別する。[L3] |
| Trotter/RTE/LCUの組合せ | 既知。単なる組合せでは独立差にならない。[L4–L5] |
| 全奇数次数に対する限定adjacent-degree最適構成 | R0の保持すべき数学的候補。[R9] |
| 既存3構成の全混合からの有限law分離 | G4-Aにより指定policy/class内で確認。[R2] |
| 強い辞書を含む実用上の一般優位 | 現在の証拠は支持しない。[R4–R5] |
| 低取得費用の一般involution向け構成法 | 新たに具体化する余地はあるが、成立未確認。[R10] |

任意の大きな凸集合と小さな凸集合を比較すれば、価格によっては差が出る。その幾何学的事実だけでは、強い方法上の新規性としては不十分である。R0/G4の価値は、有限Taylorの係数構造、一般involutionに対する構成、実装可能な保存回路、有限lawと予算を結び付けた点にある。

一方、今回の費用は固定した十分条件による予測値である。保守的error upperや予算設計方式が変わるとランキングが変わり得る。**policy-relativeな構成上の分離を、物理的な量子資源の下限分離と言い換えない**ことが、論文化の正確性に重要である。

R0.5の`METHOD_DELTA_CANDIDATE`は、指定された文献範囲の直接同値性監査の結果である。G4でCTSが勝ったことはA/J1が数学的にCTSと同値だという証明ではなく、逆にR0.5の候補分類も独立論文としての新規性を保証しない。

---

## 11. 研究としての着地点

### 11.1 現時点で完成形に最も近いもの

**限定構成・予算規則の分離と、辞書選択による限界を示すtheory/mechanism note**として、現routeを整理する。

中心となる記述は、例えば次である。

> 有限Taylorの同一operator meanに対し、次数をまたぐ係数再配分は既存ensembleの任意混合では得られない、固定予算方式下の有限資源点を生成する。一方、代数情報を利用した既知の異なる辞書には、それを上回る構成がある。構成classの最適性と、利用可能な表現全体の採用価値を区別する必要がある。

この記述は、J1を万能な改良法として宣伝せず、正と負の結果を一つの科学的な関係として残す。

### 11.2 独立論文として十分か

**現時点では、独立した強い新アルゴリズム論文の主結果として十分だとは判断しない。**

理由は、固定P3・二つのdevelopment step・一つの実装環境が中心であり、主な予算下界の道具自体は標準的、一般の取得・実装手順と実用優位が閉じていないためである。

ただし、独立論文になる可能性を否定する判断ではない。R0の一般次数の定理について、既知構成との差を命題単位で整理し、より一般的で再利用可能な構成結果がまとまるなら、理論を主とする別の完成形は考えられる。これを確認する前に「noteなら投稿できる」と約束しない。

### 11.3 今行うべきでないこと

- 数値優位が残った1Qだけを新たなprimaryとして過去へ適用する。
- B2を弱く定義し直して勝利を保つ。
- CTSが不利になるbasisやseedを結果後に探して、独立移送と呼ぶ。
- R0/G1の証明を、分子・PR＋QPEの資源改善へ転記する。
- Track Aへ結果を自動統合し、独立した研究契約を変える。

### 11.4 新しい方法研究へ進む場合

再設計の第一候補は、**情報取得とnative実装を含めて実行可能な、一般involution向けの構成法**である。ただし、これは現行J1を大きくしたものを自動採択するという意味ではない。

将来の候補には、取得可能な代数関係だけを使う構造化された表現、回路費用・誤差表を全word列挙なしに評価する方法などがある。これらは研究候補であり、今回新規性・正しさ・有用性を確立したアルゴリズムではない。具体的な入力、出力、平均保存式、取得計算量、既知法との差が書けてから、別の研究仮説として検証する。

---

## 12. G4-Cの四候補から何を選ぶか

| G4-C候補 | 今回の判断 | 理由 |
|---|---|---|
| 共通wrapper費用を戻す本検証 | 今は実行しない | x=1/4の指定CTS/J1では、同じ非負overheadで順位は戻らない。救済gridにしない。 |
| 新p・basisへの移送 | 今は実行しない | 移送するselectorや情報優位の根拠が未確定。勝つ条件を探す作業になりやすい。 |
| 合成器・seedの頑健性試験 | 今は実行しない | 限定辞書をさらに最適化する前に、より広いclassへの下界を確定する方が直接的。 |
| 情報取得・accessの評価 | 再設計する場合の最優先論点 | ただし最初は既存sourceの静的棚卸し。新しい大型dictionaryやDF計算を始めない。 |

従って、今回は**どれか一つの新条件をすぐ計算する、という選択はしない**。現在の結果を閉じる保存値確認と、再設計に必要なaccess上の事実を整理することを優先する。

これは「資料が足りないからいつまでも準備を続ける」という方針ではない。現在の固定toyに対する実用優位の実験主線はここで止める。そのうえで、新しい科学的問いを採択するに足る具体的な差があるかを検討する。

---

## 13. 次の担当と作業範囲

### 13.1 担当と目的

**次はCodex。仮称G5として、固定辞書の終了認証・成果整理・既存access棚卸しを一つの作業にまとめる。**

これはG4のretryや次の新分子pilotではない。本レビューの追加導出を独立に確認し、残るclaimを正確に閉じるための作業である。

### 13.2 固定辞書のclass-wide評価を確認する

対象は既存x=1/4、G2の7 prototypes・三precision・6頂点classと、G4の保存された同一CTS law。

確認する命題は、§7の下界とCTS上界の比較である。既存G2結果と元tableから、252-profile還元、価格区間、log下界を独立に確認し、対応するdigital classへの条件も検証する。

r_T=2150、r_CX=350、r_1Q=3500は、今回十分な余裕を与えた単純な候補である。証明書の技術的形式・厳密算術の実装はCodexに任せる。下界をより強くすること自体を目的にしない。

結果は少なくとも、以下を区別する。

- 指定digital classまで認証できた。
- 理想classまで確認できたが、digital bridgeは成立又は確認しない。
- 保存区間・class還元・CTS law対応のどこかに問題があり、追加命題は未成立。

三番目でも新angleやsampling lawで救済しない。G4の指定pointに対する既存結果と、失敗した追加class-wide claimを分けて報告する。

### 13.3 科学的成果を一つの記録へまとめる

R0の一般構成、G1の構造、G4-Aの限定分離、G4-Bのmatched CTS優越、今回の追加評価の到達範囲を、主張と証拠の対応表へまとめる。

元result、失敗記録、source、authorization、consumed marker、STOPは変更しない。過去のpositive/negative/technical分類を後から置き換えない。新しい結論は新しい研究記録に記載する。

論文草稿としての「成功物語」を作るのではなく、何が定理で、何が固定データ上の結果で、何がまだ主張できないかを明示する。独立新規性・論文化採否はCodexに判断させない。

### 13.4 既存sourceのaccessを静的に棚卸しする

将来I0路線を検討するために、既存の入力・生成器・native tableが次のどれを使っているかを明示する。

- pの明示表・samplerと、Qの回路記述・controlled実装。
- rotation生成、phase処理、basis構造、取得済み合成列。
- 全word列挙、conditional cost/errorの集計、Pauli情報の利用。
- 係数生成の算術量と、辞書・費用情報取得の計算量の違い。

既存sourceから分かる事実と、一般化した際の仮定を分ける。実problemにおけるPauli取得の不可避な下界が得られていなければMISSINGとする。新しいp・basis・分子を選ぶことや、科学的候補の採択はしない。

### 13.5 今回の範囲に含めないこと

新synthesis、新angle/seed/grid、登録LP、全面v4、DF/分子入力、新しい独立条件、trajectory、実量子sampling、Track A変更、CTSを不利にする追加penaltyは含めない。

通常のコード修正、保存値算術、source-boundテスト、証明書形式、資料整理はまとめてCodexに任せる。科学的なclass・誤差規則・targetの変更が必要なら、変更せずGPTへ戻す。

### 13.6 この作業の終点

一括作業の完了時点でmandatory STOPする。class-wide認証が成功しても、そのまま新研究のscience runへは進まない。

GPTへ戻る目的は、単なるPASS承認ではなく、(a)現routeの記録を確定する、(b)access上の事実から新しい構成・理論研究を始める根拠があるかを判断すること。新たな重要レビューを実施する場合は、その時点で利用者の開始承認を得る。

---

## 14. 継続・再設計・終了の判断基準

### 今回すでに決めること

- **現固定toy・同辞書で実用優位を求める実験主線**：一区切りとする。
- **R0/G1/G4-Aの限定理論成果**：保持し、範囲を明記して整理する。
- **全面v4・旧大規模LP計画**：進めない。
- **別のp/basis/seedを探す新実験**：今は採択しない。
- **Track B全体**：一般的な終了とはしない。ただし次の主methodも未採択。

### 再開するために必要な新しい研究内容

次のいずれかが、具体的に書けることを重視する。

1. 現行の固定辞書とは異なる、正しいfinite-mean構成と、既知法に吸収されない命題。
2. native辞書・費用・誤差を現実的な入力から取得する方法と、その計算量又は明示した予算。
3. 一般次数・一般involutionに対する、現在の限定例を超える非自明な理論結果。

「もっと大きい系なら勝つかもしれない」「seedを変えれば勝つかもしれない」だけでは、新しい科学実行へ進む根拠としない。一方、新しく具体的な構造や証明候補が生じれば、G4のtoy陰性結果だけを理由に排除しない。

### 数値閾値を後付けしない

本レビューでは、過去結果に5%/10%などの閾値を新設していない。G4により既知の同task構成が指定J1を全三座標で下回ったという事実と、下界で見える探索余地の小ささを使って、次の計算の情報価値を判断している。

将来の研究でmaterialityを使うなら、task・総費用・許容trade-offから結果前に決める。

---

## 15. 現時点で書けるclaimと書けないclaim

### 原G4に基づき書けること

> 固定native辞書とBernstein予算規則により定義した、既存3構成の任意混合・精度配分・full-support samplingおよびL1誤差課金付きdigital classに対し、x=1/4の一つのJ1有限lawはTとreadout付き1Qで認証下界を下回る。

> 同じdevelopment toyのfull P3 taskにおいて、Pauli情報を利用するmatched CTSには、指定J1 lawをT・CX・1Qで同時に下回る有限lawがある。

### 本レビューで条件付きに追加できること

> G2の理想6頂点class還元と保存価格区間を前提とし、同じ予算規則を適用すると、x=1/4の同一CTS lawは、この理想class全体への保守的T/CX/native-1Q下界を下回る。digital classへの独立延長確認は未完了である。

### 今書かないこと

- 全てのJ1/RA-RTE/一般involution法がCTSに劣る。
- G4で旧数値K3全体の不可改善が証明された。
- 原G4だけで6頂点digital class全体への三資源dominanceが認証済み。
- 物理的最小shot数又は情報理論的最小資源の分離を得た。
- 分子・DF・PR＋QPE全体で優位又は劣位が実証された。
- Pauli collectionを使わないだけで、古典取得費用が有利。
- 独立新規性、世界初、独立論文としての十分性が確定した。

---

## 16. 最終評価

G4は、研究の失敗を隠すべき結果でも、テストPASSだけで継続すべき結果でもない。

**何を残すかは明確になった。** R0の限定構成、G1の係数構造、G4-Aの有限law分離は残す。G4-Bにより、今回のtoyで指定J1を実用上の主法とする理由は失われた。さらに本レビューのclass-wide下界は、同辞書の探索を細かくして挽回する方向の価値が低いことを示している。

従って、現routeは限定理論・mechanism成果として区切り、次は保存証拠を使った終了認証とaccessの事実整理までにする。PR内部の新しいアルゴリズム研究は、その後に具体的な構成・必要情報・取得法・既知法との差を作れるかで判断する。

研究の主役を、J1の勝利やv4の完成ではなく、**どの表現classにどこまでの意味があり、その外へ出るためにどの情報と構成が必要か**に戻すことが、今回の適切な着地である。

---

## 付録A：新しいclass-wide比較の保存値算術

以下はレビュー中に実行した計算の核である。G2の保存区間とG4の保存lawを入力にしており、元の252 profilesやsource eventを一から再認証するコードではない。

```python
from fractions import Fraction as F

# G2 result_v1.json / summary['1/4']['axes'][axis]
# / same_IS_all_vertices / value.lo
phi = {
    'T': F('1174178420433982366702031039302898466229493715291256871734215304839/250000000000000000000000000000000000000000000000000000000000'),
    'CX': F('123618995165440511708818978729492825399095136955049911246959303991/1000000000000000000000000000000000000000000000000000000000000'),
    '1Q_native': F('6131077757862582816468354374674723632639525099396037595715948025967/500000000000000000000000000000000000000000000000000000000000'),
}

# G4 saved_values_audit.json / comparisons_same_optimized_axis_only
# / '1/4' / CTS_literal_M3 / '1Q' / resource_total
# 3 coordinates of ONE law. The 1Q cost includes readout.
cts = {
    'T': F('23731386882447613629426075/144115188075855872'),
    'CX': F('1069998018058777932196725/288230376151711744'),
    '1Q_native': F('248526666104596349649005303/576460752303423488'),
}
r = {'T': 2150, 'CX': 350, '1Q_native': 3500}

# Rational upper bound on exp(37/4).
z = F(37, 4)
term = partial = F(1)
for k in range(1, 61):
    term *= z / k
    partial += term
exp_upper = partial + (term * z / 61) / (1 - z / 62)
assert exp_upper < 10560

for axis in phi:
    assert phi[axis] > r[axis] ** 2
    lower = F(37 * r[axis] ** 2)
    assert lower > cts[axis]
    print(axis, int(lower), float(cts[axis]), float(lower - cts[axis]))
```

出力の下界は、T=171,032,500、CX=4,532,500、total1Qに使えるnative由来下界=453,250,000。
符号判定は全てFractionで行い、floatは表示だけに使う。

**次の独立確認が必要な部分**：G2の価格区間と252-profile還元のsourceからの再構成、CTSのexact law identity、six-vertex digital bridge。これらをこの短いコードだけで証明したことにはしない。

---

## 付録B：I0再設計時に使う、最小の研究契約の骨子

この付録は新しい実験の認可ではない。将来GPTが研究候補を作る際に、未解決の情報を明確にするための枠組みである。

**入力**：p、Qの実装記述、controlled access、rotation access、既知の代数関係、利用可能な誤差・費用情報。

**出力**：同じfinite operator meanを保つensemble、実装可能なproposal、補正weight、誤差・予算上界、取得費用の記録。

**既知対照**：同accessで構成できるordinary、PTSC-K0、return等。Pauli情報を現実に取得できる条件では、matched CTSや取得を抑えた既知手順も排除しない。

**主張候補**：係数生成だけでなく、必要な設計情報を取得する手順まで含めた方法上の差。何を最適化し、何を保証し、どの費用を計上するかを明示する。

**反証**：同じ情報・誤差・予算を使う既知構成で得られる、設計情報の取得が主張した費用でできない、制御位相を保てない、または総task費用にすると改善が消える、など。

現在のtoyの優劣から、この契約のどの結果も推定で埋めない。

---

## 参考資料

### 固定repository資料

[R1] G4 results and GPT handoff, commit `a221588f42ef3e58f63373915de95607a4ca36be`.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md

[R2] G4-A independent proof, same commit.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_A_independent_proof_20261009.md

[R3] G4-A result, same commit.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/artifacts/track_b_g4_conditional_separation/2026-10-09/result_A.json

[R4] G4-B matched CTS contract, same commit.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_B_matched_CTS_contract_20261009.md

[R5] G4 resource display and exact saved audit, same commit.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/artifacts/track_b_g4_conditional_separation/2026-10-09/resource_comparison_display.csv
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/artifacts/track_b_g4_conditional_separation/2026-10-09/saved_values_audit.json

[R6] G2 independent mathematical audit, commit `b260189b020ab7dfabb16bf424f49a6efff40d75`.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md

[R7] G2 result, same G2 commit. Values used: x=1/4, same_IS_all_vertices, value.lo, T/CX/1Q. In the retrieved file these occur around lines 187–202, 381–396, and 575–589 respectively for 1Q/CX/T.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json

[R8] G4-C design comparison, G4 commit.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_C_next_validation_design_20261009.md

[R9] R0 independent mathematical and semantic audit, commit `672d6bc667eaa7b9ca4979b012f1530499d701b8`.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md

[R10] R0.5 handoff, commit `61dd534567fda5c7348fdc688814089eb26a3561`.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md

[R11] G3 re-review v2. Local attached original `track_b_G3_scientific_review_20261009_rereview_v2.md`; its fixed repository counterpart is linked from G4.
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/research/track_b_G3_scientific_review_20261009_rereview_v2.md

### 外部一次資料

[L1] Joseph Peetz, Scott E. Smart, Prineha Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). Published 19 February 2026; publisher lists version of record 27 March 2026. 本文Theorem 1/Eqs.(5)–(6)、Methods、Discussionを参照。補足PDFの今回の独立取得は失敗。
https://www.nature.com/articles/s41534-025-01168-w

[L2] Davide Cugini, Touheed Anwar Atif, Yigit Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (13 March 2026). Theorem 1を照合。
https://arxiv.org/html/2603.13495v1

[L3] Bálint Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (31 December 2024).
https://doi.org/10.1103/PRXQuantum.5.040352

[L4] Jakob Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2 (10 July 2026 revision); PRX Quantum 7, 020332 (2026). 今回は公式書誌・概要の照合。
https://arxiv.org/abs/2503.05647

[L5] Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao, *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, arXiv:2212.04566v2; PRX Quantum 6, 010359 (2025). 今回は公式書誌・概要と既存R0.5監査を参照。
https://arxiv.org/abs/2212.04566

---

**レビュー実施状態：完了。** 新しいscopeの科学実行は未実施。G4の証拠は保持し、次の限定作業・将来の研究採択を区別する。
