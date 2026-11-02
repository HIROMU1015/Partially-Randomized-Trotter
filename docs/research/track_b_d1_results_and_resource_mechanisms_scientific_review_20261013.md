# Track B D1：原結果・保存値分析後の科学的レビュー
## finite-RTEの表現、normalization情報、合成単価、反復と資源順位の成立条件

- 作成日：2026-10-13（日本時間）
- 開始承認：利用者の「レビューを開始して」。対象はD1原結果と、その後の保存値分析。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 科学source S：`874ca12d347f8f35e5d8d64f65f24abc4a2df523`
- Authorization / 実行HEAD A：`f803291638d453873e4ab6f1ecf7bcc607f48e4b`
- D1結果commit：`fc96571881f301435a655f057709df4d6c1678e0`
- 保存値分析commit：`2dcd56737be35526f6436a52835f59d817adf30c`
- 現在の分析branch：`track-b-d1-saved-resource-analysis-20261013`。本レビュー時にremote HEADとの一致を確認。
- 運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 状態：**科学的レビュー完了。②「既存成果の発展」を、資源順位の機構・成立条件の研究として限定継続することを推奨。**
- 新しい科学実行：本レビューでは未実施・未認可。既存mandatory STOPを維持。

---

## 0. 結論

**D1は、既存成果を発展させる研究を継続する根拠になる。** ただし、closed P5の一般的優位、新しい一般PR法の成立、独立論文としての十分性を確定したものではない。

今回採用する方向は、次の問いを具体的に検証することである。

> 同じ時間発展・最終精度・confidenceを満たすfinite-RTE表現において、係数・normalizationの利益が、有限精度合成と実行回数を含む資源利益へ変わる条件は何か。どこでは、その利益が消えるか。

保存結果には、P5のT費用がpartialより低いことに加え、r=1とr=2でその内訳が異なることが残る。さらに本レビューでは、保存されたroot eventの回路骨格と価格を確認し、**同じchildについて、P5のroot価格はr=1ではpartialより4 T高く、r=2では8 T低い**という具体的な対応を確認した。全費用差をこれだけへ帰属したわけではないが、次に調べるべき「単価」の中身が具体化した。[R1,R2,R8–R11]

次は、Codexで**合成key・event構造に結び付いた価格依存性の整理と、既存r=2の保存IRを使う共通境界相殺対照の準備**をまとめることを推奨する。これは、既済の90件のfactor分解や全交点表をやり直す指示ではない。新しいr・分子・SPを追加する前に、現在の差が何に支えられているかを識別する。

本レビューは、その研究目的と比較原則を決める。新しいIR結合・相殺評価・行列検証・合成などの登録処理は、固定scope・source・資源上限と、別途の明示実行承認を必要とする。

| 論点 | 今回の判断 |
|---|---|
| D1の固定モデルの結果 | 限定された研究証拠として支持 |
| 保存値factor分解 | 記述的な恒等分解として支持。因果割合ではない |
| ②の研究継続 | 支持。特定法の勝利でなく成立条件の検証へ |
| P5が一般のPRで5.84%有利 | 認定しない |
| rを増やせばP5の利益が増幅する | 認定しない |
| 追加r grid・分子展開・SP組合せ | 今回開始しない |
| 次の担当 | Codex：機構に直結する準備を一束にする |
| 科学計算の自動再開 | 認めない。mandatory STOPを維持 |

## 1. 証拠の水準と取得範囲

### 1.1 今回取得した主要資料

D1結果報告、保存値分析報告、固定contract、保存監査、表示用資源表、D1のaccounting/pipeline source、分析sourceの関連部分、既存generator/native lowering、四つの原eventファイルの冒頭root記録、G10完成ノートの関連部分をGitHubから取得した。[R1–R13]

前回の広い②レビュー原本も、会話添付として読み直した。既存成果の発展を、独立した数学primitiveの発明だけに限定しない方針を継承する。[R14]

外部では、固定ensembleのcost-aware importance samplingとSCU/CTSの関係を一次本文で限定確認した。[P1,P2] 新たな広域新規性検索や全論文の再証明は行っていない。

### 1.2 今回実施していないこと

- 全1890 event、36合成sequence、147/150公開資料の全件byte照合を独立再実行していない。
- 原eventファイルは、費用差の具体例を確認するためroot三件ずつを主に読んだ。全rawを別プログラムで再集計したとはしない。
- 新しいHamiltonian、係数、event、trajectory、行列、IR、合成sequenceは生成していない。
- 科学runner、repository tests、native compile、T合成、sampling、LP、GPUは実行していない。
- 相手環境のworktree clean、親dirty差分保護、全保護台帳の一致はCodexの保存監査として確認した。直接ローカル環境を観測していない。

今回の数値計算は、**取得済みの保存整数・表示スカラーの比、表示用の丸め、静的な件数の算術**に限定した。rootの費用値は原保存記録である。後述の数式は本レビューでの解析であり、新実験の観測結果ではない。

### 1.3 証拠区分

本書では以下を区別する。

1. **保存結果**：D1・保存分析に記録された値や実行状態。
2. **source確認**：固定実装の静的な処理内容。
3. **本レビューの解析**：条件を明示した式の展開、保存スカラーからの算術。
4. **研究判断・提案**：結果の意味、継続範囲、次の比較設計。
5. **未確認**：新条件への移送、独立新規性、全PR費用、実samplerのonline費用等。

## 2. D1で何を比較したか

### 2.1 固定task

対象は3-system-qubitのexplicit synthetic involution providerであり、分子・DFではない。

- p=(1/5,3/10,1/2)、総dimensionless時間 τ=5/7、符号+、state=000。
- Q0=Z0、Q1=V1†Z1V1、Q2=V2†Z2V2。
- V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)。
- R_P(θ)=exp(−iθP/2)、R=Σp_iQ_i。
- degree 5、反復r=1,2、各stepのx=τ/r。
- 比較方式：ordinary、partial-return-tail、closed P3-tail、local full、closed P5、matched CTS。
- ideal physical target：U=exp(−iτR)。

同じrの各表現は、同じ理想有限平均

\[
M_r=\left[P_5(-i\tau R/r)\right]^r
\]

を対象とする。finite-bit係数まで全方式でbyte-exactに同じとは限らず、その差は共通予算へ戻す。異なるrでは有限近似M_r自体が違うが、比較先のUは同じである。[R1,R3]

### 2.2 精度と費用

各axisの誤差1/200、各axisのfailure 49/24000、各rowのresource failure 1/12000。24 axesと12 rowsを合計してfamilywise 1/20である。複素誤差は、両axisが条件を満たした場合にsqrt(2)/200以下。これらは実量子shotで測った失敗率ではなく、固定した十分予算の会計である。[R3]

probability 160 bit、root 256 bit、η=ρ=10^−12、primitive strict error 10^−6。nativeは共通literal loweringとstep内の隣接inverse cancellation。**step間fusionは許さない**。Tはnative1Qに含まれ、両者を足したscoreは使わない。

この比較は、公平な同一モデルでの資源推定として意味がある。一方、最適な全回路compiler、最小必要shot数、最適な各方式のr・精度配分、分子PR全体に対する比較ではない。

### 2.3 実行・来歴

12/12行、24 axes、1890 positive event bindings、36 keys（固定reuse19・新規17）、retry=0、outer exit=0。保存監査はIR・sequenceの価格とstrict-error記録、dyadic law、moment、積費用、terminal/source/authorizationを照合している。[R1,R4]

この完了は、欠測keyを0で補った比較や、失敗prefixだけを採用した結果とは違う。限定モデルを評価するための必要な計測は揃っている。

## 3. 結果：限定されたpositiveとtrade-offを保持する

以下はh=0の加法的native費用切片。単位は10^6 gates。表示値は丸めで、元のexact rationalが正本である。[R1,R2,R5]

| r | 方式 | T | CX | native1Q | N/axis |
|---|---|---:|---:|---:|---:|
| 1 | ordinary | 584.443449 | 34.499010 | 1569.837147 | 2088098 |
| 1 | partial | 433.770906 | 26.769358 | 1152.160879 | 1590438 |
| 1 | closed P3 | 448.433937 | 26.923137 | 1156.942453 | 1564916 |
| 1 | local full | 431.491134 | 26.388528 | 1091.078338 | 1803142 |
| 1 | closed P5 | 428.730865 | 26.219719 | 1084.098657 | 1536780 |
| 1 | CTS | 610.540632 | 16.367460 | 1526.132205 | 2675065 |
| 2 | ordinary | 515.022952 | 27.904618 | 1344.170250 | 926574 |
| 2 | partial | 456.252299 | 23.480168 | 1163.682829 | 778141 |
| 2 | closed P3 | 443.162079 | 23.591285 | 1081.327155 | 776048 |
| 2 | local full | 430.039337 | 23.532129 | 1108.697460 | 847756 |
| 2 | closed P5 | 429.626133 | 23.509518 | 1107.632166 | 773966 |
| 2 | CTS | 643.093987 | 15.227487 | 1623.187930 | 1185751 |

Tでは、各rの登録六方式内でP5が最小。CXではCTS、native1Qではr=1がP5、r=2がclosed P3である。

**P5のT改善を、すべての資源指標の改善へ読み替えない。** また、全quantum座標でのParetoと、未知のclassical費用を含む全資源Paretoは別である。

D1の主要な前進は、旧G10のfinite-only比較へ、物理targetに対するTaylor remainderと短い反復を戻しても、固定policyのT利益が残ると確認したことである。ただしr=1では旧native価格を再利用しており、新しい独立な性能確認ではない。

## 4. 費用差の意味：P5が「より正確なP5」を実装したわけではない

### 4.1 同じrでの会計

stepのnonzero確率をZ、nonzero条件付きのstep単価をd_k、各axisの全trial十分予算をNとする。全stepの古典drawを終えてから量子初期化する契約では、

\[
G_k(h_k)=2NZ^r(rd_k+h_k).
\tag{1}
\]

P5とpartialはZ=1であり、同じr内のTaylor/native bias予算も同じ。そのため、同rの比較におけるN差は、biasが改善したことではなく、主にbudget用M2上界の差に由来する。[R2,R6]

| r | N_P5/N_partial | d_T,P5/d_T,partial | G_T,P5(0)/G_T,partial(0) |
|---|---:|---:|---:|
| 1 | 0.966262124019 | 1.02289103826 | 0.988380867273 |
| 2 | 0.994634648476 | 0.946721065724 | 0.941641574412 |

r=1：shot予算が約3.37%減り、単価が約2.29%増え、積として約1.16%減る。
r=2：shot予算が約0.54%減り、単価も約5.33%減り、積として約5.84%減る。

百分率をそのまま足すのではなく、比を掛ける。例えばr=1は0.966262…×1.022891…=0.988381…である。

### 4.2 Shapley配分は因果の分離ではない

分析sourceはmultilinearな積factorのinteractionを対称に配り、保存endpointの差へのexact conservationを要求している。その計算は記述的な分解として適切である。[R7]

ただし、r=1の「shot寄与293.688%」や、r=2の「単価寄与91.051%」は、独立の実験で原因を操作した割合ではない。符号が逆の寄与が相殺すると、net差を分母にした割合は100%を超える。

またd_Tは、eventの構成、確率、回転角、native列、合成列を混ぜた条件付き平均である。91%という配分を「91%は短いwordのおかげ」「91%は合成角度のおかげ」と言い換えてはいけない。

### 4.3 研究に使える定量的な判定条件

同じr・Z=1でn=N_P5/N_partial、c=d_T,P5/d_T,partialとすると、h=0でP5が低費用になる条件は

\[
nc<1\quad\Longleftrightarrow\quad c<1/n.
\tag{2}
\]

保存Nからの算術では、r=1で単価増加を約3.4916%まで許容し、r=2では約0.5394%までしか許容しない。実際は、r=1の単価増2.2891%はその範囲内、r=2の単価は逆に低下していた。

これは保存会計に対するexactな境界の説明であって、新しい普遍的な最適化定理ではない。新しい改善率閾値を採用したものでもない。

## 5. 単価の中身：root eventに具体的な合成価格の反転がある

原eventのindex 0–2を、rごと・方式ごとに確認した。いずれもword=[]のroot群であり、child 0/1/2に対応する。**root群は集約後の群であり、生のTaylor次数0だけを意味しない。**[R8–R11]

| r | 方式 | rootのangle tangent | root event T数（child0,1,2） |
|---|---|---|---|
| 1 | partial | 2845/3717 | 132, 134, 136 |
| 1 | P5 | 2189372/2921709 | 136, 138, 140 |
| 2 | partial | 2333/6426 | 144, 146, 148 |
| 2 | P5 | 18205486/50387421 | 136, 138, 140 |

同じchildではnative回路の骨格・provider部分は同じで、回転角が異なる。child0の列は、二つの符号反転したRと二つのCXである。r=1ではP5が4 T高く、r=2では8 T低い。

ここから支持できるのは、**単価の符号変化に対応する、固定precision・固定合成計画の具体的な差が存在する**ことまでである。全eventの単価差の何%をrootが説明するかは、本レビューでは再集計していない。

同じchildのroot eventでも角度は異なるため、個々のeventが同じunitaryを実装しているわけではない。比較が同じtargetを保つのは、係数と他eventを含む平均を戻したときである。

また、一つの合成結果のT数を、その角度の最小T数や全compilerでの費用と呼ばない。r=2の価格差を「偶然」と断定する根拠も、「構造的に常に残る」と一般化する根拠もない。

次に必要な価格分析は、この具体的な観測から始める。全部のrや分子を増やす前に、どのkeyの出現頻度と価格が差を支えているかを明確にすべきである。

## 6. r=1→2：相対差の拡大は、P5の絶対費用の改善ではない

### 6.1 比較する有限近似が変わっている

| 量 | r=1 | r=2 |
|---|---:|---:|
| Taylor product上界 | 0.0005533762860 | 0.00001729308370 |
| implementation product上界 | 0.00001600000800 | 0.00003200054869 |
| axis bias上界 | 0.001138752588 | 0.00009858726477 |
| 残余統計幅s | 0.003861247412 | 0.004901412735 |

r=2ではTaylor上界が減り、native実装誤差の上界は増える。このためNが減った部分をrepresentation改善へまとめて帰属しない。[R2,R3]

sourceは、有限Taylor平均をunitaryと仮定せず、積のnorm envelopeを用いている。ε=3x^6/720とstep実装上界eから、Taylor部分(1+ε)^r−1と、実装部分(1+ε+e)^r−(1+ε)^rを分ける。各axisへ保守的factor2で戻す。今回、この予算を真のsignalや保存残差へ置き換えて小さくしていない。[R6]

### 6.2 絶対T切片

P5のr=2/r=1は約1.002088、すなわち約0.2088%増。一方、partialは約5.1828%増、ordinaryは約11.8781%減である。[R2,R5]

従って「rを増やしたらP5がより安くなったから、改善率が1.16%から5.84%へ増えた」とは説明できない。

さらに、**h=0で各方式がr=1,2からそれぞれT最小を選べる場合、P5もpartialもr=1を選ぶ。そこでの差は約1.16%であり、5.84%ではない。** これは保存された二候補内の選択であり、全r最適化ではない。

### 6.3 小stepの理想normalizationとの整合

前回の設計では、固定pの理想canonical表現について

\[
B_{partial}(x)=1+(1-\chi)x^2+O(x^4),\quad
B_{P5}(x)=1+(1-\chi)x^2+O(x^4),\quad\chi=\sum_i p_i^2
\]

を整理した。[R12,R14] 固定τ、x=τ/rなら両者のideal moment比のlogはO(τ^4/r^3)である。

D1のN差がr=2で小さくなったことはこの見方と矛盾しないが、二点で漸近則を検証したとは言えない。native価格はこの滑らかなnormalization展開に含まれない。

## 7. 全方式比較とh依存：pair交点と最小包絡線を区別する

保存されたquantum-only Paretoは、h=0でr=1がP5/CTS、r=2がpartial/closed P3/P5/CTSである。T、CX、1Q、workspaceに関する限定で、classical費用は含まない。[R2,R5]

各coordinateの共通準備・読出し単価h_kを動かすと、最小包絡線は以下のようになる。

| 比較集合 | T | CX | native1Q |
|---|---|---|---|
| r=1、六方式 | P5 | CTS→P5、h_CX≈4.327676971 | P5 |
| r=2、六方式 | P5 | CTS→P5、h_CX≈10.05625615 | P3→P5、h_1Q≈6317.245619 |
| 全12行 | r1 P5→r2 P5、h_T≈0.5868195982 | r2 CTS→r2 P5、h_CX≈10.05625615 | r2 P3→r2 P5、h_1Q≈6317.245619 |

partial/P5間のr=2 CX交点3.514949878は、その二方式の交点でしかない。その点ではCTSが全体最小のままである。この交点を「研究上の最良方式が切り替わる境界」として前面に出してはいけない。[R2]

hは物理機器で測定した値ではない。h_T、h_CX、h_1Qは別coordinateであり、Tが1Qに含まれる等の実現可能性の制約もある。任意に同じ数値を入れて資源を足し合わせたり、P5が勝つhだけを選んだりしない。

## 8. local fullが近い理由を、単純な「shotが多い」で片付けない

### 8.1 保存結果

local fullはP5よりNが大きいが、全stepがnonzeroのときだけ量子初期化する。そのため、h=0のT費用差は、P5を基準とせずlocal fullを分母にすると、r=1で約0.6397%、r=2で約0.0961%にとどまる。[R2,R5]

これは、直接normalizerを持つP5が、local生成に比べて必ず大きな量子利益を得るという解釈を支持しない。逆にclassical費用が同じであることも示さない。

### 8.2 本レビューの条件付き解析：古典段階でのzero-fill

以下はD1の会計を理解するための理想化した解析である。有限bitの二つの実装が厳密にこの形に一致するとの認定ではない。

同じevent ensemble M=Σ_e a_eV_e、a_e>0、B=Σ_e a_eに対し、canonical法は確率a_e/B、weight Bでsampleする。

一方、envelope E≥Bを使い、nonzero eventの確率a_e/E、残りを古典段階のzeroとする法を考える。nonzero weightはE、受理率Z=B/E、受理後のevent分布はcanonicalと同じ。

独立r stepでは、

\[
Z_{path}=(B/E)^r,\quad M_{2,z}=(EB)^r,\quad W_z=E^r.
\]

N*=L(2M2/s²+4W/(3s))をceiling前の同じBernstein式とすると、

\[
Z_{path}N_z^*=L\left(\frac{2B^{2r}}{s^2}+\frac{4B^r}{3s}\right)=N_{can}^*.
\tag{3}
\]

受理後のpath分布・量子実装が同じなら、量子費用の条件付き期待値も同じになる。ceiling後の期待呼出し数の差は、一axisで絶対値1未満の範囲に収まる。この式は、**すべてのzero判定を量子初期化前に終え、zeroの量子費用を0とする**場合に限る。

実際のlocal予算でBを直接使えず、U≥Bという上界をM2へ用いると、理想式では

\[
Z_{path}N_{z,U}^*=L\left(\frac{2(BU)^r}{s^2}+\frac{4B^r}{3s}\right).
\tag{4}
\]

normalizer上界の緩さを表すU/Bと、古典trialの費用が残る。D1ではさらにdyadic proposal、root rounding、range上界、有限係数targetの差があるため、式(3)だけで全差を説明しない。[R6,R12]

この解析は、zero-fill・再重み付けに関する既知の確率原理を、D1の「量子実行前拒否」へ適用したもの。独立した新しいsampling定理の優先性を主張しない。P1のerror-detection型zero-fillは量子実行後のfailureも扱うため、その課金を本式と無条件に同一視しない。[P1]

### 8.3 研究上の含意

P5の群normalizer取得とlocal生成の比較は、量子shot数だけでなく、**どこで何を知り、どこで拒否し、古典的に何回処理するか**を含む問題である。この観点は前回選んだ情報・生成・実装の研究課題に対応する。

Bを参照として用いた式の導出と、未知Bを運用predictorに無料で注入することは別。今回、旧予算へreference Bを混ぜる変更はしていない。

## 9. 現在残る限界と、その重要度

### 9.1 まず識別すべき限界

- **角度別合成価格の移送**：現在は固定precision・seed・toolの価格。rootで具体的なT差があるが、別precisionや合成規則へ移るかは未確認。
- **step間相殺**：全方式で無効にした同一モデル比較は有効だが、現実の結合回路へ順位が移るとは限らない。r=2では特に識別しやすい。
- **独立性**：p/providerは既知development。r=2の新angle取得を、独立分子validationや一般化と呼ばない。

### 9.2 目的に応じて必要となる限界

- classical online生成費用、normalizer・係数・Pauli collectionのsetup/償却費用。
- 他provider・入力分布・時間・精度へ移した成立領域。
- deterministic blockを挟んだfull PR、長時間、RPE/QPEの最終energy精度への接続。

これらを一度に全て測らなければD1が無価値、とはしない。次のclaimに必要なものを選ぶ。

保存setup CPU 0.176721秒、合成worker CPU 7.855002秒、全run CPU 631.759466秒を、per-trialの古典費用として使うことはできない。[R2,R4] CPU秒とT gate数の換算も未指定なので、そのまま一つの「総費用」へ足さない。

### 9.3 費用比較と数値誤差を混同しない

合成誤差10^−6や保存operator残差は、T費用モデルの相対誤差が10^−6であることを意味しない。exact rationalの費用差は固定モデル内で正確でも、compilerを変えたときの移送不確かさは別である。

同様に、Bernstein式のM2項が99%以上を占めるという記述は、実測量子varianceの99%を観測したという意味ではない。

## 10. 新規性・研究価値・論文化

### 10.1 何を新規性の代わりにしないか

固定ensembleの費用と二次momentの同時考慮やq∝p/√cという重要度samplingは既知である。SCU/CTSも、回路費用・測定overhead・Taylor誤差の交換を扱う。したがって、cost×shots、trade-off、ゼロtrialだけを新しい原理とはしない。[P1,P2]

一方、既知部品を使うことを不採択の自動条件にもしない。前回の②レビューは、finite-RTE固有の構成・生成・条件・適用知見を研究対象として採択している。[R14]

### 10.2 今回支持する主要claim

1. 指定P5表現は、同じphysical target・有限confidence・固定native modelへ接続できた。
2. 固定r=1,2ではP5のT費用がpartialを下回るが、shotと単価の内訳は異なる。
3. root eventの合成価格には、その符号変化に対応する具体例がある。
4. 最適方式は資源coordinateと準備・読出し単価に依存し、pair交点と全体包絡線は異なる。
5. local fullの古典zero-fillを量子実行回数へ正しく戻すと、Nの多さだけでは量子費用差を説明できない。

1–4は限定保存結果とその解釈。5の厳密な理想式は本レビューの条件付き解析であり、D1のfinite-bit全体を新たに認証した結果ではない。

### 10.3 まだ支持しないclaim

- P5がpartial/CTS family一般へ優越する。
- P5で実分子PRの総資源が5.84%減る。
- r増加により表現の利益が必ず増幅される。
- 既存方式ごとのr最適化後も5.84%差が残る。
- sourceのconstructor時間がproduction sampler費用を表す。
- Shapley配分が因果割合を同定した。
- 独立した新アルゴリズム論文として十分である。

### 10.4 目指す着地点

現在は、既存の数理構成と実装依存性を結ぶ、**方法・資源成立条件の研究**として発展させる段階である。例えば、

> 有限Taylor表現の代数的集約はいつ実装資源を減らすか：normalizer access、pre-quantum rejection、有限精度価格の役割。

という筋が考えられる。仮題であり採録可能性を保証しない。

D1一つのproviderと二つのrだけでは、独立論文の中心を十分に支えるとは評価しない。次には、現在の内訳から予測できる条件を作り、その予測を識別する前向きな確認が必要である。ただし、全条件でP5が最良になることを成功条件にしない。

## 11. 代替の次作業を比較した判断

| 選択肢 | 情報価値と問題点 | 今回の扱い |
|---|---|---|
| rを3,4,…へ広げる | 曲線は増えるが、角度価格と共有最適化の影響を未分離のまま増やす | 直ちに採択しない |
| 分子/DFへ接続 | 長期的には重要。現在の小さい差の原因を説明する前にaccess・control・他誤差を増やす | 後段 |
| SP/PAIを組み合わせる | 相互作用の候補だが、現価格差に加え新weight・channel意味論を導入 | 今回開始しない |
| 全online classical費用を測る | 全資源claimには重要。現在の最優先の機構識別とは別 | 問いを明示した限定benchmark案を後で判断 |
| 全保存factor/Pareto分析をもう一度行う | 既に完了しており重複 | 不要 |
| **合成価格とstep境界の機構を確認** | 現在の保存source・rawから具体的に問いが立つ。新しい物理入力を要求しない | **次の限定準備として推奨** |
| 研究を終了し文書化のみ | 現モデルのpositiveと具体的な機構手掛かりを残したまま早く閉じすぎる | 現時点では選ばない |

## 12. 次のCodex作業：機構検証D2の準備を一束にする

### 12.1 目的

**現在の利益がどのnative構造・合成価格に依存し、共通の簡約機会を追加したときにも関係が残るかを判別する。** 新手法探索をやり直す作業ではない。

ここでD2は本レビューでの便宜的な次段階名。既存の別研究のD2や旧runnerを再開する意味ではない。

### 12.2 保存証拠から先に整理するもの

D1原source・結果と保存値分析を読み取り専用の入力とする。全factor分析の再実行ではなく、次の追加対応を作る。

- event → root/非root、reduced word、child、angle key、nativeIR → 合成sequence価格の対応。
- nativeIRに含まれるliteral provider Tと、R placeholderから実装されるTを区別する。
- 各方式の期待合成key multiplicityを保存確率で求め、T単価を
  \[
  d_{i,T}=d_{i,T}^{literal}+\sum_k\mu_{ik}t_k
  \]
  の形で表示する。異なるeventを無理に一対一対応させない。
- 上式から、現在のlaw・Nを固定した価格の記号的変更に対する差を整理する。参照価格を動かした代数的感度と、実際に取得した合成価格を分ける。未知keyを無料としない。
- root三件の観測を、全単価差へ何割寄与したかと結び付ける場合は、使用する配分規約と確率変化を明示する。一意な因果割合とはしない。

これらは、実行済みの90 factor分解より一段下の「単価の中身」を扱う。新しい物理入力、parameter探索、再合成を伴わない形を優先する。

### 12.3 第一に準備する新しい対照：r=2の共通境界相殺

新しいrや分子の前に、既存r=2の保存step IRを順序どおりに結合し、**D1で使った同じexact adjacent inverse cancellationを、step境界にも全方式共通で適用する**限定対照を第一候補とする。

- 旧D1の `cross_step_fusion=false` は変更せず、新しい明示contractとして記録する。
- p、τ、state、degree、event law、weight、N、誤差上界、旧sequence価格を保持する。exactな回路簡約であれば旧の保守的誤差上界を維持できるが、その同値性を独立確認する。
- 相殺のために非可換な因子を並べ替えず、controlled scalar phaseを保持する。
- P5だけに簡約を与えず、六方式に同じpassを適用する。
- zero-fillは従来の全古典draw先行を維持し、零trialの処理を変えない。
- 新しいangle、合成、汎用compiler探索を追加せず、保存済みIR・sequenceで閉じる範囲を確認する。範囲外になれば理由を報告する。
- r=2の全positive pair数の静的上限は、保存support数から219,303である。これは実行件数の承認ではなく、準備時に処理量を見積もる材料。全pairの巨大JSON保存は要求せず、集約・境界state共有・streaming等をCodexが設計する。

これはまだ実行していない新しい回路評価である。単なる保存値の読取りだから自動実行してよい、とはしない。source準備と登録外テストの範囲、新しい実行scope、資源上限を定め、別途実行承認を受ける。

### 12.4 合成価格の前向きな確認

key感度から、少数の価格差だけに順位が依存すると分かった場合は、次に一つの独立した合成条件を検討できる。候補は、同じ角度に対する一つの未取得precision等である。

- 条件・比較対象・採用規則は結果取得前に固定する。
- 両方式へ同じprecisionとtool条件を与え、安いseedを探索して選ばない。
- より厳しいprecisionを使い、元の保守的bias予算・Nを維持すれば、価格だけの移送を調べる対照を作れる。誤差予算も最適化する別比較とは区別する。
- 新priceを得ても、旧resultを置き換えない。
- 同providerの合成条件確認は、分子・Hamiltonianの独立一般化ではない。

具体的precision値や新keys数を本レビューで勝手に固定しない。第一の境界相殺対照と必要性を比較し、Codexが最小範囲を提示する。何でも一度に試す計画にしない。

### 12.5 normalizer情報とclassical費用

式(3)–(4)の前提を既存finite-bit式に照合し、理想的な相殺と実際のU/B・丸め・proposal差を分ける。新normalizerや真のsignalを運用budgetへ無断注入しない。

online計測へ進む場合は、generator構築、参照全列挙、Pauli取得、random-bit消費、成功/零trial生成、cacheを分離する。validation/logging時間をalgorithm費用にしない。今回はこのbenchmarkを自動実行しない。

### 12.6 実務上の境界

- 新contractは旧contractの書換えでなく、継続する研究課題に対する新しい限定対照。
- 全結果、旧source、旧authorization、消費済みmarker、STOP、親dirty差分を保持。
- 実装の細部、cache、並列化、streaming、登録外testはCodexで一束に設計する。
- 科学的に採択した意味論を変更しない通常修正のたびにGPT詳細レビューを要求しない。
- 新しい比較対象、保証の変更、参照情報の使用、主要claimの変更が必要になった場合はその論点だけ戻す。
- 新しい科学実行・native評価・T合成は別途明示承認。全outcome後mandatory STOP。

## 13. 次に何をもって継続・縮小を判断するか

**P5が常に勝つことは条件にしない。** 結果と説明の関係を判断する。

| 得られた内容 | 次の科学的判断 |
|---|---|
| key/構造に基づく予測が、共通境界相殺や事前固定した確認条件でも成立 | 成立条件の研究として継続。次の独立入力を限定設計 |
| P5の優位が消えるが、どの価格・境界で反転するかを事前に説明できる | 使い分け・実装依存性の成果として評価。自動失敗にしない |
| 現価格の一点にだけ差があり、説明・移送ができない | 一般的な資源改善claimを縮小し、限定事例として残す |
| すべて既知の一般式の再掲で、finite-RTE固有の追加知見が残らない | 新規実験をさらに増やさず、既存ノートへ統合することを検討 |
| mean・phase・誤差会計に問題が見つかる | 性能比較と切り離し、原因を修正前に記録して科学的扱いを再検討 |

今回、新たな一律5%閾値、分子数、deadline、勝利率は置かない。効果の大きさは、次のclaimと検証不確かさに照らして判断する。

## 14. 最終判断

**②の主軸を継続する。** D1の固定taskにはpositiveと多資源trade-offがあり、保存値分析に加えてroot単価の具体的な対応も見えた。今は、以前の候補閉鎖と同じ理由で研究を休止し直す段階ではない。

同時に、r=2の5.84%をPR全体の改善率として押し出すことも避ける。次に価値があるのは、比較条件を増やすことより、**表現・normalization・合成価格・境界相殺から順位を予測できるか**の検証である。

主線はfinite-RTEの資源成立条件。FRは独立した第二候補として保持し、DF基底費用、Track A、N3へ無理に結び付けない。generic whole、Gaussian、相関sampler、その他既存closureは維持する。

**次はCodexで限定D2の準備をまとめる。新しい計算実行には別途承認を必要とし、それまでmandatory STOPを維持する。**

---

## 参考資料：内部の固定証拠

以下のURLは閲覧した固定資料の位置であり、第三者の実行環境をここで再現したという意味ではない。

- [R1] D1結果・引き継ぎ。`fc96571881f301435a655f057709df4d6c1678e0`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
- [R2] D1保存値分析・引き継ぎ。`2dcd56737be35526f6436a52835f59d817adf30c`。前半と後半を取得。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md
- [R3] D1固定contract。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json
- [R4] 保存結果・provenance監査。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/saved_output_audit/saved_output_audit_v1.json
- [R5] 保存表示用summary。表示値はexact authorityでない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/display_summary_v1.json
- [R6] D1 accounting/pipeline source。`874ca12d347f8f35e5d8d64f65f24abc4a2df523`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/accounting.py
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/pipeline.py
- [R7] 保存分析source。factor配分・N分解・包絡線・Paretoの関連部分を確認。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/scripts/tracks/algorithm_codesign/analyze_d1_saved_resources.py
- [R8] r=1 partial、root記録を含む原event先頭chunk。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/events/row_01.json.part0000.txt
- [R9] r=1 closed P5、原event冒頭のroot記録。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/events/row_04.json
- [R10] r=2 partial、原event先頭chunk。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/events/row_07.json.part0000.txt
- [R11] r=2 closed P5、原event冒頭のroot記録。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/events/row_10.json
- [R12] 既存generator/native lowering。`b9ed01455351628c9073748f5ba5751aa794b789`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g7_generator.py
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g9_native.py
- [R13] G6–G10完成ノートv1.0。今回は§5.2–7の構成・scope・旧費用表を主に確認。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
- [R14] 前回の広い②レビュー。会話添付の原本を読み直した。repository上の保存位置は既存引き継ぎに記載された以下。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8c9b5dbdf96dcc7b4af0605fc24921f47e0971bd/docs/research/track_b_existing_results_development_broad_scientific_review_20261013.md

## 参考資料：外部一次文献

- [P1] D. Cugini, T. A. Atif, Y. Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。§II/Theorem 1のcost×moment最適化、§IVのzero-fill/Discardに関する関連本文を確認。これらを新しいD1最適性定理として採用していない。
  https://arxiv.org/html/2603.13495v1
- [P2] J. Peetz, S. E. Smart, P. Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095v2、npj Quantum Information 12, 52 (2026)。CTSの有限展開・Pauli取得・samplingと誤差の関係を確認。repositoryのmatched CTSを同familyの全最適化法としない。
  https://arxiv.org/html/2407.21095v2
  https://www.nature.com/articles/s41534-025-01168-w

## 付録：保存スカラーの表示用算術

次の小計算だけを本レビューで行った。新しい科学データ・合成価格・比較条件を生成していない。

- r=1の許容単価増：1590438/1536780−1 ≈ 3.4915863%。
- r=2の許容単価増：778141/773966−1 ≈ 0.5394294%。
- P5/local fullのT比−1：r=1で約−0.6397048%、r=2で約−0.0960852%。表示済みT切片からの丸め計算。
- P5のT切片r2/r1−1：約+0.2088183%。
- r=2各方式のpair supportの静的合計：273²+264²+258²+63²+63²+24²=219303。実行認可や実行件数ではない。

原exact値の再認証と上記表示算術を混同しない。研究結果の正本は固定repositoryの原resultと分析artifactである。
