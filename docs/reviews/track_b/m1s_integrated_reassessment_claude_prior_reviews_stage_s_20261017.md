# M1-S：Claude・過去GPTレビュー・Stage S結果を統合した独立再評価

- 作成日：2026-10-17（JST）
- レビュー区分：Level 1（研究の中心的主張、新規性、継続／整理の判断）。GPTによる本セッションでの独立読解・紙上検討であり、別の研究者や別モデルによる追加レビュー実施を意味しない。
- 開始承認：ユーザー「レビューを開始して　Claude のレビューや今までのレビューも見直してほしい」。
- 主対象：アップロードされたClaude新規性レビュー、T1a/T1b原証明、Checkpoint 1/2、有限step結果レビュー、γ依存・Stage S設計レビュー、完了したStage Sの固定結果。
- 最新結果：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
- 適用運用指示：`research_project_instructions_unified_2026-10-17.md`。過去文書の2026-10-16版の適用記録は履歴として保持する。
- 実施：資料取得・読解、重要な一次論文の関連箇所確認、数式の紙上照合、既存の数値・分類の引用、研究判断の再評価、本Markdown作成。
- 未実施：科学SOURCEのimport／実行、保存H4数値の新規集計・fit・ランキング・shot forecast、Hamiltonian／signal／γ境界の再計算、数値テスト、回路生成・合成・compile、RPE simulation、GitHub書込み、契約採択、authorization／grant／marker発行。

**結論の状態：`INDEPENDENT_REASSESSMENT_COMPLETE / LIMITED_MATHEMATICAL_RESULTS_RETAINED / METHOD_NOVELTY_NOT_ESTABLISHED / FULL_THEOREM_PRIORITY_UNRESOLVED / STAGE_S_LOCAL_INTEGER_WITNESS_RETAINED / CORRECTED_TECHNICAL_NOTE_RECOMMENDED / NO_AUTOMATIC_STAGE_C / NOT_ADOPTED / MANDATORY_STOP`。**

---

## 1. 総合判断

**Claudeの「現状を強い新アルゴリズム・実用資源改善の原著論文として押し出すのは早い」という判断を支持する。一方、「既存の枠組みで表現できるからT1a/T1bの全保証も既知で、研究上の価値がない」という結論は支持しない。** Claude自身も、有限減衰のsecular保証が特定の既存定理の直接系かは未判定としている。[C, §§0,4,7]

今回の再評価は次の六点に集約される。

1. **構成上の位置付け**：完全乱択部分は方向制御された拡張HamiltonianへのqDRIFTとして厳密に表せる。ただし、部分乱択M1-S全体は、決定論的halfを残した「方向制御された決定論／qDRIFT混成対称積」である。非零の \(H_D\) を含む全系を通常の全項qDRIFTと無条件に同一視してはいけない。
2. **理論上の位置付け**：分散生成子、収縮性、Taylor剰余、Duhamel、平均化、Riesz射影は既知の道具である。一方、有限減衰を含むC-sec参照への固定モデル・固定rの一様保証は、理想unitaryへの小誤差評価と同じstatementではない。独立した定理statementのpriorityは未確定。
3. **分散の解釈**：「項内の量子分散だけが半減する」は、部分乱択では平均項を差し引いた \(P_j-h\) の量子分散として定義し直す必要がある。通常の \(\mathbb E_j\operatorname{Var}_k(P_j)\) と無条件に同じではない。原本の \(W,v,\rho\) の式を否定する指摘ではなく、解釈の精密化である。
4. **最新数値**：Stage SはG3・T=4でcap直下の整数十分shot設計にstrict削減の存在証拠を与えた。この前進は保持する。ただし実用の要求精度、実装、全roundを満たした資源改善ではない。
5. **費用に関する両レビューの修正**：整数shot削減は総費用削減の必要条件でも十分条件でもない。また共通の状態準備費用が必ず利得を薄めるわけではない。方向制御の一試行費用が下がる可能性も、増える可能性も、現状では未評価である。
6. **研究方針**：既存成果を、主張を修正した理論・信号解析のtechnical noteとしてまず確保することを推奨する。Stage Cへの自動進行は保留する。資源研究を続ける場合は、これ以上単独の補助gateを重ねるのでなく、同一推定task・利用可能情報・whole-wrapper費用・比較対象を一つの判断単位にまとめる。

この推奨は、M1-Sが一般に無効であるというNO-GOではない。独立論文が永久に不可能という判定でもない。現時点の証拠で主張できる範囲と、追加投資の合理性を分けた判断である。PR-2等の別研究との定量的な優先順位比較は今回行っていないため、「他研究の方が優れる」との採択も代行しない。

---

## 2. 何を実際に確認したか

### 2.1 添付・内部一次資料

|資料|今回の扱い|限界|
|---|---|---|
|Claude新規性レビュー [C]|添付全文、主張表・同値表現・還元・継続条件・未読範囲・付録を確認|Claudeが過去に実施したと記載する別レビューを、本セッションで再実施した事実にはしない|
|T1a原証明 [T1a]|GitHub固定版の主定理、定義、散逸性、収縮性、局所剰余、平均化の主要部分を確認|全補助例・全参照の網羅的監査、機械的形式証明ではない|
|T1b原証明 [T1b]|GitHub固定版の平均写像、二次生成子、主保証、mode／rate、ordinary／independent比較、二準位例、共鳴例、適用域を確認|全定数について独立した数値／形式検証をしたわけではない|
|Checkpoint 1 [CP1]|数学的受理範囲、新規性未認定、T1bを推奨した根拠を再読|過去の受理を新規性認定と取り違えない|
|Checkpoint 2 [CP2]|固定r、Jensen floor、等イベント比較、ρ単独の総資源STOP不採用を重点的に再読|共鳴がある場合のscalar等式とfloorを分離|
|有限step結果GPTレビュー [FP]|保存値、G6の解像性、モデル差会計、固定時間の一次位相機構を再読|H4スペクトルや係数の独立再生成ではない|
|γ科学条件レビュー [GS]|添付原本の数学、Task A/B、比較policy、過去の資源条件を再読|過去の資源案は最新版共通指示の新規契約規則を上書きしない|
|γ完了結果レビュー [GR]|添付原本および会話内の固定資料から主要結論と後続gateの論理を再確認|旧全チャットの全コマンドを監査したわけではない|
|RPE科学設計レビュー [RD]|添付・既存会話中の原資料に基づき、Task A/B、情報費用、phase／shot／費用の分離を再確認|全物理実装を認定しない|
|Stage S設計レビュー [SS]|添付全文を確認。単調性、cap極限、floor+1、no-witnessの限界を再検討|採択済み計算を再実行しない|
|Stage S結果 [SR,SM,SRAW]|固定GitHubの完了報告・保存summary・rawのprovenance／状態欄を取得。整数表は報告済み値を引用|全rawを新規に再集計・再採点していない|
|Stage S数値SOURCE [SK]|固定SOURCEの`kernels.py`を静的に確認|全runner／並列処理／全関数の形式検証ではない|

主要な判断は、原証明・直接確認した一次論文・添付レビュー・最新固定結果から行った。過去の各レビューに書かれた「検証済み」は、そのレビューの実施記録として扱い、今回独立に行った作業と区別する。

### 2.2 重要な外部一次文献

|文献|独立に確認した箇所|今回の用途|
|---|---|---|
|Campbell qDRIFT [P1]|Appendix D、Eq.(D1)、Fig.3の紙面|制御付きqDRIFTの既存性、抽象指数eventとnative費用の区別|
|Güntherほか PR [P2]|App.A.4の方向制御と独立RTE、App.B.1のπ/3条件、関連紙面|「半減trickがそのまま使えない」の意味、RPE接続の限定|
|Morisaki–Fujii [P3]|一次公開テキストのEq.(35)およびApp.Aの導出、Haar平均との区別|倍化平均演算子への適用と、C-sec保証との射程差|
|MLMC-qDRIFT [P4]|固定v2の§2.3、Eqs.(26)–(32)、fine/coarseの共有index|共有乱数をcoherent branchへ載せる先行例。ただしM1-Sと同一lawではない|
|Childs–Ostrander–Su [P5]|Eq.(6)の対称permutation積、mixing lemma|回文word・mean operatorを扱う既存原理|
|Abbottほか [P6]|coherent controlの実装依存性、cross blockの追加情報|system channelだけでは干渉が決まらないこと|
|Guほか [P7]|Theorem 1と関連modeの説明|固有位相の先導安定性と、有限時刻信号位相の区別|
|Yi–Crosson [P8]|出版本文の対象とspectral解析のscope、過去証拠との対応|決定論unitaryの結果を散逸平均写像へ無条件転用しない|
|Davies 1974 [P9]|出版社書誌・要旨、全文取得を試行|特定定理による包含は未認定。原論文全文を読了したとはしない|

MFの指定v1 PDF本体は今回の経路では取得できず、一次公開ページの取得テキストでEq.(35)と証明を確認した。原PDF bytes／hashの照合をしたとは主張しない。PRは版を明示しないPDF経路で関連箇所を確認したが、取得されたテキストと紙面でページ・図番号に差が見られたため、固定v2の全bytes一致を認定しない。本書では節名と実際に確認した数式内容を優先する。MLMCの本文は取得できたが該当ページのスクリーンショット取得に失敗したため、図を独立に視認したとは記録しない。

論文全体の新しい網羅的systematic reviewではない。30文献すべての全証明を本セッションで再読したとは主張しない。ただし、新規性判断の主要な根拠である制御qDRIFT、平均演算子bound、共有indexの構成は原論文の関連箇所へ戻って照合した。

---

## 3. Claudeの判断を、同意・修正・未判定に分ける

|Claudeの主張|本レビューの判定|理由|
|---|---|---|
|非CPなcoherence blockを扱うこと自体は新規性にならない|同意|物理的ancilla系のchannel／Heisenberg作用から厳密に表せる|
|M1-S全体はZ⊗Hに対する通常qDRIFTそのもの|**適用範囲を修正**|完全乱択部分はその表現ができるが、非零HDをexact halfで保持する部分乱択全体には決定論分割が残る|
|倍化Hamiltonianの平均演算子として表せる|同意、規格化を明示|K_jと相対時間の係数、決定論half、固定stateの行列要素を保つ必要がある|
|MF等からT1a/T1bの主要な枠組みが理解できる|同意|中心化分散、平均演算子、Duhamelは近い|
|T1a/T1bの全保証が既知定理の直接系と確定した|**その強い読み方は不支持**|Claude自身も有限減衰C-sec評価とDaviesへの直接包含を未判定としている|
|項内量子分散だけが半減する|**定義を明確化すれば支持**|部分乱択ではP_j−hの中心化揺らぎの量子分散。生のP_jの平均量子分散とは異なる|
|明示上界は分子では必ず空|**断定を弱める**|次元・gapに対して保守的だが、全分子・全stepで空と証明した資料はない|
|意味ある有限step比較は実質G3だけ|実用上の優先度に関する意見としては理解、観測の抹消は不可|G6にも条件付きで解像された絶対visibility差がある。小さい差と無証拠は違う|
|状態準備費が共通なら利得を薄める|**一般論として修正**|per-shot費用はshot数差に掛かる。共通の固定費とは異なる|
|整数shotは未検証|**時点更新**|Stage Sでcap直下の存在証拠が得られた。ただし実用taskでの優位ではない|
|原則ノート化、例外的に費用検証|条件を修正して推奨に概ね同意|停止の根拠は研究投資の優先度であって、一般NO-GOや全理論既知の証明ではない|

---

## 4. Ancilla lift：何が厳密に同じで、何が残るか

### 4.1 一つの共有event

M1-Sの原定義から、

\[
A_j=e^{i\theta P_j},\qquad
V_j=e^{-i\theta Z_a\otimes P_j}
=|0\rangle\langle0|\otimes A_j^\dagger
 +|1\rangle\langle1|\otimes A_j.
\tag{4.1}
\]

右端先作用の同じlabel列を使えば、

\[
V_{j_N}\cdots V_{j_1}
=|0\rangle\langle0|\otimes U_-
 +|1\rangle\langle1|\otimes U_+,
\]
\[
U_+=A_{j_N}\cdots A_{j_1},\qquad
U_-=A_{j_N}^\dagger\cdots A_{j_1}^\dagger.
\tag{4.2}
\]

一般には \(U_-\ne U_+^\dagger\)。この違いをancilla liftは消していない。むしろ、同じjoint lawを通常の拡張空間上のunitary列として厳密に表している。[T1b, §2; P1, App.D]

Heisenberg channelを \(\Lambda_j^\dagger(O)=V_j^\dagger O V_j\) と明示すると、

\[
\Lambda_j^\dagger(|0\rangle\langle1|\otimes X)
=|0\rangle\langle1|\otimes A_jXA_j.
\tag{4.3}
\]

したがって \(\Phi_\theta\) は、このHeisenberg channelのcoherence blockに一致する。Schrödinger pictureの同じ01 blockは逆の符号になるため、pictureとblockを省略して完全に同じと書かない。HadamardのX/Y読み出しの符号規約も保存する。

**ここから言えること**：非CPTPなblockを使ったから既存channel理論の外側にある、という新規性の根拠は成立しない。**言えないこと**：共有lawと独立lawは同じ、同じsystem marginalなら同じ位相信号、native費用も同じ、という結論。coherent controlの一般論でも実装・環境・相対位相の情報が必要である。[P6]

### 4.2 完全乱択と部分乱択

\(H_D=0\) の完全乱択では、(4.2)は方向制御されたHamiltonianへのqDRIFT列である。ただし、M1-Sの信号が読む相対時間を \(T\) とすると各branchは \(T/2\)。この因子を落として「同じ時間の既存qDRIFT」と比較しない。

部分乱択では、

\[
\mathsf D=e^{-i\delta Z_a\otimes H_D/4},\qquad
\mathsf V_\delta
=\mathsf D\left(\prod_j e^{-i\theta Z_a\otimes P_j}\right)\mathsf D.
\tag{4.4}
\]

これは、拡張Hamiltonianの決定論部と乱択部を分けた対称compositionである。\(H_D\) の全項をqDRIFTでsampleしているのではない。**正確な呼称は「方向制御された決定論／qDRIFT混成積」**であり、一般の部分乱択M1-Sを「全Hの通常qDRIFTそのもの」とする短縮は不適切。

この修正をしても、M1-Sが既知の構成原理の自然な組合せであるという評価は残る。新しいprimitiveとしての強い主張は弱い。しかし、既知class内の特定lawに新しい有用な性質が存在する可能性まで否定されない。

### 4.3 PR論文との関係

PRの方向制御half-timeの議論は、対称な決定論PFについて \(S_p(-\delta)=S_p(\delta)^\dagger\) を利用する。乱択wordにはその等式が一般にないため、乱択部へ同じ議論をそのまま移さず、独立RTEを用いる。[P2, App.A.4]

M1-Sは、その等式が一般に成立することを発見したのではない。等式を要求しない**別のcoherent平均信号**を採り、その減衰・位相を解析している。よって「PRの数学的誤りを訂正した」というclaimは採れない。新たな比較上の価値があるとすれば、その別信号が同じ推定taskをより少ない費用で実現できるか、または別の有用な信号保証を与えるかである。

---

## 5. ベクトル化とMF：還元の射程を区別する

### 5.1 平均写像の厳密な表示

列優先vecでは、

\[
\mathrm{vec}(AXB)=(B^{\mathsf T}\otimes A)\mathrm{vec}(X).
\]

従って

\[
\mathrm{vec}(A_jXA_j)
=e^{i\theta K_j}\mathrm{vec}(X),\qquad
K_j=P_j^{\mathsf T}\otimes I+I\otimes P_j.
\tag{5.1}
\]

\(K_j\) はHermitianであり、左と右の因子に同じjを使うことが相関として残る。\(\mathbb EK_j=h^{\mathsf T}\otimes I+I\otimes h\)。M1-Sの相対時間に合わせると

\[
\mathsf A_H=\frac{H^{\mathsf T}\otimes I+I\otimes H}{2},\qquad
\mathsf X_j=\frac{\lambda_R}{2}K_j.
\tag{5.2}
\]

部分乱択では \(H_D\) によるdeterministic sandwichが残る。倍化空間にしただけでouter splitting誤差がなくなるわけではない。[T1b, §§2–6]

信号は平均演算子の特定の行列要素である。\(e_k=\mathrm{vec}(|k\rangle\langle k|)\)、\(\Omega=\mathrm{vec}(I)\) とすれば \(\|e_k\|=1\)、\(\|\Omega\|=\sqrt n\)。初期vectorが規格化されたHaar-random stateとは異なることに注意する。

### 5.2 分散演算子の対応

\[
\mathsf D_R^{\rm var}
=\mathbb E\left(\mathsf X_j-\mathbb E\mathsf X_j\right)^2
=\frac{\lambda_R^2}{4}\mathbb E(K_j-\mathbb EK_j)^2.
\tag{5.3}
\]

T1bの二次散逸 \(Q=-\frac\beta2\mathbb E(J_j-L_R)^2\)、\(\beta=\lambda_R^2/(4r)\) をvecで書けば

\[
Q=-\frac{1}{2r}\mathsf D_R^{\rm var}.
\tag{5.4}
\]

これは中心化分散の既知構造そのものであり、この代数自体の新規性は主張しない。

### 5.3 MF Eq.(35)から直接得られること

MF Eq.(35)は、独立random exponentialsの平均演算子と理想unitaryとの差をtrace normで、\(t^2\operatorname{Tr}(D_p)/(2N)\) 以下に抑える。App.Aでは中心化された一回の誤差をDuhamel展開し、平均で一次項を消した後、収縮性とtelescopingを用いている。[P3]

従って完全乱択の倍化系については、

\[
|z-z_{\rm ideal}|
\le\sqrt n\,\|\overline V-U\|_{\rm op}
\le\sqrt n\,\|\overline V-U\|_1
\tag{5.5}
\]

として、固定状態信号への保守的なboundを移せる。これはHaar平均のsignal theoremを固定の基底状態へ転用するのとは違う。partial caseでは、random blockの誤差とouter deterministic splittingを別々に会計する必要がある。

### 5.4 MF Eq.(35)だけでは得られないこと

T1a/T1bは、\(N\theta^2\) または \(s\delta^2\) が有界なまま減衰が有限に残る領域で、**減衰込みC-sec参照への差がstepとともに0へ近づくこと**を評価している。理想unitaryへの差が小さいことを目的にしていない。[T1a,T1b]

\(t^2/N\) が有限で0へ行かない系列では、MFの理想unitaryへのboundをそのまま使っても、C-sec参照への \(O(\delta)\) 誤差や、共鳴blockのfloorは出ない。

したがって、

> 既知の平均演算子クラスへ含まれることは確認できるが、T1a/T1bの全statementがMFの定理の直接系であることは確認できない。

これが今回の判定である。有限次元の標準的な平均化・摂動論で証明できることは、その具体的保証が既に出版されていることとは別である。逆に、同じstatementが見つからなかったことだけで強い新規性を認定することもできない。Daviesの特定定理との全仮定・結論の対応は未検証のまま保持する。

---

## 6. T1a/T1bの数学的受理をどう見直したか

### 6.1 保持する中心構造

\(\mathcal J_jX=P_jX+XP_j\)、\(\mathcal L_R=\mathbb E\mathcal J_j\) とすれば、

\[
D_R=-\frac12\mathbb E(\mathcal J_j-\mathcal L_R)^2\preceq0.
\tag{6.1}
\]

HermitianなHS生成子の分散恒等式である。\(\Phi\) はunitaryな二側作用の凸平均でHS収縮。\(F_\delta=e^{i\delta A+\delta^2Q}\) もHermitian partが負半定値なので収縮する。局所 \(O(|\delta|^3)\) 差をtelescopingすると \(s|\delta|^3\)。\(\tau=s\delta^2\) が有界なら \(O(|\delta|)\) となる。[T1b, §§3–5]

secular pinchingとnear／far分解を加えた原本の保証は

\[
|z_s-z_s^{H,\rm sec}|
\le\sqrt n\left[K\tau|\delta|+\tau q_n
+2|\delta|\kappa_f\{1+\tau(q+q_0)\}\right].
\tag{6.2}
\]

これは**固定model、固定r、有限減衰予算の範囲の全sに一様**。任意の大系、縮むgap、rを同時に変えるfamily、任意の長時間へ一様という主張ではない。主要な構造と適用域を再確認した結果、以前の限定的な数学的受理を撤回すべき欠陥は確認しなかった。全定数の最適性、全proofの形式検証まで認定したわけではない。[CP1,CP2]

### 6.2 「分子では空」の断定

\(\sqrt n\) や逆gap依存は、実用上の保守性を示す強い懸念である。しかしこの因子だけで、すべての分子、すべてのstep、すべてのcouplingに対して上界が無内容と断定することはできない。couplingが0なら該当逆gap項は寄与せず、固定有限モデルでは十分小さいstepの非空域が原本にある。

したがって「分子への実用的な定量保証としての有効性は未実証」「大系では保守性が問題になり得る」と書くべきであり、「必ず空」と一般化しない。逆に、十分小さいstepで非空という数学的事実だけで量子資源的に有利とも言えない。[T1a, §1; T1b, §§5,11,13–15]

### 6.3 共鳴floor

対象固有値 \(E_k\) が非縮退なら \(\Pi_{E_k}(I)=E_{kk}\)。このblockに制限した \(Q_{E_k}\) は自己随伴で負半定値なので、

\[
z_s^{H,\rm sec}=e^{iTE_k}m(\tau),\qquad
m(\tau)=\sum_\ell w_\ell e^{\tau\mu_\ell},
\quad w_\ell\ge0,\quad\sum_\ell w_\ell=1,\quad\mu_\ell\le0.
\]

\(\sum w_\ell\mu_\ell=-\beta(W_k+v_k)\) とJensenから

\[
e^{-\tau\beta(W_k+v_k)}\le m(\tau)\le1.
\tag{6.3}
\]

この紙上の確認はCheckpoint 2の説明と整合する。SPがなくても非縮退対象なら参照のfloorと理想位相を得られる。一方、実際の有限step信号へは(6.2)の差を控除する必要がある。[CP2, §6]

補題の証明が短いことと、何を保証できるかを明確化する科学的効用は別問題。単独で強い新規性の根拠にしないが、ノートから削るべき無価値な部分でもない。冗長符号samplingを使った三準位反例はそのlaw内の反例であり、非冗長Pauli一般の反例へは拡張しない。

---

## 7. 最も重要な解釈修正：何の量子分散が半減するのか

### 7.1 原式

原本は、全Hの対象stateで

\[
W_k=1-\langle h^2\rangle_k,\qquad
v_k=\operatorname{Var}_j\langle P_j\rangle_k
\tag{7.1}
\]

と定義する。\(\langle\cdot\rangle_k\) は量子状態の期待値、\(\mathbb E_j\) はsampled label平均であり、両者を区別する。

\(\Delta P_j=P_j-h\) と置くと、\(\mathbb E_j\Delta P_j=0\) と \(P_j^2=I\) から

\[
W_k=\mathbb E_j\langle(\Delta P_j)^2\rangle_k,
\qquad
v_k=\mathbb E_j\langle\Delta P_j\rangle_k^2.
\tag{7.2}
\]

従って

\[
\boxed{w_k:=W_k-v_k
=\mathbb E_j\operatorname{Var}_k(P_j-h)\ge0.}
\tag{7.3}
\]

等しい総random event数Rの先導比較で \(\beta=\lambda_R^2/(4R)\) とすると、

\[
\gamma_{\rm shared}=\beta(w_k+2v_k),\qquad
\gamma_{\rm ordinary}=2\beta(w_k+v_k).
\tag{7.4}
\]

その意味で半分になるのはwの寄与であり、vの寄与は変わらない。[T1b, §§8,10; CP2, §8]

### 7.2 生のPauli項の分散とは異なる

定義を展開すると

\[
\boxed{w_k
=\mathbb E_j\operatorname{Var}_k(P_j)
-\operatorname{Var}_k(h).}
\tag{7.5}
\]

部分乱択では \(|k\rangle\) が全Hの固有状態でも、hの固有状態とは限らない。したがって \(\operatorname{Var}_k(h)\) を無条件に0とはできない。

Claudeの「項内の量子分散」という語を \(\operatorname{Var}_k(P_j-h)\) の意味で使うなら、(7.4)と整合する。通常の生の各項 \(P_j\) の量子分散 \(\operatorname{Var}_k(P_j)\) という意味に読ませると、一般のpartial caseでは誤りになる。**レビューやノートでは「中心化した乱択揺らぎの、状態に直交する量子成分」と明記することを推奨する。**

### 7.3 原本の既存二準位例でも区別が見える

T1b §11には、\(P_1=Z,P_2=X,p_1=p_2=1/2\)、\(h=(Z+X)/2\)、\(H_D=(Z-X)/2\)、\(H=Z\) という既存の紙上例がある。同原本は \(W=1/2\)、\(v=1/4\)、\(\operatorname{Var}_k(h)=1/4\) を記している。[T1b]

この既存例を(7.5)で読むと、中心化後のwと生の平均量子分散が一致しないことが分かる。新しいH4の数値計算や反例探索ではなく、原本がすでに区別していたstateの違いを、Claudeの解釈へ戻した確認である。

### 7.4 ρの意味を狭めすぎず、広げすぎない

\[
\rho=\frac{W+v}{2W}\in[1/2,1]\qquad(W>0)
\tag{7.6}
\]

は指定された先導減衰係数の比。SP下のscalar等式と、共鳴時の参照floorは別である。同じper-branch rの独立±との比較では、独立側は2倍のdraw数を使うため比較条件が異なる。\(r\)を変える有限step運用、整数rounding、shot予算、native単価、HD費用まで含む総資源比へρを転用しない。

したがって「利得は何をしても高々2倍」という全資源の断定も、「必ず半減できる」という断定も採らない。Checkpoint 2でρ単独STOP規則を不採用にした判断を維持する。

---

## 8. 有限step結果と過去の機構説明

G3だけでなくG6にも、保存半径に対して解像されたvisibility差とphase-bias増加の証拠がある。G6の絶対差は小さく、資源的重要性は未確立だが、観測自体を「意味がない」と削除してはいけない。G9は旧絶対差未解像を維持する。[FP, §§3.3–3.4]

また、T1bの固定有限減衰窓と、有限step pilotの固定 \(T=1,4\) は異なる極限である。固定Tでは \(\tau=s\delta^2=T\delta\to0\)。T1bがこの窓も含むことは、減衰差と同じ次数の位相誤差を相対的に分解した実証ではない。

過去GPTの紙上説明は、仮に対象が正確な全H固有stateなら、

\[
e^{-iTE_k}z_{\rm shared}(T)
=1+\delta\int_0^T\sum_a q_{ka}e^{it(E_a-E_k)}dt+O(\delta^2),
\]

\(q_{ka}=\langle k|Q(|a\rangle\langle a|)|k\rangle\) とし、off-diagonalな入力projector成分の振動結合から虚部の一次補正が生じ得る、というものだった。[FP, §5.2]

この機構は数理的に整合する。自己随伴な散逸生成子であっても、特定の入力と出力を持つ有限時刻の複素行列要素に一次位相が現れることと、孤立eigenmodeの固有位相の先導安定性は両立する。[T1b, §§7–8; P7]

ただしこの説明は、保存H4の全スペクトル・overlap・couplingを評価して原因を定量分解した結果ではない。Claudeの「理論で一般的に説明できる」を、H4の量的な帰属が済んだという意味にしてはいけない。ここでも、一般に生じ得る機構と、個別データの完全な説明は別である。

---

## 9. 最新Stage Sを取り込むと何が更新されるか

### 9.1 保存結果の引用

Stage Sは初回正常完了、SOURCE変更・retryなし。保存summaryと完了報告は次を記録する。[SR,SM]

|層|Stage Sの保存結果|
|---|---|
|G3・T=4|8対照で`CAP_EDGE_STRICT_INTEGER_WITNESS`|
|G3・T=1|8対照でcap直下の同数|
|G6・T=4|8対照でcap直下の同数|
|G6・T=1|8対照を旧π/3不成立としてcarry、新shot評価なし|
|G9|16対照を絶対差未解像としてcarry、新shot評価なし|

G3・T=4の既存整数表は次のとおり。NはX/Y両軸合計である。新しい比率や百分率の再集計は行っていない。

|δ [Ha^-1]|R|shared N|ordinary N|independent N|
|---|---:|---:|---:|---:|
|1/8|2|52|56|56|
|1/16|2|50|52|52|
|1/8|4|50|52|52|
|1/16|4|48|50|50|

これは4cell×2内部対照であり、8回の独立物理実験ではない。有限step→γ境界→π/3→Stage Sは、同じ保存信号を異なる設計目的で変換した一連の解析であり、独立な分子や独立SOURCEでの4回の再現とは数えない。

### 9.2 Capの数学は維持する

同一の十分設計で

\[
S_m(\gamma)=\frac{K}{[A_m^-\sin(\gamma-b_m^+)]^2},\qquad
n_m(\gamma)=\lceil S_m(\gamma)\rceil
\]

とする。正margin域ではSはγに対して連続かつ厳密減少なので、\(\gamma_c=\pi/3\) の直下極限は

\[
\boxed{n_m^{cap-}=\lfloor S_m(\gamma_c)\rfloor+1.}
\tag{9.1}
\]

端点そのものでのceilとは、S(cap)が整数の場合に異なる。固定SOURCEの`open_cap_integer`はこのfloor+1を実装している。[SK]

両方式の片側整数が解像され、\(n_s^{cap-}<n_c^{cap-}\) なら、同じγの非空開近傍でstrict削減が成り立つ。よって「π/3は開端点だから今回の陽性はすべて無効」と否定するのも誤りである。これは有効な局所存在証拠である。

逆に同数なら、より小さなγで整数差が出る可能性を排除できない。連続比のcap supremumを、整数比や総費用の全γ最大値に置き換えない。過去のStage Sレビュー本文はこの非対称性を概ね正しく保持しているが、チャットの「整数shot削減の最大限の余地」という短縮表現はこの限定を付ける必要がある。[SS, §§5–6]

### 9.3 Claudeの例外条件との関係

Claude原本はStage Sを未完了としている。この記述は最新結果に更新すべきである。しかし、Claudeの例外条件は「事前固定された共通の実用taskで、biasを支払った頑健な削減」であり、cap直下の理想Task Aの存在証拠より強い。[C, §§0.2,6.1,8,10]

今回の結果が満たしたのは、採択済みの仮想測定モデル・十分設計における局所整数差である。実際に必要なγ、状態／実装位相の余裕、whole-wrapper価格、強いbaseline、全roundエネルギー推定は未検証。したがって「例外条件が完全に満たされたからnative実験へ進む」とは判断しない。

95%は各方式行についての採択した統計設計の成功確率であり、保存半径や新規性や8対照の利得を95%で認証する数字ではない。[SM]

---

## 10. 費用会計：Claudeと過去GPTの両方を修正する

### 10.1 総費用の必要十分な会計対象

以下は、一時刻・同じ成功条件の二つの実行可能scheduleを比べるための記号的なモデルであり、現H4の未知価格へ数値を代入した結果ではない。

\[
G_s=F_s+N_s(C_s+P),\qquad
G_c=F_c+N_c(C_c+P).
\tag{10.1}
\]

Cは状態準備を除く全wrapperの一試行費用、Pは共通の一試行当たり状態準備費用、Fは一回の固定費。単位は比較可能な同一metricに限る。軸別価格が違えばX/Yを別々に合計する。時間、gate count、peak workspaceを無根拠に加算しない。

### 10.2 整数shot削減を必須gateにしてはいけない

\(N_s=N_c\) でも \(C_s<C_c\) なら、他条件が同じ場合に総費用は減る。\(N_s<N_c\) でも \(C_s\) が高ければ総費用は増える。

**従って整数shot削減は総費用削減の必要条件でも十分条件でもない。**

過去GPTの「整数削減が残る場合にだけStage C」という案内は、shot削減で単価増を相殺する限定routeとしては理解できるが、一般の資源研究の進行条件としては狭すぎる。directional Pauli回転と片枝controlled回転は異なる分解を持つため、単価の利得も損失も別にあり得る。[P1, Fig.3; P2, App.A.4]

本レビューはCが実際に小さいと計算したわけではない。「同数だから無意味」「少数shotの差だから総費用も必ず小さな改善」という双方の断定を撤回する。

### 10.3 共通状態準備費は必ず利得を薄めるか

(10.1)から

\[
G_c-G_s=(F_c-F_s)+N_cC_c-N_sC_s+(N_c-N_s)P.
\tag{10.2}
\]

\(N_s<N_c\) なら、共通per-shot費Pの増加はこの式の絶対的な費用差をshared有利側へ動かす。総費用比の変化はCやFに依存する。共通の**固定費**を加えると比が1へ近づく場合と、共通の**per-shot準備費**を加える場合を混同してはいけない。

したがってClaude §6.2の「状態準備・較正は共通で総費用比を薄める」は一般的ではない。状態準備の支払い頻度と固定費を分離して修正する。この点では過去GPTの「同じPでもshotsが違えば相殺しない」という説明を維持する。[SS, §11; RD]

### 10.4 ρやcap shotだけからの最大利得論も採らない

\(\rho\ge1/2\) は先導減衰係数についての限定であり、whole-wrapperのnative価格、補正費、候補分割の選択、状態準備費、統計policyまで含めた一般的な最大資源利得の定理ではない。capでのN差も全domainの整数比の上限ではない。これらを実用費用の一般STOP条件にしない。[CP2, §§8–9]

---

## 11. これまでのGPTレビュー・進行判断の見直し

|過去の判断・案内|維持／修正|今回の説明|
|---|---|---|
|T1a/T1bの固定適用域での数学的受理|維持|核心の収縮・平均化・係数・floorに、受理を撤回すべき欠陥を確認しなかった|
|ρ単独から総資源STOPを出さない|維持|shot、単価、状態準備、全roundを含んでいない|
|有限stepのvisibilityとphase biasを同時に評価|維持|rate単独から位相推定利益は出ない|
|元モデル差をraw総biasに二重加算しない|維持|今後の実装誤差は別台帳であり元のηと違う|
|任意の一点γで勝敗を主張しない|目的を限定して維持|唯一の外生的γがなければ研究不能という一般ルールではない。全方式に共通の事前設計や感度解析も正当な選択肢|
|γ→π/3直下を楽観的screenとして使う|維持|診断として有効。ただし実用taskを代用しない|
|cap同数を全γの整数削減不可能としない|維持|ceilによる局所plateauを全域へ一般化しない|
|Stage S陽性を受けStage C準備を推奨|**条件を狭めて修正**|陽性は入力の存在証拠。具体的な費用・taskの判断をせず自動的に次stageへ進む根拠には不足|
|整数shot削減をStage Cへの必要条件にする|**一般則として撤回**|一試行費用の利得による別経路を除外してしまう|
|多くのgate通過を独立した科学証拠の蓄積のように扱う|**避ける**|同一保存データの再表現。独立再現・一般化ではない|
|新規契約にも一律のworker／RAM／time capを要求|**最新版では採らない**|2026-10-17 §8.4に従う。既採択の旧契約は保全|

また、論理回路の条件付き資源比較に、必ず実機SPAMの測定や全ハードウェア実験が必要というわけではない。理想gate・所定state準備などを**明示された評価モデル**として置く研究は可能である。その場合、実機で成立済みと主張しないことが必要なのであって、未取得の実機情報を無条件の追加必須条件にして研究範囲を膨らませない。

同様に、独立理論論文に複数分子の数値検証が必ず必要という規則もない。要求される証拠は主張する定理・一般性・実用的優位性に応じて決まる。これは新たな理想条件を現契約へ導入する提案ではなく、過去の進行案内を一般的な研究必須条件へ読み替えないための整理である。

---

## 12. 新規性・研究価値についての独立結論

### 12.1 アルゴリズム原理

**強い新規primitiveの主張は支持しない。** 制御qDRIFT、方向制御、相関したrandom word、平均演算子という既存原理との接続が明確であり、今回のM1-Sを全く新しいsimulationの原理として提示する根拠は弱い。[P1–P6]

ただし、完全に同じpartial law、同じ比較task、同じ有限減衰保証が既に公刊されている、とまでは確認していない。特定の既知classに属することと、そのclass内の特定命題の新規性は別である。

### 12.2 理論

**新しい証明技法の主張は支持しないが、特定lawに対する定理statementの独立性は未判定として残す。** MF Eq.(35)の範囲を越えるC-sec参照、共鳴、明示量化がある。標準技法で導けることだけで研究貢献がゼロとは言えない。一方、現状では「新定理として強い独立論文に十分」と積極認定するためのpriorityと意義の論証も不足する。

そのため、正確なlaw、中心化分散の解釈、有限減衰保証、有限時間の代償を一つの短いnoteにまとめるのが現時点で堅実である。外部発表を選ぶなら、特定先行定理との比較をその主張に必要な範囲で閉じる必要がある。

### 12.3 数値

**有用な限定診断として支持する。一般的資源改善の実証としては支持しない。** G3/G6のvisibility差、phase trade-off、γ境界、Stage Sの局所整数witnessは、それぞれ限定scopeを守れば残せる。Stage SはClaude原本に対する真正の更新だが、理論priorityをそれだけで変えるものではない。

### 12.4 研究方針

**既定推奨：M1-Sの既存成果を、修正したtechnical note／研究記録として確保し、資源改善路線への新規実行をいったん保留する。**

これはClaudeの縮小推奨と大筋で一致する。ただし根拠は「T1a/T1bの全結果が既知と証明されたから」ではない。現在の中心成果が限定的で、実用taskとcostの双方を満たす情報がなく、補助gateを追加しても主たる研究判断が自動的には閉じないからである。

既存承認下の別作業を本レビューで停止・改変する命令は出していない。今後何を採択するかはユーザーが決める。

---

## 13. 継続を選ぶ場合：次の判断を一つにまとめる

資源研究をなお継続する場合、次に必要なのは「さらに一つshot指標をPASSにすること」ではなく、**同一推定taskでの実行可能scheduleと全費用の比較を定義できるか**という判断である。

最低限、次を一つの小さな契約設計にまとめる。

- Task Aの条件付き論理資源比較なのか、Task Bの対象エネルギー推定なのかを明示し、必要な精度・成功条件・利用情報をそのscopeに合わせる。
- γは、実際の要求または事前に定めた共通の最適化／配分policyから決める。Stage Sで有利な点だけを選んだ探索と、独立検証を混同しない。
- 同じ位相／実装誤差の意味論で、shared、ordinary、independentのwhole-wrapperを比べる。実用最良法に対するclaimをするなら、決定論PFやPR型RTEなど強い対照も必要。
- Nと一試行費用を同時に見る。shot差がないことを単価改善経路のno-goにせず、shot差があることを費用GOにしない。
- 未取得state／HD／synthesis／SPAM等は、選んだscopeの仮定または未確定として明示する。黙って0にしない一方、範囲外の実機保証を不要に要求しない。
- 判定不能なら何が不足しているかを保存する。新しいscopeの再実行・精度変更・追加候補は別承認にする。

この設計が既存の情報だけで定まらず、研究目的も明確でないなら、**同じDecision Packetとレビューを繰り返さずノート整理を優先する**。具体的な回路・精度契約を選べた場合のみ、Codexに最小の実装・照合を委ねる。今回、数値γ、費用metric、改善閾値、回路数を勝手に採択していない。

---

## 14. ノートに残すべき最小構成

### 14.1 中心命題の表現案（未採択）

> 固定finite-dimensional Hamiltonianをexact deterministic部とHermitian involution-tailへ分割し、同位置labelを共有した方向制御乱択を用いると、coherent Hadamard信号は固定r・有限減衰の小step系列でC-sec参照へ一様に近づく。単一pair-sum blockでは先導減衰係数がβ(W+v)となり、等random-event数の対照に比べ、中心化した乱択揺らぎの状態直交成分の寄与が半減する。対象energyが非縮退なら共鳴を許す参照floorも得られる。一方、固定有限時間の信号位相には過渡的な一次補正が生じ得るため、この先導減衰比は位相推定・総資源の優位性を直接与えない。

ここに、実際の保存H4についてのvisibility／bias、γ境界、Stage Sの存在witnessと未解決を、条件付き数値例として添える。T1bの保証を数値pilotが実証したと書かない。

### 14.2 構成案

1. 既存qDRIFT／方向制御との位置付けと、完全乱択・部分乱択の違い。
2. 共有／独立／ordinaryの正確なjoint lawと相対時間。
3. C-sec保証、SP、共鳴floor、中心化分散、適用域。
4. 固定時間の位相trade-offと、保存H4の限定例。
5. 同一の十分shot設計におけるγ境界とcap-edgeの整数witness。
6. 未達のresource claim、情報・費用会計、今後の再開条件。

方法の発明、理論statementのpriority、数値の一般性、実用の資源改善は別々のclaimとして書く。内部ノートを作るために、新しい分子計算や全RPE実装を必須にしない。

---

## 15. 確信度・残る不確実性・反対仮説

|判断|確信度|残るもの|
|---|---|---|
|完全乱択部のancilla lift、partial sandwichの区別|高（紙上代数）|実際のnative最適費用は未評価|
|中心化分散の恒等式と生の項分散との差|高（紙上代数）|Claudeの用語が中心化を意図していたなら、誤式というより定義不足の修正|
|C-sec保証はMF Eq.(35)と同じstatementではない|高|他の既存平均化定理による直接包含・priorityは未確定|
|T1a/T1bの限定数学的受理を維持|中〜高|全定数の形式検証、別SOURCEでの数値再現は行っていない|
|Stage Sの存在witnessを実用GOにしない|高|実用γ域の幅・実装誤差・cost不明|
|新primitiveとしての主張は弱い|中〜高|同一手法の全先行文献不存在は証明できていない|
|technical noteを優先する推奨|中|ユーザーの研究目的・理論成果への価値付けに依存|

重要な反対仮説は、方向制御によりwhole-wrapper価格が大きく下がり、shot差が小さくても実用的な利得が残る可能性、または有限減衰C-sec保証が既存理論にない十分に有用なstatementとして独立に評価される可能性である。いずれも現資料から否定しない。逆に、特定先行定理への厳密な包含や同一lawの既発表例が見つかれば、priorityの評価はさらに下がり得る。

**見つからないことを不存在とせず、既存技法であることを全成果既知ともせず、未知の実用費用を楽観・悲観どちらにも代入しない。**

---

## 16. 次の担当・承認・保全

今回の独立レビューは完了。**本書の推奨は未採択**であり、SOURCE・結果・正式契約・manifestの科学状態を変更していない。

推奨する次の担当はユーザーで、選択肢は「修正したノート整理を採る」か「同一taskとwhole-wrapper費用を一体化した限定比較の設計を採る」かである。前者を既定推奨とする。採択後にCodexが、既存成果を保全した別namespaceの文書整理／必要な技術準備を行う。単なる記録修正のためにGPTの同じ詳細レビューをもう一度開催する必要はない。

新規科学計算・保存値再解析・native build/compile・Task B実行は、固定された新scopeと別途明示実行承認を必要とする。Stage Sの消費済みauthorization／markerは再利用しない。

2026-10-17版§8.4に従い、新しいCodex指示で恣意的なCPU/RAM/worker/time上限を機械的に要求しない。実際の利用可能資源、科学call範囲、再現性と非干渉を確認し、重い処理は合理的に高速化する。共有環境全体や他ユーザーへの変更は禁止。既存の固定契約の資源条件は無断で変更しない。

**Mandatory STOPを維持する。**

---

## 17. 原資料と参照先

### 添付原本

[C] `m1s_prior_art_novelty_and_research_value_claude_review_20261017.md`。54,861 bytes、SHA256 `51dcfcd3a94300a4637a79d4fa3d453a07a3a16c87a000fcf5bfbce9bda0e50e`。本書でClaudeの主張として扱った部分はこの原本の§§0–12・付録A–Cによる。原本は変更していない。

[SS] `m1s_stage_s_scientific_conditions_review_20261017.md`。30,327 bytes、SHA256 `f815e886d9e4ff458a2a1398c3b41c4be2e4187e357ba58069eea9c3136b492e`。

[GS] `m1s_v1_scientific_conditions_independent_review_20261017.md`。28,470 bytes、SHA256 `a0d4a884d557e7f2337de8a88f31182b1176b7ab0d7c0bb3f2451a4712b8a575`。

[GR] `m1s_gamma_v1_completed_results_scientific_review_20261017.md`。31,978 bytes、SHA256 `18492941acc4fc7fe7fd46b6f002c21f324e52a661ec13bc0a024c5e86ad7303`。

[RD] `m1s_rpe_applicability_scientific_design_review_20261017.md`。29,915 bytes、SHA256 `3be81af2707ac60f9259f828d4428f691f0a7d45ca9ed17d5f44ab4f3f63a10b`。

共通運用指示2026-10-17版：38,191 bytes、SHA256 `d2ebdac4b0e44e871c65f240f317e6b5d3c3460f27d92f4d798992da90bf422f`。

### 固定GitHub資料

[T1a]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[T1b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[CP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md
[CP2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[FP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md
[SR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md
[SM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/saved_result_summary.json
[SRAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/artifacts/track_b_m1s_stage_s_result/2026-10-17/attempt_01/result.json
[SK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a338c4eba778c4ce1f189cfa8c8f050afb5df08/src/trottertracks/algorithm_codesign/m1s_stage_s_v1/kernels.py

- [T1a原証明][T1a]
- [T1b原証明][T1b]
- [Checkpoint 1][CP1]／[Checkpoint 2][CP2]
- [有限step結果レビュー][FP]
- [Stage S完了報告][SR]／[保存summary][SM]／[raw][SRAW]／[整数kernel][SK]
- [先行研究調査の固定索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/review_index.md)

### 外部一次文献

[P1]: https://arxiv.org/pdf/1811.08017
[P2]: https://arxiv.org/pdf/2503.05647
[P3]: https://arxiv.org/abs/2609.36694
[P4]: https://arxiv.org/pdf/2604.26865v2
[P5]: https://arxiv.org/pdf/1805.08385
[P6]: https://arxiv.org/pdf/1810.09826
[P7]: https://arxiv.org/pdf/2208.04100
[P8]: https://www.nature.com/articles/s41534-022-00548-w
[P9]: https://link.springer.com/article/10.1007/BF01608389

- [P1] Earl Campbell, *A random compiler for fast Hamiltonian simulation*. [一次PDF][P1]。制御付き発展はApp.D。
- [P2] Günther et al., *Phase estimation with partially randomized time evolution*. [一次PDF][P2]。方向制御はApp.A.4。本文・版照合の限界は§2.2参照。
- [P3] Hayata Morisaki and Keisuke Fujii, *Optimized Randomized Hamiltonian Simulation via Average-Error Analysis*. [一次公開ページ][P3]。平均演算子bound Eq.(35)とApp.A。PDF本体の独立bytes照合は未実施。
- [P4] Mohammadipour and Li, *MLMC-qDRIFT: Multilevel Variance Reduction for Randomized Quantum Hamiltonian Simulation*. [固定v2 PDF][P4]。共有fine/coarse構成は§2.3。
- [P5] Childs, Ostrander and Su, *Faster quantum simulation by randomization*. [一次PDF][P5]。Eq.(6)、mixing lemma。
- [P6] Abbott et al., *Communication through coherent control of quantum channels*. [一次PDF][P6]。実装依存性とEq.(5)周辺。
- [P7] Gu et al., *Noise-resilient phase estimation with randomized compiling*. [一次PDF][P7]。Theorem 1の仮定・対象。
- [P8] Yi and Crosson, *Spectral analysis of product formulas for quantum simulation*. [出版社本文][P8]。決定論unitaryのspectral解析。
- [P9] E. B. Davies, *Markovian master equations*, CMP 39, 91–110 (1974). [出版社ページ][P9]。今回全文未取得、直接包含の根拠としては使用しない。

### 保存・技術確認

本書のみを新しいUTF-8 Markdownとして作成した。添付原本6ファイルのbytes／SHA256を記録し、文書作成後の一致を確認した。これは文書保全の技術確認であり、科学データの数値再計算ではない。GitHub全ファイル・サーバー側worktreeを独立に再hashしたとは主張しない。GitHub操作はreadのみであり、旧科学SOURCE・結果・承認状態を変更しない。

**レビュー完了。推奨未採択。新規科学計算0。科学SOURCE変更0。GitHub書込み0。新authorization／grant／marker0。Mandatory STOP維持。**
