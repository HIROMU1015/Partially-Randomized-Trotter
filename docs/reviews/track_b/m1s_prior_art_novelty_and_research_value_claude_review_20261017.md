# M1-Sの先行研究・新規性・研究価値：Claude独立科学レビュー

**作成日**：2026-10-17（JST）  
**担当**：Claude（Anthropic）  
**レビュー対象**：M1-S（shared-label ± coherent randomization）のアルゴリズム構成、T1a／T1bの解析的結果、有限step・γ依存の数値結果、およびそれらの新規性・研究価値・論文化方針  
**状態**：`CLAUDE_NOVELTY_REVIEW_COMPLETED_NO_STANDALONE_NOVELTY_RECOMMEND_LIMITED_NOTE_NOT_ADOPTED`  
**権限と範囲**：ユーザーが依頼文書（`si.md`）で開始を明示承認した独立科学レビューのみ。新規科学計算、数値シミュレーション、保存raw結果の新しい科学的再解析、Hamiltonian・Pauli・signal・shotの新規forecast、native回路の生成・合成・compile、RPE simulation、SOURCE・契約・grant・ledger・marker・raw結果の変更、GitHubへの書込み、研究方針の正式採択の代行は行っていない。並行中のStage S（整数shot削減）の作業には干渉しておらず、その結果を推測して用いていない。**mandatory STOPを維持**する。

> **本レビューの姿勢**：研究の継続を前提にせず、Codexの調査結果や過去のGPT・Claudeレビューの追認も目的としない。一次論文、M1-Sの数学的定義、原証明、保存結果の意味論に戻って独立に判断する。主要な主張には証拠区分（§1.4）を付す。

---

## 0. 総合判断

### 0.1 新規性の判定

| 対象 | 判定 | 確信度 |
|---|---|---|
| **アルゴリズム構成** | **本質的な新規性は弱い。** M1-Sの回路はancilla liftにより「$Z_c\otimes H$ に対するqDRIFT」と厳密に一致し、Campbellの制御付きqDRIFT（$\lvert1\rangle\langle1\rvert\otimes H$）の方向制御版という自然な特殊形である。調査範囲（28一次論文）で同一の明示的構成は見つからないが、MLMC-qDRIFT（coherent枝での共有label）、PR（方向制御と乱択部の独立sampling）、COS（共有乱数による回文word）から容易に到達できる設計選択である | 中〜高 |
| **T1a／T1b** | **技法は標準的で、主要部分は既知の枠組みへ還元できる。** 平均信号はベクトル化により、Morisaki–Fujii（MF）型の平均時間発展演算子（倍化したHamiltonian $H^{\rm T}\otimes I+I\otimes H$ に対するqDRIFT）の特定の行列要素と一致する（付録A.2）。小減衰域はMFの評価を直接適用でき、有限減衰の長時間secular評価はMF・Chen・Campbellの射程外だが、標準的な平均化・摂動論の直接適用である。新しい知見として残るのは、**先導減衰係数 $\beta(W_k+v_k)$ と「項内の量子分散だけが半減する」という効率因子 $\rho$ の解釈**であり、これは単純だが情報量のある命題である | 中〜高 |
| **有限step・γ依存** | **独立した研究成果としては弱い。** 有効な比較は実質的にH4・G3の一分割に限られる（G6・G9は乱択部の1-normが極めて小さい）。visibility改善と位相biasのtrade-offは、固定時間における非secular結合の一次位相として理論的に説明できる一般機構の例示である。γ境界はHoeffdingの不等式と円板幾何による標準的な十分設計で、理論的新規性はない | 高 |
| **資源改善** | **未確立。** 整数shot、native費用、強いbaseline、多round RPEのいずれも未検証 | 高 |

### 0.2 研究方針の推奨

**既定の方針として、M1-Sを独立論文の主題とせず、T1a／T1b・有限時間trade-off・適用条件を限定ノート（technical note）として整理して区切ること（§8の選択肢5）を推奨する。研究資源は主研究（PR-2の同一精度resource／applicability研究など）へ戻すこと（選択肢6）を推奨する。**

例外として、進行中のStage Sが、**事前に固定した共通task**（位相biasを支払ったうえでの同じ精度・confidence）で整数shot削減を頑健に示した場合に限り、native費用と強いbaseline（決定論PF、PR型RTE）を含む**一点の判定**を行う価値がある（選択肢4の最小形）。その結果が肯定的であって初めて、資源改善を中心とする論文化（選択肢1または3）を再検討する。Stage Sの結果は仮定していない。

**確信度**：「独立した新規性が弱い」という判断は中〜高。研究方針の推奨は中（研究資源の優先順位というユーザー側の価値判断に依存するため）。

### 0.3 主要な理由

1. **構成の同値性**：M1-Sの物理回路は $Z_c\otimes H$ のqDRIFTそのものであり、平均信号はそのchannelのコヒーレンス・ブロック、あるいは倍化Hamiltonianの平均演算子として既存の枠組みで表せる（§3.1、付録A）。非CPなblockであることは新規性の根拠にならない（Abbottらの意味で、実装を指定すればcross blockは定まる）。
2. **理論の還元可能性**：T1a／T1bの証明は、平均演算子の積、分散生成子、Taylor剰余、telescoping、Duhamel、振動平均、Riesz射影の組合せである。結果の型（有限次元・固定モデル・逆gap・$\sqrt n$ 依存）も標準的で、分子規模では定量的に空になる。
3. **残る知見の大きさ**：$\rho=(W_k+v_k)/(2W_k)$ は、同じ乱択イベント数で共有±がordinary・独立±に対して項内分散を半分にするという明快な命題だが、PRの運用領域（固定step・多数の乱択gate）への移送、実分子での大きさ、位相biasの代償が未解決である。
4. **数値結果の射程**：H4の開発用oracle状態、事後設計、未認証の数値coverage、短い時間（$T=1,4\ {\rm Ha}^{-1}$）、実質一分割という制約のため、一般的な現象の実証ではない。
5. **費用対効果**：資源改善を確立するには、native費用、$H_D$ 実装、状態準備、bias処理、強いbaseline、多round RPE、別の分子系が必要で、得られる利得は乱択部について先導次数で最大2倍に限られる。

---

## 1. レビューの対象・開始承認・固定資料・取得状況

### 1.1 開始承認と範囲

依頼文書（zip内 `si.md`、25,376 bytes）で、新規性・科学的価値・論文化方針に関する詳細な独立科学レビューの開始が明示承認された。新規科学計算・simulation・保存結果の新しい数値再解析は承認されていない。

### 1.2 固定GitHub資料（raw取得、読み取り専用）

| 資料 | 固定commit | bytes | SHA256（先頭） | 取得状況 |
|---|---|---|---|---|
| 先行研究調査 README | 4301c0f8… | 11,555 | 0b39f6ee… | **取得・全文確認** |
| 文献一覧 bibliography.md | 4301c0f8… | 14,714 | 5c70fd78… | 取得・一部確認 |
| 一次証拠 primary_evidence.md | 4301c0f8… | 15,209 | 6924eb28… | **取得・全文確認** |
| 技術比較 comparison.md | 4301c0f8… | 7,770 | 33af4b93… | **取得・全文確認** |
| 資料索引 review_index.md | 4301c0f8… | 15,147 | 1e77ee75… | 取得・今回の判断には補助 |
| 監査 audit.md | 4301c0f8… | 7,274 | 60d910a9… | 取得・要点確認 |
| repository_evidence.json | 4301c0f8… | 33,044 | 76fb84d4… | 取得・今回の判断には補助 |
| T1a全証明 | 81d6103e… | 20,153 | 6d777495… | 取得（前回までに主要部を精読、今回同一hashを確認） |
| T1b全証明 | 81d6103e… | 32,550 | e7b621c4… | 取得（前回Checkpoint 2で全文を独立検証、今回同一hashを確認） |
| T1b Checkpoint 2 GPT独立レビュー | 81d6103e… | 52,863 | 89266825… | 取得・結論部と構成を確認 |
| 有限step結果報告 | 81d6103e… | 7,535 | 9a119228… | **取得・全文確認** |
| 有限step結果のGPT科学レビュー | 81d6103e… | 37,756 | a67ce892… | 取得・結論、§3.3、§5.2、§§8–9を確認 |
| γ依存V1結果報告 | 81d6103e… | 6,711 | 783456bb… | **取得・全文確認** |
| π/3限定gate結果報告 | 81d6103e… | 7,159 | d2f9933b… | **取得・全文確認** |
| 研究目的・研究課題 | 81d6103e… | 12,854 | 97ea6f50… | 取得・構成と研究目的を確認 |
| Shot・cost feasibility Decision Packet | fb5519bf… | 15,967 | 2a457560… | 取得・構成と冒頭のみ確認（非数値の設計資料。結果としては扱わない） |

完全なcommit hashは、調査 `4301c0f87a16be45b620304c3fa935d5aa2d721a`、研究基点 `81d6103ec89d485cb7336c3a90bd95dad66992aa`、shot・cost資料 `fb5519bf3f44392dc050feb9db44057dbc5207b1`。

### 1.3 一次論文

| 文献 | 取得版 | 今回の確認範囲 |
|---|---|---|
| Morisaki–Fujii, *Optimized Randomized Hamiltonian Simulation via Average-Error Analysis* | arXiv:2609.36694v1（PDF 375,380 bytes、SHA256 `ca8d749a…`） | **今回直接取得**。要旨、Eqs.(2)–(5)、平均時間発展演算子のbound Eq.(35)の周辺、主要語の出現（減衰・secular・長時間の解析が含まれないこと） |
| Mohammadipour–Li, MLMC-qDRIFT | arXiv:2604.26865v2（PDF 885,460 bytes、SHA256 `55e3b477…`） | **今回直接取得**。要旨、§2.3 Eqs.(29)–(32)（coherent二枝への共有label、$2Z_c\otimes O$ の測定と分散） |
| Günther et al.（PR） | arXiv:2503.05647v2 | 前回までに直接取得。App. A.4（Eq. A46と乱択部の独立RTE）、App. E |
| Kanasugi et al.（富士通） | arXiv:2603.22778v2 | 前回までに直接取得。App. B |
| Campbell、COS、SCU、Gu、Yi–Crosson、Abbott、KGW、Chen、WBC、Kim–García-Pintos、KLY、Hoeffding など | 調査の指定版 | **今回は本文を再読していない**。Codex調査の証拠カード（版・頁・式）と、筆者の既知の内容に基づいて比較した |
| Davies (1974) | — | **未取得**（Codexも書誌・要旨のみ）。特定の定理によるT1a／T1bの包含は主張しない |

### 1.4 証拠区分

| 表記 | 意味 |
|---|---|
| 【科学的事実】 | 一次論文・固定資料で確認した事実 |
| 【証明済み】 | 既存の証明（独立検証済みの範囲）で成立している事項 |
| 【保存数値】 | 保存結果の引用（新しい再解析ではない） |
| 【紙上検算】 | 本レビューで行った紙上の代数的検討 |
| 【条件付き推論】 | 明示した仮定の下での推論 |
| 【未検証仮説】 | 根拠はあるが検証されていない仮説 |
| 【推奨】 | Claudeの研究上の推奨 |

---

## 2. M1-Sの科学的背景と現在の到達点

- **構成**【科学的事実】：$H=H_D+H_R$、$H_R=\lambda_Rh$、$h=\sum_jp_jP_j$（$P_j$ はHermitian involution）。$D=e^{i\delta H_D/4}$、$A_j=e^{i\theta P_j}$、$\theta=\lambda_R\delta/(2r)$。正負branchで同位置のlabelを共有し、負枝は同じ時間順で各gateの符号を反転する（$U_-=A_{j_N}^\dagger\cdots A_{j_1}^\dagger\ne U_+^\dagger$）。平均写像 $\Phi_\theta(X)=\mathbb E_j[A_jXA_j]$、$\Psi_\delta(X)=D\Phi_\theta^r(DXD)D$。
- **理論**【証明済み】：T1a（完全乱択）とT1b（部分乱択、全Hのpair-sum射影）で、固定有限次元・固定r・小step・有限減衰予算の下、secular参照への全時刻一様な複素信号誤差bound（$\sqrt n$ と逆pair-sum gap依存）が受理された（GPT・Claudeの両Checkpoint 2レビュー）。(SP)下で先導減衰 $\beta(W_k+v_k)$、ordinaryとの比 $\rho=(W_k+v_k)/(2W_k)$。共鳴時はJensen型のfloor（非縮退の対象）。
- **数値**【保存数値】：H4（1.00 Å、STO-3G、8 spin orbitals、DF rank 12）、G3／G6／G9、$\delta=1/8,1/16\ {\rm Ha}^{-1}$、$R=2,4$、$T=1,4\ {\rm Ha}^{-1}$、zero control込み108行。G3・G6で各16対照すべてshared側のvisibilityが高い。G9は絶対差が未解像。shared側の位相biasが大きい（例：G3、$R=2$、$\delta=1/8$、$T=4$ でvisibility 0.9556対0.9234、bias −0.0079 rad対 +0.00035 rad。GPT有限stepレビュー§3.3の引用）。
- **γ依存**【保存数値】：$M_m(\gamma)=A_m^-\sin(\gamma-b_m^+)$ による同一の連続十分shot設計で、G3・G6は内部trade-off境界。π/3との限定照合でG3の16対照、G6（$T=4$）の8対照が成立、G6（$T=1$）の8対照は不成立、G9の16対照は対象外。
- **未確立**：整数shot削減（Stage Sが技術準備中）、native費用、総資源、多round RPE、別系への一般化、認証された厳密固有状態、新規性。

---

## 3. アルゴリズム構成の先行研究比較

### 3.1 三つの数学的な同値表現

【紙上検算】M1-Sの平均信号は、少なくとも次の三通りに既存の枠組みで表せる（導出は付録A）。

1. **ancilla lift（物理channel）**：方向制御gate $e^{\mp i\theta Z_c\otimes P_j}$ の列は、拡張Hamiltonian $H_{\rm dir}=Z_c\otimes H$（係数の1-normは $H$ と同じ）に対する通常のqDRIFTそのものである。Hadamard testのX／Y信号は、このCPTP channelの出力の線形汎関数であり、$\Phi$ はそのコヒーレンス・ブロック（$|0\rangle\langle1|\otimes X$ 部分）である。Campbellは制御付き発展を $\lvert1\rangle\langle1\rvert\otimes H$ に対するqDRIFTとして構成しており（App. D）、M1-Sはそれを方向制御（$Z_c\otimes H$）に替えた版である。
2. **ベクトル化（倍化Hamiltonian）**：$\mathrm{vec}(A_jXA_j)=(A_j^{\rm T}\otimes A_j)\mathrm{vec}(X)=e^{i\theta K_j}\mathrm{vec}(X)$、$K_j=P_j^{\rm T}\otimes I+I\otimes P_j$。したがって $\Phi_\theta^N$ は、倍化Hamiltonian $H^{(2)}=H^{\rm T}\otimes I+I\otimes H$ を相関した項 $K_j$（同じjを両因子に使う）でqDRIFTしたときの**平均時間発展演算子**そのものであり、信号は $\langle\bar k\otimes k|\,\overline V\,|\Omega\rangle$（$|\Omega\rangle=\mathrm{vec}(I)$、ノルム $\sqrt n$）という行列要素である。T1bのpair-sum周波数 $(E_a+E_b)/2$ は $H^{(2)}$ の固有値（の半分）であり、散逸生成子 $D_R$ はMFの分散演算子 $D_p=\mathbb E(X_j-H)^2$ を倍化系で書いたものに対応する。
3. **時間反転による表現**：実のPauli項（Yを偶数個含む。実の量子化学Hamiltonianの場合）では $A_j^\dagger=\overline{A_j}$ なので $U_-=\overline{U_+}$、干渉wordは $U_+^{\rm T}U_+$ となる。実の $\psi$ に対して信号は $\langle\overline{U_+\psi}|U_+\psi\rangle=\sum_i(U_+\psi)_i^2$ という非エルミートな双線形量である。これはsystem channelの量でないことを別の形で示すが、新しい原理ではない。

**意味**：同じ平均信号を与える既存の枠組みは存在する。しかし、**どのjoint law（共有か独立か）を選ぶかは、乱択イベント数・制御の構造・減衰・位相biasを変える実質的な設計選択**であり、表記だけの違いではない。問題は、その差が「研究上本質的な新しいアルゴリズム」と言えるほど大きいかである。

### 3.2 個別の比較

（Campbell、COS、SCU、Gu、Yi–Crosson、Abbott、KGW、Chenについては、今回は本文を再読しておらず、Codexの証拠カードと筆者の既知の内容に基づく。）

| 比較対象 | 共有するもの・law | branch・時間方向 | 観測対象 | 保証 | M1-Sとの関係 |
|---|---|---|---|---|---|
| **MLMC-qDRIFT**【科学的事実：今回確認】 | fine経路とcoarse経路が同じ乱択index列を共有 | 同じ時間方向、異なる深さ・角度 | $2Z_c\otimes O$ の期待値で経路差 $Y_\ell$ を得る（Eqs.(30)(31)）。単発分散は $4-Y_\ell^2$（Eq.(32)） | MLMCの分散削減と総gate複雑度 | **「coherent二枝に共有乱数を載せる」発想は明確な先行例**。M1-Sは同深さ・反対符号で、X／Yのコヒーレンス（干渉）から位相を得る点が異なる |
| **COS** | 一つのpermutationを対称積の前後で共有 | 同じ符号の前後逆順 | 平均channel | mixing lemma（$a^2+2b$）によるdiamond bound | M1-Sの干渉wordも回文になるが、IID復元抽出・等角Pauli・反対符号の分岐という点でlawが違う |
| **Morisaki–Fujii**【科学的事実：今回確認】 | 任意確率で項をsampling | — | 平均channel、Haar平均の誤差、Hadamard信号の誤差 | 平均時間発展演算子 $\lVert \overline V-U_t\rVert _1\le\frac{t^2}{2N}\mathrm{Tr}D_p$（全実時間・全N、Eq.(35)）。HS normに比例するsamplingの最適化 | **平均演算子と分散生成子の枠組みはM1-Sの平均信号を（ベクトル化により）含む**。ただし誤差は理想unitaryへの摂動的な評価で、有限減衰のsecular参照・共鳴・floorは扱わない（減衰・secular・長時間の解析は本文にない） |
| **PR（Günther et al.）**【科学的事実】 | 決定論部は方向制御、乱択部は正負で独立なRTE | 方向制御（Eq. A46） | 正規化したcoherent信号とRPE | Thm. B.2のRMS保証、App. Eの費用 | **最も直接の基盤**。M1-Sの差は「乱択部も同位置共有の方向制御にする」ことだけ。PRは厳密な随伴が成り立たないためこれを採らなかった |
| **SCU／CTS** | 左右のpropagatorを独立にsample | 二枝のcross term | $\mathrm{Tr}[Oe^{i\phi}V_1\rho V_2^\dagger]$ | normalizationによる $\lambda^2$ のshot overhead | 確率的unitaryの二枝cross termは既知。M1-Sは相関（共有）law |
| **Campbell qDRIFT** | IID項sampling | 制御付きは $\lvert1\rangle\langle1\rvert\otimes H$ | 平均channel、制御付きQPE | diamond bound | M1-Sは $Z_c\otimes H$ 版（§3.1の1） |
| **Abbott et al.** | 実装（変換行列）まで指定 | 二channelの干渉 | control-targetのcross block | 一般論 | cross blockは実装依存で、system channelだけでは決まらない。M1-Sの共有lawはその一つの実装にすぎず、新しい原理ではない |
| **Gu et al.** | — | — | 雑音付きunitary channelの固有値 | Hermitian Kraus雑音で固有位相が一次で不変、左右固有operator | M1-Sのmode・位相解析と同型の考え方 |
| **Yi–Crosson** | — | — | 決定論PFのスペクトル | 固有vectorの漏れとenergy shiftの分離、非対角摂動の二次評価 | T1bの三次位相（outer splitting、mode mixing）と同型の考え方 |
| **KGW** | Hamiltonian列の乱択 | — | 位相推定 | gapと隣接差、状態依存の項エネルギー分散 | 項間分散 $v_k$ と同型の量が既に現れる |
| **Chen et al.** | IID積の単一実現 | — | 演算子・状態 | 集中不等式 | ancilla liftした $H_{\rm dir}$ のqDRIFTとしてならchannel水準で適用できる。共有wordを独立な $2N$ gateとして扱うことはできない |
| **Kim–García-Pintos** | 挿入時刻 $u$ と $1-u$ の対称化 | 各step内の相関 | mixed-unitary channel | diamond誤差 | 相関による誤差相殺の別の型。label共有ではない |
| **Davies (1974)** | — | — | 弱結合極限の縮約ダイナミクス | secular（時間平均）生成子への収束 | T1a／T1bの平均化と同じ型の極限理論。**原論文は未取得で、特定定理による包含は主張しない** |

### 3.3 判定

- **既存方式の特殊例か**：M1-Sは「qDRIFTを方向制御の拡張Hamiltonian $Z_c\otimes H$ に適用する」ことと同値であり、制御付きqDRIFT（Campbell）と方向制御（PR、Reiherら）の組合せの**特殊例**である【紙上検算】。
- **数学的に同値になるか**：平均信号はancilla liftとベクトル化で既存の平均channel・平均演算子の枠組みに入る【紙上検算】。ただし、比較対象の方式（独立±、ordinary）とは平均信号そのものが異なる。
- **情報アクセス・回路費用・実装条件**：係数のみのaccessでよい点は既存qDRIFTと同じ。1 labelあたり一つの方向制御Pauli指数で両枝を動かす点は、独立±（1 labelあたり片枝制御一つ、同じ相対時間には2倍のlabel）と比べて回路構造が異なり、費用面の差になり得る【条件付き推論】。
- **差は本質的か**：**研究上の差は「乱択部へ方向制御を適用したときの信号の性質（項内分散の半減、位相bias）」に限られる**。アルゴリズムの原理としては新しくない。この差が本質的な成果になるかは、資源上の利得が確立するかにかかっている【推奨】。

---

## 4. T1a／T1bの数学的評価と新規性

### 4.1 数学的な正しさ

T1a（GPTのCheckpoint 1）とT1b（GPT・ClaudeのCheckpoint 2）は、固定有限次元・固定r・小step・有限減衰予算の適用域で受理されている【証明済み】。今回、T1a・T1bのhashが前回検証した版と同一であることを確認した。本レビューでは正しさの再検証ではなく、**既存定理との関係**を判断する。

### 4.2 既存の枠組みへ直接還元できる部分

| T1a／T1bの要素 | 既存の対応 | 判定 |
|---|---|---|
| IIDによる平均の積への分解、$\Phi^N$ | 平均channel／平均演算子の標準的な扱い（MF、Chen） | 既知 |
| 分散生成子 $D_R=-\tfrac12\mathbb E(J_j-L_R)^2\preceq0$ | MFの $D_p=\mathbb E(X_j-H)^2\succeq0$ を倍化系で書いたもの（付録A.2） | 既知の構造 |
| 小減衰域（理想unitaryへの近さ） | ベクトル化した系にMF Eq.(35)を適用すれば、$t^2/N$ に比例する誤差評価が直接得られる（ただし倍化空間のtrace normによる次元因子が付く。付録A.4） | **既知の評価から直接従う** |
| 一step Taylor剰余、HS収縮、telescoping | 標準的な技法 | 既知 |
| 相互作用表示、near／far分解、振動項の部分積分 | 古典的な平均化法、Davies型の弱結合・secular極限と同型 | 技法は既知 |
| Riesz射影による有効modeと固有値摂動 | Katoの解析的摂動論 | 既知 |
| pair-sum周波数と共鳴 | 倍化Hamiltonianのスペクトルとその縮退 | 既知の概念の言い換え |
| 三次位相の分離（outer splitting、有限角度、mode mixing） | BCHと固有値摂動。Yi–Crossonの「漏れとenergy shiftの分離」と同型 | 技法は既知 |

### 4.3 既存定理の射程外にある部分

1. **有限減衰のsecular参照への全時刻一様な評価**：MF・Chen・Campbellの評価は理想unitaryからのずれを $t^2/N$ のオーダーで抑えるもので、減衰が $O(1)$ の領域では有用でない。T1a／T1bは減衰を含む参照（C-sec）へ一様に近づくことを示す。この**statementとしては**、調査範囲で同じものは見つからない【科学的事実：調査範囲内】。ただし証明は標準的な平均化の直接適用であり、数学的に新しい技法ではない【条件付き推論】。Daviesの原論文を確認していないため、特定の定理の特殊化かどうかは判定しない。
2. **先導減衰係数 $\beta(W_k+v_k)$ と効率因子 $\rho$**：全Hの固有状態で $W_k=1-\langle h^2\rangle_k$ が正しい量であること、同じ乱択イベント数でordinary・独立±に対して因子 $\rho=(W_k+v_k)/(2W_k)$ になること。分散の分解 $W_k=v_k+w_k$（項間分散と項内の量子分散）により、**共有±が改善するのは項内分散だけ**という明快な解釈を持つ。計算そのものは対角成分の評価で簡単だが、科学的な情報量はある。
3. **共鳴時の単一rateの不成立とfloor**：三準位例（冗長な符号付きsamplingに依存）と、非縮退の対象でのJensen型の下界 $e^{-u\gamma_{\rm sh}}$。前者は射程が狭く、後者は自己随伴性と凸性から数行で従う補題である。
4. **具体的な明示定数**（$K$、$\kappa_f$、$C_{\log}$ など）：保守的で、分子規模では定量的に空になる。

### 4.4 Jensen floorなどの独立した価値

Jensen floorは、「共鳴があってもC-sec参照の振幅は単一rateの予測を下回らず、位相は厳密」という**解釈上の整理**として有用だが、技術的には自明に近く、**独立した科学的貢献とは言えない**【推奨】（この補題を前回提案したのは筆者自身であり、その評価もここで限定する）。

### 4.5 新規性が残る場合の最小の定理statement

【推奨】独立に主張し得る最小の命題は次の一つにまとめられる。

> **命題（候補）**：Hermitian involutionの乱択項をもつ部分乱択Hamiltonian $H=H_D+\lambda_R h$ について、正負branchで同位置のlabelを共有し符号だけを反転する方向制御型の乱択Hadamard信号は、固定有限次元・小step・有限減衰予算の長時間極限で、全Hの非縮退な固有値 $E_k$ に対し $z=e^{iTE_k}m(D)+O(\delta)$、$e^{-D(W_k+v_k)}\le m(D)\le1$ を満たす（単一pair-sum blockでは等号）。一方、同じ乱択イベント数の独立±・ordinaryでは先導減衰が $2DW_k$ となる。したがって、共有±は項内の量子分散 $W_k-v_k$ による減衰だけを半減し、必要イベント数は先導次数で高々 $\rho=(W_k+v_k)/(2W_k)$ 倍になる。

これに、固定時間では非secular結合による $O(\delta)$ の位相biasが生じるという補足（GPT有限stepレビュー§5.2の紙上機構）を対にすることで、「利得と代償」の両面を持つ一つのまとまった結果になる。

### 4.6 新規性の根拠として使うべきでない結果

- 平均写像が非CPであること（Abbottの意味で実装依存のcross blockにすぎない）。
- 分散恒等式、HS収縮性、Taylor剰余、telescoping、Duhamel、Riesz射影といった技法そのもの。
- $\sqrt n$・逆gap依存の明示上界（分子では空で、型も標準的）。
- 二準位例（存在例にすぎない）と三準位の共鳴例（冗長samplingに依存）。
- Jensen floor（凸性からの自明に近い帰結）。
- pair-sum共鳴という概念（倍化Hamiltonianの縮退の言い換え）。

---

## 5. 有限step・γ依存結果の科学的価値

### 5.1 一般的な現象か、特定入力の診断か

**特定入力の診断である。** 理由は次のとおり【保存数値・条件付き推論】。

1. **実質的に一分割**：GPT有限stepレビューが引用する保存値では、乱択部の1-normは G3 0.81 Ha、G6 0.026 Ha、G9 5.5×10⁻⁷ Ha で、先導減衰は $\lambda_R^2$ に比例する。G6の差は10⁻⁵程度、G9は未解像であり、意味のある比較は実質的にG3だけである。
2. **短い時間**：$T=1,4\ {\rm Ha}^{-1}$ はT1bの長時間極限ではなく、非secular結合による過渡的な位相が残る領域である。
3. **開発用oracle・事後設計・未認証coverage**：保存sector-ground stateは認証された厳密固有状態ではなく、計算は固定のH4開発データに対する事後解析で、数値coverageは条件付きengineering（`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`）である。held-out検証ではない。

### 5.2 visibility改善と位相biasのtrade-off

G3でsharedのvisibilityが高く、位相biasが大きいという観測【保存数値】は、GPT有限stepレビュー§5.2の紙上機構（固定時間では入力の他の固有projector成分と出力の振動結合から一次の位相補正が生じ、ordinaryの一次補正は実数である）と整合する。**この機構は一般的で理論的に予測できる**ので、数値結果はその例示であり、予測不能な新現象の発見ではない【条件付き推論】。

一方、**研究計画にとっての価値**は小さくない。T1bの長時間の先導rateだけでは、有限時間の位相推定の利得を予測できないことを、具体的な分子入力で示したからである。これは「rateのheadroomから位相推定の利益へは飛躍できない」という、研究の方向付けに重要な否定的・限定的知見である（外部への新規性は低い）。

### 5.3 γ境界の理論的新規性

$M_m(\gamma)=A_m^-\sin(\gamma-b_m^+)$ と $N^{\rm suff}=\lceil4\log(2/\alpha)/M^2\rceil$ は、Hoeffdingの不等式（±1測定、二側化、union bound）と、信号の円板の幾何から得られる**標準的な十分設計**である【科学的事実】。理論的新規性はない。方法論上の道具（visibilityと位相biasを同じ尺度で比べる設計）としては有用だが、十分shot数であって最小必要shot数ではなく、整数化後の削減も別条件である。

### 5.4 結論を制限する要因

開発用oracle、事後選択（保存cellの全件保持は守られているが、gridは結果を見る前の固定条件であり、それ以上の一般化はできない）、未認証の数値coverage、G9の未解像、短い時間、一分子・実質一分割。したがって、**「M1-Sが有限stepでvisibilityを改善する」という一般命題は支持できない**。支持できるのは「H4・G3の固定条件で、visibility改善と位相biasの増加が同時に生じる」という条件付きの観測である。

---

## 6. RPE・総資源改善への接続と限界

### 6.1 整数shot削減が確認された場合の価値

（Stage Sの結果は仮定しない。以下は「もし確認されたら」の条件付き評価である。）

整数shot削減が、事前に固定した共通task（同じ位相精度・confidence、biasを支払った後）で頑健に確認されれば、研究価値は**一段上がる**が、それだけでは資源改善の主張にならない【条件付き推論】。理由は、1 shotあたりの費用が方式で異なり得ること、H4・G3という一つの入力に限られること、Task A（単一時刻の位相診断）とTask B（多roundのエネルギー推定）が別であることである。

### 6.2 さらに必要なもの

| 項目 | 必要な理由 |
|---|---|
| whole-wrapperのnative費用 | 方向制御Pauli指数（ancillaを含む）と片枝制御の価格は、HWP＋catalyst、STAR、Clifford＋Tなどのモデルで異なる。同じlabel数は同じnative費用を意味しない |
| $H_D$ の実装 | T1bはexactな決定論halfを仮定する。実装PFの内部誤差・費用は未評価 |
| 状態準備・較正 | 全方式に共通だが、総費用の比を薄める |
| bias情報の取得費用 | 無補正のbiasが誤差予算に収まらない場合、補正には追加情報（oracleでない）が必要 |
| 多round RPE | 単一時刻の位相から最終エネルギーのRMS誤差・失敗率へは別の保証が要る（PR Thm. B.2型） |
| 別の分子・分割 | 乱択部の比重が大きい系での確認 |

### 6.3 強いbaselineとの比較

**不可欠である。** 特にG6・G9のように乱択部が極小の分割では、決定論PF（方向制御による半減を含む）がほぼ最適で、M1-Sの利得は原理的に小さい。乱択部が大きい分割でも、PR型RTE（既知の因子2）、重点サンプリングqDRIFT、D/R分割の再最適化を同じ条件で比べなければ、資源改善は主張できない【推奨】。

### 6.4 資源改善が不成立でも残る価値

方法論的・理論的な価値は残るが**限定的**である：(i) 乱択部への方向制御が「無効ではないが、項内分散だけを半減し、有限時間で位相biasを生む」という整理、(ii) PRの「乱択部には半減trickを適用できない」という記述の射程（厳密な随伴の有無）の明確化、(iii) 有限時間では長時間rateから位相推定利得を予測できないという注意。いずれも短いnoteで十分に伝えられる規模である。

---

## 7. 支持できる新規性の主張／支持できない主張／未判定の主張

### 7.1 支持できる主張（限定付き）

| 主張 | 根拠 | 限定 |
|---|---|---|
| 乱択部へ同位置共有の方向制御を適用した干渉信号は、小step・有限減衰の長時間極限で、全Hの固有状態の $\beta(W_k+v_k)$ で減衰し、項内分散だけが半減する | T1a／T1b、Checkpoint 2の受理 | 固定有限次元・固定r・exact $H_D$。PRの運用領域（固定step・多数の乱択gate）は未解析 |
| 同じ乱択イベント数で、共有±は独立±・ordinaryに対して先導次数で効率因子 $\rho\in[1/2,1]$ を持つ | Checkpoint 2（GPT・Claude） | 先導係数の比。shot数・native費用・総資源の比ではない |
| 非縮退の対象では、共鳴があってもC-sec参照の振幅は単一rateの予測以上で、位相は厳密 | Checkpoint 2で両者が受理 | 自明に近い補題。真の信号には誤差評価が別に必要 |
| 固定時間では、shared側にvisibility改善と同時に一次の位相biasが生じ得る（H4・G3で観測） | 保存数値とGPT有限stepレビューの紙上機構 | 一分子・一分割・短時間・開発oracle |
| PRの記述は厳密な随伴の不成立についてであり、共有方向制御の信号が無効という意味ではない | PR App. A.4と本レビュー§3.1 | 「PRの誤りの指摘」ではない |

### 7.2 支持できない主張

| 主張 | 理由 |
|---|---|
| M1-Sは新しいHamiltonian simulation／位相推定アルゴリズムである | $Z_c\otimes H$ のqDRIFTと同値な自然な特殊例（§3） |
| 非CPなcross blockを扱うこと自体が新しい | Abbottらの意味で実装依存のcross blockにすぎない |
| T1a／T1bは新しい数学的技法を含む | 標準的な技法の組合せ（§4.2） |
| T1a／T1bの上界は分子で定量的な保証を与える | $\sqrt n$・逆gap依存で空になる |
| 有限stepでのvisibility改善は一般的な現象として示された | 実質H4・G3の一分割、短時間、開発oracle |
| γ境界の式は新しい理論である | Hoeffdingと円板幾何の標準的な十分設計 |
| M1-Sは量子化学の位相推定で資源を削減する | 整数shot、native費用、強いbaseline、多round RPEが未確立 |
| 位相推定の誤差・資源を半減する | 効率因子は乱択部の先導減衰についてだけで、位相biasの代償がある |

### 7.3 未判定の主張

| 主張 | 判定に必要なもの |
|---|---|
| 同一の共有方向制御qDRIFTが文献に存在しない | 調査は28論文で網羅的でない。量子制御・雑音分光・Loschmidt echo等の分野の追加確認 |
| T1a／T1bの一様評価が特定の古典定理（Davies 1974など）の特殊化である | 原論文の取得と、離散時間・非CP写像への対応の検討 |
| PRの運用領域でも効率因子が残る | 固定step・多数の乱択gate（有限 $\alpha$）での理論（前回レビューのT1c候補） |
| 整数shot削減が残る | Stage S（進行中。結果を仮定しない） |
| native費用で利得が残る | 費用モデルを固定したwhole-wrapper評価 |
| 非冗長なPauli表で共鳴による多rateが現れる | 解析的な例または反証 |

---

## 8. 研究の継続・修正・縮小・終了の比較

| 選択肢 | 既存研究との差 | 中心的主張 | 必要な追加証拠 | リスク | 評価 |
|---|---|---|---|---|---|
| 1. 新しいアルゴリズムとして提案 | 小さい（自然な特殊例） | 共有方向制御qDRIFTによる資源削減 | 整数shot、native費用、強いbaseline、多round RPE、複数分子 | 高（新規性の否定、利得の不成立） | **現時点では不可** |
| 2. 共有乱択coherent信号の理論論文 | 中〜小（技法は既知、命題は新しい可能性） | §4.5の命題と有限時間の位相bias | 運用領域（有限 $\alpha$）の扱い、文献の追加確認、きれいな証明の整理 | 中（査読で「標準技法の直接適用」と判断される可能性が高い） | 可能だが価値は限定的。短い理論noteとしてなら妥当 |
| 3. visibility／位相bias trade-offとRPE適用条件の方法論・数値研究 | 小（Hoeffding・円板幾何は標準） | trade-offの存在と適用条件 | 複数分子・分割、認証された状態、事前登録、Stage Sの結果 | 中〜高（一般性の不足） | 現データでは一般性が足りない |
| 4. 資源改善が成立するまで限定継続 | — | 資源改善 | Stage S → native費用＋強いbaselineの一点判定 → 拡張 | 中（費用が膨らむ） | **Stage Sが肯定的な場合に限り、最小形で妥当** |
| 5. 限定ノートとして整理して区切る | — | §4.5の命題、有限時間trade-off、適用条件と限界 | ほぼ不要（既存成果の整理） | 低 | **既定の推奨** |
| 6. 別の重要な研究方向へ転換 | — | — | — | 低（M1-Sの成果はノートに保全） | **5と組み合わせて推奨**（主研究のPR-2 resource／applicability研究など） |

**推奨順位**【推奨】：5＋6を既定とし、Stage Sが事前固定の共通taskで頑健な整数shot削減を示した場合に限って4（最小形）へ進む。4の一点判定が肯定的なら、1または3を資源改善の論文として再検討する。2は、ユーザーが理論的な外部発表を重視する場合の選択肢として、5のnoteを拡張する形で検討できる。

**M1-Sの限定的な終了を、Track B全体や乱択Hamiltonian simulation一般のno-goとは扱わない。**

---

## 9. 推奨される論文化の着地点と中心的claim

### 9.1 推奨する着地点

**限定ノート（technical note）**。内部記録として、またはユーザーが望む場合はarXivの短報として。独立した原著論文の水準には、現時点の新規性・一般性が足りない。

### 9.2 中心的claim（実際に成立している範囲）

> 乱択積公式の乱択部に方向制御を適用し、正負branchで乱択labelを共有すると、Hadamard信号は「項内の量子分散だけが半減した」減衰を示す（小step・有限減衰の長時間極限）。これは同じ乱択イベント数での効率因子 $\rho=(W+v)/(2W)\in[1/2,1]$ を与え、共鳴があっても参照振幅を損なわない。一方、有限時間では非secular結合による一次の位相biasが生じ、visibilityの利得はそのまま位相推定の利得にならない（H4での具体例）。

### 9.3 タイトル案

- 「乱択部への方向制御とbranch共有乱数：Hadamard信号のsecular減衰、効率因子、有限時間の位相bias」
- *Branch-shared randomization in directionally controlled Hadamard tests: secular attenuation, an efficiency factor, and finite-time phase bias*

### 9.4 主要図（既存の成果だけで作れるもの）

1. 回路の模式図：ordinary、独立±、共有±（$Z_c\otimes H$ のqDRIFTとしての同値性を併記）。
2. 分散の分解（項間 $v$／項内 $w$）と効率因子 $\rho$ の図。
3. H4・G3の保存値によるvisibilityと位相biasの対（既存の保存値の引用。新しい集計はしない）。
4. 三準位の共鳴例：振幅の二指数混合とJensen floorの比較（解析式）。

### 9.5 採択前に必須となる検証（外部発表する場合）

- 適用域の明記：小step・固定r・exact $H_D$。PRの運用領域が未解析であること。
- 先行研究への位置付け：ancilla lift（Campbell）、ベクトル化（MF）、MLMC、PR、COSとの関係を正確に書く。
- 文献の追加確認：調査外の分野（量子制御、Loschmidt echo、雑音分光）での同等構成の有無。
- 新しい数値は不要（内部noteの場合）。外部発表で数値を載せる場合は、開発用データであること、oracle状態、未認証coverageを明記する。

---

## 10. 次に必要な最小検証、担当、STOP・承認条件

| 順序 | 担当 | 内容 | 承認・STOP条件 |
|---|---|---|---|
| 1 | ユーザー | 本推奨（限定ノートで区切り、主研究へ戻す。Stage Sが肯定的な場合だけ一点判定）を採るか判断する | 正式採択はユーザーのみ |
| 2 | Codex（既存の承認範囲内） | Stage Sの技術準備・実施は既存の承認と契約に従う。本レビューは干渉しない | Stage Sの判定基準（共通task、許容bias、confidence）を**結果を見る前に**固定することを推奨 |
| 3 | Codex | 限定ノートの草稿（既存成果の整理のみ。新しい科学計算なし） | ユーザーの採択後 |
| 4 | GPT（任意） | 本レビューの新規性判断、特に§3.1の同値表現（ancilla lift、ベクトル化）と「独立した新規性は弱い」という結論の独立確認 | ユーザーの開始承認 |
| 5 | Codex | Stage Sが肯定的な場合に限り、native費用と強いbaseline（決定論PF、PR型RTE）を含む一点の判定 | **新しい研究契約と別途の科学計算承認が必要** |

**STOPの条件**：Stage Sで、事前固定の共通taskにおける整数shot削減が頑健に示されない場合は、M1-Sの資源研究を区切り、ノートで閉じる。一点判定でnative費用または強いbaselineに対する利得が残らない場合も同様。いずれも、M1-Sの限定的な終了であり一般的なno-goではない。

**mandatory STOP**：維持する。本レビューは、Stage S、native評価、ノート作成、文献の追加調査のいずれも開始していない。

---

## 11. 確信度、重要な反論、未取得資料、一次文献

### 11.1 確信度

| 判断 | 確信度 |
|---|---|
| アルゴリズム構成の本質的な新規性は弱い | 中〜高 |
| T1a／T1bは既知の枠組みへの直接的な適用であり、技法の新規性はない | 高 |
| 残る科学的知見は「項内分散の半減」と効率因子の解釈 | 中〜高 |
| 有限step・γ結果は特定入力の診断 | 高 |
| 資源改善は未確立 | 高 |
| 研究方針の推奨（5＋6、Stage Sに条件付きで4） | 中 |

### 11.2 重要な反論と、結論が変わる条件

1. **「PRが明示的に退けた操作が実は使えると示したことは、それ自体が論文になる」**：一理ある。ただし、現時点で示されているのは小step極限の減衰係数であり、PRの運用領域での効率、位相biasの代償を含む位相推定の利得、資源上の意味は示されていない。運用領域（有限 $\alpha$）の解析と、強いbaselineに対する資源上の利得が揃えば、この反論が優勢になる。
2. **「有限減衰のsecular一様評価は、MF等の既存評価が扱わない新しい型の保証である」**：statementとしては正しい。しかし証明は標準的な平均化の直接適用で、定数は分子で空である。数学の論文としての貢献は小さく、物理的な洞察（項内分散の半減）が主な価値である。Daviesの原論文の確認で、より直接の包含が見つかる可能性もある。
3. **「H4での観測は、より大きな系や長時間で利得が大きくなる兆候かもしれない」**：先導減衰は $\lambda_R^2$ と時間に比例するので、乱択部が大きい系・長い時間では差が大きくなり得る。しかし位相biasも同時に変わり、その収支は未知である。予測ではなく検証が必要。
4. **「同一構成が文献にないので新規である」**：本レビューはこの論法を採らない。調査は網羅的でなく、構成は既知の部品の自然な組合せである。

**結論が変わる条件**：(a) Stage Sと一点判定で、強いbaselineに対する同じ精度での資源上の利得が示される、(b) PRの運用領域で効率因子が残ることが示される、(c) 文献の追加確認で同一構成と同一の解析が見つかる（その場合は新規性の評価がさらに下がる）。

### 11.3 未取得・未確認の資料

- Davies (1974) の原論文（未取得）。
- Campbell、COS、SCU、Gu、Yi–Crosson、Abbott、KGW、Chen、WBC、Kim–García-Pintos、KLY、Hoeffdingの本文（今回は再読せず、Codexの証拠カードと筆者の既知の内容に依拠）。
- MFの証明全体（今回は要旨、Eqs.(2)–(5)、Eq.(35)周辺と、減衰・secular解析の不在の確認まで）。MLMCは§2.3の該当部分まで。
- Stage Sの結果（未完了。仮定していない）。
- review_index.md、repository_evidence.jsonに列挙されたSOURCE・raw結果・監査（今回の判断には不要として未精読）。

### 11.4 一次文献

- H. Morisaki, K. Fujii, *Optimized Randomized Hamiltonian Simulation via Average-Error Analysis*, [arXiv:2609.36694v1](https://arxiv.org/abs/2609.36694v1)
- P. Mohammadipour, X. Li, *MLMC-qDRIFT*, [arXiv:2604.26865v2](https://arxiv.org/abs/2604.26865v2)
- J. Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026); [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)
- A. M. Childs, A. Ostrander, Y. Su, *Faster quantum simulation by randomization*, Quantum 3, 182 (2019); [arXiv:1805.08385v2](https://arxiv.org/abs/1805.08385v2)
- J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, [arXiv:2407.21095v2](https://arxiv.org/abs/2407.21095v2)
- E. Campbell, *A random compiler for fast Hamiltonian simulation*, PRL 123, 070503 (2019); [arXiv:1811.08017](https://arxiv.org/abs/1811.08017)
- Gu et al., *Noise-resilient phase estimation with randomized compiling*, [arXiv:2208.04100v2](https://arxiv.org/abs/2208.04100v2)
- C. Yi, E. Crosson, *Spectral analysis of product formulas for quantum simulation*, npj Quantum Information (2022), [出版論文](https://www.nature.com/articles/s41534-022-00548-w)
- A. A. Abbott et al., *Communication through coherent control of quantum channels*, [arXiv:1810.09826v3](https://arxiv.org/abs/1810.09826v3)
- I. D. Kivlichan, C. E. Granade, N. Wiebe, *Phase estimation with randomized Hamiltonians*, [arXiv:1907.10070](https://arxiv.org/abs/1907.10070)
- C.-F. Chen, H.-Y. Huang, R. Kueng, J. A. Tropp, *Concentration for random product formulas*, PRX Quantum 2, 040305 (2021); [arXiv:2008.11751](https://arxiv.org/abs/2008.11751)
- K. Wan, M. Berta, E. Campbell, *Randomized quantum algorithm for statistical phase estimation*, PRL 129, 030503 (2022); [arXiv:2110.12071](https://arxiv.org/abs/2110.12071)
- T. Kanasugi et al., [arXiv:2603.22778v2](https://arxiv.org/abs/2603.22778v2)
- E. B. Davies, *Markovian master equations*, Commun. Math. Phys. 39, 91–110 (1974)（未取得）
- T. Kato, *Perturbation Theory for Linear Operators*（Springer）

GitHub固定資料のリンクは§1.2の各commitとpathに対応する（調査：`docs/research/m1s_prior_art_survey_20261017/`、研究：`docs/tracks/algorithm_codesign/` と `docs/research/`）。

---

## 12. これまでのGPT・Claudeレビューとの相違点と理由

| 論点 | 過去の判断 | 本レビュー | 理由 |
|---|---|---|---|
| T1bの数学的受理 | GPT・Claude（Checkpoint 2）とも適用域付きで受理 | **同意** | 今回は正しさを再検証せず、同一hashを確認 |
| Claude付録A.9（$\rho$ だけによる総資源STOP規則） | GPTは一般的なSTOP規則として不採用 | **GPTの判断に同意し、前回の筆者の提案を撤回** | shot数・contrast・bias・native単価が共通という追加仮定が必要で、有限step結果は位相biasの代償を示した |
| Gate 0（$\rho$ のheadroom診断）の位置付け | 前回Claude：C＋目的限定Gate 0。GPT：同様に条件付き | **優先度を下げる** | 有限step結果により、rateのheadroomが位相推定の利得に直結しないことが分かった。研究判断を決めるのは、biasを支払った後の整数shotと費用であり、Stage Sがそれを扱う |
| 研究の継続 | GPT有限stepレビュー：位相予算と費用の成立可能性を判断する限定契約の設計を推奨し、成立しなければノートで区切る | **既定をノートで区切る方へ寄せる**（Stage Sが肯定的な場合だけ一点判定） | 新規性の分析で、理論の主要部が既知の枠組みへ還元できること（ancilla lift、ベクトル化）を確認し、残る価値が資源上の利得に強く依存すると判断したため。また有効な数値比較が実質G3だけである |
| 新規性 | GPT（Checkpoint 1・2）：独立新規性は未認定 | **弱いと判断**（未認定から一歩進めた評価） | 同値表現と技法の対応を具体的に示した。ただし網羅的な文献確認ではないので「新規性なし」と断定はしない |
| 前回Claudeの「M1-Sは $Z_c\otimes H$ のqDRIFT」 | 前回Claudeが提示、GPTが支持 | **維持し、ベクトル化（MF型平均演算子）を追加** | 理論面の還元可能性をより具体的にした |

---

## 付録A　紙上検算

### A.1 ancilla lift

$e^{-i\theta Z_c\otimes P}=|0\rangle\langle0|\otimes e^{-i\theta P}+|1\rangle\langle1|\otimes e^{i\theta P}$。乱択labelを共有した列は $\prod_ke^{-i\theta Z_c\otimes P_{j_k}}=|0\rangle\langle0|\otimes U_-+|1\rangle\langle1|\otimes U_+$ で、これは $H_{\rm dir}=Z_c\otimes H$（項 $Z_c\otimes P_j$、確率 $p_j$）に対するqDRIFTの一つの実現である（符号規約は除く）。$|+\rangle$ ancillaのX／Y測定は $\langle\psi|U_-^\dagger U_+|\psi\rangle$ を与え、その期待値は $\mathbb E[U_-^\dagger U_+]=\Phi^N(I)$。CPTP channel $\mathbb E[V(\cdot)V^\dagger]$ の $|0\rangle\langle1|$ ブロックへの制限が $\Phi$（θの符号規約を除く）である。

### A.2 ベクトル化と倍化Hamiltonian

列優先のベクトル化で $\mathrm{vec}(AXB)=(B^{\rm T}\otimes A)\mathrm{vec}(X)$。$A_j=e^{i\theta P_j}$ に対し $A_j^{\rm T}=e^{i\theta P_j^{\rm T}}$ で、$P_j^{\rm T}\otimes I$ と $I\otimes P_j$ は可換だから
$$\mathrm{vec}(\Phi_\theta(X))=\mathbb E_j\bigl[e^{i\theta K_j}\bigr]\mathrm{vec}(X),\qquad K_j=P_j^{\rm T}\otimes I+I\otimes P_j .$$
$K_j$ はHermitian（固有値 $-2,0,2$）で、$\mathbb EK_j=h^{\rm T}\otimes I+I\otimes h$。したがって $\Phi_\theta^N$ は、項 $K_j$ を確率 $p_j$ で選ぶqDRIFTの平均時間発展演算子（MFの $\overline V$）である。信号は $\langle k|\Phi^N(I)|k\rangle=\mathrm{vec}(|k\rangle\langle k|)^\dagger\,\overline V\,\mathrm{vec}(I)$。分散演算子は $\mathbb E(K_j-\mathbb EK_j)^2$ で、そのHS表示がT1aの $-2D_R$（規格化を除く）に一致する。

### A.3 pair-sumは倍化スペクトル

$H^{(2)}=H^{\rm T}\otimes I+I\otimes H$ の固有値は $E_a+E_b$（固有vector $\overline{|a\rangle}\otimes|b\rangle$）。T1bの $A=J_H/2$ の周波数 $(E_a+E_b)/2$ はその半分であり、pair-sum共鳴は $H^{(2)}$ の固有値の縮退にほかならない。

### A.4 MFの評価の直接適用（小減衰域）

MF Eq.(35)は任意のHermitian項について $\|\overline V_{p,t}^{(N)}-U_t\|_1\le\frac{t^2}{2N}\mathrm{Tr}D_p$ を与える。A.2の倍化系に適用すれば、M1-Sの平均信号と理想値のずれは、$\|\overline V-U\|_{\rm op}\|\mathrm{vec}(I)\|\le\sqrt n\,\|\overline V-U\|_1$ により $t^2/N$ に比例する量で抑えられる。ただし倍化空間（次元 $n^2$）でのtraceのため次元因子が付き、減衰が $O(1)$ になる領域では有用でない。**MFの枠組みが小減衰域を直接扱えること**を示すための紙上の対応であり、新しい定理として主張しない。

### A.5 実のPauli項での時間反転表現

$P_j$ が実対称なら $A_j^\dagger=\overline{A_j}$、$U_-=\overline{U_+}$、$U_-^\dagger U_+=U_+^{\rm T}U_+$。実の $\psi$ で $\langle\psi|U_+^{\rm T}U_+|\psi\rangle=\sum_i(U_+\psi)_i^2$。

---

## 付録B　区別した事項

| 区別 | 本レビューでの扱い |
|---|---|
| 科学的事実／証明済み／保存数値／条件付き推論／未検証仮説／推奨 | §1.4の区分で表記 |
| 先行研究比較での対象・law・情報access・保証・baseline・費用の一致 | §3.2の表 |
| 技術的PASSと科学的GO | §5.4（engineering PASSは科学的GOでない） |
| 固定H4開発データとheld-out検証 | §5.1（held-outと呼ばない） |
| 十分shot数と最小必要shot数 | §5.3 |
| 有限のnegative resultと一般的NO-GO | §5.4、§8 |
| 部分的な類似と全体の同一性 | §3.3（MLMC、MF、PRそれぞれ部分的な重なり） |
| 同じlabel数／branchあたりgate数／全Pauli指数数／native費用／総資源 | §3.3、§6.2 |
| 同一手法の不在と新規性 | §11.2（4） |
| 既知の道具の使用と全成果の既知 | §4.3（statementとして射程外の部分を区別） |

---

## 付録C　操作記録

- 依頼文書：zip内の `si.md`（25,376 bytes）を読んだ。
- GitHub：raw.githubusercontent.comからの読み取り（GET）で16ファイルを取得した。書込み、commit、push、branchの変更・merge、PR・Issueの操作は0件。
- 一次論文：arXivからMF（2609.36694v1）とMLMC-qDRIFT（2604.26865v2）のPDFを取得し、テキスト抽出して該当箇所を読んだ。
- 新規科学計算、数値シミュレーション、保存raw結果の新しい再解析、Hamiltonian・Pauli・signal・shotのforecast、native回路の生成・合成・compile、RPE simulation、SOURCE・契約・grant・ledger・marker・raw結果の変更、研究方針の採択の代行：**すべて0件**。
- 並行中のStage Sの作業には干渉していない。
- mandatory STOP：維持。

**最終状態：独立科学レビュー完了。M1-Sの独立した新規性は弱いと判断し、限定ノートで区切って主研究へ戻すことを既定の推奨とする（Stage Sが肯定的な場合に限り一点判定）。採択・計算・追加レビューの開始はいずれもユーザーの判断に残す。mandatory STOPを維持。**
