# Track B G3 科学的研究レビュー：有限J1 witness、B2条件付き分離、および研究Bの次の判断

- **レビュー日**：2026-10-09（JST）
- **研究**：Partially Randomized Trotter / Track B — Resource-Aware Reallocated Randomized Time Evolution（RA-RTE）
- **レビュー対象**：GitHub `HIROMU1015/Partially-Randomized-Trotter`
- **G3 branch**：`track-b-g3-finite-law-diagnostic-20261009`
- **G3報告commit**：`3b0fa70b47848e72ef9a9c9e13afc3164962f7cd`
- **G3基点G2**：`b260189b020ab7dfabb16bf424f49a6efff40d75`
- **G3 Phase A source**：`381449daefc7310e22729725f69656dd651b3d20`
- **G3 Phase B source**：`34fdf1c21becd96a77122a9c468b4191b9dbf2ad`
- **分類（GPT判断）**：`LIMITED_RESEARCH_CONTINUE / CERTIFY_RESTRICTED_B2_SEPARATION / CLOSE_MATCHED_CTS / NO_FULL_V4 / NO_BROAD_TRANSFER_YET`
- **判断担当**：GPT。次の実施担当はCodex。元のG3 mandatory STOPは解除したことにはせず、新しい限定作業範囲のみを推奨する。

## 1. エグゼクティブ判断

G3は、G2で見えた **未達成infimum** を、有限のdyadic sampling law・厳密なrational補正重み・Bernstein十分shot会計に接続した。さらに、G2で未取得だった既知identity-returnの12条件のphase-preserving合成を実施した。これにより、**「J1の資源差は無限大weightが必要なので実現できない」という当面の障害は、固定候補集合について解消した**。ただし量子測定・独立held-out・最終PR taskのコスト実測を済ませたという意味ではない。

今回のGPT再解析では、G3単独報告にない、研究上有用な**条件付きB2クラス分離**への道筋が見つかった。固定辞書、理想degree matching、同じ合成誤差上界、同じBernstein型の予算会計に限定すれば、**G3で固定された x=1/4 の *単一* J1 law のT総量と1Q総量は、G2のpure-profile完全性から導く、旧B2の任意混合・同等IS・任意precision分配に対する保守的下界の両方を下回る**。

これは従来の「既存の63 pure profile集合よりよい」より明確に強い、数学的に検査可能な主張である。その代わり、証拠範囲は必ず明記する。旧RA-D0 numerical K3、arbitrary dictionary、Pauli collection、state-dependent variance、別の信頼区間、最終Trotter-QPE性能に拡張しない。

**決定**：主線を一旦保持するが、まずG4でこの条件付き分離をCodexに独立・外向き有理算術で認証させ、次に同一toyでのmatched CTSを最小範囲で評価する。全面v4 LP・新しい分子grid・高次数への拡張には進まない。その後、GPTがprospective held-out条件を判断する。これは「技術PASSだからGO」ではなく、新しい科学的命題が有限の努力で成立・反証できるための限定的GOである。

## 2. 何が固定された証拠で、何が推論か

| 項目 | 区分 | 判定 |
|---|---|---|
| 288 profile、6,912有限law（Phase A）、18 return profile・162 law（Phase B） | G3実行記録 | 完了、各Phase1実行・retry0 |
| finite P3、同一controlled phase、保存列からのmean、60-bit dyadic q、rational weight | G3独立checkerつき実装記録 | 限定certificate PASS |
| Bernstein予測shot、T/CX/1Q費用、workspace | G3保存算術 | 条件付きで再現可能。量子shotは0 |
| J1の0.4–1.6%改善 | G3固定有限候補内の結果 | T/1Qで差あり、CXはなし |
| returnのCX最良 | G3同targetの限定実装比較 | この比較集合の範囲で成立 |
| 任意旧B2混合より x=1/4 の J1 がT/1Qで有利 | **今回GPTの新しい数学的推論＋固定桁区間照合** | 明示した理想B2 class・資源会計に限定して強い正のmargin。Codex正式独立監査は未実施 |
| 一般のPR本体の新methodが優れている／独立新規性が確定 | 未証明 | 判定不可 |

### 2.1 原資料への固定リンク

1. [G3 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)
2. [G3 Phase A result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/phase_A_result.json)
3. [G3 Phase B result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/phase_B_result.json)
4. [G3 same-pool comparison audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/saved_comparison_audit.json)
5. [G3 execution / publication audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/publication_audit.json)
6. [G2数学監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)
7. [G2診断結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json)
8. [RA-D0固定candidate table（21 columns/x）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ddf67756516e08f85fed1b987459a5e862676b7/artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json)。SHA256：`2f86169fc301ffa6b73f3cde7c8bd49939a4e7742d422c309bac181e5f09d0b6`。
9. [G2科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/research/track_b_G2_scientific_review_20261009.md)

**出典の識別**：本書の「G3結果」は保存報告に基づく。「今回の再解析」は独自の条件付き推論であり、G3の既定の科学分類に遡及して加えてはいない。

## 3. G3の固定条件と成果

- Hamiltonian / RTE構造：$R=\frac34 Q_0+\frac14 Q_1$, $Q_0=ZI$, $Q_1=V^\dagger IZV$, $V=\exp(-i\pi XX/16)$。2-qubit distinct-basis・controlled、$P_3(-ixR)$、$x\in\{1/8,1/4\}$、$\sigma=+1$、$|00\rangle$。
- 固定候補：ordinary、PTSC-K0、A、J1。7 prototype columns、3合成精度 $10^{-3},10^{-4},10^{-6}$。Phase Aは旧候補63/x・J1 81/xの **288 profiles**。J2・J3や旧B2連続混合はこのfinite-law段階に加えない。
- 法の生成：nonzero event係数 $a_i$ を理想group normの有限表現から組み立て、抽出確率 $q_i=k_i/2^{60}>0$、$\sum_iq_i=1$。重み $w_i=a_i/q_i$ は有理数。従って$\sum_iq_iw_iU_i=\sum_ia_iU_i$は算術的にexactで、理想係数からのずれは別のbias項へ入れる。実際の量子shotは0。
- Confidence：$m_2=\sum_iq_iw_i^2$, $L=\max_i|w_i|$, $s=1/200-b>0$, $\ell=\ln(10560)$、$n=\lceil\ell(2m_2/s^2+4L/(3s))\rceil$。各axisで$n\le10^9$。デジタル化L1 bias、synthesis errorと相対phaseを条件付きに取り込む。これはsufficient-shot**予測**であり、state-dependent実験分散の下界・実測保証ではない。
- 総資源の定義：$G_T=2n\,\mathbb E_q C_T$, $G_{CX}=2n\,\mathbb E_q C_{CX}$, $G_{1Q}=n(2\mathbb E_q C_{1Q}+5)$。同一lawについても各axisの実現費用を保存。workspaceは追加1 qubit。古典setup、60-bit random生成、203-bitまでの補正重み演算の実測費用は量子gate総量に未換算。
- Phase Bは既知returnの12 event／12 synthesis keysのみ、同sourceのpygridsynth 2.0.0とphase-preserving規約、既存offdiagonal O2再利用。CTS、DF、分子、他xは未評価。
- 保存済み数値は、毎条件で異なるwinnerの座標最小。すべてを単一lawが同時に達成したとは言わない。

### 3.1 有限pool内の比較結果

| x | 資源 | 既存3構成のpool最小 | J1 pool最小 | known return最小 | 判断 |
|---|---|---:|---:|---:|---|
| 1/8 | T | 179,987,284.56 | 179,160,991.73 | 187,808,511.12 | J1最良 |
| 1/8 | CX | 4,281,301.64 | 4,310,150.09 | 4,245,968.98 | return最良 |
| 1/8 | 1Q | 465,290,176.32 | 463,389,609.27 | 481,297,026.22 | J1最良 |
| 1/4 | T | 177,548,958.29 | 174,720,368.17 | 189,825,055.94 | J1最良 |
| 1/4 | CX | 4,610,549.46 | 4,723,948.73 | 4,448,986.39 | return最良 |
| 1/4 | 1Q | 468,173,605.23 | 461,712,911.04 | 495,013,282.20 | J1最良 |

同じxでもT、CX、1Qのpool最小は異なるlawで達成される。一つのJ1が返した**1Q向け最良law**について、比較は次のとおり。

| x | law | T総量 | CX総量 | 1Q総量 |
|---|---|---:|---:|---:|
| 1/8 | Aの1Q選択 | 179,987,284.56 | 4,483,213.00 | 465,290,176.32 |
| 1/8 | J1の1Q選択 | 179,183,039.36 | 4,465,884.67 | 463,389,609.27 |
| 1/4 | Aの1Q選択 | 177,548,958.29 | 5,718,904.22 | 468,173,605.23 |
| 1/4 | J1の1Q選択 | 174,820,322.72 | 5,661,819.95 | 461,712,911.04 |

このJ1 lawは**選択されたAの1Q法**には3資源で改善しているが、任意B2法を含む全3D Pareto frontierへのmembershipを意味しない。特にCX最小のreturn／PTSCと比べればJ1のCXは悪い。

## 4. 今回導出したB2理想classの保守的資源下界

### 4.1 適用クラス

ここで$\mathcal B_2^{\mathrm{ideal}}$は、元のordinary/PTSC-K0/Aの3構成の非負凸混合と、各active prototypeの保存済み3精度への非負配分を認め、元の固定degree matchingを**厳密**に満たすクラスと定義する。各prototypeの内部word conditional law、synthesized IR、cost/error bound、phase、finite P3 targetはG2/G3で固定されたもの。抽出確率は、各正係数eventについて任意の正規化されたfull-support lawを許す。新dictionary、signed cancellation、新角度、target変更、Pauli collection、operator error boundの再最適化は含めない。

$P$をこのclassの一つの有限representation、event係数を$c_i\ge0$、一回路のnative費用を$C_i\ge0$、保存synthesis errorから定めるbias上界を$b(P)$、$s(P)=\epsilon-b(P)>0$とする。抽出$q_i>0$、補正weight$c_i/q_i$、$m_2=\sum c_i^2/q_i$。G3と同じBernstein sufficient-shotルールを使うと、

$$
n\ge\ell\left(\frac{2m_2}{s(P)^2}+\frac{4L}{3s(P)}\right)
\ge\frac{2\ell m_2}{s(P)^2},\quad \ell=\ln(10560).
$$

T/CXは総量$G_C=2n\sum q_iC_i$。Cauchy–Schwarzにより

$$
\left(\sum_i\frac{c_i^2}{q_i}\right)\left(\sum_i q_i C_i\right)
\ge\left(\sum_i c_i\sqrt{C_i}\right)^2.
$$

従って（整数shot切上げ、range上乗せは下界のため落としてよい）、

$$
\boxed{G_C(P,q)\ge4\ell\left[\frac{\sum_i c_i\sqrt{C_i}}{\epsilon-b(P)}\right]^2}. \tag{1}
$$

1Qでは$G_{1Q}=n(2\sum_iq_i C_i+5)=2n\sum_iq_i(C_i+5/2)$だから、同じ議論で

$$
\boxed{G_{1Q}(P,q)\ge4\ell\left[\frac{\sum_i c_i\sqrt{C_i+5/2}}{\epsilon-b(P)}\right]^2}.\tag{2}
$$

*重要*：これは実際の測定法に対する情報理論的下界ではなく、**保存したBernstein sufficient-shot設計式で予算を決める方式に限定した下界**。別の分散上界、observable、より強いcertificationをB2だけに認めた場合に適用できない。

### 4.2 任意B2混合から63 pure profilesへの厳密な還元

G2独立数学監査は、非負price

$$
K(P)=\sum_i c_i\sqrt{C_i},\qquad b(P)=\sum_i c_i d_i
$$

と、固定dictionaryの理想係数混合について、正の$s(P)$を持つ任意の混合より悪くない**pure vertex × pure precision profile**が存在することを示した。理由は、$K$と$s$が混合についてaffineであり、非負$K$では正の$s$を持つ凸混合が全構成の$K/s$最小を下回れないためである。各groupのprecision分配は、G1のpositive-group decompositionでpure profileの凸混合へ持ち上げられる。

この補題をTについては$h_j=\mathbb E[\sqrt{C_T}]$、1Qについては**共通readout加算後**の$h_j=\mathbb E[\sqrt{C_{1Q}+5/2}]$に適用する。両者とも非負線形priceなので、B2の3頂点／active groups 2,3,3／各3精度に対応する

$$
3^2+3^3+3^3=63
$$

profileだけで、式(1)(2)の右辺のB2全体に対する最小値を取得できる。これはG3のfinite-law最適性や元RA-D0 numerical K3の「完全性」ではない。

### 4.3 保存tableからの照合結果

G2 `candidate_table_v1.json` の各groupの$sqrt(a_g^2+b_g^2)$、各event conditional label probability、native cost、`d_upper`を参照した。GPT側で63 profileについて、平方根をBigInt integer-square-rootで挟む**30桁固定小数点外向き区間演算**を行った。TについてG2の元の最小値と一致し、1Qについては新しく$C+5/2$を価格に反映した。両資源のminimizerは$x=1/4$のA、全active group $10^{-3}$だった。

まず自然対数の通常の表示値での概算：

| x=1/4 | B2理想class下界の概算 | G3の同じJ1 1Q選択law | マージン概算 |
|---|---:|---:|---:|
| T | 176,744,841.96 | 174,820,322.72 | B2下界−J1 = 1,924,519.24 |
| 1Q | 466,052,533.94 | 461,712,911.04 | B2下界−J1 = 4,339,622.90 |

さらに対数値を意図的に弱く評価した$\ell=\ln(10560)>9.2$だけを使った。63 profileを外向き区間で下から評価し、各整数部分として得た値は以下である：

| x=1/4 | **保守的B2下界** | G3の単一J1 lawの厳密保存値 | 下界側との差 |
|---|---:|---|---:|
| T | **175,508,109以上** | `6298565922079926703836975 / 36028797018963968`（約174,820,322.72） | >687,786 |
| 1Q | **462,791,435以上** | `33269921505865913553690245 / 72057594037927936`（約461,712,911.04） | >1,078,523 |

この導出が正しく実装されているなら、**同一のJ1 finite lawについて$G_T(J1)<\inf_{P\in\mathcal B_2^{ideal},q}G_T(P,q)$かつ$G_{1Q}(J1)<\inf_{P\in\mathcal B_2^{ideal},q}G_{1Q}(P,q)$が同時に成り立つ**。有限の旧B2 profile一覧だけを見た差ではない。この数式主張とrooted-costの独立監査をG4の最優先とする。

#### 認証上の留保

- これは今回GPT側の再解析・コード照合であり、リポジトリ保存の第三者独立certified proofではない。使用した固定桁の端点仕様・rounding direction・全63候補チェックをCodexが別実装で再構築し、proof artifactを保存する必要がある。
- `$\ell>9.2$`の根拠も有理Taylor enclosure等の独立証明へ落とす。浮動小数点の`Math.log`を証明書にしない。
- 数値K3のmembership tolerance、sampler rounding、coefficient L1 toleranceを許した**全ての**B2_numまで式(1)(2)の最小値が保たれることは未検証。現在のclaimは厳密理想classに限定する。転移のためにはその差分の保守的誤差限界を証明する。
- G3のJ1 lawは理想平均と一致するわけではなく、$\sum a_iU_i$のデジタル平均と理想$P_3$との差をbiasに入れた**近似的実現**である。B2だけをexact mean、J1だけを近似で比較する問題を避けるため、B2_numへ拡張する場合は同じ誤差予算を適用する必要がある。
- `native gate cost`は保存合成列に依存し、別synthesizer・compiler・hardware-native costでの優位を保証しない。
- T/1Qでの2軸分離を得たとしても、CXでは既存法がより良く、3D全体のPareto支配や最終アプリケーションの優位は出ていない。

### 4.4 $x=1/8$への過大一般化をしない

同じ弱い下界を$x=1/8$へ適用するとTで約$1.79\times10^8$、1Qで約$4.63\times10^8$程度となり、G3 J1 1Q法を下回るほどには強くない。**下界で分離できないことはB2が勝つことの証明ではない**。$x=1/8$では限定pool改善のみが現時点で確定する。

## 5. 科学的意義と、新規性・実用性の区別

### 5.1 何が新しく前進したか

G1はdegree-local係数の理想多面体と新しい極点の存在を示した。G2は理想IS価格の下でJ1が元3頂点よりわずかに有利であることを示したが、特にTではinfimumだった。G3によりfinite positive q、rational correction、range・integer shotが保存され、**少なくとも固定候補集合内で「実装可能な抽出lawが存在する」という障害は除去**された。さらに今回の再解析は、特定の予算会計で全B2理想classに対する二資源分離を狙えるという新しい論理を与えた。

この結果を**一般的なPR最適化の新理論**と呼ぶことはできない。mean保存とIS最適化の一般原理は既知で、特定synthesizerのcost notchによって一例が生じた可能性も残る。現時点の強みは、**数理的に説明可能な自由度と、強い同model内下界を持つ有限反例・分離例**を作れることにある。

### 5.2 公平な先行研究比較

- **Cugini, Atif, Subasi (2026)** は randomized quantum protocol に対するresource-optimal ISとbias不変性を一般的に扱う。$q\propto c/\sqrt C$やCS下界は既知であり、新規claimへ含めない。[arXiv:2603.13495](https://arxiv.org/abs/2603.13495)
- **Koczor, PRX Quantum 5, 040352 (2024)** は非理想量子操作のdictionaryを使うprobabilistic synthesis／凸最適化を扱う。線形制約で辞書を選ぶこと自体には新規性を置けない。[DOI](https://doi.org/10.1103/PRXQuantum.5.040352)
- **Peetz, Smart, Narang, npj Quantum Information 12, 52 (2026)** のSCU/CTSはPauli collectionを利用する、同targetの強い比較対象。論文のchannel simulationと本研究のcontrolled first operator momentを混同せず、finite P3にspecializeした実装を同cost/errorへ揃える。[DOI](https://doi.org/10.1038/s41534-025-01168-w)
- **Zhao, Yuan, Quantum 5, 534 (2021)** はHamiltonian simulationでの構造的簡約・相殺の既知原理。$\chi$依存のreturn自体は独立新methodとは主張しない。[DOI](https://doi.org/10.22331/q-2021-08-31-534)
- R0.5の独立prior-art監査は、general involution I0情報でのA構成とPauli closure I1のcollected CTSとの差、およびPTSC-K0同target比較を既に注意している。これを拡大解釈しない。[R0.5 novelty audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)

### 5.3 研究としての有効な主張と、まだ支持されない主張

**支持し得る（ただしG4独立certを要する）**：固定7-prototype、有限P3、保存誤差会計と合成費用、Bernstein sufficient-shotの下で、degree-local representation J1の一つの有限lawが、旧B2理想係数classの全混合をTと1Qの二座標で同時に上回る、条件付きの具体例がある。

**支持されない**：現実の任意B2_num／arbitrary LCUs／CTSに勝つ、全資源で最適、Trotter-QPE総資源で勝つ、DF分子へ転移、漸近優位、量子測定shotで観測した改善、世界初の独立アルゴリズムを確立した。

**論文化可能性**：一般の構成・取得可能性・強い既知法との差・独立条件への再現性が伴えば、制約付きの新method研究候補にはなる。一つの小toyと1%前後のnotchだけなら、単独の主要claimとしては弱く、R0/G1/G2の理論と合わせたmechanism noteや補足資料化を優先するべき場合がある。

## 6. 代替方針の比較と採否

| 方向 | 現在の判断 | 科学的理由 |
|---|---|---|
| A. 旧RA-D0 v4全productionと全gridを開始 | **NO-GO** | 本件ではG2 pure-profileとG3 finite witnessを使う狭い証明経路があり、数万LPに直行する情報価値が低い |
| B. $x=1/4$のJ1対全理想B2の二座標分離を独立certする | **最優先GO** | 旧B2を閉じられていない点への最小・強い反証。既存保存表だけで実施可能 |
| C. 同一toyのmatched CTS/collected comparatorを閉じる | **Bが通った場合の限定GO** | 強い既知対照後の価値を測る。I1の追加情報／取得コストは区別 |
| D. new x、new Rのprospective transfer | **準備のみGO、実行は次GPT判断** | 開発条件でのsmall gainが別合成費用でも再現するか重要。ただしB/Cの意味を先に閉じる |
| E. より一般のreturn-aware PRへ転換 | **保留** | 既知returnはCXで強いが、新規method差は未特定。無条件の主線変更は時期尚早 |
| F. Track Bを閉じてR0/G1の理論noteに縮小 | **現時点では保留** | 今回限定の分離候補に追加の情報価値がある。G4後に再判断 |

## 7. 次の担当・Codex一括作業（G4）

**科学責任者はGPT、次の実行担当はCodex。次作業の目的は既存の限定結果をpublication-grade evidenceへ近づけ、強い対照による反証機会を閉じること。** 細かな実装形式、solver選択、検査fixture・ログ形式はCodexが自律設計してよい。固定すべき科学条件だけを示す。

### Workstream A（必須）：B2 class分離の独立証明書

1. 旧保存table・結果commitを読み取り専用で固定し、今回§4の式(1)(2)、$C_{1Q}+5/2$の扱い、63 pure profile還元、$s>0$、zero-costへのCSを独立に監査する。**今回GPTで書いた結果を前提に結論を誘導せず、反例を探す。**
2. $x=1/4$、T・1Q、G3で選ばれた**同一J1 1Q用law**の厳密保存有理費用を使用し、B2側の各profileの平方根・bias・最小値をFraction／有理区間で外向き認証する。$\ln(10560)>9.2$の厳密証明を含める。すべての63 profile、採用候補、正のgap、元のG2 minimaとの一致を保存する。
3. G3 J1側のdyadic sum、weight、mean tolerance、相対phase、bias、n、joint T・1Q費用が同一lawであることをmanifestと照合する。元R1/RA-D0のmarker、source、resultは不変。
4. B2 idealとB2 numerical K3／mean toleranceの違いを明示し、どこまで下界を転移できるかを**数学的に保証できる場合にのみ**記録する。できなければ別classのまま扱う。
5. 仮定・算術境界・ref/data identityが閉じない場合は`SCIENTIFIC_WITNESS_NOT_CERTIFIED`等の記述的分類を付け、**G4以降を自動実行せずSTOP**。閉じた場合も、限定classの定理が立つだけで新methodの一般GOではない。

### Workstream B（A成功時だけ）：同一toyに対する最小matched CTS

- Pauli collectionが可能なこの2-qubit toyで、**同じfinite $P_3$のfirst operator moment**を保ち、controlled phaseを正しく数えること。元CTS論文のchannel版をそのまま別targetとして比較しない。
- CTSが利用するPauli closure／full collectionをI1アクセスと明記し、J1のgeneral-involution I0とは情報モデルと古典取得費用を分けて表示する。情報条件が異なる場合は同一コストを偽装しない。
- 同じx、精度・synthesis手続き、native T/CX/1Q、same confidence、coefficient・gate bias、state/readout、sample lawを可能な限り揃える。新たな高次文献の探索や広いdictionary最適化は行わない。
- 追加合成に必要なangleなどは結果前に固定し、既存データを可能な限り再利用する。comparisonが同じ科学targetとして成立しない場合は無理に数値を作らず、具体的阻害要因を記録する。
- CTSがJ1を支配しても、Pauli取得を要しないI0 classでの条件付き結果を否定したことにはならない。ただし一般的な「実用上最善」の主張は縮小する。

### Workstream C（実行なし）：prospective transfer preregistration準備

- Bが済んだ時点で、既存の$x=1/8,1/4$・$p=(3/4,1/4)$とは異なる**未使用のinvolution law**（例：3以上の非可換involution、異basisの制御回路）を候補として一つに絞るための*設計資料*を作る。実行してよいとは限らない。
- 将来の**primary T**を従来のprimaryと整合的に扱い、1Qをsecondary、CXを主要なtrade-off座標として、結果前に固定する。今回の1Qはposthoc development由来であり旧研究の事前登録primaryとはしない。
- J1が勝ちやすい構造をデータを見ながら選別しない。失敗でも有益な対象、baseline accessとnative費用の公平性、materialityの判断原理、gate compilerを替えたときの頑健性、計算上限を提案する。
- 新しいheld-out数値計算・分子DF計算・全v4登録LPを**このG4作業から自動実行しない**。B後の研究上の意味をGPTへ返す。

G4のA+B+Cは可能な限り一括してCodexが実施できる。Aの証明が不成立ならB/Cを無理に進めず、失敗情報と反例を返す。BでCTSの圧倒的支配など主仮説の意味を大きく変える結果が出た場合も、無条件に次段階へ進まない。

## 8. 次にGPTへ戻る条件・GO/STOP

1. **G4 Aで新しい数学的主張に反例が出た場合**：GPTに即時返し、旧B2 class・費用・誤差認証の問題を修正するか、当該claimを撤回する。B2分離を事実として継続しない。
2. **G4 AがPASSした場合**：条件付き理論的分離を研究記録へ追加できる。ただし一般実用GOではない。Bの結果と併せて研究価値を評価する。
3. **G4 BのCTSがよりよい場合**：I0/I1の情報モデル差・setup cost・controlled semanticsを吟味し、一般新method claimを縮小。必要なら別のI0 competitorに対象を移す。
4. **CTSでもJ1に条件付きfrontier余地が残る場合**：次GPTがprospective transferを許可するか判断する。固定toyでの結果の後追いではなく、独立入力・固定指標・強い同情報baselineが必要。
5. **G4で証拠を閉じても新規性・materialityの改善見込みが乏しい場合**：一連のR0最適性、G1 6頂点、G2/G3機構分離を限定理論noteとして保存し、別のPR内部研究候補へ移ることを検討する。

## 9. 研究方針の変更点と変えない点

- **G2以前**：J1は狭いfinite candidate poolで改善、full B2は未比較なので有用性未判断。よって全面v4はSTOP。
- **G3で追加された新情報**：finite dyadic J1が存在し、returnのT/CX/1Q比較も実装済み。さらに今回G2 pure-profile定理を資源下界へ適用すると、x=1/4では一つのJ1 lawが理想B2の全混合よりTと1Qで有利である、**形式化可能な新しい条件付き結果**が出た。
- **変更する判断**：強い数学的命題をG4の主検証目標として採択する。全B2最適化を待つ必要がない限定ルートを優先。
- **維持する判断**：全面v4は未承認。独立新規性未確認。無制限のsynthesis・新分子・PR task移送は未承認。一般involution classとPauli I1 classを混同しない。

## 10. 文献・研究資料

- [G3 source・結果・監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)。保存条件とprovenanceの一次根拠。
- [G2 mathematics](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)。pure-profile補題の根拠。
- Cugini, D., Atif, T. A., Subasi, Y. (2026), *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495](https://arxiv.org/abs/2603.13495)。固定samplingの費用×二次モーメントの一般結果。
- Koczor, B. (2024), *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352, [DOI:10.1103/PRXQuantum.5.040352](https://doi.org/10.1103/PRXQuantum.5.040352)。probabilistic dictionary/convex synthesisとの違いを検討する一次資料。
- Peetz, J., Smart, S. E., Narang, P. (2026), *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52, [DOI:10.1038/s41534-025-01168-w](https://doi.org/10.1038/s41534-025-01168-w)。Pauli collection/CTSとの比較。
- Zhao, Y., Yuan, X. (2021), *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534, [DOI:10.22331/q-2021-08-31-534](https://doi.org/10.22331/q-2021-08-31-534)。return関連の既知簡約・相殺。

---

**最終裁定**：`GO` for *one bounded proof-and-strong-baseline packet*、`STOP` for full-v4/old-registered LP/chemical expansion。次の担当はCodex。G4結果が新しい科学的結論を持った時点で次GPT研究レビューを行う。
