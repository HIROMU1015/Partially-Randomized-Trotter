# GPT独立科学レビュー：新しいHamiltonian simulationアルゴリズムの研究方向（D1・D2固定）

**作成日**：2026年10月16日（JST）  
**区分**：重要科学レビュー／**Level 1（高）**  
**開始承認**：ユーザーの「レビューを開始して」  
**レビュー状態**：**COMPLETED — 科学的な推奨を提示。正式採択・研究状態変更・計算許可ではない。**  
**科学計算**：**0件**。紙上の数式操作、研究資料・文献の読解、既存結果の論理的比較のみ。保存された数値の再集計、分子計算、シミュレーション、数値最適化、コンパイルは一切行っていない。  
**GitHub**：読み取りのみ。既存のsource、results、decision、authorization、STOPを変更していない。

---

## 0. 結論と研究上の意思決定

### 0.1 判定

**結論：Claudeの判断Cを「現時点で直ちに正式採択できる独立した新アルゴリズムは特定されていない」という狭い意味で支持する。一方、「D1・D2の下に新しい算法研究の有望な問題がない」「追加の文献・数学的技術監査すら行う理由がない」という広い意味では支持しない。**

特に、Claudeレビュー `レビューoutter(3).md` の次の科学的根拠には修正が必要である。

1. **OP1、体数の障壁**：入れ子交換子演算子が3・4体の項を持つことと、その**基底固有状態での期待値**が3・4体RDMを必要とすることは同値ではない。後者について二分割PFで明示的な反例がある（§4）。
2. **T-tail、Chebyshev条件**：`σ_V ≲ ε√α` は一発のQPE出力が平均近傍に収まることを保証する**十分条件**の一つであり、一般的な**必要条件**ではない（§5）。ただし、T-tailを新算法として採択しないという実務判断は維持する。
3. **混合スペクトル推定量とD1**：`⟨0|P_k V|0⟩` のような非対角の量を測るために、LCU/block-encodingが数学的に必須という主張は正しくない。controlled time evolutionと各Pauli項のHadamard test、古典フーリエ解析でアクセス可能である（§6）。重要な懸念は**情報価格**と**安定性**であり、D1上の一律な不可能性ではない。
4. **制御なし位相推定の探索の非網羅性**：Kuji等だけでなく、Clinton等の位相復元型の制御なしQPEも存在する。両方式は既知なので新規性そのものにはならないが、「参照状態とのコヒーレントな重ね合わせが作れなければ制御費用の削減の余地がない」という一般化は強すぎる（§7）。

これらは**新アルゴリズムが完成したことを意味しない**。逆に、既存PFの係数・順序・fragment名を変えるだけで新規性を主張してはならない。推奨する次段階は、**新算法のK1ではなく、対象を限定した一度限りのCodexによる「数理・一次文献・合法アクセス」の静的スクリーニング**である（§11）。この技術監査で独立した生成規則が見つからなければ、現時点の探索はPAUSE相当の提案としてユーザー判断に戻す。

### 0.2 正式に区別する判断

| 判定対象 | GPTの判断 | 確度・境界 |
|---|---|---|
| Claude C：「即採択できる独立算法なし」 | **支持** | 中〜高。現在得られた候補・証拠に限る |
| Claude C：「D1・D2内に研究機会なし」 | **支持しない** | 中。非網羅的探索、数学的な過剰な一般化がある |
| O1：Track Aと高次PFの統合比較 | **今回の新算法選定から除外** | 比較研究としての価値とは別問題 |
| O2：第1研究の論文化 | **今回の新算法選定から除外** | 独立研究としての価値は否定しない |
| T-tailを直ちに新算法採択 | **NO-GO** | 強い先行性の重複、精度・状態コピー価格の負担が未解決 |
| OP1〜OP3を単に「不可能」とする | **NO** | 演算子体数、保証、実装資源を区別すべき |
| 新科学計算・K1 | **NOT AUTHORIZED** | 今回のレビュー開始承認は計算許可でない |
| 独立した静的screen | **条件付き推奨** | Codex、ユーザーによる作業範囲の承認後、計算なし・one-shot |

---

## 1. 固定した研究課題と合法な情報モデル

**D1（変更禁止）**：block-encoding、qubitization、およびそれらを中核とするQSVT・量子walkを新しいアルゴリズムの中心手続きとして採用しない。先行研究・強い比較対象として参照することは可能。D1と無関係な通常のcontrolled time evolution、Pauli回転、fermionic Gaussian演算、Hadamard test、通常の統計的測定を自動的に禁止しない。

**D2（変更禁止）**：有限基底の元の第二量子化電子Hamiltonian `H` の**絶対基底エネルギー `E₀(H)`**を、誤差 `ε`・失敗確率 `α` の下で推定する。反応エネルギー、gap、励起スペクトルを最終タスクに置き換えない。gapやスペクトルは補助的な解析量として利用し得る。

**比較の情報契約**：厳密基底状態 `|0⟩` のコピーを利用できるという既存前提は維持。ただし、コピー・準備の費用を計上し、古典的な真の `E₀`、exact ground vector、全RDM、gap、resolventを無料の設計入力にしない。コピーを入手できることと、その未知状態を含む**コヒーレントな重ね合わせを安価に準備できること**は異なる。

**費用の公平性**：RZ数、RZ層、CX、T/Toffoli、論理qubit、状態コピー、成功確率、古典オンライン情報費用を別座標で報告。異なるデバイスモデルや誤差目標で測った倍率を合算しない。最終目的は「新しい生成規則によって、同じ絶対E₀タスクで総資源を低減すること」であり、operatorノルム、漸近係数、signal精度、証明上界の一つの改善だけを成功としない。

---

## 2. 実際に確認した資料・来歴・読解範囲

### 2.1 ユーザー提供のClaude完成資料

- `レビューoutter(3).md`：**今回レビューの主対象**。全章§0〜§12と結論・式・参照先を確認。2026-10-16 Claude完成版。最新版の判断C、OP1〜OP3、T-tail、前回O1撤回が記されている。**本資料はユーザー添付から取得**したもので、今回GPTがGitHubに新規公開されていると確認したものではない。
- `レビューoutter(2).md`：前回の研究全体レビュー。O1の統合比較とT-tail新規要素候補を提案。今回は最新版により撤回された部分を歴史的資料として扱う。

### 2.2 GitHubで実際に取得して照合した固定資料

2026-10-16、本GPTレビュー中にGitHub連携で内容を取得した。下記は**実取得済み**であり、単なるURLの存在確認ではない。ただし全資料の全行を逐語的に独立再監査したという意味ではなく、研究状態・数理式・主要claimに関連する範囲を確認した。

| 資料 | 固定commit | 科学的な用途 |
|---|---|---|
| [A3正式採用記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-16_qc_sos_pf_a3_state_adoption.md) | `dbe51f1870...` | SOS-PF現分割生成算法のCLOSEDと他研究の独立性 |
| [A3正式decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json) | 同上 | 正式状態・非許可事項 |
| [SOS-PF後GPTレビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md) | 同上 | A3の科学的根拠、二分割 `d=a/12`、A2と新算法の区別 |
| [SOS-PF後Claudeレビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_claude_scientific_review_20261016.md) | 同上 | 先行Claude判断の境界 |
| [統合研究引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md) | `78b50c515...` | CLOSED/STOP/HOLD、Track A/B等の研究契約の識別 |
| [研究目的・研究課題](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/%E7%A0%94%E7%A9%B6%E7%9B%AE%E7%9A%84%E3%83%BB%E7%A0%94%E7%A9%B6%E8%AA%B2%E9%A1%8C.md) | 同上 | 研究目的、正式な保留・終了状態 |
| [2026-10-13研究方向レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/quantum_chemistry_ground_energy_algorithm_direction_review_2026-10-13.md) | 同上 | 過去の算法探索と既存候補の失敗範囲 |
| [SOS-PF K0監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md) | `58d867270...` | 二分割係数、実装・理論の境界、`UNDETERMINED_DATA_REQUIRED` |
| [SOS-PF K0.5監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md) | `01bdc56a29...` | A2用有限step保証とKang–Suとの差 |
| [第1研究固定総括](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md) | `cd810189...` | 第1研究の独立した主要claim、追加計算不要の記録 |
| [第2研究方向freeze](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md) | `1d01e772...` | 第2研究Direction C |
| [Track A H6独立結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md) | `0dd5c3d...` | signal task上の限定的PR優位の範囲 |
| [Track B D1-D3監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md) | `d3a232a...` | 既報・費用・保証と新算法要件の境界 |

過去のグローバルmanifest既知FAIL、別系統の研究状態、K1-F未認可を変更しない。Claude最新版そのもののremote公開を本レビューの必要条件とはしない（ユーザーが原文を直接添付したため）。後でGitHub研究記録として保存することは、Codexに対する**別途の限定作業**である。

### 2.3 外部文献の確認範囲

一次文献の出版社・arXivページにより題名・著者・要旨を確認。**Hejazi等の論文とCasares等SPRINT原稿、THRIFTの本文の関連節を読解**。それ以外は主に要旨／公開概要までであり、未読定理の詳細を断定しない。文献は末尾§13にまとめる。

---

## 3. Claudeレビューで支持できる内容

1. **研究目的への再整合**：O1は手法比較、O2は既存研究の論文化であって、独立した新アルゴリズムを作ることの代わりではない。前回O1の「第一推奨」を取り消した判断を支持。
2. **T-tailの慎重な扱い**：厳密基底状態上の平均・分散を使うこと自体、Kivlichan–Granade–Wiebeのrandomized Hamiltonian phase estimationと強く重なる。Hamiltonian項を除く／軽くする機構の新規性は低い。ただし**方式が厳密に同一と証明したわけではない**。決定論的除去＋測定補正という差分の重要性は、回路費用に実質的な改善がなければ論文の中心claimにならない。
3. **直接測定を正面から比較すべき**：厳密基底状態がアクセス可能な情報モデルでは、`⟨H⟩_0=E₀` の直接測定は合法な競合手段である。ただし高精度で必ずQPEを上回るという話ではない（§8）。
4. **既存の失敗は繰り返さない**：PF小改変、H-E1、SOS-PF現探索、旧N1/THRIFT/C1などは、それぞれ正式なCLOSED/STOP等の状態を維持する。単純な名称変更で再始動しない。
5. **新算法としての着地点を要求**：具体的な回路/生成手順、合法な入力、保証する絶対E₀タスク、改良する資源座標、最強の既知baseline、比較のためのデータ取得費用が同時に書けないなら、候補を「採択」しない。

---

## 4. 重大な修正I：入れ子交換子の演算子体数とground期待値を分離する

### 4.1 Claude OP1で正しい部分

一般の電子Hamiltonianで`k`体と`l`体の正規順序化された粒子数保存演算子の交換子は高々`k+l−1`体。したがって3個の2体DF fragmentを含む二重交換子に4体成分が生じ得る。これは、**任意の多fragment PFで高次相関を安く完全評価するとは限らない**という実際の障壁を示す。

### 4.2 過剰な一般化への紙上反例：一体＋二体の二分割

有限基底の `H=A+B`、`A`を一体演算子、`B`を二体演算子とする。`|0⟩` は `H|0⟩=E₀|0⟩` を満たす。対称二次PFを

\[
S_A(h)=e^{-ihA/2}e^{-ihB}e^{-ihA/2}
\]

とする。BCHで、`S_A(h)=e^{-ih(H+h²D+O(h⁴))}`、

\[
D=\frac1{24}[A,[A,B]]+\frac1{12}[B,[A,B]]
\]

を得る。`C=[A,B]` とすればground固有状態について `⟨[H,C]⟩_0=0` なので、

\[
\langle [B,C]\rangle_0=-\langle[A,C]\rangle_0,
\qquad
\boxed{d=\langle D\rangle_0=-\frac1{24}\langle[A,[A,B]]\rangle_0}. 
\]

また`[A,[A,H]]=A²H−2AHA+HA²`から

\[
\langle[A,[A,H]]\rangle_0=-2\langle\delta A(H-E_0)\delta A\rangle_0,
\]

従って

\[
\boxed{d=\frac1{12}\langle\delta A(H-E_0)\delta A\rangle_0\ge0}.
\]

**重要**：`[A,B]` と `[A,[A,B]]` はいずれも高々二体演算子である。従ってこの`d`は1体・2体RDMで完全に決まる。`D`のもう一方の項`[B,[A,B]]`に一般に3体項があっても、**ground期待値では二体演算子の期待値へ正確に還元される**。

この結果は、Claude自身が引用したK0の`d=a/12`と整合する。しかし、Claude§8.1の「fragmentに二体を含む限りbiasには三体以上が必須」「二分割でも三体相関に依存する」という論理は成り立たない。

**重要な制限**：`B`を全相互作用の和にすると`e^{-ihB}`が安く実装できるとは限らない。上記はまず数学的な**必要情報に関する反例**であり、実装可能な新分割算法の存在証明ではない。多数の非可換DF fragmentを使うPFでは高次相関が残り得る。また`d`の正確な期待値を既知基底状態のコピーから求めるのと、**状態情報なしに認証付き上界を出す**ことは別問題である。

### 4.3 研究に残る本質的な問い

新アルゴリズムの可能性は「Dの形式体数」ではなく、**元のHamiltonianのground期待値における`[H,X]`の消去関係を利用し、誤差係数の測定費用とPF回路単価を同時に減らせる、実装可能なfragment構成があるか**という点に限って残る。ただしこれはHejazi等のenergy-specific PF設計、既存H-E1、SOS-PFの方向と重なり、新規性が確認されるまで研究候補として採択しない。

---

## 5. 重大な修正II：T-tailの集中条件は十分条件であって必要条件ではない

`H=H_c+V`、`|0⟩`をHの厳密ground状態とすると、Claudeの恒等式

\[
 m=\langle H_c\rangle_0=E_0-\langle V\rangle_0,
 \quad \mathrm{Var}_0(H_c)=\mathrm{Var}_0(V)=\sigma_V^2
\]

は正しい。`H_c`について1回の理想QPEをすると、固有値の標本`E`が出る。Chebyshevから

\[
\Pr(|E-m|>\varepsilon_s)\le\sigma_V^2/\varepsilon_s^2
\]

なので、`σ_V≤ε_s√α`は失敗確率を`α`以下にする**十分条件**である。しかし一般の必要条件ではない。

反例として、`H_c`のspectral measureが

\[
\mu=(1-\delta)\delta_m+\frac\delta2\delta_{m+L}
            +\frac\delta2\delta_{m-L}, \quad 0<\delta<\alpha,
\]

である場合を考える。平均は`m`、分散は`σ_V²=δL²`であり、`L`を大きくすれば`σ_V`はどれほど大きくもできる。一方、`L>ε_s`のとき、1回の標本が`m`から`ε_s`以内に入る確率は`1−δ>1−α`である。**大きな分散が小さな高失敗確率を必然的に意味するわけではない。** これは数値計算ではなく、確率分布の紙上構成である。

**研究上の帰結**：Claude§7.2の「`σ_V≤ε√α`が必要」という文言は「Chebyshevに基づく分布非依存の十分条件」に置き換えるべき。ただし、未知のスペクトル分布の高集中性を低費用で**認証**する一般手順は示されていない。単一サンプルは通常平均を返さず、spectral widthの広がりによってHeisenberg型の精度が崩れ得る。Kivlichan等との強い先行重複も残るので、**T-tailの新算法としての採択は支持しない**。

---

## 6. 重大な修正III：混合スペクトル推定量はD1だけを理由に排除できない

Claude§6(i)は、`H_c`の射影`P_k`を用いて

\[
 E_0=E_k^c+\frac{\langle0|P_kV|0\rangle}
                         {\langle0|P_k|0\rangle}
 \quad(p_k=\langle P_k\rangle_0>0)
\]

という正しい恒等式を示したうえで、非unitaryな`V`を測るためLCU/block-encodingが必要として退けている。しかし**その必要性は示されていない**。

`V=\sum_j v_j Q_j`をPauli基底（各`Q_j`がunitary）に分解する。理想の時間発展`U_c(t)=e^{-itH_c}`に対して

\[
 C(t)=\langle0|U_c(t)V|0\rangle
       =\sum_jv_j\langle0|U_c(t)Q_j|0\rangle
       =\sum_k e^{-itE_k^c}\langle0|P_kV|0\rangle,
\]

\[
 P(t)=\langle0|U_c(t)|0\rangle
     =\sum_ke^{-itE_k^c}\langle0|P_k|0\rangle.
\]

各相関`⟨U_c(t)Q_j⟩`は、**controlled`U_c(t)Q_j`に対する通常のHadamard test**で複素成分を測定できる。時間信号`C(t),P(t)`から古典的なスペクトル分離で`⟨P_kV⟩`と`p_k`を取り出せる。**アルゴリズムの中核にblock-encodingを置く必要はない**。D1下でも「定義上実装不能」ではない。

しかし、有利とは限らない。`|v_j|`の係数和による測定ノイズ、分解項数、有限時刻でのスペクトル分離、`p_k`が小さい場合の不安定性、`H_c`用のcontrolled発展、`H_c`自身のPF bias、QPEと直接測定の強い対照が残る。**一般にはε依存が測定由来の`ε^{-2}`になる可能性が高い**。この案は、現時点では「ClaudeのD1不可能性判断の修正」であって、原理的新算法・改善保証・新規性の認定ではない。

---

## 7. 誤差保証、制御の情報モデル、既存文献による探索範囲の修正

### 7.1 二次PFの非負ground biasは全PFのno-goではない

二分割の標準対称二次PFで`d≥0`なのは正しい。一定の順序平均でも正の主要biasが残り得る。しかし、**設計された新しい積公式一般で固有位相の低次誤差を相殺できない**という結論には拡張できない。Hejazi–Soni–Shokrian Zini–Arrazolaは、QPEの固有値誤差を専用の目的関数にして、通常の同次数PFより改善する9項の2次PFを明示している（arXiv:2412.16811、本文§II）。これは「既知の強い対照」であり、過去の閉じた係数改変を再開する理由ではない。

### 7.2 `S(0)=I`の縮退と基底位相の解析性

Claude OP3の直観「基底固有位相の解析性は一般的norm boundより広い可能性がある」は合理的だが、数学上は`S_2(0)=I`で**すべての固有値が重複**するので、`S_2(z)`の単純固有値の通常の摂動論を`z=0`へそのまま当てることはできない。局所的なprincipal logarithmが定義できる円板で

\[
H_{\mathrm{eff}}(z)=\frac{i}{z}\log S_2(z),\quad H_{\mathrm{eff}}(0)=H
\]

と置き、`z=0`の可除特異性を処理したうえで、元の`H`の対象sectorで孤立したground固有値に解析的摂動論を適用するのが適切である。**局所解析性の存在**と、実用的な刻み幅を含む**数値的・認証可能な大きな領域**の証明は別問題。A2は`NOT_ADOPTED`のままとする。

### 7.3 制御なしQPEについての既存研究

Kuji等の2025年論文は、adiabatic state preparationの対称性を使って基底状態と既知エネルギー参照状態の重ね合わせを作る方式である。これとは異なるClinton等の2026年PRX Quantum論文（arXiv:2410.21517）は、**入力信号を増やし、位相復元を利用してcontrolled evolutionを回避する既知方式**を提案している。従って「制御回避は参照状態との単純な重ね合わせしか道がない」という視野は不十分。ただし、論文は**既知**であり、実行に要求する複数の入力信号・基準の調達、絶対エネルギーの原点の確定、shot費用の条件がD1・D2の固定契約と一致するかは、要旨だけから決められない。

`H→H+cI`で**無条件の`e^{-iHt}`だけ**を使う量子実験では、各回の全体系にかかる`e^{-ict}`は全体位相となり、測定統計から定数シフト`c`を識別できない。元の`H`の係数や基準エネルギーが古典的に既知なら別の情報が追加されている。絶対`E₀`推定では、単なるスペクトル差の推定と原点の校正とを分けて説明する必要がある。これは制御なし方式の一般的不可能性を主張するものではない。

### 7.4 先行研究の強さ

- **SPRINT/GRADE**（Casares等、2026）：near-integrable PF、randomization、processing、対称性保護、QROM・tail処理を統合している。DFや前処理の小改変を新機構とする際の強い先行性チェックが不可欠。ただし論文の主数値例はX-ray吸収であり、**今回の絶対E₀**での優位を直接実証したわけではない。
- **THRIFT**（Bosse等、2025）：時間依存のinteraction pictureを使う、弱相互作用系における改良PF。いわゆるGaussian回転枠を単に導入しただけなら既知原理に属する。過去の研究内THRIFT CLOSEDを再開しない。
- **CFQM**（Moreno Casares等、2024）：時間依存の非可換発展をcommutator-free指数積で扱う一般枠組み。化学DFへの移植だけで独立算法の新規性が生じるとは限らない。
- **corrected PF**（Bagherimehrab等、PRX Quantum 2026）：効率的なcorrector設計は既知。特にcorrected PFを固有値誤差で比較する問題も文献中に示されているため、「固有位相で最適化する」という言い換えだけでは新規性はない。

以上は外部一次文献で確認できた「既報の範囲」であり、最新の全先行研究についての網羅的な新規性審査ではない。

---

## 8. 直接測定という競合手段の正しい位置づけ

Claudeのgrouped direct measurementについて、各群の測定分散を`σ_γ²`、必要な独立shot数を`N_γ`とすると、独立測定で`Var(\hat E)=\sum_γσ_γ²/N_γ`。全shotが同じ価格なら、誤差目標`ε`の最適shot配分は

\[
N_{\rm DM}\simeq\frac{z^2}{\varepsilon^2}\Big(\sum_\gamma\sigma_\gamma\Big)^2.
\]

しかしgroupごとに状態準備・基底回転などの**1 shot費用`c_γ`**が違う場合、実行資源はLagrange最適化により

\[
\boxed{C_{\rm DM}^{\rm opt}\simeq
\frac{z^2}{\varepsilon^2}
\Big(\sum_\gamma\sigma_\gamma\sqrt{c_\gamma}\Big)^2}
\]

となる（Gaussian近似の信頼係数`z`；厳密な失敗確率保証にはtail boundなどを別途設計する）。`c_γ`には状態準備も含める。

また、もし`H=cI+\sum_γG_γ`で`G_γ\succeq0`かつ`\|G_γ\|\le f_γ`と書ければ、`\mathrm{Var}(G_γ)\le f_γ\langle G_γ\rangle`なので、`Λ=\sum f_γ`、`E_g=E₀-c≥0`に対して`\sumσ_γ\le\sqrt{ΛE_g}`。ClaudeのPSD上界は数学的には成立するが、**shift `c`と分解に依存し、一般に鋭いとは限らない**。

一方、条件を満たす量子位相推定は総コヒーレント発展時間に概ね`O(ε^{-1})`のHeisenberg型依存を持つ一方、PF step誤差・合成費用・state preparation価格が必要となる。したがって、**direct measurementの存在は時間発展を使う算法を不要にするno-goではない**。両者を同一のコピー価格、同一精度・失敗確率、同一native costで比べることが必要である。ここではその比較数値を生成していない。

---

## 9. D1・D2内で残る科学的可能性：具体的な研究候補と未解決問題の区別

以下は**仮説・評価の対象**であり、採択済み新アルゴリズムではない。

| 方向 | 合法な候補の型 | 既報との差を示すための最低条件 | 科学的な最大リスク | 現判定 |
|---|---|---|---|---|
| **H1：固有値誤差の低体数化とPF回路の同時設計** | `⟨[H,X]⟩_0=0`の関係を利用して、groundで重要な誤差演算子を安く測れる低体数の量へ縮約できる**native-fast-forwardableな分割と積公式**を生成 | 単なるHejazi型energy-specific PF、既存H-E1、SOS-PFの係数変更ではない、新しい**入力→出力回路生成規則**。2RDM化だけでなく`C_step/h_max`または情報込み総費用の優位 | 全interaction `B`のexpが安価でない、補正情報が高価、既報と重複 | **未構成・静的screen対象** |
| **H2：混合スペクトル相関による低費用な絶対E₀推定** | `C(t)=⟨U_c(t)V⟩`と`P(t)=⟨U_c(t)⟩`によるcoherent/measurementハイブリッド | D1非抵触で、`p_k`分母・spectral filter・測定費用込みで**全H QPEと直接測定の双方に対する優位**。強い先行性確認 | `ε^{-2}`shot価格、small overlap、既報の相関・推定法へ還元 | **導出は成立・算法利益未確認** |
| **H3：Gaussian枠での時間依存DF primitive** | `H=h+V`のone-body回転枠で`V_I(t)=e^{iht}Ve^{-iht}`をexactに処理し、特殊な積分構造を使う | THRIFT・CFQM・SPRINTより安価な具体的**native指数化回路**およびstep誤差と制御付き費用の証明 | frame変換・再対角化にGivens回路がかかり、通常PFより高価 | **条件付き調査のみ。既存THRIFT CLOSED維持** |
| **H4：制御なし絶対E₀推定の合法access** | multi-signal phase retrievalを化学の絶対エネルギーと接続 | Clinton/Kujiにない情報取得とゲート費用低減を与え、古典的原点・reference・入力状態価格を明示 | 既知法の単なる適用、絶対位相の識別不能、コピー数増大 | **先行調査対象、候補採択なし** |

**優先度の決め方**：H1は今回特定したClaudeの数学的論理の穴から直接生じるため、最も限定的な*静的数学・先行研究の照合*に値する。ただし既存H-E1のCLOSEDと重なりやすいため、**まず差分がないことの検証**が中心となる。H2は数学的にD1外ではないという修正が重要だが、現時点で費用の目途は弱い。H3・H4は既存文献が強く、網羅的に探索を拡大する理由は弱い。

### 9.1 どのようなときだけ新算法の研究候補として認めるか

1. **生成規則**：入力電子積分・軌道・DF fragment・既知の測定情報から、出力PF/時間発展/推定器回路が一意または手順に従って生成される。
2. **合法情報**：exact E₀、exact gap、RDM、フル固有vectorを無料入力にしない。状態コピー費用を含む。
3. **中心claim**：`Pr(|\hat E-E₀|≤ε)≥1−α`（または定義した経験的評価）の意味を確定。biasとoperator誤差を混同しない。
4. **改善のレバー**：より小さい`C_step`、大きい`h_max`、少ないコピー、より軽いcontrol/synthesis等、**強い既報**と区別されるもの。単なる可測性は不足。
5. **科学的新規性**：Hejazi、CPF、Kivlichan、THRIFT、SPRINT、CFQM、既存研究の停止候補への厳密なreduction testを通過。
6. **対照の公平性**：同じ絶対E₀タスク・成功確率・情報accessで、既存高次PFと適切な測定・乱択手法等を比較する。ただしその比較だけを新アルゴリズム研究の中心成果としない。

少なくとも**具体的な生成規則と回路費用の整合した優位見込み**が得られない限り、K1の科学計算へ進まない。

---

## 10. 研究の方向性と論文化の着地点

現時点で論文化できる独立した新アルゴリズムは特定できていない。これを明示的な**否定的・保留の科学判断**として記録する。

- **H1が将来成立する場合のみ**：例として「元の絶対E₀推定に対して、ground状態の交換子同値関係を用いたDF-native積公式生成により、有限step誤差と情報取得を会計した総資源を強いbaselineより減らす」が中心claimとなり得る。今のところ定理も回路も存在せず、論文主張にしてはならない。
- **H2が成立する場合のみ**：既報と異なる混合スペクトル推定手続きが、分子の具体条件で時間発展QPEとdirect measurementを上回る条件付き結果。現時点では基本恒等式以外の優位は立証されていない。
- **静的screenで上記の生成規則が成立しない場合**：新算法シリーズを無理に続けず、今の探索を**一時停止することを推奨**。ただしTrack A/Bや第1・第2研究を新算法の代わりに自動採択せず、それらは別の科学的・運営上の選択として扱う。

**古い研究状態は正式変更しない**。A3 `CLOSED`、A2 `NOT_ADOPTED`、旧N1/THRIFT/第一候補等のSTOP・CLOSED・PAUSED・HOLDはユーザー承認なしに解除しない。

---

## 11. 次の担当、推奨するCodex作業、Mandatory STOP

### 11.1 推奨作業（新規実験ではない）：一度限りの技術的screen

**次の担当：ユーザー（実施可否の承認）→Codex（資料・数学の静的調査）**。GPTによる本Level 1判断は完了。Codexは今回の科学的結論の採択を代行しない。

Codexへの作業範囲（別途ユーザー承認があったときのみ）：

1. **OP1修正の独立紙上検算**：`H=A(one-body)+B(two-body)`の二分割S2で`d=-⟨[A,[A,B]]⟩/24`と`d=a/12`を確認。多fragmentで2RDMに落ちる条件を**式と合法なfragment指数化コスト**で分類。シミュレーション・数値行列による検証は実行しない。
2. **H1のprior-art差分検証**：Hejazi 2412.16811、CPF 2026、既存H-E1、SOS-PF、SPRINTの固定資料を照合し、すべて既知に還元する場合は**NO_GO_NO_INDEPENDENT_GENERATION_RULE**と記録する。探索中に新しい候補を採択しない。
3. **H2の合法accessと価格の式**：Pauli分解＋Hadamard testで`⟨U_cV⟩`がblock-encodingなしに得られること、`p_k`・測定分散・filter時間のoverhead、直接測定・全H QPEへの比較が設計できるかを文献・紙上で記録。専用の数値計算は禁止。
4. **H3/H4の最小先行性確認**：THRIFT、CFQM、SPRINTおよびClinton/Kujiを短い対応表で確認する。先行との差が書けなければ深掘りを止める。無制限の候補生成をしない。
5. **成果と保全**：参照可能な資料索引、論理の成立/不成立/未証明、実装可能性、未取得情報、STOPを書いた1つの短いMarkdownを新branchに保存。**commit/pushはこの限定技術記録にのみ、ユーザーがその公開を含め明示承認した場合に許可**。source/results/旧監査/authorization/global manifestに触れず、remote SHAとURLを報告する。

### 11.2 停止条件

- 数理式の訂正が既に以前のK0に含まれ、**新たな生成原理・回路優位を生み出さない** → その枝を直ちに終了。
- 最も近いprior artが同じ入力→出力生成規則を与えている → その枝を終了。
- 有望に見える案に**未承認の数値実験・保存値再集計**が必要 → 事実・理由・必要な計算の案だけ記載してSTOP。
- 既存CLOSED/STOP系列を実質的に再開する必要 → 再開を実行せずユーザーとGPTの科学判断へ戻す。
- 明確な合法accessまたはコスト会計が閉じない → 「未確定」記録でSTOP。**未確定をGOと表現しない**。
- 終了後はMandatory STOP。K1、コード実装、科学計算、formal研究採択は自動開始しない。

### 11.3 GPTに再度戻る条件

新しい**具体的な回路生成規則**、強い文献との独立差分、理論上の資源改善の式がすべて確認できた場合。または数学的な重要修正が既存研究全体の主要claimを変更する場合。そのときに**新たなLevel 1レビューの必要性を判定し、明示承認を得る**。単なる調査完了・commitだけを理由に再レビューしない。

---

## 12. 反論・不確かさ・科学的GO/STOPの位置づけ

**反論A：「Claudeの候補なしは正しく、数学の修正は実装に影響しない」**：十分あり得る。実際、二分割`B`の高速指数化ができない可能性、measurement overhead、既報性によりH1/H2は全滅し得る。したがって正式算法採択はせず、**極めて小さな静的screen**だけを提案した。

**反論B：「既存の第1・第2研究／Track A/Bを先に進めるべきだ」**：プロジェクト全体の成果最大化では合理的な場合がある。しかし今回は独立した新アルゴリズムの**科学的探索の可否**が対象であり、既存研究の論文化・比較をその答えと混同しない。並行作業の価値は別の決定事項。

**反論C：「2RDMで主要係数を表せるならcertificateも得られる」**：誤り。RDMの推定費用、入力状態依存性、統計誤差、有限h剰余の保証、branch・phase overlap、`H_c`の指数化費用が別に残る。低体数化は**一つの障壁の除去**であり、完成した保証ではない。

**反論D：「T-tailの分散条件が必要でないなら強い新算法になる」**：誤り。例外的な高集中spectral measureを事前に保証する低費用手順がない。先行性とコピー会計の問題はそのままである。

**反論E：「混合スペクトル相関は新規算法である」**：認めない。通常のHadamard testとフーリエ分解による数学的な実装可能性を示しただけ。大きいノイズ・filter overheadを上回る独立機構は未発見。

**今回の科学判断の確度**：二分割2RDM反例＝高、Chebyshev必要条件の修正＝高、Hadamard相関のD1適合＝高（一般費用優位は不明）、T-tail新規性の低さ＝中〜高（原論文の重複範囲の詳細照合は未実施）、**新算法即採択なし**＝中〜高、**1回の静的screenの情報価値**＝中。

---

## 13. 一次文献・今後の参照先

1. **Kivlichan, Granade, Wiebe**, *Phase estimation with randomized Hamiltonians* (2019), arXiv:1907.10070. [原文／要旨](https://arxiv.org/abs/1907.10070)。本レビューでは要旨と公開本文冒頭を確認。importance samplingと変更Hamiltonianによる位相推定の既報。
2. **Hejazi, Soni, Shokrian Zini, Arrazola**, *Better product formulas for quantum phase estimation* (2024/2025), arXiv:2412.16811. [原文](https://arxiv.org/abs/2412.16811)。**本文§I–II**で固有値誤差を対象とする新しい二次PFを確認。
3. **Kohda et al.**, *Quantum expectation-value estimation by computational basis sampling*, Phys. Rev. Research **4**, 033173 (2022), arXiv:2112.07416. [要旨](https://arxiv.org/abs/2112.07416)。直接測定の強い競合手法。
4. **Kuji et al.**, *Robust phase estimation of the ground-state energy without controlled time evolution on a quantum device*, Phys. Rev. A **111**, 042618 (2025), [出版社概要](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.042618)。adiabatic state preparationを利用する参照状態重ね合わせ方式。
5. **Clinton et al.**, *Quantum Phase Estimation Without Controlled Unitaries*, PRX Quantum **7**, 010345 (2026), [出版社概要](https://journals.aps.org/prxquantum/abstract/10.1103/7qcr-znl2)／[arXiv](https://arxiv.org/abs/2410.21517)。複数信号の位相復元方式。**元の黒箱exact ground stateコピー契約への適合は未審査**。
6. **Casares et al.**, *Theory and practice of Trotter product formulas for quantum chemistry* (2026), arXiv:2606.30741. [原稿](https://arxiv.org/abs/2606.30741)。SPRINT/GRADE、near-integrability、randomization、processing、QROMの強い既報。**本文の紹介節と構成節を確認**。
7. **Bosse et al.**, *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nat. Commun. **16**, 2673 (2025), [出版社全文](https://www.nature.com/articles/s41467-025-57580-5)。THRIFTのinteraction-picture積公式。
8. **Moreno Casares, Shokrian Zini, Arrazola**, *Quantum simulation of time-dependent Hamiltonians via commutator-free quasi-Magnus operators*, Quantum **8**, 1567 (2024), [出版社概要](https://quantum-journal.org/papers/q-2024-12-17-1567/)。CFQMの既報。
9. **Bagherimehrab et al.**, *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, PRX Quantum **7**, 033018 (2026), [出版社概要](https://journals.aps.org/prxquantum/abstract/10.1103/phk5-slkl)。corrected PFと固有値評価の公開展望。本文の詳細証明は未精読。
10. **Günther et al.**, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026), [出版社概要](https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq)。既報のPR-QPEを比較に置く必要があるが、今回の新算法にはしない。
11. **Abe et al.**, *Evaluating higher-order product formulae for molecular ground-state energy estimation* (2026), [arXiv:2605.30967](https://arxiv.org/abs/2605.30967)。対象プロジェクトの決定論高次PF比較。元の論文・第一研究・Track Aは**同一結果として合算していない**。

---

## 14. 実施・承認・保存状態の最終確認

- **レビュー実施：COMPLETE、Level 1**。Claudeを機械的に追認せず、数学的な反例・理論的制限・文献の新規性・研究着地点を独立に評価した。
- **D1・D2：変更なし**。block-encoding/qubitization中心の新算法を提案しない。最終タスクは元の絶対`E₀`。
- **新規科学計算：0**。本書の数式は紙上の推論・初等代数であり、新しい分子数値・回路費用・benchmarkは生成していない。
- **既存SOURCE/成果・状態：不変**。GitHub書込みなし。A3／A2／旧系列の状態変更なし。
- **Codex静的screen：推奨のみ、未承認・未実行**。文書保存・branch・commit/pushもユーザーの別の明示指示が必要。
- **Mandatory STOP継続**。新算法・新計算・新しい正式研究方向は自動開始しない。
