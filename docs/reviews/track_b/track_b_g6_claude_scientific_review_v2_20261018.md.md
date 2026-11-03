# Track B 局所Return Aggregation生成＋有限bit構成：独立新規性・研究価値・独立論文化可能性に関する重要科学レビュー（完成版）

| 項目 | 内容 |
|---|---|
| レビューID | TB-RA-FB-REVIEW-20261018-CLAUDE |
| 作成 | Claude（Anthropic）による独立レビュー。2026-10-18 JST（開始依頼、「続けて」、自己点検依頼の3セッション） |
| 版 | v2（v1の自己点検による改訂。結論・確信度・推奨は不変。改訂内容は末尾の改訂履歴） |
| 開始承認 | ユーザー依頼文（`ssi.md`）による明示承認。継続は同一scope内 |
| 種別 | 研究上の推奨。研究方針の正式採択・論文投稿GO・追加科学計算の承認を代行しない |
| 本レビューで行ったこと | 固定commit資料と一次論文の取得・読解、SOURCEの静的読解、既存数学の紙上検証、紙上の漸近解析と論理的反例の検討、科学的解釈 |
| 本レビューで行っていないこと | 新規科学計算、保存値の数値再解析・再ランキング、最適化、SOURCEのimport・実行・修正、compile・合成・sampling simulation、契約・authorization・grant・marker・claimの発行、GitHub上の変更（いずれも0件） |
| 既存状態 | Track B D＋E（10/15採択）、現行探索 mandatory STOP、研究④の新テーマ選定保留、研究①②③：すべて不変。本レビューはSTOP解除・実験再開を意味しない |

## 根拠ラベル

| ラベル | 意味 |
|---|---|
| 〔固定〕 | 固定commitの資料で直接確認した事実 |
| 〔SOURCE〕 | SOURCEの静的読解で確認したこと（import・実行なし） |
| 〔数学検証〕 | 既存の数学をClaudeが紙上で独立に検証したこと |
| 〔保存結果〕 | 既存の科学計算・保存結果の転記（表示値のみ。再計算・再順位付けなし） |
| 〔紙上推論〕 | Claude自身の新しい紙上推論（解析的展開・論証。数値計算ではない） |
| 〔仮説〕 | 未検証の仮説 |
| 〔文献推測〕 | 先行研究との比較から推測されること |
| 〔推奨〕 | 研究方針に関する推奨 |

---

## 0. 最終的な科学的判断（冒頭回答）

**問い：「Track Bの局所Return aggregation生成＋有限bit構成は、現存する独立した査読付き論文の中心的貢献として成立するか。」**

**判断：C — 技術ノート・補助成果としての価値を残し、Track B単独の査読付き論文化は当面見送る。**

| 選択肢 | 判断 | 確信度 |
|---|---|---|
| A：独立アルゴリズム論文として成立する具体的貢献が確認できる | **不成立** | 高（約90%） |
| B：範囲を限定した構成・方法論論文の可能性 | **現存構成を中心に据える形では推奨しない** | 中〜高（約75%） |
| C：技術ノート・補助成果として保持し、単独論文化は当面見送る | **推奨** | 中〜高（約75%） |
| D：資料・理論上の不足により判断保留 | 不要。判断を左右する構造は既存資料と紙上検証で確定でき、残る不確実性はCを覆す方向よりも補強する方向にある | — |

### 0.1 最も強い根拠

1. **〔紙上推論〕Track B固有の要素は漸近的な資源価値を持たない。** PR／統計的位相推定で費用を支配する多step・短step領域（1stepのパラメータ $x\to0$、$c=rx^2$ 固定）では、r step積推定量の二次momentの対数は、ordinary（Wan–Berta–Campbell型）で $2c$、normalizerを既知とするreturn-aware正準sampler（partial return、closed P3＋tail、closed P5、normalizer既知のfull）で $2(1-\chi)c$、G6のordinary-envelope zero-fill生成で $(2-\chi)c$ となる（§7.1 命題R）。$\chi=\sum_ip_i^2$ は乱択部の衝突確率である。partialとfullの係数質量差は1stepあたり $O(\chi x^4)$ で、r step合計では $O(\chi c\,x^2)\to0$。さらにfull ensembleを重要度重みで抽出する任意のsamplerの1step二次momentは $B_{\rm full}^2$ 以上なので、envelopeやproposalを改良しても漸近的に意味のある差でpartialを上回れない。**一般次数の局所Green query＋normalizer不要のzero-fill生成という独自部分は、より単純な二次return吸収に漸近的に支配される。**
2. **〔数学検証〕〔文献推測〕数学要素はすべて既知か初等的である。** 形式母関数はAomoto–Kato（1988）の自由積Green関数の $\mathbb Z_2$ 特殊化、Euler pairingはWan–Berta–Campbell／PTSC、$Q^2=I$ による高次項の吸収はZhao–Yuan §4.2、Pauli収集と層samplingはCTS、zero-fillと費用×momentの関係はCugini–Atif–Subaşı、辞書分解と重みの交換はSPSにある。Track B固有の補題（挿入上界、$0<x\le1$ での正値性、受理率 $\ge1/(2\sqrt2)$、有限bitのbias・range上界、P5閉形式）は正しいが初等的である。有限bit構成は重要度重みによる実proposal補正と誤差会計の標準的実装で、独立した理論貢献ではない。
3. **〔保存結果〕保存証拠も「return-aware方式全般 vs ordinary」の差を示すにとどまり、一般構成固有の利益を示さない。** G10 v3の同次数比較では、一般full returnは各次数のclosed対照に勝たない（T切片 +0.19%／+0.64%／+2.91%）。既存S4レビュー§5.6でも、closed P5のordinaryに対する利益（h=0のT −26.6%）の大半はpartial returnで得られ、P5との差は1.16%にとどまる。いずれも $\chi=19/50$ の3-system-qubit synthetic、$x=5/7$、r=1–2の固定条件である。

### 0.2 最も重要な反論

- **有限x・少step（early fault tolerance等）は漸近論の外にある。** $x\le1$ の大きいstepでは $O(x^4)$ 項や角度依存の合成価格が効き、一般fullが有用な入力があり得る。→ 応答：差分項はいずれも少なくとも一回の隣接衝突を要し $\chi$ に比例する。局所queryが不可欠になるのはLが大きく全reduced語の列挙やclosed形式が困難な場合だが、そのとき $\chi\ge1/L$ は小さくなり得る。逆にχが大きい少数項の場合は列挙・closed形式（P5のL²+1群）で足りる（§7.3の「挟み撃ち」）。ただしこの構造は定性的で、「少数の大項を乱択部に残す」設計では成立しない。PRでは通常大項は決定論側へ移るが、Track Bの実入力では確認されていない。
- **一般involution（非Pauli）への適用性は独自である。** → 応答：PR原論文の乱択部はPauli表現である（App. E.2）。Pauli表現では反交換による相殺など $Q^2=I$ 以外の関係を使う低次収集（CTS型、Zhao–Yuan型）が、二次の主項で自由積集約を上回り得る（§7.4）。非Pauli表現を乱択部に用いる具体的PR設計は、Track Bでまだ提示されていない。

### 0.3 残る不確実性

1. 実PR乱択部の衝突確率 $\chi_R$（未計算。計算には別途承認が必要）。
2. Pauli-aware低次収集対照を同一targetで比較した証拠はない（G9/G10のmatched CTSは、Pauli展開でℓ1が1を超える非Pauli fixtureでの比較）。
3. 一般mでの $|B_{\rm partial}-B_{\rm full}|=O(\chi x^4)$ は概略証明で、m依存定数は未評価（m=3の厳密係数は本レビューで再導出し、既存GPT G6レビューの係数と一致した）。
4. 全結合の文献上の先行性（網羅的priority調査はしていない）。ただし結論は先行性に依存しない（§8.3）。

### 0.4 次に投資する科学的価値

- 追加の数値実験・SOURCE実装・provider追加への投資は推奨しない〔推奨〕。
- 情報価値が最も高い最小作業は次の二つ（本レビューでは実施せず、いずれも別途承認が必要）。
  1. **E1（新計算なし）**：§7の漸近比較・χ依存性・Pauli表現での位置付けを技術ノートv1.0の限界節として追記し、系列を構造的理由で区切る。
  2. **E0（O(L)の古典計算、要承認）**：既存の分子Hamiltonian係数とPRの分割から実乱択部の $\chi_R$ を求める。漸近的な乱択部費用削減の上限は正準return-awareで $\chi_R$、G6 zero-fillで $\chi_R/2$ であり、これが小さければRA系列の資源主張を終了する（§11）。

---

## 1. レビュー対象・目的・最終状態

### 1.1 対象

$$M=P_m(-i\sigma xR),\qquad R=\sum_{i=1}^{L}p_iQ_i,\quad Q_i^\dagger=Q_i,\ Q_i^2=I,\ p_i>0,\ \sum_ip_i=1,$$

奇数 $m$、$0<x\le1$、$\sigma=\pm1$ の有限Taylor operator meanに対するTrack Bの構成。

- G6：形式return集約、局所係数query（自由積Green関数）、偶数parent／奇数childのEuler pairing、ordinary envelopeからのzero-fill生成、有限bit確率と補正weight。
- G9：P5専用の少数群（L²+1群以下）生成。
- G7／G8／G9／G10・D1–D3・Parent allocation S4：条件付きnative費用・十分N・保証・価格の保存比較。

### 1.2 目的

依頼文§1.4の三点：(1) 既知研究と区別できる科学的・アルゴリズム的貢献の有無、(2) その貢献が単独査読論文の中心を支える程度か、(3) 不十分な場合に何が決定的に不足し、追加投資に合理性があるか。

### 1.3 最終状態

**完成版**。結論は§0。本レビューは研究上の推奨であり、D＋E・mandatory STOP・他研究の状態を変更しない。

---

## 2. 固定資料・取得・検証状況

### 2.1 Repository（sandbox内の読み取り専用cloneから `git show` で取得。import・実行なし）

| 資料 | 固定commit | 読んだ範囲 |
|---|---|---|
| 論文化向け技術調査（必須9点＋README） | `aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14` | README、paper_claim_sheet、prior_art_method_comparison、results_usability_map、minimal_evidence_proposal、handoffは全文。technical_classification、literature_scope、source_static_read_inventoryは全構造。evidence_indexは80参照の全一覧・欠測4項目・計数 |
| G6 独立数学監査、finite-bit・access監査 | `bc649c592c8c2b625b49094b85c27e380faa3cf4`（`28cfabb1` の同名ファイルとSHA256一致） | 全文 |
| G6 prior-art・method delta、G6結果handoff | 同上 | 全文 |
| ReturnKernel SOURCE `return_aggregation.py` | `28cfabb1d47e0e1824bce1e154a9c289738fa2b9` | 全254行（静的読解） |
| `g7_generator.py` | `ab2549f41b3546fb3940342a2162dd9ee93699c4` | 1–140行（partial／P3 arm、quartic helper） |
| G7結果、G8結果、G9 v2結果、G9契約、G10 v3結果、G10 claim map | `bc649c59…` | G7は主要部、G8は冒頭、G9・G10は全文 |
| 技術・方法ノート v1.0（docs/manuscripts） | `bc649c59…` | 本文全体と付録A–Cの冒頭 |
| GPT G1・G6レビューの小x展開箇所、G10公開・endpointレビュー | `101bf342…`、`bc649c59…` | G6 §5–6、G1 §6、G10公開 §1・§4.2・§8 |
| Parent allocation closure（D＋E）、資源構造・ボトルネック | `101bf3426fcd04cc0c01c0d202d0916268765d97` | closureは全文、資源構造は前半（系列別整理）を精読し後半は見出し確認 |
| S4科学レビュー完成版 | `e3086f48c5598c8bb31b0587e807ca6c1b3d2754` | 見出し全体、冒頭、§2、§4.5、§5.6 |
| S4 Evidence Packet | `53c48317d16851333c0e0354b0dfd8ef91e1d524` | 見出し構成のみ（数値はclosure・S4レビューの転記を使用） |

**取得上の注意（重要）**：依頼文§2.2のG6監査URLのcommit `bc649c592c8b2b625b49094b85c27e380faa3cf4` は、clone・SHA指定fetch・raw URLのいずれでも取得できなかった（404／not our ref）。調査資料（results_usability_map、evidence_index）が用いるcarrierは `bc649c592c8c2b625b49094b85c27e380faa3cf4` で、**1文字（`8b2`／`8c2`）異なる**。後者から取得したG6監査2文書は、G6監査commit `28cfabb1` の同名ファイルとSHA256が一致した（先頭16桁 `c1d547bcd15d6b02`、`9520ab74c6134441`）。依頼文のURLは誤記と判断した。

### 2.2 一次文献（本文PDFを今回取得）

| 文献 | 版 | 読んだ範囲 | 取得bytesの照合 |
|---|---|---|---|
| K. Aomoto, Y. Kato, Green functions and spectra on free products of cyclic groups, Ann. Inst. Fourier 38(1) (1988) | numdam AIF_1988__38_1_59_0 | (1.6)前後、Lemma 1.1系のcorollary、(1.16) | SHA256がCodex記録と一致 |
| K. Wan, M. Berta, E. T. Campbell, arXiv:2110.12071 | v2 | App. C（(C1)–(C5)）、Algorithm 2、truncation記述 | 一致 |
| Q. Zhao, X. Yuan, Exploiting anticommutation in Hamiltonian simulation, arXiv:2103.07988 | v2 | §4.1末–§4.3（式(22)–(40)） | 一致 |
| P. Zeng, J. Sun, L. Jiang, Q. Zhao（PTSC）, arXiv:2212.04566 | v2 | §IV.A（式(42)–(49)）、Proposition 3の記述 | 一致 |
| J. Peetz, S. E. Smart, P. Narang, Quantum Simulation via Stochastic Combination of Unitaries（SCU／CTS）, arXiv:2407.21095 | v2 | Theorem 1、CTS手順、Methods IV.B、Supp. Note 3 | PDF取得（Codexは同版のHTML） |
| D. Cugini, T. A. Atif, Y. Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495 | v1 | Theorem 1（(9)–(10)）、§IV ZeroFill(L)・Discard(L)、Corollary 1冒頭 | PDF取得（Codexは同版のHTML） |
| B. Koczor, Sparse Probabilistic Synthesis of Quantum Operations（SPS）, arXiv:2402.15550 | v2 | Statement 1、Statement 3冒頭 | PDF取得（Codexは同版のHTML） |
| J. Günther et al., Phase estimation with partially randomized time evolution（PR）, arXiv:2503.05647 | v2 | App. A.2（(A18)–(A29)）、App. E.1b、E.2、E.2c | 一致 |

全論文の精読、全証明の独立再証明、著者SOURCEの取得・実行は行っていない。追加の文献検索は行っていない（理由は§8.3）。

### 2.3 次回取得可能性

第1セッションで本レビュー用作業ディレクトリへ保存した作業記録（1,260 bytes、06:41:39 UTC作成）が、第2セッション開始時に同一パスで残存していたことを確認した。完成版も同ディレクトリに保存し、保存後に存在・bytes・SHA256・節構成・GitHubリンク先の固定commit内での存在を検証した。検証結果は同ディレクトリの `レビュー作業記録.md` §10に記録する（本ファイル内に自己申告として書かず、別ファイルに保存後の実測を残す）。v2は同ディレクトリの `v2/レビュー.md` として排他的に作成し、v1（`レビュー.md`）は変更せず保持する。v2の検証結果は作業記録§11に記録する。

---

## 3. Track Bの到達点と従前判断との関係

### 3.1 到達点（〔固定〕〔保存結果〕）

| 系列 | 確立したこと | 限界 |
|---|---|---|
| G6 | 有限 $P_m$ の形式return集約恒等式、short-step正値性、自由積Green関数による局所係数query、parent pairing、ordinary envelope＋zero-fillによるnormalizer不要生成、有限bit補正weightとbias・range上界 | native・controlled実装と資源価値は未検証 |
| G7／G8 | 局所生成をcontrolled provider・Rz取得へ接続（条件付き記号費用、短いon-demand trace） | 物理controlled-Q費用、大Lのonline費用は未取得 |
| G9 v2 | P5少数群生成。固定3-qubit syntheticでclosed P5のT切片が登録6方式で最小、matched CTSはCXで有利 | cheap I1 synthetic、1入力 |
| G10 v3 | m=3／5／7の同次数比較。一般fullは各次数のclosed対照に勝たない。m=7の固定辞書下界 | 同上 |
| D1–D3 | 保証・価格・境界簡約の寄与分離、順位の反転例 | 一般selectorはない |
| S4 | 配分＋ISのbest-to-bestで約14%／19%のT切片低下（N・CX増） | r=1、h_T=0、異なるproposal、多stepに懸念 |
| 10/15 D＋E | 限定成果の保持と現行探索の区切り | — |
| 10/18 Codex調査 | A：`A_INSUFFICIENT_EVIDENCE`、B：`B_DESCRIPTIVE_RESULTS_ONLY`、単独論文化保留を推奨 | 新定理・新解析は意図的に0件 |

### 3.2 従前判断との一致点と相違点

| 従前判断 | 本レビューとの関係 |
|---|---|
| GPT G6レビュー（10/10）：ordinaryからの $O(\chi x^2)$ 改善の主項はpartial returnでも得られ、全集約の追加はm=3で $O(x^4)$。zero-fillの保守予算は $1+\chi x^2$ の因子で不利 | **一致**。本レビューは多step漸近へ拡張し（対数二次momentの $2(1-\chi)c$ 対 $(2-\chi)c$）、envelope改良でも救済できないことを示す（§7.1）。 |
| GPT G10公開・endpointレビュー（10/11）：独立原著としての投稿や追加性能探索は推奨せず、技術・方法ノートとして完成 | **一致**。本レビューは、その判断の理由を「中心貢献が十分に切り出されていない」から「独自要素に漸近的資源価値がない」という構造的理由へ強める。 |
| 10/14 方向レビュー（Parent allocationを第一候補） | S4後の結果と§7.6の多step論から、配分系の単一step利益は長時間PRへ移送されないと判断する（10/15 D＋Eと整合）。 |
| S4完成版レビュー（e308）：多stepの頂点下界は総T比の証明ではない | その限定は妥当。本レビューは追加条件（独立step積推定、回転を含む事象のT単価の正の下界）を明示したうえで総費用への含意を§7.6に記す。 |
| Codex調査（10/18）：A `A_INSUFFICIENT_EVIDENCE` | **相違**。Codexは新定理・新解析0件という制約下で「証拠不足」と分類した。本レビューは承認された紙上推論で漸近比較を行い、Aの独自要素は資源価値を持たないと判断する。従って「Aは未確定」ではなく「Aは現存構成では成立しない」とする。 |
| Codex調査：B `B_DESCRIPTIVE_RESULTS_ONLY`、単独論文化保留 | **一致**。 |

---

## 4. G6の数理構造・局所生成の検証

### 4.1 集約恒等式（〔数学検証〕PASS）

$(-i\sigma xR)^n/n!=(-i\sigma x)^n/n!\sum_{w\in[L]^n}p(w)Q(w)$ で、$Q_i^2=I$ のみから $Q(w)=Q(\mathrm{red}(w))$。$n-|u|$ は偶数で $(-i\sigma)^{n}=(-i\sigma)^{|u|}(-1)^{(n-|u|)/2}$ なので

$$a_u=\sum_{\substack{|u|\le n\le m\\ n-|u|\ {\rm even}}}(-1)^{(n-|u|)/2}t_nP_n(u),\qquad t_n=\frac{x^n}{n!},\qquad M=\sum_u(-i\sigma)^{|u|}a_uQ(u).$$

語の並べ替えは不要で、operatorの順序と位相を保持する。

### 4.2 挿入上界 $P_{n+2}(u)\le(n+1)\chi P_n(u)$（〔数学検証〕PASS）

$|u|\le n$ で長さ $n+2$ のraw語がuへreduceするなら隣接同labelが存在する（無ければraw語自体がreducedで長さ $n+2>|u|$ となり矛盾）。最初の隣接pairを削除する写像は、対象語から（長さnでuへreduceする語、n+1個のgap、label i）の組への**単射**であり、重みは $p_i^2$ 倍になる。従って和は $(n+1)\chi P_n(u)$ 以下。多重被覆があるため等式ではない（監査の記述どおり）。

### 4.3 short-step正値性（〔数学検証〕PASS、$0<x\le1$ に限定）

$c_n=t_nP_n(u)$ について $c_{n+2}/c_n\le\chi x^2/(n+2)\le1/2$。項の絶対値が単調減少する有限交代和なので、$l=|u|$、$P_l(u)=p(u)$（長さlでuへreduceする語はu自身のみ）として

$$t_lp(u)\Bigl(1-\frac{\chi x^2}{l+2}\Bigr)\le a_u\le t_lp(u).$$

$x>1$ への延長は主張されておらず、本レビューでも行わない。

### 4.4 自由積Green関数（〔数学検証〕PASS、既知式の特殊化）

部分積 $g_k=Q_{I_1}\cdots Q_{I_k}$ は $\Gamma=\mathbb Z_2*\cdots*\mathbb Z_2$ のCayley木上の右乗算walkで、左平行移動がwalkの自己同型であるため、任意の頂点 $v$ から隣接頂点 $vs_j$ への初到達生成関数は根からの $F_j$ に等しい。最初の一歩で分解すると

$$F_j=zp_j+\sum_{k\ne j}zp_kF_kF_j\ \Longrightarrow\ F_j=\frac{zp_j}{1-z\sum_{k\ne j}p_kF_k},\qquad G=\frac1{1-z\sum_kp_kF_k},$$

木の測地線を必ず通るので $\sum_nP_n(u)z^n=G\prod_{j\in u}F_j$。〔SOURCE〕`ReturnKernel.__init__`（147–160行）の再帰 $f_i[n]=p_i\mathbf 1_{n=1}+\sum_{b=1}^{n-2}f_i[b]\,(C[n-1-b]-p_if_i[n-1-b])$、$G[n]=\sum_{a=1}^{n-1}C[a]G[n-1-a]$ は、$F_i=zp_i+zF_i(S-p_iF_i)$、$G=1+zSG$（$S=\sum_kp_kF_k$）の係数比較と一致する。`word_series`、`_mass`、`parent`（childは空parentで全label、非空parentで先頭label以外）も式どおりである。

〔文献推測〕Aomoto–Kato の(1.6)は自由積上のGreen関数比の乗法性、(1.16)は各因子のmultiplierを他因子のreturnでshiftした引数で評価する形（OCR表示からの読み取りで $f_j(z)=F_j(z-X(z)+X_j(z))$）を与え、$\mathbb Z_2$ 因子では上の $F_j$ へ帰着する。G6監査の対応導出と整合する。木・自由積上のrandom walkのこの種の生成関数は教科書的事実であり、新しい数学的貢献ではない。

### 4.5 parent pairing（〔数学検証〕PASS）

$Q_i^2=I$ より $d_u\sum_iq(i|u)e^{-i\sigma\phi_uQ_i}=a_uI-i\sigma\sum_ia_{iu}Q_i$（$\cos\phi_u=a_u/d_u$、$\sin\phi_u=s_u/d_u$）。各奇数reduced語は先頭を除いた唯一の偶数suffixを持つので、重複も欠落もない。回転generatorは単一の $Q_i$ だけで、奇数語全体をinvolutionとは仮定していない。〔SOURCE〕`phase_i_power = len(word) % 4`（偶数長で $i^{|u|}=(-1)^{|u|/2}$）と `circuit_word_time_order = reversed(word)` は、位相とcircuit時間順の扱いに一致する。

### 4.6 envelope・受理率・厳密減少（〔数学検証〕PASS）

$a_u\le t_lp(u)$、$s_u\le\sum_{i\in C(u)}t_{l+1}p_ip(u)\le t_{l+1}p(u)$ から $d_u\le b_lp(u)$（$b_l=\sqrt{t_l^2+t_{l+1}^2}$）。長さlのreduced語のIID質量は1以下なので $B_{\rm full}\le B_{\rm ord}$。受理率は $A_u\ge a_u/(b_lp(u))\ge1/\bigl(2\sqrt{1+x^2/(l+1)^2}\bigr)\ge1/(2\sqrt2)$。$m\ge3$、$x>0$ では $a_\emptyset\le1-c_2+c_4<1$ により空parentで厳密な不等号になる。

### 4.7 zero-fill推定量（〔数学検証〕PASS）とその代価（〔紙上推論〕）

$\Pr(u\ {\rm accepted})=(b_l/B_{\rm ord})\,p(u)\,A_u=d_u/B_{\rm ord}$、受理時 $X=B_{\rm ord}Y$ より

$$\mathbb E X={\rm Re/Im}\,{\rm Tr}(\rho M),\qquad \mathbb E X^2=B_{\rm ord}B_{\rm full},\qquad |X|\le B_{\rm ord}.$$

全試行を分母とすること、accepted-only平均を禁じること、未知の $Z=B_{\rm full}/B_{\rm ord}$ を予算へ無料使用しないことは正しい。

normalizerを既知とする正準sampler（$d_u/B_{\rm full}$ で直接抽出）の二次momentは $B_{\rm full}^2$ である。従って**normalizer不要の代価は、二次momentで因子 $B_{\rm ord}/B_{\rm full}=1/Z$** であり、小xで $1+\chi x^2+O(x^4)$（GPT G6レビュー§6.1と一致）。G10の上界 $U$ は予算上の $Z_+$ を鋭くするが、**実際の二次moment $B_{\rm ord}B_{\rm full}$ 自体は変えない**。この点が§7の多step漸近で決定的になる。

〔保存結果〕〔数学検証〕G9 v2の保存reference値はこの関係と整合する。general local fullの受理率 $Z$ の表示値 0.857766137921、digital second moment reference 1.935363620466、closed P5（同じ理想ensemble、受理率1）の reference 1.660089378201 は、恒等式 $m_2^{\rm full}\,Z=B_{\rm full}^2=m_2^{\rm P5}$ を表示精度で満たす。本レビューで行ったのは恒等式の成立確認だけで、新しい量の算出や再順位付けではない。

### 4.8 計算量主張の区別（〔SOURCE〕〔数学検証〕）

| 項目 | 確認結果 |
|---|---|
| 前処理（算術op） | 各n・各iでbについてO(m)の和 → $O(Lm^2)$ 有理演算、係数保持 $O(Lm)$。PASS |
| 一parentのquery | `word_series` はl回の畳み込み（各 $O(m^2)$）で $O(lm^2)\le O(m^3)$。全childはL回の畳み込みで $O(Lm^2)$。合計 $O(m^3+Lm^2)$。PASS |
| bit complexity | 係数の分母は $D^m\,{\rm den}(x)^m\,m!$ を割り、bit長は $O(m(H_p+H_x+\log m))$。多項式。PASS（有理演算数をbit演算数・実時間と同一視しない点も主張どおり） |
| online sampling | 1試行：次数とraw語で $O(m)$、reducedのときのみquery。空parentは入力ごとに固定で前計算できる。ordinaryは1試行 $O(m)$ でqueryなし。**古典online費用はordinaryより大きい**。ただし追加のqueryは長さ2以上のreduced parentが提案されたとき（確率 $O(x^2)$）に限られ、label多重集合でcacheできるため、小xでは償却後の追加分は決定的でない〔紙上推論〕 |
| 回転取得 | 角度はlabel histogramと先頭labelに依存し、種類数のO(m)保証はない。ただし短step漸近では受理事象の大半が空parentの共通角 $\phi_\emptyset$ で、その合成列は再利用できる〔紙上推論〕 |
| 非列挙性 | ordinaryのIID生成も非列挙である。G6の非列挙性はordinary proposalから継承した性質で、対照に対する分離ではない |

### 4.9 対象の限定（〔固定〕〔数学検証〕）

targetは有限 $P_m$ で、$e^{-i\sigma xR}$ とのTaylor remainderは別会計である。対象はoperator first moment（Hadamard test）であり、ensembleのchannel一致では代用できない。controlled実装ではparent位相と積の順序を保持する必要があり、system-only $Q_i$ から無料でcontrolled oracleは得られない。

---

## 5. P5専用生成とfinite-bit保証の検証

### 5.1 P5閉形式（〔数学検証〕PASS）

- $P_2(\emptyset)=\chi$、$P_3(i)=p_i(2\chi-p_i^2)$（$jji$ と $ijj$ の和から $iii$ の重複を引く）、$P_4(\emptyset)=2\chi^2-\mu_4$（$aabb$ と $abba$ の和から $aaaa$ の重複を引く）。
- 一return等式 $P_{l+2}(u)=p(u)\{(l+1)\chi-\sum_tp_{i_t}^2\}$：長さ $l+2$ でuへreduceする語はuへ一対を挿入したものに限られ、同一語を二重に生むのは文字 $i_t$ の直前と直後への $i_ti_t$ 挿入だけである。
- ここから $A_{jk}=t_2-(3\chi-p_j^2-p_k^2)t_4$、$b_i^{jk}=p_i\{t_3-(4\chi-p_i^2-p_j^2-p_k^2)t_5\}$、$S_{jk}=(1-p_j)t_3-\{(1-p_j)(4\chi-p_j^2-p_k^2)-(\mu_3-p_j^3)\}t_5$、長さ4で $a_u=t_4p(u)$、$a_{iu}=t_5p_ip(u)$、$\tan\phi_u=x(1-p_j)/5$ を確認した。
- $L=1$ の極限で $P_3=P_4=P_5=1$ となる（$P_5(i)=p_i(5\chi^2-4\chi p_i^2+2p_i^4-2\mu_4)$ も1）。$P_5(i)$ の一般式は原導出とG9のfirst-adjacent-deletion oracle照合に依拠し、本レビューでは極限検査のみ行った。
- 群数はroot 1＋ordered pair $L(L-1)$＋先頭label $L$ で $L^2+1$。群構築 $O(L^2)$、条件付き抽出 $O(L)$、現行の群選択は線形scan（$O(L^2)$）であり、技術ノートの自己記述どおり。

**評価**：P5閉形式は正しく、受理率1・normalizer既知の正準samplerを与える点で一般生成より実用的である（G9/G10の保存結果と整合）。しかし低次の語の組合せ計算であり、新しいアルゴリズム原理ではない。また§7.1のとおり、その漸近的利益はpartial returnと同じ主項に由来する。

### 5.2 finite-bit構成（〔数学検証〕PASS）

- 各dyadic質量が理想値の $(1\pm\eta)$ 内にあり受理を下向きに丸めるとき、$\pi_{u,i}\ge(1-\eta)^{l+3}\alpha_{u,i}/B_{\rm ord}$、$|\tilde W|\le(1+\rho)B_{\rm ord}/(1-\eta)^{m+2}$（$l\le m-1$）。
- 実proposal $\pi$ で割るため $\pi\tilde W=\tilde\alpha$ が厳密に成り立ち、確率の丸めはbiasを生まない。meanは $\tilde M=\sum\tilde\alpha V$ であり、元の $M$ との差は $\sum_u|\tilde d_u-d_u|\le\rho B_{\rm full}$。
- $B_{\rm ord}\le\sum_nt_n\le e^x<3$ によりglobal normalizerなしでrange・biasを保守的に認証でき、二次momentは $\le\kappa B_{\rm ord}B_{\rm full}\le\kappa B_{\rm ord}U$（$\kappa=(1+\rho)^2/(1-\eta)^{m+2}$）。
- 角度：$s_u/a_u\le2x/(l+1)\le2$、$\phi=2\arctan[r/(1+\sqrt{1+r^2})]$ の内部引数は $2/3$ 未満でalternating級数による囲い込みが可能。角度誤差τはinvolution回転のoperator差τ以下に戻る。native誤差 $\delta_e$ は $\sum_e\tilde\alpha_e\delta_e$ として別会計。
- 精度不足はfail closed、ランダムなbit追加やretryはない。

**評価（問題Bへの回答）**：実dyadic確率と理想確率の区別、係数近似と重要度重みの整合、zero-fillでの全試行分母、normalizer未知時の保守的moment・range、近似係数でのmean保存と理想targetへのbiasの区別、controlled位相・順序、精度の取得とcertification——いずれも正しく丁寧に処理されている。しかしこれは「実proposalを重要度重みで補正し、係数近似を明示的biasとして会計する」という標準的手法の正確な実装であり、**全体としても独立した理論・アルゴリズム貢献には当たらず、engineering統合（実装の正しさ）に還元される**と判断する。独立成果として残り得たのは、例えば必要bit数の上界と下界が一致する計算量定理や、既知手法を越えるcertification方式であるが、現資料にはいずれも含まれない。論文に含めるなら付録の「certified finite-precision implementation」としての価値である。未取得事項は、任意入力に対する精度選択器、大Lでの1試行あたりの有理演算・root計算の実時間、物理controlled-Q費用。

---

## 6. 既知手法との厳密な方法差比較

### 6.1 一次文献との対応（〔文献推測〕と記した箇所以外は該当節で確認した事実）

| 既知研究（確認箇所） | 既知の内容 | Track B構成との関係 |
|---|---|---|
| Aomoto–Kato 1988（(1.6)、(1.16)） | 自由積のGreen関数の乗法性と、他因子のreturnによるmultiplierのshift | §4.4の $F_i,G$ はその $\mathbb Z_2$ 特殊化。数学的に包含 |
| Wan–Berta–Campbell v2（App. C、Algorithm 2） | 隣接Taylor次数のEuler pairing、次数とIID labelからの非列挙生成、1segmentの正規化 $\le e^{x^2}$ | ordinaryそのもので、G6のproposal。returnは収集しない |
| Zhao–Yuan v2（§4.2 (24)–(29)、§4.3） | $H_l^2=I$、$H_lH_{l'}H_{l'}=H_l$ を用い、高次語のうち $I$ や $H_l$ に等しいものを低次係数へ吸収。反交換による相殺も利用 | 「returnを吸収する」原理そのもの。決定論的truncated Taylor LCUであり乱択samplerではない。G6は全reduced語へ一般化し、使う関係は $Q^2=I$ のみ |
| PTSC v2（§IV.A (42)–(49)） | 0次と1次のみのpairing。1-normは $\approx e^{3(\lambda x)^2/2}$ | returnを収集しない。Track Bに対する対照としては弱い |
| SCU／CTS v2（Thm 1、Methods IV.B、Supp. Note 3） | Taylor級数をPauli基底で展開・収集（実係数は直接、虚係数はrotationへ）、古典 $O(K^M)$。層samplingは非列挙だが層間の簡約を逃す | Pauli代数の全関係（$Q^2=I$、反交換等）を使う。G6は層間を含む全returnを局所queryで保つが、使う関係は $Q^2=I$ のみ |
| Cugini–Atif–Subaşı v1（Thm 1、§IV） | 固定protocolで $N C_q\ge(\mathbb E_p c^{1/2})^2$、等号は $q\propto p/c^{1/2}$。ZeroFill(L)／Discard(L) | G10の固定辞書下界はThm 1の直接適用。G6のzero-fillは量子実行前の古典coinだが原理は同じ |
| Koczor SPS v2（Statement 1、3） | 辞書の係数の絶対値に比例して抽出する不偏推定、L1と合成費用の交換 | S4の「安いpure事象と重みの交換」はこの原理に近い |
| Günther et al. PR v2（App. A.2、E.2、E.2c） | RTEの正規化 $\le e^{\tau^2r}$、$r=\lceil t^2\rceil$ で定数overhead。乱択部は**Pauli表現**で、DFは決定論側のみ | Track Bの非Pauli involution fixtureは、PR原設計の乱択部とは表現が異なる |

### 6.2 四つの軸での判定

| 軸 | 判定 |
|---|---|
| 数学的包含 | 全構成要素が既知理論に包含されるか、その初等的帰結。Track B固有の補題（挿入上界、正値性、受理率下界、有限bit上界、P5閉形式）は正しいが初等的 |
| 生成手順の同値性 | 「ordinary envelope＋非reduced語のzero＋局所Green query＋parent／child pairing＋有限bit補正」をこの形で一体化した記載は、確認した節には見当たらない。ただし既知部品の直接的な組立てであり、完全一致の不在は新規性の証拠ではない |
| 情報アクセス差 | ordinaryと同じp表・controlled $Q_i$・controlled回転を要する。有利なaccess分離はない。Pauli代数が使える場合は、使える関係が少ない分むしろ不利になり得る（§7.4） |
| 計算量差 | 古典：1試行あたりordinaryより多い（局所query。ただし§4.8のとおり小xでは償却される）。CTSの全展開 $O(K^M)$ より安いが、CTSの層samplingも多項式。量子：1stepの回路単価は主項で同じで、差は二次momentの指数だけ（§7.1） |

### 6.3 問題Aの各問いへの回答

1. **Green型再帰は既知理論のどの特殊化か**：自由積のGreen関数（Aomoto–Kato §1）の、$\mathbb Z_2$ 因子・重み $p_i$ への特殊化。木上の初到達分解と同じ。
2. **既知結果から直接従う範囲**：係数 $P_n(u)$、parent／child構造、交代和による正値性は、既知の生成関数と初等的不等式から直接従う。非負性は $x\le1$ の短step条件付き。局所queryは生成関数の切断係数計算そのもの。
3. **ordinary IID・CTS層samplingとの差**：G6の生成はordinaryのIID proposalに局所受理を加えたもので、非列挙性はordinaryから継承している。CTS層samplingとの実質差は「層をまたぐ全returnを保つ」点だが、その資源上の効果は1stepあたり $O(\chi x^4)$ にとどまる（§7.1）。
4. **P5専用経路**：normalizerを $O(L^2)$ で得て受理率1の正準samplerにできる点で一般生成より有用。ただし原理は低次の閉形式計算。
5. **同情報・通常最適化を既知法に認めた後の優位**：二次の主項 $(1-\chi)$ はpartial return（Zhao–Yuan型吸収の二次への適用）で得られ、一般構成固有の生成・取得上の優位は残らない。

---

## 7. 同情報・同最適化条件での資源価値

### 7.1 多step・短step漸近（命題R、〔紙上推論〕）

固定奇数 $m\ge3$、$\chi<1$、1stepのパラメータ $x$、step数 $r$、$c=rx^2$ とする。1stepの係数質量は（既存レビューの展開と独立に一致）

$$B_{\rm ord}=1+x^2+O(x^4),\qquad B_{\rm partial}=B_{\rm P3+tail}=B_{\rm P5}=B_{\rm full}=1+(1-\chi)x^2+O(x^4).$$

ordinaryは $\sqrt{1+x^2}+\tfrac{x^2}{2}\sqrt{1+x^2/9}+O(x^4)$。return-aware方式はroot $\sqrt{(1-\chi x^2/2)^2+x^2+O(x^4)}=1+(1-\chi)x^2/2+O(x^4)$ と、長さ2のparentの総質量 $(1-\chi)x^2/2+O(x^4)$ から得る。独立step積推定量の二次momentは次のとおり。

| 方式 | 1stepの二次moment | r stepの対数 |
|---|---|---|
| ordinary（WBC型） | $B_{\rm ord}^2$ | $2c+O(cx^2)$ |
| normalizer既知のreturn-aware正準sampler | $B^2$ | $2(1-\chi)c+O(cx^2)$ |
| G6 zero-fill（ordinary envelope） | $B_{\rm ord}B_{\rm full}$ | $(2-\chi)c+O(cx^2)$ |

**帰結。**

- (i) 同じ二次moment予算での必要step数は $r_{\rm can}/r_{\rm ord}\to1-\chi$、$r_{\rm zf}/r_{\rm ord}\to1-\chi/2$。乱択部の総費用（1試行の回路単価 ∝ r、試行数 ∝ 二次moment）をrについて最適化すると $r^*\propto aT^2$、総費用 $\propto aT^2$（$T=rx$、$a=1,\ 1-\chi,\ 1-\chi/2$）。**乱択部費用の漸近的削減率の上限は、正準return-awareでχ、G6 zero-fillでχ/2である。**
- (ii) partialとfullの係数質量差は1stepあたり $O(\chi x^4)$。m=3では本レビューでroot（$(\chi-\mu_3)x^4/6$）と長さ2のparent（$\sum_jp_j^2(1-p_j)(2-p_j)\,x^4/36$）の寄与を再導出し、その和が既存GPT G6レビューの厳密係数 $\sum_ip_i^2(1-p_i)(8-p_i)\,x^4/36\le\tfrac29\chi x^4$ に一致することを確認した。一般mは概略証明にとどまる：差分項は空parentのchild係数の $O(x^3)$ 補正、長さ2以上のparentの相対 $O(x^2)$ 補正、長さ4以上のtailからのみ生じ、いずれも少なくとも一回の隣接衝突（確率 $O(\chi)$）を含む。m依存の定数は未評価。r step合計では $O(\chi c\,x^2)\to0$。
- (iii) full ensembleの固定事象を重要度重みで抽出する任意のsamplerは、Cauchy–Schwarzにより1stepの二次moment $\ge B_{\rm full}^2$ を持つ。従って**envelopeやproposalを改良しても、一般fullはpartialを漸近的に意味のある差で上回れない**（差は $O(\chi cx^2)\to0$）。G6 zero-fillはさらに $e^{\chi c}$ の因子だけpartialより不利になる。
- (iv) 1stepの回路単価は主項で一致する（受理事象の大半は1回転で、偶数長parentによる追加の $Q$ 積は確率 $O(x^2)$）。角度依存の合成価格差（D3・S4で数T規模が観察されている）は系統的な方式差ではない。

**PRとの接続**：PR原論文は $r=\lceil t^2\rceil$（$\tau\approx1/t$、$\tau^2r\approx1$）で定数overheadを取る（App. A.2 (A29)）。これは上の $x\to0$、c固定の領域に当たる。

### 7.2 保存結果の読み方（〔保存結果〕〔紙上推論〕）

- G10 v3（m=5）のT切片（百万、技術ノート§7.1の表示値）：ordinary 371.09、partial 275.44、closed P3＋tail 284.75、full 273.99、closed P5 272.24、matched CTS 387.63。return-aware方式間の差はordinaryとの差に比べ小さく、一般fullは各次数のclosed対照に対し +0.19%（m=3）、+0.64%（m=5）、+2.91%（m=7）（技術ノート§7.2）。
- S4レビュー§5.6：P5 commonはordinary commonに対しh=0のT −26.6%、正規化Bが1.502→1.288、partial returnとの差は1.16%。
- これらは $\chi=19/50$（fixtureの $p=(1/5,3/10,1/2)$ から $0.04+0.09+0.25$。入力パラメータの初等算術）、$x=5/7$、r=1–2の結果である。方向は§7.1と整合するが、定量的照合は新規計算になるため行っていない。
- D1：P5対partialのT差はr1約 −1.16%、r2約 −5.84%で、後者は主に回路単価（角度依存の合成価格）の差と記録されている（資源構造文書）。一般構成固有の系統的利益の証拠ではない。

### 7.3 一般局所構成の「挟み撃ち」（〔紙上推論〕〔仮説〕）

- 局所Green queryが**不可欠**になるのは、全reduced語（長さ $\le m-1$、長さlで $L(L-1)^{l-1}$ 個）やclosed形式の群数が大きい、すなわちLが大きい場合である。
- $\chi\ge1/L$（Cauchy–Schwarz）であり、PRの分割（大きい項を決定論側へ）では乱択部の $\chi_R$ は小さくなる傾向にある〔仮説：Track Bの実入力で未確認〕。$\chi_R$ が小さければreturn-aware全体の利得（§7.1(i)）も小さい。
- χが大きいのは少数項に質量が集中する場合で、Lが小さければ列挙やP5型closed形式で足り、Lが大きく少数の大項がある場合はPRの設計上その大項が決定論側へ移る〔仮説〕。
- 従って、一般局所構成が不可欠かつ有益な入力領域は狭い。例外は「多数の小項＋乱択部に残された少数の大項」で、PRの通常の設計とは異なる。

### 7.4 Pauli表現での位置付け（〔紙上推論〕〔文献推測〕）

$Q_i$ がPauli文字列なら、二次項は符号を除き $\tfrac{x^2}{2}R^2=\tfrac{x^2}{2}\bigl[\chi I+\sum_{i<j}p_ip_j\{Q_i,Q_j\}\bigr]$ で、反交換対は消え、可換対は $2p_ip_jQ_iQ_j$ にまとまる。自由積集約は $Q_iQ_j$ と $Q_jQ_i$ を別の語として残すため、長さ2の非identity部のℓ1は $1-\chi$、Pauli収集では $w_{\rm comm}=\sum_{i\ne j,\ [Q_i,Q_j]=0}p_ip_j\le1-\chi$ となる。規格化の $x^2$ 係数は自由積集約で $1-\chi$、Pauli収集で $\tfrac12(1-\chi)+\tfrac12w_{\rm comm}$ であり、反交換対に重みがあれば**Pauli-aware低次収集の方が厳密に良い**。これはZhao–Yuanの反交換利用、CTSのPauli収集の原理に含まれる。ここで比較しているのは理想係数質量の $x^2$ 係数であり、native費用や古典前処理（Pauli対の収集は $O(L^2)$、CTSのM次展開は $O(K^M)$）は含まない。

PR原論文の乱択部はPauli表現なので、そこでの最強対照はPauli-aware低次収集＋ordinary tailであり、G9/G10のmatched CTS（非Pauli fixtureで $Q_1=(IZI+XYI)/\sqrt2$、$Q_2=IIZ/\sqrt2+(IXY-ZYY)/2$ とPauli展開によりℓ1が膨らむ条件）とは異なる。非Pauliのinvolution表現（DF型の回転反射）を乱択部に用いる場合も、同じDF leaf内の可換構造など、自由積より強い関係が残り得る〔仮説〕。

### 7.5 依頼文§3問題Cの9項目

| 項目 | 直接確認されたこと | 未検証の移送 |
|---|---|---|
| 1 回路単価 | 固定synthetic、literal加法会計、共通compilerでの保存T／CX／1Q | whole-circuit compile、実provider、batch合成 |
| 2 正規化・moment・range | 有理数で保存。理想式はG6で証明 | 多stepは§7.1の紙上予測のみ |
| 3 十分N | Bernstein十分予算（実測shotsでも必要最小shotsでもない） | 実task |
| 4 Taylor誤差・native bias | 同次数内は同target。D3で保証変更がrの選択を変える例 | 同一指数近似精度での方式比較、PR全体 |
| 5 有限bit保証 | bounded構成、fail closed、保存会計で整合 | 任意入力の精度選択器 |
| 6 受理率・zero-fill | 理想式と保存値で確認。代価は $B_{\rm ord}/B_{\rm full}$ | 多stepでの $e^{\chi c}$ の不利（§7.1） |
| 7 classical setup・online・normalizer | 小入力の短いtrace | 大Lのonline費用（`NOT_MEASURED`） |
| 8 state preparation・readout | h付きaffine会計。S4のT交点はordinary約27.3、P5約11.3 T/trial | 実PRのh |
| 9 多step反復 | 未実証（S4はr=1、D系列はr=1,2） | §7.1・§7.6の紙上推論のみ |

### 7.6 Parent allocation S4の約14%／19%（〔保存結果〕〔紙上推論〕）

- 〔保存結果〕best common対best noncommonのbest-to-best（proposalが異なる）、r=1、$h_T=0$、strict 1e-6の固定条件。代表pairでNは約1.72／3.34倍、CXは約1.43／2.56倍（S4レビュー§2.2）。配分単独・RA固有・全資源・実分子の改善とは解釈しない（closure・S4レビューと同判断）。
- 〔紙上推論〕利益の機構は、T不要のpure事象（反射）を作りL1増加と引き換えにする既知のSPS・PAI型の交換である。頂点配分の1step係数質量は $1+x(1-p_j)+O(x^2)$（S4レビュー§4.5）で、r stepの二次moment下界は $e^{2\tau(1-p_j)}$（$\tau=rx$）とrによらず総時間に対して指数的に増える。一方、pure事象の確率は1stepあたり $O(x)$ なので、回転を含む事象のT単価が正の定数で下から抑えられる限り（受理事象の大半は依然として回転を含む）、T削減率は $O(x)\to0$。従って独立step積推定の下では、**一次でL1を増やす配分は長時間PRで漸近的に不利**であり、S4の利益は単一stepの交換として扱うべきである。e308の記述は下界にとどめた慎重なもので、ここで加えたのは上記の追加条件付きの総費用推論である。
- 共通角と非共通角の合成機会：共通角では同一角の合成列を全childで再利用でき、多stepでは同角・可換な回転のbatch合成（WBCが言及するHamming weight phasing等）の機会も対照側に生じ得る。S4はr=1でこの機会を対照に与えていないため、移送を論じる際は対照にも同じ最適化機会を与える必要がある〔仮説〕。

---

## 8. 独立新規性と論文の中心的貢献の判定

### 8.1 A：独立アルゴリズム論文 — 不成立（確信度：高）

新しい乱択時間発展アルゴリズムとしての中心貢献に必要な「既知法に対する資源上の独自優位」が、独自要素（一般m局所query＋zero-fill）について漸近的に存在しない（§7.1(ii)(iii)）。資源上の利得は二次return吸収（既知原理）に帰属し、PRのPauli乱択部ではPauli-aware収集が主項で上回り得る（§7.4）。

### 8.2 B：範囲限定の構成・方法論論文 — 現存構成中心では推奨しない（確信度：中〜高）

仮に主張するなら、最も狭く正確な命題は次である。

> 有限奇数次Taylor多項式 $P_m(-i\sigma xR)$（$R=\sum_ip_iQ_i$、$Q_i^2=I$、$0<x\le1$）に対し、$Q^2=I$ のみを用いた形式return集約表現は、全係数正、$B_{\rm full}\le B_{\rm ord}$、受理率 $\ge1/(2\sqrt2)$ のnormalizer不要な不偏sampler、有限bitでのbias $\le\rho B_{\rm full}$・range $\le3(1+\rho)/(1-\eta)^{m+2}$ を持つ。ただし多step漸近での二次momentの対数係数は、ordinaryの2から正準samplerで $2(1-\chi)$、zero-fillで $2-\chi$ へ下がるのみで、前者は二次return吸収で既に達成される。

| 項目 | 内容 |
|---|---|
| 数学的・実装上の支持 | G6の証明（本レビューで独立検証）、ReturnKernel SOURCE、G9/G10の保存会計 |
| 最も強い既知対照 | 正準partial return、closed P3・P5（Zhao–Yuan型吸収の二次への適用）。Pauli表現ではPauli-aware低次収集（CTS低次＋ordinary tail）。ordinary WBC |
| 未解決の反論 | 漸近的支配（§7.1）、χ依存性（§7.3）、Pauli表現での劣位（§7.4）、古典online費用、controlled-Qの物理費用 |
| 実用価値が成立し得る条件 | 少step・$x\lesssim1$、乱択部のχが大きく、非Pauli involution表現が必要で、Pauli・可換構造が使えない入力。現時点で具体的PR設計は未提示 |

この命題の後半（限界）を正直に含めると、主たる内容は「一般構成は漸近的に単純な構成と同等」という限定的負結果になり、査読付き原著の中心としては弱い。限界を含めなければ不正確になる。

### 8.3 文献上の先行性について

全結合の網羅的priority調査は行っていない。しかし結論は先行性に依存しない。仮に全結合が未発表でも、(a) 資源上の価値は既知原理の部分に帰属し、(b) 独自部分は漸近的価値を持たない。従って追加の文献探索は判断を変えないと判断し、実施しなかった（依頼文§2.4の「中心主張を左右する資料を優先」に沿った判断）。「完全一致が見つからない」ことを新規性の根拠には用いていない。

### 8.4 C・D

C（技術ノート・補助成果として保持し、単独論文化は当面見送る）を推奨する（§10）。Dは不要（§0）。

---

## 9. 反論・代替説明・重要な未解決事項

### 9.1 本判断への反論と応答

| 反論 | 応答 |
|---|---|
| 有限x・少step（early FT）では一般fullが有用かもしれない | 差分は衝突確率に比例し、G10（$x=5/7$、$\chi=19/50$）でも一般fullは勝っていない。可能性は残るが、中心貢献を支える証拠はない |
| 非Pauli involutionへの一般性は独自 | 一般性は事実だが、その一般性が必要かつ有益なPR設計は未提示。DF型でも自由積より強い関係が残り得る |
| normalizer不要は大規模入力で有用 | partialのnormalizerはχだけで決まりO(L)で得られる。normalizer不要の利点よりzero-fillの代価 $e^{\chi c}$ が大きい |
| 有限bitの厳密性には価値がある | 実装の正しさとしての価値は認める（技術ノートの付録）。独立した理論貢献ではない |
| 既知部品の組合せでも新アルゴリズムになり得る | 原理的にはそのとおりだが、本件では組合せの独自部分に資源価値がないことを§7で示した |

### 9.2 観察された利得の代替説明

| 観察 | 代替説明 |
|---|---|
| return-aware方式のordinaryに対する約1/4規模のT低下（fixture） | 衝突確率 $\chi=19/50$ が大きい入力での、$Q^2=I$ による二次の吸収効果。一般構成固有ではない |
| closed P5・partial・fullの間の数%以内の差 | 角度依存の合成価格（D3・S4で数T規模）、予算方式（$U$ 対 既知normalizer）の差 |
| S4の14%／19% | pure事象＋既知ISの交換、proposalの違い、h=0切片 |
| CTSに対するT利益 | 非Pauli fixtureでのPauli展開によるℓ1膨張。Pauli表現のPR乱択部へは移送されない |

### 9.3 重要な未解決事項

1. 実PR乱択部の $\chi_R$。
2. Pauli-aware低次収集対照との同target比較。
3. 一般mでの $O(\chi x^4)$ 差の厳密化。
4. 大Lでの1試行あたり古典費用、物理controlled-Q費用。
5. 非Pauli乱択部を用いるPR設計の具体性。

---

## 10. 独立論文化・限定的技術論文・保留の選択肢

| 選択肢 | 必要な追加証拠 | 本レビューの判断 |
|---|---|---|
| 新しい乱択時間発展アルゴリズムとしての論文 | 同情報・同最適化の強い対照に対し、多stepで残る資源優位 | 不成立。§7の漸近論が独自要素の優位を否定する |
| 既知要素を接続した限定的な構成・方法論論文 | §8.2の命題を中心に、漸近的限界と適用域を明示 | 推奨しない。限界を含めると中心貢献が弱く、含めないと不正確 |
| 実装可能性・資源診断を中心とする技術ノート | 既存v1.0に§7の漸近比較・χ依存性・Pauli位置付けを追記 | **推奨（C）**。内部記録とするかtechnical reportとして公開するかは著者判断 |
| Track B単独論文化を当面保留 | — | **推奨（C）**。再検討は§11 E3の前提が満たされた場合のみ |

---

## 11. 追加研究の情報価値・最小検証・STOP条件

〔推奨〕本レビューは新規計算・実装・実験を承認しない。以下は判断材料の提案で、採否はユーザーが決める。

| 候補 | 何の判断が変わるか | 必要な証拠・強いbaseline | 失敗条件・STOP | 承認 |
|---|---|---|---|---|
| **E0：実乱択部の $\chi_R$ 取得**（最小・情報価値最大） | return-aware系列が実PRで意味を持ち得るか。漸近上限は乱択部費用の $\chi_R$（正準）、$\chi_R/2$（G6 zero-fill） | 既存の分子Hamiltonian係数とPRの決定論／乱択分割から乱択部の正規化重みを求め、$\sum_ip_i^2$ を計算（O(L)の古典計算） | $\chi_R\times$（乱択部が総費用に占める割合）が小さい場合（目安0.05以下。D1–D3・G10で観察された角度依存価格の揺らぎと同程度以下）はRA系列の資源主張を終了 | 保存科学データの新規解析に当たるため別途承認 |
| **E1：技術ノートへの限界節追記**（新計算なし） | 系列の区切りを構造的理由で確定 | 本レビュー§7.1命題R、§7.3、§7.4 | — | Codexによる文書追記の承認（GitHub変更を伴う） |
| E2：一般mの $O(\chi x^4)$ の厳密証明 | 結論は変わらない見込み | 理論作業 | 推奨度は低い | 理論作業の承認 |
| E3：数値・native検証（条件付き） | 実PRでのreturn-aware利益の有無 | E0で $\chi_R$ が十分大きく、かつ非Pauli乱択部を要する具体的PR設計がある場合のみ。対照：ordinary、正準partial、closed P3・P5、Pauli表現ならPauli-aware低次収集、DF型なら可換構造を使う対照。同じ合成器・精度・cache・batch・whole-circuit最適化、同じ指数近似精度（Taylor biasを戻す）、PRのr選択則による多step、prep h込み | return-aware全体の利得が $\chi_R$ 相当の上限に届かない、Pauli-aware・可換対照に劣る、または一般fullが正準partial・closedを上回らない（§7.1の予測どおり）場合はSTOP | 研究方針・契約・実行の別途承認 |
| **E4：見送り**（推奨デフォルト） | — | — | — | 不要 |

Bの方法論（一般selector）については、既知のshots×回路価格会計を越える新しい評価原理の手掛かりがなく、追加投資を推奨しない。D3・S4の保存診断は技術ノート内の事例として保持する。

---

## 12. 次の担当、必要な承認、参考文献、取得できなかった資料

### 12.1 次の担当

1. **ユーザー**：本レビューの推奨（C）の採否、E0・E1の承認可否。
2. **Codex**（承認時のみ）：E1の文書追記、E0の計算準備。SOURCE変更・新契約・科学計算は各承認が前提。
3. **GPT**（必要時のみ）：E0の結果を受けた再判定。

### 12.2 本レビュー完成によって自動承認されないもの

研究方針の正式採択、論文投稿GO、新規科学計算（E0を含む）、SOURCE実装・実行、GitHub上の文書追加、新規契約・authorization・grant・marker・claim。Track BのD＋E、研究④の新テーマ選定保留、研究①②③の状態は維持される。

### 12.3 参考資料（固定版）

**Repository**（すべて `HIROMU1015/Partially-Randomized-Trotter` の固定commit）

- [論文化向け技術調査 README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/README.md)、[paper_claim_sheet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/paper_claim_sheet.md)、[prior_art_method_comparison](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/prior_art_method_comparison.md)、[results_usability_map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/results_usability_map.md)、[minimal_evidence_proposal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/minimal_evidence_proposal.md)、[handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/handoff.md)
- [G6 独立数学監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、[G6 finite-bit・access監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)、[G6 prior-art・method delta](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)
- [ReturnKernel SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py)、[g7_generator.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py)
- [G7結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)、[G8結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md)、[G9 v2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)、[G9契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)、[G10 v3結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)、[G10 claim map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g10_prior_art_and_claim_map_20261010.md)
- [技術・方法ノート v1.0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md)、[G10公開・endpointレビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/research/track_b_G10_v3_publication_and_endpoint_review_20261011.md)
- [GPT G1レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G1_scientific_review_20261009.md)、[GPT G6レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md)
- [Parent allocation closure（D＋E）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)、[資源構造・ボトルネック](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md)
- [S4科学レビュー完成版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md)、[S4 Evidence Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md)

**一次文献**

- K. Aomoto, Y. Kato, Green functions and spectra on free products of cyclic groups, Ann. Inst. Fourier 38(1), 59–85 (1988). <https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf>
- K. Wan, M. Berta, E. T. Campbell, A randomized quantum algorithm for statistical phase estimation, arXiv:2110.12071v2. <https://arxiv.org/abs/2110.12071v2>
- Q. Zhao, X. Yuan, Exploiting anticommutation in Hamiltonian simulation, arXiv:2103.07988v2. <https://arxiv.org/abs/2103.07988v2>
- P. Zeng, J. Sun, L. Jiang, Q. Zhao, Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations, arXiv:2212.04566v2. <https://arxiv.org/abs/2212.04566v2>
- J. Peetz, S. E. Smart, P. Narang, Quantum Simulation via Stochastic Combination of Unitaries, arXiv:2407.21095v2. <https://arxiv.org/abs/2407.21095v2>
- D. Cugini, T. A. Atif, Y. Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1. <https://arxiv.org/abs/2603.13495v1>
- B. Koczor, Sparse Probabilistic Synthesis of Quantum Operations, arXiv:2402.15550v2. <https://arxiv.org/abs/2402.15550v2>
- J. Günther et al., Phase estimation with partially randomized time evolution, arXiv:2503.05647v2. <https://arxiv.org/abs/2503.05647v2>

### 12.4 取得できなかった・精読していない資料

| 資料 | 状態 | 結論への影響 |
|---|---|---|
| 依頼文のG6監査URL（commit `bc649c592c8b2b…`） | 404。1文字違いの正しいcarrier `bc649c592c8c2b…` で取得し、`28cfabb1` と同一bytesを確認 | なし |
| GPT G6／G7レビュー付属selfcheck | 未提供（既存記録どおり） | なし |
| 外部手法の著者SOURCE | 未取得・未実行 | なし |
| S4 Evidence Packet（215,202 bytes） | 見出しのみ。数値はclosureとS4レビューの転記 | 小 |
| S4完成版レビュー（110,900 bytes） | §1–2・§4.5・§5.6と見出しのみ | 小 |
| 資源構造・ボトルネック（33,110 bytes） | 後半は見出し確認のみ | 小 |
| G8結果、G7・G8の既存GPTレビュー、D1–D3個別資料、10/14方向レビュー全文、次アルゴリズムDecision Packet | 冒頭のみ、または未精読（必要な数値は資源構造文書・closureの転記） | 小 |
| Ross–Selinger（arXiv:1403.2975v3）、Kesten・Woessなど古典文献 | 未取得 | なし |
| 追加の文献検索 | 未実施（§8.3） | なし |

---

## 付録：本レビューでの計数と保全

新規科学計算 0、保存値の数値再解析・再ランキング 0、最適化 0、SOURCEのimport・実行・修正 0、compile・合成・sampling simulation 0、新規契約・authorization・grant・marker・claim 0、GitHub上の変更 0。行ったのは、入力パラメータの初等算術（$\chi=19/50$）、保存表示値の間の恒等式成立の確認（§4.7、1件）、紙上の解析的展開・論証のみである。repositoryは読み取り専用cloneとして参照し、既存SOURCE・raw結果・科学契約・レビュー原本・消費済みclaim・STOP・他研究の成果に触れていない。

---

## 改訂履歴

| 版 | 内容 |
|---|---|
| v1（2026-10-18、第2セッション） | 完成版の初版。`レビュー.md` として保持（変更なし） |
| v2（2026-10-18、第3セッション） | ユーザーの自己点検依頼による改訂。結論（C）・確信度・推奨・参照資料は不変。変更点：(a) §4.7にG9 v2保存reference値の恒等式 $m_2^{\rm full}Z=m_2^{\rm P5}$ の成立確認を追加（zero-fillの代価 $1/Z$ の保存証拠）。(b) §7.1(ii)でm=3の $x^4$ 係数を本レビューで再導出し既存係数と一致することを明記（§0.3も同旨に修正）。(c) §7.4に符号の注記と、比較が理想係数質量の $x^2$ 係数に限られる旨・古典前処理費用の注記を追加。(d) §4.8・§6.2に古典online費用の償却（追加queryは確率 $O(x^2)$ の事象に限る）を追記。(e) §5.2の評価に、独立成果として残り得た内容の例を追記（公平性のため）。(f) §7.3第3項に〔仮説〕ラベルを付与。(g) ヘッダ・§2.3・付録に版管理と恒等式確認の計数を反映 |
