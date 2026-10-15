# SOS-PF（候補A）K0・K0.5後の独立科学評価と今後の研究方針：重要科学レビュー（完成版）

**Claude（Anthropic）による独立科学レビュー — 2026-10-16（JST）**

| 項目 | 内容 |
|---|---|
| 依頼文書 | `si.md`「Claude重要科学レビュー依頼：SOS-PF K0・K0.5後の独立科学評価と今後の研究方針」（依頼文書作成日2026-10-16 JST） |
| 開始承認 | `si.md` 添付＋「実行して」 |
| 完了状態 | **完成版**（一回の実行で作成。`WAITING_FOR_SOURCE` に該当する必須資料の欠落なし） |
| 正本（固定commit） | K0：`58d8672702284d73321cc67a26c4640adaddde5f`、K0.5：`01bdc56a2970dcb749390b0540846143f775d6f9`、研究状態・目的：`78b50c515d9df9556f261e74ca7c57af461b7199` |
| 実施したこと | 固定資料の実取得とSHA256照合・本文読了（範囲は§1.3）、Kang–Su arXiv v3の原文取得と関連節の読解、ACM出版版の取得試行（HTTP 403）、K0・K0.5の主要式の紙上検算、追加の紙上導出（区分を明示）、A1/A2/A3の比較 |
| 実施しなかったこと | 新規Hamiltonian生成、SDP求解、固有値・期待値・RDM計算、PF数値シミュレーション、最適化・certificate探索、回路compile、T合成、保存科学数値の再集計、GitHubの変更、候補の採択、K1、正式状態の変更 |
| 主結論 | **第一推奨はA3（SOS-PFを「新しい分割生成アルゴリズム」候補として区切り、成果を保存）**。A2は主テーマとして不推奨。A1は固定規則による一回の反証テストに限り条件付きで検討可（§0） |

---

## 0. 結論の要旨

### 0.1 推奨

1. **【推奨・第一】A3：SOS-PFを「新しいPF-native分割生成アルゴリズム」の候補としては区切り、K0・K0.5の数学的成果を保存する。** これはSOS-PF一般の不可能性の宣言ではない。理由は三つある。
   - (i) K0・K0.5を経ても、既知の構成（SOS Gram SDP、DFの定数shiftによるPSD化、平方完成・一体再配分、Gram因子化・回転、複素Gramの実化）から区別できる独立した生成規則 `R` は特定されていない。自然な候補はいずれも既知SDPの制限か既知操作の変形である（§5）。
   - (ii) 「SOS HamiltonianをPF型の構成で時間発展し、化学のHermitian一体平方と固定粒子数normを使う」という広い構成はKang–Su（arXiv v3）が既に含む（§4.1）。同論文が扱わないのは**SOS gapを使う基底エネルギー特有の保証**であり、これは分割生成（A1）ではなく保証理論（A2）の側の差分である。
   - (iii) 実際の量子資源を左右するのは、分割変更による**実際の固有位相bias**と**native step単価**の二つだけであり（§5.2）、これは静的な文献・数理調査では決まらない。K0.5の結論と同じく、静的調査をこれ以上重ねても判断は変わらない。
2. **【推奨】A2（保証理論への転換）を研究の主テーマにはしない。** 保証理論には、閉じ得る具体的な数学問題が二つ残る（§6.2：gapを要しない基底状態特化の有限step保証、およびgapを用いるがCauchy半径に縛られない有限step保証）。しかし、いずれも同じ回路の実費用を変えず、構成要素は既知で、新規性は限定的である。ユーザーが研究テーマの変更（新アルゴリズム→保証理論）を明示的に受け入れる場合に限り、限定的な理論ノートとして選択肢になる。
3. **【条件付き】A1について、未固定の `R` の探索は行わない。** ただしユーザーが「PF限定・絶対ground energy」というスコープを維持し、かつ「SOSのnear-frustration-free構造がPFの実biasと証明付きstep数に移るか」という一点の事実を判断材料として求める場合に限り、**事前に固定した規則 `R_0` による一回の反証テスト（K1-F、§7.3）** を別承認で検討してよい。目的は継続／停止の判断であり、新規性の確立ではない。結果がどうであれ、その後は自動的に次段へ進まない。
4. **【推奨】研究全体**：次の候補を作る前に、ユーザーが二つのスコープ判断（block-encoding/qubitization除外の扱い、taskを絶対ground energyに限るか）を行い、「狙う費用レバーと目標倍率」を入口条件とする方向選定を一回だけ行う（§7.4）。K0・K0.5で確認済みの論点を繰り返すCodex作業は推奨しない。

### 0.2 本レビューで新たに加えた独立所見

1. **【導出】SOS gapの利益は、PFでもspectral amplificationと同じ形の比で現れる。** K0の一様norm上界どうしを比べると、主要係数の上界比は $K_{\rm SOS}E_g/C_{\rm raw}\approx\frac98\,E_g/\Lambda$（$\Lambda$ はfragment norm総和）になり、gapを用いるbias経路での証明付きstep数は $\sqrt{E_g/\Lambda}$ 倍になる。これはSOSSAの $\Lambda\to\sqrt{2\Lambda E_{\rm gap}}$ と平行する（§3.4）。ただし比較相手は最も弱い状態非依存上界であり、実誤差の比ではない。
2. **【導出・概略】gapを要しない基底状態特化の保証では、利益は $E_g$ の平方根でしか効かない。** spectral weight補題と基底状態residualの評価を組み合わせると、証明付きstep数は状態非依存の同種評価に対し $(E_g/\Lambda)^{1/4}$ 倍、gapを用いるbias経路に対し約 $(\alpha' E_g/\Lambda)^{-1/4}$ 倍（$\alpha'$ は失敗確率）となる（§3.8）。**gap証拠の有無で、SOSの利益は二乗根だけ変わる**。
3. **【文献】Kang–Su v3は、自らのSOS simulationの費用を、gap amplification法（Low et al.）とは「比較不能」と述べている**（§1.6、本レビューで原文確認）。同論文のSOS simulationは全stateに対するcommutator scalingであり、SOS gapを使っていない（§4.1）。
4. **【導出】有用領域の窓**：証明付き区間 $[L,U]$ で $G=U-L\le2\varepsilon$ なら古典的に問題は解けており、$G\gtrsim\Lambda$ なら利益がない。SOS-PFが意味を持つのは $2\varepsilon<G\ll\Lambda$ の場合に限られる。また $G$ はQPEのalias解消に必要なエネルギー窓も与え、その条件 $hG<2\pi$ はCauchy半径 $h<\log(3/2)/\Lambda$ よりはるかに緩い（§3.6、§3.8）。
5. **【判断】実際の量子資源のレバーは「実bias」と「native step単価」の二つに帰着する**（§5.2）。前者は分割を変えたときのground-state揺らぎ、後者はfragment数・幅・実化で決まり、どちらも計算なしには決まらない。
6. **【訂正】前回のClaudeレビュー**（同一会話で作成した前回の `レビュー.md`）には、二分割の係数の簡約漏れ、Gram実化に伴うfragment数の誤り、Kang–Suの見落とし、候補Bの粗い評価があった。訂正を§9にまとめる。

### 0.3 判断の分離（問F）

| 判断 | 結論 | 確度 |
|---|---|---|
| K0・K0.5の主要式は数学として妥当か | 二分割 $d=a/12\ge0$、多fragmentの辞書と $K_{\rm SOS}$ 行和上界、条件付き有限h式、Gram実化のrank、定数shiftの回路不変性はいずれも成立（§3） | 高（紙上検算。一様上界の一部は未検算） |
| 広い「SOS＋PF」構成は先行研究と異なるか | 異ならない（Kang–Su §1.6・§6） | 高 |
| SOS gapを用いる化学PFのground-energy保証は先行研究と異なるか | 検索・読了範囲では未発見。ただし構成要素は既知で、初等的な組合せ | 中 |
| 学術的価値はあるか | 保証理論の限定的ノートとしてはあり得る。新アルゴリズムとしては未確立 | 中〜低 |
| 量子計算資源が減るか | **判定不能**（実bias・native単価が未知。計算が必要） | — |
| 研究として採択すべきか | 現時点では採択しない（A3） | 中〜高 |

---

## 1. 対象・承認・資料・完成状態

### 1.1 依頼範囲と遵守事項

si.md §4.1に従い、本書は重要科学レビューのみである。新しい科学計算・科学的再集計（Hamiltonian生成、SDP、固有値・RDM、PF数値シミュレーション、optimizer探索、compile、T合成、保存数値の再集計・再ランキング）、GitHubのcommit・push、科学source・結果・method status・manifestの変更、候補の正式採択、K1、PAUSED/CLOSED/STOP/HOLDの解除・変更をいずれも行っていない。本書の推奨はユーザーの判断材料であり、作業指示や実行認可ではない。

### 1.2 共通指示・テンプレート

`研究レビュー依頼テンプレート.md`（2026-10-15版）と `research_project_instructions_gpt_codex_review_approval_2026-10-12.md` は本セッションに添付されていない（si.mdも自動提供を仮定しないと明記）。したがって原文は確認しておらず、si.mdの記述（特にチャット出力規則）に従った。矛盾がある場合は最新版共通指示と当該依頼でのユーザー明示指示が優先される。

### 1.3 取得した資料と読了範囲

すべてraw.githubusercontent.comから固定commit指定で取得し、ローカルでSHA256を計算した。K0.5の必須3資料のSHA256は、公開provenanceに記録された値と一致した。

| 区分 | 資料 | commit | path | bytes | SHA256 | 読了範囲 |
|---|---|---|---|---|---|---|
| 必須A（K0） | K0資料索引 | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/README.md` | 3,989 | `b512ca23a18eb81d13ddfe8c8ef5abe3685a829c260b1da58e6e6ef757719561` | 全文 |
| 必須A（K0） | K0技術監査 | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/technical_audit.md` | 32,094 | `6ba1ae010b24874368c9bac695626aad75157bb659718dfa859f5c000b28e2b7` | 全文 |
| 必須A（K0） | K0文献registry | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/literature_registry.json` | 14,642 | `9d93a0b88c6d8ca3203882a64ec3b45db1d91f5c0b940270893c7e98fd4217fa` | 全エントリ（章節・仮定・関係・限界） |
| 必須A（K0） | K0 method status | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/method_status.json` | 5,034 | `2163efc7074968ffbe9f0b7969bd5270262ce5f9b35f49f9b8fb2ec0e7434042` | 全文 |
| 必須A（K0） | K0 GPT向け引継ぎ | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/gpt_handoff.md` | 6,028 | `11134bf1801fcc9496abc00141937252aa827b450f2e3138e77bd76f456d75c2` | 全文 |
| 必須A（K0） | 未認可K1草案 | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/k1_design_draft.md` | 9,783 | `d92dec4b5d2b4cf747f5d6d30bc4c5d06d13211b0940dd995cb91432d0581cf4` | 全文 |
| 必須A（K0） | K0 provenance | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/provenance.json` | 10,492 | `ec16fb828da66818f1ae34a412619e9ed9363a5d812d9a4f2d45196f1ea9396d` | 取得のみ（本文は未精読） |
| 必須A（K0） | K0に渡された前回GPT独立レビュー | `58d86727` | `docs/research/qc_sos_pf_k0_20261015/supplied_gpt_review.md` | 41,261 | `aaab84272fd1e2d1dca8d92ddb191e78d5daca3aeff896fff88e8eed303b224d` | 全文 |
| 必須A（K0） | K0.5追加技術調査報告 | `01bdc56a` | `docs/research/qc_sos_pf_k05_20261015/technical_report.md` | 21,581 | `99f00a89fb8fbd5b157823166f5ad46a718074b76e452fa94a92b612db8c0a10` | 全文 |
| 必須A（K0） | K0.5追加文献対応表 | `01bdc56a` | `docs/research/qc_sos_pf_k05_20261015/literature_correspondence.md` | 14,162 | `f073ad907ecb8ff479ecbbc9bce7b83887efb14c9201ad86f8b4f0f9e71c0bf1` | 全文 |
| 必須A（K0） | K0.5 GPT向け判断資料 | `01bdc56a` | `docs/research/qc_sos_pf_k05_20261015/gpt_decision_handoff.md` | 8,309 | `13b4a9323271f1065f4281982fd1623409bfa6de4fd3cf92cfc6bb350b4ed5c8` | 全文 |
| 必須A（K0） | K0.5資料索引 | `01bdc56a` | `docs/research/qc_sos_pf_k05_20261015/README.md` | 3,632 | `95d9260cf0e04f7560f2938cd8b243a4735f090f2d61c695e0870f5575a7c9f7` | 全文 |
| 必須A（K0） | K0.5公開provenance | `01bdc56a` | `docs/research/qc_sos_pf_k05_20261015/provenance.json` | 10,522 | `401498d23fd2fc7748824f5996bc85dce483fde550184b60e4cb6819cac9a2af` | 必須3資料のSHA256・公開変更点の部分 |
| 必須C | 正式採用済みの研究状態 | `78b50c51` | `artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json` | 3,825 | `46b8835d849ddd7abd75a5146840eaa8f06ef0034198c2e97f139a797728f824` | 全文（前回作成時にも読了） |
| 必須C | 研究目的・研究課題 | `78b50c51` | `docs/research/研究目的・研究課題.md` | 65,154 | `c43a7bb22a077a3de3022b021cca4b78a2e3cddd4a930b059b24930d80076772` | 取得・SHA照合のみ（本文は同一会話の前回作成時に関連節を読了） |
| 必須C | C1〜C4・Track A/B等の統合引継ぎ | `78b50c51` | `docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md` | 31,677 | `8c81a505ae8e86dbd3d890e044e9c3da29fbc18b1def661121e132bcef527573` | 取得・SHA照合のみ（本文は前回作成時に読了） |
| 補助 | M案後のGPT科学再評価 | `78b50c51` | `docs/reviews/quantum_chemistry_pf/qc_m_k0_claude_scientific_rereview_and_direction_20261015.md` | 38,976 | `2fdd740fe998c6f279034e723197288697f1c5e01573e8a78a0a21eb2b32f086` | 取得・SHA照合のみ（本文は前回作成時に読了） |

**外部一次文献（本レビューで原文を取得したもの）**

| 文献 | 取得結果 | 読了範囲 |
|---|---|---|
| Kang–Su, *Quantum Matrix Arithmetics with Hamiltonian Evolution*, arXiv:2510.06316v3（v1 2025-10-07、v2 2026-01-22、v3 2026-09-01） | HTML 2,530,653 bytes、SHA256 `34e062fcbcd44980eabceef94c583fcc6e1e5d66fc678ebb1dab358ea5b14313`。PDF 1,014,702 bytes、SHA256 `d24877bad57e8f797c609c7f6917d3ad50dae6e8583e11f250941a5b29ec4573`。いずれもK0.5の記録値と一致 | 目次、§1.6本文、Corollary 27・29の主張、§6.3冒頭、Lemma 7の主張、Lemma 43の主張、ground energyへの言及箇所。全証明・Lemma 9 Eq.74・Eqs.271–274は未読 |
| 同 ACM Trans. Quantum Comput. 7(4), 1–86（DOI 10.1145/3816436、online 2026-08-13） | DOI landing・PDFとも**HTTP 403**。Crossref metadata（CC BY 4.0表記）は取得。Unpaywallは `is_oa=true`・publishedVersionと返すがPDF URLは空 | **本文未取得**。arXiv v3との差分は**未確認（UNVERIFIED）** |

その他の文献（Hejazi ×2、Mizuta–Kuwahara、Yi–Crosson、Low、King、Rubin–Low–DePrince、Casares、Childs et al.）は、K0・K0.5の文献台帳・対応表の記録と、同一会話の前回レビュー作成時に確認した要旨・本文一部に依拠し、今回は原文を再取得していない（§4.2で区別する）。今回の検索で新たに2026年のSOS spectral amplification関連2件（第一量子化への適用 arXiv:2607.15358、guiding state付きground-energy推定の最適下界 arXiv:2608.24493）を検索抜粋の範囲で確認した。

**未取得・注意事項**

- Claudeの前回レビュー `レビューouter.md`：si.mdの指示どおり、本セッションで取得できると仮定しない。ただし本セッションの出力領域に、同一会話で前回Claudeが作成した `レビュー.md`（92,629 bytes）が残っていた。改行数は745（末尾行の数え方次第でGPTレビューの「746行」と整合し得る）、SHA256は `6057cdafea9d51bcba28e05fb3a71d92e41012556418b6d8483f263346a90177` である。ただしユーザーによる改名・編集の有無は確認できないため、**`レビューouter.md` との同一性は未確認**として扱う。前回文書は本完成版の保存前に作業領域へ退避した。§9の訂正はこの前回作成文書に対するものである。なお、本完成版の保存により出力領域の同名ファイルは置き換わる。
- ACM出版版本文：上表のとおり未取得。
- K0・K0.5原本中の「ローカルのみ」「commit/push 0件」等は各調査完了時点の履歴であり、現在の未公開を意味しない（si.md §2.2の指示どおりに読む）。

---

## 2. 研究目的・研究全体の到達点・正式状態

### 2.1 目的

元の有限基底電子Hamiltonianのground energyを、**同じ目標精度・成功確率・合法な情報アクセス条件**で、既存の強い方法より低費用で求めることである（si.md §1）。誤差係数や上界の改善だけでは、この目的の達成とみなさない。

### 2.2 到達点と正式状態（事実）

| 系列 | 正式状態・結果 |
|---|---|
| 第一候補（一体項再配分PF） | 現固定構成 `CLOSED`。H4で登録10%未達、同情報compact BCHへ `REDUCES_TO_KNOWN` |
| H-E1 | 現定義の探索終了 |
| M-a/M-b | 追加探索 `STOP`、K1未実施 |
| PF小改変探索系列 | `PAUSED` |
| 旧D案（DF因子の等価回転） | `HOLD` |
| Track A/B、第1・第2研究、N1〜N3、THRIFT、C1〜C4 | 各既存状態のまま（変更なし） |
| **SOS-PF（候補A）** | 未採択。K0：`UNDETERMINED_DATA_REQUIRED`、`K0_PROTOTYPE_CANDIDATE=false`、`overall_no_go_theorem=false`。K0.5：完了、判定変更なし。K1：未認可・未実施。新規科学計算は0件 |

### 2.3 SOS-PF系列が示したこと・示していないこと

- **示したこと**：PSD fragmentの基底状態分散がSOS gapで抑えられること、二分割S2の主要ground係数が非負であること、多fragmentの明示的な係数上界、条件付きの有限step式、Hermitian一体平方の回路接続、Gram実化のrank増、定数shiftだけでは回路・実bias・overlapが不変であること、Kang–Suという近接先行研究の存在、gapを要しない既知の保証経路の整理。
- **示していないこと**：独立した有利な分割生成規則、化学で有用な有限step保証、同じ情報条件の強い対照に対する総費用改善。

---

## 3. 問A：K0・K0.5の数学的主張の独立確認

### 3.1 設定（K0 §3.1の記号）

対象sector上で $H'=H-E_{\rm SOS}I=\sum_{j=1}^MF_j$、$0\preceq F_j\preceq f_jI$、$H'|0\rangle=E_g|0\rangle$、$\phi_j=\langle F_j\rangle$、$\sigma_j^2=\langle F_j^2\rangle-\phi_j^2$、$\delta F_j=F_j-\phi_jI$、$w_j=\|\delta F_j\|\le f_j$、$\Lambda=\sum_jf_j$。$E_g$ はSOS下界からのoffsetであり、励起gap $g_0=E_1-E_0$ とは別の量である。

### 3.2 PSD分散（K0 Eq.1）：成立・既知

$F_j^2\preceq f_jF_j$ から $\sigma_j^2\le f_j\phi_j-\phi_j^2\le f_j\phi_j$、$\sum_j\phi_j=E_g$。初等的帰結で、独立新規性はない（K0・GPTレビューと同判断）。

### 3.3 対称BCHの規約と二分割係数（K0 Eqs.2–4）：成立

$\log(e^{X/2}e^Ye^{X/2})=X+Y-\tfrac1{24}[X,[X,Y]]+\tfrac1{12}[Y,[Y,X]]+O(5)$ は標準の対称BCH（Yoshida型の $\alpha_3$）と一致する。$X=-ihA$、$Y=-ihB$ で三次項に $(-i)^3=i$ が付くので、$S_{ABA}=\exp\{-ih[H'+h^2D+O(h^4)]\}$、$D=\tfrac1{24}[A,[A,B]]-\tfrac1{12}[B,[B,A]]$ となる。符号を独立に追って確認した。

$(H'-E_g)|0\rangle=0$ より $(\delta A+\delta B)|0\rangle=0$、すなわち $\delta B|0\rangle=-\delta A|0\rangle$。よって $a:=\langle\delta A(H'-E_g)\delta A\rangle=\langle\delta B(H'-E_g)\delta B\rangle$ で、$\langle[A,[A,B]]\rangle=\langle[B,[B,A]]\rangle=-2a$、

$$
d=\langle D\rangle=-\tfrac{a}{12}+\tfrac{a}{6}=\tfrac{a}{12}\ \ge0 .
$$

**判定**：成立。非負性はPSD分割を必要とせず、非縮退ground branchなら一般の二分割で成り立つ（$H'-E_g\succeq0$ だけを使う）。すなわち二分割の対称S2では、主要次数でground固有値は上へずれる。PSDを使う大きさの評価 $d\le\frac{W}{12}\frac{f_Af_B}{f_A+f_B}E_g$（$W=\|H'-E_g\|\le f_A+f_B$）は、さらに簡約すると $d\le f_Af_BE_g/12$ となる。縮退groundでは $P_0DP_0$ を部分空間内で扱う必要がある（K0の記述どおり）。この非負性の名前付きの既報は今回確認していないが、標準的な一次摂動の直接帰結であり、独立成果とはしない。多fragmentへ符号を移せないことはK0の注意どおりである。

### 3.4 多fragmentの係数辞書と $K_{\rm SOS}$（K0 Eqs.5–10）：成立、ならびに比較の尺度（追加）

- **辞書（Eqs.5–6）**：$D=\sum_{j<M}\{\tfrac1{24}[F_j,[F_j,R_j]]-\tfrac1{12}[R_j,[R_j,F_j]]\}$、$R_j=\sum_{k>j}F_k$ は、外側から対称BCHを順に適用する再帰（内側S2の誤差はこの次数で加算）と一致する。raw項数 $\sum_{r=1}^{M-1}(r+r^2)=(M^3-M)/3$、$\sum|\kappa|=\sum_r(r/24+r^2/12)=M(M-1)(4M+1)/144$ を独立に検算した。
- **行和上界（Eqs.7–9）**：四つの語を両端のベクトルnormで評価し、非負対称行列 $B_{\rm SOS}$ の二次形式 $x^{\mathsf T}Bx$（$x_j=\sqrt{\phi_j}$、$\|x\|^2=E_g$）に入れ、最大行和でspectral normを抑える論法は正しい。同一添字の対角項の二重加算も、元の評価式と整合することを確認した。
- **一様上界（Eq.10）**：$K_{\rm SOS}\le(M^2-1)f^2/8$ のうち、前半（$[F_j,[F_j,R_j]]$ 由来の行和 $\le(M-1)f^2/8$）は確認した。後半の行和式は今回独立に再導出していない。

**【導出】比較の尺度**：状態非依存で最も素朴な上界は $|d|\le\|D\|\le4\sum|\kappa|f^3=:C_{\rm raw}=M(M-1)(4M+1)f^3/36$ である。一様normの下で

$$
\frac{K_{\rm SOS}E_g}{C_{\rm raw}}\le\frac{9(M+1)}{2M(4M+1)}\cdot\frac{E_g}{f}\ \xrightarrow{M\gg1}\ \frac98\,\frac{E_g}{Mf}=\frac98\,\frac{E_g}{\Lambda}.
$$

$p=2$ では証明付きstep数は係数の平方根に比例するので、gapを用いるbias経路では $\sqrt{E_g/\Lambda}$ 倍程度となる。これはSOS spectral amplificationが費用を $\Lambda$ から $\sqrt{2\Lambda E_{\rm gap}}$ に下げる構造と平行している。行和上界が $\sum_j\phi_j=E_g$ という**予算の共有**を使うため、素朴な三重和より一桁 $M$ 分有利になる点が鍵である。

ただし次の限定が付く。(a) 両辺とも一様normの最悪評価であり、実際の $K_{\rm SOS}$ や $\|D\|$ とは大きく異なり得る。(b) 公平な状態非依存の対照は $C_{\rm raw}$ ではなく、sector内の実際の入れ子交換子norm、Monte Carlo上界（Blunt et al.）、Childs et al.の明示S2上界であり、これらに対する比はより不利になり得る。(c) $E_g$ は実際に用いるPF-native分割のgap（$G=U-L$）でなければならない。(d) これは漸近係数の比であり、有限stepの保証（§3.6、§3.8）や実誤差の比ではない。

### 3.5 $h^4$ 項・overlap（K0 Eqs.11–12）：条件付きで成立

非縮退・孤立groundの摂動展開と、語の右端に $\|F_j|0\rangle\|\le\sqrt{f_jE_g}$ を使う評価規則（$\|QD|0\rangle\|\le B_D\sqrt{E_g}$、$|\langle T\rangle|\le A_TE_g$）は正しい。$h^4$ 係数の二項に相殺があるため、上界の改善から実残差の減少を結論しない、というK0の注意も妥当である。

### 3.6 有限stepのCauchy＋Schur式（K0 Eqs.13–17）：論理は成立、実用性は構造的に乏しい

確認した論法：$|z|\le R=\log(3/2)/\Lambda$ で $\|S_2(z)-I\|\le e^{|z|\Lambda}-1\le1/2$、主logのnorm $\le\log2$、$H_{\rm eff}(z)=i\log S_2(z)/z$ は偶関数で解析的、円周上の上界 $M_R=\log2/R$ とCauchy係数評価から $r_4(h)$（Eq.13）。Weylで最下固有値が $E_g$ から $v=h^2N_D+r_4$ 以内、$Q$ block最小値は $E_g+g-v$ 以上、ゆえに距離 $g-2v$、Schur補元から Eq.15、overlap Eq.17。いずれも正しい。$\|hH_{\rm eff}\|\le\log2<\pi$ なので、この領域では位相の折り返しも生じない。

抜け・限界として次を確認した。

| 点 | 内容 |
|---|---|
| 解析半径 | $R\sim1/\Lambda$（全fragment norm和）。化学では実用的な $h$ よりはるかに小さい。fragmentを中心化（$\Vert F-c\Vert $ を最小化）すれば半径は高々2倍程度しか改善しない |
| 剰余 | $r_4\sim(\log2/\log^5\tfrac32)\Lambda^5h^4$ はSOS gapに比例しない。$E_g$ が小さいほど剰余が相対的に支配する（K0.5 §5と同判断） |
| gap証拠 | Schur部分は認証済み励起gap下界 $g$ を要する。ground用のSOS下界 $L$ は $E_1$ の下界を与えない |
| 縮退・対称性 | spin多重項等ではground部分空間全体で扱う。sectorごとの保存則の確認が必要 |
| スカラー位相 | $E_{\rm SOS}$ の位相を最後に復元する処理は正しい。制御版では相対位相として扱う |

**判定**：数学的に誤りはない。しかし半径・剰余がglobal normに支配されるため、この経路から化学で有用な証明付きstepは出ない。これは定数の粗さではなく、**複素解析的な展開を全演算子に対して行う手法そのものの限界**である。有用な有限step保証を狙うなら、実時間の誤差表現を基底状態に直接作用させる別経路が必要である（§3.8）。

### 3.7 K0.5のgapを要しない保証：何を上界づけているか

| 経路 | 上界の対象 | 励起gap | 基底状態特化 | 有限h | 評価 |
|---|---|---|---|---|---|
| 逆log＋Weyl（Kang–Su Lemma 7） | $H_{\rm eff}$ と $H'$ のoperator差、したがって全固有値 | 不要 | なし | normの小ささ条件下 | 既知。状態非依存で弱い |
| QPE出力分布の安定性（telescoping） | 出力分布の差 $\le\sum_q\varepsilon_q$ | 不要 | なし | 可 | 既知。step数に比例して誤差が蓄積し、最も弱い |
| Childs et al. 明示S2上界 | $\Vert S_2(h)-e^{-ihH'}\Vert \le h^3C_2$ | 不要 | なし | 全ての実 $h$ | 既知の強い対照 |
| spectral weight補題（K0.5 §3.3） | 目標位相から離れたPF固有位相の重み $\le\eta_s^2/[4\sin^2(\omega/2)]$ | 不要 | $\eta_s$ 次第 | 可 | 既知の補題。**$\eta_s$ を基底状態特化で抑えれば有用**（§3.8） |
| Cauchy＋Schur（K0） | 特定branchのbias・overlap | 必要 | あり（主項） | 半径 $\sim1/\Lambda$ 内 | §3.6 |

K0.5の整理（gap証拠がないことはground-energy task全体の不可能性を意味しない）は正しい。

### 3.8 【導出・概略】gapを要しない基底状態特化の経路と、gapの価値

**(a) QPEへの接続**：理想QPE（$U=S_2(h)$ の正確な冪）に厳密groundを入力すると、出力位相は $U$ の固有位相が $|0\rangle$ のspectral measureに従って分布する。spectral weight補題により、推定エネルギーが目標から $\varepsilon_b$ 以上離れる確率は

$$
P_{\rm far}\le\frac{\eta_s(h)^2}{4\sin^2(h\varepsilon_b/2)}+(\text{読出し由来の失敗}),\qquad \eta_s(h)=\|(S_2(h)-e^{-ihH'})|0\rangle\|
$$

で抑えられる。冪の回数に依存せず（同じ固有分解を読むため）、励起gapも単一branchの追跡も要らない。目標近傍の固有位相が複数あっても、エネルギーの精度には影響しない。aliasの解消には事前のエネルギー窓が要るが、SOS下界と変分上界の区間 $[L,U]$（幅 $G$）が窓を与え、条件 $h(G+2\varepsilon_b)<2\pi$ はCauchy半径の条件よりはるかに緩い。

**(b) 残差の基底状態特化評価（方針）**：S2の実時間の厳密な誤差表現（Childs et al.型）は、各誤差項を「ユニタリ×（共役された）入れ子交換子×ユニタリ」の積分で表す。これを $|0\rangle$ に作用させると、右端のユニタリ $V$（fragment指数の部分積）は $\|(V-e^{i\chi})|0\rangle\|\le h\sum_l\tau_l\sigma_l+\dots$ で位相に近く、入れ子交換子は定数を除いた $[\delta F_a,[\delta F_b,\delta F_c]]|0\rangle$ の形になり、各語の右端が $\delta F|0\rangle$（norm $\sigma$）となる。したがって

$$
\eta_s(h)\le h^3\Gamma_3+h^4\Gamma_4+\cdots,\qquad \Gamma_k\ \text{は}\ \sigma_j\ \text{に線形}
$$

という形が得られ、$\sigma_j\le\sqrt{f_j\phi_j}$ とCauchy–Schwarz（$\sum\phi_j=E_g$）から $\Gamma_3\le\hat\Gamma_3\sqrt{E_g}$ となる。解析半径の制約はない。ただし**剰余項の明示定数は今回導出していない**。

**(c) スケーリングの目安**（一様norm、最悪評価）：$\Gamma_3\sim M^{5/2}f^{5/2}\sqrt{E_g}/12$、状態非依存の $C_2\sim M^3f^3/9$ より

| 経路 | 証明付き $h^2$ の条件 | 状態非依存の同種評価に対するstep数比 | 必要な証拠 |
|---|---|---|---|
| gap不要・基底状態特化（本節） | $h^2\lesssim\varepsilon_b\sqrt{\alpha'}/\Gamma_3$ | 約 $(E_g/\Lambda)^{1/4}$ | $G=U-L$、fragment norm |
| gapを用いるbias（§3.4） | $h^2\lesssim\varepsilon_b/(K_{\rm SOS}E_g)$（有限hの妥当性が別途必要） | 約 $(E_g/\Lambda)^{1/2}$ | 上記＋認証済み励起gap |

両者の比は約 $(\alpha'E_g/\Lambda)^{-1/4}$ で、例えば $E_g/\Lambda=10^{-3}$、$\alpha'=10^{-2}$ ならgap不要経路は約18倍のstepを要し、状態非依存の同種評価よりは約5.6倍少ない（記号上の例示で、分子の計算値ではない）。**gap証拠の有無で、SOS gapの利益は二乗根だけ変わる。** gap不要の評価が $\sqrt{E_g}$ でしか効かないのは、残差が固有ベクトルの混合（エネルギーには無害）と位相のずれを区別できないためである。

**(d) gapを用い、Cauchy半径に縛られない経路（方針のみ）**：$\langle0|S_2(h)|0\rangle e^{ihE_g}=1-ih^3d+R(h)$ の剰余 $R$ を実時間表現で評価すると、両端に $\delta F$ を持つ語だけが残り $O(E_g)$ と見込める。これを認証済みgapとKato–Temple型の位相版不等式に組み合わせれば、$|E(h)-E_0|\le h^2K_{\rm SOS}E_g+O(h^3E_g)+O(h^4E_g/g)$ 型の有限step保証が得られる可能性がある。未導出であり、gap証拠の取得が最大の難所である。

### 3.9 PF-native SOS・sector・定数shift（K0 §§6.1–6.4）：成立

- **Gram実化**：$G=CC^\dagger$、$C=A+iB$ とすると $G=AA^{\mathsf T}+BB^{\mathsf T}+i(BA^{\mathsf T}-AB^{\mathsf T})$。ゆえに $\operatorname{Re}G=AA^{\mathsf T}+BB^{\mathsf T}$、$\operatorname{rank}\operatorname{Re}G\le\min(m,2\operatorname{rank}G)$。例 $G=\begin{psmallmatrix}1&i\\-i&1\end{psmallmatrix}$（rank 1、$\operatorname{Re}G=I$ はrank 2）も確認した。
- **残余一体項**：$\operatorname{Im}G$ は実反対称で、$i\sum_{\mu<\nu}(\operatorname{Im}G)_{\mu\nu}[E_\mu,E_\nu]$、$[a^\dagger ua,a^\dagger va]=a^\dagger[u,v]a$ からHermitianな一体演算子になる。そのPSD化は下界を悪化させ得る。
- **最初から実対称Gram**をHermitian一体基底上で使えば（PF-native SOSを直接定式化すれば）、平方数は $\operatorname{rank}S$ で残余は生じない。ただし最適値は一般の複素SOSより悪くなり得る（K0の記述どおり）。
- **sectorと理想項**：粒子数を保存する $f$ についての $f(\hat N-\eta)+(\hat N-\eta)f^\dagger$ は圧縮で0になり、各fragmentがsectorを保存すればPFから省ける。spin固定時は各fragmentの保存則の確認が必要。
- **定数shiftの不変性（K0 Eq.19）**：$e^{-ih\sum_j\alpha_j}S_2(h;\{F_j\})=S_2(h;\{H_j\})$。正しい。**DFの定数shiftによるPSD化は、同じ回路に対する保証の書き換えであり、実bias・overlapを変えない。**

### 3.10 問Aのまとめ

| 主張 | 判定 | 区分 |
|---|---|---|
| PSD分散補題 | 成立 | 既知の初等結果 |
| 対称BCHの規約・$D$ の符号 | 成立 | 標準 |
| 二分割 $d=a/12\ge0$、$d\le f_Af_BE_g/12$ | 成立 | 標準摂動の直接帰結（名前付き既報は未確認） |
| 多fragment辞書・項数・$\sum\vert \kappa\vert $ | 成立（検算） | 標準 |
| $K_{\rm SOS}$ 行和上界 | 成立 | 初等的 |
| 一様上界 Eq.10 | 前半のみ検算 | — |
| $K_{\rm SOS}E_g/C_{\rm raw}\approx\frac98E_g/\Lambda$ | 本レビューの導出 | 一様normの目安 |
| Cauchy＋Schurの有限h式 | 論理は成立。実用性は構造的に乏しい | 標準手法の組合せ |
| gap不要の各保証 | 成立 | 既知 |
| gap不要・基底状態特化の残差経路 | 主項の形のみ導出（剰余の定数は未導出） | 本レビューの方針 |
| Gram実化のrank・残余一体項 | 成立 | 線形代数 |
| 定数shiftの回路不変性 | 成立 | 恒等式 |

---

## 4. 問B：先行研究との具体的な差分

### 4.1 Kang–Su（arXiv:2510.06316v3）：本レビューで原文を確認した範囲

| 箇所 | 原文で確認した内容（要約） | SOS-PFとの関係 |
|---|---|---|
| §1.6 | 行列算術の手法を量子simulationに適用すると、従来PFのcommutator scalingを受け継ぎつつ各Trotter stepの実装を改善できる。SOS Hamiltonian $H=\sum_k(\sum_jA_{jk})^\dagger(\sum_jA_{jk})$ を、要素のHamiltonian block encodingから、和（PF）→積（群commutator）→外側の和（PF）で構成し、結果は制御つき時間発展としてQPEに使える | 「SOSをPF型の構成で時間発展し、QPEに使う」という広い構成は既知 |
| §1.6（化学への適用の評価） | 化学SOSのsimulation費用は、qubitization系より良いが、gap amplification法（Low et al.、arXiv:2502.15882）とは**比較不能**と述べている | SOS gapを使う低エネルギーの利得は同論文の評価に含まれない |
| Corollary 27 | 一般SOSの制御つき時間発展を精度 $\varepsilon$ で、要素の制御つきHamiltonian block encodingへの問い合わせ $(\alpha_{\rm comm}+\alpha_\infty)t(n_K\alpha_\infty t/\varepsilon)^{o(1)}n_Kn_J$ 回で構成 | 全stateに対するoperator誤差。SOS offset・励起gapへの依存はない |
| Corollary 29 | $\Vert H\Vert \le1$ のHermitian $H$ について、$H^2$ のHamiltonian block encodingを $O(\log(1/\varepsilon))$ 回の問い合わせで構成（Hamiltonian squaring） | 平方の指数化をQSVT型で行う。SOS-PFの「軌道回転＋対角二次位相」による直接指数化とは実装経路が異なる |
| §6.3冒頭 | 化学SOSをLow et al.の形 $H=\sum_{r,c}\mathrm{Quad}^2(W_{rc})$（DFTHC型）で扱い、一体項と平方内のidentity shiftを省く。Quadの指数をGivens回転 $O(n)$ 個で、平方をCorollary 29で実装し、粒子数sectorのseminormで入れ子交換子を評価 | **gap最適化されたLow et al.のSOS形をPF型simulationの入力に使う点まで既知**。ただしgapを誤差評価に使っていない |
| Lemma 7 | $\Vert e^{-iJ}-e^{-iK}\Vert \le\Vert J-K\Vert $、逆向きはnorm条件の下で成立 | K0.5のgap不要経路の部品 |
| Lemma 43（付録C.2） | 正規行列 $J$ について $\max_{\psi_\eta}\vert \langle\mathrm{Quad}(J)\rangle\vert =\max_{\vert S\vert =\eta}\vert \sum_{k\in S}\lambda_k(J)\vert $ | 固定粒子数normの評価は既知 |
| ground energy | 応用先としての言及、Green関数推定（§5.2）でground stateとenergyを入力とする | SOS gap依存のground-energy保証は与えていない |

**アクセスモデル**：Kang–Suの「Hamiltonian block encoding」は、拡大したHermitian演算子 $\begin{psmallmatrix}0&A^\dagger\\A&0\end{psmallmatrix}$ の時間発展（制御つき）を基本アクセスとし、最大2個（化学では1個）の補助qubitを使う。通常のunitary block encodingとは異なるが、QSVT型の多項式変換を用いる。従来の研究範囲（block-encoding/qubitization中心の方法を対象外）との関係はユーザーの判断事項であり、本書は比較対象として参照するにとどめる。

**K0.5が指摘した有限定数の注意**（§6.3で平方内のidentity shiftを省くと交差一体項が残ること、Eq.271→274のanticommutator係数）は、本レビューでは当該式を再導出しておらず、**未検証**として引き継ぐ。論文全体の誤りの認定とはしない。

**出版版**：ACM版（7(4), 1–86、online 2026-08-13）はDOI・PDFともHTTP 403で本文未取得。Crossref・Unpaywallのmetadataは取得したが、本文同一性は推論しない。v3（2026-09-01）は出版online日より後の版であり、**v3と出版版の数学的・アルゴリズム的差分は未確認**である。

### 4.2 その他の主要文献との対応

（今回原文を再取得していないものは、K0台帳・K0.5対応表の記録と、同一会話の前回レビュー作成時の確認に依拠する。）

| 文献 | 主な結果・仮定 | SOS-PFとの関係：既知の部分 | 残り得る差分 |
|---|---|---|---|
| Hejazi–Soni–Shokrian Zini–Arrazola（arXiv:2412.16811） | 任意次数PFの固有値摂動、外側がHの入れ子交換子はenergy誤差に寄与しない、2項custom PF、低energy節（局所・PSD・有界degree）。対象固有値のgapを全経路で仮定 | 主要係数を期待値で評価する方法、低energy評価の構想 | 非局所化学でのSOS gap比例の評価、gap不要の有限h保証 |
| Hejazi–Shokrian Zini–Arrazola（arXiv:2402.10362） | 低energy部分空間に制限したnormと漏洩評価（局所、PSD特化あり） | 制限norm・漏洩の証明構造 | 密な化学平方へ有利な定数を移すこと |
| Mizuta–Kuwahara（PRL 135, 130602） | PSD・k-local・extensiveな項で、射影した入れ子交換子がエネルギーに一次（補足Theorem S1）、有限時間の低energy誤差 | **「PSD＋低energyで誤差がエネルギーに一次」という原理は既知** | 非局所・全supportの平方での有用な定数 |
| Yi–Crosson（npj QI 8, 37） | QPEでの固有値・固有ベクトル摂動、gap条件 | 個別branch解析 | gap証拠の取得法は与えない |
| Low et al.（PRX 15, 041016） | 化学SOS、Gram SDP、DFTHC、SAで $\Lambda\to\sqrt{2\Lambda E_{\rm gap}}$ | 化学SOSの構成、gap最適化 | PFでのgap利用（block-encoding側の結果） |
| King et al.（PRL 136, 110601） | SOSSA、termwise SA、energy／phase推定 | SOSと低energy利得の枠組み | PF固有の定理ではない |
| Rubin–Low–DePrince（JCTC 22, 8751） | weighted SOS、v2RDM双対、粒子数ideal、BLISS | SOS生成、sector ideal | PF-native閉包・費用保存は与えない |
| Casares et al.（arXiv:2606.30741） | SPRINT/GRADE、近可積分・乱択・対称性保護・QROM | 強いPF対照、native価格 | 推定は厳密有限h証明ではない |
| Childs et al.（PRX 11, 011020） | S2の明示有限時間上界（arXiv v3 §5.1 Eq.150、Prop.16） | 状態非依存の強い対照 | 基底状態特化ではない |
| Dutkiewicz et al.（arXiv:2607.15358、2026） | 第一量子化の電子構造でSOS gap増幅によるQPE gate数の漸近改善（検索抜粋） | SOS＋SAの適用範囲が広がっている | PFではない |
| arXiv:2608.24493（2026） | guiding state付きground-energy推定の最適下界。SAについても平方根演算子への問い合わせ下界を論じる（検索抜粋） | SA側の理論の成熟 | PFではない |

### 4.3 差分の判定

| 主張 | 状態 |
|---|---|
| SOS（gap最適化済みの化学SOSを含む）をPF型の構成で時間発展し、QPEに使う | **既知**（Kang–Su §1.6・§6.3、Low et al.のSOS形を入力） |
| Hermitian一体平方の指数化 | **既知**（DF-Trotterの標準技術。Kang–SuはQSVT型二乗化） |
| 固定粒子数sectorのnorm | **既知**（Kang–Su Lemma 43 ほか） |
| PSD項・低energyでPF誤差がエネルギーに一次 | **既知**（局所Hamiltonian：Mizuta–Kuwahara等） |
| エネルギー推定特化のPF誤差解析 | **既知**（Yi–Crosson、Hejazi） |
| 非局所化学PFでSOS gapに比例する基底エネルギー誤差評価 | 検索・読了範囲で**未発見**。ただし初等的な組合せであり、未発見であることを新規性の根拠にしない |
| gap不要・基底状態特化・有限hの化学PF保証（§3.8） | 検索・読了範囲で**未発見**。要素は既知 |
| 独立した有利なPF-native分割生成規則 | **未特定**（§5） |
| ACM出版版とarXiv v3の差 | **未確認** |

**結論**：残り得る差分は、分割の生成（A1）ではなく、**SOS gapをPFのground-energy保証に使う理論（A2）**の側にある。既知部分があることをSOS-PF一般の不可能性の根拠にはしないが、A1の独立新規性を支える先行研究上の空白は見当たらない。

---

## 5. 問C：新しいアルゴリズムとして成立するか

### 5.1 生成規則 `R` の候補と既知法への関係

| 候補規則 | 内容 | 既知法との関係 | 独立した改善機構 |
|---|---|---|---|
| $R_{\rm shift}$：DF＋定数shift | 一体部分を固定sectorの最小値でshift、平方はそのまま | 回路は元のDFと同一（恒等式） | なし（保証の書き換えのみ） |
| $R_{\rm complete}$：平方完成・一体再配分・平均場中心化 | $\lambda(L-b)^2$ と補償一体項 | 既知の一体再配分（第一候補と同族、CLOSED構成は再開しない） | 下界は単調に改善しない。選択規則は未提示 |
| $R_{\rm SDP}$：一般SOS Gram SDP | v2RDM双対、spin-free基底 | 既知（Low、Rubin） | PF-native化で劣化し得る |
| $R_0$：PF-native SOS SDP（Hermitian一体基底上の実対称Gram） | 既知SDPの錐を、平方がそのまま指数化できる形に制限 | 既知SDPの直接の制限 | 最適値・rank・幅は未知（経験的な問い） |
| $R_{\rm reg}$：幅・rankを正則化したSDP | $K_{\rm SOS}E_g$ の凸上界を最小化 | $R_0$ の変種。rank・native費用は一般に非凸 | 未確認 |
| $R_{\rm rot}$：Gram因子の回転選択 | $\sum Q^2$ を保つ回転 | 既知（旧D案と同種、HOLD） | 選択規則・対照にできないことが未提示 |
| $R_{\rm real}$：複素Gramの実化 | $\operatorname{Re}G$ と残余一体項 | 線形代数 | rank最大2倍・残余で費用増 |
| Kang–Su構成 | Hamiltonian block encoding、QSVT型二乗化 | 既報 | accessが異なる |

**判定**：K0・K0.5と同じく、既知の構成から区別でき、かつ有利な機構を伴う独立規則は特定できない。最も自然な固定規則 $R_0$ も既知SDPの制限であり、その価値は「新しい規則」ではなく「制限してもgapが小さく保たれ、実biasとstep単価が改善するか」という経験的な事実に依存する。

### 5.2 費用レバーの分解

| レバー | SOS-PFでの機構 | 何に効くか | 現状 |
|---|---|---|---|
| (i) 代表係数 $K_{\rm SOS}E_g$ | 予算共有により $E_g/\Lambda$ の比（§3.4） | 証明付きstep（gap証拠がある場合） | 上界上は有望。実値は未知 |
| (ii) 有限step境界 | Cauchy経路は実用性なし。実時間残差経路はgap不要で $\sqrt{E_g}$、gapありで $E_g$（§3.8） | 証明付きstepのみ | 未導出 |
| (iii) 実bias | 分割を変えると基底状態でのfragmentの揺らぎ $\sigma_j$ が変わる | **実際の量子資源**（推定に基づくstep選択） | **不明**。計算でしか分からない |
| (iv) native/FT価格 | 平方数（$\operatorname{rank}S$）、幅、実化、制御、合成精度 | **実際の量子資源** | 増える可能性。未評価 |
| (v) 情報・certificate価格 | 古典SDP、検証つき変分上界、数値証拠の検証 | 古典資源 | 多項式だが未評価 |
| (vi) overlap・branch・失敗確率 | overlap損失は $E_g/g^2$ 型。gap不要経路ではspectral weightで扱う | 反復回数 | 厳密ground前提では影響は小さい |

**【導出】実biasの決まり方**：§3.3〜3.4の評価で定数を除けるのは任意の分割について成り立ち、$|\langle[F_a,[F_b,F_c]]\rangle|\le2(\sigma_a\sigma_cw_b+\sigma_a\sigma_bw_c)$ は**実際のfragmentの基底状態揺らぎ**で書ける。PSDとSOS gapは、この揺らぎの**上界**を与えるだけである。DF fragmentの実際の揺らぎがSOS fragmentより大きいか小さいかは分からない。従って、SOS表現を使えば実biasが下がるという主張には、現在どの理論的根拠もない（上界が下がることとは別）。

**真の費用レバーは(iii)と(iv)の二つ**であり、どちらも静的な調査では決まらない。(i)(ii)(v)(vi)は証明付き見積もりと保証の品質を左右するが、同じ回路の実費用は変えない。加えて、有用領域の窓 $2\varepsilon<G\ll\Lambda$（§0.2）を満たさない系では、SOS-PFに意味はない。

### 5.3 問Cの結論

新しいアルゴリズムとしては**成立していない**。有利な $R$ は未特定であり、データ不足ではなく構成原理の不足である（K0の判断を支持）。数値最適化やパラメータ探索で $R$ を探す継続は推奨しない。唯一残る問いは「固定した $R_0$ で実biasとnative単価がどうなるか」という事実の確認であり、これはアルゴリズムの新規性ではなく継続判断の材料である（§7.3）。

---

## 6. 問D：保証理論へ主題を変える価値

### 6.1 A2が何をする研究か

A2は**同じ分割・同じ回路**について、取得可能な証拠から、既知の鋭い保証より大きな $h$ を認証する研究である。実際の固有位相biasは変わらない。改善されるのは、保証つきのstep選択と、公平な資源比較のための証明付き見積もりである。新しいHamiltonian simulationアルゴリズムを作る研究とは主張が異なる。

### 6.2 閉じ得る具体的な数学問題

| 課題 | 内容 | 主な難所 | 期待される結果の形 |
|---|---|---|---|
| T1：gap不要・基底状態特化の有限h保証 | §3.8(a)(b)：spectral weight補題＋実時間誤差表現の基底状態への作用＋SOS分散評価 | 共役による補正項と剰余の明示定数、予算共有の最適な使い方 | $P_{\rm far}\le\eta_s^2/[4\sin^2(h\varepsilon_b/2)]$、$\eta_s\le h^3\hat\Gamma_3\sqrt{G}+\dots$。全ての実 $h$ で有効 |
| T2：gapを用い、Cauchy半径に縛られない保証 | §3.8(d)：Rayleigh位相の実時間展開＋Kato–Temple型の位相版不等式 | **認証済み励起gapの取得**、位相版不等式の定数 | $\vert E(h)-E_0\vert \le h^2K_{\rm SOS}G+O(h^3G)+O(h^4G/g)$ 型 |
| T3：既存DF回路の保証 | DFの定数shift表示でT1/T2を適用 | DFの定数shiftでは $G$ がfragmentごとの最小値の和で決まり、一般に大きい | 改善は小さい見込み（下記） |

**T3についての構造的な所見（【導出】）**：同じ回路では、定数を除いたfragmentは変えられないので、使えるgapは $G=U-\sum_j\min_{\rm sector}(H_j)$（各fragmentを独立に最小化した和）に固定される。DFでは平方項の最小値は0に近く、Coulomb反発の寄与を含む $E_0$ との差は大きいと見込まれる。したがって**既存DF回路に対するA2の利得は小さく、SOS gapの利得は分割を変えた場合（A1の問い）にしか現れない**。A2を独立に進めても、結局A1の経験的な問いに戻る。

### 6.3 新規性と価値

T1・T2の構成要素（spectral weight補題、状態依存のTrotter誤差評価、PSD分散、Kato–Temple型評価）はいずれも既知であり、新しいのは化学SOS-PFへの具体的な組合せと定数である。論文化の現実的な着地点は、「SOS gapがPFのground-energy保証にどの形で移るか（gapありで $\sqrt{E_g/\Lambda}$、gapなしで $(E_g/\Lambda)^{1/4}$）」を定理と記号的比較で示す短い理論ノートである。実際の分子で保証がどれだけ締まるかを示すには計算が必要で、それなしでは実用的価値を主張できない。

### 6.4 「実誤差を変えないcertificate」を成果としてよい条件

1. 研究上の問いが、明示的に保証・認証である（資源削減を主張しない）。
2. certificateが合法な情報だけで得られる（真のgap・$E_0$・RDMのoracleを使わない）。
3. 最も強い状態非依存の保証（実際の入れ子交換子norm、Monte Carlo上界、Childs et al.）に対して、現実的な分子で数倍以上の改善がある。
4. 論文の中心claimが「保証理論」として書かれる。

### 6.5 テーマ変更の大きさと判断

A2への転換は、「元のground energyをより安く求める新アルゴリズム」から「既存回路の保証付きstep選択の理論」への**大きな主題変更**である。si.mdの最終目的とユーザーの希望（新アルゴリズム）から見て、主テーマにはしないことを推奨する。ユーザーが保証理論それ自体に価値を置く場合に限り、T1を中心とする限定的な理論ノートとして選択肢になる（数値計算は不要だが、実用性の主張はしない）。

---

## 7. 問E：A1/A2/A3の比較、推奨、次の作業

### 7.1 比較

| 評価軸 | A1：生成算法として継続 | A2：保証理論へ転換 | A3：区切る・保存 |
|---|---|---|---|
| 独立新規性 | 低：$R$ 未特定。広い構成はKang–Su、生成は既知SDPの制限 | 中〜低：要素は既知、化学SOS-PFへの組合せと定数 | — |
| 理論的意義 | 未確定 | 中：SOS gapがPFのground-energy保証へ移る形（gapありで $\sqrt{E_g/\Lambda}$、なしで $(E_g/\Lambda)^{1/4}$）を明確化 | 既存成果（§2.3）の保存 |
| 実現可能性 | 静的作業では進まない。計算が必要 | T1は紙上で可能。T2はgap証拠が難所 | 即時 |
| 必要資源 | SDP・小系の厳密計算（別承認） | 理論作業のみ（実用性の主張には計算が必要） | なし |
| 失敗の情報価値 | 固定 $R_0$ の反証なら中：PFでSOS利得が出ない理由が分かる | 低〜中 | — |
| 論文の中心claim | 「近frustration-free分割でPFの資源を削減」：現証拠では立たない | 「SOS gapとPFのground-energy保証」：短い理論ノート | なし（内部記録） |
| 元の目的への寄与 | 成立すれば直接 | 間接（保証のみ） | なし（資源を他方向へ） |

### 7.2 推奨（条件つきで明確化）

1. **既定の推奨はA3**：SOS-PFを「新しいPF-native分割生成アルゴリズム」候補としては区切り、K0・K0.5の数学（§3.10の成立項目）、Kang–Suとの対応、PF-native化の制約を保存する。SOS-PF一般の不可能性とはしない。追加のK0型静的調査は行わない。
2. **A2は主テーマにしない**。ユーザーが保証理論それ自体を成果として望む場合に限り、T1を中心とする限定的な理論ノートを選んでよい（§6）。
3. **A1は、固定規則 $R_0$ による一回の反証テスト（K1-F）に限り、次の条件をすべて満たす場合に検討してよい**：(C1) ユーザーがPF限定・絶対ground energyというスコープを維持する。(C2) 「SOSの利得がPFの実費用に移るか」という事実判断に計算資源を割く価値を認める。(C3) 結果にかかわらず自動的に次段へ進まないことを受け入れる。(C4) 結果を新規性の根拠に使わない。一つでも満たさなければK1-Fは行わずA3とする。
4. 候補B・C、旧D案、PAUSED系列、閉じた第一候補構成を自動的に再開・採択しない。

### 7.3 K1-F（固定規則による反証テスト）の設計案 — **未認可・提案のみ**

**目的**：固定規則 $R_0$ の下で、PF-native化後のSOS gap、fragment数・幅、実際の主要ground係数と有限hの実bias、証明付き $h$、native単価proxyを、DFの定数shift表示と同条件で比べ、「SOS構造がPFの実費用に移るか」を反証可能な形で判定する。

**固定規則 $R_0$（実行前に固定する内容）**

| 項目 | 内容 |
|---|---|
| 入力 | 保存済みHamiltonianのみ（新規分子生成なし）。K0のK1草案の入力方針（最大2入力、8 modes以下）に従い、入力hashを事前固定 |
| 生成基底 | spin-freeのHermitian一体演算子 $\{I,E^+_{pq},iE^-_{pq}\}$ |
| SDP | $L$ を最大化。制約は $H-LI-\sum_{\mu\nu}S_{\mu\nu}B_\mu B_\nu-(\text{粒子数保存の理想項})=0$（正規順序化した係数一致）、$S\succeq0$ は実対称。solver・許容誤差・数値証拠の検証（残差normを $L$ から差し引く規則）を事前固定 |
| 分解 | $S$ の固有分解。切捨ては事前に定めた閾値以下のみとし、残差を誤差予算に計上 |
| PF | S2、順序は固有値降順などの事前規則。対照はDF＋定数shift（同じ入力・同じS2） |
| 上界 $U$ | 事前に固定した古典的変分法（真値ではない）。真の $E_0$・ground vector・gapは診断層に封印 |

**指標**：(1) $G=U-L$（PF-native）とDF定数shiftの $G$、(2) $M$、$\Lambda$、$K_{\rm SOS}$、$C_{\rm raw}$、(3) 診断用の実際の $d$ と事前固定 $h$ gridでの $E(h)-E_0$、overlap、(4) 証明付き $h$（Childs et al.明示上界＋spectral weight、§3.8のT1が導出済みならそれ、oracle gapによるSchur式は診断のみ）、(5) Givens数・対角位相数などのnative単価proxy。

**判定規則の例**（数値はユーザー／GPTが実行前に確定する）：両入力で「実際の $|d|$ がDFの半分以下」かつ「step単価proxyの比が1.5以下」かつ「$G/\Lambda\le10^{-2}$」のときに限り、A1の発展（スケール可能なrank・幅制御つき $R$ とT1/T2の理論）を検討する価値ありとする。それ以外はSTOPし、否定的結果として記録する。

**上限・STOP**：SDPは各入力1回（計2回）まで、新規分子・compile・T合成なし、attempt 1／retry 0。係数一致・PSD・sector保存の検証失敗、SDPの非実行可能、PF-native $G$ がDF定数shiftの $G$ 以上、のいずれかでSTOP。追加分子、SDP階層、Gram回転探索、候補B/Cへ自動移行しない。

**情報価値**：否定的なら「SOSの利得がblock-encodingには移りPFには移りにくい」ことの具体的な根拠となり、PFとblock-encodingの戦略判断に使える。肯定的でも $R_0$ は既知SDPの制限なので新規性は確立せず、スケール可能な $R$ の設計という次の問いを開くだけである。実行には計算（SDPを含む）の別途明示承認が必要。

### 7.4 研究全体として次に価値がある方向

これまでの系列（第一候補、H-E1、M案、C1〜C4、N1〜N3、THRIFT、SOS-PF）に共通するのは、PFの係数・分割・表現の設計空間が先行研究で密に埋まっており、独立した新規性が残りにくいことである。一方で、低energy構造による大きな利得（SOSSA：4〜195倍）はblock-encoding側で実現している。研究目的（新アルゴリズムによる低費用化）を見失わないため、**次の候補を作る前に、ユーザーが次の二つを判断する**ことを推奨する。

- **D1：block-encoding/qubitization除外の扱い**。従来の特定研究方向での除外を維持するか、少なくともKang–Su型のHamiltonian-evolution算術や、block-encoding法を対象・対照として含めるか。本書はこの制約を解除も拡大もしない。
- **D2：taskの範囲**。絶対ground energyに限るか、エネルギー差（gap・反応エネルギー）も含めるか。

そのうえで、「狙う費用レバーと目標倍率」を入口条件とする方向選定を一回だけ行う。参考となる方向（**いずれも未採択・先行性未確認**）：

| 方向 | 前提となる判断 | 内容 | 主なリスク |
|---|---|---|---|
| E1：Hamiltonian-evolution型アクセスでのSOS spectral amplification | D1で除外を緩める場合 | Kang–Suは自らのSOS simulationをgap amplification法と「比較不能」としている。Kang–Suの算術（少数ancilla、制御つき時間発展）でSAの $\sqrt{\Lambda E_{\rm gap}}$ 型の利得を実現できるかは、SOSの着想を新アルゴリズムへつなぐ最も直接的な問い | 近い研究グループによる既報・進行中の可能性、SA問い合わせ下界（arXiv:2608.24493）、費用会計の差 |
| E2：エネルギー差でのTrotter誤差相殺の保証 | D2でtaskを広げる場合 | Bay-Smidt et al.はPPP模型のgap推定で回路深さ約一桁減を報告。§5.2の揺らぎの見方は、bias差がfragment揺らぎの差で決まるという機構を示唆する | taskの変更。以前のP-C（geometry差）は別task・別理由で停止 |
| E3：レバー先行の資源分析（計算なし） | いずれの場合も可 | 公開済みの資源見積もり（SPRINT、SOSSA、DF/THC qubitization等）から、代表分子でPF-QPEの費用がどこに集中し、どのレバーに10倍以上の余地があるかを文献ベースで整理 | 新しい研究成果ではなく、方向選定の材料 |

### 7.5 次の担当と承認境界

| 担当 | 作業 | 承認 |
|---|---|---|
| ユーザー | A1/A2/A3の採否、K1-Fの要否（§7.2の条件）、D1・D2の判断 | — |
| GPT | 本レビューの独立検証（任意）、D1・D2判断後の方向レビュー（別開始承認） | 別途 |
| Codex | **必須の新しい技術作業はない**。K0・K0.5の論点の再調査は不要。K1-Fが承認された場合に限り、入力hash、$R_0$、判定規則、上限、証拠検証規則を固定する契約準備 | K1-Fの計算は別の明示承認 |
| Claude | 求められた場合の再レビュー | 別途 |

Mandatory STOPを維持する。K1、SDP、新しい計算、GitHub変更、採択はいずれも本書の推奨によって開始されない。

---

## 8. 問F：反論・残存リスク・確度・決定不能の原因

### 8.1 第一推奨（A3）への最も強い反論

**反論**：「§3.4で示したとおり、SOS-PFはPF側で唯一、spectral amplificationと同じ $\sqrt{E_g/\Lambda}$ 型のスケーリングを持ち得る機構である。安価な計算を一度もせずに区切るのは、大きなレバーを捨てることになる。」

**応答**：その比は上界どうしの比であり、実biasは実際のfragmentの揺らぎで決まる（§5.2）。PF-native化（実化・残余一体項）、fragment数の増加、有限stepでの妥当性（gap証拠）のいずれもが利得を消し得る。さらに $R_0$ の新規性は低く、ユーザーの目的（独立した新アルゴリズム）を満たさない。ただし反論には一理があるので、A3を既定としつつ、この反論に直接答える最小の検証（K1-F）を条件付きで残した。確度：中。

### 8.2 その他の反論

- **「A2には閉じ得る定理（T1）があり、計算なしで論文になる」**：可能性はあるが新規性は限定的で、既存DF回路への利得は小さく（T3）、価値はA1の経験的な問いに依存する。保証理論そのものを望む場合のみ推奨する。
- **「K1-Fは、K0.5が警告した数値による救済ではないか」**：規則・判定・上限を事前固定し、結果を二値の継続判断にのみ使うので、$R$ の探索ではない。それでも新規性を優先するなら行わないのが妥当である。
- **「Kang–Suがgapを使っていないなら、A1にも空白があるのではないか」**：Kang–SuはLow et al.のSOS形（gap最適化を経た表現）を入力としてPF型に時間発展しており、gapを使わないのは誤差評価の側である。空白は保証（A2）にあり、分割生成（A1）にはない。

### 8.3 決定不能の原因の区別

| 未確定事項 | 解決に必要なもの |
|---|---|
| 分割変更による実biasの増減 | 計算（K1-F相当） |
| PF-native化後の $G$、fragment数、幅、native単価 | 計算（SDPと静的会計） |
| 現実の分子での $G/\Lambda$ | 計算（PF-native化後の値は文献にない） |
| T1・T2の剰余定数と保証の締まり | 理論作業（T2はgap証拠の取得法も） |
| Kang–Su出版版とv3の差 | 出版版本文の取得 |
| block-encoding除外の扱い、taskの範囲、保証理論を成果とみなすか | ユーザーの選好・スコープ判断 |

---

## 9. 前回Claudeレビュー（同一会話で作成した前回の `レビュー.md`）の訂正

| 前回の箇所 | 前回の記述 | 訂正（指摘元） |
|---|---|---|
| §5.3.3 | 二分割で $d=-a/12+b/6$、$\vert d\vert \le\frac16\Vert H'\Vert f_{\max}E_g$ | $\delta B\vert 0\rangle=-\delta A\vert 0\rangle$ より $a=b$、したがって $d=a/12\ge0$、上界は $d\le f_Af_BE_g/12$。前回の上界は安全だが鋭くない（GPTレビュー、K0） |
| §5.5 | 一般PF-SOSのfragment数 $M\le\operatorname{rank}G$ | 複素Gramを実化すると $\operatorname{rank}\operatorname{Re}G$ は最大 $2\operatorname{rank}G$、さらに残余一体項が残る。実対称Gramから始める場合に限り $M=\operatorname{rank}S$（GPTレビュー、K0） |
| §0.1・§4.2・§5.7 | 化学Hamiltonianの積公式にSOSを移した研究は検索範囲で見つからない | Kang–Su（arXiv v1、2025-10）を見落としていた。同論文はLow et al.の化学SOSをPF型の構成で時間発展する。未発見と言えるのは「SOS gapを使うground-energy保証」という狭い主張だけである（K0、本レビュー§4） |
| §6.2 | 候補Bの利得を $(E_g/f)^{3/4}$ と概算 | $\Vert A\Vert ^2=\Vert \sum X_\gamma^2\Vert \le\sum\Vert X_\gamma\Vert ^2$ でfragment数に依存し、補助register・制御・符号branchの費用も未算入。概算は利得を過大に見せていた（GPTレビュー） |
| §5.4・§5.5 | 構造化SOS（平方完成）を最小変種として扱い、保証の改善と分割の変更を一部で混同 | 定数shiftだけでは回路・実bias・overlapは不変（K0 Eq.19）。平方完成は分割を変えるが既知の一体再配分の範囲にある |
| §5.3.6 | 上界比を $E_g/f_{\rm typ}$ 程度と記述 | 予算共有を使う行和上界と素朴な三重和上界の比は約 $\frac98E_g/\Lambda$（本レビュー§3.4）。ただし公平な対照はより強い状態非依存上界である |
| §10.2 | K0の通過条件に「証明付き総費用で2倍以上」を含めた | 正式gateではなく感度分析の参考とするのが適切（GPTレビュー、K0） |

前回の§5.3.7（順序平均の符号）は、二分割では $a=b$ と整合し、訂正を要しない。

---

## 10. 正式状態と承認境界

次の状態をいずれも変更しない：第一候補の現固定構成 `CLOSED`、H-E1現定義の探索終了、M-a/M-b `STOP`・K1未実施、PF小改変探索系列 `PAUSED`、旧D案 `HOLD`、N1 `STOP`、N2 `HOLD`、N3 `CLOSED`、THRIFT現template `CLOSED`、Track A/B・第1/第2研究・C1〜C4等の独立研究の状態、SOS-PF候補Aの未採択、K0 `UNDETERMINED_DATA_REQUIRED`、K1未認可・未実施、既存のMandatory STOP・one-shot・retry=0・消費済み認可・旧source/結果の保護・global manifestの既知FAIL。本書は、K1、SDP、新しい計算、GitHub変更、研究採択のいずれも認可しない。

---

## 11. 参考文献・元資料

### 11.1 固定GitHub資料

- [`docs/research/qc_sos_pf_k0_20261015/README.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/README.md)（commit `58d86727`、3,989 bytes、SHA256 `b512ca23a18eb81d…`）
- [`docs/research/qc_sos_pf_k0_20261015/technical_audit.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)（commit `58d86727`、32,094 bytes、SHA256 `6ba1ae010b248743…`）
- [`docs/research/qc_sos_pf_k0_20261015/literature_registry.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/literature_registry.json)（commit `58d86727`、14,642 bytes、SHA256 `9d93a0b88c6d8ca3…`）
- [`docs/research/qc_sos_pf_k0_20261015/method_status.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/method_status.json)（commit `58d86727`、5,034 bytes、SHA256 `2163efc7074968ff…`）
- [`docs/research/qc_sos_pf_k0_20261015/gpt_handoff.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/gpt_handoff.md)（commit `58d86727`、6,028 bytes、SHA256 `11134bf1801fcc94…`）
- [`docs/research/qc_sos_pf_k0_20261015/k1_design_draft.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/k1_design_draft.md)（commit `58d86727`、9,783 bytes、SHA256 `d92dec4b5d2b4cf7…`）
- [`docs/research/qc_sos_pf_k0_20261015/provenance.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/provenance.json)（commit `58d86727`、10,492 bytes、SHA256 `ec16fb828da66818…`）
- [`docs/research/qc_sos_pf_k0_20261015/supplied_gpt_review.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/supplied_gpt_review.md)（commit `58d86727`、41,261 bytes、SHA256 `aaab84272fd1e2d1…`）
- [`docs/research/qc_sos_pf_k05_20261015/technical_report.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md)（commit `01bdc56a`、21,581 bytes、SHA256 `99f00a89fb8fbd5b…`）
- [`docs/research/qc_sos_pf_k05_20261015/literature_correspondence.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/literature_correspondence.md)（commit `01bdc56a`、14,162 bytes、SHA256 `f073ad907ecb8ff4…`）
- [`docs/research/qc_sos_pf_k05_20261015/gpt_decision_handoff.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/gpt_decision_handoff.md)（commit `01bdc56a`、8,309 bytes、SHA256 `13b4a9323271f106…`）
- [`docs/research/qc_sos_pf_k05_20261015/README.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/README.md)（commit `01bdc56a`、3,632 bytes、SHA256 `95d9260cf0e04f75…`）
- [`docs/research/qc_sos_pf_k05_20261015/provenance.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/provenance.json)（commit `01bdc56a`、10,522 bytes、SHA256 `401498d23fd2fc77…`）
- [`artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json)（commit `78b50c51`、3,825 bytes、SHA256 `46b8835d849ddd7a…`）
- [`docs/research/研究目的・研究課題.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/%E7%A0%94%E7%A9%B6%E7%9B%AE%E7%9A%84%E3%83%BB%E7%A0%94%E7%A9%B6%E8%AA%B2%E9%A1%8C.md)（commit `78b50c51`、65,154 bytes、SHA256 `c43a7bb22a077a3d…`）
- [`docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md)（commit `78b50c51`、31,677 bytes、SHA256 `8c81a505ae8e86db…`）
- [`docs/reviews/quantum_chemistry_pf/qc_m_k0_claude_scientific_rereview_and_direction_20261015.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/reviews/quantum_chemistry_pf/qc_m_k0_claude_scientific_rereview_and_direction_20261015.md)（commit `78b50c51`、38,976 bytes、SHA256 `2fdd740fe998c6f2…`）

### 11.2 外部文献

- C. Kang, Y. Su, *Quantum Matrix Arithmetics with Hamiltonian Evolution*, arXiv:2510.06316v3 (2026-09-01). https://arxiv.org/abs/2510.06316 ／ ACM Trans. Quantum Comput. 7(4), 1–86 (2026), DOI 10.1145/3816436（本文未取得、HTTP 403）。
- K. Hejazi, J. Soni, M. Shokrian Zini, J. M. Arrazola, *Better product formulas for quantum phase estimation*, arXiv:2412.16811. https://arxiv.org/abs/2412.16811
- K. Hejazi, M. Shokrian Zini, J. M. Arrazola, *Better bounds for low-energy product formulas*, arXiv:2402.10362. https://arxiv.org/abs/2402.10362
- K. Mizuta, T. Kuwahara, *Trotterization is Substantially Efficient for Low-Energy States*, Phys. Rev. Lett. 135, 130602 (2025); arXiv:2504.20746. https://arxiv.org/abs/2504.20746
- C. Yi, E. Crosson, *Spectral analysis of product formulas for quantum simulation*, npj Quantum Inf. 8, 37 (2022). https://doi.org/10.1038/s41534-022-00548-w
- G. H. Low et al., *Fast Quantum Simulation of Electronic Structure by Spectral Amplification*, Phys. Rev. X 15, 041016 (2025); arXiv:2502.15882. https://arxiv.org/abs/2502.15882
- R. King, G. H. Low, R. Babbush, R. D. Somma, N. C. Rubin, *Quantum Simulation with Sum-of-Squares Spectral Amplification*, Phys. Rev. Lett. 136, 110601 (2026); arXiv:2505.01528. https://arxiv.org/abs/2505.01528
- N. C. Rubin, G. H. Low, A. E. DePrince III, *Near-Frustration-Free Electronic Structure Hamiltonian Representations and Lower-Bound Certificates*, J. Chem. Theory Comput. 22, 8751–8766 (2026)（書誌は前回GPTレビューの記載による）; arXiv:2602.05069. https://arxiv.org/abs/2602.05069
- P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741. https://arxiv.org/abs/2606.30741
- A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, *Theory of Trotter Error with Commutator Scaling*, Phys. Rev. X 11, 011020 (2021); arXiv:1912.08854. https://arxiv.org/abs/1912.08854
- N. S. Blunt, A. V. Ivanov, A. J. Bay-Smidt, *A Monte Carlo approach to bound Trotter error*, arXiv:2510.11621. https://arxiv.org/abs/2510.11621
- A. J. Bay-Smidt et al., *Quantum simulation of nanographenes and Trotter error cancellation*, arXiv:2605.00745. https://arxiv.org/abs/2605.00745
- A. Dutkiewicz et al., *Spectral amplification for ground-state energy estimation of electronic structure in first quantization*, arXiv:2607.15358（検索抜粋のみ）。https://arxiv.org/abs/2607.15358
- *Optimal Lower Bound for Ground-State Energy Estimation with a Guiding State*, arXiv:2608.24493（検索抜粋のみ）。https://arxiv.org/abs/2608.24493
- R. D. Somma, S. Boixo, *Spectral gap amplification*, arXiv:1110.2494. https://arxiv.org/abs/1110.2494

### 11.3 実施内容の明記

本書の数学的判断は紙上の検算・導出であり、数値検証は行っていない。§3.4・§3.8のスケーリングは一様normの最悪評価に基づく目安であり、分子についての計算値ではない。§3.8(b)(d)と§6.2のT1・T2は方針であり、剰余項の明示定数を含む完全な証明ではない。Kang–Su出版版、共通指示、レビュー依頼テンプレートの原文は取得していない。
