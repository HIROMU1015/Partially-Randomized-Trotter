# Track B／Parent allocation S4：科学レビュー完成版
## Claude独立レビューを基にした批判的検証・改訂統合版（訂正済み70 law比較）

**レビュー状態：改訂完成版。** 保存済み70 lawと理論的含意を評価したが、一般的な科学的GO、独立新規性、他入力への優位性は認定しない。研究縮小・探索中断は**条件付きの推奨**であり、未承認の新しい研究方針・実行計画を自動採択するものではない。

**成立経緯と責任範囲：** 2026-10-15のClaudeによる独立レビューを原稿とし、GPT-6が `レビュー修正点.md` のF1–F10に沿って数理・解釈・権限境界を批判的に検証して改訂した。**Claude本人がこの改訂を承認・執筆したことを意味しない。** Claudeによるデータ再集計・文献閲覧履歴は原レビューの記録として保持し、GPTがその全件を独立に再実施したとは主張しない。GPTによる改訂では、新しい登録科学計算・合成・operator guard・N再予測を行っていない。

| 項目 | 内容 |
|---|---|
| 作成日 | 2026-10-15（JST） |
| 原稿作成 | Claude（Anthropic）による独立レビュー。原稿の実施・取得範囲は同レビューの自己記録に基づく |
| 改訂・完成版編集 | GPT-6。論理・数理・解釈上の主要な問題点を検証して修正。改訂理由は§13に明記 |
| 改訂日 | 2026-10-15（JST） |
| 原レビューの依頼 | 利用者がClaudeに渡した `指示.md`（24,455 bytes）と、その後の明示的な開始指示 |
| 完成版の依頼 | 利用者がClaudeへ戻さず、GPT側でレビュー原稿と修正点を統合するよう依頼 |
| 対象 | `HIROMU1015/Partially-Randomized-Trotter` の Track B Parent allocation。特に S4 訂正済み比較（v3 control 50＋S4 訂正 20＝70 law） |
| Claude原稿で実施したこと | 固定commitからの資料取得・照合、source読解、保存値の新しい派生集計（§1.4）、紙上導出、引用文献の一部閲覧。原稿記載の実施履歴として扱う |
| GPT改訂で確認したこと | 原稿・修正点、GitHubのS4最終Evidence Packet・70 law CSV・実行引き継ぎ・S4候補生成source・evidence manifest等の対象資料取得、数理的主張の適用条件と結論の論理関係 |
| 未実施・未認定 | 本改訂で新しい登録科学数値計算・回路生成・合成・operator guard・N forecast・workerを実行していない。Claude原稿の全70行・全論文・全実装をGPTが独立再監査したとの認定はしない。GitHubの既存source・結果・marker・STOPは不変更 |

### 根拠の種類の表記

本文の主張には次の印を付ける。

- 〔数理〕紙上で導出できること（本レビューでの導出を含む）
- 〔保存〕固定commitの保存結果（JSON／CSV）で確認したこと
- 〔source〕固定sourceに書かれていること（静的読解）
- 〔文書〕Codex報告・過去GPTレビューの記述（本レビューで再監査していない部分）
- 〔一次文献〕外部一次資料の本文で確認したこと
- 〔派生集計〕Claude原稿が保存値から**新たに実施した**比・NC proxy・価格分布・post-hoc順位などの数値分析（§1.4）。既存resource mapの原記録、事前登録された独立実験、exact認証とは区別する
- 〔解釈〕解釈・仮説
- 〔未確認〕取得・検証していないこと

参照記号 [R…]（repository）と [L…]（文献）は §12 に固定URL・bytes・SHA256付きで列挙する。affine relation の番号 [840] などは S4 `resource_map.json` の `/combined_comparisons/affine_relations/<index>` を指す。§12のR9・R21は複数ファイルに一意な枝番を付けて整理する。

**根拠の階層：** 「〔保存〕」「〔source〕」「〔一次文献〕」の記述は主としてClaude原稿が指定固定資料から確認した履歴を継承したものであり、必要に応じてGPTが一部のGitHub固定資料を追加取得した。本文の一般的なno-go、実用環境での優位／非優位は、未検証の領域へ拡張しない。

---

## 1. レビュー情報・資料取得状況

### 1.1 固定commitと役割

| 役割 | commit | 日時（JST） |
|---|---|---|
| S4最終比較Evidence Packet・70 law索引CSV | `53c48317d16851333c0e0354b0dfd8ef91e1d524` | 2026-10-15 15:05 |
| S4結果・371監査・chunk検証の公開 | `fe0f3f077342bf7a557033e4dfe2734b7ead6091` | 2026-10-15 14:38 |
| S4 science source | `4ccc1832f63cd323b23fc05623f7e0b58cf34f37` | 2026-10-15 08:52 |
| S4 authorization-only実行HEAD（A4） | `71cf1b76492ed01fc5bb8363246b5f13f7e166d7` | 2026-10-15 14:31 |
| v3 Evidence Packet | `17bfcaac5a53d648ee94de0067de24d719ed3911` | 2026-10-15 08:04 |
| v3結果公開（50 controlの元） | `cd055232f9612982a60953341b9cbf6168dd0bd8` | 2026-10-15 03:00 |
| 先行研究対応監査 | `bf1366ac5832136ab8038d160e9a0285da69e79e` | 2026-10-14 23:47 |
| Track B研究方向レビュー | `4d80b45693a1ea7c14d7cbf9035e7358788953bd` | 2026-10-14 09:37 |

Claude原稿では8 commitの存在を確認済みと記録している。S4 Evidence Packetのcommitは `main`（Claude確認時のHEAD `0babed0`）上ではなく、branch `track-b-parent-allocation-s4-evidence-packet-20261015` 上にある。fe0f3f0→53c4831 の差分は原稿§12.3の記録を保持した。GitHubのbranchやHEADは将来変化し得るため、科学的引用では上表の固定SHAを使う。

### 1.2 取得経路と照合

- 公開repositoryを `git clone --filter=blob:none https://github.com/HIROMU1015/Partially-Randomized-Trotter.git` で取得し、各ファイルを `git show <commit>:<path>` で固定blobから取り出した。GitHubの通常表示（Web UI）は使っていない。GitHub REST APIは未認証のrate limitに達したが、git経路で全件取得できたため影響はない。
- 指示書の資料A〜Dの19件はすべて取得できた（§12.1にbytes／SHA256）。
- `resource_map.json` はgitから全体を取得し（3,228,746 bytes、SHA256 `8281cf66c1ce7b5abf1a46ba6d4c4725c994eabf3dadcc143a9877387cc117c5`）、さらに `evidence_manifest.json` の順序に従って `resource_map.json.part0000.txt`〜`part0006.txt` の7 chunkを連結し、bytes・SHA256がmanifestの値および全体取得と一致することを確認した〔保存〕。一部のchunkだけを読んだ状態ではない。
- `inherited_rows.json` の50 rowは、v3 `resource_map.json` のrowsからconfig_04／05／10／11の20 rowを除いたものと、順序を含めてオブジェクトとして完全一致した〔保存〕。
- 判断のために、指定外の保存資料も読んだ。v4の `candidate_freeze.json`・`guarded_cache.json`・`inventory.json`・`events/config_04,05,10,11.json`、v3の `resource_map.json`・`candidate_freeze.json`・`guarded_cache.json`・`events/config_00〜14.json` である。
- 読んでいないもの：S3 science source（`53a36adf…`）の `design.py`／`pipeline.py` 本体（旧KKTの不具合は [R13] の引用、[R4] の記述、保存配分の幾何（§4.3(d)）で確認した）、監査script本体、`publication_receipt.json`、focused testsの実行記録、D1〜D3とDF系列の原資料。
- Claude原稿の作業経緯：最初のセッションはツール呼出しの上限に達してファイル作成前に停止し、その際にClaudeは暫定要点をチャットへ書いた。これは当初 `指示.md` の**「チャットにはレビュー本文を書かず、ファイルのみ出力」**という出力契約と一致しなかった。経緯を消さずに記録する。本書はClaudeの当該セッションではなく、GPTによる改訂完成版である。
- GPT改訂で確認した範囲：原稿・修正提案と、S4最終Evidence Packet、70 law索引、S4実行引継ぎ、source `design.py`、manifestのGitHub取得を確認した。Claude原稿にある独立取得・再計算・文献読解の全操作は反復していない。

### 1.3 外部一次資料の確認範囲

arXiv API（export.arxiv.org）で版・公開日・要旨を取得し、arXiv HTML（Wanのみ PDF をテキスト化）から該当箇所を抽出して読んだ。全文の精読ではなく、網羅的なpriority調査でもない。

| ID | 論文 | 版と公開日 | 本文で確認した箇所 |
|---|---|---|---|
| [L1] | Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495 | v1, 2026-03-13 | §II Eq.(7)–(10)・Theorem 1、§III Eq.(22)–(23)・Theorem 2、§V Eq.(60)（qDRIFTへの適用） |
| [L2] | Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, arXiv:2402.15550（PRX Quantum 5, 040352） | v2, 2024-10-31（初版2024-02-23） | §II Statement 1–2、Eq.(3)（複数ゲートでのoverheadの積）、Eq.(6)（BPDN／LASSO）、低T数Clifford+T列ライブラリによる回転合成の記述とFig. 1 |
| [L3] | Peetz, Smart, Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095（npj Quantum Inf.） | v2, 2026-07-06（初版2024-07-30） | §II.3 Eq.(4)–(6)・Theorem 1、CTS（convex Taylor sampling）の定義、Supplementary Note 5 Eq.(S15) |
| [L4] | Wan, Berta, Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071（PRL 129, 030503） | v2, 2022-07-13 | Appendix C Eq.(C1)–(C5)（共通角pairingと、r stepの総重みが一stepの重みのr乗になる構造）、同一角の回転の実装費用（Hamming weight phasing）に関する記述 |
| [L5] | Zeng, Sun, Jiang, Zhao, *Simple and high-precision Hamiltonian simulation by compensating Trotter error with LCU*, arXiv:2212.04566（PRX Quantum 6, 010359） | v2, 2025-03-27 | Eq.(74)–(75)（恒等項の比例分割によるpairing） |
| [L6] | Koczor, Morton, Benjamin, *Probabilistic Interpolation of Quantum Rotation Angles*, arXiv:2305.19881（PRL 132, 130602） | v2, 2023-11-14 | 要旨（三つの離散角設定の確率的選択と、overheadの最適性） |
| [L7] | Kiumi, Koczor, *TE-PAI: Exact Time Evolution by Sampling Random Circuits*, arXiv:2410.16850（Quantum Sci. Technol. 10, 045071） | v3, 2025-09-24 | 要旨・序論（±Δとπの二種類の角だけのPauli回転、FT実装でのT-state数推定、深さと測定overheadの交換） |
| [L8] | Granet, Dreyer, *Continuous Hamiltonian dynamics on digital quantum computers without discretization error*, arXiv:2308.03694（npj Quantum Inf. 10, 82） | v2, 2023-11-17 | 要旨とEq.(1)–(4)付近（回転角を自由に選び、既知の減衰因子とshot増で補う構成） |
| [L9] | Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647（PRX Quantum 7, 020332） | v2, 2026-07-10 | 要旨のみ（研究全体の文脈確認） |

[L1]–[L5] は指示書で指定された候補、[L6]–[L9] は本レビューで追加した近接文献である。

### 1.4 保存資料の読取り、Claude原稿の派生分析、今回のGPT改訂を分離する

**Claude原稿で実施済みの派生分析**は下記1–8である。保存された数値を使った分析だが、「科学計算は一切なし」とは表現しない。新しい量子シミュレーション・native合成・operator guard・登録N予測は含まない一方、事前契約にない新しい数値の集計・post-hoc選択を含む。これらをS4で事前登録済みの新しい実験結果やexact certificateと混同しない。元原稿ではfloat64を使用しており、厳密な数値認証がない結果は探索的診断に留める。

1. 保存G・N・CX・native1Qの比と差（固定CSV [R2] のexact欄を基に表示用丸め）。
2. 保存された有理配分・λ区間・丸め距離の小数表示への換算。
3. 保存event表（v3 `events/config_00〜13`、S4 `events/config_04,05,10,11`）の係数和B、canonical期待T、$S=\sum_e\tilde c_e\sqrt{T_e}$、root／非root内訳の派生集計。
4. 保存cache 48 key（v3 36、S4新12）のRZ 1本あたりT数分布。
5. 保存W・Δ・νからのrange項とvariance項の比 $(2/3)W\Delta/\nu$。
6. 保存proxy価格 (136,138,140) を用いたKKT摂動改善の概算。
7. 保存pairwise affine relationの一部交点の読取り（新しい交点・包絡線は作っていない）。
8. 70 lawのN最小や構成内最良候補のpost-hoc検索・並べ替え。

**今回のGPT改訂：** 数理的関係と推論の射程を見直し、Claude原稿の表と派生値を参照した。新たな登録入力数値分析、NCのexact再集計、全$h$包絡線、r>1モデル計算は行っていない。これらを今後Codexが行う場合は、「保存データだけだから無承認で科学数値計算を実行可能」とはせず、対象・source・計算上限・承認を別途確認する。

---

## 2. エグゼクティブ結論

### 2.1 判断（研究上の暫定推奨であって、採択済みGO/STOPではない）

- **D＋Eを第一候補とする：Parent allocationを条件付きの限定的成果として整理し、同じsynthetic入力での新しいone-shot数値探索は当面中断することを推奨する。** 同じ形式の追加候補を増やすより、既存証拠と適用条件を確定する方が情報価値が高い。ただし研究系列全体の一般的no-goを認定せず、方針の変更には利用者の承認が必要。確信度：中。
- **B（内部KKT／halfway）：今回の固定価格proxyとその近傍を、次の主たる探索対象として優先しない。** Smooth proxyが予測する改善が、角度依存のnative価格変動に比べ非常に小さい。全ての連続内部配分やhardware-aware設計の不可能性は示していない。確信度：中〜高（限定範囲内）。
- **A（頂点＋IS）：有限条件でのT低下は有効な保存結果として保持する。** 一般的な量子計算資源改善として継続するには、実際のstep数、$h$、pure事象のnative価格を先に固定する必要がある。確信度：限定命題は高、一般化判断は中。
- **C（Return aggregationとの共同最適化）：直近の優先度は低い。** P5とordinaryの両方に配分利益があり、RA固有の追加機構は独立に識別できていない。確信度：中。
- **Track B全体：** 別のアルゴリズム構成を選ぶ前に、目標task（PR／RPEの乱択部分）の現実的な費用構造（$r$・各step $x$・$h$・basis change・同角共有やnative合成）を既存資料から整理する方針を推奨する。これは単なる原稿仕上げのためではなく、次に価値のある科学的問いの選別を目的とする。確信度：中。

**重要：** 「探索を当面中断する」という推奨はユーザーの研究方針を既に更新した意味ではなく、新たなsource実行・登録科学計算を承認しない。

### 2.2 根拠の要約

1. **KKT方向の滑らかなproxy利益より、今回の固定条件での角度依存native価格変動が大きい。**〔保存〕〔派生集計〕〔数理〕
   - 保存48角のRZ 1本のT数は主として64〜72、外れ値84。root 1事象のTはおおむね2本のRZとbasis change（0／2／4 T）で決まる。
   - P5訂正KKTのroot価格は [144,142,132]（common [136,138,140]）となり、同一proposalの$h=0$切片は約0.75〜0.78%下がった。ordinary訂正KKTのroot価格はcommonと同じで、差は約+0.004%。**これは実装固定時の正当な費用差であるが、滑らかなKKT最適化の利益と同一視できない。**
   - 旧halfwayと訂正KKTは配分が近いにもかかわらずcanonicalのT切片がordinary約0.48%、P5約3.9%異なる。fixed-tool・precision・seed下の**決定的な角度依存価格差**であって、反復測定した統計的ノイズ分散ではない。
   - 保存proxy価格での摂動評価によれば、smoothな固定価格KKTの相対利益はrootの$S$で$10^{-5}$程度と概算される。これは探索的な派生評価であり、全providerに共通の上界ではない。
2. **頂点＋ISのT切片低下は訂正集合でも保持される。**〔保存〕〔派生集計〕　普通のordinaryで約14.0%、closed P5で約19.0%（各armのbest common対best noncommon）。6頂点構成の構成内bestはそれぞれcommonに勝っているが、代表pairではNが約1.72／3.34倍、CXが約1.43／2.56倍になる。
3. **保存されたT改善の主要機構は、安価なpure事象を作る配分と既知ISとの組合せで説明できる。**〔一次文献：Claude確認〕〔派生集計〕　解析IS適用可能なlawでは、保存費用比と$(\sum_e\tilde c_e\sqrt{C_e})^2$の比が数値的に近い。一方、T=0事象がある場合は解析ISが適用外になり、正則化proposal格子による制約を受ける。有限confidenceや$h>0$での普遍的最適性は示さない。
4. **$h=0$のT切片は無料・低T事象を相対的に優遇する。**〔数理〕　ゼロT事象へproposalを集中させる極限ではNが発散しても$2N\mathbb E[C_T]$は一般に**有限の非ゼロ下限へ近づく**。利益を無限に拡大するわけではない。ただし共通準備費用$h>0$を課すと$2Nh$は発散し得るため、切片だけでは総資源を代表できない。
5. **多stepの一次normalization負担は重大な懸念だが、総T費用比の指数下界を証明したわけではない。**〔数理〕　独立step積推定量で固定頂点を用い$x=\tau/r$とすると、係数質量由来の二次moment下界は$e^{2\tau(1-p_j)}$へ近づき、commonの対応する質量下界は1へ近づく。ただし最適proposal・実際のnative費用・$h$・実装taskを含む総費用の倍率には追加条件が必要。
6. **Return aggregation自体のcommon比較での優位と、RA固有の配分利益は区別する。**〔保存〕〔解釈〕　P5 commonはordinary commonより低Tで、保存比較の指定資源では優位。partial returnとの差はh=0のTで約1.16%に留まる。P5とordinaryの頂点改善率の差が、新しい独立機構を証明しているわけではない。
7. **独立新規性は未認定。**〔数理〕〔一次文献：Claude確認〕〔解釈〕　Euler分解・線形分割・固定辞書cost-aware ISや安い回路との重み交換は既知理論と近い。今回の特定rootへの適用・生成手順の完全同値や全他方式に対する包含は網羅的に調べていないため、**新しい資源現象が絶対に存在しない**とは断定しない。

### 2.3 確信度の内訳

- **高：** 任意配分のEuler mean保存、commonのL1最小性、S4の70 law正式集合、原報告に示されたbest-to-best数値。
- **中〜高：** 今回の固定価格proxy改善が角度依存native価格差より小さいという限定観察、頂点＋ISのT切片利益の既知ISによる定性的説明。
- **中：** 多step・$h$によって条件付き利益の実用性が制限されるという方向性、Parent allocation新規one-shotを当面中断する研究上の推奨。
- **未判定：** 全$h$の最適包絡線、PR/RPE/DFの実装条件でのend-to-end利益、独立新規性の網羅的priority、全内部配分familyの限界。

---

## 3. 研究全体の経緯と今回の位置付け

### 3.1 経緯（固定資料からの要約）

| 日時（JST） | 出来事 | 資料 |
|---|---|---|
| 〜10-13 | Return aggregation（G6〜G10）、D1〜D3の保証・価格比較、DF補正などの系列。D3では、同じ有限P5を実現するP3+tailとclosed P5の順位が、root角の合成価格差で約0.170%逆転した | [R15] §3–4〔文書〕 |
| 10-14 09:37 | 研究方向レビュー。「回路費用を考慮したpairing・係数配分」を第一候補に選び、配分family・L1最小性・固定価格KKTを導出 | [R15] |
| 10-14 23:47 | 先行研究対応監査。Euler／ISは既知、SPSは線形埋め込みで比較可能、完全同値は未確認 | [R14] |
| 10-14 | 科学設計レビュー。ordinary／P5 × {common、3頂点、KKT、halfway} と参照3方式、5 proposal、ε_axis=1/200、α_axis=1/40、主指標 $G_T(0)$ を採択 | [R12] |
| 10-15 00:43〜02:58 | v1技術停止、v2 CPU停止、v3で15構成・70 lawを完遂（新合成16、918監査PASS） | [R11] |
| 10-15 08:04〜08:35 | v3 Evidence Packet。GPTレビューが頂点＋ISの限定的positiveと、KKT実装の目的関数不一致（raw Cに $C^2$ 用の式）を指摘 | [R10][R13] |
| 10-15 08:52〜15:05 | S4でKKTを訂正し、影響した4構成・20 lawだけを補完（新合成12、371監査PASS）。S4 Evidence Packetと70 law索引 | [R4][R3][R1][R2] |

### 3.2 過去の系列から引き継ぐこと

[R15] の整理（本レビューでは原資料を再監査していない）のうち、今回の判断に直接効くのは次の三点である〔文書〕。

- **SP系列**：回路を安くする代わりにweightを増やす交換で大きな限定利益が出たが、長いwrapperでは反転した。今回の頂点＋ISも同じ型の交換で、同じ型の制約（hと多step）を受ける（§4.5、§6.4）。
- **D3**：root角1本の合成価格で順位が0.17%動いた。今回の内部候補でも同じ現象が0.5〜4%の幅で起きている（§5.3）。Track Bでは、数%以下の差を比べるとき、角度依存価格変動が繰り返し支配的になっている。
- **G5／G10**：固定classでの強い対照の重要性。今回はcommonに同じproposal機会を与えており、この点は守られている。

### 3.3 今回のレビューの位置付け

訂正後の内部候補が揃った最初のレビューであり、Parent allocationを続けるかを決める節目である。過去のGPTレビュー [R13] の「頂点＋ISの限定的なT利益を支持し、KKTの不一致を先に正す」という判断は、手順としては妥当だった。訂正の結果、内部候補は設計上の効果を検出できる実験になっていなかったことが分かった（§5.3）。これは [R13] の誤りというより、固定価格proxyの効果の大きさを事前に見積もっていなかったことによる。本レビューは、その見積もりを紙上で補い、研究判断を更新する。

---

## 4. 数学的検証

### 4.1 任意配分による平均保存（係数・符号・位相・順序・端点）

偶数parent $u$ について、指示書の記法で

$$M_u=\zeta\Big(aI-i\sigma\sum_i b_iQ_i\Big)Q(u),\qquad Q_i^2=I,\quad \zeta=(-i\sigma)^{|u|}\in\{\pm1\},\quad \sigma\in\{\pm1\}.$$

$z_i\ge0$、$\sum_iz_i=a$ に対し $c_i=\sqrt{z_i^2+b_i^2}$、$\phi_i=\operatorname{atan2}(b_i,z_i)\in[0,\pi/2]$ とおくと、$Q_i^2=I$ から

$$e^{-i\sigma\phi_iQ_i}=\cos\phi_i\,I-i\sigma\sin\phi_i\,Q_i,\qquad c_i\,e^{-i\sigma\phi_iQ_i}=z_iI-i\sigma b_iQ_i .$$

child ごとに足すと $\sum_i(z_iI-i\sigma b_iQ_i)=aI-i\sigma\sum_ib_iQ_i$ なので

$$M_u=\sum_i c_i\,\zeta\,e^{-i\sigma\phi_iQ_i}\,Q(u)$$

が成り立つ〔数理〕。異なる $Q_i$ の可換性は使っていない。回転は $Q(u)$ の左から掛かる（時間順では $Q(u)$ の後に回転）。

端点と退化：$z_i=0$、$b_i>0$ なら $\phi_i=\pi/2$、$e^{-i\sigma\pi Q_i/2}=-i\sigma Q_i$ で、係数 $b_i$ のpure事象になる。$b_i=0$ のchildに配った質量は恒等（pure parent）にまとめる。

sourceとの対応〔source〕[R18]：`ParentAllocation.__init__`（136–160行）は、common点で比 $s/a$ と因数分解した区間 $d\,b_i/s$（$d=\sqrt{a^2+s^2}$）を使い、一般点で比 $b_i/z_i$、pure端点、pure parentを作る。`phase_i_power` は $|u|\bmod4$ で $\zeta$ と一致し、pure childでは $\sigma=+1$ のとき3、$\sigma=-1$ のとき1を足す（$-i\sigma$ と一致）。`native_plan`（368–402行）はancilla上の位相ゲート（S、Z、S†）→時間順のword（`reversed(word)`）→回転の順に並べ、演算子 $\mathrm{rot}\cdot Q(u)$ と一致する。回転は $V_i$・制御RZ・$V_i^\dagger$ で、負の回転は保存列の実際のadjoint（Wを保持、`up_to_phase=false`）を使う（[R19] `native_accounting`、`realized_time_gates`）。以上から、符号・位相・順序・端点の扱いは採択設計と整合していると判断する。ただしsourceの全行認証ではなく、関連関数の静的読解である。

既知性：共通角 $z_i=ab_i/s$ の場合は、[L4] Eq.(C4)–(C5)（ラベルに依存しない角 $\theta_n$ で $1+i\frac{x}{n+1}P$ をpairing）、[L3] Eq.(S15)（$I+iL_sP'=\sqrt{1+L_s^2}\,e^{i\theta P'}$）、[L5] Eq.(74)–(75)（恒等項を $\eta_2/\eta_\Sigma$、$\eta_3/\eta_\Sigma$ で比例分割して同一角でpairing）にそのまま現れる〔一次文献〕。一般の $z$ への拡張は線形分割の直接の帰結で、新しい一般定理ではない〔数理〕。

### 4.2 正規化：commonはfamily内のL1最小で、非commonの利益は単価との交換だけ

$$L(z)=\sum_i\big\|(z_i,b_i)\big\|_2\ \ge\ \Big\|\Big(\sum_iz_i,\ \sum_ib_i\Big)\Big\|_2=\sqrt{a^2+s^2}.$$

等号は全 $b_i>0$ のchildで $z_i/b_i$ が共通のとき、すなわちcommon点だけで成り立つ〔数理〕。

保存値もこれと整合する〔保存〕[R2]。係数質量Bは、ordinary common 1.502093に対しKKT 1.502132、halfway 1.502103、頂点1.692〜1.855。P5 common 1.288445に対しKKT 1.288477、halfway 1.288453、頂点1.461〜1.617。**非common候補のどれも正規化では得をしていない。** 観測されたT低下はすべて一試行あたりの期待単価の低下から来ており、二次moment νとNはcommon以上である（§5）。

### 4.3 KKTの訂正

**(a) 採択proxyの性質**〔数理〕。$S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}$、$C_i>0$、$b_i>0$、$a>0$ とする。各項は $z_i$ について厳密凸（二階微分 $\sqrt{C_i}\,b_i^2/(z_i^2+b_i^2)^{3/2}>0$）である。さらに実行可能な有限単体 $0\le z_i\le a$ 上では二階微分に一様な正の下限があり、この制限上では強凸でもある。よって単体上の最小点は一意である。境界 $z_i=0$ では $\partial S/\partial z_i=0$ なので、$z_j>0$ のchildから質量を移すと必ず下がり、最小点は内部にある。内部の停留条件 $\sqrt{C_i}\,z_i/\sqrt{z_i^2+b_i^2}=\lambda$ から

$$z_i=\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}},\qquad 0<\lambda<\sqrt{\min_iC_i}.$$

$\sum_iz_i(\lambda)$ は0から極での発散まで狭義単調増加なので、解は一意である。

**(b) S4 sourceとの整合**〔source〕[R16]。`kkt_candidate`（14–69行）はraw価格Cを受け、$t=\lambda^2\in[0,\min C)$ を128回二分し、各childの $z_i(t)=b_i\sqrt{t/(C_i-t)}$ を256 bitの外向き区間で囲む。和の上限が $a$ 未満なら下端、下限が $a$ を超えれば上端を更新し、曖昧なら打ち切る。区間 $[\mathrm{lo},\mathrm{hi}]$ は常に $t^*$ を含み（$z_i(t)$ の単調性による）、`ideal_z_intervals` はloでの下限とhiでの上限で作るので、真のKKT点を含む。中点を正規化し、分母 $2^{64}$ の最大剰余法で有理化して $\sum z=a$ をexactに保つ。目的関数の文字列も `sum sqrt(raw_event_T_C_i)*sqrt(z_i^2+b_i^2)` である。**採択proxyとS4 sourceは一致している**と判断する。

**(c) 保存された証明書の意味**〔保存〕〔派生集計〕[R9a]。ordinaryは $\lambda\approx9.5786$（$\lambda^2\approx91.75<136$）、P5は $\lambda\approx9.4204$（$\lambda^2\approx88.74$）。理想KKT点からの丸め距離の上界はordinary約 $3.1\times10^{-20}$、P5約 $2.4\times10^{-20}$ で、`simplex_residual=0`、`exact_KKT_optimum=false`。これは「保存配分は固定価格proxyのKKT点に $10^{-20}$ の精度で一致する」ことの証明書であり、native費用の最適性については何も言わない。proxyの価格（旧common角での価格 136／138／140）は新角度で±4〜8 Tずれるので（§5.3）、モデル誤差は丸め距離より何十桁も大きい。

**(d) 旧proxyとの関係（非同値性と幾何）**〔数理〕〔保存〕。旧実装の停留条件は $C_iz_i/\sqrt{z_i^2+b_i^2}=\ell$ で、$\sum_iC_i\sqrt{z_i^2+b_i^2}$ のKKT、すなわち採択目的の価格を $C_i^2$ に置き換えたものにあたる [R13] §4.3。不等価格では採択目的と一致しない。ただし価格の差が小さいときは、commonからの移動方向は同じで、相対的な価格差が約2倍になるぶん歩幅が約2倍になる（(e)の摂動式）。保存配分はちょうどそうなっている（小数は保存exact有理数 [R9a][R21b] の表示換算〔派生集計〕）。

| arm | 旧KKT − common | 旧halfway − common | 訂正KKT − common | 訂正halfway − common |
|---|---|---|---|---|
| ordinary | (+0.01165, +0.00344, −0.01509) | (+0.00583, +0.00172, −0.00755) | (+0.00570, +0.00181, −0.00752) | (+0.00285, +0.00091, −0.00376) |
| closed P5 | (+0.00982, +0.00300, −0.01283) | (+0.00491, +0.00150, −0.00641) | (+0.00482, +0.00157, −0.00639) | (+0.00241, +0.00079, −0.00319) |

commonはordinary (0.2, 0.3, 0.5)、P5 (0.17919, 0.26995, 0.45627)（$a_0=139129/153664\approx0.9054$）。旧halfwayと訂正KKTの差の最大成分は、ordinary 1.2×10⁻⁴、P5 9.6×10⁻⁵ である。**8個の内部点は、commonから同じ方向に0.5×・1×・1×・2×の歩幅で並んだ一次元の点列**になっている。

**(e) 固定価格proxyで得られる改善の大きさ**〔数理〕。common点のまわりで $z=z^0+v$、$\sum_iv_i=0$ と展開すると、$w_i=\sqrt{C_i}$ について

$$S(z^0+v)=S^0+\frac ad\sum_iw_iv_i+\frac12\sum_iw_i\frac{s^3}{d^3b_i}v_i^2+O(v^3).$$

二次近似の最小値は

$$\frac{\Delta S}{S^0}=-\frac12\Big(\frac as\Big)^2\frac{A-H}{A}\ \approx\ -\frac18\Big(\frac as\Big)^2\frac{\operatorname{Var}_b(C)}{\bar C^2}$$

で、$A$、$H$ は $b$ で重み付けた $w$ の算術平均・調和平均である。保存proxy価格 (136, 138, 140) と、保存common配分から読める $a/s$（ordinary 1.4、P5 約1.33）を代入すると、root部分の $S$ の改善はordinary約 $3.1\times10^{-5}$、P5約 $2.8\times10^{-5}$、全体のNC（$\propto S^2$）では約 $5\times10^{-5}$、$G_T(0)$ で約1万Tのオーダーである〔派生集計〕。旧proxyの点（約2倍の歩幅）は、採択proxyの二次近似では改善がほぼ0に戻る（$S(z^0+2v^*)\approx S^0$）。

**結論**：訂正は正しく行われた。しかし採択proxyが予測する滑らかな改善は、この価格構造（旧commonのchild間のT単価差はbasis changeの0／2／4 T）では非常に小さく、S4の少数候補のnative価格変動から独立に検出・帰属することが難しい。さらに(a)により、固定価格proxyは境界（pure事象）を決して選ばない。したがって、頂点で起きている「回転が消えて価格が不連続に下がる」機構を原理的に表せない〔数理〕。

### 4.4 ISの三つの最適化を区別する

1. **固定辞書・固定係数でのIS**〔一次文献〕[L1]。$NC_q=\mathbb E_q[c]\,\mathbb E_q[w^2]\ge(\mathbb E_p[c^{1/2}])^2$、等号は $q\propto p/c^{1/2}$（Eq.(7)、(9)、(10)）。係数で書くと $\nu\,\overline C\ge(\sum_ec_e\sqrt{C_e})^2$。Theorem 2（Eq.(22)–(23)）により、実装を固定したままならISはbiasを変えない。
2. **辞書を変える外側の配分最適化**〔数理〕。$\min_z\big(S_{\rm rest}+S_{\rm root}(z;C(z))\big)^2$。$C(z)$ は角度ごとの合成価格で、境界で不連続に下がり、内部では角度ごとに非滑らかに変動する。[L1] の定理はこの外側問題を扱っていない。配分を変えると実装も変わるのでbias不変も移せない（S4でも配分ごとに重み付きstrict native誤差が $2.36\times10^{-7}$〜$4.48\times10^{-7}$ と変わる〔保存〕）。
3. **有限confidenceの十分N**〔数理〕〔派生集計〕。$N=\lceil\log(2/\alpha)\,(2\nu/\Delta^2+4W/(3\Delta))\rceil$。range項と分散項の比は $(2/3)W\Delta/\nu$ で、保存値からP5 common canonical 0.24%、P5頂点0解析IS 0.29%、ordinary頂点2正則化（$C_*/16$）0.25%。残余幅Δは70 lawで0.0046301859〜0.0046306114（相対差 $10^{-4}$ 未満）、ceilの影響は $10^{-6}$ 程度である。したがって

$$G_T(0)\approx2\log(2/\alpha)\cdot\frac{2}{\Delta^2}\cdot\nu\,\overline{C_T}\,\big(1+O(3\times10^{-3})\big)$$

となり、**この比較では有限confidenceの順位は [L1] のNCの順位とほぼ同じ**である。§5.5の保存値がこれを裏付ける。

例外はT=0の事象を含む辞書である（ordinary／P5の頂点1・2、matched CTS）。$q\propto c/\sqrt C$ が定義できず、NCの下限は達成されない。下限に近づけるとWとNが発散し、range項とhが効く。S4の正則化 $\kappa\in\{C_*/16,\ C_*,\ 16C_*\}$（$C_*\approx143.28$）はこの領域での固定の選択で、最適ではない（§5.5）。

### 4.5 多stepでのnormalization下界と総T費用を区別する

〔数理〕**ここでの議論は、独立に各stepの事象を抽出し、weightを掛ける積推定量を仮定する。** これはS4の $r=1$ 契約そのものではなく、将来の応用可能性を検討するための紙上の条件付き議論である。状態に依存した分散・相関sampling・whole-circuit cancellationは含めない。

ordinaryではrootの$a=1$、$b_i=xp_i$を用いる。closed P5についても小$x$で$a=1+O(x^2)$、$b_i=xp_i+O(x^3)$という展開が成立する場合、$\tau>0$を固定して$x=\tau/r$としたとき、同じchild $j$へ全parent質量を与える頂点の係数質量は

$$L_{V_j}(x)=\sqrt{a^2+b_j^2}+\sum_{i\neq j}b_i=1+x(1-p_j)+O(x^2),$$

commonでは

$$L_{\rm common}(x)=\sqrt{a^2+(\sum_ib_i)^2}=1+O(x^2).$$

このため係数質量の積の二乗は

$$\prod_{s=1}^{r}L_{V_j}(\tau/r)^2\longrightarrow e^{2\tau(1-p_j)},\qquad
\prod_{s=1}^{r}L_{\rm common}(\tau/r)^2\longrightarrow1.$$

各stepのlawで確率$q_e>0$とweight$w_e=c_e/q_e$を用いる場合、Cauchy–Schwarzから$\nu_s=\sum_eq_ew_e^2\ge(\sum_ec_e)^2=L_s^2$。独立積推定量では$\nu_{\rm total}=\prod_s\nu_s$であるため、頂点側の二次momentには上記指数因子の**下界**がある。common側でも対応する**係数質量下界**は1に近づく。

**重要な限定：** 下界の比較だけで、実際のcommonの$\nu_{\rm total}$、最適IS後の$G_T$、$h$、step回路単価や非独立samplingの資源費用を確定できない。従って、**「最適化後の総T費用が頂点で少なくとも$e^{2\tau(1-p_j)}$倍不利」という一般命題は証明していない。**

頂点はcommonが抑えていた一次のnormalization負担を再導入する。これは多stepへの移送に対して強い懸念であるが、S4の$r=1$でのT利益を直ちに否定するものではない。長時間・多数stepのPR／RPE／QPEに対して成否を断定するには、実際のstep構造、global IS、$h$および回路単価を別途固定する必要がある〔未確認〕。

標準的な各stepを独立に実行する場合、pure事象の係数が$O(x)$しかないことから一stepで安価事象へ移る確率も通常$O(x)$になる、という元レビューの補助的直観は保持する。ただし、その議論を任意proposal・非IID手法に対する総費用の厳密な倍率へ昇格させない。

### 4.6 配分の効果・ISの効果・best-to-bestを混ぜない

- **配分固定でのIS効果**：同じ配分のproposal間の比較（[R1] §4、付録Aの390 relation）。
- **proposal規則固定での配分効果**：同じproposal名の配分間の比較（[R1] §3）。ただし同じ名前でも、係数・辞書・価格が変わるので実際のqは別物である。例えばP5 commonの解析ISはほぼcanonicalと同じだが、P5頂点0の解析ISは試行のおよそ3/4〜8割をT=2／4のpure事象に割り当てる（保存eventからの概算〔派生集計〕）。
- **best-to-best**：各側の最良proposal同士（保存 `mechanisms` [R5]）。
- canonicalで不利でIS併用で有利な候補は「配分単独の改善」ではない。S4では、頂点6構成のうちcanonicalでcommonを下回るのはP5頂点2（−2.26%）だけで、他の5構成はcanonicalではcommonより高い〔保存〕。

---

## 5. S4結果の比較分析

### 5.1 正式集合の確認（指示書 §4 問い1）

〔保存〕[R2][R5][R7]

- CSVは70行で、origin `V3_CONTROL_UNCHANGED` が50、`S4_CORRECTED_INTERNAL` が20。全行 `ELIGIBLE`、acceptance 1、workspace_beyond_system 1、online_classical_cost `NOT_MEASURED`。
- 5構成（config_02、03、08、09、14）は4 lawずつ（解析ISが無い）、他の10構成は5 lawずつで、10×5＋5×4＝70。適用外の5 slotは行として存在せず、0費用やnegativeとして数えられていない。理由はいずれも「positive supportにT=0の事象がある」で、保存eventではpure $Q_0=Z_0$ 事象（T=0）またはCTSのT=0事象が該当する。
- S4 `resource_map.json` の `excluded_old_internal_configurations` はconfig_04／05／10／11、`combined_comparisons.affine_relations` は7,245件（=70·69/2·3）で、旧raw-C proxyの20 lawは含まれない。`finite_registered_best=config_07:analytic_IS`、`scientific_adoption=false`、`priority_established=false`、`materiality_threshold=null`。
- 50 controlはv3のrowと完全一致した（§1.2）。S4の新lawは同じ契約（入力、精度、ε、α、bias式、proposal規則、C*）で計算されている。`supplement.saved_controls` がv3契約との一致を検査し〔source〕[R17]、371監査がC*とbaselineの不変を確認している〔保存〕[R8]。Taylor上界は70 law共通の `3125/16941456`、係数誤差は中心armで $10^{-77}$ 程度、参照armで $8\times10^{-12}$（8ρ）で、いずれも無視できる〔保存〕。

**判断**：正式比較集合は指示どおり50＋20＝70 lawで、旧20は除外され、適用外slotの扱いも正しい。

### 5.2 各構成の保存最良値（h=0、T、百万T）

〔保存〕[R2]、比は〔派生集計〕。構成内最良のproposalは本レビューのpost-hocな読取りであり、sourceが保存した選択は `mechanisms` の2組と `finite_registered_best` だけである。

| config | arm | 配分 | canonical | 構成内最良（proposal） | 同armのcommon最良との比 |
|---|---|---|---:|---:|---:|
| 00 | ordinary | common | 258.668 | 258.666（reg 16C*） | 基準 |
| 01 | ordinary | 頂点0 | 321.175 | 248.426（解析IS） | −3.96% |
| 02 | ordinary | 頂点1 | 274.357 | 223.411（reg C*/16） | −13.63% |
| 03 | ordinary | 頂点2 | 261.890 | 222.431（reg C*/16） | −14.01% |
| 04 | ordinary | 訂正KKT | 258.678 | 258.676（reg C*） | +0.004% |
| 05 | ordinary | 訂正halfway | 258.714 | 258.702（reg C*） | +0.014% |
| 06 | closed P5 | common | 189.773 | 189.772（reg C*） | 基準 |
| 07 | closed P5 | 頂点0 | 202.320 | 153.776（解析IS） | −18.97% |
| 08 | closed P5 | 頂点1 | 199.467 | 158.969（reg C*/16） | −16.23% |
| 09 | closed P5 | 頂点2 | 185.475 | 155.138（reg C*/16） | −18.25% |
| 10 | closed P5 | 訂正KKT | 188.349 | 188.293（reg C*/16） | −0.78% |
| 11 | closed P5 | 訂正halfway | 194.545 | 194.545（reg 16C*） | +2.52% |
| 12 | partial return | common | 191.990 | 191.974（reg C*） | P5 common比 +1.16% |
| 13 | closed P3+tail | common | 198.484 | 198.484（canonical） | P5 common比 +4.59% |
| 14 | matched CTS | common | 270.223 | 237.609（reg C*/16） | P5 common比 +25.21% |

### 5.3 訂正KKT／halfway：角度依存native合成価格変動の影響が大きい（指示書 §4 問い3）

**同一proposalの差**〔保存〕[R1] §3：

- ordinary：common − 訂正KKTは5 proposalで −9,178〜−9,777 T（全h≥0でcommonが劣らないL分類）、相対+0.0035〜0.0038%。common − halfwayは −32,314〜−45,399 T（L）。
- closed P5：common − 訂正KKTは +1,424,624〜+1,487,345 T（X分類。h=0でKKTが0.75〜0.78%低い）。common − halfwayは −4,771,464〜−4,777,191 T（halfwayが2.51〜2.52%高い）。

**root事象の保存価格**〔保存〕[R9b,R9d–g][R21c,R21f–t]（T数。順にchild 0, 1, 2。括弧内はRZ 1本のT数）：

| arm | common | 旧KKT | 旧halfway | 訂正KKT | 訂正halfway |
|---|---|---|---|---|---|
| ordinary | 136, 138, 140（68, 68, 68） | 132, 142, 136（66, 70, 66） | 132, 138, 140（66, 68, 68） | 136, 138, 140（68, 68, 68） | 140, 142, 136（70, 70, 66） |
| closed P5 | 136, 138, 140（68, 68, 68） | 140, 142, 144（70, 70, 70） | 140, 146, 144（70, 72, 70） | 144, 142, 132（72, 70, 64） | 140, 142, 144（70, 70, 70） |

rootの回転事象のTは「2×（RZ 1本のT）＋basis changeのT（child 0, 1, 2 で 0, 2, 4）」になっている。制御RZをCXで挟んだ2本のRZと、$V_1=R_{XX}(\pi/4)$、$V_2=R_{XX}(\pi/4)R_{ZZ}(\pi/4)$ のTである〔保存〕〔source〕[R18] `native_ir_g9`。保存cache 48角のRZ 1本のT数は、64が1本、66が7本、68が13本、70が22本、72が4本、84が1本だった〔派生集計〕[R9a–g][R21a–t]。

**内部8点の結果**〔保存〕〔派生集計〕（構成内最良の $G_T(0)$ の、同armのcommon最良との比）：

| 歩幅（commonからの方向は共通） | ordinary | closed P5 |
|---|---:|---:|
| 0.5×（訂正halfway） | +0.014% | +2.52% |
| 1×（訂正KKT） | +0.004% | −0.78% |
| 1×（旧halfway。訂正KKTとほぼ同じ配分） | −0.48% | +3.27% |
| 2×（旧KKT） | −0.93% | +2.51% |

読み取り：

1. 歩幅との単調な関係がない。ほぼ同じ配分の2点（1×の2行）の差がordinaryで0.48 pt、P5で4.05 ptあり、他のどの差よりも大きい。
2. 差はroot価格の違いでほぼ完全に説明できる。例えばP5訂正KKTでは、係数最大のchild 2（係数0.565）が140→132 T、child 0, 1が+8, +4 Tになった。rootの事象はcanonicalで全試行の約88%を占めるので、期待単価は約−1.06 T（139.49→138.43）、相対−0.76%である。
3. ordinary訂正KKTは価格がcommonと同一だったので、残ったのは設計どおりの効果だけである。canonicalの期待単価は−0.0103%（安いchildへ質量を移した効果）、一方Nは+0.014%（ν +0.0052%、Δの縮小による+0.0088%）で、差し引き+0.0037%になった。Δの縮小は、新しい合成列のstrict誤差（重み付きnative誤差 3.35×10⁻⁷→4.37×10⁻⁷）による。**設計上の微小な改善と同程度の、合成列ごとのstrict誤差上界の変化がNの保証会計に効いている。これは統計的なnoise分散を測定した結果ではない。**
4. native1Qでも内部候補は大きく動く（canonicalでordinary KKTは−5.1%、P5 halfwayは+6.4%）。RZ列ごとのClifford数の違いによる可能性が高い。配分設計の直接の寄与として切り出せていないため、Tで差がない候補を後付けで1Qの成功例と再分類しない。

**判断：** 訂正KKT／halfwayの有限比較は、**固定したnative価格下の数値的な優劣**を示す。その一方、滑らかな固定価格proxyの設計原理自体の一般的な有効・無効を識別するには不足する。旧20 lawは当初採択したsqrt(C) proxyとは異なる目的の候補であり、訂正20 lawと同一アルゴリズムの再試行とはしない。P5訂正KKTの0.75〜0.78%低下は**固定合成条件で実際に得られた条件付きT費用利益**であるが、改善をKKT理論の滑らかな効果へ因果帰属できない。変化は、角度に応じて決定的に異なるnative列価格とstrict誤差会計の影響を大きく受ける。連続内部配分family全体のno-goは示されていない。

### 5.4 頂点：回転の除去とISの組合せ

rootの保存事象〔保存〕[R21a–t]：

| 構成 | 回転事象（tan、T） | pure事象（係数、T） |
|---|---|---|
| ordinary 頂点0 | child 0：1/7、**168**（RZ 84） | child 1：0.2143、2／child 2：0.3571、4 |
| ordinary 頂点1 | child 1：3/14、142 | child 0：0.1429、**0**／child 2：0.3571、4 |
| ordinary 頂点2 | child 2：5/14、140 | child 0：0.1429、**0**／child 1：0.2143、2 |
| P5 頂点0 | child 0：約0.1483、140 | child 1：0.2023、2／child 2：0.3419、4 |
| P5 頂点1 | child 1：約0.2234、142 | child 0：0.1343、**0**／child 2：0.3419、4 |
| P5 頂点2 | child 2：約0.3776、136 | child 0：0.1343、**0**／child 1：0.2023、2 |

- 回転を一つにし、他のchildをClifford程度のpure事象にすることで、事象価格の極端な不均一（0〜4 T対約140 T）が生まれる。commonの事象はすべて約136〜180 Tで不均一が小さく、ISはほとんど効かない（P5 common：canonical 189.773、解析IS 189.782、reg C* 189.772 百万T）〔保存〕。
- canonicalでは頂点6構成中5構成がcommonより悪い。ISを併用すると6構成すべてがcommonを下回る（§5.2）。**代表的な大幅利益は「配分が価格の不均一を作る」ことと「ISがそれを使う」ことの組合せで生じる。** ただしP5のvertex_2はcanonicalでもcommonを下回るため、「どの頂点でも配分単独の改善はない」とは言わない。
- 大きさは残った回転1本の合成価格に左右される。ordinary頂点0はatan(1/7)の列が84 T（外れ値）だったため−3.96%にとどまり、他のordinary頂点（−13.6%、−14.0%）より大幅に小さい。
- T=0のpure事象（$Q_0=Z_0$ はbasis changeが不要）がある頂点1・2では解析ISが使えず、正則化IS（最良は常に最小の $\kappa=C_*/16$）にとどまる。NC下限との差は§5.5。

### 5.5 NC proxyは保存費用をほぼ完全に再現する

〔派生集計〕保存eventの係数（digital）と保存T価格から $S=\sum_ec_e\sqrt{T_e}$ をfloat64で計算し、$(S/S_{\rm common})^2$ を保存 $G_T(0)$ の比（構成内最良／同armのcommon最良）と並べた。

| 構成 | $(S/S_{\rm common})^2$ | 保存Gの比 | 最良proposal | 解析IS |
|---|---:|---:|---|---|
| ordinary 頂点0 | 0.9602 | 0.9604 | 解析IS | 可 |
| ordinary 頂点1 | 0.8315 | 0.8637 | reg C*/16 | 不可（T=0） |
| ordinary 頂点2 | 0.8280 | 0.8599 | reg C*/16 | 不可（T=0） |
| ordinary 訂正KKT | 0.9999 | 1.0000 | reg C* | 可 |
| P5 頂点0 | 0.8102 | 0.8103 | 解析IS | 可 |
| P5 頂点1 | 0.8030 | 0.8377 | reg C*/16 | 不可（T=0） |
| P5 頂点2 | 0.7835 | 0.8175 | reg C*/16 | 不可（T=0） |
| P5 訂正KKT | 0.9921 | 0.9922 | reg C*/16 | 可 |
| P5 訂正halfway | 1.0252 | 1.0252 | reg 16C* | 可 |
| P5 common／ordinary common | 0.7334 | 0.7337 | — | 可 |

- 解析ISが使える辞書では、有限confidenceの保存費用比がNC下限の比と0.1%以内で一致する。§4.4の評価（range項0.3%未満、Δほぼ一定）と整合する。
- T=0事象を含む辞書では、固定のκ格子がNC下限から3〜4%離れている。**頂点間・arm間の順位は、このproposal格子の扱いにも依存している。** NC下限で比べると、P5では頂点2（0.7835）が頂点0（0.8102）より良い。保存最良が頂点0になったのは、頂点0にT=0事象がなく解析ISが使えたためでもある。
- 〔解釈〕S4の資源差は「配分によって辞書の価格分布を変え、既知の最適ISを当てる」ことで定量的に説明でき、有限confidence会計に固有の新しい効果は見当たらない。今回の固定条件では、**実際の価格が取得済みであれば**NC $(\sum c\sqrt C)^2$ は有力な探索的診断指標になる。ただし価格取得前に同精度で候補を選べるわけではなく、T=0事象、$h>0$、有限confidence、異なるproviderへの適用は別に検討する。

### 5.6 ordinaryとclosed P5：Return aggregation固有の利益か（指示書 §4 問い5）

〔保存〕〔派生集計〕

- **RA自体の利益**：P5 commonはordinary commonに対し、h=0のTで−26.6%（Gの比0.7337）。正規化Bが1.502→1.288（B²の比0.736）に下がり、一試行の期待単価はほぼ同じ（139.9対139.5）。T・CX・native1Qのすべてで、全h≥0にわたってP5が劣らない（[51]–[53]、`RIGHT_NONWORSE_ALL_NONNEGATIVE_H`）。**これは頑健な、正規化による改善である。**
- ただしpartial return（config_12）との差はh=0で1.16%で、P5はT・CX・1Qすべてで全h≥0にわたって劣らない（[3768]–[3770]）。RAの利益の大半はpartial returnでも得られている。
- **配分の利益のarm差**：best-to-bestでP5 −19.0%、ordinary −14.0%。2×2の比で見ると、RAの効果はcommonで0.7337、最良頂点で0.6913、相互作用比は0.942（post-hoc）。NCで見ると、rootが全体の $S$ に占める割合はP5で87.5%、ordinaryで81.4%で、rootだけを変える配分はP5のほうが効きやすい。さらにP5のrootは $s/a$ が少し大きい（tan 0.749対0.714）。この二つの量的な違いで説明でき、RAと配分の間に別の機構があることを示す証拠はない〔解釈〕。
- partial returnとP3+tailには配分の自由度を与えていないので、「RAが配分の利益に必要か」は検証されていない〔未確認〕。

**判断：** この登録比較でのordinaryとP5の差は主としてRAによる正規化の変化と整合する。一方、Parent allocationとRAの間の**独立した新しい相互作用**を識別した証拠はない。partial／P3+tailに同じ配分自由度を与えた公平な追加比較は未取得である。

### 5.7 best-to-bestの定義と「約14%／19%」の保持（指示書 §4 問い6）

- 定義：sourceの `comparisons()` が、armごとにcommon側と非common側でそれぞれh=0のT切片が最小のlawを選び、差を取る（[R16] 143–156行）。左右でproposalは異なってよい。
- 訂正集合でも保存 `mechanisms` はv3と同じ2組（ordinary：config_00:reg 16C* 対 config_03:reg C*/16、P5：config_06:reg C* 対 config_07:解析IS）で、差のexact値も同一である〔保存〕[R5][R21a]。訂正20 lawはどちらの最良も更新していない。**約14.0%／19.0%は訂正集合でも数値として保持される。**
- 頑健性〔保存〕〔派生集計〕：
  - 符号：頂点6構成すべてでbest-to-bestが負（−3.96%〜−18.97%）で、単一の当たりではない。
  - 大きさ：合成価格（ordinary頂点0の84 T）とproposal格子（T=0事象）で数pt動く。
  - N：1.72倍（ordinary）、3.34倍（P5）。
  - CX：1.43倍、2.56倍。保存pairのCXは全h≥0でcommonが劣らない（[841]、[3724]）。
  - native1Q：h=0で0.850倍、0.863倍。1Qの交点は78.6 T、20.6 T（[842]、[3725]）。
  - h：保存pairのTの交点はordinary 27.31、P5 11.32（[840]、[3723]）。
- **結論**：「同じ十分予算policy・固定native価格・h=0のT切片で、頂点＋ISがcommon＋同じproposal機会を約14%／19%下回る」という限定命題は保持できる。全資源での優越、h≥0での一様な優位、配分単独の効果、一般化は保持できない。

### 5.8 $h=0$での最良 `config_07:analytic_IS` の意味（指示書 §4 問い4）

- **保存事実：** 登録70 law、$h=0$、固定十分N policy、加法的native Tモデル、pygridsynth 2.0.0・$\epsilon=10^{-6}$・seed 0、3-qubit synthetic・$r=1$の中で、`config_07:analytic_IS` が最小である〔保存〕。
- **特定二lawの交点：** `config_07:analytic_IS` と `config_09:regularized_1/16` の保存affine relation [4164] は $h\approx0.63$ T/trialで交差し、それより大きい$h$では後者が低Tとなる。また`config_07`内の別proposalが$h\approx2.84$で追い越す[4140]。これは`config_07:analytic_IS`が**全70 lawの最良たり得る$h$領域の上限**を与える。
- **未取得：** $0\le h<0.63$の**全域**で他の68 lawにも勝つかは、全70 lawの下側包絡線を取得していないため未判定。元レビューの「最良なのは$h<0.63$だけ」は、この必要条件を十分条件と誤読させるため訂正する。
- 他の入力・未登録law・既知アルゴリズム全体への最適性、実際の準備費用$h$における最良性も認定しない。

### 5.9 3-qubit syntheticからの移送（指示書 §4 問い8）

対象は既知のdevelopment fixture（$p=(1/5,3/10,1/2)$、$x=5/7$、$r=1$、状態000、$Q_0=Z_0$、$V_1=R_{XX01}(\pi/4)$、$V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4)$）で、held-outではない〔保存〕[R20]。利益が実証されたのは大きな単一step、pure事象が安いprovider、$h=0$の保存費用切片という条件の下であり（§4.5、§6.4、§6.7）、分子量子化学・DF・一般PR・QPE全体へ無条件に移送できない。多stepに対する正規化由来の不利は条件付きの数理的懸念であり、実際の総T費用の一般的no-goではない。

---

## 6. 資源と保証のレビュー

### 6.1 物理targetと誤差会計

〔source〕〔保存〕[R16][R20][R2]

- 有限target $P_5(-ixR)$ と物理target $e^{-ixR}$ は区別されている。差はTaylor剰余の上界 $|x|^6/6!=3125/16941456\approx1.845\times10^{-4}$ で、全law共通に会計される（$\|R\|\le1$ とHermite性による積分剰余で成り立つ〔数理〕）。
- axis biasは $\beta=2(\delta_{\rm Tay}+e_{\rm coef}+\sum_e\tilde c_e\delta_e)$。係数誤差は中心armで $10^{-77}$ 程度、参照armで $8\times10^{-12}$、重み付きstrict native誤差は $2.36\times10^{-7}$〜$4.48\times10^{-7}$ で、β≈3.69×10⁻⁴（ε_axis=5×10⁻³ の約7.4%）のほとんどをTaylor剰余が占める。統計残余幅は $\Delta=\epsilon_{\rm axis}-\beta\approx4.630\times10^{-3}$。
- strict native誤差はexactなatan角へのguardを含むので、角度丸めを二重に足していない（`native_angle_error_already_in_strict_guard=True`）。dyadicなqの丸めは $w=\tilde c/q$ で補償され、mean biasは0（`probability_rounding_mean_bias=0`）。
- 配分を変えると実装とstrict誤差が変わり、Δも変わる（§5.3のordinary KKTではNに+0.009%効いた）。[L1] Theorem 2のbias不変はproposalだけを変える場合の命題で、配分間には移せない〔一次文献〕〔数理〕。

### 6.2 十分Nの意味

- $N$ は各axisについて、二次moment ν（状態に依らない分散の上界）とrange Wに基づくBernstein型の十分条件で、$\alpha_{\rm axis}=1/40$ のもとでRe／Imの各axisの誤差が $\epsilon_{\rm axis}=1/200$ 以下になる。複素の失敗率は1/20。必要最小shot数でも実測shot数でもない（量子shotは0）。
- 70 lawにαを割り振らないのは、lawを量子outcomeと独立に選ぶ「選んだ一つのtask」の保証だからで、[R12] §9.3の議論は正しい〔数理〕。ただし「最良lawの選択」は予測値に基づく選択であり、量子測定の統計的有意差ではない。
- shot cap（10⁹／axis）は超えていない（最大Nはconfig_01:解析ISの3,072,408）〔保存〕。

### 6.3 T・CX・native1QとPareto

〔保存〕

- Tはnative1Qに含まれており、合算していない（sourceの `per_trial_native` はT、CX、1Qを別々に持つ）。
- best-to-bestの2組では、Tと1Qは非commonが低く、CXはcommonが全h≥0で劣らない。**頂点＋ISは (T, CX, 1Q) の同時Pareto優位にならない。**
- P5訂正KKT（config_10:canonical）はconfig_06:reg C* に対しTとCXで全h≥0劣らない（[3831]、[3832]）が、1Qではh≈12,996まで劣る [3833]。差は限定的で、固定native合成列の価格差が結果に強く影響している（§5.3）。

### 6.4 準備・読出し費用hと包絡線

- $G_j(h)=2N(\overline{C_j}+h)$ はhについて一次で、傾きは $2N$。hが大きいほどNの小さいlawが有利になる。70 lawでNが最小なのはconfig_06:canonical（680,240）で、P5のcommon／KKT／halfwayのlawが680,240〜680,727に集まっている〔保存〕。
- 保存pairのTの交点〔保存〕[R5]：

| left | right | Tの交点 h* | 分類 |
|---|---|---:|---|
| config_06:reg C* | config_07:解析IS | 11.32 | X |
| config_06:reg C* | config_07:reg C*/16 | 19.27 | X |
| config_06:reg C* | config_08:reg C*/16 | 21.39 | X |
| config_06:reg C* | config_09:reg C*/16 | 33.51 | X |
| config_06:reg C* | config_09:reg C* | 38.14 | X |
| config_06:reg C* | config_10:canonical | — | R（KKT側が全h≥0で劣らない） |
| config_06:canonical | config_10:canonical | 14,839.8 | X |
| config_07:解析IS | config_09:reg C*/16 | 0.63 | X |
| config_09:reg C* | config_10:canonical | 34.78 | X |
| config_00:reg 16C* | config_03:reg C*/16 | 27.31 | X |
| config_00:reg 16C* | config_03:reg C* | 22.86 | X |
| config_00:reg 16C* | config_02:reg C*/16 | 19.53 | X |

record番号は順に [3723]、[3726]、[3738]、[3750]、[3753]、[3831]、[3390]、[4164]、[5430]、[840]、[843]、[828]。

- 読み取り：保存pairの11.32は「全頂点がcommonに勝つ$h$の上限」ではない。P5頂点2の特定正則化lawは、`config_06:reg C*`との**二者間比較**では$h\approx33.5$～38まで低Tの例がある。$h$が増えるとNの小さいlawは相対的に有利だが、**全70 lawの大域的な勝者は包絡線を計算しない限り確定しない。**
- 全70 lawの最適包絡線は保存されておらず、本レビューでも計算していない〔未確認〕。上の表は保存pairの読取りで、包絡線の代わりにはならない。
- 実際の$h$は未取得である。状態準備、読出し、決定論的部分など、各試行に共通に必要なT費用は該当する場合$h$に入る。今回の**指定したlaw同士**の比較では数十Tで利益が反転する例があるが、これだけで全頂点・全proposalの利益が消える$h$を確定できない〔解釈〕。

### 6.5 $h=0$のT切片は安価事象を優遇するが、費用が無限に下がるわけではない

〔数理〕固定したevent辞書に$C_e=0$の事象が少なくとも一つあり、正費用の事象も存在するとする。正費用の集合を$I_+$とし、その全proposal質量を$\varepsilon\to0^+$へ縮め、各$e\in I_+$について$q_e=\varepsilon r_e$（$r_e>0,\ \sum_{I_+}r_e=1$）とする。残りの質量をzero-T事象へ割り当てる。これは**連続確率の形式的な極限**であり、S4の固定160-bit dyadic proposalや有限shot capで無条件に実現できるlawではない。

このとき

$$\nu=\sum_e\frac{\tilde c_e^2}{q_e}=O(\varepsilon^{-1}),\quad
W=\max_e\frac{\tilde c_e}{q_e}=O(\varepsilon^{-1}),\quad
\overline C_T=\sum_eq_eC_e=O(\varepsilon).$$

残余統計幅$\Delta>0$を一定に保つ固定辞書のproposal変更では、十分Nも$O(\varepsilon^{-1})$で増大する。そのため

$$G_T(0)=2N\overline C_T$$

は**一般にはゼロまで下がらず、正費用のevent群によって決まる有限の非ゼロ極限**へ近づく。Cauchy–Schwarzから

$$\nu\overline C_T\ge\left(\sum_{e\in I_+}\tilde c_e\sqrt{C_e}\right)^2$$

が成立する。ただし有限confidenceのrange項があるので、このNC下界だけで厳密な$G_T(0)$極限値を確定したことにもならない。

一方$h>0$なら$2Nh\to\infty$となり得る。**問題は「$h=0$の期待Tが無限にゼロへ下がる」ことではなく、shot数と共通準備費用を含む実際の総資源の価値を、$h=0$の単一切片が十分に表さない場合があること**である。

現在のParent allocationには$\sum_i z_i=a$という制約があり、任意のpure恒等事象を自由に追加するfamilyではない。頂点配分はzero-T／低Tのpure事象を生むが、それにより無条件の一般的最適性が得られるわけではない。元Claude原稿が示した非pairing rootの「ordinary common NC比約6%」は**その原稿の保存価格からの探索的な概算**としてのみ保持し、独立に認証された実行可能lawの費用とも、S4正式候補の優越証拠とも扱わない。

### 6.6 古典費用

- 保存されているのは、候補生成CPU（各arm約0.02秒）、law／ledgerのCPU（config_04／05で約1.8秒、config_10／11で約0.3秒）、実行全体のwall 9.45秒などで、いずれも共有の資源表作成の費用である〔保存〕[R3][R5]。
- 選んだ方式を実際に回すときのonline sampling費用は、全lawで `NOT_MEASURED` であり、0とはみなさない。global ISには全事象の価格表が要る。このfixtureでは事象数が63（P5）〜273（ordinary）と小さいが、大きい問題での表の取得・初期化・samplingの費用は未測定である〔未確認〕。

### 6.7 native合成への依存

- 価格はpygridsynth 2.0.0、ε=10⁻⁶、seed 0、`up_to_phase=false`（Wを保持）、符号付き逆は実列のadjoint、制御位相はancillaのS／Z／S† で会計されている〔保存〕[R20][R19]。
- RZ 1本のT数は固定tool・precision・seedの下で角度に応じ主に64〜72、外れ値84へ変化し、この**決定的な角度依存native価格差**が内部候補の有限比較に大きく影響した（§5.3）。別の$\epsilon$・seed・合成器での順位安定性は未測定であり、「容易に符号が入れ替わる」は仮説に留める。頂点＋ISの6構成で符号が揃うという保存事実から、別provider全体の頑健性までは保証できない。
- T=0事象の有無はproviderの性質（$Q_0=Z_0$ はbasis change不要）で決まり、解析ISの適用可否と頂点間の順位を左右した（§5.4、§5.5）。
- 頂点の利益は「pure事象の価格 ≪ 回転事象の価格」に依存する。ここではbasis changeが0〜4 T、RZ 2本が約136 Tである。basis changeが高価な入力（例えばDFの軌道回転のように、$V_i$ 自体に合成回転が必要な場合）ではpure事象も高価になり、利益は縮むか消えると予想する〔解釈〕〔未確認〕。
- 費用は各gateの加法計数で、whole-circuit compilerの最適性は主張されていない。同一角の回転をまとめて安くする実装（[L4] は同一角の回転にHamming weight phasingを使う議論をしており、[L7] はcatalystを使う）では、角度が揃うcommonが有利になり得る。頂点や内部配分は角度の種類を増やす。現在のモデルはこの効果を含まない〔一次文献〕〔解釈〕。

---

## 7. 先行研究との比較・新規性評価

### 7.1 要素ごとの対応

| 要素 | 近接する一次資料と確認箇所 | 関係 | 残る差 |
|---|---|---|---|
| 任意zの平均保存 | [L4] App. C Eq.(C4)–(C5)、[L3] Suppl. Note 5 Eq.(S15)、[L5] Eq.(74)–(75) | 共通角は特殊点として一致〔一次文献〕。一般zは線形分割＋Eulerの直接の帰結〔数理〕 | 新しい数学はない |
| 固定辞書の費用付きIS | [L1] Eq.(7)–(10) Theorem 1、Eq.(22)–(23) Theorem 2、Eq.(60) | S4の解析IS lawはTheorem 1の下界に一致し、保存Gの比を0.1%で再現（§5.5） | T=0事象・有限confidence・hでの扱いは、本レビューで確認した [L1] の範囲の外 |
| 配分（辞書の変更）で正規化と安価なgateを交換 | [L2] Statement 1–2・Eq.(3)、(6)・低T数ライブラリでの回転合成、[L6]（三つの離散角）、[L7]（±Δとπの二種類の角、FTでのT-state推定）、[L8]（回転角を自由に選び、減衰因子とshot増で補う） | 「高価な連続角の回転を安い（Clifford・離散角・低T）操作へ置き換え、正規化とshot増で補う」という機構は既知〔一次文献〕 | Parent allocationは、Euler-paired Taylorのrootで恒等質量を配るという特定の構造への適用 |
| 固定価格KKT | [R15] §8.3、[R14] §4（標準的な凸最適化） | 標準的な凸最適化 | 本レビューの摂動評価では、この価格構造での効果は $10^{-5}$ のオーダー |
| Return aggregation | repository内のG系列 [R15]（外部との対応は本レビューでは未確認） | — | 正規化改善は保存データで頑健。partial returnとの差は小さい |
| strictなfinite-bit／native会計 | — | 実装の品質 | 新規性の根拠にはならない |

### 7.2 指示書の四分類による評価

- **新しい数学：** 今回の平均保存恒等式と固定価格KKTは既知の線形分割・Euler恒等式・標準凸最適化の直接の帰結であり、調べた範囲では独立の一般定理と認定できない〔数理〕〔一次文献：Claude確認〕。
- **具体的な情報アクセス・生成法：** rootだけを変えるため、既存のP5 samplerに局所係数処理を加えて候補を生成できるという実装上の差は残る [R14] §7。ただしglobal ISに全eventの価格表が必要で、その取得・online費用での優位は実証されていない〔未確認〕。
- **新しい資源上の現象：** **独立した新現象の証拠は現在不足している。** 保存されたT差は既知の費用付きISと安価な事象の生成で定量的に説明可能である一方、調査対象外の情報アクセスやコンパイルを含む全手法との完全同値・普遍的no-goは示していない〔解釈〕。
- **既知機構の特定構造への組合せ・応用：** 現在の主要な位置付けはこの分類である。価値が残るなら、Euler-paired Taylorのrootにおける適用条件（$x,r,h$、pure事象／回転価格比）の定量化と実装可能性の識別である。

### 7.3 限界

指定5本と追加4本の該当箇所を読んだだけで、網羅的なpriority調査ではない。上の分類は「本レビューで確認した範囲で既知の機構に含まれる」という判断であり、同一の手順が既に公表されていることの証明ではない。逆に、完全一致を見つけられなかったことを新規性の根拠にもしない。

---

## 8. 重大な懸念と判断保留事項

| 重要度 | 事実 | 根拠 | 結論への影響 | 解決策 |
|---|---|---|---|---|
| 高 | 内部候補（旧・訂正とも）の差は固定条件下の角度依存T価格変動の影響が大きく、滑らかなproxyの効果（$10^{-4}$〜$10^{-5}$）を単独識別できない | §5.3、§4.3(e)。ほぼ同じ配分の2点で0.48%／3.9%の差 | KKT／halfwayの成績を滑らかな内部配分設計の一般的成否に使えない。P5訂正KKTの0.75〜0.78%は固定native価格下の実際の利益だがproxyの因果効果とは区別する | 新しい注記文書で旧20／訂正20を区別し、角度依存価格変動への感度を示す。Bを当面優先しないことを提案する |
| 高 | 頂点＋ISの利益は、r=1、x=5/7、h≈0、pure事象が安いproviderの組合せでのみ示された | §4.5（多stepで $e^{2\tau(1-p_j)}$）、§6.4（交点11〜38 T）、§6.7 | PR／RPE／分子への無条件移送は支持されない。normalization下界だけで総T費用の指数的不利を確定したものではない | 適用境界を紙上で導出する（§10 C5）。新しい実行は境界が実用域を含む場合だけ |
| 高 | 主指標 $G_T(0)$ は費用0の事象に退化的に有利 | §6.5 | $h=0$切片の優位はその指標では有効だが、実総資源優位としては$h$とshot負担を指定する必要がある | 主指標を「hの範囲付き」に改めるか、目標taskのhを先に固定する |
| 中 | T=0事象を含む辞書では、固定κ格子がNC下限から3〜4%離れる | §5.5 | 頂点間・arm間の順位（最良が頂点0か頂点2か、P5とordinaryの差）が格子に依存 | 保存値のみのNC再集計で差を明示する（§10 C2）。格子の追加は新しい科学設計として別に扱う |
| 中 | hの包絡線が未取得で、保存pairの交点だけでは「頂点が勝つ範囲」を代表できない | §6.4（0.63〜38 T、config_10のR分類） | 「h≈11まで」という記述は過小にも過大にもなり得る | 保存値のみの包絡線計算（§10 C4）。post-hocと明記する |
| 中 | 費用モデルが加法的・角度ごとで、角度を揃える実装の利点を含まない | §6.7、[L4]、[L7] | commonの真の費用を過大に見積もっている可能性 | 結論の適用範囲に明記する。必要なら別モデルで感度を確認（新しい科学設計） |
| 中 | 調査した範囲で独立新規性の証拠が不足 | §7 | 新しい一般原理として主張できず、特定構造への応用価値を条件付きで検討 | 限定的成果として整理する（D） |
| 低 | RAの必要性は未検証（partial／P3+tailに配分の自由度なし） | [R12] §6.2、§5.6 | 「P5が不可欠」とは言えない | 継続しないなら不要。続けるならpartialにも同じ配分を与える |
| 低 | online古典費用・全事象表の取得費用が未測定 | §6.6 | end-to-endの優位は主張できない | 大規模入力へ進む場合にだけ測る |
| 低 | 監査PASS（918／371）は来歴と保存値の一致を示すが、proxyの妥当性は示さない | KKTの不一致は監査ではなくレビューで見つかった [R13] | 監査を科学的妥当性の証拠に使えない | 監査と科学検証を報告上で分ける（現状でもおおむね分けられている） |

保留事項（本レビューでは判断しない）：目標taskのhとstep構造、DF入力でのbasis changeの価格、全hの包絡線、別の合成器・ε・seedでの順位、online古典費用、網羅的なpriority。

---

## 9. 研究方向の選択肢比較と推奨

### A：頂点配分＋ISの資源利益を、情報条件・費用条件付きで発展させる

- **新しい科学的問い**：価格の不均一を作る分解＋ISは、多step・h>0・現実のbasis change価格のもとでも総費用を下げるか。下げるならその条件は何か。
- **既存研究との差**：機構は [L1]＋[L2]／[L6]／[L7]／[L8] で既知。差は、Euler-paired Taylorのrootという構造と、適用条件の定量化だけである。
- **現在の証拠**：単一のsynthetic fixture、r=1。6頂点すべてでbest-to-bestが負で、NCで説明できる。交点は11〜38 T。多stepでは正規化由来の二次moment下界に指数因子が現れるが、総T費用比の一般下界ではない。
- **最小追加検証**：紙上の適用境界の導出（§10 C5）と、保存値のみのNC再集計（C2）。新しい計算は、境界が実用域を含むときだけ行う。
- **成功条件（未採択案）**：目標taskの$x,h,$価格比、step数のもとで、有効な不確かさ・感度の定義に照らして意味あるT低下が残り、関連する既知対照とも比較可能であること。「価格差の2倍」は現時点では科学的閾値として採択しない。
- **中断・縮小条件（未採択案）**：対象taskにおける適用条件が理論的に満たせない、または定めた費用規則の下で改善を支持できない場合。$h$が数十Tという値はこのfixtureでの指定pairの交点であり、普遍的な失敗閾値ではない。
- **予想リスク**：高い。現時点で他provider・多step・実$h$へ移送できる証拠はない。新規性も未認定。
- **必要な計算資源**：紙上と保存値だけならCPU数秒。移送確認をするなら、新しい合成keyが数十本程度。
- **研究としての着地点**：「単一step・小さいhでだけ効くT削減の条件」を述べた限定的な技術ノート。

### B：内部配分（KKT surrogateを含む）をさらに最適化する

- **proxy最適化とnative有限confidence費用最適化の違い**：今回の固定価格proxyは境界での価格の不連続を表せず、対象価格構造では滑らかな改善見込みが$10^{-5}$程度と小さい（§4.3）。native費用には角度依存の非滑らかな価格差があり、これを扱う別のangle-aware設計の価値は今回の有限比較から否定も肯定もできない。
- **既存研究との差**：本件の固定価格KKT式自体は標準的な凸最適化。別の実装条件・価格aware手法の差分は未評価。
- **現在の証拠**：旧・訂正の8内部候補では歩幅との単調関係が見えず、今回の角度依存価格変動が支配的に作用する（§5.3）。これは固定されたtool／precision／seedでの観測であり、確率的ノイズ分散ではない。
- **最小追加検証**：現行の固定価格KKT近傍へ同条件で追加候補を増やす必要性は低い。別のangle-aware設計を検討するなら独立の研究設計が必要。
- **成功条件（未採択案）**：別条件に展開するなら、価格取得の情報条件と新規性を明確にした上で、固定proxy効果を超える改善が現実の費用・精度条件で持続すること。
- **中断条件の提案**：本fixtureでのsmoothなKKT近傍探索は、追加投資の情報価値が小さいと判断して優先順位を下げる。全内部配分のno-goとはしない。
- **予想リスク・計算資源**：明確な角度費用モデルなしに続けると、特定合成列の価格差を後追いする探索になりやすい。
- **着地点**：「この価格構造では固定価格KKTが予測する改善は小さく、native費用は角度依存変動に敏感」という限定的な技術的知見。
- 補足：角度をCliffordやTの角に合わせて安くする「角度を意識した配分」は別の問いで、[L2]／[L6]／[L7] と重なる。BではなくA／Dの文脈で扱う。

### C：Return aggregationとの結合を深掘りし、ordinaryとの差を識別する

- **新しい科学的問い**：配分の利益にRAが必要か。
- **既存研究との差**：RA自体はrepository内の系列。外部との対応は未確認。
- **現在の証拠**：RAの正規化改善は頑健（全h、全資源）だが、partial returnとの差は1.16%。配分利益のarm差は量的に説明がつく（§5.6）。
- **最小追加検証**：partial returnとP3+tailに同じ頂点配分を与える比較（新しい計算）。
- **成功条件（未採択案）**：partial／P3+tailに同じ機会を与えても残るP5固有の資源利益が、予め定めた感度・費用条件に対して安定に示されること。
- **失敗・終了条件**：Aを続けないなら、この問いは自動的に優先度を失う。
- **予想リスク・計算資源**：中程度。情報価値はAの結果に従属する。新しい合成keyが十数本。
- **着地点**：RAの正規化利益の確認（既に得られている）。

### D：Parent allocationを限定的成果として整理し、他のTrack B改善へ優先順位を移す

- **整理する内容**：(1) 平均保存の配分family（既知式）、(2) 固定価格proxyの小さい改善と角度依存native費用の差、(3) 頂点＋ISによる限定的T利益と$h$・多step・価格比による適用条件、(4) 保存条件でのRAの正規化改善、(5) 実価格でのNCが有限confidence費用比を近似するという探索的知見。
- **次の優先順位の決め方**：候補を選ぶ前に、目標taskの現実的な費用構造（step数とstepあたりのx、h、basis changeと回転の価格比、角度依存価格変動）を既存資料（D1〜D3、DF系列）からまとめる。そのうえで、(i) 多stepで正規化を悪化させない、(ii) 角度依存価格変動より大きい、(iii) 現実のhで残る、の三条件を満たす候補に絞る。paired randomization、回路合成、sampling、構成前処理のどれを選ぶかは、この整理の後に決める。
- **既存研究との差・現在の証拠**：本レビュー全体。
- **予想リスク**：低い。
- **計算資源**：文書作業のみ。
- **着地点**：短い技術ノートと、次の方向の選定基準。

### E：追加投資の情報価値が低ければ、現在の探索を区切る

- **未採択の研究推奨：** Parent allocationの同一fixtureに対する新しいone-shotは当面中断し、追加実行の要否を対象taskの費用条件から再判定する。現時点で一般的no-goや恒久的な探索終了を宣言しない。
- 区切ることは、技術停止を性能negativeへ書き換えることではない。既存の結果・closure・markerはそのまま残す。

### 推奨順位と根拠

1. **D＋E（条件付き第一候補）**：限定的成果として整理し、同一条件の追加one-shotを当面中断するよう提案する。研究方針の正式変更は利用者承認後。根拠は §5.3、§4.5、§6.5、§7。
2. **Aに関係する適用条件の検討（§10 C2〜C5）**：有用な可能性があるが、C2–C4は**新たな派生数値分析**であり、実行前に範囲と承認を確定する。C5は解析的な仮定を明示して別途検討する。
3. C、B：優先しない。

「拡張前の整理・探索中断」を推すのは、現状の同一fixtureで候補を増やしても新しい科学的判断に直結しにくいためである。ただし判断を変え得る情報が適用境界**だけ**だと証明したわけではなく、現実の$h$・step構造・native価格・情報アクセスも残る。論文化の可能性は、この結論からは「既知機構の条件付き適用例」以上のものを見込みにくい。ただし論文化を優先する理由もないので、これは判断材料にしていない。

---

## 10. 次のCodex作業案・承認とmandatory STOP

### 10.1 研究方針の推奨（未採択）

1. 本fixtureでの追加native one-shotを自動継続せず、Parent allocationの得られた成果と限界を文書に整理する。
2. 固定価格KKT／halfway近傍の内部配分探索は、直近の優先研究テーマから外すことを提案する。これは**全ての内部配分のno-go**ではない。
3. 頂点＋ISの限定的なT利益は保持し、対象となるPR／RPE taskの$h,r$、pure／rotation価格を把握してから次の拡張価値を判定する。
4. 新しい研究主線・実行契約・閉鎖判断の採択権限は利用者と明示的に承認された科学レビューにあり、本書の作成だけではsourceやmarkerの状態を変更しない。

### 10.2 次のCodexへ任せ得る作業の区分

**C1：技術文書の整理（新規科学計算を含まない）。** 旧raw-C proxy20と訂正sqrt(C) KKT20を別実装の有限候補として明記する。頂点＋ISの14%／19%に「$h=0$、十分N、N/CXの増加、$r=1$、development synthetic、固定native tool」という条件を併記する。既存原記録を書き換えず、追補の注記文書に限る。自律的な研究GO/STOPは付けない。

**C2：NC proxyのexact再集計（保存データからの新しい科学的数値分析。別途承認が必要）。** 70 lawに対し$S=\sum_e\tilde c_e\sqrt{C_e}$を外向き区間で求め、保存$G_T(0)$比、range項比と比較する。Claude原稿のfloat64派生集計を数値的に再認証できる可能性があるが、**実施済みとはしない**。解析ISとの差0.5%等は閾値案であり未採択。

**C3：保存cache価格の角度感度整理（内容に応じ承認を区別）。** 既存keyのpath・hash・T数を転記する索引作業は通常の技術資料整備である。一方、価格分布の新しい再集計、近い配分点の反事実的比較、価格変動を用いた科学的閾値の決定は**新規派生数値分析／科学判断**となるため別途範囲と承認を固定する。「価格差の2倍」は統計的2σではない未採択案である。

**C4：全$h$包絡線（保存係数からの新しい科学的数値分析。別途承認が必要）。** 70 lawの既存切片・傾きから、T・CX・native1Qを**別座標**として下側包絡線を計算する。結果後の探索的解析であること、個別二者交点からは全域の勝者を認定しないことを明記する。**本レビューでは未計算**。

**C5：多stepの適用境界（紙上の解析を先行、登録入力の新規数値予測は別途承認）。** 積推定量と非IID手法の対象区分、$x,r,h$、pure価格・rotation価格、step間のweight構成を固定し、頂点＋ISがcommonに勝ち得る条件を解析的に導く。単なるnormalization下界を総T費用の倍率と同一視しない。数値代入やregistered$\tau$のforecastは新たな科学的数値分析として別途承認を要する。

C1以外は**自動実行を推奨する指示ではない**。既存S4 source／result／authorization／消費済みmarker・STOPを不変に保つ。

### 10.3 将来の新しい科学計算（実行承認のない設計候補）

多stepなどの追加科学計算が本当に必要と研究上判断された場合、少なくとも以下を事前固定する必要がある：

- 物理target／有限演算子target、step積推定量の保証、$r$と$\tau$、入力provider、精度・failure定義。
- 各条件に公平なcommon／非common／既知代替、確率proposal機会、pure・rotation価格、$h$、classicalオンライン費用の扱い。
- 精密な「改善した」判定規則、数値区間、source/contract manifest、resource caps、one-shot、retry=0、mandatory STOP。
- 新規registered candidate generation、N forecast、合成、guard、workerは**新source凍結後のユーザー別途承認まで禁止**。

Claude原稿§10.4の探索例――$r\in\{1,2,4,8\}$、$T_{RZ}\in\{64,68,72\}$、$h=50$ T、$r=4$でのgate、CPU60秒、二段階の価格モデル→native価格検証――は**未採択の実験設計案**として記録する。入力値・閾値に新たな科学的根拠がつかない限り、自動実行条件・一般的停止基準として用いない。元原稿の「2倍」「0.5%」も同様である。

### 10.4 次にGPTへ戻る条件

- 新しい科学的な研究方向・主評価指標・比較契約を採択する必要があるとき。
- 既存の結論を変更し得る数学的反例、保証の疑義、保存資料の意味論的不一致が判明したとき。
- ユーザーの別途承認によるC2–C5の科学的分析や新規計算の結果が揃い、その解釈に新しい重要判断が必要になったとき。
- 現実的なPR/RPE taskの$h,r$、native費用構造を整理し、Parent allocationの適用可能性や新規性判断が変わり得るとき。

**mandatory STOP：** 原結果の変更、旧runのretry、新scientific worker、native synthesis、operator guard、unapproved re-ranking/NC計算/全$h$包絡線、追加seed／precision／provider／DF／分子／GPU／LP／trajectory／whole-circuit compile等を自動開始しない。レビュー開始承認と計算実行承認を分離する。

---

## 11. 主張できること／できないこと

### 主張できること

| 主張 | 根拠 |
|---|---|
| 任意の非負配分で理想平均（有限P5）は保存される | 〔数理〕§4.1。既知のEuler恒等式の直接の帰結 |
| commonは配分family内でL1正規化を最小にし、非commonの利益は単価の低下との交換に限られる | 〔数理〕§4.2、〔保存〕全非commonでBが増加 |
| S4 sourceは採択した√C重みのKKTを正しく実装し、保存配分はそのproxyのKKT点に $10^{-20}$ の精度で一致する | 〔source〕〔保存〕§4.3 |
| 正式比較集合は50＋20＝70 lawで、旧20と適用外5 slotは正しく扱われている | 〔保存〕§5.1 |
| h=0・固定十分N・加法的native Tモデル・このsynthetic fixtureで、頂点＋ISはcommon＋同じproposal機会よりT切片が約14%（ordinary）／19%（P5）低い | 〔保存〕§5.7 |
| 代表的な大幅T利益は、安価事象を作る頂点配分とISの組合せによって生じる | 〔保存〕§5.4。P5 vertex_2はcanonical単独でも改善する例外 |
| その資源差は、実価格での固定辞書ISのNC proxyにより今回の保存条件で定量的に近似できる | 〔派生集計〕〔一次文献：Claude閲覧〕§5.5。float64に基づく探索的分析 |
| 訂正KKT／halfwayの費用差は角度依存native価格変動の影響が大きく、滑らかなKKT proxyの因果的な有効性を一般化できない | 〔保存〕〔派生集計〕〔数理〕§5.3。価格は固定条件下の決定的な値 |
| closed P5はordinaryに対し、正規化の低下によりT・CX・1Qで全h≥0にわたって劣らない | 〔保存〕§5.6 |

### 主張できないこと

| 主張 | 理由 |
|---|---|
| Parent allocationの平均保存式が独立の新しい一般定理である、または全他手法と異なる資源原理が既に実証された | §7。調べた範囲では根拠不足、網羅的priorityも未実施 |
| 滑らかな固定価格KKTがordinary／P5のnative総費用を一般的に改善・悪化させる | §5.3。有限角度価格の寄与を分離できていない |
| P5訂正KKTの0.75〜0.78%低下が、角度依存価格変動と独立したsmooth KKT最適化の効果である | §5.3。固定条件での費用利益自体は保存事実 |
| 頂点＋ISが全資源・全hで優位 | §6.3、§6.4 |
| 14%／19%がPR／RPE／QPE／分子DFでも得られる | §4.5、§6.4、§6.7、§5.9 |
| `config_07:analytic_IS`が$0\le h<0.63$全域または任意の$h$で最良である | §5.8。$h\approx0.63$は特定二lawからの全体最良領域の**上限**であり、下側包絡線は未取得 |
| 連続配分familyのnative最適解、またはno-go | 有限候補だけであり、proxyは境界を表せない |
| RAが配分の利益に不可欠 | §5.6（partial／P3+tailに配分の自由度がない） |
| end-to-endの実行時間・総量子費用での優位 | online費用は未測定、hは未取得 |
| 有限confidence会計に固有の新しい最適化効果 | §4.4、§5.5（NCと一致） |
| 全内部配分のno-go、研究系列の恒久的終了、科学的GO・独立新規性の認定 | 本書は一般的no-goを示さず、探索中断・縮小は未採択の研究推奨 |

---

## 12. 出典と資料取得ログ

### 12.1 repository資料（固定commitからgitで取得。bytes／SHA256は本レビューでの再計算）

| ID | 資料 | commit | bytes | SHA256 |
|---|---|---|---:|---|
| R1 | [S4最終比較Evidence Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md) | `53c4831` | 215,202 | `cfbd5f3c4fc8f94a0b805e99dc69c5e8a4dcc10c58f4f824ca295070f0f8149f` |
| R2 | [訂正済み70 law索引CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/artifacts/track_b_parent_allocation_review_packet/2026-10-15/s4/law_comparison_index.csv) | `53c4831` | 152,587 | `140ffa352df00673be09d4178782320199901077746a244ebb6250c66596539e` |
| R3 | [S4補完実行結果・GPT引き継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md) | `fe0f3f0` | 14,720 | `e334134ed6e33a90c091391a10e5bbdfb3331f4476ab751150aaba9e7e5be2c7` |
| R4 | [S4 KKT修正・技術準備報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/tracks/algorithm_codesign/parent_allocation_s4_kkt_repair_and_supplement_preparation_20261015.md) | `fe0f3f0` | 11,881 | `a975c0c104ae9cc22ac35a6fc2b5a968acebf420bde3ed2626d91e15de46bf9f` |
| R5 | [S4 resource_map.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/resource_map.json) | `fe0f3f0` | 3,228,746 | `8281cf66c1ce7b5abf1a46ba6d4c4725c994eabf3dadcc143a9877387cc117c5` |
| R6 | [S4 evidence_manifest.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/evidence_manifest.json) | `fe0f3f0` | 11,563 | `4760a476006d977c1e636626dfdc794690020ebb1809d0202323df95dc404650` |
| R7 | [S4 inherited_rows.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/inherited_rows.json) | `fe0f3f0` | 428,444 | `9e51307ba5a3fe918606da681972df9d8e551bbbcb37f1704aa0cb8f01ef1db6` |
| R8 | [S4保存監査（371 checks）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json) | `fe0f3f0` | 90,750 | `d832c80fd33807f4de0f56f8f83c8755396829c7fc35a05e9f6367613de56103` |
| R9a | [S4 candidate_freeze.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/candidate_freeze.json) | `fe0f3f0` | 10,088 | `c9624113afaec7af550842ef1fc25eef3dd347c8c1dbfc07ce909984360d7e0c` |
| R9b | [S4 guarded_cache.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/guarded_cache.json) | `fe0f3f0` | 57,972 | `24c72a9855a73bf5c86ef18907fa2b3bb894dae2f9e9b557b961d53b5c55f57c` |
| R9c | [S4 inventory.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/inventory.json) | `fe0f3f0` | 24,136 | `b1d0ce222e52afacd5bbd770938522824809a930bf6ab5b3f26f4111820831bb` |
| R9d | [S4 events/config_04.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/events/config_04.json) | `fe0f3f0` | 644,324 | `90234a8357e8e9f283364e99e97e3f319d2fb87d8bc445f9b398a6fc53853bb6` |
| R9e | [S4 events/config_05.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/events/config_05.json) | `fe0f3f0` | 644,340 | `e330e3fdf3777aeb802490c94da936a50087f7f53e0c97ed81110676511a6cbc` |
| R9f | [S4 events/config_10.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/events/config_10.json) | `fe0f3f0` | 161,223 | `660c1b6d38f835f1064714d1234d3d8e11be4d5f111e60ef7271b8292a17ba79` |
| R9g | [S4 events/config_11.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/events/config_11.json) | `fe0f3f0` | 161,326 | `03e005ee0da6279cfa0d1204a39b923d98c1fcf701b1cb314c265f52c97387a8` |
| R10 | [v3比較Evidence Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/17bfcaac5a53d648ee94de0067de24d719ed3911/docs/tracks/algorithm_codesign/parent_allocation_v3_comparison_evidence_packet_20261015.md) | `17bfcaa` | 20,509 | `b72e7e9b13bf0c61962060cb210be28efad169563c5532a11039ad58d8db7c07` |
| R11 | [v3実行結果・監査報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md) | `cd05523` | 10,949 | `b05f6d5b97cb15e1d5744702ba3a5ba01fb69554ccaec262ca23f9966b00e8d3` |
| R12 | [v3以前に採択した科学的設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/research/track_b_parent_allocation_scientific_design_review_20261014.md) | `cd05523` | 41,911 | `11d1e389190091221fd2a282e931c229a52bda1dcc8c8f30ec5d3bcf7aa853bf` |
| R13 | [前回GPT科学レビュー（KKT不一致の指摘）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/research/track_b_parent_allocation_v3_scientific_review_20261015.md) | `fe0f3f0` | 39,180 | `450bdc2e34662f21e8416fbd2d0c1a65590eb4a840627da40c5141af01fe0584` |
| R14 | [近接先行研究との対応監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf1366ac5832136ab8038d160e9a0285da69e79e/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md) | `bf1366a` | 23,267 | `53e8cd9c3943f72a217fade27f054ef1c3a8d7660c34407c39b0bcf6fe3f4210` |
| R15 | [Track B研究方向GPTレビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md) | `4d80b45` | 44,878 | `16af1fa58ec3d0b12dba5e2e1f8fda574a5723eeff3875e16688ef4ebe445e51` |
| R16 | [S4 parent_experiment/design.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_experiment/design.py) | `4ccc183` | 10,580 | `11c5c37a0a8d0b7a9c5d1760463b33377fda86912b13a6407976db8c9777caca` |
| R17 | [S4 parent_experiment/supplement.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_experiment/supplement.py) | `4ccc183` | 11,816 | `2e06d01ece8d0e8b2da2059dcf4de6e1cf3a051318eff664b668bb08039a49c5` |
| R18 | [parent_allocation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py) | `4ccc183` | 25,788 | `6ebecaf81a446dcdeaafa89b9bd0018ac8d11d23695da57d58b5683b99a4709d` |
| R19 | [parent_allocation_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation_native.py) | `4ccc183` | 13,351 | `f2a7ddbf313aaca7a7d7e9e9adeb991da00238c4a55aac5698112ba20844188e` |
| R20 | [S4固定実行契約 execution_contract_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v4/execution_contract_v4.json) | `fe0f3f0` | 12,608 | `f76a10c6f04972494d5b7607a1a32f425f47c52b5bdb6083f8a983a827056e7d` |
| R21a | [v3 resource_map.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json) | `cd05523` | 3,647,912 | `a781242e3136c51253862d9569388c0ac7fc9093d4c1f28c2fbe378523ba94fe` |
| R21b | [v3 candidate_freeze.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/candidate_freeze.json) | `cd05523` | 14,060 | `7f182ce8f3bb98b6473cb8bfc12ba25f1c7a0316d6ec41e3b732b2d1f53c85ee` |
| R21c | [v3 guarded_cache.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/guarded_cache.json) | `cd05523` | 94,722 | `a955b48bdb3fde8da695047e5d0a065711ab88e08ee94279ac14c8005f56b51e` |
| R21d | [v3 inventory.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/inventory.json) | `cd05523` | 37,393 | `08be791ba592d506947bc97b89782c6766da31c43791e7e03e704701bcb91101` |
| R21e | [v3 evidence_manifest.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/evidence_manifest.json) | `cd05523` | 31,706 | `6352e5103e306193d34dd762537f360fe13c52bcecb53aa8406c31a2bb80bdc4` |
| R21f | [v3 events/config_00.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_00.json) | `cd05523` | 642,658 | `3384b78fd27848ae070a87393e62c8263058c615aa81606f37e781508c8aad61` |
| R21g | [v3 events/config_01.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_01.json) | `cd05523` | 642,622 | `06b8c9972363dcc373b7ddbcbd22c10ca7516af49d4b4e312cc388a939e6d7dc` |
| R21h | [v3 events/config_02.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_02.json) | `cd05523` | 642,618 | `3e9ab7fb8a7bff59758db63d5b85e54aa9ac3b90dbac5472aff1e89227c4f6f8` |
| R21i | [v3 events/config_03.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_03.json) | `cd05523` | 642,622 | `e1faee2a9b8b3ff119568f2a8a2f0e830b129e28b6ba40c7bc924e5eba1fd43d` |
| R21j | [v3 events/config_04.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_04.json) | `cd05523` | 644,332 | `75246dfd9ae0ad9315406b1fa8fcd0fd482ce03d97d218a41592a77bde634e60` |
| R21k | [v3 events/config_05.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_05.json) | `cd05523` | 644,347 | `182c42c752054af799270497dec14211625b49987a16921062a910a3b72fc8b9` |
| R21l | [v3 events/config_06.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_06.json) | `cd05523` | 159,729 | `e64b9092dee9d1ce0a2a1ff584c1034cea4165c2a77e7897ba539f406c93e317` |
| R21m | [v3 events/config_07.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_07.json) | `cd05523` | 159,529 | `bf638b1335ddf37dd3e0cc9a3ff98bd4caaa72a2fe1ddbe956ae52b367cae762` |
| R21n | [v3 events/config_08.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_08.json) | `cd05523` | 159,518 | `cc1fcc57cae325ba127c3de73af9ee1c9ec032b295fd48c512f8ab79aca161c2` |
| R21o | [v3 events/config_09.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_09.json) | `cd05523` | 159,522 | `862166a77b19b0307b2af2af5b8f92efd3048c9f21c39abf5ad5404addbedac8` |
| R21p | [v3 events/config_10.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_10.json) | `cd05523` | 161,227 | `a0b7b970b72260159ad7f8b2b99cdec7f626e379c019b31d23b74c704d0f5ac5` |
| R21q | [v3 events/config_11.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_11.json) | `cd05523` | 161,345 | `cc73d208093e9ca220d74dc4a687f7b15ce1528d141223070fa98789f0e00b15` |
| R21r | [v3 events/config_12.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_12.json) | `cd05523` | 626,822 | `076d1a107a05d9c86a638b2e96add0730d10f41a9d8fcd7e92644aba49f63f5b` |
| R21s | [v3 events/config_13.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_13.json) | `cd05523` | 612,022 | `5edf36582b2a3d7ed1a6124abae31ef99bac6acf039a7ca76ba1a2124176d648` |
| R21t | [v3 events/config_14.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/events/config_14.json) | `cd05523` | 36,036 | `2d8d10c8a01bdc6bc854175e3fad6b3001b92b7bc0852184fa9559e5943a6974` |

指示書の資料A〜Dのうち、URLで直接指定された「S4固定結果ディレクトリ」（`https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4`）はディレクトリで、上の個別ファイルと §12.3 のchunk照合で確認した。

### 12.2 外部一次資料

| ID | 文献 | 確認した版 | URL |
|---|---|---|---|
| [L1] | D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms* | arXiv:2603.13495v1（2026-03-13） | https://arxiv.org/abs/2603.13495v1 ／ https://arxiv.org/html/2603.13495v1 |
| [L2] | B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024) | arXiv:2402.15550v2（2024-10-31） | https://arxiv.org/abs/2402.15550v2 ／ https://arxiv.org/html/2402.15550v2 |
| [L3] | J. Peetz, S. E. Smart, P. Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, npj Quantum Inf. (DOI 10.1038/s41534-025-01168-w) | arXiv:2407.21095v2（2026-07-06） | https://arxiv.org/abs/2407.21095v2 ／ https://arxiv.org/html/2407.21095v2 |
| [L4] | K. Wan, M. Berta, E. T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, PRL 129, 030503 (2022) | arXiv:2110.12071v2（2022-07-13） | https://arxiv.org/abs/2110.12071v2 ／ https://arxiv.org/pdf/2110.12071v2 |
| [L5] | P. Zeng, J. Sun, L. Jiang, Q. Zhao, *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, PRX Quantum 6, 010359 (2025) | arXiv:2212.04566v2（2025-03-27） | https://arxiv.org/abs/2212.04566v2 ／ https://arxiv.org/html/2212.04566v2 |
| [L6] | B. Koczor, J. Morton, S. Benjamin, *Probabilistic Interpolation of Quantum Rotation Angles*, PRL 132, 130602 (2024) | arXiv:2305.19881v2（2023-11-14） | https://arxiv.org/abs/2305.19881v2 |
| [L7] | C. Kiumi, B. Koczor, *TE-PAI: Exact Time Evolution by Sampling Random Circuits*, Quantum Sci. Technol. 10, 045071 (2025) | arXiv:2410.16850v3（2025-09-24） | https://arxiv.org/abs/2410.16850v3 |
| [L8] | E. Granet, H. Dreyer, *Continuous Hamiltonian dynamics on digital quantum computers without discretization error*, npj Quantum Inf. 10, 82 (2024) | arXiv:2308.03694v2（2023-11-17） | https://arxiv.org/abs/2308.03694v2 |
| [L9] | J. Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026) | arXiv:2503.05647v2（2026-07-10） | https://arxiv.org/abs/2503.05647v2 |

いずれも §1.3 に記した箇所だけを読んだ。全文の精読、PDF図表からの数値の読取り、版間の差分比較はしていない。

### 12.3 取得・照合ログ

| 手順 | 結果 |
|---|---|
| `git ls-remote`（公開repository） | 成功。main HEAD `0babed0` |
| GitHub REST API（未認証） | rate limitで応答なし。git経路で代替し、影響なし |
| `git clone --filter=blob:none` と8固定commitの存在確認 | すべて存在 |
| fe0f3f0→53c4831 の差分 | 2 files changed, 991 insertions(+)。変更path：`artifacts/track_b_parent_allocation_review_packet/2026-10-15/s4/law_comparison_index.csv`、`docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md` |
| resource_map.json の7 chunkをmanifest順に連結 | 各chunkのbytes／SHA256がmanifestと一致。連結結果 3,228,746 bytes、SHA256 `8281cf66c1ce7b5abf1a46ba6d4c4725c994eabf3dadcc143a9877387cc117c5`。manifest値・git全体取得との一致：一致 |
| inherited_rows.json と v3 resource_map rows の照合 | 50 row。順序を含めオブジェクト完全一致：一致 |
| arXiv API（export.arxiv.org）による版・公開日の取得 | [L1]–[L9] すべて取得 |
| arXiv HTML（[L1]–[L3]、[L5]–[L8]）とPDF（[L4]）の取得・該当箇所の抽出 | すべて取得。抽出箇所は §1.3 |
| 最初のセッション | ツール呼出しの上限でファイル作成前に停止し、暫定要点をチャットに記載（ファイル未作成と明記）。本ファイルで置換 |

本書の作成以外に、repositoryへの書込み、sourceの修正、authorizationの発行、markerの操作、pushは行っていない。新しい科学計算・合成・guard・N予測・sampling・worker実行も行っていない。

---

## 13. 改訂履歴・手続的結論

### 13.1 本完成版の編集範囲と責任

- **原本：** 2026-10-15付のClaude独立レビュー `レビュー.md`（元ファイルは別に保持）。
- **修正根拠：** GPTによる `レビュー修正点.md`（F1–F10）。
- **完成版編集：** GPT-6による校訂・批判的な数理／科学的解釈の整合確認。Claudeが本改訂へ再回答した、修正を承認した、または原本を自主改稿したことを意味しない。
- **原データ保全：** S4の50 control＋訂正20 law、旧proxy20 lawの除外、正常適用外5 slot、報告中の費用表・固定条件・GitHub path／bytes／SHA256、Claudeが取得・読解した一次資料の記録は、個別に誤りが示されない限り原稿の値を継承した。
- **再監査の限界：** 今回GPTはGitHubの主要固定資料の取得可能性を確認したが、Claude原稿にある全行のraw再集計・全source・全一次文献本文を一括再監査したとは主張しない。したがって§1.4の派生float集計はClaude原稿から継承した**探索的結果**であり、S4の認証されたledger fieldとは区別する。

### 13.2 主要な訂正と理由

| 修正ID | 対象 | 完成版での取扱い |
|---|---|---|
| F1 | $h\approx0.63$ | 固定二lawの交点は全70 law最良範囲の上限のみ。下側包絡線は未取得（§5.8） |
| F2 | 多step | normalization／二次momentの下界と最適化後総T費用の倍率を分離（§4.5） |
| F3 | zero-T極限 | $G_T(0)$は一般に正の有限極限へ近づき、$N$と$2Nh$が発散し得る（§6.5） |
| F4 | 終了・新規性 | 全連続配分familyのno-goを否定し、今回の価格・入力・有限候補に限る研究上の優先順位推奨へ限定（§2・§7・§9） |
| F5 | native価格 | 「noise」ではなく固定合成条件下の決定的な角度依存価格変動（§5.3・§6.7） |
| F6 | 算術の権限 | Claudeによる新しい保存値の派生集計を明示。将来のC2–C4・登録値分析は別途ユーザー承認（§1.4・§10） |
| F7 | 閾値 | $r=4$、$h=50$ T、2倍、0.5%などは未採択の設計候補（§9・§10） |
| F8 | 引用 | §12のR9・R21の重複を一意な枝番に修正（§12） |
| F9 | Claude初回チャット出力 | 当初のファイル限定契約に反した履歴をそのまま開示（§1.2） |
| F10 | 強凸性 | $f''>0$による厳密凸性と有限単体上の一様正下界を区別（§4.3） |

### 13.3 本完成版から直ちに実施してよいこと／いけないこと

本書は**研究判断のための完成した科学レビュー記録**であり、**計算実行のauthorizationではない**。未採択のD＋E方針を選ぶか、条件付きでAを続けるか、別テーマへ移すかの判断に使用できる。

ただし、S4の原source／結果／authorization／消費済みmarker／STOPを変更せず、別途承認のない登録入力の数値再集計、NC exact化、$h$包絡線、r>1評価、追加native合成・guard、分子／DF計算等を開始しない。**mandatory STOP、one-shot、retry=0を維持する。**

**完成版としての結論：** Parent allocationの有限条件付きT利益は保存し、同一fixtureでの追加探索を当面中断することを第一候補として推奨する。しかし、それは全ての内部配分や多step構成の不可能性を証明するものではなく、実際のPR/RPE taskにおける費用構造と未取得条件次第で、優先順位の再検討余地を残す。
