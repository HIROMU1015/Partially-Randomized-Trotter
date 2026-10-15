# Parent allocation：限定的成果と現在の数値探索の区切り

2026-10-15 JST。**D＋Eをユーザーが正式採択したことを記録する新しい研究記録**である。DはParent allocationの限定的成果を保持すること、Eは現在の数値探索をいったん区切り、次の改善方向を検討する資料へ研究実務を移すことを意味する。[今回の指示原本](inputs/track_b_de_direction_preparation_user_instruction_20261015.txt)をbyte保存した。新しい候補の採択、実装、科学計算は認可されていない。**mandatory STOP**。

完成版はClaude独立レビューを基にGPT-6が改訂した記録で、Claude本人による改訂承認やCodexによる派生分析の再認証ではない。完成版レビューではD＋Eは承認前の推奨だった。今回の明示採択を、その原本を書き換えずに追加記録する。過去の「GPT判断待ち」「条件付き第一候補」は当時の状態であり、原結果の科学classificationを変更する理由にはしない。[完成版レビュー][PA-REVIEW]／[当初v3レビュー][PA-V3-REVIEW]。

## 1. 何を改善しようとしたか

Return aggregationでは、同じ有限Taylor演算子を異なる回転辞書・sampling lawで生成できる。Parent allocationは、Euler pairingに与えるeven parentの係数をchildごとに配分し、正規化の増加を許して回転を安くする余地を調べた。中心比較はordinaryとclosed P5、参照はpartial-return、closed P3+tail、matched CTS。既知のcost-aware ISにも各方式で同じ選択機会を与えた。[科学設計と訂正の経緯][PA-V3-REVIEW]／[最終比較証拠][PA-PACKET]。

ここでのP3+tailも**同じfinite P5 target**を実現する。Taylor次数を3へ落として精度を緩めた対照ではない。closed P5とgeneral local fullは同じ理想return ensembleを異なるnormalizerアクセス・有限bit生成経路で扱う。[G〜D研究ノート][G-D-NOTE]。

## 2. 数学的に保持する成果と、その既知性

### 平均演算子・位相の保存

既存監査の記号で、parent/childrenの寄与を

\[
\zeta\bigl(aI-i\sigma\sum_i b_iQ_i\bigr)Q(u)
\]

とする。\(Q_i^2=I\)、\(b_i\ge0\)、\(z_i\ge0\)、\(\sum_i z_i=a\) の下で

\[
c_i=\sqrt{z_i^2+b_i^2},\quad \phi_i=\operatorname{atan2}(b_i,z_i),\qquad
\sum_i c_i\zeta e^{-i\sigma\phi_iQ_i}Q(u)
=\zeta\bigl(aI-i\sigma\sum_i b_iQ_i\bigr)Q(u).
\]

これは**理想平均の恒等式**であり、有限bit係数・合成誤差が自動的にゼロになるという主張ではない。\(z_i=0\) のpure端点は \(-i\sigma Q_i\) で、controlled実装ではその位相と積順序を保持する。任意のword全体をinvolutionと仮定していない。[既知法との対応監査][PA-ART]。

\(\sum_jb_j>0\) のcommon配分 \(z_i=a b_i/\sum_jb_j\) は、このfamilyの係数L1 \(\sum_i\sqrt{z_i^2+b_i^2}\) を最小化する。child別配分は角度・係数・辞書価格を変える自由度であり、L1改善自由度ではない。この平均保存は既知Euler pairingと線形分割の帰結、L1最小性は三角不等式の帰結である。[PA-ART][PA-ART]。

### 訂正KKTが最適化したもの

採択された固定価格proxyは

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2},\qquad
z_i=\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}}.
\]

S4はこの式へ訂正し、探索区間・外向き数値区間・有理化・端点処理を接続した。固定旧価格はcommon側の完全eventのT単価 \([136,138,140]\)。新角度の価格を見る前に候補を固定しており、native価格取得後の候補選び直しはしていない。128回の二分、256bit根区間、share64有理化を用いた候補で、exact simplexと距離certificateを保持する。**連続native費用の最適解を得たわけではない**。[S4結果と修正根拠][PA-RESULT]。

旧S3は同じraw \(C_i\) を入力しながら \(\sqrt{C_i^2-\lambda^2}\) を用い、別目的 \(\sum_i C_i\sqrt{z_i^2+b_i^2}\) の候補を生成していた。旧20 lawはそのproxyの実行記録としてv3に保持し、訂正結果には含めない。halfwayも訂正KKTに依存するため4構成・20 lawを補完した。これは同一KKTのseed違い再試行ではない。[PA-RESULT][PA-RESULT]／[訂正前後のidentity索引][PA-PACKET]。

cost-aware ISの既知のleading proxyは \((\sum c_e^2/q_e)(\sum q_eC_e)\)。全 \(C_e>0\) の固定辞書なら \(q_e\propto c_e/\sqrt{C_e}\) がその最小化を与える。KKTはこの既知原理の外側に置いた固定価格proxyである。range、bias、ceil付き十分N、\(C=0\)、実際に角度で変わる合成価格まで含めた総費用最適性は従わない。[PA-ART][PA-ART]／[D系列の先行研究対応][D-ART]。

SPSの一般的な線形分解最適化、SCU/CTS、Euler pairingとの共通部分は既知として保持する。operator first momentとchannel/processの違いは比較時にembedding・controlled位相・情報取得条件へ戻す。数学的に埋め込めることだけで、全生成手順・取得費用の完全同値や独立新規性のいずれも認定しない。[PA-ART][PA-ART]。

## 3. S4の正式な証拠集合と検証範囲

| 項目 | 固定条件・証拠 |
|---|---|
| 正式比較集合 | 旧v3から不変の50 control law＋S4訂正20 law＝70 law、15構成 |
| 対象外の履歴 | 旧raw-C proxy20 lawはv3に保持。正常適用外のanalytic IS 5 slotは失敗・negativeではない |
| 入力 | 既知development 3-system-qubit synthetic、\(p=(1/5,3/10,1/2)\)、\(\tau=x=5/7,r=1\)、finite P5、初期状態000、\(\sigma=+1\) |
| native構造 | \(Q_0=Z_0,V_1=R_{XX01}(\pi/4),V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4),Q_i=V_i^\dagger Z_iV_i\)。controlled位相を保持 |
| 分子条件 | 分子・geometry・basis・DF rank・split \(L_D\) は非該当。独立held-out分子結果ではない |
| 精度・confidence | primitive誤差\(10^{-6}\)、pygridsynth 2.0.0/seed0/strict phase。axis許容誤差1/200、axis失敗確率1/40、一つのcomplex signalで失敗確率1/20以下 |
| 科学source／authorization | S4=`4ccc1832f63cd323b23fc05623f7e0b58cf34f37`、A4=`71cf1b76492ed01fc5bb8363246b5f13f7e166d7` |
| 既済の証拠 | S4：12新合成key・11 metadata再利用・24root診断・672bindings、371保存監査PASS。v3：918監査PASS。今回は再実行していない |

正本は[実行報告][PA-RESULT]、[保存監査][PA-AUDIT]、[70 law索引][PA-CSV]、[最終Evidence Packet][PA-PACKET]。同名config_04/05/10/11でも、旧v3 namespaceと新v4 namespaceを区別する。固定commit・namespace・row path・hashをidentityとする。

## 4. 保存値で確認された利益と交換条件

| 保存比較 | h=0の記録 | そのまま言えないこと |
|---|---|---|
| ordinary best common `config_00:regularized_16` 対best noncommon `config_03:regularized_1/16` | noncommonの期待総Tが約14%低い。NとCXは増える | proposalが異なるため純粋なallocation-only効果ではない。全座標優越ではない |
| P5 best common `config_06:regularized_1` 対best noncommon `config_07:analytic_IS` | noncommonの期待総Tが約19%低い。NとCXは増える | 同上。pure端点が安いproviderと固定価格・proposal集合に条件付く |
| 訂正ordinary KKT／halfway対common | 同一proposalでT改善を示していない | 連続内部配分の一般的な不可能性ではない |
| 訂正P5 KKT対common | 同一proposalのh=0で小さいT差が記録されている。halfwayはcommonより高い | proxyの連続最適性、実価格に対する最適性、他providerへの移送は未確立 |
| 全70 lawの保存選択 | h=0の有限集合最良は`config_07:analytic_IS` | 全h最良、全配分family最良、実PR最良ではない |

約14%／19%は[完成版レビュー][PA-REVIEW]の丸めた要約値で、基礎となる保存二lawとT/CX/native1Q・N・moment・bias・affine recordは[PA-PACKET][PA-PACKET] §§4–7へ追跡できる。今回は比率を新規計算していない。同一proposal比較とbest-to-best比較を混同しない。同じproposal名は同じ生成規則の比較であり、辞書・価格が変われば実確率qまで同一ではない。

新しい角度の価格取得と旧cache検証は科学実行時に記録されている。合成角度の変化とproposal変更の両方が費用に関係するため、「normalizationが増えても常に安い」または「既知ISだけで全差を説明済み」という一般化はしない。係数質量・second moment・range・native/coefficient biasは別の会計量である。十分Nは保証規則からの予算で、実量子shot数や必要最小shot数ではない。[PA-PACKET][PA-PACKET]。

### 準備・読出し費用

S4の保存モデルは、資源座標jごとに \(G_j(h_j)=2N E[C_j]+2Nh_j\)。T、CX、native1Qは別座標で、足し合わせない。上表のT利益は \(h_T=0\) の切片。保存best-common対best-noncommonのT交点はordinary約27.313、P5約11.322 T/trialである。一方、P5の別頂点lawを含めると `config_07:analytic_IS` が最良であり得る領域には約0.63という別の二law交点による上限がある。**0≤h<0.63全域で全70 lawに勝つ証明ではない**。全70 lawの下側包絡線は未取得で、今回は計算していない。[PA-PACKET][PA-PACKET]／[完成版の訂正箇所§5.8][PA-REVIEW]。

online classical costは`NOT_MEASURED`。setupの保存時間、12新keyの取得費用、実量子試行時のonline生成費用、状態準備は別項目であり、未測定部分を0にはしない。

### 多stepへの懸念と限界

完成版レビュー§4.5の紙上議論は、**独立step・weight積・同じ頂点を反復**する条件に限る。\(x=\tau/r\) とすると頂点の係数質量は \(1+x(1-p_j)+O(x^2)\)、commonは \(1+O(x^2)\)。対応する質量積の二乗の極限 \(e^{2\tau(1-p_j)}\) と1は、二次momentの**下界**の比較である。最適IS後の総T費用、実際のcommon moment、非IID生成、準備費用、回路相殺を含む倍率ではない。S4はr=1なので多step実証もしていない。[PA-REVIEW][PA-REVIEW]。

## 5. 確認範囲を超えないための区分

| 区分 | 保持する内容 |
|---|---|
| 数学・実装上の成果 | 任意の適格配分の理想mean保存、commonのfamily内L1最小性、端点と位相を含む実装、訂正proxy候補の有理化 |
| 固定条件の数値成果 | 上記70 lawの条件付きT利益・N/CXとの交換、同一proposal／best-to-bestの区別 |
| 既知原理 | Euler＋線形分割、固定辞書のcost-aware IS、標準KKT。SPS/SCU/CTSとの近接性 |
| レビューの追加的・探索的内容 | 完成版レビューにあるfloat64のNC診断、価格統計等は同レビューに帰属するpost-hoc分析。認証済みresource mapへ昇格させず、今回exact再集計しない |
| 未確立 | 独立新規性、連続familyのnative最適性、他provider・分子DF・PR/RPE/QPE全体の資源優位、whole-circuit/compiler最適化、online費用 |

技術的停止のprefixと有効な性能比較を混同しない。v1/v2の保存・process不具合、旧KKT目的不一致、S4の正式な訂正比較は異なる記録であり、過去のmarker・分類・STOPを残す。

## 6. D＋Eとして何を区切り、何を残すか

**保持するもの**は数理構成、mean/phase検証、70 lawの有限条件付き資源地図、価格・proposal・h・多stepに関する境界である。**現在区切るもの**は同じdevelopment入力で候補、precision、proposal、rを増やし続ける数値探索である。これは今回ユーザーが採択した研究判断であり、Codexが新たに一般的な性能negativeを認定したものではない。

現在の固定proxyの内部候補は、実際のnative価格・finite confidence・準備費用を一括最適化していない。頂点＋ISのT差だけで独立した方法貢献や実用的総費用改善を確定できず、同じfixtureの候補追加だけでは未解決事項を閉じない、という直前レビューの限定を引き継ぐ。[PA-REVIEW][PA-REVIEW]。

将来、例えば既知IS/SPSとの差を持つ具体的な生成機構、非列挙での取得保証、実際のPR taskでのnative/preparation条件、量子費用を含む新たな成立条件が得られた場合は再検討材料になり得る。これは再開承認や新しい実験採択ではない。再検討時は最初に本書、[70 law正本][PA-PACKET]、[先行研究対応][PA-ART]、[完成版レビュー][PA-REVIEW]を読む。

次の準備資料は[資源構造調査](track_b_resource_structure_and_bottlenecks_20261015.md)と[未採択の方向群Decision Packet](track_b_next_algorithm_research_decision_packet_20261015.md)。重要な科学レビューはユーザーの明示的開始承認後、科学実行は別途明示承認後。**今回は資料整理のみで終了する。**

## 固定資料の参照先

参照IDのcommit・path・bytes・SHA256は[取得索引](../../../artifacts/track_b_de_direction_preparation/2026-10-15/v1/evidence_index.json)に記録した。リンク先はbranch先端ではなく固定commitである。

[D-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[G-D-NOTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-14_track_b_return_aggregation_G_D_research_note.md
[PA-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
[PA-AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json
[PA-CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/artifacts/track_b_parent_allocation_review_packet/2026-10-15/s4/law_comparison_index.csv
[PA-PACKET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[PA-RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md
[PA-REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[PA-V3-REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/research/track_b_parent_allocation_v3_scientific_review_20261015.md
