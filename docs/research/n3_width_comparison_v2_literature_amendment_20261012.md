# N3幅比較 v2：検証付きRDM・weighted SOSの文献対応追補

2026-10-12レビュー系列。Codexによる一次資料・式対応であり、新規性採択のレビューではない。旧[v1対応表](n3_literature_correspondence_v1_20261012.md)を保持する。

| 一次資料・確認範囲 | 今回の対応と限界 |
|---|---|
| Chaykin et al., JCTC16(2020),7342–7356、DOI10.1021/acs.jctc.0c00497。[著者所属機関](https://tore.tuhh.de/entities/publication/fce317e4-612d-4332-bfff-02471b71c072)の書誌・abstract | N-representabilityを使うSDP lowerでも浮動小数誤差により不正なlowerが生じ得ること、VSDPで検証することを確認。原典全文／全定理／性能実装は未取得。N3の有理checker一般を新規性としない。今回のSOS residual修復をVSDPそのものと呼ばない |
| Rubin–Low–DePrince, arXiv:2602.05069v1。[取得HTML](https://arxiv.org/html/2602.05069v1)第II.1式3–5、III.2式47–50、第IV節 | weighted SOS、number等式に掛けるpolynomial、v2RDM dualとの対応を確認。今回の粒子/正孔/D/Q/G Gramとreal quadratic number multiplierは明示した固定level。論文の全spin-adapted階層・分子数値・量子資源改善を実装／再現したとはしない |
| CVXPY/SCS公式資料：[solver設定](https://www.cvxpy.org/tutorial/solvers/index.html)、[CVXPY1.6.5](https://pypi.org/project/cvxpy/1.6.5/)、[SCS3.2.8](https://pypi.org/project/scs/3.2.8/) | 新専用venvで固定版を導入。solver statusやobjectiveをcertificateにせず、元係数でPSD/scalar/residualを検証。数値solverの提案能力と有理検証を分ける |

科学レビューで採択した方向は、同等/強い既知保証との取得費用の比較である。
Fantope同値表示、粒子数付きSOS、厳密lowerの検証という一般概念を「初めて」と呼ばない。
OCCのSOSへの包含は、R=Kκ−τI+Z、hole Gram=Z、particle Gram=R、pair Gram対角κJ、τ(N−νI)の式から独立に実装照合した。
同じ数学的族を解くSDPの有限実装が未判定でも、OCCが理想SDPより強いとは表示しない。
低次数SOSが理想的にOCC証拠を含むことと、固定solver・bit・repair budgetでその証拠を取得できることは別である。
原典Overton–Womersley/Coleman全文未取得は維持。文献の網羅的先行性や論文主要claimは次の新科学結果後の判断事項。
