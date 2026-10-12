# N3：一次文献と数理・taskの対応 v1

2026-10-12 JST。Codexによる資料収集・式対応。新規性・論文化・baseline採択の科学レビューではない。
完了済みレビュー第10.2節の資料を補充し、数学的同値、情報・保証の違い、未取得を区別する。
先行研究の不存在を確認したとは主張しない。以下は論文の長文引用ではなく、今回確認した箇所の短い対応記録である。

## 確認した一次資料

| ID・固定版／URL | 全文の確認箇所・取得状況 | 数式・入力情報・保証・費用の対応 |
|---|---|---|
| L1 Liang et al., arXiv:2104.00257v1 | [HTML](https://arxiv.org/html/2104.00257v1)第1節式1.1、2節Theorem2.1 | orthonormal X上のtrace最小化は最小固有値和。N3のσνの原理は既知。射影するHamiltonianや外部sector下界を選ぶ方法を、この式だけから同定しない |
| L2 Vu et al., NeurIPS2013 | [公式PDF](https://papers.nips.cc/paper_files/paper/2013/file/81e5f81db77c596492e6f1a5a792ed53-Paper.pdf)第2節式2、4節Lemma4.1、参考文献22–24 | Fantopeの制約・rank-ν射影の凸包・最大固有値和。−Kへの適用はN3の最小和と同値。ADMMはsparse-PCA目的と統計仮定を伴い、N3誤差保証や取得費用へ直結しない |
| L3 Dusson–Sigal–Stamm, arXiv:2105.02058v1 | [HTML](https://arxiv.org/html/2105.02058v1)仮定A–C、Theorem1.1/1.2、Lemma2.1 | Feshbach–Schurの等スペクトル関係・孤立固有空間/gapを使う摂動誤差。N3もSchurを使うがPは外部空sector、利用下界はc。既知Schur原理を独立寄与としない。gap情報取得を共通U/cだけの対照へ無償移植しない |
| L4 Chamaki et al., SQuISH, arXiv:2211.16522v1 | [全文PDF](https://arxiv.org/pdf/2211.16522v1)IIのRDM式5–6、III.A手順1–6／式7–8、III.C/D式10–11、AppendixB p12を取得。p3を画像確認 | 近似wavefunction/RDMでHamiltonianの項を順位付けし反復追加。N3の一mode空射影とはtaskが違う。実用停止は隣接反復のenergy差、主要benchmarkでは正確energyを参照したとの説明がAppendixBにある。この停止規則をN3の係数からの有理射影誤差certificateと同じ保証に数えない |
| L5 Greiner–Gauss–Eriksen, arXiv:2406.11343v1 | [HTML](https://arxiv.org/html/2406.11343v1)2.1–2.3節・式3–5、3節 | active/reference spaceとMBE increments、経験的lognormal/回帰・importance samplingからscreening誤差を推定。取得済みCASCI等のincrement情報、sampling費用、推定保証をN3のexact PSDと区別。実分子coverageは今回N3へ移さない |
| L6 Overton–Womersley, SIAM J.MAA13(1992),41–45, DOI10.1137/0613006 | [出版社](https://epubs.siam.org/doi/10.1137/0613006)書誌・abstract、L2の参考文献24で対応。全文は未取得 | 対称行列の最大固有値和・凸包の数学原理。原典の全文未照合をL2全文で隠さない。N3固有の保証生成算法が本文にないと断定しない |
| L7 Coleman, RMP35(1963),668–686, DOI10.1103/RevModPhys.35.668 | [出版社書誌](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.35.668)を確認。公式全文APIは401で未取得 | 1-RDM/N-representabilityの歴史的原典候補。原典の定理番号・normalizationを未照合のまま採用しない。現在のFantope同値性は取得済みL2と数学証明を根拠にする |

L4は前回abstract中心だったが、今回は公式PDFの11,591,687 bytes、SHA256 `51c7e47a32255f9b1c251719d8ac243e375efc69a2ae1ee7c0e661210ed232ca` を取得。
[取得receipt](../../artifacts/n3_theory_task_integration_v1/2026-10-12/literature_download_receipt.json)。全文と抽出textは一時資料として/tmpに保持し、著作権のある全文をrepositoryへ再配布しない。
L4の初期web PDF/HTML取得失敗と、その後の公式PDF取得成功を区別する。L7の401は全文未取得であり、論文自体の不在ではない。

## 数学的対応の技術整理

OCCのσν(K)は、1-RDMの0≼γ≼I,trγ=ν上の線形最小化と同じ最適値を持つ。
この局所primitiveに相関状態の1-RDMを取得する必要はなく、係数Kのスペクトルを使える。2-RDMの完全N-representability問題を解いたという意味ではない。
旧族のνλmin(K)はγ≽0,trγ=νへ緩和した最適値であり、占有上限の欠落を示す。
既知のFantope原理を使う一般SDPもこのprimitiveを表現できる。Z/τのdual表示を「SDPにできること」の新規性としない。

N3固有として判断が残るのは、係数trial U・外部下界c・一mode構造・有限bit検証・縮約予算の組合せにどの独立した算法／理論差があるかである。
既知原理の組合せが現在の主張を実質的に包含する可能性を、閉じたものとして扱わない。

## 科学比較の候補と未採択事項

| 対照候補 | 利用する情報／取得費用 | 現時点の扱い |
|---|---|---|
| 同じU,cのNORM＋FIVE＋全OLD | 同一係数・bound証拠、小行列PSD・scalar評価 | 既存の登録対照。S1での理想OR下端を解析的照合済み |
| Fantope/一般SDPによる同じσν | 同一K、浮動小数solver＋exact rounding/PSD検査費用 | 数学的には同じprimitive。実装対照または包含の判断材料。自動で独立新baselineを採択しない |
| density情報を残すRDM/SDP、一般Schur評価 | 同じHamiltonian係数でも追加relaxation・sector/gap情報・大きい行列処理が必要になり得る | 強対照候補。情報と費用を明記してGPTが必要性を判断。今回計算なし |
| SQuISH／MBE active-space選択 | 近似state/RDM、反復solve／increment sampling等 | taskの先行例。N3と同保証の資源対照にするには変換規則が必要。今回採択なし |

コスト・oracle・保証対象を変える比較は、既存内部旧族との分離証拠で解決されていない。
文献原典二件の全文未取得は未解決事項として残す。次のGPT判断は、取得済みの対応とこの限界を含めて行う。
「網羅的な先行性調査完了」「全既知法に優位」とは報告しない。
