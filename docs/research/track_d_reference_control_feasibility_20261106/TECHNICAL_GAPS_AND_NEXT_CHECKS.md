# Track D：技術的Gapと必要最小限の次確認

2026-11-06 JST。調査結果を研究採択へ変換しない。下記の後続作業は提案であり、自動実行しない。今回の許可は6文書の検査・公開照合まで。既存STOPは全て保持する。

## 1. D-1〜D-4の位置

| 区分 | 今回資料から言えること | 言えないこと |
|---|---|---|
| D-1：追加削減の機構 | 真空保持DF pathなら、symmetric baselineでも中央RTEに残るCRZ/CRZZ・CZとancilla接続を通常演算へ交換する具体的対象がある | compiled正味利益、全RPE優位、独立新規性、科学的GO |
| D-2：既存法との重複 | Gaussian制御除去・basis共有・D halfの対称制御による任意角負担削減は既存。ここだけをReference固有利益とは呼べない | symmetric方式とReferenceの全費用が常に同等、または全D pathの残余CX/depth利益がないという一般結論 |
| D-3：reference条件 | 一般X/Y Pauli factorは真空を保存しない場合があり、普遍的な真空方式は成立しない。custom basis/mixed pathのreference資格は未確認 | 別referenceや特殊full wordにも不可能、全DF registryが適格という断定 |
| D-4：資料不足 | 実SE型PR回路、phase補正の実装、同条件full-wrapper compile、routing/合成精度の比較が不足 | 不足を失敗・価格0・優位へ読み替えること |

中心問いへの到達点は **「条件付きの残余機構を特定したが、正味full costはD-4」**。D-1〜D-4は対象範囲ごとの整理であり、研究方向の採択・棄却ではない。

## 2. 科学計算なしで解消できるGap

| Gap | 参照・再利用可能なもの | 最小限の静的確認／担当 |
|---|---|---|
| 対象pathのreference資格 | `df_rte_tail.py`、physical Gaussian producer、typed `BasisProvider`、保存requestの公開metadata | Codex：対象SOURCEとencoding、各complete factorのvacuum-ray条件を一件の候補に限定して対応付ける。Hamiltonian/stateを作らない |
| 既知phaseの取りこぼし | `RTEEvent.phase`、`unsigned_rotation_angle`、product signs、constant/extracted identity metadata、actual adjoint | Codex：公開metadataからscalarがどこに会計されるかを追跡する。新しい分子phaseの数値評価はしない |
| baselineの合成可能範囲 | legacy `boundary_optimized`、native `symmetric_directional`、理論A/Bの閉境界 | Codex：同一SOURCEでemit済みの最適化と、別構成の許容変形を分ける。費用を再計算しない |
| 過去H4 SOURCE binding | H4 v5 manifest closure、native87の固定bytes、元inventory | Codex：必要な関数だけのbyte/役割を追加照合可能。全依存の再監査や過去run再認定を必須にしない |
| 接続接口 | native/legacy wrapper、mixed actual adjoint、phase保持compile、serializer | Codex：必要最小の接口差・register順を記述する。新builderは実装しない |
| 強対照との重複 | SE §II/III、Simon–Love §III、PR App.A.4/E | Codex：同じpathを保つ変形とrandom lawを変える変形を引き続き区別可能 |

上記の主要条件は本6文書で既に整理した。全repositoryの不存在証明や全面的な文献再調査は残タスクにしていない。公開された特定pathに候補を絞る場合の追加静的bindingが、数値なしで詰められる範囲である。

## 3. 新実装・回路科学実行を伴う最小検証案

ユーザー／GPTが継続対象を選び、**別の実装・実行許可を与える場合のみ**、次の順に範囲を絞る。現段階で新しい実行契約・grant・承認証明の仕組みは作らない。

1. **Codex：小さい人工例でgadget correctness。** 既存DF event/actual-adjoint/wrapperを再利用し、SE原著Eq.2の2層CSWAPと既知phase補正を接続する。正負Z/ZZ、identity、order0/高次product、異basis、非可換D/random、複数step、両axisを代表する最小例だけを使う。ordered pathのactual operator、χω、条件付きX/Y確率を直接controlと照合。一般Pauliとcustom非粒子数保存basisを負対照にする。これは新builder/buildと新operator評価を含むため今回は実施しない。
2. **Codex：一つの同条件full-wrapper native価格比較。** correctnessが成立した同じexplicit pathについて、既存diagonal-only＋可能な共有/fusion、native symmetric-directional、Referenceの各armを比較する。同じprecision・compiler・topology、state準備除外。CSWAP2層、追加register、補正P、cut損失を含むRZ/CX/depthとqubitを別に記録する。Tを判断する場合だけ合成profileを同じ精度へ固定する。新compileは独立した科学実行許可が必要。
3. **ユーザー／GPT：上記を受けて次の対象と必要な研究判断を決定。** 局所価格に意味が残る場合だけ、指定lawの期待価格・全round精度/成功率への接続の要否を決める。Codexは勝手に全grid、sampling、保存再会計、長RPEへ進まない。

同じoperator・lawを保持する案なので、最初からHamiltonian/state/referenceの分子生成、E₀/gap情報、全roundの新しいshot最適化、希少eventの大規模samplingは不要。人工correctnessが技術PASSでも科学的GOではない。

## 4. 許可とreviewの区別

| 作業 | 次担当 | 現在の許可 |
|---|---|---|
| 本6文書の作成・文書検査・専用branchへの公開・remote SHA/blob照合 | Codex | 今回依頼で許可済み。これを完了してSTOP |
| 追加の限定静的照合だけ | Codex | 継続依頼の対象を明示すれば実施可能。STOP後に自動継続しない |
| 新builder実装・新operator評価・build/compile・simulation | Codex | 今回範囲外。対象・条件を定めた別許可が必要 |
| 保存値の再会計・再ranking、sampling/新科学計算 | Codex | 今回範囲外。別の実行許可が必要 |
| 正式な研究方向選択、重要科学レビュー | ユーザー／GPT | 今回実施していない。重要レビューの許可は科学実行許可と独立 |

再利用資産の旧SOURCE、raw、grant、ledger、marker、archive、mandatory STOP、retry条件を改変しない。研究A/B/Cや旧P-A/C1等をTrack D名で再開しない。後続計算が許可される場合は、既存performance方針に従いdominant workloadと実資源を調べ、共有環境に影響しない範囲で高速化する。過去runの資源上限を新規作業へ一律流用しない。
