# C4 finite-RTE proposal：GPT判断用引継ぎ

2026-10-14 JST。Codexの限定技術監査資料。GPT科学レビューは開始していない。C4の正式採択・終了・論文化価値の判断ではない。

## 1. 技術監査の答え

現C4の最適則・whole-path費用・平均保存・backward DPには、独立した算法上の差を確認できなかった。**REDUCES_TO_KNOWN**。一般DF/native IRを多項式有限状態へ圧縮する構成、安いrange/tail保証、D1の実際の総費用利益は **UNDETERMINED**。限定classでの既知sampling手順は **CONDITIONAL**、実平均を保つweightの意味論とD1 source上の接続点は **SUPPORTED**。

Phase Aの停止条件に従い、Phase Bの新constructor具体化・production実装・科学benchmarkへ進んでいない。単なる「指数列挙なし」も新規性の十分条件ではない。係数law/価格が安く、価格比が制御できるclassには、標準rejection/zero-fillが強い既知対照になる。

[技術監査全文](qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md)、[状態map](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/method_status.json)、[一次文献registry](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/literature_registry.json)、[固定参照identity](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/reference_identity.json)、[scope](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/scope.json)。

## 2. 判断材料の更新点

- [Resource-optimal 2026](https://arxiv.org/html/2603.13495v1)の本文・定理を取得し、一般費用、合成操作、bias、error detection、signed PECを確認した。C4の確率則への新規性claimは支持されない。一般有限bitサンプラーの効率保証とは区別する。
- 2023 cost-aware qDriftは時間も変える別channelであり、確率則の違いだけでC4を独立法と扱えない。PRHSはcorrelated signed列とresource-optimal IS接続を既に述べる。PR/CTSの非列挙/層samplingも公平な対照に含める。
- 標準rejectionの受理lawと、global normalizer不要のzero-fill weightを係数未代入で照合した。理想moment×量子費用productへ到達し得るが、range・古典費用・finite bits・D1十分shotsの最適性は示さない。
- 原C4のDPは状態・遷移・係数因子化が与えられれば既知weight pushingへ還元する。元O(JrS²)の記述はevent alphabet/bit長/初期化/Jを不足している。
- D1の実native passはinverse stack。個別pathの線形処理と、全prefixを多項式有限状態へmergeできることは別。任意inverse grammarには指数個の区別が必要なcaseがある。一般DFの不能性や固定P5の不能性と解釈しない。
- 全path proposalでcorrelationを入れるとD1のstep moment/range/acceptanceのべき乗会計は再利用できない。同じfinite mean、new whole-trial M2/W/acceptance/native費用のcertificateが必要。

## 3. GPTに残す科学的問い

1. 現C4を「既知ISの適用・実装課題」として扱うか、独立算法への投資を見送るか。技術還元と研究の正式終了は別である。
2. 限定native grammarの圧縮やcheap range/tail certificateに、既知のrejection/weighted-automata法を超える具体的な課題があるか。未知であることだけを継続理由にしない。
3. 同じ費用情報・有限状態・合法な最適化を許したPR/CTS/D1の強い対照にも利益を認めたとき、C4固有の改善機構を定義できるか。
4. classical setup/draw/probability/weight、rare tails、controlled回路、state preparation、測定を含めても、元量子化学ground-energy taskへ意味のある利益を接続できるか。
5. C1の既存否定的診断とC4の技術還元を踏まえて、C1を終了・再設計するか、別方向を選ぶか。今回Codexはこの判断を代行していない。

## 4. C1との切り分けと不足情報

C1 baseは5ff55e749ceb33e54da0b3c70ac6ac08599ebb82。C1結果・source・認可・STOPは保全し、集計・数学探索・再実行をしていない。C4の一般sampling法の成立は、C1の構成やselector問題を解決する証拠ではない。

判断を変え得る追加情報は、実際の回路grammarと価格アクセス、polynomial state/J/bit complexityの証明、安いM2/W/tail保証、同情報強対照との差、ground-energy全費用への接続である。それらの取得を今回の後続作業として無断に開始しない。新しい分子・固有値・PF・compile・sampling benchmark・保存事後集計を必要とするなら、目的と最小scopeを別に固定して明示実行承認を受ける必要がある。

## 5. 次の担当と承認境界

今回の技術監査は完了。次はユーザーによる別開始承認を受けた **GPTの重要科学判断**。同じC4の候補探索を反復しない。実行認可・C4の正式採択は未発行。新しい数値結果はなく、詳細なGPT科学レビュー記録も作成していない。

Branch qc-c4-finite-rte-technical-audit-20261014、基準C1 5ff55e74/C4 fc063696/D1 2dcd5673。公開payload/SHA・主要資料のGitHub取得確認は[公開receipt](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/publication/receipt.json)。
旧source/結果/認可/STOP/global FAIL、N1 STOP・N2 HOLD・N3 CLOSED・THRIFT現template終了を維持する。独立branchへcommit・pushし、remote byte確認後 **Mandatory STOP**。

## GitHub公開確認と最終STOP

[再取得receipt](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/publication/receipt.json)：監査資料payload `60da5738d48e0faaedd3de125b0350301935dc3a` の21変更blobs、361保護files、C4/D1固定参照10filesをGitHubからbyte/SHA照合した。独立cacheにlocal alternatesはなく、公開は指定repositoryの独立branchのみ。receiptは後続metadataとして保存し、最終remote取得で全リンクを照合する。新しい数値・数学プログラム・science source実行やGPTレビューは追加せずMandatory STOP。
