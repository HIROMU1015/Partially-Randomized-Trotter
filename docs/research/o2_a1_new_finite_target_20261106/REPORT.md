# A1新finite-target契約設計：報告

2026-11-06 JST、SERVER1。専用branch `codex/o2-a1-new-finite-target-contract-20261106`。
**契約草案を作成した。新targetは未採択、科学実行は0件。** 旧opt2来歴調査を再実行せず、固定[6381f81の結論][OLD]を引き継いだ。

## 成果物

- [A1_NEW_FINITE_TARGET.md](A1_NEW_FINITE_TARGET.md)：保存DF・state・preparation・実Q・selected full複素信号の一貫した定義。
- [A1_NEW_SCIENCE_EXECUTION_CONTRACT_DRAFT.md](A1_NEW_SCIENCE_EXECUTION_CONTRACT_DRAFT.md)：公平な4方式比較、finite-confidence、費用、後日の科学callとSTOP。
- [SOURCE_CANDIDATES.json](SOURCE_CANDIDATES.json)：再利用元の固定commit・SOURCE bytesの一覧。将来runnerの実行SOURCEは未固定。
- [CONTRACT_STATUS.json](CONTRACT_STATUS.json)：設計、未取得数値、判断事項、科学call=0、STOPの機械可読記録。

## 定義したことと残ること

| 区分 | 今回の到達点 |
|---|---|
| A：静的に定義した事項 | 保存DF H4/1.00 Å/STO-3G/legacy DF12/4電子number sector/8qubit/L_D3/m0/q1/δ.02/r1/K2/threshold0の候補。SOURCEの順序・phase・actual Q・w=1、同じselected full signalを比較する規約 |
| B：科学取得が必要 | 新partition/preparation/basis/catalog/λ/B/CDF/Q/support/ν、state/engine誤差、価格、bit、variance/N/費用。今回未取得 |
| C：ユーザー判断 | 新targetの正式採択、ε/α/η/train/pilot/cap/allowance、条件付き取得法、主RZ、実現費用か期待費用保証か、N1/N2/N3の許可範囲 |

一次targetを**actual-Qでのselected logical full interrogationのattenuated複素mean**とした。algebraic reference・保存DFの真時間発展・元Hへのenergy保証は別接続とする。
この選択は新しい契約案の中心的な判断事項で、旧草案のalgebraic finite-targetを黙って継承したものではない。全方式が同じ一次targetを推定するため、初期mean比較にPF/RTE物理biasや半径下界の認定を必須にしない。state/engine/centering誤差は引き続き会計する。

旧opt2と完全同一という主張はしない。旧partition/preparation/tail hashを新runの受理条件にせず、新run内の保存identityを正本とする。
旧hashの再現計算、旧guardの削除、threshold/基底丸めの変更はしていない。旧価格・bias・半径・N=510を新targetへ無条件移植しない。
引き継ぐのは保存DF snapshot、既存algorithm/interface、phase/readout/価格profile候補と調査結果である。旧数値結果は歴史的証拠として保持する。

## 比較・confidence・価格の設計

3control、B0/B1-O/B1-OL/A1、5層と層内CV相殺を明記した。actual Qからν/層質量を一貫して取得する。
強いbaselineを不必要な古典rejectionで不利にしないようexact-grid直接条件付き取得を第一候補とし、旧rejection/attempt capを選ぶ場合の規則も保持した。取得法は採択事項であり、今回adapterを実装していない。

提案は独立OLS train→固定独立cal→fresh固定N prod。全方式に同じvariance-sensitive保証を認める。
standalone診断のε=1/4・全8比較のα=1/20、train1024、B1 pilot64/層、η1/8、prod cap8192、CV range guard、engine allowanceは具体的な**未承認候補**として記載した。新Nを算出したわけではない。
[Maurer–Pontil v1](https://arxiv.org/pdf/0907.3740v1)のTheorem10と既存草案のbounded Bernstein/Hoeffdingを適用条件付きで再利用した。新定理・重要科学レビューではない。

full価格は同じselected policy・axis・Qiskit1.3.0 opt2 seed17から取得する案。主RZ、全stageの費用と古典費を含む。初期案は実現campaign費用であり、期待費用のconfidenceには別途有効な価格range/独立取得が必要。

## 次の最小scopeと障害

最小の次回推奨は**N1のみ**：新runnerの最小実装・人工検証とSOURCE固定後、同NPZから一回のpreparation、完全payload、actual CDF/Q/support/ν/層質量、新policy basisの対応を取得してSTOPする。
state solve・sampling・full interrogation build/compile・測定・variance/N/費用はN1に含まない。ユーザーが明示許可したscope内でのみ次段階へ進む。

残る技術差分は、完全pre-threshold serializer、新identity用薄いrunner、exact conditional取得、confidence/有限配分、state/measurement engineと同じpath/価格のbinding。既存A1/DF/RTE/RPEを再利用し、coreや旧runnerは変更していない。
N2/N3へ進む前には、state payload・sector/layoutとground精度、engine error allowance、未取得価格を閉じる必要がある。現時点でmeasurement-readyや期待総費用の優位を認定しない。
source guard/入力/新数値を勝手に埋めた「実行可能」とは呼ばないが、後日対象scopeを許可すれば実装・固定・取得へ進めるcall単位を具体化した。

## 保全・公開・STOP

[保全記録](PRESERVATION.json)に元mainのHEAD/statusと既存2,136ファイル、旧A1 evidenceの不変を記録する。
今回は静的SOURCE/既存資料の閲覧・数式設計・文書/JSONの構造確認のみ。NPZ load、準備、CDF/moment/normalizer、数値再解析、RNG、solver、回路build/compile、測定、variance/N/費用は0件。新しい人工科学テストも不要なため実行していない。
専用branchへcommit/公開し、3文書の本文・blobとtreeを再取得照合する。元main・他worktree・共有環境、旧SOURCE/raw/result/grant/ledger/marker/STOP、C1凍結・S2/R1判定を維持する。
**mandatory STOP。契約草案の公開を、新targetの正式採択・科学計算の実行許可と扱わない。** 承認証明書・新grant発行等を求めない。

[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6/docs/research/o2_a1_opt2_provenance_20261106/REPORT.md
