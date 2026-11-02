# SOS-PF K0.5：限定技術調査の公開資料

K0.5調査：2026-10-15着手、2026-10-16完了（JST）。公開作業：2026-10-16。
次のClaude科学レビューが既存成果と根拠を直接取得できるようにする資料公開である。
今回の公開作業では科学的内容の再検討、新しい文献調査・科学計算を行っていない。

## 資料索引

1. [K0.5追加技術調査報告](technical_report.md)
2. [追加文献対応表](literature_correspondence.md)
3. [GPT向け判断資料](gpt_decision_handoff.md)
4. [公開用provenance・原本のbyte数とSHA-256](provenance.json)

必須3資料はローカル原本とbyte単位で同一である。READMEと公開用metadataだけを公開用途に整備した。
原本中の「ローカルのみ」「commit・pushなし」「公開未認可」等は、K0.5限定調査の実施時点の記録である。
今回の別途明示承認は既存成果の公開・独立照合だけを許可する。調査時点の記述を遡って書き換えていない。

## K0の固定参照

Repository：`HIROMU1015/Partially-Randomized-Trotter`。
本公開branch：`qc-sos-pf-k05-publication-20261016`。
基点は既存K0固定commit `58d8672702284d73321cc67a26c4640adaddde5f`。
K0資料を再作成せず、このcommitから継承する。

- [K0資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/README.md)
- [K0技術監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)
- [K0文献台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/literature_registry.json)
- [K0 method status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/method_status.json)
- [K0 GPT引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/gpt_handoff.md)
- [未認可K1計画草案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/k1_design_draft.md)

## 保持する状態と取得限界

K0の正式判定は **`UNDETERMINED_DATA_REQUIRED`** のまま。K0.5調査・今回の公開とも新規科学計算は0件。
K1は未認可・未実施、SOS-PF候補Aの研究正式採択なし。既存のmethod status・研究状態・認可・STOPを変更していない。

K0.5実施時点でClaude原本は未取得である。ユーザーの提供履歴や別資料の引用を原本取得の証拠に置き換えない。
ACM出版完全本文・PDFは403で未取得、arXiv v3との数学的・アルゴリズム的差分は未確認という記録を保持する。
文献原文の再取得や科学的判断の更新は今回行っていない。

公開ファイルは上記3資料・README・provenanceの計5件。
実機固有のcacheパス・workspace保全baseline・ローカル保全receipt・論文cache本体は公開しない。
原文との差と取得記録の公開範囲はprovenanceに記す。

**Mandatory STOPを維持する。公開と独立照合の完了後、Claude科学レビュー、GPT独立レビュー、
研究の継続・終了判断、K1実行へ自動的に進まない。それぞれ別途承認が必要である。**
