# A2 SPEC-01〜03人工技術検証：判断用ハンドオフ

2026-10-18 JST。**`SPEC01_03_IMPLEMENTED_SYNTHETIC_PASS`。** [技術報告](technical_report.md)／[SOURCE/API](source_api_and_scope.md)／[テスト結果](synthetic_test_results.json)／[固定証拠](evidence_index.json)／[公開receipt](publication_receipt.json)。

## 到達点

SPEC-01〜03のSOURCEと独立checkerが利用可能になった。人工raw binary64からexact modelとcanonical identity、固定N・PF/scalar、LowerCertificate/FragmentCapCertificateのL・全f_j・Λを生成・検算できる。最終38 unit/property/negative/mutation tests、1489 subtestsは全PASS。checkerをgenerator関数なしで動かせること、故意のgenerator cap誤りやhash付直し後の不正scalar/capも拒否することを確認した。

independenceはbinary64復元、Hermiticity、cap加算、L/scalar、PF検査の別実装まで。schema/hash/型/Python runtimeと既知公式は共有しており、形式証明や完全独立再現とはしない。人工fixtureは手計算・stdlib bit oracle・小規模exact occupation/PSD oracleで照合したが、実入力のcorrectness・cost・認証値の良さは一切評価していない。

## 残る制限

- callable入口は人工raw envelopeのみ。実分子/npz/DFHamiltonianとのadapterはない。scope tagは自己宣言で、元データの真正性を認証しない。
- Hermiticityはexact受理のみ。実保存入力がこれを満たすか未確認で、黙って対称化してはいけない。
- capは保守的な係数三角上界。Lが十分よいか、Λが小さいか、bit/整数処理費が安価かは未検証。
- trusted model_id・期待N/PF順・元raw/完全inventoryは呼出側で固定する。来歴不明の欠落blockや二重加算済みscalarを自動発見できると主張しない。
- exact ground access、回路・state preparation、元分子保証、数値stepや総費用はない。
- U/W、K_SOS、Cbar2、step比較、QPE/native/cost等のSPEC-04〜08は未実装。未知値を0やPASSとして出力しない。
- 返却partial recordの自動永続化・OS強制終了からの回復runnerは未実装。将来callerで保存仕様を採択する必要がある。

今回の人工scopeを妨げる未解決の技術不具合は確認されなかった。合格範囲を超えた保証は上記のまま未検証である。

## 次に進めるか・必要な承認

**次の担当はユーザー。** SPEC-04〜07の限定SOURCE実装と人工テストを、別途依頼するか判断する。今回の合格はそれらの自動実装・実行を承認しない。

進める場合の最小作業は、既存の[設計7af83266…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_a2_certificate_connection_design_20261018/source_api_connection.md)に従い、現在のmodel/sector/partition/PF IDsへ、SPEC-04のdeterminant変分U/W、SPEC-05のK0辞書/行和、SPEC-06のgrouped構造Cbar2、SPEC-07の既存RF/EW2/strong counterとstep enclosureを接続することである。将来のCodexはこのSOURCEのfixed commit/bytesを基点に、独立checkerと人工coverageを維持する。ここではそのSOURCEを作成していない。

実入力certificateを望む場合は、対象raw/model・sector・PF順・SOURCE identity、精度/証拠水準、取得情報policy、保存/STOP/retry範囲を正式契約として固定し、さらに別途の科学実行承認を得る。人工test合格を根拠に保存DFへ「試しに適用」してはいけない。A2採択、研究GO、新しいGPT重要レビュー、native/QPE/state計算も別の範囲である。

## 状態・保全・公開

正式状態の正本は[decision67576526…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)。A1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行、Exact Encoding現構成REDUCES_TO_KNOWN_METHOD、その他STOP/CLOSED/PAUSED/HOLD、①②③は変更しない。

基点7af83266…、branch `qc-a2-spec01-03-synthetic-20261018`、作業領域 `/tmp/prt-qc-a2-spec01-03-synthetic-20261018`。ユーザー指示は[byte保存原本](inputs/user_instruction_a2_spec01_03_synthetic_20261018.txt)。固定SOURCE/test/資料・証拠へのリンクはevidence indexからたどれる。SOURCEのpackage namespace・新test・今回の専用報告以外を変更せず、一般索引や研究状態も更新しない。

本文commitをpushしてSOURCE/test/資料・固定参照の公開bytes/SHA256を照合後、receiptを子commitへ保存する。最終remote HEADと全公開ファイルを再照合してチャットで報告する。receiptに自己hashを埋め込まない。親worktreeのHEAD・local branches・registry・index/exclude・tracked bytes/原本、独立領域の全base blobを開始snapshotと照合する。

**実分子科学計算、実入力certificate生成、保存科学値再解析、SPEC-04〜08、科学実行契約・authorization/grant/marker、新科学レビュー、A2採択は各0件。** 今回の人工generator/checker/test実行は明示承認内の技術検査。新しい科学実行へ移行せず、**mandatory STOP**。
