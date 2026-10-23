# 保存値と来歴の読取規約

このファイルは既存規約の所在説明であり、数式・数値条件の変更や新しい科学会計ではない。
原規範は[共通プロトコルv1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/docs/research/track_a_pr_resource_evaluation_protocol_v1.md)、
主な列定義は[既存paper manifest説明](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/artifacts/resource_applicability/track_a_paper_preparation/2026-10-19/paper_data_manifest.md)を参照する。
各時点のv2草案や採択・追記案は元のstatusのままとし、この集約で採択しない。

|項目|保存規約と注意|
|---|---|
|taskと精度|固定時間のcosine/sine signal精度。化学的energy精度や全RPE round精度へ換算しない|
|bias|主系列の通常signed biasと、非相殺などの別scopeを区別する。厳密bias certificateと経験値を区別する|
|allowance|原SOURCEの経験的numerical allowance、倍率、log margin、headroomを保持。新しい厳密上界へ読み替えない|
|normalization|stepのb、累積B、raw/corrected信号と対応SOURCEを区別。欠測から逆算しない|
|shot|SOURCEで保存された解析的shot数と整数化。量子measurement shotを実samplingした結果とは扱わない|
|費用|両軸について G = N_cos C_cos + N_sin C_sin。Cは軸別のfull Hadamard wrapper compiled RZ単価。状態準備Pを含むかは原scopeに従う|
|PR標本|同一trajectoryのcosine/sine pairingを保つ有限標本平均。保存SEは原cohortのpaired費用のMCSEで、母平均優位や選択後coverageの保証ではない|
|PF単価|直接compile、検証済みmodel、条件付き保存価格を原区分のまま残す。すべて直接実測とは呼ばない|
|単位|原数値と単位を維持。compiled RZ×解析shotをFT T count・Toffoli数・物理運用費へ換算しない|
|統計配分|主系列のalpha_total=.05/alpha_axis=.025は原会計の設定。費用標本SE、標本数設計alpha_C、QPE/RPE統計誤差、決定論PF係数Cと混同しない|
|cohort|探索n8、旧fixed anchor/union n32、独立P、outer32 pilot/探索を混ぜない。別資料にある同じH8観測のaliasを独立証拠として加算しない|
|nullと欠測|null、不適格、未保存、未登録、未検証、対象外、FAIL、STOPを区別。欠測を0やFalseへ変えない|
|join key|系・geometry・basis・入力生成版・Hamiltonian/state/snapshot identity・sector・DF政策/実rank/順序・PF公式・L_D/q/r/Kと配分・T/delta・bias/allowance・shot会計・control/native/compiler/runtime・cohort・PF対照集合・結果commitを使う。候補IDやrank数だけで結合しない|
|SOURCE|科学SOURCE、取得SOURCE、会計SOURCE、資料公開commit、実行時HEADを区別。hashの一致だけから元private入力の内容や実行hostを推定しない|

本bundleの`evidence_index.csv`は既存ファイル全体のSHA256とGit blob IDを示す。
値を転記した原科学表のrow/field対応は、固定参照先の既存paper manifestを使う。
今回、新しい比・差・平均・SE・適格性・winner・shot・費用の算出はしていない。
原文に未公開とある資料でも、その後の固定commitで取得可能になっている場合があるため、記述時点と今回の取得状態を分けて読む。

取得検証段階：

- `RAW_BYTES_FETCHED_AND_MATCHED`：今回GitHubからrawを取得し、ローカル固定bytesと一致。
- `VERIFIED_COMMIT_TREE_LOCAL_OBJECT`：GitHubのcommit/tree一致を確認し、その固定treeのローカルblobをhash計算。個別rawの今回取得は未実施。
- `ANCESTOR_SOURCE_OF_VERIFIED_SNAPSHOT`：原SOURCE公開commitが確認済み公開snapshotの祖先であり、原manifestのSHA256とローカルbytesが一致。科学再実行は未実施。

本作業は作者本人によるmetadataと保全の技術確認である。独立科学レビュー、外部再現、最終evidence freezeではない。
