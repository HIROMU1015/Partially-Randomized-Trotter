# PR α検証：保存入力のseal（数値解析は未承認）

2026-10-18 JST。利用者の「次の作業に進んで大丈夫」を、前回提示した保存入力の結合・sealへの承認として実施した。B-1/B-2の実データ数値解析は前の添付指示により別途明示承認が必要であり、今回は実行していない。

**最新入力はrun03。B-1は14条件すべての保存入力、B-2は3,720条件中3,660条件の保存入力をsealした。B-2の残る60条件は既知の旧H6 B0I非相殺分解・会計不足として保持した。二項確率・shot・RZ費用・順位の新規再解析、sampling、signal、compileはすべて0。有効production grantは未発行でmandatory STOP。**

[前回の実装準備](pr_alpha_followup_preparation_20261018.md)と監査e62ddb59を再利用した。[最新metadata](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/)を正本とし、旧版を上書きしない。

## 固定した入力とSOURCE

SOURCE：`93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb`。branch：`track-a-alpha-followup-preparation-20261018`。保存scalarをコピーする[入力codec](../../src/trotterlib/pr_alpha_scalar_input.py)、[binder](../../scripts/bind_pr_alpha_saved_inputs.py)、[コピー検査script](../../scripts/check_pr_alpha_input_copies.py)を追加した。新しい科学関数や旧科学SOURCEの変更はない。

固定入力19件（旧計画・manifestを含む）をcommit・Git blob・bytes/SHA256へ結合した。うちprivate原本3件は、旧H6 signal、旧H6 B0I signalコピー、旧H8 signalである。既存の公開input receiptとすべて一致した。旧H8のproduction版・resume01版は固定hashに一致せず採用せず、resume02版の原本で一致を確認した。原本の書換え・値の代入は行っていない。

private scalar packと実行planはproject専用private領域に保存し、Gitには入力hash・scope・field来歴だけを保存する。private directoryは0700、各pack/planは0400。raw/corrected signal・reference値やprivate path mapping、有効grant原本・認証情報を公開しない。保存h/N一覧は元から公開されているaccounting欄の抽出で、新規shot計算ではない。

[コピー検査](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/copy_verification.json)は、B-1の14行とB-2の完全3,660行、計3,674レコードを原本フィールドへ照合した。JSONの正準表現の等値でfloatの再解釈や辞書の別値への置換を拒否する。原本19件のhash、保存baseline全欄、raw/corrected/reference、allowance、log_B/margin、bias recipeの素材、paired CSVの行番号とseed/event/compiler、PF単価と説明metadataを確認した。数式を実データで評価した検査ではない。

|入力欄|来歴・表現|
|---|---|
|z_approx / raw / reference|保存signalの対応軸を直接コピー。決定論的B=1のみ、rawはcorrectedの恒等コピーと明記|
|B_corr|保存log_Bからexpを評価するrecipe。準備時はexpを評価しない|
|B_budget|保存log_B＋倍率×saved log_B_marginのexp recipe。B_corrと別の役割を固定|
|通常bias|保存total_signedの軸別絶対値recipe|
|非相殺bias|H6は保存signed_partsの絶対値和recipe、H8は旧axis_envelopes CSVの保存magnitudeをコピー|
|u / h / N / α|uとmarginは保存signal、h/N/αと状態・証拠分類は固定baseline全欄からコピー|
|PR費用|確認union8/union6の元CSVから同じreplica/seed/event/compilerの2軸RZ整数をコピー。n32の平均・SEは計算しない|
|PF費用|各条件の保存実測/model/mixed単価と来歴をコピー。新model評価なし|

実行時のrecipe評価はgrant確認とone-shot marker消費の後で行う。B-2はlog-domain行で不要なexpを評価しない。B-1は保存h/Nを使い、shotを再計算しない。B-2の基準α=.05での再現照合も、実データについては今回まだ実施していない。

## B-1：承認可能になった範囲

[入力binding](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B1_input_binding_UNAPPROVED.json)、[保存h/Nの28軸一覧](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B1_saved_shot_scope.csv)。前回の代表14条件・通常bias・allowance倍率1・総α=.05を維持した。候補を再選択していない。14/14条件に保存raw/corrected/referenceとbaselineが揃った。

承認後の上限は28二項軸評価、CDF/SF各tail合計最大56 call、logCDF/logSF最大56 call。入力recipe decode最大14条件、normalization decode最大14条件。新しいshot・費用再会計・Monte Carlo・sampling・compileは0。数値境界、underflow、α超過は既存runnerの判定・STOPに従う。

B-1の入力不足は解消したが、確率条件の数値検証は未実施。bias/u/normalizationの証明書不足は入力sealだけでは解消しない。

## B-2：承認案と欠測の影響

[入力binding](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B2_input_binding_UNAPPROVED.json)。総α gridは**.01/.05/.10の未承認案**、2軸均等配分、倍率1、通常/経験的非相殺を分離する。H6は決定論136＋PR確認8、H8は決定論184＋PR確認6。εはH6=.05/.01/.005/.001/.0005、H8はさらに.000625を含む。前回の固定集合を維持した。

入力3,720条件のうち3,660条件が完全、60条件は旧H6 B0Iの12候補×5精度の非相殺分解・baseline不足である。欠測を除外してfull-domain winnerを決めず、MISSINGとして3α分の180出力行を残す。このためH6非相殺の全候補最小値・PR/PF順位はUNRESOLVEDとなる設計である。通常biasやH8へ欠測を波及させない。

承認後の出力上限は11,160会計行。現入力で凍結shot関数を呼べるのは最大10,980回（3,660×3α、うち基準αは3,660回）、残り180行は欠測placeholder。費用評価はeligibleな行に限り、保存確認n32のpaired G_i平均・SEと保存PF単価を用いる。欠測単価を0にしない。最良候補・順位・削減率の新計算は今回0。

## 依存・検査・保全

各stageは実際に使う5つのSOURCE fileだけをhash照合する。B-1専用wrapperの変更でB-2を停止せず、その逆も同様。Python executableと使用math/fractions/statisticsを固定し、SciPy/NumPyはB-1だけに結合した。B-2はstdlib会計でありSciPy変更を停止理由にしない。共有core/codec/runtimeなど実際の共通依存が不正な場合は影響するstageを停止する。無関係なHEAD/status/worktreeや一時的なCPU競合をgateにしない。

[人工62件](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/synthetic_tests.json)がPASS（新規21＋関連回帰41）。[新規テスト](../../tests/test_pr_alpha_scalar_input.py)と[関連回帰テスト](../../tests/test_pr_alpha_followup.py)を使用した。前のA監査11件/16件は再実行していない。[CLI8検査](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/CLI_checks.json)はsealed planの表示、無効grantによる実入力実行拒否、V2人工入力だけのparent/child一回実行・native exit0・再実行拒否を確認した。主agent自身の検査であり、独立reviewではない。

metadata準備中に、H6のn8/n32平均の併記、H8のbias別説明metadata差、旧H8の文字列factor、確認表への決定論対照の再掲を検出した。[履歴](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/metadata_preflight_events.json)を残し、数値の再評価やsource値の書換えで解決していない。

最初のsealは文字列factorのjoin漏れを含む履歴として保持した。run02は入力が揃った版、run03はさらにstage別依存を限定した最新版。旧pack・plan・receiptを上書きしない。[保全監査](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/preservation_audit.json)に、旧版入力hashと旧科学結果・前回準備成果物の不変を記録する。

## 次の担当と判断

- **ユーザー：**B-1の14条件・保存h/N・28軸上限・SOURCE/runtimeを確認し、B-1の一回実行を明示承認する。B-1承認をB-2へ流用しない。
- **ユーザー：**B-2のα grid案・候補/ε/bias・60欠測の扱い・11,160出力行と最大10,980会計callを確認し、必要ならB-2を別途明示承認する。
- **Codex：**承認されたstageにだけ独立grant・未使用output/markerを結合し、fresh gate後に一回実行、native終了・結果を記録してmandatory STOPする。

今回の入力sealへの承認を、実データ解析の承認へ拡張しない。現時点では有効production grantを発行せず、科学workerを起動しない。
