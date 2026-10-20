# Publication scope

今回の利用者指示は、固定済み保存結果からA〜Dを別版として作成し、リポジトリ保存・専用commit・公開承認済み範囲のpushまで進めるもの。対象は新しい6図、表示データ、出典写し、caption、検証・STOP、新規描画scriptのみ。元の科学結果、証拠台帳、主表、Figure 2、grant、ledger、marker、STOPを変更しない。

入力の公開状態は2026-10-20に`git ls-remote origin`で確認した。次の公開branch先端は指定commitと一致した。

|固定commit|公開branch|
|---|---|
|`c5ad79457aea87cc36b38022a2821ebb42eb276d`|`track-a-pr-pf-absolute-cost-v2-20261019`|
|`f2d1a86c431f835db700f348ce618d8604e7ee23`|`codex/h2-pr-resource-preparation-20261019`|
|`8fbe7d77556c2353be837dbd9ed0fddc02ee81c4`|`codex/h2-results-integration-20261019`|
|`3983fe4f1ede003953735e757d2c098c12b4644f`|`codex/h6-noncancellation-stage-b-rerun-20261020`|
|`295b93d50662a0f8af3ec469df43099a3f04af60`|`codex/h4-pr-cost-variability-20261019`|

既存主表commit `1923fce8f7461513d1bc47b5061e87cd70917e87`のCSVは公開済み統合資料のH4/H6/H8行と全フィールド一致する。入力写しは指定公開結果・CSV・集計・文書・rendererのみで、未公開raw、grant、実行ledger、私的receiptを追加しない。

Bの保存文書中の「その段階で新規公開未承認・pushなし」は当時の履歴としてそのまま写した。現在の公開branchに同じ固定commitが存在することを別途確認しており、過去の状態記述を書き換えない。今回のpush対象に元結果の再commit・再pushは含めない。

新規branchは`track-a-presentation-figure-updates-20261020`、親は公開済みA v2 `c5ad794...`。他の入力commitをmergeせず、新規の表示成果物だけを追加する。既存のmain/master・公開図・結果を更新しない。詳細科学レビュー・最終evidence freeze・主表8条件への正式変更・追加科学計算は承認対象外であり実施していない。
