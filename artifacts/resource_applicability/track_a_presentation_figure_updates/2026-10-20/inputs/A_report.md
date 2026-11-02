# PR/PF絶対RZコスト比較：v2・5精度（2026-10-19）

利用者の「プロトコルは最新v2、対象はいったん5精度、プロットできるものだけ」という指示に基づき、保存済みの比較を表示する。新しい科学計算・shot会計・候補選択・順位計算は行わない。

基準はv2 snapshot `c0808b254aa02a94c7482f5c6e353b54532a54f8` の[プロトコル](track_a_pr_resource_evaluation_protocol_v2_DRAFT.md)。元文書のDRAFT状態は保持し、今回の作図指示をプロトコル全体の正式採択や新しい科学stageの承認へ読み替えない。

[成果物一覧](../../artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/README.md)に5枚のPNGとSVG/PDF、データ表、スライド用captionを保存した。

## 図とデータの範囲

横軸は水素原子数、縦軸は解析shot込みの保存総RZコスト `G_RZ`（対数）。PRは青丸・実線、deterministic PFは橙四角・破線。PRのerror barは同じcohortの**保存paired SEの±1 SE**であり、新しいCIではない。直接実測のPFを塗りつぶし、検証済みモデル推定のPFを白抜きにした。数値ラベルは表示用に3桁へ丸め、exact tokenはCSVに保持する。

|精度ε|独立確認パネル|探索パネル|選択されたPFの単価証拠|
|---|---|---|---|
|0.05|H4/H6/H8|H2/H4/H6/H8、各n8|全表示点が直接実測|
|0.01|なし|H2/H4/H6/H8、各n8|H6は検証済みモデル、他は直接実測|
|0.005|なし|H2/H4/H6/H8、各n8|H6は検証済みモデル、他は直接実測|
|0.001|なし|H2/H4/H6/H8、各n8|H6は検証済みモデル、他は直接実測|
|0.0005|H4/H6/H8|H2/H4/H6/H8、各n8|全表示点が直接実測|

**26比較・52点・5精度**。両端2枚では独立確認と探索を左右のパネルに分け、同一ε内は共通の縦軸範囲にした。中間3枚は探索だけで、独立確認の欠測を埋めない。各εで縦軸の表示範囲は異なる。線は離散点の視認性補助で、補間値・fit・scaling則を表さない。

全点はlinear chain、隣接1 Å、STO-3G、T=.8、P=0、通常signed bias、allowance倍率1、tol-only1e-8 DF。実rankはH2/H4/H6/H8で3/10/19/28。full cos/sin Hadamard wrapper、symmetric_directional、元constant＋tail identity一回、reference不回転を既存条件として保持する。compiler設定は記録上Qiskit1.3.0、rz/sx/x/cx、opt1、seed17で、実runtime/profileの完全同一性は別問題である。

`G_RZ`は両軸のcompiled RZ×保存解析shotの合計として**既に保存された値**を引用した。実量子shotを実行した費用、energy/RPE全round、非Clifford数、FT T countではない。

## 採用した比較

### 独立確認6比較

既存[掲載用主表](../../artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/main_table.csv)の6行をexact tokenのまま転記した。PFは各Pで事前固定された対照で、新たに最良PFを選んでいない。

|系|ε=.05のn・L_D・δ|ε=.0005のn・L_D・δ|元のcohort|
|---|---|---|---|
|H4 DF10|1108 / 3 / .8|78 / 2 / .1|INDEPENDENT_H4_CONFIRMATION|
|H6 DF19|48 / 4 / .4|104 / 4 / .1|H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1|
|H8 DF28|48 / 8 / .4|48 / 7 / .05|INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48|

H4粗精度のNOT_REPRODUCEDとH6粗精度のPF有利方向のREPRODUCEDを保持する。REPRODUCEDをPR有利の同義語にしない。H8の2比較は既存Figure2B/H8_CONFIRM_1・3と同じ証拠で、追加の独立証拠ではない。

### 探索20比較

保存済み`comparisons`が指定するPRとPFを転記した。候補一覧から新しいargminや旧cohortとのunion winnerを作っていない。

|系|採用した保存結果|選択範囲・cohort|固定snapshot|
|---|---|---|---|
|H2 DF3|A66の5精度比較|新EXPLORATION_N8、D70保存winner。FP n1を混ぜない|829d697ae7905b3ea90a61da9d2d4da15d8b757a|
|H4 DF10|A50の5精度比較|全5表示winnerは新EXPLORATION_N8。保存FP n32とのpoolingなし、D190保存winner|85cb5b4d1808434165c421115ed09dab3b63dd00|
|H6 DF19|新A16の通常signed・倍率1|新EXPLORATION_N8とstrengthened_S2_S4の保存比較|64c85f7a833b5950ef2ccae623708b73081edcd6|
|H8 DF28|新A16の通常signed・倍率1|新EXPLORATION_N8とstrengthened184の保存比較|64c85f7a833b5950ef2ccae623708b73081edcd6|

H6/H8は選択されたPF単価が直接実測の点でも、登録対照集合の他候補にmodelが残る。全格子直接実測最良や大域最適性を主張しない。比較集合の大きさ、候補域、選択効果は系ごとに異なる。

H2はv2に記載されたA完了の探索結果としてだけ表示し、未完了Pの点を作らない。既存の論文主表・Methods・CEMへH2を追加していない。

## 除外と欠測

- H2独立P：v2固定時点で未実行・未seal。S_STATSの提案nを完了標本として扱わない。
- H4/H6/H8の中間3精度の独立P：今回の主表に該当比較なし。探索値を独立確認へ昇格させない。
- H4 legacy DF12：新tol-only DF10と別入力。今回の5枚から全て除外する。
- 旧PR n32、確認union n32、H4 FP n32、H2 FP n1：採用した新A・独立Pと混合せず、追加の系列にもしていない。
- 古い/弱いPF集合：H6 old32・H8 old32へ戻さず、保存済み強化集合との比較を使う。
- 非相殺bias、allowance倍率10、主5精度以外、m5_best等の追加公式：今回の共通表示条件の対象外。未解決・欠測を0で補完しない。

機械可読の[採否表](../../artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/data/selection_decisions.csv)と[比較表](../../artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/data/comparisons.csv)に対応を保存した。

## 注意点と来歴

全生成規則の一致は未認定で、state solverや正規化・残差gateに差がある。分子サイズだけの因果効果や厳密scalingを主張しない。有限標本・探索選択効果・未観測rare費用は残り、paired SEは全supportや母平均順位を保証しない。

H8のPR完了は公開結果/報告のscopeで保持するが、全実seed除外・grant消費・runtime/provider/input bytes・全worker exit/reap・最終byte照合を公開詳細receiptから再検証したものではない。原profile bytesと旧compiler出力の完全同一性も未確認。元の`REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS`を維持する。

`data/input_sources.json`は15個の実読取Git objectのcommit/path/SHA256を記録する。`data/source_records.json`は採用した元recordを保存し、`data/plot_points.csv`は52描画座標をexact tokenに結び付ける。分数からfloat・表示文字列への変換、軸範囲、注記位置、error barの描画だけを行い、mean/SE/shot/総費用/比/CI/順位は新規に計算していない。

描画は科学moduleをimportしない[専用script](../../scripts/plot_pr_pf_absolute_cost_v2.py)。Matplotlibは今回のworktree内の`.plot-venv`、font cacheは`.plot-cache`だけを使用する。共有package・OS設定・他jobに変更なし。既存SOURCE・結果・ledger・marker・STOP・旧Figure2は変更しない。

保存値照合・採用行のscope/件数・5精度の全52点・出力SHA・画像形式・34個のlocal linkの構造検査を実施し、PNG全5枚を目視確認した。これはCodex自身の技術確認で、別担当の独立科学レビューではない。再現コマンドと環境は成果物README・requirementsに記録した。

終了条件は成果物保存・local commit、公開pushなし。プロトコル正式採択、最終evidence freeze、詳細科学レビュー、新しい科学stageへ進まない。

次の作業なし。Codexはlocal commit後 **mandatory STOP**。
