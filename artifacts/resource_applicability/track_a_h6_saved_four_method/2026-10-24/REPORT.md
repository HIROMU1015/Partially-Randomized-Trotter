# H6 Full S2・Full Yoshida4・PR S2・PR Yoshida4：保存済み結果の回収

4方式すべての総費用Gが固定済み資料に存在したため、ユーザーの「計算済みなら飛ばしてよい」という指示に従い、再計算を省いた。保存値をH2/H4の完了出力と並べ、出典・条件・欠測を整理した。**今回の実H6科学call、追加会計、順位計算、sampling、compile、simulation、新規grantはすべて0。** 新しい4方式同時順位だけは任意の別承認単位として準備した。旧strict COMPAREはBLOCKED、mandatory STOP。

## 対象と保存値

linear H6、隣接1 Å、STO-3G、tol-only DF1e-8、保存観測rank19、system12 qubits・ancilla wire12。T=.8、ε_sig=.0005、α_axis=.025、P=0、USUAL_SIGNED、symmetric-directional。単価Cは状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ、Gは保存analytic shot会計による総RZ。探索cohortを独立確認cohortへ格上げせず、H2/H4/H6間でも標本を混合しない。

| 方式 | 固定候補 | prefix | q | 保存trajectory数 |
|---|---|---:|---:|---:|
| Full S2 | H6_FULL_19_8_2nd | 19 | 8 | 1 |
| Full Yoshida4 | H6_FULL_19_4_4th | 19 | 4 | 1 |
| PR S2 | M5_MIN_a08e8bfb6642eb93 | 5 | 8 | 33 |
| PR Yoshida4 | M5_MIN_37ff3428bb58f5cd | 5 | 4 | 33 |

PRは既存prefix5/K4/ABS_TIME_PER_OUTER32。q8/q4は過去の保存family選択を引き継いだもので、今回新しく最適化・再選択したものではない。q2/Yoshida4、m5_best、B0/B0IをFull19や指定Yoshida4の代用品にしない。

以下は保存fieldの転記。積・和・比率・SE・順位の新規計算をしていない。

| 方式 | C（cosine=sine） | N_cosine | N_sine | 保存G_total | 保存paired G SE |
|---|---:|---:|---:|---:|---:|
| Full S2 | 311350 | 123734418 | 195006858 | 99240096282600 | — |
| Full Yoshida4 | 469386 | 73377606 | 75521478 | 69891145442424 | — |
| PR S2 | 184315.9696969697 | 124124833 | 195407304 | 58894875680499.97 | 483152643434.11163 |
| PR Yoshida4 | 186918.57575757575 | 86409465 | 88925894 | 32773435584223.242 | 171357449504.90616 |

4/4は元資料のELIGIBLE判定。Fullの選択endpointにN_total・軸別Gが保存されていないため、その表示fieldはnull/空欄を維持した。総Gの欠測ではなく、新たな加算・乗算も必要ない。FullのSEも未定義を0にしない。PRの単価SE・軸別G・N_totalは保存済み値を[H6 CSV](four_method_saved_values.csv)へ転記した。

[H2/H4/H6の12行CSV](H2_H4_H6_saved_output_table.csv)はH2/H4の既存8行をfield単位で保存し、H6の4行を追記した。H4はDF10、H2はDF3であり、異なるallowance入力や候補cohortを単一標本として再解析していない。新しいH6の方向逆転判定・4点順位の出力はまだない。

## 出典・SOURCE・identityの範囲

- 共通Full採択データ：`f75a7c25d584b69d90de454731a7f343a421e059` の `track_a_h246_pf_common_adopted/2026-10-22/dataset.json.gz`。H6候補pointerは `/candidates/381` と `/candidates/450`、ε=.0005の保存endpoint。
- Full native監査：`90f1cd55a0d963953696bd0a27d9d041ff939bb7` の `track_a_pf_common_H6_results/2026-10-22/`。catalogのSHA、native inventory、input identity、end audit、STOPに結合した。
- PR費用：`8613b3ecda2030a677b1690fe7aa3da23dec4873` の `track_a_h246_saved_cost_results/2026-10-21/evaluation.json`。解析SOURCEは `9a77d2b0a0b87e246e85f085b37f5923c18e27bb`、元PR実行SOURCEは `5a448d94ea389ada6e2b2755de6a215ed2034679`。
- 回収adapter SOURCE：`9a22f0604281f6c602e819418c936d9844830262`、基準commit：`868ee7ced6818b2531a3aa9c12498f02bce55dc3`。H4/H2の既存seal・会計・rankコードを保持し、新しい数値公式を実装していない。[SOURCE一覧](SOURCE.json)。

[保存metadata照合](source_binding_audit.json)ではHamiltonian・状態・snapshot SHA、rank/policy、geometry、basis、T、compiler、controlがFull/PRの元資料間で一致した。[回収projection](saved_results_projection.json)とcatalogに9固定資料のSHA・byte数を保存し、Fullのraw結果SHAをnative inventoryと照合した。

Full S2 raw SHAは `00fcab25cd4d9b2cb5330ea1d91fc3cc118ee88e45159643c68eb488620366d7`、Yoshida4は `feb81bf60ff4bf3e215341deb383d022f8b8ec08655da7c34700e38516c35f7e`。両方とも旧SOURCE `b7662487185086aeaa1b298f82089e194b7270db` の **AUDITED_COMPLETE_PAIR_FROM_INTERRUPTED_STAGE** である。後続aggregate SOURCE `bdf6772aade7063c65d69f24dfdc5e303a2819a2` はこのcomplete pairを採用している。旧parentのexit/reapはnullのまま、recovery workersのexit0/reaped=trueとは区別する。今回の回収によって元stage全体の正常終了や完全な過去dynamic runtime identityを認定していない。

保存referenceの末尾差、empirical u=1e-9、reference絶対誤差bound=null、有限探索平均と母平均保証の違いを保持する。単一状態での条件付き探索結果であり、作用素保証・厳密certificate・母平均優位・全候補最適性・次数変更の因果説明を主張しない。

## 任意の順位出力だけを行う未承認契約

**今回の成果物を読むために実行する必要はない。** 保存済みGを対象にH4と同じ機械可読の4方式順位が必要な場合だけ、次の別契約を使用する。

| 項目 | 固定値 |
|---|---|
| series | H6_FOUR_METHOD_SAVED_RESULTS_RANK_ONLY_V1 |
| SOURCE | 9a22f0604281f6c602e819418c936d9844830262 |
| canonical contract SHA256 | cab91e6d525743675ba5d5c1e5b743ff96727a6f88da33a625b4e5306004e5c0 |
| input manifest SHA256 | 1a651da436911ab400cd81feceaeb0de8a4cfbb696cd98df53c9fd78308a473f |
| 対象 | 上記H6固定4候補の保存G |
| exact ledger | bootstrap1 / bundle_load1 / ranking1（計3 event） |
| 新しい科学operation | ranking1のみ |
| その他の科学call | allowance/N/G/統計/reference/prepare/signal/sampling/build/compile/simulation等すべて0 |
| runtime | 封印済みisolated Python3.11、971 file seal |
| authority | 未承認、新規grant発行0、専用root未作成 |
| STOP | 一回のみ、retry=false、終了後mandatory STOP |

[契約案](execution_contract_DRAFT.json)、[無効authority template](authority_templates.json)、[metadata-only preflight](metadata_preflight.json)。旧保存順位はsubset PF基準とm5/PRの順位であり、現在のFull19/PR5の4候補同時順位として転用しない。既存H2/H4のgrant・完了rootは再利用しない。

`scripts/h6_saved_four_method.py` の入力検査・承認・ledgerは既存実装を流用した。任意の `rank_rows` はH4 `Backend.rank` のコード本体を変更せずH6候補namespaceに結合し、さらに元H2のrankを呼ぶ。新しい科学会計を含まない。共通運用指示 `research_project_instructions_unified_2026-10-23.md` §8.1の「対象・条件・SOURCE/input・必要な科学call範囲・STOP条件を固定した上でユーザーの別途明示承認」により、この新順位は承認前には実行しない。既存値の引用は同節が認める技術作業として完了した。

## 技術検証・保全

人工10テストmethodがPASS（複数の取り違え拒否をparameterize）。H2/H4混入、DF/geometry/prefix/q/公式/cohort、単位・shot整数・欠測・SE、旧grant、承認前の実入力read拒否、再会計予算0、コード本体流用を確認した。[test_results.json](test_results.json)、[ログ](technical_tests.txt)。封印isolated runtimeの人工smokeもPASS、人工3 event・実H6call0、実run root生成なし、loaded closure PASS。[人工runtime記録](isolated_artificial_smoke.json)。作者自身の技術検査であり、独立した科学レビューではない。

SOURCE/input/runtimeの971 fileと、完了H2/H4の既知SOURCE・入力・結果・grant/ledger/marker/STOP・固定artifactを合わせた1,305 fileを終了時にSHA照合し、変更0を確認した。[保全監査](preservation_audit.json)。既存manifest entry・index本文も保持。既知の対象だけを監査し、他worktreeの通常更新や全体HEAD凍結を要求しない。SHA処理は利用可能CPU affinityに合わせて並列化し、共有環境や他processを変更していない。[資源観測](resource_observation.json)。private原本・内部絶対path・seed・有効grantを公開packetへ含めない。[file inventory](file_inventory.json)。

## 次の担当と終了状態

- ユーザー：保存済み4方式とH2/H4/H6表を確認する。新しい機械可読のH6同時順位が必要な場合だけ、上記固定契約のranking1を明示承認する。
- Codex：承認があればfresh metadata gate後に専用の未使用grant/rootで順位だけを一回実行し、終了監査する。現在は実行しない。既存C/N/G/SEの再計算は不要。

詳細GPT/Claude科学レビュー、最終evidence freeze、旧strict COMPARE解除を開始していない。[STOP.json](STOP.json)を保存してmandatory STOP。
