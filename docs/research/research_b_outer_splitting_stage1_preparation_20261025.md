# 研究B：Outer splittingの実装準備・人工検証・限定科学契約案

2026-10-25 JST。**技術準備・人工検証を完了。科学実行契約は未採択・未承認で、科学処理は未実行。**

## 1. 起点・範囲

起点は `3f5d60122f158b3320fadab86498963950f00734`、独立worktreeのbranchは `codex/research-b-outer-splitting-preparation-20261025`。本書を含む公開commitは実装候補SOURCEであり、科学実行SOURCEの正式固定ではない。
[Stage 1報告][RESULT]と[静的費用・outer調査][SURVEY]を確認した。完走科学SOURCE `758a05ed6aac58a5945af5a5330c792a842b174d`、保存後CLI修正 `5bef9c9cbcd7cfd595ec82874531ad3c7982d014`、結果公開 `51b1387996b1b53feb550efa7cb98aadcf5f9189` を区別する。新SOURCEで科学再実行したとは扱わない。CSV／rawを再集計していない。

既存結果の範囲はsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0、L_D=2 weight-ranked、δ=±0.09、q=2,3、n_D=1,2、r_RTE=2／K=2、complex128、36行。分子geometry／AO basis／分子DF rank探索は非該当。高次D近似のfull operator誤差が約5.1–7.6×10⁻⁷に残り、A2との差は小さく、A3対A4は丸め誤差水準でINCONCLUSIVEだったという既存報告の記載を引き継ぐ。Outer律速は仮説で、一般の支配誤差と認定しない。

共通運用 `research_project_instructions_unified_2026-10-25.md` は元workspaceの実体を全文確認した。承認はユーザーの実行許可であり、証明書・原文保存・packet／receipt／grantは追加しない。旧SOURCE・契約・結果・STOP、他研究・共有環境を保全し、C1最優先を維持する。

## 2. SOURCEと再利用

| 資産 | 実装・適合性・限界 |
|---|---|
| [`outer.py`](../../src/trottertracks/b2d_evaluator/outer.py)（新規） | `OuterCase`、`DStage`、`RTEStage`、`OuterPlan`、4方式のstage、合法融合、heterogeneous normalization、`build_outer_graph`／`evaluate_outer` |
| [`outer_backend.py`](../../src/trottertracks/b2d_evaluator/outer_backend.py)（新規・休眠） | 既存DenseBackendを継承し、stage別RTE、独立数学積gate、弱摂動診断nodeだけを追加。科学constructorは今回未起動 |
| `b2d_adapter/{adapter,_published}.py` | DContext／TailOccurrence／PhaseOperation、既存S2展開・対称列生成を再利用。Actual inverseとR障壁を維持 |
| `b2d_evaluator/{_baselines,plans,evaluation,dense_backend}.py` | A2／A3の既知列、旧半stepとの列照合、既存Node／Graph、memoization、全gate先行のevaluate、入力検査・指数cache・operator／signal処理を再利用 |
| `trotterlib/rte.py` | `make_rte_config`／`finite_rte_operator_moments` を休眠backendから接続。Sampler／Taylor演算を再実装しない |
| `df_partial_s2.py`／`df_partial_s2_repeated.py` | Preparation／phase／共通幅q反復を静的確認。不均一stageに共通N^qを流用しない |
| 既存Input package／launcher／preflight／scope | 原本は変更なし。新graphのSOURCE・scope・fresh outputへの実行bindingは条件採択後に接続する |

Published PF repositoryの固定commit `20d64c2fea1ed2d2188777df4f3d962ee94df486` の [product_formula.py][PF] をGitHubから取得し、移植済みYoshida4 generatorを確認した。既存 `_baselines.yoshida_4th_list` と `symmetric_s2_sequence` を呼び、係数を再探索・最適化しない。A3は同版のpublished Y8m10bを再利用。A4／YP8m8は新しい標準集合に含めない。

旧固定5-armには任意outer列の入口がないため、固有nodeのglueだけを独立追加した。共通DAG／数値backendの全面再実装なし。旧科学実装ファイルの変更は0件で、新規2 module、人工test、本書、関連索引だけを追加する。新launcher、認証基盤、manifest、科学artifactは作成しない。

## 3. Stage・時間・融合

`mathematical_stages` は数学的積の左から右、plan.bodyはappend順。各macroの数学列を反転してappendし、macroを繰り返す。`plan.mathematical_operations` はbodyの逆順。Dは `tau=duration/n_D` の既存A2／A3 kernelをn_D回連結する。

| 方式 | 数学列（macro幅δ） | R回数／macro |
|---|---|---:|
| Outer-S2 | D(δ/2) R(δ) D(δ/2) | 1 |
| Refined-S2 | 幅δ/fのouter-S2をf回 | f（明示必須、既定倍率なし） |
| Outer Yoshida4 | J2(aδ) J2(bδ) J2(aδ)、J2(t)=D(t/2)R(t)D(t/2) | 3 |
| SABA2 | D(c1δ) R(δ/2) D(c2δ) R(δ/2) D(c1δ) | 2 |

Yoshidaは既知 `a=1/(2−2^(1/3)), b=−2^(1/3)/(2−2^(1/3))` を再利用し、中央Rの負時間を保持する。既存floatをFractionとして保存し、和を1にする再補正はしない。そのため実stage時間和にはfloat丸め由来の微小な差があり得る。`cost_metadata.D_time/R_time` に実際の和を残す。SABA2は既知 `c2=1/√3, c1=(1−c2)/2`。c2のfloatからFractionのc1を定め、2c1+c2=1を保つ。[原論文 §4・Table I][SABA]

公称Tは全方式で同じ `T=δq`、refinedは `δ'=T/(qf)`。Scalar／抽出identityは公称Tについて一度だけ付与する。**理想Yoshida4の四次保証は近似D＋finite RTEに自動継承されない。SABA2は弱摂動向けで一般四次法ではなく、weight-ranked splitだけで適用条件は保証されない。**

境界融合は同一context・内部family・tauの隣接DStageだけ。Tauを保って反復数を足し、融合後literal列が元の連結と同一になるようにする。異なる時間幅のD近似を、時間を足した一つのPFに置換しない。Rは無条件の障壁でactual adjointへの入力も拒否する。内部blockの追加最適化はしない。
q=2、n_D=1でのD run／R occurrence／D kernel総反復は、outer-S2が3／2／4、refined f=2が5／4／8、Yoshida4が11／6／12、SABA2が5／4／6。静的構造でありnative価格・depth・shotではない。

## 4. Finite RTE・normalization・phase

各R occurrenceに固有ID、signed duration、r_RTE、K、tolerance、seedを保持し、`RTEStage.make_config` が実時間を符号のまま既存factoryへ渡す。現在のplan生成は共通r／Kを各specへ明示する。下位specは異なるr／Kも表せるが、初回案ではstage別の最適化を行わない。

各stageのcorrected meanをA_j、normalizationをN_jとするとrawはA_j/N_j。独立occurrenceについてDを交互に挟むordered積を作り、`M_corr=(∏N_j)M_raw` を照合する。Normalization nodeには全occurrenceの依存を順序付きで含め、実N_j列・積・attenuationを返す。Mean共有は同じstage-specの場合だけで、共有してもoccurrenceの積から項を落とさない。共通N^qを仮定しない。

`independent_occurrences=True` を明示しないplanは拒否する。これは平均対象の条件であり許可証明ではない。Seedはoccurrenceごとに区別するが、samplingは今回0件。将来samplerを接続する場合はstreamと独立drawを確認し、seedの違いだけで独立性の証明としない。本案はanalytic momentsのみ。

負時間meanを正時間meanのinverseとせず、fresh負時間sampleを実現済み列のactual inverseにしない。Scalar／event phaseとcontrol-zero規約をmetadataに残し、位相fitで消さない。新random law／finite-bit certificateは追加しない。**Controlled circuitの数値検証は未実施**で、新graphも既存Settings同様system operator用である。

## 5. Correctnessと実行境界

`evaluate_outer` は既存evaluateを呼び、schemaだけを新outer用に識別する。全correctness nodeを完了してからoperator_error／signalsを始める。失敗は例外を伝播し、比較継続・retryをしない。休眠backendは次を用意したが、数値correctnessは未検証。

1. 既存入力gate：Hamiltonian／D/R／scalar identity、sector／state、Hermiticity、精度、非可換性。
2. 弱摂動診断：||D||、||R||、比、||[D,[D,R]]||、||[R,[R,D]]||。有限値を確認するが、自動weak閾値判定なし。
3. 全D stageをadapter展開を使わない数学的S2積と照合。Exact-D/R outer列もappend積とは別の右乗算による数学積と照合し、phase fitなし。
4. 全occurrenceのN_jを使ったrawのrescalingとcorrected meanの整合を検査。不一致なら比較前STOP。
5. 全gate PASS後、exact physical、exact-D/R outer、近似D＋ideal-tail、finite corrected、rawを分離して比較。

Outputsはouter／ideal-tail／correctedのphysical operator誤差、finite-RTE差、normalization整合、physical／ideal-tail／corrected／rawのX/Y・半径・複素signal直接誤差。信号は既存 `_signals` へ接続する。非有限値・normalization range不足はINCONCLUSIVEとしてSTOPする想定で、科学的優劣を自動認定しない。

**新outer専用の本番launcher・scope bindingは未接続。** 旧 `run_b2d_stage1.py`／preflight／scopeを転用・解除していない。旧scopeが新graphを拒否することを人工確認した。条件採択後に、既存fresh-output排他作成・失敗保存・STOP部品へ新graphを最小接続する。未採択gridを起動可能な既定値として埋め込まない。

## 6. 人工テスト

新規 `tests/management/test_b2d_outer_preparation.py` **19件PASS**、既存6ファイル **72件PASS**、合計 **91件PASS**。CPython 3.11.0rc1、ローカル人工確認でありCIや科学性能検証ではない。

| 対象 | 確認内容 |
|---|---|
| Stage・D内部 | 既知係数、負R、SABA2の2 R、数学／append順、正負T、refinedの同一T、A2／A3旧列と独立参照側literal列 |
| Fusion | 同一tau連結一致、異tau／family／context非融合、R障壁、tailのinverse拒否 |
| RTE・mean | Signed factory、固有occurrence／seed、異なるmock N_jの積、raw／correctedとordered依存、相関条件の拒否 |
| Phase・実行境界 | 全T scalar、event／control-zero metadata、科学import barrier、opaque mock、旧scope拒否、全gate先行、失敗時比較／retryなし |
| 既存保存・STOP | 旧launcherのfresh output競合、入力／gate失敗、途中失敗の部分保存、保存失敗時STOP、SOURCE／scope拒否の回帰 |

標準ライブラリのみ、numpy／scipy／Qiskit／化学系／trotterlib等のimportを禁止した。新backend接続は `__new__` とopaque stubで確認し、科学constructorを呼んでいない。Hamiltonian／state／matrix exponential／commutator norm／signal／Gaussian基底回路／native build・compileは全て0件。保存結果の新数値解析も0件。

```bash
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_outer_preparation.py -v
```

既存回帰は同オプションで `test_b2d_{adapter,evaluator,input_preflight,stage1_launcher,execution_ready,simple_execution}.py` を個別実行し、各15、14、16、15、4、8件PASS。既存mock資産を再利用した。人工PASSを科学的改善の証拠としない。

## 7. 限定科学契約案（未採択・未承認）

| 条件 | 推奨する最小案・未確定事項 |
|---|---|
| 入力 | Stage 1と同じsynthetic 2-mode、one-body＋3 DF、N=1／mode0、L_D=2 weight-ranked。実identityは新実行内で検査 |
| Recipe | 既存InputRecipe、`tests/test_df_partial_s2_repeated.py:_nontrivial_basis_case`、定義commit `52ba88687f92d089921b207e60aec006f04141de`／blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`。別fixture探索なし |
| T | ±0.18を初回候補（旧δ=±0.09、q=2に対応）。新gridとしては未採択。±0.27や追加時刻は自動追加しない |
| q／refinement | 基本q=2、refined f=2を最小案。f=3はYoshidaとR回数を合わせる代替案で、自動追加しない。q／fは判断待ち |
| Outer集合 | Outer-S2／refined-S2／outer Yoshida4／SABA2の4方式。効果未確認 |
| D内部 | A2、n_D=1が基本。A3、n_D=1を同じ4方式の内部精度対照とする案。主8行、対照込み16行。A3採否・n_Dは未採択。A4なし |
| RTE | 全signed stageでr_RTE=2、K=2、既存tol=1.0 metadata、独立occurrence、analytic moments。各stageのbiasとN_jを保存。新samplingなし |
| 精度・gate | complex128、atol=1e-12、係数dropなし、phase診断無効を推奨。改善認定／弱摂動閾値ではない |
| 参照・診断 | Exact-D/R classical outer、独立列gate、同じHとψ、ideal-tail／finite／raw、operator／複素signal、二重commutator。Exact参照を安い量子oracleに数えない |
| 費用 | Stage数、D run、kernel／block数、signed absolute time。Native価格・depth・shot・同資源budget判定は含めない |

### 科学call scopeの具体案

上の候補条件を**架空identityだけのmetadata**で展開した静的数。入力行列・実RTE configは作っていない。条件を変えた場合は再展開してからSOURCE・契約を固定する。

| 境界 | A2主集合8行 | A2＋A3対照16行 |
|---|---:|---:|
| 共有DAG dispatch | 342 | 874 |
| 共有exponential key／backend.expm | 94 | 490 |
| Stage-spec別RTE config／moments | 各24 | 各24 |
| `signal.expectation` | 32 | 64 |
| `validate_input`／`outer_diagnostics` | 各1 | 各1 |
| 独立D stage数学gate（共有後） | 20 | 40 |
| Outer数学順gate（共有後） | 8 | 8 |
| Raw／corrected rescaling gate | 8 | 16 |

Exact physical参照は2種類のTに各1個、計2個を共有し、同じstage-specのmoment、同じblock／time指数、D内部gateも共有する。R occurrenceは省かない。A4／published-full-word gateはない。旧 **941 dispatch／448 keyは流用しない**。上表の境界は入れ子を含み、足して総call数・総費用としない。

入力生成は既存loaderの10高位境界（fixture、split、preparation、sector、full、D block×3、tail、state）を再利用する案。内部の小規模Gaussian basis／sector materializationも科学処理であり、**将来の明示実行許可に含めてから**起動する。新native PF回路やfull controlled wrapperの生成許可と区別する。
条件採択後、新graph node集合・RTE stage-specの登録を既存scope記録／保存へ最小接続し、入力展開を含む実登録境界と出力サイズを確認する。旧scopeをそのまま使用しない。Graph、入力identity、依存version・精度・環境、比較行、correctness、各N_j、部分結果・失敗理由・provenance・STOPを必要な範囲で保存する。環境のbitwise一致・承認証明は要求しない。

### 保全・失敗・資源

- Fresh output候補は `artifacts/research_b_outer_splitting/<execution-date>/<fresh-attempt>/`。排他的作成、既存run／STOPの上書き・resume・削除なし。
- 入力／独立参照FAILは比較前STOP。非有限値・識別力不足はINCONCLUSIVE。改善なしは科学的結果として保存し、係数／入力／精度を自動変更しない。
- 初回案はretry／resumeなし、完了・失敗後mandatory STOP。復旧例外が必要なら対象と同条件fresh attemptをユーザーが別途許可する。旧Stage 1の復旧例外は流用しない。
- 同じTの誤差比較と同資源budget比較を区別する。単一窓から収束次数、SABA2一般四次、scientific GO、native／全RPE総資源優位を認定しない。
- 実行前にCPU／RAM／GPUと支配処理を確認し、既存cache・sector・vectorized演算を利用。重ければ独立task並列化を準備する。恣意的なCPU／RAM／worker／wall上限なし。共有環境・他ユーザーを変更しない。

## 8. 未確認事項と次の担当

新outerでの実入力identity、数値的D／outer correctness、弱摂動／commutatorの関係、finite RTE込みのoperator／signal改善、complex128の識別力、native／shot費用は未確認。既存Stage 1 PASSや人工PASSから補完しない。

**研究Bのユーザー：** T、q／f、n_D、A3対照の採否、r／K、精度と比較集合を確認する。
**Codex：** 条件決定後、新graphのscope／入力／fresh-output保存を既存部品へ最小接続し、科学SOURCE・call scopeを固定する。研究Bチャットでユーザーが明示的に科学実行を許可した場合だけ実行する。Native compile、sampling、Stage 2、保存結果再解析へ自動進行しない。

今回のSOURCE・人工test・本書をcommit／pushし、remote SHA・内容照合後mandatory STOPする。通常の研究B作業として統括報告は行わない。

[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
[PF]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[SABA]: https://arxiv.org/pdf/astro-ph/0005074v1
