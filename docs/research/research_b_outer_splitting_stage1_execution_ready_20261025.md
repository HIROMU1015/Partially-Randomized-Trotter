# 研究B：Outer D/R Splitting 16行科学検証・最終実行準備

2026-10-25 JST。**技術接続・人工検証・metadata preflightを完了。科学実行は未承認・未実行。新outerの実operator／signal結果は存在しない。**

## 1. 固定SOURCEと引継ぎ

| 対象 | 固定版 |
|---|---|
| 作業起点 | `2e46050bd85e2df86c980e501f727ff56d10823e`、[前段の準備・契約案][PREP] |
| **今回の科学SOURCE候補** | **`b85dd328423de5b1159f11939835f04db90f3187`**、[SOURCE][SOURCE] |
| 独立branch | `codex/research-b-outer-stage1-execution-ready-20261025` |
| 前段B2-D実行 | [Stage 1結果][RESULT]、実際の完走SOURCE `758a05ed6aac58a5945af5a5330c792a842b174d` |
| 参照調査／PF | [静的費用・outer調査][SURVEY]、published PF固定 `20d64c2fea1ed2d2188777df4f3d962ee94df486` |

科学SOURCEを先にcommitして固定し、本書・索引は後続の文書commitで公開する。科学checkoutは上のSOURCEそのものを使う。旧SOURCE・raw・STOP・契約を遡及変更せず、保存CSV／rawの新しい数値再解析もしていない。前段のA3対A4はINCONCLUSIVEのままで、outerが一般の支配誤差とは認定しない。
共通運用は前作業で全文確認した `research_project_instructions_unified_2026-10-25.md` を適用する。ユーザーの承認は実行許可であり、原文保存・certificate・packet・receipt・grant・管理者操作は要求しない。C1最優先、他研究・共有環境の非干渉を維持する。

## 2. 完成した接続

| SOURCE | 変更／再利用 |
|---|---|
| `outer_execution.py`（新規） | 固定16 case、SOURCE／input binding、完全なgraph照合、新scope、`OuterRunContext`／`OuterScopeLedger`。既存RunContextとScopeLedgerを継承 |
| `scripts/run_b2d_outer_stage1.py`（新規） | `--inspect` はmetadataのみ、科学入口は明示的な `--execute --source-commit … --output …` |
| `launcher.py` | `launch_outer` とouter接続分岐を追加。既存排他的mkdir、Input loader、入力保存、Evidence、部分結果・失敗保存、STOPを共用 |
| `scope.py` | 入力10境界を共通定数へ抽出。Registry検証、RTE identity、比較／保存node種別の最小override接口を追加。旧36行経路の条件は維持 |
| `outer.py`／`outer_backend.py` | 前段のstage／RTE／D内部PF／独立参照を再利用。独立outer参照をcaseから数学順に別構成。Rawとphysical、近似D＋ideal-tailとexact outerの直接差を追加し、5対象のsignalを明示 |
| 既存部品 | `evaluation.py` の共有DAG・全gate先行、`dense_backend.py` の入力検査・expm cache・signal、Input package、DF preparation、finite-RTE config／momentsを再利用。新Hamiltonian生成器、PF探索、sampler、signal estimatorは作らない |

既存36行B2-Dに対するsource変更は共通部品の必要な接口拡張に限定し、旧callerの関数署名・固定scopeを維持した。科学SOURCE候補は新commitとして区別する。前段の実装・技術能力から、数値的正しさ・改善を認定しない。

## 3. 未承認の最終科学実行契約

以下は今回**実装準備条件として固定**した内容である。契約作成や技術準備の許可は、科学実行許可ではない。

| 項目 | 固定内容 |
|---|---|
| Fixture | 既存synthetic 2-mode、one-body＋3 DF因子。分子geometry／AO basis／分子DF rank探索は非該当 |
| 定義SOURCE | 既存InputRecipe、`tests/test_df_partial_s2_repeated.py:_nontrivial_basis_case`、commit `52ba88687f92d089921b207e60aec006f04141de`、blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043` |
| Sector／state | N=1、mode0 occupation、既存sorted OpenFermion JW sector順 |
| D/R分割 | L_D=2、weight-ranked prefix、`lambda_frobenius_squared`、係数dropなし、scalar／identity分離 |
| Physical時間 | T=+0.18、−0.18（厳密なFraction ±9/50） |
| Outer | S2／refined-S2／Yoshida4／SABA2、基本q=2、δ=T/q=±0.09 |
| Refinement | f=2、幅T/(qf)、qf=4 step、同じT |
| D内部 | A2 Yoshida4／A3 published Morales Y8m10b、n_D=1 |
| RTE | 各signed stageでr_RTE=2、K=2、tolerance metadata=1.0、seed基点41、occurrence別spec、analytic momentsのみ |
| 精度／判定 | complex128、correctness atol=1e-12、phase診断無効、記述的診断のみ |
| 行数 | 2 T × 4 outer × 2 D内部 = **16行** |

A4、f=3、n_D=2、追加時間、別fixture／sector、PF係数探索は含めない。入力はfresh runで既存loaderから生成し、同じphysical Hamiltonian、state、sector、D/Rを全行で共有する。新入力identity・数値正しさはまだ未検証。

### Stage・作用順・phase

数学的積は `J2(t)=D(t/2)R(t)D(t/2)`。Yoshida4は `J2(aδ)J2(bδ)J2(aδ)`、既存generatorの `a=1/(2−2^(1/3))`、`b=−2^(1/3)/(2−2^(1/3))` を再利用し、中央負時間Rを維持する。Float係数を無断で補正しない。公称Tと実際のD／R時間和は別のmetadataに保存する。
SABA2は `D(c1δ)R(δ/2)D(c2δ)R(δ/2)D(c1δ)`、`c2=1/√3`、`c1=(1−c2)/2` の既存列で、1 macroにRが2個。弱摂動向けで一般四次保証はない。理想Yoshida4の保証もinexact D＋finite RTEへ自動継承しない。

Append順と数学的積順を区別する。隣接Dは同一context・family・tauのkernel連結だけを融合し、異なる時間幅の近似を時間和の一つのPFへ置換しない。R障壁越しの融合は禁止。Scalar／抽出identity phaseは公称Tに一度付与し、event phaseとcontrolled relative phaseの規約を保持する。今回の数値経路はsystem operator用で、controlled回路を生成しない。

### Finite-RTEの平均対象

各Rの実signed durationから既存factoryでconfigを作る。Corrected mean A_j、raw mean A_j/N_jを区別し、独立occurrenceのordered D/R積について `M_corr=(∏N_j)M_raw` を確認する。各N_jと出現順・回数を残す。共通N^qを仮定せず、同じstage-specの計算共有でも物理occurrenceを落とさない。
負時間meanを正時間meanのinverseにせず、fresh負時間sampleをactual inverseと呼ばない。新sampling law・finite-bit保証は導入しない。Taylor bias、D近似、outer splitting、raw attenuationを分けて記録する。

### Correctnessの順序

1. Fixture／split／preparation、Hamiltonian identity、sector／state、Hermiticity、state norm、必要な非可換性を既存入力gateで確認する。
2. 弱摂動診断 `||D||, ||R||, ||R||/||D||, ||[D,[D,R]]||, ||[R,[R,D]]||` を保存する。有限値を要求するが、恣意的なweak閾値は設定しない。
3. 全D内部PFを、adapterのappend展開を使わない独立数学積と照合する。
4. Caseから別構成した未融合の数学順exact-D/R outer積と、実現列のappend積を照合する。参照のD指数はexactなので、近似Dの不正融合を正当化しない。
5. 全occurrenceのnormalization積によるraw／corrected rescalingを照合する。
6. **全gate PASS後だけ**16行のoperator／signal比較を開始する。失敗したら比較せず保存・STOP。

### 評価対象・出力

同じψ、Tについて、(1) exact physical、(2) exact-D/R outer、(3)近似D＋ideal-tail、(4)finite corrected、(5)rawを区別する。Physicalを基準とする4種のoperator誤差、ideal-tail対exact outerのD近似差、corrected対ideal-tailのfinite-RTE差、normalization整合を保存する。
全5対象のX/Y・半径・複素signalを既存signal estimatorで扱い、4種のphysical直接signal誤差、D近似のsignal差、finite-RTEのsignal差を記録する。位相をfitして誤差を消さない。小さな記述差を科学的GOや総資源利益へ格上げしない。

## 4. 新scopeの実確認

以下は実装SOURCEからmetadataとして再生成し、全経路mockでattempted／completedの一致まで確認した値である。科学callを実際に消費した記録ではない。

| 登録境界 | 件数 |
|---|---:|
| 比較行 | 16 |
| 共有DAG dispatch | **906** |
| 共有exponential key／backend.expm | **490** |
| Stage-spec別RTE config／moments | 各24 |
| Signal expectation | **80** |
| 入力10高位境界 | fixture／split／preparation／sector／full各1、D block3、tail／state各1 |
| 入力gate／weak診断node | 各1 |
| 独立D数学gate／outer数学順gate／rescaling gate | 40／8／16 |

前案874から増えた32 dispatchは、各16行のraw-physical operator誤差とD近似直接差。Signal64→80は各行でexact outerを含む5対象にしたためで、比較条件・PF係数は変えていない。旧36行の941 dispatch／448 keyは流用していない。
Exact physicalは2 Tに各1 key、同じD block／timeとRTE specを共有する。独立outer参照の未融合D時間も登録490 key内にあることをmock cacheで確認した。全occurrenceをordered積へ残す。親子・入れ子の境界を足して全jobの独立科学call数としない。

入力preparationには既存のone-body／DF Gaussian basis構成、tail basis circuit／sector変換を内包する。これらも**将来の科学実行許可に含める処理**であり、今回は起動していない。内部Gaussian operation、BLAS、Taylor／matrix productのleaf呼出し総数は独立に計数していない。固定SOURCEの高位境界内に包摂し、scope外の追加計算を許可する意味ではない。

## 5. Fresh output・保存・STOP

SOURCE・fixture定義・固定全条件・graph全体・scopeを照合し、生成後には実input context／配列identityをbindingする。新context・scopeを旧36行と分離し、dependency versionsと実行環境を記録する。環境のbitwise一致や承認証明は要求しない。
Outputは下記専用pathを排他的mkdirで一度だけ作る。既存dir／STOPがあれば拒否し、上書き・resume・削除しない。

`/tmp/prt-research-b-outer-stage1-science-20261025/artifacts/research_b_outer_splitting/20261025/once-16rows`

保存は既存の9ファイル：binding.json、inputs.json、inputs.npz、graph.json、events.jsonl、partial_results.json、result.json、status.json、STOP.json。共通Evidenceを再利用し、途中まで完了した行・失敗理由・call実績を保持する。保存失敗時もSTOPの保存を試み、正常完了と報告しない。
既存の出力枠を再利用する（計16 MiB、JSON文書1 MiB、event行64 KiB、STOP予備256 KiB）。Logの件数枠は新scopeから3100 recordsと算出する。CPU／RAM／worker／wallに新しい人工上限を設定しない。

正常完了、科学的FAIL、INCONCLUSIVE、技術例外の全てでmandatory STOP。Retry／resumeなし。改善なしは科学的結果として記録し、入力／精度／係数／gridを自動変更しない。旧Stage 1の復旧例外を流用しない。必要な復旧や別条件・Stage 2は別途ユーザー判断へ戻す。

## 6. 人工検証とmetadata preflight

**既存91＋今回14＝105件PASS**（ローカル、CI／科学性能証拠ではない）。今回の14件は `tests/management/test_b2d_outer_execution.py`。固定16行、全条件／stage-spec、graph・scope変異拒否、仮input名／scalar metadataのbinding、全登録callのmock照合、5対象signal glue、全gate先行、入力／数学／rescaling失敗で比較0、途中結果保持、INCONCLUSIVE、保存失敗STOP、fresh競合、CLI誤起動防止、承認証明を要求しない通常実行引数を確認した。

```bash
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_outer_execution.py -v
```

人工テストはCPython3.11.0rc1で実施した。既存7ファイルも同じオプションで個別実行：adapter15、evaluator14、input_preflight16、stage1_launcher15、execution_ready4、simple_execution8、outer_preparation19、計91件PASS。旧36行経路も回帰PASS。標準ライブラリとopaque mockだけを使い、科学moduleのimportを禁止した。科学的な行列／state／実RTE operator／signal／Gaussian circuit生成、native build／compile、保存数値再解析は0件。

さらに固定SOURCEのclean detached checkoutで、実行予定の既存環境を使った**読取り専用metadata preflight**をPASSした。科学import guard下でSOURCE、fixture bytes、16行scope、依存availability／versions、fresh output非存在を確認し、科学module loadは0、output作成も0。
環境は既存 `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、CPython3.11.9、NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Qiskit Nature0.7.2／OpenFermion1.6.1／OpenFermion-PySCF0.5／PySCF2.7.0／Numba0.60.0／Matplotlib3.9.2。共有環境は変更していない。
32 CPU、空きRAM約52.7 GiBを確認した。将来の小規模2×2 sectorでは既存vectorized演算・指数／moment共有cacheと全correctness先行を使う。不要なprocess分割やCPU／thread制限を加えず、実行直前の空き資源・非干渉を再確認する。実測の科学速度倍率は主張しない。

## 7. 許可後にだけ使う単一command

科学SOURCEのclean checkoutを用意済み。次にユーザーが研究Bチャットでこの固定契約の科学実行を明示的に許可した場合だけ、Codexが以下を起動する。**今回は起動していない。** `--execute` は誤起動防止で、許可を証明するものではない。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-stage1-science-20261025/scripts/run_b2d_outer_stage1.py \
  --execute \
  --source-commit b85dd328423de5b1159f11939835f04db90f3187 \
  --output /tmp/prt-research-b-outer-stage1-science-20261025/artifacts/research_b_outer_splitting/20261025/once-16rows
```

`--inspect` はmetadataのみ。実行許可ファイルの保存・発行操作は不要である。

## 8. 未確認事項と完了時境界

新outerの実input identity、A2／A3数学的数値correctness、独立outer順・normalization整合、弱摂動commutator診断、complex128での識別力、finite-RTE込みのoperator／signal改善は**未確認**。前段PASSや人工105件から補完しない。Native full wrapper、controlled費用、shot数、finite-bit certificate、全RPE費用は範囲外。
次の担当は**研究Bのユーザー**：固定16行契約の科学実行を許可するか判断する。許可された場合だけ**Codex**が同一SOURCE・条件で一回実行し、correctness先行、結果・失敗・実績保存、remote公開・照合後STOPする。追加研究へ自動進行しない。
今回はSOURCEと本書を公開・remote照合後mandatory STOPし、通常の研究B作業として研究統括への報告は行わない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b85dd328423de5b1159f11939835f04db90f3187
[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e46050bd85e2df86c980e501f727ff56d10823e/docs/research/research_b_outer_splitting_stage1_preparation_20261025.md
[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
