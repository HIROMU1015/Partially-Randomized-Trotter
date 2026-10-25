# 研究B：A0独立高精度検証の実行結果

2026-10-25 JST。**A0はCOMPLETE／EMPIRICAL_CONSISTENCY。B0はSTOPを維持し、実行していない。** 固定8行×90/128桁を一回で完了し、修正・中断・retry／resumeはない。保証付きu_num、native費用、shots、総資源優位を認定しない。

## 1. SOURCE・条件・原本

| 項目 | 固定値／保存先 |
|---|---|
| 科学SOURCE | `1e3921ba16e5d531e9a3010cffa9ff717609a93a`、実行中・実行後の変更なし |
| 実行契約 | [3c1e10f1cのA0仕様・scope・command](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c1e10f1c061b9b4d15e2690403fd26b841cbc06/docs/research/research_b_outer_a0_b0_execution_preparation_20261025.md) |
| 入力snapshot | 結果公開commit `33715848bfb83d49ec4e54d6eef481012a7441e4` の[旧22行原本](../../artifacts/research_b_outer_rte22/20261025/analytic-once/) |
| A0 raw8ファイル | [independent-once/](../../artifacts/research_b_outer_a0/20261025/independent-once/)：binding／inputs／graph／events／partial_results／result／status／STOP |
| 全8比較の列抽出 | [COMPARISON.csv](../../artifacts/research_b_outer_a0/20261025/COMPARISON.csv)、128桁の保存文字列と90/128・旧値比較を保持 |
| Process記録 | [execution_logs/attempt-1/](../../artifacts/research_b_outer_a0/20261025/execution_logs/attempt-1/)：preflight／stdout／stderr／time |

公開branchは `codex/research-b-outer-a0-results-20261025`。結果worktreeは準備文書commit3c1e10f1cから分離し、実行したscience checkoutは旧SOURCEのまま保全した。旧36／16／22行のSOURCE・raw・契約・grant／ledger／marker／STOPを変更していない。原本8ファイルとprocess4ファイルの公開用コピーは実行場所のbytesと一致する。CSVと以下の表は今回の実行で保存された数値の列抽出・表示であり、operator／signal／boundの再評価ではない。

系は既存synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1。分子geometry／AO basis／分子DF rank探索は非該当。T=±0.18の両符号、S1=SABA2/q4/A2/r2K2、F1=refined-S2/q2/f8/A2/r2K2、B3=outer-Yoshida4/q2/A2/r2K2、R4=outer-Yoshida4/q2/A3/r2K4の8行。新fixture・係数・gridを追加していない。

## 2. 実行とscopeの確認

実行直前のclean checkoutでHEAD、保存入力5ファイルのGit blob、固定scope、依存versionとorigin、fresh output非存在を確認した。NPZは実行開始後にだけ読み、complex128 primitiveの各実部・虚部を二進有理数としてliftした。時間・scalarのFractionと既存float由来PF／SABA係数を保持し、理想実数係数・再DF分解・再対角化へ置換していない。保存physical Hamiltonianは独立targetのままで、D+R+cIへ上書きしていない。

既存 `precision_context`／`exact` を使った局所contextで、90桁Taylor exponentialと128桁Padé exponential、独立した数学順のmatrix product、finite polynomial／r回積、scalar phase、state expectation、SVDによるoperator 2-normを評価した。DenseBackendのcomplex128演算は呼んでいない。同じmpmathと保存モデルを共有するため、独立アルゴリズムの経験的比較であって別実装・別入力による外部再現ではない。

| 登録境界／実績 | 完了数 |
|---|---:|
| 行×精度 | 8×2＝16 |
| 高位target／signal | 各80 |
| DAG node | 3,400 |
| 共有exponential | 412 |
| Explicit multiply | 2,348 |
| Operator residual 2-norm | 96 |
| D／outer unitarity診断node | 40 |
| Stage normalization／Taylor bound | 各24 |
| 全体normalization／bound | 各16 |
| Snapshot／reference初期化 | 各1 |
| 90/128と旧値の比較 | 8 |
| Event記録 | 6,820、上限6,852内 |

全kind／identityのattemptedとcompletedが契約の登録数に一致し、failedは空。16件の途中結果、8比較、最終result、statusとSTOPを保存し、evidence_write_failuresは空だった。親子・library内部の演算を足して独立科学job数と呼ばない。未登録・超過call、入力不一致、追加精度・条件への自動拡張はなかった。

## 3. 90/128桁の整合とnominal bound

8比較すべてで、保存signal・operator residual・normalization・合成boundの90/128桁差が固定目安1e-40以内だった。全比較指標の最大差は**1.843847384×10⁻⁸⁹**（F1、T<0）。これはfull matrix全成分のinterval包含やcertified errorではなく、登録済み比較指標の経験的安定性である。

以下は128桁保存値の表示。Finite-RTE差はcorrectedとideal-tailのoperator 2-norm、nominal boundはstage別正項Taylor remainderの合成値。旧complex128の同じoperator指標を比較のため併記する。

| ID | T | 旧complex128 finite-RTE差 | A0 finite-RTE差 | A0 nominal bound | Consistent |
|---|---:|---:|---:|---:|---|
| B3 | +0.18 | 4.896109706e-10 | 4.896109544e-10 | 4.901855972e-10 | true |
| B3 | −0.18 | 4.896109712e-10 | 4.896109544e-10 | 4.901855972e-10 | true |
| R4 | +0.18 | 3.589109754e-16 | 5.523329131e-16 | 5.528097987e-16 | true |
| R4 | −0.18 | 3.631917410e-16 | 5.523329131e-16 | 5.528097987e-16 | true |
| S1 | +0.18 | 5.098455973e-13 | 5.077665782e-13 | 5.078650737e-13 | true |
| S1 | −0.18 | 5.098455973e-13 | 5.077665782e-13 | 5.078650737e-13 | true |
| F1 | +0.18 | 6.620415605e-14 | 6.347082459e-14 | 6.347720334e-14 | true |
| F1 | −0.18 | 6.620435837e-14 | 6.347082459e-14 | 6.347720334e-14 | true |

S1／F1の旧浮動評価で見られたnominal bound超過は、今回の保存binary64モデルに対する高精度評価では再現しなかった。Bound自体の旧floatとの差はS1約4.11e-29、F1約2.36e-28に対し、finite-RTE operator差の旧値との差は約2.08e-15／2.73e-15だった。超過が通常floatのbound値だけに起因したとする説明は支持されず、operator評価側の数値誤差の影響と整合する。

ただし、exponential、積の蓄積、norm評価それぞれの寄与は個別に分離していない。旧値を修正・置換せず、今回の独立参照と旧値の差として保存した。理論boundの一般的正しさや元の化学Hamiltonianに対する保証を新たに証明したわけではない。

## 4. Physical signalと小残差

以下も128桁保存値で、正負Tはこの表示桁で一致した。Rawを含む5対象のmatrix／complex signal、両精度全16記録、stage boundとdiagnosticsはraw JSONに残す。

| ID（両T） | Corrected対physical operator誤差 | Corrected physical coherent signal直接誤差 | D近似operator差 | Raw対physical operator誤差 |
|---|---:|---:|---:|---:|
| B3 | 1.751922586e-9 | 6.756867015e-10 | 1.345301963e-9 | 3.657596933e-4 |
| R4 | 1.437122887e-10 | 7.003832053e-11 | 2.303954862e-19 | 3.657596933e-4 |
| S1 | 1.456690275e-9 | 7.132180054e-10 | 1.658080440e-11 | 1.396355407e-5 |
| F1 | 7.954210004e-9 | 3.538258858e-9 | 6.227794508e-14 | 6.981808439e-6 |

R4の旧D近似差約1.57e-14は、今回約2.30e-19だった。旧complex128ではこの小差を適切に識別できていなかったことを示す限定診断であり、D誤差ゼロや一般的な収束次数を意味しない。R4 finite-RTE差は約5.52e-16で非零として残る。全5対象signalの旧complex128との差の最大は約1.69e-14（R4）、corrected physical signal誤差の旧値との差は最大約2.34e-15だった。

今回の4候補・同じ保存モデルに限れば、corrected signalの大きさはR4、B3、S1、F1の順だった。22行全候補の新しいランキング、同精度達成、native／shot／総費用の優位を評価した結果にはしない。

## 5. 保存モデルの前提と保証の限界

128桁の入力診断には次が残った。高精度化しても保存primitiveの丸めは消さず、値を対称化・再正規化していない。

| 診断 | 保存値の表示 |
|---|---:|
| H_phys − D − R − cI のoperator 2-norm | 4.403221638e-17 |
| RのHermiticity差 | 4.609643360e-18 |
| DのHermiticity差 | 0 |
| Normalized tailのoperator 2-norm | 1.000000000000000001680096819… |
| State norm | 1 |
| Scalar phaseの旧binary64との差 | 2.003845178e-17 |

厳密なunitary／normalized-tailを仮定する理論の前提は、保存binary64値のexact liftでは完全な等号にならない。Nominal Taylor boundは丸め方向を保証したinterval上界ではなく、微小な前提差を含むcertified boundにもしていない。今回の余裕内で経験的整合したという結論を維持し、**全16記録・8比較でu_num=null、certified=false**とする。90/128桁一致を保証付きu_numやγへ読み替えず、ε=1e-8／1e-9の資格、必要shot数、有限bit certificate、全RPE総資源は未確定のままである。

## 6. 実行環境・実績・保全

既存venvのCPython3.11.9、mpmath1.3.0、NumPy1.26.4、Linux x86_64。Module origin、SOURCEとsnapshot5ファイルblobをpreflight／bindingへ保存した。依存install・共有設定の変更はない。実行直前は32 CPU、空きRAM約52.7 GiB、/tmp空き約330 GiB。GPU driverは通信不能だったが、固定CPU高精度backendを使用した。

2×2 sector、412 exponentialとD／outer productの共有を使用し、数秒の小行列処理をprocess分割していない。CPU／RAM／worker／wall-timeに人工的な低い上限は設定しなかった。Whole-process wall **2.36秒**、user CPU3.55秒、system CPU2.95秒、最大RSS61,424 KiB、exit0。Stderrは空。速度倍率の比較測定は行っていない。

技術中断0、SOURCE修正0、再実行0。準備段階の人工138件PASSは既報であり、今回SOURCE変更がないため同じ人工suiteを繰り返していない。今回の保存後確認はscope／event数・row数・全比較flag・STOP・保存失敗0・原本copy一致を確認したもので、科学再実行ではない。

実行commandは契約どおり次の一回だった。Stdout／stderrと`/usr/bin/time -v`のprocess記録は上記artifactへ保全した。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-a0-science-20261025/scripts/run_b2d_outer_a0.py \
  --execute \
  --source-commit 1e3921ba16e5d531e9a3010cffa9ff717609a93a \
  --output /tmp/prt-research-b-outer-a0-science-20261025/artifacts/research_b_outer_a0/20261025/independent-once
```

この出力は使用済みで、上書き・resume・再起動しない。B0 checkout、B0 SOURCE `8d0cb96df9d1f6229f207550634c7e11aff2851c`、B0 output候補には科学実行による変更を加えていない。

## 7. 最終状態・次の担当

最終状態は**A0 COMPLETE、scientific_status=EMPIRICAL_CONSISTENCY、runtime_status=COMPLETED_DESCRIPTIVE_MANDATORY_STOP**。全8比較で固定精度安定性とnominal bound整合を確認できた。保存入力の理論前提差、rigorousな数値包含、誤差寄与の個別分離、精度を含む総資源保証は未確認である。

Codexは結果・研究記録を公開しremote SOURCE／成果物を照合した後にmandatory STOPする。次の担当は研究Bのユーザーで、この結果を踏まえて保証付き包含A1を必要とするか、別許可単位B0を実行するか判断する。**B0は引き続きSTOP**。A1、B0、期待費用MC、shots、追加精度／gridへの自動移行は行わない。
