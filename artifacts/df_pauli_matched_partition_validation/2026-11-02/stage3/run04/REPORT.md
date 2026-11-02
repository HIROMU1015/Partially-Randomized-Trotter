# P3 Stage 3 run04：独立32 trajectory/方式・Full128価格確認

2026-11-02 JST。**承認範囲の価格確認は完了。新64 Full bodyすべてPASS、128/128 compile成功。**
[STOP](STOP.json)：`STOP_STAGE3_PRICE_CONFIRMATION_COMPLETE_NO_AUTOMATIC_FOLLOWUP`。
新sampling64回・64 trajectory・2048 event位置。新T/DF対角化・Givens合成、E/R/Empty、旧価格の再compile、旧pilotのpooling、自動retry、新signal/B/N/Gはすべて0。
旧Stage 2 finite biasの未解像状態とStage 3 run01/run02/run03の全結果・STOPを保全した。

## 科学条件・SOURCE・保存証拠

linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10。
案A：共通Gaussian T_eff保持、Pauli184残差M_BとDF rotated Z/ZZ360残差・10 basisのM_C。
T=exact binary64 .8（3602879701896397/4503599627370496）、S2/q8/r4/K4、32 event/trajectory。
ancilla8、cosine/+Im sine、measure込みfull wrapper、P=0。state準備費用を除く価格だけで、新しい状態信号を生成していない。

- 新科学SOURCE：`1cfce5e961e3d063741a1e0656052b10c7b27392`。人工9 tests PASS後に公開・bytes照合し、そのSOURCEで1 batchを実行。
- 承認対象の設計：`60515a33ce6231e9ff4ffdb40124ed845373351c`の[限定案][PLAN]。旧完成結果：`90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5`の[36価格報告](../run03/REPORT.md)。
- 保存native SOURCE：`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`、run02の81 native PASSを再利用。
- Stage 2：結果`8bcf5851f99f980dc08c5885ace54874b554c815`、科学SOURCE`4bfbbaad1f8178077489bf06b085377256d6fb80`。
- H4保存anchor：`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、元生成SOURCE`0c329b30a81accb35c212be8defa66fee0dd0f12`。
- [SOURCE/runtime/input来歴](source_and_inputs.json)：SOURCE32ファイル、固定科学入力22、保存native artifact7のbytes/hashを照合。

P seed320201–320232、DF320301–320332を昇順index0–31へ固定し、各seedのPCG64/np.choice flat32を1回使った。
旧pilotのseedと混ぜず、各armの新32本だけが以下の集計対象。
同じindexのP/DFは独立標本。cos/sinは同一trajectory内でeventを共有し、各軸32標本である。
価格差を見た後の後続設計であり、pilot以前からの事前登録や検出力保証とは呼ばない。

## 保存Gaussian接続と新Full correctness

run02 archive内の保存half/actual inverse/controlled half/DF basis QPYを原bytes・fingerprintと照合して復元した。
基底gate順・qubit位置・basis ID/hash・成分順・符号・確率を保存lawへ接続し、別gauge・再Givens合成・再対角化なし。
T内部identityと共通外側scalarを元の位置で一度だけ適用し、Full-onlyは旧trajectory_partsのFと同じouter順で合成した。

独立oracleに必要な参照halfは保存Tから1回だけ指数計算し、保存system halfの行列を1回評価した。
[半指数照合](reference_half.json)：spectral差 `6.8105919646146522e-16`、
固定PASS閾値 `2.2737367544324539e-13` でPASS。基準・phase政策は緩和していない。
64 Fullを保存辞書由来の独立DictionaryOracleへcomplex128で照合し、[全64結果](native_preflight.json)を保存した。
最大spectral差 `2.0570383026010926e-14`、最大合成engineering不確かさ `2.192866777114695e-09`。すべて元の積誤差政策でPASS。
全PASS後だけ128 wrapper compileを開始した。これは全law・全compiled作用精度ρの保証ではない。

人工検査は新規5＋既存process4＝9 PASS。人工2-modeのみで、saved QPY/basis hash復元、
Full-onlyと旧Fのliteral一致、order0/2/4・scalar・+Im・測定observableのcompiler回帰、spawn復元、平均/SE/共分散を検査した。
人工PASSと今回のH4 native PASS、科学的な費用観測、全law保証の有無を区別する。

## 価格：新32標本/arm、各軸の標本平均±SE

[price_statistics.json](price_statistics.json)／[summary CSV](price_summary.csv)／[全128 metrics](prices.json)／[個別価格CSV](compiled_prices.csv)。
表の±は標本SEで、信頼区間や母平均上界ではない。平均のexact Fraction・不偏標本分散・全metricはJSON/CSVに保持する。
両軸のRZ/CX/depth平均とSEはこのcohortでは同値だったが、軸価格を別々に記録している。

| 方式 | 軸 | RZ 平均±SE | CX 平均±SE | depth 平均±SE |
|---|---|---:|---:|---:|
| P | cosine | 3017.93750 ± 9.41376 | 1155.12500 ± 4.81696 | 1552.75000 ± 5.21841 |
| P | sine | 3017.93750 ± 9.41376 | 1155.12500 ± 4.81696 | 1552.75000 ± 5.21841 |
| DF | cosine | 8559.71875 ± 150.19704 | 2771.87500 ± 44.79989 | 3942.18750 ± 67.69463 |
| DF | sine | 8559.71875 ± 150.19704 | 2771.87500 ± 44.79989 | 3942.18750 ± 67.69463 |

観測範囲（両軸の主指標は同範囲）：

| 方式 | RZ min–max | CX min–max | depth min–max |
|---|---:|---:|---:|
| P | 2902–3109 | 1108–1205 | 1493–1611 |
| DF | 6789–10178 | 2226–3246 | 3087–4674 |

方式間差は独立標本のP−DFであり、paired差やpaired SEではない。

| 軸 | metric | 平均差 P−DF | 独立標本SE |
|---|---|---:|---:|
| cosine | RZ | -5541.78125 | 150.49176 |
| cosine | CX | -1616.75000 | 45.05811 |
| cosine | depth | -2389.43750 | 67.89547 |
| sine | RZ | -5541.78125 | 150.49176 |
| sine | CX | -1616.75000 | 45.05811 |
| sine | depth | -2389.43750 | 67.89547 |

共通保持・同じtupleを使う今回の独立確認でもPauliの標本平均価格がDFより低い。
辞書・sampling law・native残差実装・basis reuse・compiler非加法性を含む観測であり、表現だけの独立した因果効果ではない。
Stage 2保存BはPauli39.125909965750068、DF34.527999187486905で、今回再評価していない。
単回価格差とnormalizationを結合したN/G、元の純RTE対DF B3の一般的説明・順位は評価していない。

## 同trajectoryのcos/sin共分散と希少event

各arm/metricの不偏標本共分散は以下。軸間の依存を無視して将来の重み付きGの分散を足し合わせない。
今回はN/G・新しい重み付き価格を計算せず、共分散を後続用の証拠として保存した。

| 方式 | metric | cos/sin標本共分散 exact |
|---|---|---:|
| P | RZ | `703279/248` |
| P | CX | `1485/2` |
| P | depth | `27014/31` |
| P | rz_depth | `19119/248` |
| P | cx_depth | `102079/248` |
| P | size | `386002/31` |
| DF | RZ | `716117711/992` |
| DF | CX | `3981947/62` |
| DF | depth | `36367223/248` |
| DF | rz_depth | `29650791/992` |
| DF | cx_depth | `2457143/248` |
| DF | size | `2803867223/992` |

[event_coverage.json](event_coverage.json)に全64本の次数coverageを保存。新cohortの合計：

| 方式 | order0 | order2 | order4 |
|---|---:|---:|---:|
| P | 968 | 56 | 0 |
| DF | 967 | 57 | 0 |

order4は両方式で未観測。coverageを理由に再抽出・forced event混入・top-upしていない。
未観測tail寄与を0とはせず、`population_mean_upper=null`、`population_tail_bound=null`。
標本SEは無条件母平均やrare-tailを非漸近に保証しない。追加標本数による所定精度達成も事前保証していない。

## 実行・原値保全・終了

compilerはQiskit1.3.0、basis rz/sx/x/cx、level1、seed17、backend/coupling/layout/routing指定なし。
compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` は全128件で一致。
専用runtimeはnumpy1.26.4/scipy1.14.1/qiskit-nature0.7.2/OpenFermion1.6.1/Python3.11.0rc1。
native taskはspawnしcomponent oracleをprocess内cache、compilerはspawn/max_tasks_per_child=1。
全128 compileに異なる128 PID、各call1、例外0、全worker join済み、process exit0。
実worker数63は起動時のaffinity/loadから選び、共有環境/他利用者・libraryを変更していない。
wall `69.227148`秒。完了後の追加sampling/compileや科学retryなし。

```json
{
  "sampling_calls": 64,
  "trajectories": 64,
  "event_positions": 2048,
  "native_Full_PASS": 64,
  "reference_half_exponentials": 1,
  "saved_half_matrix_evaluations": 1,
  "compile_completed": 128,
  "new_eigendecompositions": 0,
  "new_Givens": 0,
  "new_E_R_Empty": 0,
  "old_prices_recompiled": 0,
  "old_pilot_pooled": 0,
  "new_signal_B_N_G": 0,
  "automatic_retries": 0
}
```

[native_evidence.tar.xz](native_evidence.tar.xz)は全704 memberの可逆保存：新native64 QPY、input128/output128 QPY、
価格metrics128、compile worker128、native worker64、trajectory64。
QPYは元gzip bytesのままで、[archive_members.json](archive_members.json)の各hashに対応する。
元のlocal rawも保持し、公開archiveから元bytesを復元できる。旧archiveへ上書きしていない。
[verification.json](verification.json)では704 member・320 gzip QPY・SOURCE/input/旧STOP・集計とCSVの転記を検証する。
検証でQPYの回路再build/再compileは行わず、raw値を修正していない。

Stage 2 finite bias両軸未解像、finite-bit実効law平均、実compiled/全lawρ、母平均tail・N/Gは未解決のまま。
今回の価格確認から全roundのRPE/energy総資源、一般的方式優位・最適性を認定しない。
次の担当はユーザー：保存bias/Bと今回の価格統計を使うStage 4の条件付きN/G会計を行うか、その仮定・精度範囲を判断する。
Codexは別の明示指示まで新N/G・高精度化・追加標本・新条件・主要claim採択・論文改訂を開始しない。
**限定価格確認は完了。公開・remote一致確認後mandatory STOP。**

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60515a33ce6231e9ff4ffdb40124ed845373351c/docs/research/h4_p3_poststage3_cost_and_stage4_design_20261102.md
