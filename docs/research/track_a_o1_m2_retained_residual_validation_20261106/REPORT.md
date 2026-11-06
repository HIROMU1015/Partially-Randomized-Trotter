# O1 M2：微小係数保持修正後の実H4完全残差検証

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**微小実係数の全消失は解消したが、総合状態はNOT_QUALIFIED。** 固定SOURCEの残差判定は保存binary64入力由来のanti-Hermitian成分を検出した。原raw・閾値・SOURCEを変更せず、同じ入力のoperator-only原因診断まででmandatory STOPした。

入口は[assessment.json](../../../artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/assessment.json)。[原result.json](../../../artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/result.json)のNOT_QUALIFIEDを保持しており、診断でPASSに書き換えていない。

## SOURCE・再利用・保存入力

実行SOURCEは **`cee130ad8c24868a88ad09778f664aa01ae50154`**。新branchは`codex/track-a-o1-m2-retained-residual-validation-20261106`。rawの`source.base=a21be7…`は前段の履歴であり、今回の実行SOURCEを表すHEADではない。本報告とassessmentでその区別を明示する。

[新runner](../../../scripts/validate_o1_m2_retained_residual.py)、局所unit-monomial JW mapper、既存adapter、c1_stage1のspin lift/κ/scale、既存の独立sparse CAR・raw J/U density core・Pauli label bit actionを再利用した。新runではこれら固定SOURCEを変更していない。

保存core/入力はcommit`aced1647b81e8d28091058cdc7b6e3e1ee54896a`にある同じupdate005と元H：

| 役割 | 元path | SHA256 |
|---|---|---|
| 全8frameのJ/U | `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz` | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| 元h/g/scalar | `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz` | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |

保存core生成SOURCE`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`と今回の実行SOURCEは別である。入力bytesを事前・loader内で照合した。linear H4、間隔1Å、STO-3G、中性singlet、4空間/8 spin modes、full256とNalpha=Nbeta2のsector36、途中射影なし。base＋全8frameをcoreとし、旧LD2 tailは使わない。

Python3.12.3、NumPy2.5.4、SciPy1.18.1、OpenFermion1.8.1、Qiskit2.6.2の既存専用runtimeをread-only利用。global EQ_TOLERANCEは1e-8のまま、共有環境は不変更。CPU/RAM/workerの独自上限は設けていない。[timing](../../../artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/timing.txt)はwall7.25秒、user19.08秒、sys0.32秒。原runner終了code1、[console](../../../artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/console.log)は空。例外ではなく固定判定の不適合である。

## 事前許容値とoperator結果

準備SOURCEで固定したL33288、guard8、binary64 u=2^-53、operand scaleから評価した。総H判定値は **8.011293947336095e-9**、残差自身はunit floorなしの **2.0490161611727855e-18**、rtol0。残差scaleは6.930386640756261e-8。残差最大entry2.2651962304775996e-9に対して分解能条件`tau_R < .01 max|R|`は通過し、ゼロmapper・反符号を区別できる。人工atolの転用・結果後の閾値変更はしていない。

| 比較 | full256最大entry差 | sector36最大entry差 | 固定判定 |
|---|---:|---:|---|
| raw density全core＋R vs H CAR | 1.0658141036401503e-14 | 2.1094237467877974e-15 | PASS |
| density core vs normal-ordered core | 6.661338147750939e-15 | 2.220446049250313e-15 | PASS |
| physical Pauli vs CAR | 5.773159728050814e-15 | 1.887379141862766e-15 | PASS |
| core Pauli vs CAR | 2.6645352591003757e-15 | 1.7763568394002505e-15 | PASS |
| **R Pauli vs R CAR** | **8.391724815037804e-17** | **8.272462576064399e-17** | **FAIL** |
| **R CARのHermiticity defect** | **1.6783449630075609e-16** | **1.6544925152128798e-16** | **FAIL** |
| R mappedのHermiticity defect | 0 | 0 | PASS |
| Pauli core＋R vs H CAR | 6.661338147750939e-15 | 2.1094237467877974e-15 | PASS |

総H側のPASSで残差自身のFAILを覆い隠さない。微小残差の下流加算scaleだけから作ったtau_Rでは、元H/coreの大きい数値からの差や元入力のadjoint非対称性を覆えないことが判明した。これは許容式の限界であり、今runで値を緩める理由にしない。上記基準は工程scaleの数値検査であってrigorous enclosureではない。

## 微小項・scalar・一体補正・符号・sector

physical/core/Rはそれぞれ**360 nonidentity Pauli項**を保持した。前段R=0項という全消失は解消した。Rの電子identityは **4.6249802598818945e-11**、独立CAR trace/256と差0。核scalarはphysical/coreが2.29310124732、Rが0で、一度だけ含まれる。

原gから独立要素和で求めたκとの差は1.232595164407831e-32、`R_h−.5 contract(R_g)`は1.0408340855860843e-16、`R_g−(g_original−g_fit)`は0。R_h最大8.000222706527893e-11、R_g最大7.998633243176693e-10。微小な一体項をゼロへ置換していない。全mapped operatorのsector外へのentryは0。残差gの符号は旧保存field`g_fit−g_original`の逆である。

physical/coreのidentityは核scalar込みtotalで、それぞれ−0.3314778134168105 / −0.3314778134630603。電子寄与は既知核scalarを引いて区別する。Rは核scalar0なので、上記identityがそのまま電子寄与である。

## 同一入力のoperator-only原因診断

不一致説明のため、[診断script](../../../scripts/diagnose_o1_m2_residual_hermiticity.py)で同じ保存入力・同じcore・同じunit JW/CAR参照を使った。固定mapperや原runnerを再定義せず、raw・閾値・入力を変更しない診断である。新fit・別rank・別物理条件・修正版での追試ではない。

[hermiticity_diagnosis.json](../../../artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/hermiticity_diagnosis.json)には、既存実Pauli係数化を通さないcomplex JWも比較した：

| 診断 | full256最大entry差 |
|---|---:|
| complex JW（虚部を除去しない）vs raw R CAR | **8.271806125530277e-25** |
| fixed real mapper vs Hermitian part `(R+R†)/2` | **4.1359030627651384e-25** |
| raw Rのanti-Hermitian part `(R−R†)/2` | **8.391724815037804e-17** |

complex Pauli係数に非zero虚部が368個、最大3.642919299551295e-17、虚部L1は4.948974519785009e-16あった。既存`PauliHamiltonian.real_coefficient`は虚部の許容値1e-12以下を実部へ写すため、raw Rのanti-Hermitian部分が除去される。これは今回のtiny real coefficient切捨てとは別である。

入力tensorにも、physical h−hT最大1.1102230246251565e-16、core h−hT最大4.781127026519142e-17、R h−hT最大1.3877787807814457e-16がある。core/Rのg adjoint差は5.551115123125783e-17。binary64で保存・構成された大きい元H/core由来の非対称性であり、意図する物理Hamiltonianの非Hermiticityを示す主張ではない。

この診断は「実mapperがHermitian部分を極めてよく再現する」という原因説明であり、raw残差の完全一致PASSへ変更するものではない。`(R+R†)/2`を新しい物理残差として採択したり、入力をsymmetrizeして原結果を上書きしたりしていない。

原result SHA256 `15baeb87b13f475ed09d86899e354c189e3fe6061a97cb3d4447bee635544af1`、診断result `145a77b1bae2902e37ac62a66dde0dbfb0109e0ba49dc58698c75c1159ac3dd1`、診断SOURCE `037c8f355d2a30c9431d2df1f86d90d93612925c724f0dd1fc06051e269a1e5f`を保存した。

## 再現入口・未解決・STOP

新runnerは以下で実行した（出力は既に存在し、runnerは上書きを拒否する）：

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/o1-m2-retained-run-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m2-retained-run-cache-20261106/mpl \
/tmp/track-a-o1-m2-runtime-20261106/bin/python scripts/validate_o1_m2_retained_residual.py \
  --core ../track-a-o1-m2-h4-operator-correctness-20261106/artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/update005_candidate.npz \
  --input ../track-a-o1-m2-h4-operator-correctness-20261106/artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/input.npz \
  --output artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/result.json
```

診断は同じcore/input引数を`diagnose_o1_m2_residual_hermiticity.py`へ渡し、出力だけ`hermiticity_diagnosis.json`とした。新科学処理は固定runner1回と同一入力operator原因診断1回。元実験の再実行・SOURCE修正・閾値変更・別runtime追試はない。準備時22人工テストを実H4 PASSの代用にしていない。

未解決は、元SOURCE精度とHermiticity規約を含めた残差の科学的定義・判定条件である。**次の担当はユーザー**：保存binary64のanti-Hermitian丸め成分をどう扱うか、明示Hermitian規約と入力精度由来の誤差予算を採択する。**Codex**は必要なら、その決定に基づく新SOURCE・事前判定条件を準備し、別途許可されたoperator-only検証を行う。単なるposthoc許容値拡大で今回をPASSへ変更しない。

今回はraw・診断・assessment・本報告を独立branchへ公開し、remote SHA・全変更file内容・cleanを確認して**mandatory STOP**。state解析、PF/RTE信号、回路build/compile、sampling、shot・総費用、新fit/rank探索には進まない。主agentの検査であり、独立研究レビューではない。旧SOURCE・raw・grant・marker・STOPと他作業は保全した。
