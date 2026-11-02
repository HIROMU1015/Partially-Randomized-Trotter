# Rectangular Givens候補A：PS48の科学実行結果

**固定48-wrapperの科学的正しさは全件PASS。ただし候補Aの一律適用による費用削減は支持されなかった。**
index21 `IIZI` とindex25 `IZZI` では各8件すべてのRZ・CX・depthが減った一方、index20 `IIIZ` とindex24 `IZIZ` はRZ・CXが増加した。
48件の等重み平均はRZ +7.8571%、CX +1.9802%、total depth −0.7124%。これは選択されたsubsetの記述値で、全F2の確率重み付き期待費用ではない。
一回の承認・grantで正常完了し、修正・retry・resumeは0件。科学実行は終了し、追加実験へ進まない。

## 条件・SOURCE・実行範囲

SOURCEは準備commit **`770e53addda63a03fa8c6c5d5168f7062067c572`** のまま。
[準備・固定PLAN](pr_rectangular_givens_preparation_20261024.md)、[候補A設計](pr_gaussian_singleton_synthesis_design_20261024.md)を再利用した。
baseline結果は `dc7378ab34015287215bcf4ce52147e46d94ecd9`、測定SOURCEは `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。

4 mode・2粒子・occupied(0,1)、L_D=1、残差2fragmentの固定合成Hamiltonian。
分子geometry／AO basis／分子DF rankは非該当、rank探索なし。
T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0。ここでF1は残差fragment F1。
PS index20～25、partner0 `P:collect:IIII`／2 `P:collect:IXXI`、両operator順序、X/Yを用いた。
support1/2各24-wrapper、計48、24 evolution、候補Gは6種類。

Qiskit1.3.0／Nature0.7.2、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree=1、qubits_initially_zero=False。
F2の位相保持compiler接続、full wrapperのphysical状態準備・controlled evolution・readoutを維持した。
変更はsingletonのGaussian recipeだけで、Hamiltonian・選択operator・符号・順序・sampling lawは保持した。同frame共有runへの展開は今回行っていない。

preflightでSOURCE113ファイル、保存B4ファイル、baseline192ファイル、依存50件とNature helper2ファイルを照合した。
新しいPS48専用one-shotを消費し、保存48 QPYの中心部から同じ角度 **`0x1.4534303f567e2p-6`** を取得した。
係数符号・元support・順序・identity/scalar位相のbindingも通過。λからの角度再推定はしていない。
元90/128桁独立referenceで照合済みの48 targetを再利用し、高精度referenceの新規評価は0件。

## 科学gateの結果

| 検査 | 実績・最大残差 | 判定・許容差 |
|---|---:|---|
| 6基底のprojector `W Π_T W† = V V†` | 8.04920e-16 | PASS、1e-10 |
| 6基底のactual adjoint | 5.55112e-17 | PASS、1e-10 |
| 6基底の全16次元unitarity | 3.33067e-16 | PASS、1e-10 |
| raw full-wrapper operator、48件 | 7.77195e-16 | PASS、1e-10 |
| raw X/Y信号、48件 | 1.44329e-15 | PASS、2e-10 |
| compiled full-wrapper operator、48件 | 1.99349e-14 | PASS、5e-10 |
| compiled X/Y信号、48件 | 6.24500e-15 | PASS、5e-10 |
| raw／compiled unitarity、各48件 | 1.77636e-15／3.55271e-15 | PASS、1e-10 |

rtol=0、phase fittingなし。controlを含む32×32 wrapperを保存独立targetへ直接照合したので、systemの全16 Fock状態を含む。
固定2粒子sectorや一つの初期stateだけのoperator検査ではない。global／control枝のrelative phase、負係数、元のidentity/scalar位相を含めて通過した。
6基底のprojector・adjoint・unitarityと、support1/2各24件のraw／compiled operator・signalはすべてPASS。
件数は作業単位が異なるため、単一の「科学call総数」には合算しない。

新buildは48 full-wrapper、compile要求48、実transpile36、同一full IRのalias12。
全logical caseを保存referenceへ検査し、cacheで科学gateを省略していない。
baselineのbuild／compile、law／Hamiltonian再生成、新しいsampling、γ／N計算は0件。

## Compiled資源：support別と全48件

以下はfull wrapperの**case等重み算術平均**。各行はbaseline → 候補A（変化率）。
shot期待値、全F2のlaw重み付き期待値、N加重費用を意味しない。

| subset | RZ count | CX count | total depth | RZ depth | CX depth | size |
|---|---:|---:|---:|---:|---:|---:|
| support1、24件 | 67.5000 → 74.3333 (+10.1235%) | 30.6667 → 32.0000 (+4.3478%) | 84.2500 → 92.2500 (+9.4955%) | 39.5000 → 43.5000 (+10.1266%) | 27.6667 → 28.5000 (+3.0120%) | 130.5000 → 144.3333 (+10.6003%) |
| support2、24件 | 84.1667 → 89.2500 (+6.0396%) | 36.6667 → 36.6667 (0%) | 102.9167 → 93.5833 (−9.0688%) | 46.8333 → 43.8333 (−6.4057%) | 33.1667 → 30.1667 (−9.0452%) | 164.5000 → 173.7500 (+5.6231%) |
| 全48件 | 75.8333 → 81.7917 (+7.8571%) | 33.6667 → 34.3333 (+1.9802%) | 93.5833 → 92.9167 (−0.7124%) | 43.1667 → 43.6667 (+1.1583%) | 30.4167 → 29.3333 (−3.5616%) | 147.5000 → 159.0417 (+7.8249%) |

quantum-only depthを含む全8指標・各caseの整数費用・差分と、各groupのexact有理平均は
[CASE_RESULTS.csv](../../artifacts/pr_rectangular_givens/2026-10-24/CASE_RESULTS.csv)、
[RESULT_SUMMARY.json](../../artifacts/pr_rectangular_givens/2026-10-24/RESULT_SUMMARY.json)を正とする。
measurement countは全caseで1のまま。
RZは20件で改善・28件で悪化、CXは改善16・同値16・悪化16、total depthは改善26・同値10・悪化12だった。

## 改善・悪化したcaseと特徴

各atomは固定partner2種類・両順序・X/Yの8件。平均差は候補−baseline。

| index／元support | word | 平均ΔRZ | 平均ΔCX | 平均Δdepth | 観測 |
|---|---|---:|---:|---:|---|
| 20／(3) | IIIZ | +27 | +8 | +31 | 全8件で3指標とも悪化。RZ +56.25% |
| 21／(2) | IIZI | −7.25 | −4 | −6 | 全8件で3指標とも改善。RZ −8.9231% |
| 22／(2,3) | IIZZ | +8.5 | 0 | −4 | RZは全8件増加、depthは6件減少・2件増加。指標間tradeoff |
| 23／(1) | IZII | +0.75 | 0 | −1 | identity相手はΔRZ +4、Pauli相手は−3または−2。境界／順序に依存 |
| 24／(1,3) | IZIZ | +17.25 | +4 | 0 | RZ/CXは全8件増加。Pauli相手のdepthは順序で−6／+6 |
| 25／(1,2) | IZZI | −10.5 | −4 | −24 | 全8件で3指標とも改善。RZ −11.7978%、depth −21.7687% |

例：`PS_025_002_X` はRZ 93→81、CX 40→36、depth 121→89。
`IIIZ`は元support(3)からcanonical T=(0)へ移す候補で、選択部分空間だけの合成が常に安いわけではない。
この増加は、設計時に挙げたTへのmode移動と、元frameの特殊構造を使えるbaselineの有利さに整合する。
ただし、mode移動・Givens分解・compiler cancellationそれぞれの因果的な寄与は分離していない。
index23/24のpartner／order依存も、basis単独の費用だけでfull-wrapper資源を予測できないことを示す記述的結果である。

## 追加の古典計算費用

pipeline wall **21.187秒**、起動・終了を含むwhole-job wall **21.93秒**。
whole-job CPUはuser 528.19秒＋system 15.28秒＝**543.47秒**。
buildは24 worker、compileは32 worker、job内thread1。実際の32 CPU affinity、約56.6 GBのavailable RAMと既存RSS記録から選んだ。
worker／CPU／wall／RAMの人工的な小さい上限や共有環境変更は行っていない。

build/check journalのCPU合計151.764秒、compile/check journalは273.135秒。
実transpile部分のwall合計は1.876秒で、compile/check全体の費用と区別する。
残りのwhole-job CPUにはworker起動・import・検査・serialization等が含まれ、journal合計だけで全費用を表さない。
並列jobのwall合計もpipeline wallへ単純加算しない。
最大process RSSは368,380 KiB（約360 MiB）。これは全worker合計のpeak RAMではなく、aggregate peak RAMは未計測。
baselineと同条件で古典CPUを再測定していないので、候補合成の古典的speedupは主張しない。

## 技術的判断・限界・次の担当

**候補Aを全singletonへ一律に置換して全F2へ展開する作業は、現時点では優先しない。**
全48件平均でRZ/CXが増え、support2にもRZとdepthのtradeoffがあるため、一律の資源削減経路としての根拠が不足している。
一方、index21/25の固定対照では正しさを保ちつつ3指標が一貫して改善し、選択的な使用を検討する技術的余地はある。

次は**ユーザー／GPT**が、一律展開を見送るか、元support・partner／orderを含む適用規則の設計を別に承認するか判断する。
今回の結果を見て選ぶ規則は事後選択であり、一般化や全F2の改善を認定しない。
新規則の実装・全F2期待費用比較には、条件・対象caseを別に固定した承認が必要。
今回、その実装・計算・新しい重要科学レビューは開始していない。
γ・η・α・intervalの未資格状態は変わらず、認証N・N加重比較・Physical同精度比較・一般的優位・独立新規性は未確立。

## 成果物・保全・STOP

- [結果と全基底残差・group統計](../../artifacts/pr_rectangular_givens/2026-10-24/RESULT_SUMMARY.json)
- [全48 caseの費用・残差](../../artifacts/pr_rectangular_givens/2026-10-24/CASE_RESULTS.csv)
- [SOURCE／入力／raw provenance](../../artifacts/pr_rectangular_givens/2026-10-24/PROVENANCE.json)
- [保存結果だけの集計SOURCE](../../artifacts/pr_rectangular_givens/2026-10-24/summarize_saved_ps48.py)
- [Mandatory STOP](../../artifacts/pr_rectangular_givens/2026-10-24/STOP.json)

科学SOURCEは変更せず、結果資料と保存結果の集計utilityだけを新規追加した。
48 caseのbaseline費用は、実行で使った保存compiled JSONと公開F2 CSVの双方に一致することを確認した。
新rawには候補G／raw・compiled QPY、再利用target、全case残差、attempt journal、角度、費用、STOP、hash inventoryを保持する。
承認原文・grant・marker・内部locatorは非公開で保護する。
旧SOURCE・raw・grant・ledger・marker・STOPと元workspaceを保持した。
公開結果は固定commitに結びついたlocal実行証拠で、CIや外部独立再現ではない。

**公開・保全確認後mandatory STOP。次の担当はユーザー／GPT。**
全F2展開、別候補、新fixture、追加科学計算、γ保証・N計算へ自動移行しない。
