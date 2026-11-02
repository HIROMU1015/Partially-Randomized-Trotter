# 研究C・C4：B2角度結合baselineの限定科学実行・保存Rustiq比較

2026-10-24 JST。状態：**C4_B2_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP**。一回の承認済みrunで正常完了し、全登録gateがPASS。retry／resumeは0。

**連続する同一Pauli controlled rotationの単純角度結合B2は、全4case・X/Y両axisの全8 compiled資源指標で、保存Rustiq R1と一致した。** このfixtureで先に観測された削減は単純結合だけで再現でき、B2を超えるR1の追加compiled資源差は残らなかった。Rustiq内部の一意な機構同定、一般的な新規性・優位性、独立研究の正式終了を意味しない。

## 1. 固定対象と追跡先

Repository：`HIROMU1015/Partially-Randomized-Trotter`。専用branch：`codex/research-c-angle-fusion-b2-execution-20261024`。

- 科学実行SOURCE／公開準備commit：`0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162`。実行中・実行後にSOURCEを変更していない。
- B2実装commit：`c7b6bc17f73d7b1dd2c9221272b4a5d858b5c472`。比較設計正本：[`b7b70f6a`の設計書][DESIGN]。
- 固定契約：[`C4_B2_SAVED_COMPARISON_V1`][CONTRACT]。SHA256：`f03839377158a44477a898fdc35af7d6a3b96bf953e86275915bf97324df035f`。
- 保存対照：[`f95148e3`の旧結果][OLD]。旧science SOURCEは`b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9`。
- 新packet：[`artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`][PACKET]。同所の[approval][APPROVAL]、[実行前preflight][PREFLIGHT]、[provenance][PROV]、[全call集計][SUMMARY]、[attempt ledger][LEDGER]、[STOP][STOP]を参照。

物理対象は保存F1の**4 mode・2粒子、occupied(0,1)**、density-square合成Hamiltonian、split `L_D=1`・残差2fragment。分子geometry／AO basis／分子DF rankは非該当。時間は`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`の一点。SS M2/M3/M4・PP M0、設計済み10領域と8 full measured Hadamard wrapperだけを対象とした。

CompilerはQiskit1.3.0、native `rz,sx,x,cx`、opt1・seed17・approximation_degree1.0・qubits_initially_zero=false。backend／target／coupling／layout／routing／scheduling指定なし。元Hamiltonian・sampling law・finite target、ordered events、元basis/run/singleton、相対位相、prep/readoutを保持した。

## 2. 承認とpreflight

今回のユーザーが固定公開commit・契約SHA・B2の4case／10領域／8wrapperを明示承認した。旧契約の`scientific_execution_authorized=false`は準備時点の記録のまま保全し、今回の承認だけを新しい`USER_EXPLICIT_C4_B2_EXECUTION`・SOURCE・契約hash・fresh output・run IDへbindした。旧grantと技術的retry許可は流用していない。

最初の科学call前にclean HEAD、41 SOURCE hash、7保存入力とlock、旧packetのmanifest＋379登録ファイル、保存reference14 bundleのschema・文字列grammar、元4case/10region、model/profile/tolerance、Python3.11.9・50 package version・Qiskit API/native binaryを技術照合しPASSした。保存入力／SOURCE集合identityと全hashは[preflight][PREFLIGHT]と[provenance][PROV]に保存した。独立worktree・process・cache/temp/outputを使用し、共有venv・package・他jobは変更していない。

承認後のmotif復元では元event、atom、angle hex、occurrence、run/basis/singleton、phase、registerを保存reportと照合した。SOURCEの再hashは各stage barrierでも行った。旧と新はSOURCE commitこそ異なるが、共通35 SOURCE bytesと科学条件が一致し、B2のopt-in差分を新packetの`source_diff.patch`で追跡できる。

## 3. 科学的正しさ

B2は元planを保持したemit時の局所結合であり、control展開前の最大連続同一generatorだけに`math.fsum`を適用した。singletonの角は元floatのまま、signを二重適用せず、異Taylor orderの角を代表値×回数にしない。並替え、basis/run/macro/product/identity境界越え、modulo、phase無視を行わない。既存controlled emitter、basis actual inverse、D融合、scalar/Taylor/product/identity位相、actual adjointとfull wrapperを再利用した。

保存128桁referenceの十進real/imag文字列を128桁contextへ直接復元してから既存binary64変換を使用した。新しい独立referenceと90/128再比較は0で、旧精度資格・hashを再利用した。局所では同じactual Operatorを保存独立targetと保存結合前literalの両方へ絶対比較した。fullはcontrol0=Iを含む完全な`diag(I,U)`、adjointはactual circuit inverse、compiledはterminal measureだけを外した完全なprep/evolution/readout operatorを検査した。

| 保存gate群 | 比較数 | 最大max-entry | 最大Frobenius | 最大spectral | 結果 |
|---|---:|---:|---:|---:|---|
| 局所：保存独立reference | 10 | 2.22045e-16 | 9.93014e-16 | 2.22045e-16 | PASS |
| 局所：保存結合前literal | 10 | 1.11036e-16 | 6.28075e-16 | 1.11036e-16 | PASS |
| full controlled | 4 | 1.33229e-15 | 3.88e-15 | 1.33641e-15 | PASS |
| actual adjoint | 4 | 1.11031e-15 | 3.93663e-15 | 1.11066e-15 | PASS |
| compiled full wrapper | 8 | 1.0872e-14 | 4.70305e-14 | 1.45875e-14 | PASS |
| unitarity | 14 | 2.55351e-15 | 7.71003e-15 | 2.55963e-15 | PASS |

行列・unitarityの判定は絶対max-entry **1e-10**、rtol0。Frobenius/spectralは記録指標であり、別の判定閾値へ置き換えていない。Raw complex X/Y 4pairの最大誤差は`1.55435e-15`、compiled4pairは`4.92067e-15`、固定tol **2e-10**を通過した。X/Y各8回を実評価しており、一方のaxisで代用していない。位相fit・global-phase同値だけの判定・事後的許容差緩和は0。

全数値は新しいoperator再評価ではなく保存reportから整理した。[gate_summary.json][GATES]および個々の`reports/`に元の精度で保存している。原本のQPY/NPY/target・report・journal・STOPは報告作業で変更していない。

## 4. B2と保存B0/B1/R1のcompiled資源

全4case・両axisの正しさがPASSした後に既存`circuit_metrics`で取得した値。表中は**保存B0/B1→新B2（保存R1と同値）**。各caseのX/Yは全8指標が一致しているため一行に表示し、8record全体の値は[paired_metrics.csv][CSV]と[paired comparison][PAIR]に保持する。

| case | axis | RZ | CX | RZ depth | CX depth | total depth | quantum-only depth | size | measure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SS_motif_2` | X/Y共通 | 139→135 | 66→62 | 74→70 | 58→54 | 172→164 | 171→163 | 286→278 | 1→1 |
| `SS_motif_3` | X/Y共通 | 115→113 | 51→49 | 63→61 | 45→43 | 136→132 | 135→131 | 223→219 | 1→1 |
| `SS_motif_4` | X/Y共通 | 87→83 | 48→44 | 47→43 | 40→36 | 108→100 | 107→99 | 176→168 | 1→1 |
| `PP_motif_0` | X/Y共通 | 59→59 | 36→36 | 39→39 | 34→34 | 78→78 | 77→77 | 108→108 | 1→1 |

B0とB1は全対応recordで同値。B2は保存R1と**全8wrapper・全8指標で完全一致**し、整数差は全て0。SSの3caseではB1よりRZ/CX/depth等が小さく、PP M0は全route同値だった。異caseの値を集めてlaw期待値や新たな重み付きwinnerへ変換していない。認証shot、全round、T countは計算していない。

これは単純角結合が観測削減を再現するという**十分性**の証拠であり、R1が内部で同じ操作だけを行ったという証明ではない。今回のfixture/profileでは、B2を超えるR1のcompiled費用差は観測されなかった。Rustiqの独立した方法論的新規性や一般的優位はこの結果から認定しない。

## 5. Raw IRと通常compilerの切り分け

契約内の保存B1 raw QPY decode8件と、新B2の既存raw8件に対するmetric診断。旧回路のbuild/compile/simulationはしていない。表はB1→B2であり、Gaussian compositeを含むraw表現の値をnative compiled資源と混同しない。

| case | axis | raw RZ | raw CX | raw total depth | raw size |
|---|---|---:|---:|---:|---:|
| `SS_motif_2` | X | 40→36 | 8→4 | 54→46 | 85→77 |
| `SS_motif_2` | Y | 40→36 | 8→4 | 55→47 | 86→78 |
| `SS_motif_3` | X | 30→28 | 6→4 | 41→39 | 72→68 |
| `SS_motif_3` | Y | 30→28 | 6→4 | 42→40 | 73→69 |
| `SS_motif_4` | X | 24→20 | 8→4 | 40→32 | 58→50 |
| `SS_motif_4` | Y | 24→20 | 8→4 | 41→33 | 59→51 |
| `PP_motif_0` | X | 8→8 | 12→12 | 36→36 | 46→46 |
| `PP_motif_0` | Y | 8→8 | 12→12 | 37→37 | 47→47 |

SSでは融合ledgerに対応してraw primitiveが減り、その差は固定opt1 compile後にも残った。従ってこのprofileがB1でB2と同等の費用削減を全て実現済みだったとはいえない。PPでは融合なしの負対照としてraw/compiledとも同値。旧pass別traceがないため、Qiskitのどのpassが何をしたかやRustiqの内部処理は同定していない。

## 6. 実際のcall・資源・古典費用

| 科学request | 実績／契約 |
|---|---:|
| saved motif restore／対象law arm | 4／2 |
| 保存128桁bundle復元／保存literal NPY load | 14／10 |
| target matrix変換／signal変換 | 22／4 |
| B2 local／full build | 10／4 |
| actual adjoint／full measured wrapper | 4／8 |
| Operator materialization | 26 |
| 絶対operator比較／unitarity | 36／14 |
| raw／compiled Statevector signal | 8／8 |
| X/Y complex pair gate | 8 |
| cold／warm request | 8／8 |
| 実transpile／warm追加transpile | 8／0 |
| correctness後compiled metric | 8 |
| 保存B1 raw QPY decode／raw metric | 8／16 |
| 新独立reference／precision再比較 | 0／0 |
| B0/B1/R1再実行／Rustiq／R2 | 全て0 |
| sampling／law生成／認証shot／全round／T count | 全て0 |
| scientific attempt／retry／resume | 1／0／0 |

全journalは356行で契約4096行以内、completed group数は契約と一致。科学コードの追加実行やscope変更はなかった。UI上の中断時には科学processが既にexit0で終了しており、その後は保存reportの整理・公開だけを続行した。科学runのresume／retryではない。

CPU affinity32、起動時load約1.02、空きRAM約55 GB、CPU/memory pressureはほぼ0。最大stage fanout10に対応するworker capacity10、Rayon3・BLAS/OpenMP/Numba/Qiskit各1で、既存process-poolとstage barrierを使った。GPUはdriverに接続できず、既存CPU経路を選択した。人工CPU/RAM/GPU/worker/wall capは設けていない。

実行wall `19.013127`秒、parent CPU `4.308602`秒、children CPU `52.665506`秒。journalで観測した最大process peak RSSは**367924 KiB**で、同時aggregate RAMではない。plan／fold／emitter、whole builder、cold lowering/transpile、warm lookupの各task時間は[classical_costs.json][TIME]と元journal/traceに保存した。fold/emitterはbuilderの内数なので加算しない。R1旧単発timingとはworker/load/実施時期が揃わず、古典費用の速度優位は未判定である。

## 7. 証拠・限界・次の判断

`summary.json`／`STOP.json`は実行が書いた原本を保持する。新packetにはapproval・ユーザー承認本文・実行前preflight・環境・SOURCE/input/profile hash、元event/angle/plan/fusion ledger、byte同一の保存reference14件、QPY/NPY、gate、資源比較、raw IR、時間/RSS、attempt journal/ledgerを保存する。`OUTPUT_HASH_MANIFEST.json`は新packet全ファイルのhashを固定し、自分自身は除外する。SOURCE commitを参照するので、結果commit自身への循環参照はない。

保存report整理用の`artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/summarize_saved_results.py`は既存C4 summarizerのJSON/journal方式を参照したstandard-library utilityで、QPY/NPYのload・build・compile・Operator/Statevector・reference評価を行わない。契約に合わせてcase/axis別の整数比較だけを扱い、旧helperの全case費用合計は使わない。再照合は次のcommandで行える。

```bash
/usr/bin/python3 artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/summarize_saved_results.py
```

この証拠はcommitされるローカルrunであり、CI再生成・外部独立再現ではない。固定4caseと一点δ以外、分子系、他compiler/coupling、law重み付き期待費用、認証N、全round総資源へ外挿しない。理論的な総コスト優位や一般的no-goを意味しない。

設計正本の「全8wrapperの全指標でB2=R1」に該当する。**提案**は、既知の角度結合を実装改善として保持し、C4を独立研究として一旦区切るかをユーザー・研究C・統括が正式判断すること。この再現結果の説明のためにR2を追加する必要はなく、R2・追加case等は実行しない。独立研究の終了は今回自動採択していない。

元workspace、旧SOURCE commit・raw・結果・grant・ledger・marker・STOPは保全する。共通概要・validation status/manifest・索引と当日noteには今回の新しい結果を追記し、旧結果statusは変更しない。公開commit・本文・必要な証拠hashを照合した時点で**mandatory STOP**。次の担当はユーザー／研究C／統括、Codexは別途指示前に新科学へ進まない。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7b70f6a5e3ee4ae4a55b215b49de0390e5732e5/docs/research/research_c_rustiq_angle_fusion_baseline_design_20261024.md
[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162/docs/research/research_c_angle_fusion_b2_contract_20261024.json
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f95148e3f176efc6b3fd81a93d043ccd20255343/docs/research/research_c_rustiq_execution_20261024.md
[PACKET]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/
[APPROVAL]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/approval.json
[PREFLIGHT]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/preflight.json
[PROV]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/provenance.json
[SUMMARY]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/summary.json
[LEDGER]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/attempt_ledger.json
[STOP]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/STOP.json
[GATES]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/gate_summary.json
[CSV]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/paired_metrics.csv
[PAIR]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/comparison_summary.json
[TIME]: ../../artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/classical_costs.json
