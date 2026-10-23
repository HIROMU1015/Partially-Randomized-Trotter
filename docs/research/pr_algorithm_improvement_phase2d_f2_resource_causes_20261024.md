# 限定F2：PPが低コストになった回路上の理由

**結論：PPでは残差イベントのGaussian基底変換・復帰が不要で、局所H/Sと共通のcontrolled Pauli回転を使える。K=0・r=2では1軌道で選ぶ成分は2個であり、Pauli展開の成分数増加が、そのまま1-shot回路長の増加にはならない。** これがSOURCEから特定できる主要な費用差の機構で、保存結果とも整合する。ただし、Gaussian変換・共有・確率変更・compilerのそれぞれが期待費用差の何割を生んだかは未分離である。

調査は固定SOURCEの静的読解と保存記録の参照・case照合のみ。新しい集計、統計解析、順位計算、回路build／compile、数値再解析、科学SOURCE変更は行っていない。以下の「保存事実」「SOURCE上の機構」「解釈・未測定」を区別する。

## 1. 対象と既存結果

結果commitは [`dc7378ab34015287215bcf4ce52147e46d94ecd9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/dc7378ab34015287215bcf4ce52147e46d94ecd9)、測定SOURCEは `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。
4 mode・2粒子・occupied(0,1)、L_D=1、残差density-square 2fragmentの固定合成Hamiltonian。geometry／AO basis／分子DF rankは非該当、合成fragmentは2個に固定しrank探索なし。T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0、faithful identity tail。
Qiskit1.3.0／Qiskit Nature0.7.2、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree=1、qubits_initially_zero=False。位相保持修正後のfull-wrapper結果だけを参照する。

以下は[既存実行報告](pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)の転記で、再集計ではない。X/Yそれぞれの1-shot期待費用で、prep・controlled evolution・readout・measure込み。

| arm（F0／F1） | 保存canonical L | 期待RZ | 期待CX | total depth |
|---|---:|---:|---:|---:|
| SS（frame／frame） | 14 | 70.825335 | 32.696703 | 86.953886 |
| SP（frame／Pauli） | 58 | 45.161578 | 26.354128 | 60.338726 |
| PS（Pauli／frame） | 26 | 53.950984 | 27.813111 | 69.415121 |
| PP（Pauli／Pauli、fragment間collect済み） | 51 | 35.445140 | 23.528787 | 49.627289 |

保存事実は全7指標の **PP < SP < PS < SS** と全4armの固定許容差内PASS。順位は今回再計算していない。精密値・残差は[RESULT_SUMMARY](../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json)、Lは[Bの保存要約](pr_algorithm_improvement_phase2d_binding_execution_20261023/INPUT_LAW_SUMMARY.json)を参照する。新しい検証PASSや厳密証明を追加したわけではない。

## 2. PPの費用を下げる具体的な機構

| 処理 | frame成分 | physical Pauli成分 | 根拠・限界 |
|---|---|---|---|
| 演算子表現 | Gaussian回転したZ／ZZ、supportは1または2 | 物理基底のIXYZ word、supportが長くなる場合もある | `MixedAtom`、`make_fragment_alternatives` |
| 基底への移動・復帰 | Gaussian B†とB。singletonはsupport completion、同基底連続runはfull frame | XにH、YにH/S、I/Zには変換なし。それぞれactual inverseで往復 | `BasisProvider.basis`、`build_event_sequence` |
| controlled中心部 | Z-supportのparity ladderとancilla制御回転 | 同じrecipe | `append_central`。Pauliが中心部だけでも常に安いとは言えない |
| 同基底の共有 | 連続する同じframe key内のみ | 同じlocal basis key内。IとZは基底選択が共通 | 異なるwordでも基底を共有できるが、frame/local間の共有はない |
| 成分のcollect | Sのorigin fragmentを保持 | Pを選んだfragment間で同一wordをexact collect、exact zeroのみ除外 | `canonicalize_arm`。PPのsampling weightも変わる |

中心部は、supportのparityをCXでtargetへ集め、`RZ(θ) → CX(anc,target) → RZ(-θ) → CX(anc,target)`を実行し、parityを戻す。**Gaussian変換そのものをancillaで全面制御しているわけではない**。両表現の差は、共通中心部の外側に必要な基底変換とsupport構造にある。identityはancilla位相へまとめられる。

Gaussian経路は `native_gaussian → U_to_qiskit_ops_jw_givens → BogoliubovTransform(U.T)`。固定版NatureはRZと隣接XXPlusYYを生成し、QiskitのXXPlusYY公式定義にはCXとRY・位相操作が含まれる。native rz/sx/x/cxへ下げると、この往復がRZ・CX・depthとして現れる。一方Pauli側の基底変換は局所Cliffordで、物理Zだけのwordでは基底変換を要しない。これはPPに有利な具体的処理差である。**削減量をXXPlusYY個数から推算してはいない**。[固定Nature定義](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py#L252-L259)、[固定Qiskit定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/xx_plus_yy.py#L143-L160)。

K=0ではeventはrotation1個・product0個で、r=2のordered pairを適用する。**L=51のPPで51成分を順に実行する構造ではない**。Lは全列挙・保存・前処理の規模とsampling lawには効くが、1軌道のevent個数とは別である。成分数の少ないSSが自動的に安い、という期待はこの実装には当てはまらない。

## 3. 共有・fusion・compilerについて確認できる範囲

- **実装済み**：隣接・同一basis keyのrunをB†／複数中心部／Bで囲む。単独frame成分にはsupportに必要な列を保持するcompletionを使う。従って「SSは基底共有やsupport最適化を全くしていない」という説明は誤り。
- **今回の制約**：q=1のためmacro間のD半ステップfusionは発生しない。K=0・r=2で共有できる非identity中心部はこの隣接pair内に限られる。別frame、frame/local間はkeyが異なり、identityもrun区切りになる。SOURCEは演算子順序を入れ替えず、中心回転角の明示的な融合も行わない。
- **compiler依存**：位相を保持する公式definition展開後、同じopt1 profileでfull-wrapperをcompileしている。Pauli局所ゲートや隣接回転の簡約は可能だが、保存結果にはpass別の費用差や最適化無効時の対照がない。特定passの効果量や「Pauli側のcancellationが順位の主因」とは断定できない。
- **古典cacheとの違い**：BasisProviderのcacheは生成済み基底回路のコピー再利用。13,006 unique compile／670 aliasも同一IRの古典処理再利用で、実行時に量子ゲートを省く操作ではない。量子費用を減らす同基底run共有と混同しない。

保存[CASE_RESOURCES.csv.gz](../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/CASE_RESOURCES.csv.gz)の特定行とBのatom metadataを照合した例を示す。全行の分類集計ではなく、構造を説明するための既存整数値の転記である。

| 保存case ID（X軸） | 選択成分の構造 | compiled RZ／CX／depth |
|---|---|---|
| `SS_000_000_X` | identity／identity | 27／16／38 |
| `SS_000_008_X` | identity／F1-frame `IIIZ`、singleton | 44／22／54 |
| `SS_001_001_X` | F0-frame `IIZI`を2回、同基底run | 51／28／65 |
| `SS_001_008_X` | F0-frame `IIZI`／F1-frame `IIIZ`、別基底 | 82／36／97 |
| `SP_000_001_X` | identity／physical `IIIZ` | 28／18／42 |
| `PP_001_004_X` | physical `IIIZ`／`IIZI`、共通local Z基底 | 29／20／45 |

この照合はGaussianを含むcaseの具体的な費用と、共有条件が実際の選択成分に対応することを示す。arm間では演算子・角度・確率が異なり、SS内の比較でも選択成分とsingleton/full recipeが異なるため、表の差をGaussian単独費用や共有効果の推定値にはしない。

**raw→compiledの差も最適化効果量ではない。** 例えば `SS_000_008_X` の保存rawはRZ10／CX2／depth19、compiledは44／22／54。`SS_000_000_X` のraw RZ/CXは0だが、compiledは27／16である。raw metricは未分解XXPlusYYやCPの内部RZ/CXを数えない。DのCPやreadout等もnative化されるので、これを「compileで費用が悪化した」「Gaussianだけの追加費用」と解釈できない。共通prep/readout診断を主費用から引くことも、full-wrapper fusionが非加法的なので行わない。

## 4. 混在辞書が中間になった理由と、断定できない点

**SP／PS < SS** は保存事実。片方をPauli化すると、そのfragment由来の非identity選択にGaussian往復を要しなくなることが、中間費用になる構造的な説明である。**PP < SP／PS** も保存事実で、混在armには残るframe基底費用とframe/local境界がPPにはない。ただし選択確率・support・collectも同時に変わるので、期待費用差を基底費用だけに帰属しない。

**SP < PSの原因はまだ一意に分離できない。** SPはF0をframeで残し、PSはF1をframeで残す。fixture定義ではF0は(0,1),(1,2)のplane rotation、F1は(2,3),(0,2),(0,1)であり、両fragmentは同じ構造ではない。これはF1の基底変換負担を疑う技術的理由になるが、support completion・full-run・sampling weightが異なる。元のplane rotation数だけから「F1のcompile費用が大きい」と認定しない。

PPのfragment間collectは保存atomの `P:collect:IIZI` のoriginsがF0/F1であることにも対応し、同一wordの重複・符号をまとめる経路である。混在armではS側を物理Pauliへ展開してcollectしない。**collectがあることは確認できるが、その順位への寄与量やcollect単独による改善率は未測定**。小さいLだけを費用差の原因とはしない。

以上のmechanismは固定SOURCEの性質であり、保存順位を説明する整合的な解釈である。順位の再現範囲は今回の小さいfixture・短いr・K=0・native basis・compiler・無routing条件まで。別fixture、大きいsupport、長い同frame run、別backend／coupling、最適化したGaussian合成へ一般化しない。独立新規性や一般的なPauli優位も認定しない。

## 5. λ・Btotと統計負担

| arm | 保存λ（丸め） | 保存F2/K0 Btot（丸め） |
|---|---:|---:|
| SS | 0.155739 | 1.000242547 |
| SP | 0.203588 | 1.000414480 |
| PS | 0.198515 | 1.000394081 |
| PP | 0.208101 | 1.000433059 |

[既存F2 law](../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/F2_laws.json)と実行報告の転記。F1/K=2のlawは用いていない。SSはλ・Btotが小さい一方、PPは1-shot費用が小さい。λは成分確率と回転角にも関わるため、4armを同一確率で比較した表ではない。
η／α／γ／intervalの資格、共通finite targetへの有限ビット誤差予算は未成立。認証NとN加重費用は未取得のままであり、この表から同精度総資源の勝者や統計負担との損益分岐を計算しない。数値的一致はγの厳密保証の代用にならない。

## 6. 有望な改善箇所と、追加承認が必要な作業

以下は実装上の候補であり、新アルゴリズムの採択・科学的GOではない。

| 候補 | 変更対象・期待する効果 | 判断に不足する証拠／次の作業 |
|---|---|---|
| supportに合わせたGaussian合成 | 現行completionは決定的な標準ベクトル補完で、native費用最小化ではない。選択Z／ZZに必要な変換とactual adjointをより安く実装する余地 | singleton／full-run別の費用内訳、その後に固定operator・符号・位相を保つ候補回路の対照が必要 |
| 実回路費用を考慮した表現選択 | λや成分数だけでなく、Gaussian往復・Pauli support・共有条件を考慮 | まず既存caseとlawを型・support・同基底条件で照合し、重み付き寄与を再集計する必要。認証Nは別問題 |
| 基底間の接続と中心部の簡約 | 現行は完全一致keyの隣接共有のみ。境界でB間の変換や連続controlled回転を直接合成する余地 | 軌道順序・sampling lawを変えずoperator等価を検証する新build／compileが必要。単純な並べ替えや長いrunへの条件変更は未承認 |

**次の最小追加作業の候補は保存データだけの寄与分析**：既存atom／case／probabilityを照合し、identity・frame singleton・同frame pair・異frame pair・frame/Pauli・Pauli/Pauli、support、F0/F1別の確率重み付き費用を出す。これにより高費用caseの型とsampling weightの影響を区別できる。ただし今回禁止された新しい数値再集計なので、実施していない。新しい回路を作らずに済むが、Gaussian単独費用やcompiler passの因果効果までは識別できない。

その分離が研究判断に必要になった場合に限り、同じ登録軌道・精度・wrapper・compiler条件で、共有有無や合成方式だけを変える限定対照のbuild／compileを別途設計・承認する。compilerを変更しただけの順位、identityのD移動、K/rやfixture変更を今回の結論に混ぜない。

## 7. 保全と引継ぎ

新しい数値結果・SOURCE・runner・grantは作成していない。既存実行SOURCE、raw、使用済みgrant／ledger／marker／STOPは別作業領域で保持し、本調査から書き込んでいない。公開するのは本メモと既存索引・現状・研究ノート・文書inventoryの追記だけである。

**mandatory STOP。次の担当はユーザー／GPT**：この技術的説明から、保存データの寄与分析へ進む価値を判断する。実施を選ぶ場合は、その数値再集計を明示承認する。Codexは承認後に既存CSV／law／atom記録を再利用して担当する。候補回路の実装・build／compile、γ保証、認証N比較、新しい重要科学レビューには、それぞれ必要な別承認を要し、自動移行しない。

主なSOURCE参照（すべて測定時の固定commit）：[回路・共有](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/circuits.py)、[辞書・collect・確率](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/prototype.py)、[fixture](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/fixture.py#L165)、[event recipe](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trotterlib/rte.py#L1594)、[Gaussian接続](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trotterlib/df_trotter/ops.py#L213)、[compiler](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/gates.py#L85)、[測定・集計定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/f2_execution.py)。
