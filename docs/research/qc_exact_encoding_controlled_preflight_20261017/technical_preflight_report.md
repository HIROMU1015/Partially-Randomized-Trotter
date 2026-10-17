# ④ exact fermion encoding・sectorとcontrolled時間発展：限定技術事前検証

2026-10-17 JST。Repository: `HIROMU1015/Partially-Randomized-Trotter`。資料読解・固定SOURCE/APIの静的照合による技術報告。新しい科学レビュー・証明・算法・科学条件の採択ではない。

**確認できたのは、既知の符号化と既存のJW/DF回路基盤を接続するための条件・不足機能である。具体的な新しい回路生成規則は、今回の入力資料にも本報告にも存在しない。** 現在具体的に指示できる実装は既知手順の接続であり、それだけでは独立した新算法の構成差にならない。一方、既知手順を超える構成が不可能という結論も出していない。

実施境界：新規科学計算、保存科学値の再解析、回路build/simulation/native compile、数値correctness test、新しい科学SOURCE、実験契約、execution authorization/grant/markerはすべて0件。停止済み系列は再開しない。成果公開後mandatory STOP。

関連資料：[SOURCE/API明細](source_api_inventory.md)、[固定証拠・静的検査](evidence_and_checks.json)、[公開照合記録](publication_receipt.json)。

## 1. 基点と確認範囲

- ④の[既存研究範囲][coverage]・[残る問題][landscape]・[GPT引継ぎ][handoff]：`56f1cc55c59ab7bc42c6b24e0953675de9b29b1d`。SOURCEの静的照合もこの固定snapshotを使った。
- 4系列の[戦略引継ぎ][four]・[重複対応表][overlap]：`d09ba606a9fceef806a9313e3f7c43cd21d19e3f`。他系列の新しい結果を取り込む監査は行わない。
- ユーザー指定の[GPTレビュー原本](inputs/qc_algorithm_strategy_ii_iii_iv_scientific_review_20261017.md)：43,206 bytes、SHA256 `8388dd1ba3a14ff147323af1f74ad65280508d9fe1a3894442a5ad9be99907f2`。原本を取得して読解し、bytesを変えず保存した。主に§§7–10、13–16の範囲・留保を継承する。
- [2026-10-17共通運用指示][instructions]：ユーザーのローカル原本と固定公開copyのbytes/SHA256一致を確認。§8.4を適用し、新しい人工的CPU/RAM/worker/wall上限や共有環境変更は設けていない。過去の固定契約の上限・STOPは変更しない。
- 既存依存関係は `requirements.txt` のQiskit 1.3.0／Nature 0.7.2／OpenFermion 1.6.1を基準にした。現行web docsの版だけから機能を推定せず、対応tagをcommitに解決して一次SOURCEを読んだ。import・インストール・互換性試験は未実施。

対象分子、basis、rank、grid、数値精度、許容閾値、費用の主指標は新たに選ばない。過去の分子数値の再評価・転記による優劣判断も行わない。

## 2. 生成問題の入力・出力と保持条件

### 2.1 入出力

入力として必要なのは、元の有限basis電子Hamiltonianの係数とscalar、fermion mode順、符号・単位・時間発展の規約、正当化された対象sector、利用可能なstate access、時間と要求誤差である。DF入力を再利用する場合も、その固定DFモデルと元のintegral Hamiltonianとのモデル差を消してはいけない。「exact encoding」は渡されたモデルの表現が正確という意味で、DF打切りやactive-space近似を遡って正確にするものではない。

出力はencoded Hamiltonian行列だけでは足りない。実行可能なgate列または生成手順、適用code/sector、control branchの位相、ancillaの初期・終了条件、必要ならencode/decode、前処理と反復費用の記述が必要である。準備unitaryと逆が入力に含まれる場合と、既に供給されたground copiesだけの場合を区別する。古典E0、古典固有vector、RDM、gapを無料入力に追加しない。

### 2.2 sector・codespace・元ground energy

レビュー§9.1のisometryを、論理空間から元の対象sectorへの `V` とすれば、保持条件は `V†V=I`、`HV=V H_enc` である。GSEのように補助qubitを使う場合には、向きを逆にして、物理対象spaceからqubit codeへの `J`、`J†J=I_S`、`JJ†=P_code`、`H_enc J=J H_S` と書く方が安全である。qubit全空間と物理sectorの次元が等しいと仮定しない。これは既知のintertwining条件の整理で、新しい証明ではない。[既存レビュー§9](inputs/qc_algorithm_strategy_ii_iii_iv_scientific_review_20261017.md)、[taperingの原論文][taper-paper]。

以下を別に確認する。

1. 元Hが対象spaceを保ち、符号化がそのspace上で正確であること。
2. 対象sectorの最低energyが、ユーザーの求める元問題のground energyに対応すること。固定Nだけ、固定NとSz、固定S²まで指定する場合は異なる。
3. 対象より広いparity sectorやcodespace外の低energyを、対象groundとして読まないこと。正しくcode内に準備して閉じた時間発展を使う場合、code外energyだけで動力学が無効になるわけではない。全qubit空間で無制約ground探索する場合には、別途code制約等が必要になる。
4. code外のpadding、ancilla、復号、読出しがcoherentな位相・state access契約と整合すること。

### 2.3 絶対原点とcontrolled相対位相

規約を `U(t)=exp(-itH)`、`H=E_off I+H_dyn` とすると、scalarの因子は `exp(-it E_off)` である。control-on-1の `diag(I,U)` では、この因子はcontrolの相対位相となる。nuclear/constant energy、identity項、対角回転の半角scalar、taper後にscalarへ移る項、basis変換と合成で生じる位相を追跡する必要がある。

既存コードはすでに、対角primitiveのglobal phaseをcontrolled時にcontrolのPhaseGateへ移し、energy shiftを回路global phaseへ入れる。したがって「global phaseを記録する」というだけでは未実装の新機構ではない。Qiskit 1.3.0のcontrolled化SOURCEも定義のglobal phaseをcontrolへ移す経路を持つ。ただし入力時点で消した位相を復元してくれるわけではない。[明細I02/I03/I07](source_api_inventory.md)、[Qiskit固定SOURCE][qk-control]。

`Operator.equiv` はglobal phase同値なので、control化前の一致だけでは絶対energyの検査に不十分である。完全なcontrolled演算子に適用する場合とは区別する。将来の検証では、scalar台帳とcontrol両branchを含む作用を対象にする必要がある。今回はその検証を実行していない。[Qiskit 1.3.0 SOURCE][qk-operator]。

### 2.4 fragmentの閉包とPF誤差

全Hのsector保存から、各fragmentを個別に圧縮した演算がunitaryだとは限らない。レビュー§9.2の条件 `(I−Π)UV=0` を、**独立した論理操作として圧縮するblockの境界**で確認する必要がある。各native gateが常に元sector内に留まることまで要求するものではない。ancillaや中間spaceを利用して最後に戻る既知実装を排除しない。

非冗長Pauli展開と共通Pauli Z2 symmetry等では、各項の保存が成立する既知条件を利用できる。DF/Gaussian blockでは、N、spinの各量、code stabilizerのうち実際に保存するものを個別に扱う。一般Bogoliubov変換、単なるN保存変換、spin-adapted変換を同一視しない。

同じ閉じたfactorを同じ順序・係数・位相で移すexact encodingでは、対応するPF固有位相とPF biasは変わらない。回路実装費の改善と、近似積そのものの誤差改善を混ぜない。順序や分割を変える案は、②や停止済みPF系列との重複判定が必要な別の変更である。[レビュー§§9.2–9.4](inputs/qc_algorithm_strategy_ii_iii_iv_scientific_review_20261017.md)。

### 2.5 同じtaskでの費用と有限精度

レビュー§9.4の費用式 `G_m^(g)=F_m^(g)+Σ_(r,a) N_(r,a)(C_(m,r,a)^(g)+P_(m,r,a)^(g))` を使う場合、初期変換・code準備・復号が固定費Fか各試行費Pかを実際のaccessに合わせる。controlled evolutionはCに含める。同じ信号・誤差・confidence契約でscheduleを共通にできるときだけNを共通とし、qubit数、gate種、depth、古典前処理時間を無根拠に一つの数へ合算しない。今回はどの項も数値化しない。

数学的にexactなencodingでも、係数入力、変換生成、回転合成、PF、state準備、統計的読出しの誤差は別である。全HamiltonianのN保存は個別Pauli回転のN保存を一般には保証しないため、N圧縮後に使うblockを先に定義する必要がある。exact mappingだけを理由にshot scheduleやenergy精度を同一とみなさない。

## 3. 既知方法で可能なことと適用条件

| 方法 | 既知の出力・保持範囲 | 今回の問題に自動では付かないもの |
| --- | --- | --- |
| JWとGaussian/Givens | fermion演算子のqubit表現、数保存一体変換の構造化回路。既存DF基盤が直接利用する | interacting H全体のexact exponent、別encoding上の同じ安いGaussian回路、要求sectorの選択・準備 |
| Z2 tapering／ParityMapper | commuting symmetryの固有値を指定したoperator削減 | 固定N/S²を丸ごと指定する圧縮、全fragment・state・制御・費用の接続 |
| BKSF／GSE | graphとcodeを伴うeven fermion algebraの表現 | 任意parity/Nの無条件対応、任意のGSE構成を既存BKSF API一つで実装、code準備とcontrolled oracleの総価格 |
| number/Sz sectorの古典圧縮 | 対象basisでの行列・vector・matrix-free solver | 小qubitレジスタ上での効率的gate生成・準備・復号 |
| spin-adapted Paldus変換 | spin-free HのN/S/Mをラベルするbasis変換と、論文で構成された演算 | 一般二体interacting Hに対する無条件の安い直接controlled時間発展 |

**tapering**：対称性sectorを使うqubit除去は既知である。二つのspin-population parityを固定しても、一般にNα/Nβの整数値そのものを固定したspaceとは異なる。ground-energy読出しに必要なsectorの同定を省略しない。[Bravyiほか][taper-paper]、[Nature固定ParityMapper][qn-parity]。

**GSE**：Setiaほかの主構成はconnectedなeven-degree graph、局所Majorana、向きとloop stabilizerでcodeを指定する。本文で説明されるeven-parity表現と、さらに条件を強めた誤り訂正定理のgraph要件を混ぜない。qubit追加とweight低下の交換を含み、一般の密な化学係数で総費用が下がるという結論ではない。[GSE原論文][gse-paper]。

**分子GSEと軌道変換**：Brownほかは、分子系のGSE、軌道回転、graph/path構造を既に結び付けている。本文のdepth proxyを、こちらのstate preparation・controlled絶対energy・全round費用の実測値へ転用しない。論文全体がこちらの未提示構成を包含する／しないという判定は行わない。[原論文v1][brown-paper]。

**Paldus取得状況の更新**：元レビューはP05本文未取得だった。今回v2 PDFの導入および§IV A–Cの必要箇所を取得・読解した。spin-free HのN/S/M分解とCSF準備は既知の具体的構成であり、単なる「spin sector回路がない」という空白認定はできない。一方、§IV B.2は一体u(d)のgroup actionと一般interacting Hを区別し、§IV Cのfast-forwardingも対象classを限定する。basis変換の存在だけで一般化学Hの安いexponentを得たとは扱わない。94頁全体の定理・回路の独立検証ではない。[Quantum Paldus Transform v2、特に印刷pp.38–40][paldus-paper]。

## 4. 既存SOURCE/APIの接続状況

[明細](source_api_inventory.md)に固定path・関数・上流commitを記録した。静的照合で重要なのは次の点である。

1. `DFHamiltonian`、JWの構造化Gaussian、DF diagonal primitive、identity policy、controlled scalarの既存基盤は再利用候補として実在する。実行を伴う再認証はしていない。
2. `PhysicalSector` は古典matrix-free solverのbasis restrictionである。`spin_sector` のN/Sz指定を、S² spin-adapted encodingやencoder circuitと呼べない。
3. OpenFermionの当該Sz参照pathはデフォルトで交互spin modeとMSB側mode-index規約を使う。一方、NatureのParityMapper自動二qubit削減はblock spin orderを前提とする。単に `num_particles` を同じ値で渡す接続は安全でない。fermionic mode再順序付けと単なるbit-endian変更を区別し、係数・state・symmetry・符号を一緒に対応付けることは技術的に必要だが、それ自体を新算法とはしない。
4. Nature 0.7.2の `BogoliubovTransform` はJWのみ対応する。N×Nの数保存入力とN×2Nの生成消滅混合入力を区別する。JW circuit生成器をそのままtaper/GSE generatorと見なせない。
5. taper APIはsymmetry非可換な補助operatorを `None` とし、設定によって除外する。未対応をゼロ演算子・保存された量と誤認せず、項の対応を保つ必要がある。operator mapperの存在から、対応する全回路・準備・制御があるとは言えない。
6. 旧grouped-Pauli builderにはidentityをskipする経路がある。既存の定数台帳なしで絶対energy用途へ直結しない。別の現行DF pathがscalarを処理することと矛盾しない。
7. 旧表現研究には、Gaussian circuit、条件付き回転、sector/leakage、Lie closure、clean-workspace、制御後の位相を監査するprototypeがある。ただし一部はdense matrix/Statevectorから位相を合わせる有限規模診断であり、それを一般入力の効率的な位相生成規則として無料利用できない。

固定snapshotのtracked `src/`・`scripts/`を対象に、具体的なtaper/BKSF API名等も検索した。統合された新encoded generatorの呼出しはこの範囲では確認できない。ただし全branch・全履歴・未取得原本の機能不存在を証明する検索ではない。依存libraryが持つ機能と、repository内で接続・検証済みの機能を分ける。

## 5. 障害の分類：技術接続と科学的に未提示の規則

| 論点 | 既知方法で解決する部分 | 現資料で閉じない部分 | 現段階の扱い |
| --- | --- | --- | --- |
| mode/basis/phase規約 | 明示的な変換仕様とscalar台帳で整合させる | この研究用adapterは未実装・未検証 | 技術的不足。新SOURCEは今回作らない |
| symmetryとsector | exact symmetryと固有値、code stabilizerの既知構成 | 対象groundを含むsector・利用可能stateの契約 | 入力条件が未指定。勝手に選ばない |
| fragmentとcodeの閉包 | 保存する論理block単位で既知intertwining条件を使う | 一般の選択fragmentが同じ安いgate集合へ閉じる根拠 | 具体blockがないので判定不能 |
| compressed exponent | known encodingと通常のPauli/Gaussian実装はbaselineになる | 相互作用項の入力から、既知baselineを越える直接生成規則 | **主要な未提示事項**。回路が小さくなるだけでは埋まらない |
| 制御・準備・復号 | 既知のcontrolled化やcode準備手順を価格に戻す | 入力accessのまま要求信号を保つ安い具体手順 | oracleや逆を無料で追加しない |
| 利益・新規性 | 同一taskで資源ごとに比較できる | 構成差とその費用機構、既報包含、実利益 | 未認定。欠測を新規性と呼ばない |

厳密表現から効率的な回路生成までの不足と、ソフトウェア上のadapter欠落は異なる。前者を後者の実装で解いたことにはできない。現在は既知方法の再実装へ必ず還元するという一般判定も、未知の有利な算法が残るという認定もできない。**提案された具体手順がないため、方法差の有無の判定対象そのものが未完成**という限定結論である。

## 6. ④の停止系列および②との重複

| 既存系列 | 既に扱った内容 | 新構成の代わりにならない作業 |
| --- | --- | --- |
| ④ N1／旧表現S12–S14 | 表現selector、sector/leakage、Lie代数、条件付き可解block | 同じcatalogにtaper/GSE名を追加してscanするだけ |
| ④ N3／active space | occupation/sector変更や縮約の限定検討 | 切断した別Hを元の絶対energy問題と同定する |
| ④ H3／THRIFT・mixed access | frame/Gaussian共役、既存oracleとその制御費 | 同じ共役をencoding名で再開する |
| ④ 一体再配分PF、H-E1、M、SOS-PF、H1–H4 | 各固定構成の監査と停止・非採択の記録 | 順序・分割・符号を変更して停止系列へ戻る |
| ② Gaussian積・basis fusion | ordered operationを保つGaussian積生成、既知の反射・隣接相殺 | 同じ処理の再記述や再compile |
| ② sampling・Parent allocation、③M1-S | law/variance/価格交換、共有labelと信号診断の既存成果 | 固定されたlawやcoherenceの診断を別名で繰り返す |

根拠は[④研究範囲][coverage]、[4系列対応表][overlap]、[Gaussian積の固定closure][gaussian-closure]。今回、これらの科学結論を新たに認証せず、停止範囲を広い一般no-goへ拡張もしない。H1–H4のPAUSED、N1の現scan STOP、N2 HOLD、N3現系列CLOSED、THRIFT現template CLOSED、②の既存区切り、③の統合後STOPはすべて維持する。

## 7. 次に具体化しなければならない一つの問題

今後ユーザーが科学的構成作業を別途指示する場合、対象は次の形まで絞る必要がある。

> 指定された元電子Hと正当なsectorについて、**どの論理blockを、どの入力から、どの順序・位相で生成するのか**を一つ提示し、その手順が既知encoding＋通常のcompilerから何を省く／共有するのかを明記する。codeへの入出力、control scalar、要求誤差と前処理を含めても、その構成差が残る条件を示す。

これは新しい生成規則の提案ではなく、未提示の構成を判断可能にするための入出力上の不足の記述である。新しい基本gate、全分子への利益、完成した総費用証明を先に要求するものではない。一方、単なるqubit数やPauli weight、疎な行列、既知手順を呼ぶadapterだけではこの不足を埋めない。

科学的に残る事項は、具体的なH class/論理blockと手順差の提示、その仮定の妥当性、先行構成との包含、主な資源座標、テーマとして進めるかの判断である。routineなAPIの名前・version・既知の位相処理経路は明細で解決した。新しいbenchmark、数値条件、runnerや実行契約を先回りして作成しない。

次の担当はユーザーである。本報告を受け、具体的な構成作業を別途依頼するか判断する。依頼されるまでCodexは追加survey、科学レビュー、SOURCE実装、停止研究の再開、科学計算へ移らない。具体化後も正式採択と数値/compile実行承認は別である。

## 8. 確認結果と限界

- 固定資料・原本SHA・SOURCEの関数/行・依存tagとcommitの対応・文書リンク・追加差分を静的確認した。検査明細は[台帳](evidence_and_checks.json)に記録する。
- 数式は原レビューと既知方法の条件の引継ぎであり、新証明ではない。実H入力のsector閉包、生成回路の位相、数値correctness、runtime、gate数・費用・性能は未検証。
- Paldus本文の必要箇所は今回新たに確認したが、旧レビュー原本の「未取得」という時点記録は書き換えていない。他の未取得Claude原本、出版版差分、全branch/文献の網羅監査は行っていない。
- 本報告だけで研究テーマ・新算法・優位性・一般的な不可能性を採択しない。既存SOURCE/raw/科学契約/authorization/grant/ledger/marker、親worktreeと他作業領域は変更しない。

**限定技術事前検証を終了し、mandatory STOP。**

[coverage]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md
[landscape]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/open_problem_landscape.md
[handoff]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/gpt_decision_handoff.md
[four]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d09ba606a9fceef806a9313e3f7c43cd21d19e3f/docs/research/qc_four_track_strategy_handoff_20261017/strategy_handoff.md
[overlap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d09ba606a9fceef806a9313e3f7c43cd21d19e3f/docs/research/qc_four_track_strategy_handoff_20261017/overlap_and_progress_map.md
[instructions]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/research_project_instructions_unified_2026-10-17.md
[gaussian-closure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md
[taper-paper]: https://arxiv.org/html/1701.08213v1
[gse-paper]: https://arxiv.org/html/1810.05274v2
[brown-paper]: https://arxiv.org/html/2511.09322v1
[paldus-paper]: https://arxiv.org/pdf/2506.09151v2
[qn-parity]: https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/qiskit_nature/second_q/mappers/parity_mapper.py
[qk-control]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/add_control.py
[qk-operator]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/quantum_info/operators/operator.py
