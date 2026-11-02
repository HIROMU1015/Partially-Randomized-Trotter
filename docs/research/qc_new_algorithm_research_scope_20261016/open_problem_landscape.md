# 残る科学的問題と未確認事項

2026-10-16 JST。D1・D2内の研究課題選定に直接関係する限定技術調査。[既存の検討範囲](research_coverage_map.md)を前提にする。**以下は採択候補の提案・順位表ではなく、既知構成と未確認の接続を区別する問題表である。具体的な新アルゴリズムは構成していない。**

## 今回の一次文献照合

既存K0/K0.5の文献監査は再実施せず、下記8原論文の必要箇所へ限定した。HTML本文を読んだ範囲を明記し、論文全体や出版版との同一性をbyte認証したとはしない。arXiv HTMLの生成日表示と論文version日付を区別する。

| ID・原論文・取得版 | 確認箇所と既知の構成 | 今回の境界確認 |
| --- | --- | --- |
| L1 Kökcüほか、[Fixed Depth Hamiltonian Simulation via Cartan Decomposition, 2104.00728v4](https://arxiv.org/html/2104.00728v4) | Hamiltonian Algebra、Cartan Decomposition、Thms.1–2、Determining Parameters、App.F。交換子閉包・可換代数への変換から時間非依存depthの回路を構成する | depthの時間非依存性はsystem size・代数次元・古典前処理の効率性を保証しない。一般相互作用化学Hの低費用化は本資料から未確立 |
| L2 Patel–Yen–Izmaylov、[Extension of exactly-solvable Hamiltonians using symmetries of Lie algebras, 2305.18251v2](https://arxiv.org/html/2305.18251v2) | §§II、III.2–III.3、V、App.A。対称性多項式を係数とする可解Hamiltonian、条件付き対角化／測定 | 既に旧レビューS13/S14が扱った文献。測定時のmid-circuit分岐とcoherent controlled evolutionの総費用は同じでない。新発見としない |
| L3 Rajabzadeh Dizajiほか、[Hamiltonian simulation in Zeno subspaces, 2405.13589v1](https://arxiv.org/html/2405.13589v1) | Implementation and circuit complexity、Eqs.11–15、Relation to QDRIFT。PREP/SELECTと反復ancilla測定／kickを使う | Eq.15は小時間発展の和のblock encodingを明示する。H自体のblock encodingでないという名称差でも、block-encodingを中核とする本構成はD1外の比較資料 |
| L4 Kimmelほか、[Hamiltonian Simulation with Optimal Sample Complexity, 1608.00281v1](https://arxiv.org/html/1608.00281) | Thms.1、5、App.B/C・Thm.20。状態ρのcopiesとpartial SWAPからρ生成の時間発展・controlled版を近似 | 供給されるρはHamiltonianとして使う状態。元化学Hのground projectorのcopiesは、Hのshift/normalizationを符号化したρのcopiesではない。一般copy下界をground-energy推定の一般下界に転用しない |
| L5 Zhao–Rubin–Miyake、[Fermionic partial tomography via classical shadows, 2010.16094v3](https://arxiv.org/html/2010.16094v3) | Thm.1、App.B.4。fermionic Gaussian測定でRDM／observable期待値を推定 | 元Hの期待値の取得は既知の対照。全k-RDMの同時推定保証と、既知の一つのHのenergy推定に必要な費用は別 |
| L6 Setiaほか、[Superfast encodings for fermionic quantum simulation, 1810.05274v2](https://arxiv.org/html/1810.05274v2) | 導入、Superfast Encoding、GSEのgraph/code条件。encoded物理部分空間で元fermion Hと対応する表現 | Pauli weight削減は追加qubit・graph degree・code制約を伴う。一般の密な分子係数にそのまま低費用を保証しない |
| L7 Bravyiほか、[Tapering off qubits to simulate fermionic Hamiltonians, 1701.08213v1](https://arxiv.org/html/1701.08213v1) | §§VII–VIII、particle/spinおよび複数Z2 symmetryによるqubit除去 | sectorを指定した符号化は既知。指定sector内energyが要求する元問題のground energyに対応する条件を残す。sector選択自体を解いたことにはならない |
| L8 Majsak–McNulty–Oszmaniec、[A Simple and Efficient Joint Measurement Strategy for Estimating Fermionic Observables and Hamiltonians, 2402.19230v3](https://arxiv.org/html/2402.19230v3) | §§II.2–II.4、III、IV.1–IV.2、補足C。fermionic observable／実電子構造型Hのjoint測定構成 | 同一copy上で複数observableの情報を得るjoint POVMは既報。複数copyを同時操作するcollective測定とは区別。対象係数・Majorana構造、layout、variance条件を一般化しない |

L1/L3〜L8は今回の範囲境界を具体化する追加対照。L2は過去の取得記録を発見したため、その必要箇所のみ確認した。L8で通常のfermionic joint測定まで既知と確認できたので、「joint測定」という名称だけを新しい課題にする追加探索は停止した。引用は要点の要約に限り、原文・図・数値結果を転載しない。

## 観点C1：代数・保存量・軌道構造

**問い：** 元の相互作用電子Hに対し、安い前処理から効率的なcoherent時間発展を保証する構造を認識・利用できるか。

既知の可解Lie代数、symmetry-polynomial block、Gaussian一体変換、occupation条件付きSU(2)、THRIFT mixed accessは既に構成または調査済み。プロジェクトでは旧表現研究S12–S14が、小Lie closure、可解mixed oracle、分子係数からの条件付き回転まで検討していた。したがって「化学の対称性を使う」という広い領域は未検討ではない。

取得資料で閉じていないのは、**どの相互作用化学クラスに対し、係数からの認識・代数記述・変換生成・controlled化・誤差精度までが安いか**という接続である。L1は代数次元と前処理を無視できないこと、L2は対称性依存可解性自体が既知であることを示す。これらから一般化学Hが効率的／非効率的のどちらとも結論しない。

新アルゴリズムにつながるには、既知のGaussian/Patel/THRIFTに含まれる操作を列挙するだけでなく、対象クラスと入力サイズに対する独立した構成・精度・費用根拠が必要になる。これは必要証拠の列挙であり、その構成を今回設計したものではない。D1は直接回路による代数分解なら適合し得る。D2は元Hを保持し、近似・sector・scalarの誤差を計上できる場合に限る。

今後の静的調査で確認できるのは、明示されたクラスの定理・出力回路・既知法への包含と前処理条件。GPTの判断事項は、従来のmixed-block課題と区別できるクラスが実際に提示されたか、既存停止系列の再開になるかである。現時点の分類は②③④で、新領域・有望性は未認定。

## 観点C2：PF以外の時間発展

**問い：** 小指数演算子積の係数／順序を変える以外の構成が、現入力モデルで元Hのcoherent evolutionを低費用化できるか。

旧K0はinteraction picture、THRIFT、CFQM、SPRINT/GRADE、乱択／signed meanを既に比較した。既知指数演算子の積や測定値の線形結合を、名称だけで新しいcoherent propagatorとしない。CFQMの和の指数を、既存の単一DF-square primitiveと同じ価格では数えられない。

今回確認したL3のZeno法は、測定を使うにもかかわらずPREP/SELECTによるblock embeddingを構成の中核とするのでD1内の新候補には入らない。これは当該構成の分類で、全測定型simulationの不可能性ではない。LCUという用語を一律禁止するのでもない。coherent block embeddingを必要とする実現と、古典postprocessingされた期待値を区別する。

L4のsample-based simulationはPFの時間刻み変更とは異なる既知構成で、partial SWAPそのものはD1の禁止primitiveではない。しかし、供給する状態ρ、copies取得費用、Hへのshift/scaleの対応、controlled精度が契約に必要。exact ground copiesだけではこの入力は供給されない。係数からρを安く準備できることは今回未証明・未評価であり、強い新oracleを無料で追加して低費用化とはしない。

今後静的に確認するなら、提示された具体的構成の入力・出力・誤差・成功確率・primitiveを読み、D1内かを先に確定できる。新しい構成原理や同一入力での利益は新たな科学的証拠が必要。現時点では**D1・D2・既存入力を満たす新しい有利な非PF生成規則を特定していない**。新手法探索を無制限に続けず、比較条件をGPTへ戻す。

## 観点C3：量子回路の実装構造

**問い：** 元Hを変えずに、fermion encoding・exact symmetry sector・回路レイアウトによって、高価な操作が必要になる構造を変えられるか。

Gaussian/Givens、DF、swap network、basis fusion、native synthesis、strict controlled phase、SPRINT/GRADE、joint境界最適化は既存基盤／監査の対象。新compilerがないことを科学的未検討の根拠にはしない。L6/L7によるencoded algebraやtaperingも既知の具体的構成である。

今回確認した基盤G9と既存表現レビューには、L6/L7等を**同一の元H・state access・絶対energy精度・成功確率で、encoded preparation/control/readoutまで含め比較する**十分な記録を確認できなかった。これを狭い⑤とする。encoding一般の新規性や未解決性を認定するものではない。

未確認なのは、化学係数の密さ／軌道表現に対するgraph構造、code/sector準備、物理部分空間外の作用、ancilla、制御、scalar phase、測定までを戻したとき、削減したPauli weightやqubit数が総費用へ残るか。古典のfixed-sector行列が小さくなることから、安い量子encoded oracleは導けない。正確なsector内作用はD2に適合し得るが、対象sectorに元問題のgroundがあるという契約が必要。

新アルゴリズムと主張するには、既知mappingを選ぶ工学的比較を超える、元Hに対する回路生成規則と費用機構が必要。静的な次の確認は、具体的なmappingのisometry・対応H・phaseと元基盤への入出力仕様まで。回路生成・gate数評価・compileは別の科学作業となる。GPTは、この問いが独立した研究問題か、既知実装の比較に留まるかを判断する。

## 観点C4：絶対基底energyを得る情報

**問い：** exact ground copiesを利用できる場合、元Hのenergyを読むために必要な情報アクセスと量子操作をどう区別すべきか。

入力契約を分ける。ground copies、準備unitaryと逆、controlled evolution、既知energy referenceとそのcoherent superposition、元H係数からの直接測定、複数copyの同時操作は同一資源ではない。D2ではenergy originを追跡する。H+cIに対するuncontrolled channelの不変性は、既知係数を用いた直接測定を否定しない。

H2/H4およびClinton/Kuji/QFD等の近接関係は完了K0 S1/S2で調査済み。L5/L8により、fermionic RDM・元H期待値を得る非QPE測定も具体的な既知対照であると確認した。joint observable測定をcollective-copy測定と呼び替えない。exact eigenstateでのHの分散が0でも、H全体をprojectiveに測る回路が無料で与えられるわけではなく、分解項の測定分散が0になるとも限らない。

検討記録が不足する可能性を残すのは、**この入力差を固定した上での、既知測定法・coherent情報取得・複数copy操作の同一absolute-energy費用比較**である。一般state property learningの優位やL4のsample complexityを、化学ground energyでの優位・下界へ移していない。exact ground copiesでの期待値はE0だが、近似／guiding stateなら期待値をE0と同定する別保証が必要。

必要な構成原理の証拠は、Hに依存した実行可能な測定／読出し手順、phaseまたは絶対offsetの扱い、copiesとcoherent preparationを混ぜない精度・成功保証、全準備・測定・古典処理費用。現時点では新手順を提示していない。D1は個々の測定構成で確認し、D2には適合し得る。ただしsimulation回路生成を改良せず測定へ研究対象を移すことが、ユーザーの求める新Hamiltonian simulation研究に含まれるかはGPTとユーザーの科学的判断に残す。今回勝手に拡張しない。

## 観点C5：理論の未閉鎖条件と生成算法の違い

**問い：** 化学構造に依存する理論の穴を埋めることが、実際の生成・情報取得費用の改善へつながる条件は何か。

有限stepの有用なgap/remainder認証、commutatorの鋭さ、overlap/branchとphase、normalizationと長時間signal、rare eventまでの費用は未閉鎖だが、既にK0/K0.5、C1〜C4、高次PF、第1・第2研究、Track A/Bの検討対象である。これらをまとめて「未検討の理論」とはしない。

特にSOS-PF A2はNOT_ADOPTEDであり、新しい保証研究として開始しない。既存のgap-free operator/output保証と、孤立ground branchを追跡するgap依存保証は区別する。Track BのM1-S理論も、固定有限involution・対象state・共有乱択law等の受理範囲を保ち、一般DFやRPE総費用が解決したとはしない。

未確認の接続は、係数から安く入手可能な証拠で、generator選択／回路／測定条件が実際に変わり、強い既知対照を超えるかである。誤差上界だけの改善と新生成算法を分ける。既存理論の文献対応と入力条件は静的に確認できるが、役立つcertificateの存在・効率・精度での総費用差は現資料にない。新しい数学的証明または別途承認された評価が必要で、ここでは作成しない。

D1/D2への適合は改善対象の実現法次第。GPTは、保証理論自体を研究目的として採択するか、実際の新回路生成につながる問いだけを採るかを判断する。これはA2や既存停止系列を再開する指示ではない。

## 未取得・未認証の情報

- Kang–Su ACM DOI `10.1145/3816436` 出版版はK0.5で403未取得。arXiv:2510.06316v3と出版版の数学的差分は未確認のまま。今回再取得・再監査しない。
- 過去レビューが留保したPR論文の新版全文・出版差分などを、今回読んだことにはしない。L1〜L8も出版版との全差分監査を行っていない。
- 完了K0で未取得とされたClaude原本 `レビューoutter(3).md` は今回も取得していない。SOS-PF A3時に別途保存されたClaude原本があることから、この別原本を取得済みとはしない。
- L4のNature掲載ページは今回の直接openでInternal Error。arXiv v1本文で上記必要箇所を確認した。掲載版との差分は未確認。
- 全branch・非公開資料・全関連文献の網羅的取得、collective-copy energyに特化した最適性定理、一般化学Hの効率的Cartan構成は確認していない。不存在・一般no-goを主張しない。
- 新たなHamiltonian、数値gap、RDM/期待値、最適化、PF実行、回路生成、compile、保存科学数値の再集計は0件。比較に必要な科学数値は未取得として残す。

**限定調査をここで終了する。** 追加の検索だけで新アルゴリズム成立を主張できる材料は得られていない。次のLevel 1判断点は [GPT引継ぎ](gpt_decision_handoff.md)に集約し、公開・独立照合後Mandatory STOPとする。
