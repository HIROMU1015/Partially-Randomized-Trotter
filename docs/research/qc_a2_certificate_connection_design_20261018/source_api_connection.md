# A2 SOURCE/API接続表と実装計画の境界

2026-10-18 JST。[接続仕様](certificate_connection_spec.md)／[比較・費用設計](comparison_and_cost_design.md)／[handoff](decision_handoff.md)。既存SOURCEの機能・不足は[preflight対応表の固定正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c62fe99d29e7edcaf0cb655d789d297890709a29/docs/research/qc_a2_practical_preflight_20261017/certificate_source_matrix.md)を再利用する。今回は機能の再監査・実行をしていない。

以下の既存ファイルの正本commitは `db1819587df36ee8833b37227ca0083faa0408ec`。bytes/SHA256・function読取範囲は前回evidence indexのSOURCE_01–SOURCE_18に固定されている。新仕様へ接続する際はそのidentityを照合する。**`SPEC-*`は今回設計した作業単位であり、実在する関数・実装済み機能の名前ではない。**

## 1. 既存APIと必要adapter

| 既存ファイル／実在API | 利用できる部分 | 新しく必要な処理・検査 | 作業ID |
|---|---|---|---|
| `df_hamiltonian.py: DFHamiltonian`、snapshot writer/loader in `rte_connected_cluster_cost_validation.py` | c、q、signed λ、gの保存fieldとfile identity。loaderは既存数値representationへの入口 | rawからexact dyadicをdecodeし、dtype/shape/規約/Hermiticity/全blockを検証。既存loaderのfloat変換や`.hermitized()`をexact対象の正本にしない | SPEC-01 |
| `df_hamiltonian_from_integrals`、`df_decompose_from_integrals`、`_parse_low_rank_result` | full/correction、spin-orbital/spatial、truncation metadataの規約を接続する参考 | 初期certifierは固定DFを受け取り、DF生成を呼ばない。由来のtransform tagを保存し、truncationをcへ加えず、元分子の同値証拠と分離 | SPEC-01 |
| `df_hamiltonian_to_model`、`interaction_operator_from_df_model` | 既存回路bridge/再構成規約 | full payloadとcorrection-onlyを別adapter型にする。外側cとonebodyの二重加算、tensor factor/normal orderを検査。既存callerのバグ認定ではない | SPEC-02 |
| `PhysicalSector.number_sector`、`_build_excitation_table` | 古典basis orderingの参考 | 固定Nの文法証拠・mode/JW順とlogical fragment保存。基底外skipを保存則証明に使わない。quantum sector encoderへ格上げしない | SPEC-02 |
| `diag_hermitian`、`_block_operator_norm_upper_bound` | 後続の数値proposalや回路parameter生成の参考 | exact係数三角capを新実装し、q上下端・u・PSD shift・L/f/Λのledgerとcheckerを作る。float eig/helperをcertificateとして再利用しない | SPEC-03 |
| `df_fragment_weight`、`_pf_phase_rate_upper_bound` | 既存のpartition proxy/phase tracking用途に限定 | f_j/Λの代用品としての流入を拒否。phase-rateの真energy入力を証拠の無料入力にしない | SPEC-01/03 |
| `solve_df_ground_state`、state cache funcs | 古典state/energyの保存規約を知る参考 | 初期Uは明示determinantのexact expectationから生成・独立検証。solver残差受理やHF/FCI値を厳密上界にしない。ground quantum accessも別管理 | SPEC-04 |
| K0文書 §4のraw辞書・行和（既存専用certifierは未確認） | 固定係数1/24・−1/12、添字、B更新・行和評価 | raw辞書、sqrt上界witness、対角重複を保持する行和generatorとchecker。算術・bit/保存費の計測 | SPEC-05 |
| K0.5 §3.4、EW2監査 §5（既存専用certifierは未確認） | 同順序C2、centered strong counter | center capsとexact grouped commutator/JW係数の構造経路。両経路のnorm証拠、min選択・費用を検証。float normal-orderだけを根拠にしない | SPEC-06 |
| `pf_decomposition.iter_pf_steps`、`product_formula.trotter_2nd_list` | 固定palindromic S2のstage規約 | 表示積↔回路時系列、半stage、全時間係数1、零fragment保持、PF IDと辞書の順序をexactに検査 | SPEC-02/07 |
| `build_df_blocks`、`build_one_body_gaussian_block`、`df_squared_diagonal_primitives`、`append_diagonal_primitives` | Gaussian/対角平方、signed λ、RZ/RZZ、control上のPhaseGate | PSD F=B+sIと元Bの対応、L+Σs=c、対角化/回転の残差と合成精度ledger。Nature/Givens/dense fallbackのbackend/versionを宣言 | SPEC-08（後段） |
| `QiskitDFPartialS2CircuitBuilder`、`QiskitDFPartialS2RepeatedCircuitBuilder` | deterministic-only、controlled phase、固定kernelの反復部品 | A2は全項deterministic。random-tail/RTE平均は拒否。fixed h powers、scalar/frame、QPE scopeとの接続は別検証 | SPEC-08（後段） |
| `rpe_hadamard_interrogation.py`、`df_rpe_resource.py` | 既存wrapper/費用metadata。state preparationを含まないという境界 | QPE readout/alias/powersとstate prepの未実装・未評価を正しく残す。過去RPE/PR費用をそのままA2総費用へ移さない | SPEC-08（後段） |

別系列のN3 rational witness checkerの存在は前回確認済みだが、それを改修・実行しない。本計画の独立checkerが実装済みになったとも扱わない。general SOS・state-prep・新PF分割の探索は本表の作業に含まない。

## 2. 生成と検証を分離した実装単位

| ID | 後続に実装する入出力 | 必須の独立検査 | 依存 |
|---|---|---|---|
| SPEC-01 | raw→ModelIntake、exact coefficient table、canonical model digest | 別decode経路でraw bits/正規形/shape/Hermiticityを照合。暗黙修復・未指定cast・二重correctionを拒否 | なし |
| SPEC-02 | context/sector/PFとscalar inventory→IDs・文法witness | 全fragmentのlogical N保存、mode順、stage順、scalar総和、全block対応 | 01 |
| SPEC-03 | exact係数→q上下端/u、K0 shift、L/f/Λ | checkerが正確なcap・符号・scalar帳簿を再生成。証拠の値を信頼しない | 01–02 |
| SPEC-04 | fixed occupied list→U証拠→同じcontextでW | occupation符号/遷移集約、norm=1、N、expectationとsigned λを別処理で検証 | 01–03 |
| SPEC-05 | 同順序f→raw κ、sqrt witnesses、行和、K_SOS_upper | 辞書version/全件・対角二重加算・各u²≥z・最大行和 | 02–03 |
| SPEC-06 | 同じfragment algebra→grouped C2 norm候補/Cbar2 | center cap、exact CAR/JW/Pauli符号と相殺、全norm候補の正当性と最小値、公開情報の共通性 | 01–03 |
| SPEC-07 | 全受理bundle＋共通task policy→route条件/step enclosure/比較record | exact有理数predicate、open domain、特異case、context mismatch、no-advantage/未解像保存 | 03–06 |
| SPEC-08 | access/circuit/readout/cost ledgerのadapter | logical exact証拠と数値回路残差を分離。条件付きstepと実行/総費用の状態を別々に検査 | 後日採択したtask・実装範囲 |

01–07も新しい科学SOURCEであり、今回実装していない。08のnative/QPE実装・実験は今回準備しない。ファイル名・公開API名・packageを勝手に既存namespaceへ追加しない。

## 3. 将来の検査coverage

後続の技術実装を別途承認する場合、実装だけでなく検査の許可範囲も固定する。少なくとも、exact decode（subnormal/符号付き0/非有限）、複素Hermitianの受理/拒否、positive/negative/zero λ、N端点、scalar/correction混用拒否、同一添字のB二重加算、sqrt witnessの逆向き拒否、grouped相殺・Pauli符号、M=1/Λ=0/W=0/Cbar2=0、厳密open endpoint、identity違い拒否、部分保存・中断を検査対象にする。

これは将来のsynthetic/数値テスト一覧の仕様であり、今回そのテストを作成・実行したものではない。実H4等の保存入力を自動fixtureにしない。SOURCE設計検査、人工数値検査、実入力certificate生成、科学比較をそれぞれ明示して承認範囲へ含める必要がある。

generatorとcheckerの同じ実装の二回呼出しを「独立検証」と称しない。criticalなraw decode・CAR/JW符号・scalar台帳・辞書の共有箇所は明示し、checker自身の検査範囲と限界を残す。今回の仕様は形式証明・platform認定ではない。

**Mandatory STOP。** この表は実装済み能力の追加ではなく、後続実装計画の入力である。
