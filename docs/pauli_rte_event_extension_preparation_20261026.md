# H4/H6 純RTE総event数拡張：実行前の準備

2026-10-26。branch `codex/pauli-rte-event-extension-preparation-20261026`、基点 `288d2206d689a9c24516844cba2100c2d9845a32`。

**準備のみ完了。実H4/H6のlaw生成・状態作用・sampling・compile・新N/G・再ランキングは0。科学実行は未承認で、無効計画のまま停止する。**
添付指示§8は「新しい科学計算の実行承認とは扱わない」「明示的な実行承認を取得すること」と定めている。承認後の結果はまだないため、Hybrid優位の維持・縮小・逆転はいずれも未判定。

## 固定資料・保存入力の確認

次のGitHub固定資料を実際に取得し、内容が指定commitのGit blobと一致することを確認した。

- [H6最新比較 288d2206](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/docs/h6_rte_fairness_20261025.md)
- [Hybrid/RTE比較 90c1c608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3/docs/pauli_hybrid_rte_comparison_20261025.md)
- [H4探索 d3fdfad8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md)
- [H2/H6元研究 8fd1ceb2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/docs/pauli_h2_h6_resource_comparison_20261025.md)

[固定GitHub照合](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/fixed_github_documents.json)にblob/SHA/URLを保存した。H4探索資料・runnerは基点treeにないため、d3固定Git objectと既存snapshotを確認した。別branchの結果を現在のSOURCEと誤認しない。
保存H4は8qubit/184非identity項、H6は12qubit/918項。両者のraw state/full-H referenceを別roleで参照し、shape・ファイルSHAだけを構造照合した。Hamiltonian/state/referenceは生成・再正規化しない。
[baseline CSV](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/saved_baseline_quotes.csv)は、既存の選択済みHybrid/RTE確認行20件を文字列で引用しただけで、新しいmin/G/比/統計を計算していない。H6旧未確定2行・H4旧科学監査の状態・全旧STOPも保持する。

## 実装と再利用

`pauli_rte_event_extension.py` はL_D=0/q=8/K2,4に限定した接続である。旧searchのr<=8 validatorは変更しない。新しい私有namespaceで候補ID検証だけを拡張し、元law_descriptorとCDF/grid mass処理をそのまま使う。r8の人工descriptor/configは旧処理と完全一致した。
PauliSignalAdapter/FixedSignalConfig/RTE sampler本体は正整数rを扱う。step_time=.8/8、RTE短時間はstep_time/r、event数は8rで、元のbinary64計算順序を変えない。q/rが同じsだけで完全に同値とは仮定せず、q=8を固定する。
Kは最大偶数event次数であり、対応するpaired多項式の次数はK+1。S2のPF次数とは区別する。Signed Pauli product、order phase、native constant phase、control、wire配置、cosine/+Im sineを維持する。

- **H4 mean**：d3のFLINT mean-matrix/vector処理を原bytesのsnapshotとして再利用。元64→96→128bit、contraction/L2 recenter・B一回を保持する。r拡張のために古い現branchの遅いPython-vector処理へ戻さない。
- **H6 mean**：既存tight L2 matrix-free処理を再利用。Arb scalar128bit＋binary64中心53bit、符号付きPauli permutation、凸event平均のnorm<=1という仮定はrに依存しない。rを増やした実際のuは承認後に評価し、閾値を緩めない。
- **回路**：既存AuditedPauliSignalAdapter→D2 central-control→full Hadamard wrapperを使い、同じsamplerの一回呼出しで8r eventを得る。元のH4/H6別SeedSequence規約（seed base20261025、探索101/確認202）を維持する。
- **N/G・統計**：既存Fraction上界・任意精度整数ceil、同一shot十分条件・1%guard、paired統計・cohort別min helperを使用する。候補固有B/bias/u/Nを他候補へ転用しない。

数値環境にQiskitを追加せず、lawをcompiler環境で生成して数値環境へ渡す構成を保つ。Compiler環境はQiskit1.3.0/NumPy1.26.4/SciPy1.14.1、数値環境はpython-flint0.9.0/NumPy1.26.4。共有環境・package・他者processを変更していない。
新しいhelperは数値・trajectoryのproduction接続APIを用意したもので、実行中の科学driverではない。承認後はこのkernelから新driver/active batch/SOURCEを固定し、既存output固定CLIや旧STOPを再開しない。

## 人工検証の範囲

12件PASS（新7＋再利用5）。2～3qubitのみ。

- r8の旧config/law一致、r16/32/64/128の有限law/角度/normalization計算の既存adapter対応、KとK+1の区別。
- 旧validatorを拡張していないこと、H4/H6候補領域、元seed規約とcohort分離。
- r128/s1024でもevent数が一致し、bodyが元sampler/adapterの構成と一致。
- 人工compiled回路のglobal/relative phase・control・両軸測定が期待値と一致。
- H4高速Arb接続とH6 L2接続が既存Arb oracleを包含し、共通raw norm/非再正規化/B一回を保持。
- 既存任意精度整数会計の人工境界・guard検査5件。

最初の人工meanテストはArb環境にQiskitがないためimport失敗した。失敗logを保全し、人工lawをcompiler側で作る既存の環境分離へ合わせてPASSした。失敗した実分子callはない。人工PASSを実分子での費用・精度の証拠としない。

## 承認対象の候補と上限

新しい科学条件は固定H4/H6・1.00Å/STO-3G、Pauli-JW、S2/T=.8、q8/L0、同一raw state/full-H reference、central/Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、topology/routing指定なし。ε=.05/.01/.005/.001/.0005、α_axis=.025、状態準備を除くRZ×shot、仮想ρ=0を維持する。

| 対象 | 初期候補 | 条件付き追加 | 有限上限 |
|---|---|---|---|
| H6 | r16/32/64、K2（s128/256/512） | r128/K2（s1024）、有望rのK4 | 8候補 |
| H4 | r16、K2/4（s128） | r32、K2/4（s256） | 4候補 |
| H2・Hybrid | 保存結果のみ | 新しい科学実行なし | 0候補 |

新12候補は既存registryに重複なし。r8は人工互換性検査にだけ用い、実分子の旧成功候補を再実行しない。

- **H6 s1024**：s512がいずれかの精度で算定可能探索min、またはそのminとの差が2 combined-SE以内の場合にだけ追加。
- **H6 K4**：現在最良または2 combined-SE以内のK2の同じrで、bias不適格/幅未確定、bias＋uがε_axisの5%以上、またはHybridとの差が2 combined-SE以内のときだけ追加。5%は候補追加の判断規則であり、shot/誤差guardの変更ではない。最大4つのK4候補。
- **H4 s256**：s128が保存s64より改善/2 combined-SE以内、Hybridとの観測順位変更、またはHybridとの近接がある場合だけ追加。明確な悪化かつ順位不変ならH4を終了。
- 各追加は該当する保存結果と理由、ID/input/SOURCE/callをbatch前に固定。最後まで上限最良なら「最適域未確認」と記録し、全域最適性は主張しない。

| 科学callの範囲 | 初期5候補の上限 | 条件付き追加を含む総上限 |
|---|---:|---:|
| 新law descriptor | 5 | 12 |
| Target状態作用 | 9 | 20 |
| 探索paired trajectory | 40 | 96 |
| 新確認paired trajectory | — | 4608 |
| 新paired trajectory合計 | — | 4704 |
| 両軸compile合計 | — | 9408 |

Target上限はH4各候補3 precision、H6各候補1作用を含むworst caseで、機械的に全precisionを使わない。H4は5精度の判定にWIDTH_AMBIGUITYがなければ早期終了。H6は既存128/53bitの一作用で未確定なら保留し、新allowanceを仮定しない。新Hamiltonian/state/reference生成、H2/Hybrid新科学実行はいずれも0。
探索は各8本。精度別nomineeの和集合を確認前に固定し、極小近傍の判断に必要な隣接s候補だけ追加確認する。最大12候補×384本という上限であり、全候補を最高nへ進める計画ではない。
確認は48→96→192→384。固定Hybridまたは隣接RTEとのG差が2 combined-SE以内、名目t95半幅がGの1%超、かつ新候補側の不確かさが旧比較候補のfloorを上回る場合だけ延長。上限で近接なら未解決。旧成功cohortの再実行・探索とのpoolingは行わない。

## 科学結果の扱い・次の作業

新r>8候補と保存Hybridはq/r/Kが一致しないため、主比較はfamily比較である。新Hybrid r>8は範囲外なので、新しいmatched値はnull/未取得とし、既存matched対照は引用にとどめる。
新しいG・N・順位や図は承認後に取得する。図はlog-s/log-G、B/N対s、必要ならC/N tradeoff。保存Hybridの水平線とcohort/n・名目誤差範囲を明記する。
実ρ、母平均費用tail、大域最適性は未保証のまま。Near-tie・数値未確定・bias不適格・未測定を分け、nullを0/∞にしない。順位逆転も有効な研究成果として扱い、Hybrid優位を採択しない。

[無効の一括実行計画](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/execution_plan_DRAFT_DISABLED.json)、[候補CSV](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/candidate_plan.csv)、[SOURCE/input参照](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/prospective_SOURCE_inputs.json)、[技術確認](../artifacts/pauli_rte_event_extension_preparation/2026-10-26/verification.json)。Future outputは `artifacts/pauli_rte_event_extension/2026-10-26`（未作成）。

**現在はPREPARATION_STOP、execute_enabled=false。** ユーザーが上記一括科学範囲を明示承認した後に、Codexがfresh active contract/driver/SOURCEを固定し、同範囲の計算・必要な独立確認・比較CSV/図/報告・公開・remote照合を連続実施する。通常batchごとの再承認は求めない。
科学条件変更・call上限超過・重大な不整合・共有環境への影響が必要なら停止する。失敗科学callの自動retryはしない。承認証明書・grant発行機構は作らず、旧科学SOURCE/raw/grant/ledger/marker/STOPは保全する。
次の担当は**ユーザー：一括科学実行の承認判断**。**Codex：承認後のみ実行、結果公開・local/remote SHA確認後mandatory STOP**。主要claim採択・詳細レビュー・論文変更へは進まない。
