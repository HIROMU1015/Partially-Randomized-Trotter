# H2・H4・H6：Hybrid PR対全ランダム化RTEの比較準備

2026-10-25。専用branch `codex/pauli-hybrid-rte-preparation-20261025`。
基点は最新統合 `e2b9215fd7432e7406ef8c7b8b5f55fba563a696`。

**保存証拠の棚卸しと、一括実行範囲の準備まで完了した。実分子データからの新しい方式別min・比・順位、sampling、compileは未実行。**
今回の添付指示§8は「準備が整ったら」「一度だけ明示的な実行承認を取得」と指定する。
以下の準備は実行承認を証明する仕組みではなく、ユーザーが対象・負荷・限界を判断するための科学的な範囲定義である。

## 1. 保存資料の入口

- [候補・精度・cohortごとの棚卸しCSV](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/candidate_evidence_inventory.csv)：1580行、306登録PR候補。
- [15条件の方式別証拠充足性](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/coverage_by_precision.json)：保存適格性・G有無の件数とID。Gの大小を計算していない。
- [固定GitHub出典・ローカル参照](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/source_manifest.json)、[archive回収元](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/archive_members.json)。
- [再利用SOURCE・入力6件・runtime参照](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/prospective_SOURCE_inputs.json)。
- [無効の一括実行計画](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/execution_plan_DRAFT_DISABLED.json)。
- [棚卸し照合](../artifacts/pauli_hybrid_rte_preparation/2026-10-25/inventory_verification.json)。

対象はlinear H2/H4/H6、隣接1.00 Å、STO-3G、各保存Hamiltonian/raw ray/full-H reference、S2、T=.8。
Pauli central-rotation、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topology/layout/routing指定なし。
full Hadamard wrapperのRZを価格とし、状態準備費用を除く。各候補の保存B・bias・allowance・Nを保持する。
主会計は仮想ρ=0の `G=N_cos*C_cos+N_sin*C_sin`。今回この式を実データで新しく評価していない。

## 2. 取得済み候補と独立確認

| 分子 | M | RTE登録 L_D=0 | Hybrid登録 0<L_D<M | 保存独立確認 |
|---|---|---|---|---|
| H2 | 14 | 13 | 51 | RTE1候補・Hybrid1候補、各96 paired trajectory |
| H4 | 184 | 13 | 128 | RTE2候補、各384；Hybrid2候補、各48 |
| H6 | 918 | 7 | 94 | Hybrid4候補、各48；RTE0候補 |

L_D=0でもconstant phaseを保持する。L_D=Mは全項保持で非自明なランダム化tailがなく、今回の登録PRの2群には含まれない。
H4 Hybrid128には旧p027/q8/r4/K4のhistorical baselineが1件含まれる。
H4の実explorationは140候補（RTE13＋Hybrid127）。旧baselineの256標本は別cohortで保持し、探索8本や独立確認へ改名・poolしない。

H2保存確認ID：`PR_p0000_q1_r2_K4` と `PR_p0001_q1_r4_K4`。
H4保存確認ID：`p000_q8_r8_K2`、`p000_q8_r8_K4`、`p036_q2_r8_K2`、`p036_q2_r8_K4`。
H6保存確認ID：`PR_p0684_q1_r8_K4`、`PR_p0684_q1_r8_K2`、`PR_p0078_q8_r8_K2`、`PR_p0078_q8_r8_K4`。
候補IDと保存標本数の引用であり、方式別の新しい最良候補を選んだものではない。

### 精度ごとの確認欠測

| 分子 | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---|---|---|---|---|
| H2 | 両方式の適格確認あり | 同左 | 同左 | 同左 | 同左 |
| H4 | 両方式の適格確認あり | 同左 | 同左 | Hybrid確認候補2件ともbias不適格 | 同左 |
| H6 | RTE確認なし | 同左 | 同左 | 同左 | 同左 |

H4の厳しい2精度には未確認の適格Hybrid登録候補が存在する。
既存のp36の標本数を増やしても、その保存bias不適格は解消しないため、別の登録候補を探索保存値から選ぶ必要がある。
H2は保存確認の両方式を再利用すればよい。H6ではRTE探索結果を確認済みの値として表示しない。

H4は探索nominee固定後の独立確認を主表とし、確認後の最小候補を別に保存している。
H2/H6の旧主表は凍結確認集合内の観測最小で、確認後選抜を含む。
今後の方式別確認minも事後選抜として明示し、元H4 primary nomineeの主表を上書きしない。

## 3. 数値不適格・会計domain・価格欠測の区別

H6 RTE7候補の保存状態には、bias適格でもG/Nがnullの行がある。
詳細candidate_resultsの各軸 `rho_zero.reason` には `SAFE_INTEGER_LIMIT`、最厳条件の一例には `FIXED_GUARD_FAILED` が保存されている。
例えば `PR_p0000_q1_r4_K4` はε=.01/.005/.001でSAFE_INTEGER_LIMIT、ε=.0005で数値未確定のラベルを持つ。
これは回路単価を0とする根拠や、科学的な不適格・無限費用の判定ではない。
本準備では保存B exactやN、G、reasonを転記しただけで、Nの上限変更・新scalar下界・比・順位は計算していない。

H4 CSVは欠測を文字列`null`、H2/H6 CSVは空セルとして保存する。
両方を同じ欠測として接続し、正規化前の元ファイルは変更していない。
保存値がある項目だけを収録し、H4 catalogのB decimalを新たにexact分数へ変換して補完する処理も行っていない。
棚卸しには価格SEとG SEを別列にし、探索・確認・歴史的baselineを分離した。

## 4. 承認後に行う保存値再解析

1. molecule/ε/group/cohortを固定し、適格かつ保存Gが有限の行だけで方式別minを求める。exact分数と全tie IDを保持する。
2. G欠測・数値未確定・SAFE_INTEGER_LIMITのIDは別に残す。計算可能な部分集合のminを全登録候補の確定minと呼ばない。
3. `R=G_Hybrid/G_RTE` を同一分子・精度・cohortで計算する。一方が欠測・不適格なら比をnullにする。
4. 探索min、確認済み集合の事後min、旧H4 primary nomineeを別に表示する。
5. H4歴史的baselineは別の参照列とし、実exploration140候補のminへ混入させない。登録141全体との関係を明記する。
6. 保存最良決定論的PFを第三基準として引用し、代表候補の保存B/bias/N/価格/Gを並べる。旧PFを再計算しない。

前記min・比はまだ実分子データに適用していない。
新helper [pauli_hybrid_rte_comparison.py](../src/trotterlib/pauli_hybrid_rte_comparison.py)は人工データだけで確認した。
旧 `finite_minimum` はH4 CSVにないguard rangeまで要求するため、その値を作って埋めず、exact Fraction・tie保持の規約だけを小さいhelperへ再利用した。

## 5. 不足確認の最小範囲とcall上限

追加候補は承認後の探索保存Gだけで選ぶ。未確認group/εの方式別nomineeを各1件選択し、同一IDを重複除去したunionを標本取得前に凍結する。
候補IDを今回の無承認ランキングで先取りしていない。登録済みq/r/K/prefixを追加探索しない。

- H2：新sampling 0。既存2候補の確認を再利用。
- H4：ε=.001/.0005のHybrid側だけ、最大2 distinct候補。
- H6：5精度のRTE側だけ、最大5 distinct候補。
- 合計最大7候補。新target/reference/Hamiltonian/state/Arb状態作用は0。

各追加候補は48 paired trajectoryから開始し、必要な場合だけ総数96→192→384へ進む。
48は既存H6・H4 Hybrid確認と比較できる初期規模である。
保存価格SEは利用可能であり、承認後に保存G差・分散を使って固定した延長規則を適用する。準備中に新しい標本数推定は行っていない。

延長は、Hybrid/RTEの観測G差がcombined SEの2倍以内かつ新候補の名目t95半幅がGの1%超の場合だけとする。
既存比較候補の不確かさが支配して新候補の追加では解けない場合、成功済み旧cohortを再実行せずnear-tie/未解決で停止する。
上限は新trajectory最大2688、両軸compile最大5376。これは科学call範囲の上限であり、CPU/RAMの制限ではない。
実際の候補重複、保存証拠の再利用、早期終了により減らす。
これらの名目SE/逐次停止は厳密な選抜後同時信頼保証・母平均winner保証ではない。

元seed規約を維持し、未確認candidate IDのconfirmation tag202・未使用indexだけを使う。
H4は元の `[20261025,3101,L_D,q,r,K,202,index]`、H6は `[20261025,6,L_D,q,r,K,202,index]`。
過去seed・価格recordとの衝突をdispatch前に確認する。探索/確認のpoolingは行わない。
新確認価格のGは同じtrajectoryのcos/sinを組み合わせ、保存Nで重み付けする。軸SEを独立と仮定して加算しない。
旧N、B、bias、allowanceは候補固有の保存値を再利用し、shot法則や数値domainを変更しない。

H6の無意味な大量compileを避けるため、既に正当化された保存下界がある場合だけ事前screenに使う。
観測最小や探索平均を母平均価格の下界へ昇格させない。そのような下界がなければ、固定nomineeを小さい確認cohortで評価するか、確認不足と表示する。
保存domain問題を解くための新しいN実装・高精度状態計算は、この提案範囲に含めない。

## 6. 再利用SOURCE、入力、科学batch

固定結果はGitHubの次のrefから実際に取得した。

- 統合：`e2b9215fd7432e7406ef8c7b8b5f55fba563a696`（報告、15行CSV）。
- H4：`d3fdfad880902b7ff35b710d22e1c9759f7087c2`（報告、catalog、precision CSV、nominations、best、入力参照）。
- H2/H6：`8fd1ceb283309a7e515ff4a533a253260bdd0f78`（報告、batch catalog、precision CSV、各final_results）。

既存Git blobの一致を確認して、同じSHAの既存snapshotを再利用した。
H2/H6のconfirmation plan/resultsとH2 extension planは、保存archiveのpart/archive/member SHAを照合し、新namespaceへJSON5件だけを回収した。
既存trajectory価格recordの場所・保存標本SE・cohortを参照し、価格列を再集計していない。QPYの再decode・再compileも行っていない。

再利用先は、`h4_pauli_pr_search.central_trajectory`、Pauli adapter、D2 control、Hadamard wrapper、既存compiler、H2/H6 process worker・paired統計、`h4_conditional_cng.paired/cost_summary`。
元law・normalization・bias/referenceのnumerical kernelを作り直さない。
古いrunnerは旧出力へ固定されているため、そのentry pointやSTOP済みbatchを再起動しない。
承認後の最小driverは再利用kernelを新namespaceへ接続し、候補ID・保存target/N selector・driver SOURCE・runtime・seed/index・call範囲をsampling前に固定する。
このdriver接続は固定科学条件内の技術実装であり、batchごとにユーザー承認を求め直さない。

入力model/reference6件、再利用kernelのSHAと準備helperを `prospective_SOURCE_inputs.json` に固定した。
対象外worktreeのHEADや共有環境を凍結しない。CPU/RAM/worker数の人工上限は置かず、利用可能資源に応じて独立native workerを並列化する。
OS・共有venv・他ユーザーのprocess/fileは変更しない。

## 7. 技術確認、保全、現在の停止位置

棚卸し1580行のID/ε/cohort一意性、H2/H6正本の保存G/N/B/bias文字列、H4正本の文字列/null、固定入力SHAを照合した。
人工7件（null表記、cohort分離、欠測を順位へ変換しない、exact tie、不適格除外、比の方向/混在拒否）はPASS。
これらは著者Codexの技術検証であり、独立科学レビューではない。
旧SOURCE・raw・結果・grant・ledger・marker・STOP・旧統合CSVに変更はない。

**新科学call 0、新実データmin/ratio/ranking 0。実行計画は無効のまま。**
今回の段階では方式別優劣やハイブリッド化の有用性を新しく判定していない。
仮想ρ=0、有限価格標本、母平均tail未保証、大域最適性未証明、選抜方式差・分子別state/referenceを保持する。
R2-C証明、正式certificate、重要claim採択、論文原稿、GPT詳細科学レビューへ移行しない。

次の担当はユーザー：保存値の新方式別再解析と上記不足確認を合わせた範囲を一度だけ明示承認する。
Codexは承認後、その固定範囲内の実装・比較・確認・検証・公開まで連続して行い、結果保存/remote照合後にmandatory STOPする。
