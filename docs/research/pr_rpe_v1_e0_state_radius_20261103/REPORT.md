# RPE V1：共通E₀・入力状態・信号半径の限定再解析

2026-11-03 JST / LOCAL / 主エージェントのみ。前回preflightの後、ユーザーの明示実行許可を受け、固定保存値のV1再解析を実施した。V2/V3、詳細科学レビュー、中心claimの正式採択は行っていない。

**元DF Hamiltonianの同じE₀をreferenceにした円周位相残差は、Partial／Full各18roundで最終opt2のPF予算内だった。Partialのfinite-RTE位相差と保存phase boundも最終RTE予算内だった。一方、物理H基底状態を入力した場合の厳密な単位PF半径は保証できない。Fullの保存観測半径はモデル下界1を全18点で下回り、7点は既存matrix数値許容差1e-10を超える差を持つ。** この差を消したり許容差を変更したりせず残した。V1全体のenergy解釈は`CONDITIONAL`、energy branchは`UNRESOLVED_BRANCH`である。

## 成果物と範囲

- [V1_ROUND_RESULTS.csv](V1_ROUND_RESULTS.csv)：固定36roundの保存値と再解析値。physical/control、exact/PF/corrected/attenuatedを別項目として保持する。
- [VALIDATION.json](VALIDATION.json)：固定入力・SOURCE参照、metadata identity、tuple、人工テスト、保存値再現、数値summary、操作範囲。
- [reanalyze_saved_signals.py](reanalyze_saved_signals.py) と [人工テスト](test_saved_signal_reanalysis.py)：標準ライブラリだけの保存scalar/complex値用補助処理。

物理条件はlinear H4、隣接1.0 Angstrom、STO-3G、legacy DF rank12、8qubit、保存4電子sector（dimension70）。比較はPartial L_D3とFull L_D12、δ=.02、m0..17、q1..131072。Partialのr/Kは保存最終opt2選択を固定し、Fullはtail-free r0/K0である。別分子・rank・geometry・PF次数・候補・予算を追加していない。

資料は前回調査commit `953b3bbcd05c495ff24d29c7b7e26ea0714907b7` のREPORT、検証案、前提表、round対応表、EVIDENCE_INDEXを使用した。科学入力・元SOURCE carrierは `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。取得commitをclean科学実行SOURCEとみなしていない。[前回REPORT][PRIOR_REPORT]、[V1候補][PRIOR_PLAN]

AGENTS.md、PROJECT_MAP.md、正式共通運用指示2026-10-25の§8.1・§8.4・§10はpreflightで実読した。同じ固定範囲の実行で再承認や承認証明の仕組みは追加していない。既存の採択済み有限coherent-signal研究、Phase0〜3、G01/G02、前回RPE調査とSTOPを保全した。

## 1. 共通energy referenceと入力identity

共通E₀は保存された元DF Hamiltonianの4電子sector基底エネルギー `-2.1663874486347616` Ha。これは新しいenergy推定値ではない。Hamiltonian hashは `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、snapshotファイル参照SHA-256は `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。両者は違う対象のhashである。

ABLGRID、PF3、PF12の元H・E₀・sector metadataが一致し、物理H基底状態fingerprintも `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90` で一致した。ABLJSONとPF3/PF12のsnapshot参照、M06-F初期／拡張computeの記録hashも一致する。NPZの新decode・H生成・対角化は不要で、実行していない。[ABLGRID][ABLGRID]、[PF3][PF3]、[PF12][PF12]

Full18点の診断自体にはstate fingerprint欄がない。元runnerの定義は同snapshot、4電子sector、元Hの最低固有状態を用い、PF12に同一物理状態fingerprintがある。しかし、そのfingerprintをFull診断自身の保存欄として補わない。Full18点の入力同一性はSOURCE/provenance対応を含む条件付き記録であり、独立した状態byteの再確認ではない。[ABLJSON][ABLJSON]、[ablation SOURCE][ABLATIONSRC]、[元runner][ABLRUN]

## 2. 状態・信号の区別

Partialには元H物理基底状態と、PF-step固有vectorのうちphysical overlapが最大のcontrolが別々に保存される。δ=.02のcontrol fingerprintは `25f120a614a34c9924c8d60a2ce5c25a04eb3164096ce95c2c861a82da211f7c`、保存overlapは `.9999999999396312`。これは振幅overlapの保存値で、数学的な状態同一性や新しい確率保証へ変換していない。[ABLGRID states][ABLGRID]

全18選択tupleについて、physicalとcontrolの双方にexact、PF、finite-RTE corrected、attenuated event-meanの4種類がある。control上の元H exact signalは元H基底状態のpure E₀信号とは限らないため、そのまま同一referenceとして代用しない。各信号を個別に共通E₀へ比較した。

PartialのphysicalとcontrolのPF位相の差は、今回の円周比較で最大 `9.787792799697157e-11` rad。高い一致はこの保存条件の数値的支持であり、物理入力がPF固有状態である証明ではない。

Fullのδ=.02のPF有効固有状態controlは主入力内にない。PF3/PF12のcontrolは保存holdout δ=.0125,.025,.05,.1,.2,.4のもので、別δのstate・固有位相を移送しない。Full control比較は`UNRESOLVED`とし、新PF演算子や固有状態を生成していない。

資源モデルの `exact_effective_partial_s2_eigenstate_input_assumed` と `unit_unattenuated_survival_signal_radius_assumed` は、PF有効固有状態・単位基準半径を外部仮定とする。今回主に再解析した物理H基底状態への適用は別の接続問題である。[ACCOUNT][ACCOUNT]、[PF12][PF12]

## 3. 位相再解析の定義・符号・branch

既存SOURCEと同じ `U(t)=exp(-itH)`、`z=<psi|U|psi>`、`t=q*delta`を使った。保存E₀から作る `z_E0=exp(-iE0*t)` は数学的referenceであり、新しい物理simulation信号ではない。各保存zについて `Arg(z*conj(z_E0))` のsigned残差とabsolute残差を求めた。元 `_phase_distance` と同じ円周位相差で、符号を保存する列を追加した。[finite SOURCE][FINITESRC]、[PF SOURCE][PFDELTASRC]

Hadamard sineはSdg→Hで虚部を読む規約。元Hのconstantと抽出identityはstep／反復回路に既に含まれる。追加のscalar減算、identity再加算、位相の二重補正はしていない。[wrapper][WRAPPER]、[step][DFSTEP]、[repeated][REPEATED]

Partialの物理exact signalと共通E₀ referenceの残差は最大 `2.7755575615628933e-16` radで、符号・referenceのsanityと整合した。PF-vs-exact、corrected-vs-PFの位相差と保存半径は、元matrixの`numerical_atol=1e-10`以内で再現した。予算の比較はaccountingの`1e-15`を別欄に保持し、raw excessとallowance込み判定を両方保存する。

**残差はprincipal Argだけであり、unwrapやbranch復元は実行していない。** 円周上で近いことから正しい長round energy branchを認証しない。全36roundのenergy解釈に`UNRESOLVED_BRANCH`を残す。新しい最終energy値や成功率を計算していない。

## 4. 最終opt2予算との数値比較

以下は今回の再解析結果と、元opt2に保存された予算の比較。単位はrad。

| 対象 | 全18round中の最大値（今回） | 保存最終予算 | V1の数値判定 |
|---|---:|---:|---|
| Partial physical PF対共通E₀ | .013988443007835971 | .014050012886294481 | NUMERICALLY_CONSISTENT：全18点 |
| Full physical PF対共通E₀ | .014016007066802567 | .014077830256906622 | NUMERICALLY_CONSISTENT：全18点 |
| Partial physical corrected対PF | 3.247208072224522e-11 | 9.899584668737594e-8 | NUMERICALLY_CONSISTENT：全18点 |
| Partialの保存finite-RTE phase bound | 9.899484670427213e-8 | 9.899584668737594e-8 | NUMERICALLY_CONSISTENT：全18点 |
| Full finite-RTE | NOT_APPLICABLE | 保存0.0 | NOT_APPLICABLE：tail-free |

Fullには元々PF位相誤差の直接保存欄がなかったが、保存された18個の物理PF複素信号と共通E₀から今回再解析できた。新operatorやH生成へ進んでいない。ここでFullの残差は、保存信号に対する新しいV1解析値であり、元artifactに既にあった欄と偽っていない。[Full保存信号][ABLJSON]、[最終予算][OPT2JSON]

Partial controlのPF対共通E₀、corrected対PFも各保存予算内。controlの最大PF残差は `.013988443003100962` rad。physical/control間の違い、元H exact signalとpure E₀ referenceとの違いはCSVの独立欄に残した。

corrected/attenuated信号の共通E₀への残差も各signal recordにある。既知の正normalizationによる減衰とfinite-RTE complex誤差を分けて扱い、normalization補正をPF bias補正として扱わない。Taylor残差・lambda・Bの再生成、PF係数fit、β配分変更はしていない。

この数値整合性は、当該保存信号・指定round・固定予算についての経験的検証である。D6経験係数やPF proxyを厳密上界へ昇格していない。PF3/PF12の `qpe_spectral_diagnostics_pass=false` は元のままVALIDATIONへ保持し、`overall_pass=true`で上書きしない。

## 5. 半径とモデル下界：一致と不一致を別に残す

| 対象 | 今回再解析したPF半径の範囲／差 | 解釈 |
|---|---|---|
| Partial physical PF | [.9999999998289882, 1.000000000072514]、1からの最大絶対差1.7101176030820398e-10 | near-oneの保存数値。厳密な単位下界ではない |
| Partial PF control | 1からの最大絶対差1.810958050185718e-10 | 数値的に1超の値もそのまま保存し、clampしない |
| Full physical PF | [.9999999998284864, .9999999999999536]、最大不足1.7151358111533455e-10 | 全18点でモデル下界1をraw数値では下回る |
| Partial event-mean対保存モデル下界 | 全18点で観測半径がモデル下界以上 | 指定保存点での数値的支持。新しい厳密下界ではない |

Partial最終q131072の観測半径は `.5727013937488484`、モデル下界は保存値 `.5727013919240477`。PF基準半径と、normalization減衰を含む観測半径は同じものではない。corrected、attenuated、保存保守下界、model lower boundを別欄にした。[CSV](V1_ROUND_RESULTS.csv)

Fullでモデル下界1との差がmatrix数値許容差1e-10を超えるのは、m=6,7,8,9,10,11,14、q=64,128,256,512,1024,2048,16384の7点。このモデル半径との数値比較を`INCONSISTENT`として残した。他11点もrawでは1未満である。いずれも「厳密に1以上」と認定していない。

これはsaved-value再現検査の失敗ではない。元のFull半径をそのまま再現した上で、**model radius=1との違い**を見ている。旧WP04の別の実用的判定 `deterministic_unit_radius_model_deviation_below_1e_8` は元どおり通過記録を保持する。1e-10と1e-8の用途を混ぜず、今回の7点をPASSにするための閾値変更はしていない。[旧WP04判定とSOURCE][ABLRUN]

physical PFはH基底状態を使い、modelはPF有効固有状態を仮定している。また保存controlに1超の半径もあり、長いmatrix powerの浮動小数点評価を含む数値である。小さい差をそのまま新しい物理的破綻と断定せず、状態差と数値誤差の寄与はこの保存値だけでは分離できないとする。その分離のために行列・固有状態を再構築していない。

したがって、**元Hのexact ground stateという入力条件からrho_star_lb=1を自動的に正当化できない**。保存最小半径を新しいrigorous lower boundにもしていない。単位半径の保証は`CONDITIONAL`のまま、数値不一致の列は隠さず保持する。

## 6. V1判定と最終energyモデルの解釈

| 接続 | V1判定 | 根拠・制限 |
|---|---|---|
| 保存元Hの共通E₀定義 | NUMERICALLY_CONSISTENT | H/E₀/sector/fingerprintのmetadata一致。Full診断自身のstate fingerprintは未記録 |
| 物理PFの円周位相残差対最終PF予算 | NUMERICALLY_CONSISTENT | 両候補全18点。Fullは今回保存complex値から新解析 |
| Partial finite-RTE actual／保存bound対最終RTE予算 | NUMERICALLY_CONSISTENT | 選択済み全18 tuple。rigorous新保証ではない |
| Fullのモデル単位半径との数値比較 | INCONSISTENT：7点（atol=1e-10） | rawでは全18点が1未満。旧1e-8実用判定とは別 |
| exactな単位reference radius・PF固有状態入力 | CONDITIONAL | 高overlap・near-one・数値PASSは数学的同一性や厳密下界の証明ではない |
| Full δ=.02のPF有効固有状態control | UNRESOLVED | 主入力に対応controlなし。再生成なし |
| Full finite-RTE比較 | NOT_APPLICABLE | tail-free endpoint |
| 長round energy branch／最終成功率 | UNRESOLVED | V1は円周残差のみ。V2を実行していない |

既存18roundモデルは、同じ元DF E₀に対して、今回確認したphysical信号のPF/RTE成分が指定予算内にあるという経験的支持を得た。ただし、元モデルの入力stateと半径の仮定を、そのまま物理H基底状態の厳密な資源保証として扱う接続は未完了である。V1全体を一つの無条件PASSにまとめない。

N・α・β・費用・順位は保存モデルを変更していない。約4.858%点差と区間重複という前回解釈にも新しい再会計・順位付けを加えていない。PRの低費用を本検証の合格条件にしていない。

## 7. V2前に残る本質的問題

最初に明確化すべきは、元Hの物理基底状態を用いるのか、候補ごとのPF有効固有状態を外部利用できるとするのかという**実際の入力状態と、正当な正の信号半径下界の対応**である。今回のnear-one数値と予算内残差は支持証拠だが、単位下界を保証しない。Full controlの同δ記録不足とFull診断の直接fingerprint不足も残る。

その上で、同じ入力・信号・系統誤差条件を、保存最終N/α・alias-free window・branch規則に接続する必要がある。4round成功や旧WP04の長round統計結果は保全し、それを最終opt2の新しい成功率へ代入していない。CA/10は当時の暫定目標のままで、新しい正式energy目標を採択しない。

次担当はユーザー。V1の条件付き解釈・半径の不一致を踏まえて、入力state／半径条件の整理やV2候補を進めるかを判断する。追加の科学的再解析・計算、詳細レビュー、V2/V3は別承認の候補であり、今回開始していない。

## 技術検証・SOURCE再利用・原本保全

人工テストは9件PASS。q/r/K、L_D、state label、signal kindの誤結合、符号、正normalization、principal Arg、欠測/null/ゼロ/非有限値、rawとallowance付き予算比較の分離を確認した。両候補の最初・最後の保存点でsanityを確認してから全36roundへ適用し、保存位相差・半径の再現は全点で1e-10以内だった。

既存finite/PF/ablation/accounting/RPE4/opt2解析の関連関数・validator・6組の既存testを静的確認した。元moduleやtestには行列生成・Qiskit・状態作用等へ入るものがあるためimport／pytestしていない。元phase-distance・radius定義と選択規則を再利用し、今回必要な保存scalar値だけを扱う最小補助コードに限定した。元SOURCEは変更していない。

SOURCE carrier、元dirty実行provenance、元input hashと、新しい補助コード／test hashはVALIDATIONで区別する。旧38 SOURCE-file照合は前回固定資料として再利用し、新しい全raw監査はしていない。今回の入力／必要SOURCE object17件は固定commit/path/blob/size/SHA-256を照合した。

```bash
python3 docs/research/pr_rpe_v1_e0_state_radius_20261103/reanalyze_saved_signals.py --evidence-repo '<repository-containing-the-fixed-objects>' --sanity
python3 docs/research/pr_rpe_v1_e0_state_radius_20261103/reanalyze_saved_signals.py --evidence-repo '<repository-containing-the-fixed-objects>'
```

再現時も自動fetchせず、固定objectがない場合は停止する。標準ライブラリのbinary64 scalar/complex処理であり、保存値の新しいphase/radius評価を含む。再実行は該当する科学実行許可の範囲内で行う。本書はclean CI・rawからの再生成証明ではない。

処理は小さい保存値の列への一回の変換で、状態・行列・samplingの並列workloadはない。CPU/RAMの人工上限は設けず、不要なworker poolやGPU jobを起動していない。共有環境・他ユーザーの設定、process、job、filesystemを操作していない。

原証拠はread-only、成果物は独立Git作業領域の新branch `codex/pr-rpe-v1-e0-state-radius-20261103` に置く。親commitは前回調査953b3bbcd。元worktreeのSOURCE/input/H/state/reference/raw/QPY/grant/ledger/marker/STOP、one-shot/retry契約、Phase0〜3/G01/G02、前回RPE成果物、他branch/worktree・保全資料は変更していない。原repoのcheckout/reset/clean/gc/prune/fetchは行っていない。

公開前に機密情報・file差分と原本状態を確認し、専用branchへcommit/push後にremote SHAと公開本文を照合する。検証後のSHA・全文一致は最終チャットで報告する。内部host/IP・絶対パス・秘密情報・private rawを公開しない。

**Mandatory STOP：公開・照合後にV1を終了する。** branch復元、最終energy値／成功率計算、shot再計算、N×C再会計、α/β再最適化、compile、sampling、Hamiltonian/state/operator生成、詳細科学レビュー、V2/V3、最終資源優位の採択へ移行しない。

[PRIOR_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/953b3bbcd05c495ff24d29c7b7e26ea0714907b7/docs/research/pr_rpe_energy_bridge_readiness_20261103/REPORT.md
[PRIOR_PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/953b3bbcd05c495ff24d29c7b7e26ea0714907b7/docs/research/pr_rpe_energy_bridge_readiness_20261103/MINIMAL_VALIDATION_PLAN.md
[ABLGRID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json
[ABLJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json
[PF3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot/h4_sto3g_d100_rank12_ld3_v5.json
[PF12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot/h4_sto3g_d100_rank12_ld12_v5.json
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[INITPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012/run_provenance.json
[EXTPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/run_provenance.json
[FINITESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/finite_rte_signal_validation.py
[PFDELTASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/pf_delta_validation.py
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_resource_accounting.py
[RPE4SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_four_round_phase_validation.py
[OPT2EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py
[WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_interrogation.py
[DFSTEP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2.py
[REPEATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2_repeated.py
[ABLRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_research_direction_ablation.py
