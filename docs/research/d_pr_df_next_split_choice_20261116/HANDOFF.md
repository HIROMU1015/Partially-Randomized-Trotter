# 統合担当／次担当へのHANDOFF：PR-DFはLD7を先に提案

**推奨はLD7。比較した最終2案はLD3とLD7。** LD5/9は今回選ばず未評価で残す。LD7は元fragment2..6をLD2 tailからheadへ移し、tail[7,8,9]の領域を一度に調べる。価格が安いと予測した選定ではない。[REPORT](REPORT.md)／[CANDIDATE_COMPARISON](CANDIDATE_COMPARISON.json)／[NEXT_FINITE_PLAN](NEXT_FINITE_PLAN.json)。本案は未採択・未実行。

## 次担当の固定入口

| 用途 | commit / path / selector |
|---|---|
| 元候補差分 | f42f6e9b97372dacb132d278b31044c391e7b921 / docs/research/d_pr_df_ld_split_cost_plan_20261115/CANDIDATES.json / rowsのL_D=3,5,7,9 |
| prefix7 λ/scalar出所 | 7a8bc7f132e166390c943e6b32c92cc6abe1baf4 / artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json / T20_reference_files.H4 |
| その保存context | /home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json / SHA256=8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f / prefixes.7 |
| inputの元型 | 7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75 / docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json / inputs |
| LD1/2 metadata callerの型 | a9abf8de75bf3432fcd714e868e5c14c8884b8f8 / docs/research/pr_rpe_first_signal_menu_20261113/PREPARE_MENU.py / prepare（そのまま実行しない） |
| 元κ/整数化 | 510fbacd01a2d855202de4c045d8ba1972e5ff3d / src/trottertracks/pr_rpe_rmse.py / paper_event_mapper,schedule |
| canonical組立 | 437ba4acdb82708b6fc98e3cd119a2a734f7cf64 / src/trottertracks/pr_rpe_rmse_integration.py / a_spec,generate_from_a_spec |
| signal backend・saved loader | fefd548ecd56f60d72031b3d385f6f5e6ee95d06 / src/trottertracks/pr_rpe_signal_provider.py / load_h4_context,saved_h4_reference,evaluate |
| 13 Full-H reference pair | bba150834ca30bdb186fbfd58c55a8d51177b855 / docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json / rounds[m].diagnostics.reference_pair |
| own-prefix価格backend | e827367629ca6c2920d9d908fc1631314399db86 / src/trottertracks/candidate_price_adapter.py / from_b,NativeRuntime,Adapter（同commit fastと束ねる） |
| 価格caller部品の実績 | ea9e66092465d39dfcd4aeb6fea8e9f09808cabb / scripts/run_pr_df_ld_split.py / 既存LD2の実行入口をそのままLD7へ流用しない |
| exact fit算術 | d83737b6d5241962ce39c81d39c56af3506c0cdd / src/trottertracks/price_model_validation.py / _solve |
| 正の有理数配分 | e094b2d60c5c4468711dea244fa2f91fb587ac12 / scripts/run_pauli_ld2_reallocation_q3_fixed.py / generate（Pauliのcase/価格を流用しない） |
| 実Q3 worker経路 | 005dc8c019061c07ba2ac8da389af0117f7aa27b / scripts/run_pr_df_ld_split_q3_resume.py / 元27e70efe workerをSOURCE固定読込 |
| 既存6点比較 | 8ec7c882806809460d38a638a891bafcf18fafd1 / docs/research/d_pr_df_ld_split_q3_resume_20261116/SUMMARY.json |

全機械可読locatorは[SOURCE_LOCATORS](SOURCE_LOCATORS.json)。context値はλ_R=0.0003129539143720037、**tail identity** s_R=4.5302064712666724e−05。Hamiltonian constant cと足したtotal scalarではない。今回はc+s_Rを新計算せず、後続SOURCE/INPUTSで両quantityを束縛する。

## 先に確認する条件

新しいLD7 configuration/R列/physical IDは未生成・null。 後続では生成用SOURCEとprefix/input/mapper指定をcommit固定してから候補を生成し、生成結果を固定してからsignalへ進む。元H/state/snapshot・元順、one-body別head、S2/.1/K2/元κを保ち、元mapperを使う。**後続metadata生成後、全13 R=1と単一S2 pieceを検査できた場合だけ、以下の価格案へ進む。** 本turnではR=1を数値的に確認していない。条件外なら科学call前STOP・再計画し、κ/R/feature/cellやprefixを無断で変更しない。

LD3/7とも現行13信号・価格/Q3は確認範囲で未同定。旧signalでT.8/q8/Δ.1/S2/K2の部分一致recordはあるが、Rは4/8/16/32/64。新candidate/lawが未定義なので受入0。旧compiled価格はlegacy lowering/別T,q,R,K,公式等で不採用。旧metadata件数はpath数ではない。B0I p7/PR-Pauli LD7の証拠とも混ぜない。

## 一つの段階的有限案

1. **候補＋signal**：後続でPR configuration1・13rowをSOURCE固定し、own raw/corrected/exact-tailを13round取得。保存reference13 pairを再利用。新signal最大13 provider round-evaluation（complex出力最大39、normalizer13）、新Full-H action0、trajectory sampling0。入力/normalizer b>1・norm/phase/closure gate不成立で停止。
2. **同じNで比較**：各軸V0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]` でLD7のQ3を最大1件。旧LD1/LD2 V0は再利用。これが適合する有限anchorを得た場合に価格・独自A/Bへ進む。未適合で全Nが不可能とは結論しない。
3. **own価格**：m0 T.1/q1、m6 T6.4/q64で各normal fresh4 path×X/Y=最大16 compile。座標は保存LD2 scheduleからの継承値であり、新R/ID/seedはown LD7正本から固定する。own full-wrapper C平均に、軸別`C_ref=alpha+beta*q`をexact fit、26正・finite価格を固定。R1ではqR=qなので別q/qR係数は識別不能。LD2係数は移さず、Sは診断に保持してもfull Cへ再加算しない。
4. **価格holdout**：価格凍結後、m3 T.8/q8の独立fresh2 path×X/Y=最大4 compile。軸別2path平均と凍結Pの誤差を記述し、許容差・モデルPASS/FAIL・refitなし。1path欠測へ平均を変更しない。m7..12の価格は依然q64を越える未検証外挿。
5. **自身の配分を選択**：own V0モデル総額の3/4を予算に、元の正の有理数breakpoint A_SHAPE/B_PRICEを各1本、1≤N≤V0で生成。各軸の価格を使い、整数列・実費用・残額をQ3前に固定。uniqueな新Q3最大2。V0/A/Bの最小適合モデル費用を、保存LD1/2の同じ3点protocolの結果と比較。

**全体上限：signal13 round、価格fresh10 path・20 compile、fit2・価格26、配分2、Q3最大3。** 既存LD1/2の6結果と価格・費用は再利用。LD2追加長round compile0、LD3/5/9の新規取得0。全roundの直接compileやCI/厳密E[C]を初回条件にしない。m7..12のモデル依存とI/Q・有限cohortの違いを明記する。

元のsignal/provider・native/from_b・matching fast・新ZZ・P0・測定付きfull controlled wrapper・Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・総scalar一回を維持。prefix7の実入口を人工fixture/fake compilerで確認し、実行code/入力/cohort/seed/出力先を科学call前に固定。`ld12_price_requests.request_for`と旧固定LABELSにはLD7対応がないので、新callerにgeneric from_b経路を束縛する。元P2 fresh以上の作用保証やfull-wrapper statevector PASSを追加しない。

Q3は3.11.0rc1の元AST guardを通る既存venvを優先。実constructor/child/保存の経路を人工検査してからcallする。信号・価格・各LD own signal・decoder/target/loss/閾値を固定し、Bの進行中shot探索を待たない。

## LD2長round・次の判断

既知のm7..12占比99.857098%は保存値のまま。LD2の追加一点がモデルより高い・低い・近いのいずれでも、この新領域を調べる選択はLD7のまま。異常なら技術対応が必要だが、LD3選択の根拠にはならない。**今回の次案ではLD2長round取得を待ち条件にせず後回しにする。** 既存の金額順位を保証したという意味ではない。

将来、own配分でLD7がLD2より低い適合モデル費用なら次の候補をLD9へ、高い/同額ならLD3へする案。これは候補選択を更新する規則で、未評価3/5/9の価格を判定しない。対案LD3を先に調べるなら、安ければLD5へ、局所で悪化/同額ならLD7で別領域へ進む。現時点で各分岐の結果は不明。

## §10.5とSTOP

f42の五つの価格台帳・旧signal inventoryを再利用し、今回は四分割descriptor/hash・近い旧signal candidate metadata・元SOURCE呼出し経路・最近のref/job/output差分だけを追加した。[DELTA_CHECK](DELTA_CHECK.json)。PR-Pauli次分割案a50dce8fと決定論p7のshot探索e7d24c21は別scope。次回は新prefixの同一configuration/physical/law/axis/SOURCE、mean/receipt/cohort/seed、model/N/Q3、進行jobとoutput/cacheだけを差分照合し、同等物は出所付きで再利用、判定不能は保留する。

R1条件外、identity/SOURCE/scalar不一致、signal/gate/保存不良、V0未適合、cal/holdout欠測、rank又はpositive-price不成立で新規投入停止・部分保全。無断retry・seed差替え・追加path・clip・再fit・事後feature変更・閾値変更はしない。A/Bの不適合/不確定は有限結果として残し、追加列へ進まない。

統合ロードマップ本体は統合担当へ渡す。「次分割を未選定／LD2長roundを自動的な先行依存」から「LD7をLD3との比較で先行提案、R1 domainは後続metadataで確認、LD2長roundは選択を変えないので後回し」へ整理できる。正式E[C]/CI/G/winner・全L_D最適性は未認定。

今回の新科学計算は全て0。固定source/raw・m4独立holdout・旧STOPを保全し、計画のcommit/push・remote一致後mandatory STOP。有限案は実行していない。
