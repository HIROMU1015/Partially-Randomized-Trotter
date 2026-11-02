# V3 preflight：研究判断を変え得る最小追加候補

2026-11-03 JST / LOCAL。**提案であり、科学実行契約・parameter freeze・停止判断の変更・実行承認ではない。** 今回は新しい価格解析・compile・samplingを実施していない。V1/V2の物理信号・branch・成功率は再実行しない。保存証拠は[EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)、対象72軸は[PRICE_EVIDENCE_MAP.csv](PRICE_EVIDENCE_MAP.csv)を参照する。

## 判断と提案の順序

現在の条件付き点推定と全保存区間の重なりを報告するだけなら、追加科学計算は不要であり、このpreflightで完了できる。約4.858%の点差から、同じ暫定energy精度・成功率の条件付きcompiled-RZ差をさらに判別したい場合にだけ、下記R1を最初の候補とする。**今すぐ追加compileが不可欠とは結論しない。**

保存後半3roundの寄与率はPartial .9139324437497927、Full .8302589410503727。まず対応するr32／tail-freeの長q移送を判断対象とする根拠はあるが、どの価格を何標本追加すれば費用順位が決まるかは保存資料だけでは不明。r1/q16,32の証拠不足を見つけたことだけで、その局所補完を最優先のcompileへ固定しない。[OPT2JSON][OPT2JSON]

| 提案順 | 候補 | 新しい保存値再解析 | 新compile | 今回の状態 |
|---|---|---|---|---|
| 0 | 現在の条件付き結論・undeterminedを維持 | 不要 | 不要 | 本preflightで完了 |
| 1 | R1：固定scheduleの共有誤差・未保証移送の判別力を調べる | 必要。ユーザー別承認後のみ | 不要 | 最初に提案する限定作業 |
| 2 | R2：判別力がある場合だけ長q構造を狙う限定holdout | 選択とholdout評価に必要 | 条件付きで必要 | R1結果とSTOP変更判断を待つ |
| 3 | R3：r1/q16,32の局所domain補完 | 新holdout評価が必要 | 条件付きで必要 | 比較上の情報価値が確認された場合のみ |

これは保存寄与率と証拠構造に基づく**定性的な提案順**。新しい費用感度・誤差伝播・再ランキングで得た数値順位ではない。R1が低い判別力を示せばR2/R3を行わず終了する。

## R1：保存価格を用いた固定planの依存関係・判別力評価

- **判別したいこと**：費用差の判断を妨げるものが、共有較正のsampling成分、有限holdoutのmodel discrepancy、未知の長q systematicのどれで、どの成分を限定すれば比較主張が変わり得るか。無条件独立や誤差の完全相殺を仮定しない。
- **最終対象**：Partial L_D3/δ.02/K2、保存18round全q/r、特にm15–17のr32/q32768,65536,131072。Full L_D12/δ.02/r0/K0の同じ18q。cosine/sine、policy、compiler、N/α/βを固定する。
- **既存証拠**：q1/2 mean/SE、各r q8、Full q16/32、A0 r32 q16/32、既存費用と寄与率、全scenarioの重なり。q1/2を多roundへ共有し、cos/sinとpolicyがpairedであることはSOURCEで分かる。[OPT2JSON][OPT2JSON]、[A0][A0]、[COMPSRC6][COMPSRC6]
- **不足**：保存最終比較は候補差の共分散modelや長q systematicの実測上界を持たない。aggregateはtask fingerprint索引で、paired個別価格vectorの代わりにはならない。既知のLOCAL initial/fresh32保存先ではtasks/checkpoints/worker_resultsのdirectoryが見つからず、今回rawを復元・他環境探索していない。
- **既存結果だけで判断可能か**：点選好・保存区間重複・追加compile未決定までは可能。最小holdout・必要精度・順位の頑健性を数値的に選ぶには新再解析が必要。
- **最小の新作業候補**：保存価格と固定planだけを用い、共有較正を反映した価格差の不確かさと、未保証のsystematic幅を明示的に残す感度を評価する。詳細paired vectorがなければ共分散はUNKNOWNのままとし、取り得る依存条件を明記する。unknownを0、全round独立、全候補完全相関へ置換しない。新Hamiltonian/state/signal、sampling、compile、再最適化は含めない。
- **再利用**：`AffineAxisCostModel`、`_axis_model`/`_pair_model`、`_measured_model_half_width`/`_comparison`の既存定義と保存roundを基礎にする。必要な差分は固定planの依存成分を分ける薄い解析に限る。`evaluate_coherent_extension_analysis`や`_grid_search`をそのまま実行するとHamiltonian準備・schedule等の再探索を含むので使わない。既存全runnerを無造作にimportしない。[ABLATIONSRC][ABLATIONSRC]、[DECISIONSRC][DECISIONSRC]、[OPT2SRC][OPT2SRC]、[OPT2EXTSRC][OPT2EXTSRC]
- **研究判断への影響**：改善対象が費用比較を判別できるか、単なる局所SE低減にとどまるかを見分け、R2/R3の必要性を絞る。未知systematicの仮定だけで得た分離を科学的優位としない。samplingだけを減らしても未評価の長q幅が支配するならcompileへ進まない。
- **STOP・承認・担当**：これはM06-Fの局所compiler精度向上STOPを変更せず検討できる**新しい保存値再解析候補**。ユーザーが対象と再解析範囲を明示承認した後だけCodexが実施する。今回R1は未実行。R1の完了後も自動compileへ進まない。

## R2：価格モデルの長q構造を狙った限定holdout（条件付き）

- **判別したいこと**：q1/2 affine式の傾き・境界相殺・trajectory長による系統的なずれが、保存短q域を越えた段階で現れるか。単に同じqのSEを小さくする目的とは分ける。
- **対応する最終tuple**：Partialのm15–17 `(32768,32,2),(65536,32,2),(131072,32,2)`、Fullの同q `(q,0,0)`、δ.02、両軸。最大qを直接compileする前提は置かない。
- **既存証拠**：Partial r32 latest fresh q1/2+n32 q8とlegacy n8 q16/32をA0で照合済み。Full同compiler q1/2/16/32、保存holdout RZ残差0。A0のselected q32点残差は約4.911%、標準化残差2.012である。初めから「未検証だったq16/32を再実施する」計画にしない。[A0][A0]、[LEGACY][LEGACY]
- **不足**：r32・tail-freeともq>32の直接価格がない。長qへの一様保証、理論的なcompiler線形則、backend移送の保証もない。
- **候補範囲の考え方**：R1で判別力が確認された場合、例えばq64またはq128のr32／tail-free同条件holdoutを限定候補にできる。これは**r32回路族の橋渡しprobe**であり、最終scheduleのq64/128はPartial r1なので、そのroundの直接価格とは呼ばない。qや標本数・seed・実行回数を今ここで固定しない。必要ならr16等との識別候補もR1で比較する。
- **取得すれば限定できるもの**：選んだ新q・同一tuple・同一compilerでの直接平均とsampling SE、その点でのaffine残差。結果が不適合なら現価格モデルの短qからの移送仮定を反証し得る。
- **取得しても限定できないもの**：q64/128を通過してもq32768〜131072の未知systematicは残る。最大qを含む一様保証へ昇格しない。新holdoutは費用順位を必ず決定するものではない。Fullの保存ゼロ残差も全長qの証明ではない。
- **再利用**：Partialは `_compile_paired_requests`、`compute_full_opt2_randomized_cell`、Fullは `_compile_deterministic_point`。full wrapper・measurement境界・identity phase・policy・trajectory bindingを保持。fitは既存 `fit_affine_holdouts` とA0の固定holdout照合を基礎にし、holdoutをfitへ混ぜない。既存runnerはq/maximum_repetition_countや固定fingerprint/contractを持つため、旧SOURCEは編集せず、選択された条件に必要な差分だけ別版で用意する。[COMPSRC6][COMPSRC6]、[OPT2SRC][OPT2SRC]、[FITSRC][FITSRC]、[LINEAGESRC][LINEAGESRC]
- **STOP変更の科学的理由**：R1が、局所precisionではなく長qのmodel structureを限定することに判断上の価値があると示した場合に限る。M06-Fは区間重複を理由に局所compiler精密化を停止済みで、long-q holdoutは条件付き。新namespaceを作るだけでその停止を解除しない。
- **承認・担当**：ユーザーが研究上の追究価値と既存STOPを変更する範囲を判断し、限定compile・関連sampling/新holdout評価を別途明示承認する。Codexは承認後に必要なSOURCE/入力/試行範囲/保全/STOPを具体化し実施する候補。今回grantや未承認実行契約を作成していない。

## R3：最終r1/q16,32のdomain補完（より低い提案順位）

- **判別したいこと**：選択済みm4/q16/r1/K2、m5/q32/r1/K2の価格が、r1 q1/2 affine式からどれほどずれるか。δ.02、L_D3、selected-policy、両軸、固定compiler。
- **既存証拠**：r1 q1/2較正n8、q8 holdout n8。A0 r32 q16/32はrが違うため直接証拠ではない。
- **不足／判断可能性**：同r1 q16/32のopt2直接価格は固定証拠内で未発見。証拠domainの不足は今回のmetadataで確定できるが、その補完が総費用差へ及ぼす数値的重要度は未評価。
- **新作業・限定できるもの**：必要なら同tupleの新compiled価格と固定fitに対するholdout評価。r1の短q域の解釈を改善するが、主要な後半r32長q問題は解決しない。費用差を判別する目的なら、R1で価値が示されない限り行わない。
- **再利用／STOP／承認**：R2と同じpaired builder・既存affine/holdout照合を使う候補。既存局所精密化STOPを再開する理由が別途必要。ユーザーによる再開判断と新compile/解析の明示承認後だけCodexが行う。今回未実行。

## 今回採用しない追加作業

q1/2の標本を機械的に増やす、51taskを再compileする、全18q・最大q131072を直接取得する、全r/全metricを追加する、別compiler/backend/H12へ広げることは提案の既定路線にしない。保存局所precision gateは既に通過済みで、追加precisionだけで重複区間や未知systematicが解消するとは期待できない。

状態準備、入力状態と単位半径下界、外部alias、実backend/noiseは独立した残存条件。価格検証だけで解決しないため、このV3の限定候補に混ぜない。共通準備Pが相殺しないことと保存break-evenは引用にとどめ、新しい準備費用解析を始めない。[V1REPORT][V1REPORT]、[V2REPORT][V2REPORT]、[OPT2DOC][OPT2DOC]

## 次の担当とmandatory STOP

本preflightは成果物公開とremote本文照合で完了する。**ユーザー**は、条件付き点推定とundeterminedで研究記録を閉じるか、R1の限定保存値再解析を追究するかを判断する。**Codex**は新しい明示指示を受けた範囲だけを担当する。今回の作業から科学的再解析・compile・再較正・費用再会計・詳細科学レビュー・中心claim採択へ自動進行しない。**既存M06-F STOPを維持し、公開照合後mandatory STOP。**

## 固定参照

[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[COMPSRC6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope.py
[DECISIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_decision_cost.py
[FITSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope_extension.py
[LEGACY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json
[LINEAGESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_proxy_lineage_reconciliation.py
[OPT2DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[OPT2EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[OPT2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2.py
[V1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
