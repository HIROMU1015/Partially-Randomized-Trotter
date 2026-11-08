# Complete-step cut選択の実装と人工correctness

2026-11-08。利用者の候補B実装・人工検証指示に基づく。基点は `2131917656446bd22f294946d7c1064a47332e4c`。実H4入力は使用していない。

## 再利用と追加箇所

| 処理 | SOURCEと今回の扱い |
|---|---|
| Whole-path構築 | `track_d_c3_whole_path.prepare_steps` に既存の `cut` を渡すだけ。`physical_segment`、head/mixed-head、step融合、actual inverseを変更しない |
| Strong対照 | `StrongBuilder`、directional center、relative Gaussian、同basis共有、q−1境界融合をそのまま使用。両cutで同一演算子を確認 |
| Reference | `track_d_h4_reference.reference_gadget/control_one` の二層CSWAP、Aのactual inverse、P(+χ)を再利用 |
| 人工系 | `track_d_c7_2_toy.context/event/check/structure` と既存exterior Fock lift。2-mode用のN指定は既存toyと同じ一時patchで、自process内に限る |
| Descriptor | 固定C7-4 `64072ba81bb70db1df1b34e94adb08e78926f24b:src/trottertracks/track_d_c7_4_models.py` の `event_features` をGit bytesからread-only import。旧fitやrunnerは呼ばない |
| P0 | 既存 `c1_h4_native_fair.compile_checked` → MPFR160宣言global-phase compiler。別compilerは実装しない |
| Checker | 既存 `reference_column_check`、`induced_observable`、`expected_observable`、C7-2の成分/observable判定とDAG/max-plus特徴 |

新規は `track_d_cut_selection.py` のdescriptor選択器、`track_d_cut_artificial.py` の固定2人工fixtureへの接続、薄いrunnerと負テスト。旧C3/C5/C8-1やO2のSOURCEは変更していない。旧C7-2 CLIは54固定probeと別originを前提とするため、今回の2ケースをそのCLIへ増設せず、新しい有限runnerに接続した。新gadget、Hamiltonian builder、Gaussian compiler、state simulator、sampling pipelineはない。

## Compile前に固定した規則

正本は [PLAN.json](PLAN.json)。各stepの既知metadataを投影し、C7-4でbasis-run数 `r_i` とsupport work `w_i` を数える。固定負荷は `l_i=1+r_i+w_i`。候補は中央 `floor(q/2)` と左右1境界のうち内部cutのみ。

候補cのscoreは `max(sum(l[:c]),sum(l[c:]))+1`。最後の1は共通のcut損失proxy。同点は中央に近い方、その後小さいc。係数fit、compiled depth、P0 DAG、Born値は入力にしない。特徴計算はone-step counterとして再利用し、C7-4の価格モデルは使わない。scoreは価格式・誤差保証ではない。

`select_cut(ordered_events,event_features)` はdecisionとdescriptor hashを返す。`build_selected(context,steps,decision)` はhash、q、内部近傍cutを照合して既存builderを呼ぶ。未対応/不正descriptorはSTOPであり、都合のよいfallbackはしない。現在のcohortは人工専用で、実H4科学activationではない。

## 演算子同値性

Chronologicalな同じ列を `B=U_{c-1}…U_0`、`A=U_{q-1}…U_c` と切るので、任意の合法cで `AB=V`。既存DF半path条件 `A|v⟩=e^{ia}|v⟩`、`B|v⟩=e^{ib}|v⟩` と `χ=a+b` を維持する。

H後のgadgetの二枝は、共通reference真空を省略すると `e^{ib}A†|ψ⟩/√2` と `e^{i(χ-a)}B|ψ⟩/√2`。したがってX+iYのcoherenceは `e^{i(χ-a-b)}⟨ψ|AB|ψ⟩=⟨ψ|V|ψ⟩`。これは任意の正規化入力に対するideal恒等式であり、A/B可換性は不要。[既存の測定同値性整理](../track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md)に沿う。half-ray条件をwhole-V-rayだけへ緩和していない。

## 固定人工入力と検査

分子geometry/AO basis/分子DF rank/L_Dは非該当。既存C7-2の2-mode・3 native block、frame角(.23,−.37,.41)、scalar .17、extracted identity .04、h=.1、τ=.2を再利用した決め打ちpath。q=6、全step16 event、rotation角atan(.2)、既存plus/minusの符号とframeを保持。分子RTE samplingではない。

| Case | 6 stepのword | 中央→選択 |
|---|---|---|
| run_imbalance | (+−)^8 を3step、+^16を3step | 3→2 |
| balanced_runs | (+−)^8 と(−+)^8を交互に6step | 3→3 |

両fixture、規則、許容差は最初のcompile前に固定。全ての候補cutをcompileして事後minを選んでいない。

- 8 unit/negative tests PASS：近傍選択、中央fallback、短q、descriptor拒否、oracle/価格field非使用、不正cut/hash拒否、H/nonvacuum gate拒否、ancilla/clbit位置と負χ検出。Yは既存Sdg→H、bit0→+1/bit1→−1。
- 110数値比較 PASS：独立exterior-lift基準、ordered V/AB、actual A†、scalar/identity/signed rotationとχ、half vacuum phase、strong invariance、全4 system入力列の全32出力、X/Y observable/complex coherence、P0 full-unitary/global phaseとcompiled leakage。
- 最大成分差 `1.3831338278274048e−13`、最大observable Frobenius差 `7.148334763322419e−14`。既存の成分abs1e−10、observable Frobenius2e−10、rtol0、phase fitなしを変更していない。
- 非可換性 `||AB−BA||_F` は偏りcaseで中央 .178187/選択 .537037、均衡case .351000。raw reference leakageは全検査で0。compiledについても全入力observableと全reference出力成分を確認した。

これは局所binary64数値検証で、全H4 pathへの丸め誤差証明やO2認証engineの更新ではない。

## 技術修正と再現入口

run01は環境metadata取得で、未使用 `threadpoolctl` のdistribution metadataを必須扱いしたため検査・compile前に停止（全compile0）。未導入/未使用と記録する修正だけを行い、同じPLAN・fixture・規則・閾値でrun02を実施した。旧失敗は保全し、環境へのinstallは行っていない。初回失敗時SOURCEは [PRECHECK_HISTORY.json](../../../artifacts/track_d_cut_artificial/2026-11-08/PRECHECK_HISTORY.json) を参照。

人工検証を再現する場合の明示入口（今回は完了済みで、再実行しない）：

```bash
PYTHONDONTWRITEBYTECODE=1 <SOURCE.json記載のPython> scripts/run_track_d_cut_artificial.py --output <存在しない新出力>
```

SOURCE/input hashと依存version、commandは [run02 SOURCE](../../../artifacts/track_d_cut_artificial/2026-11-08/artificial_run02/SOURCE.json)、全人工QPY/構造は同directoryのpayload archive/indexに保存した。
