# 保存する成果と限界

本書は既存の理論監査・SOURCE・実行結果・完了GPTレビューの要約である。新しい検算・再解析・科学レビューを行わない。[方針決定](decision_record.md)、[終了・再開条件](handoff.md)、[固定provenance索引](evidence_index.json)。

## 理論・認証実装

| 成果 | 保存内容・条件 | 認定しないこと |
|---|---|---|
| RF0 | 共通不変sector上H=LI+G、G=ΣF_j、0≤F_j≤f_jI、Λ=Σf_j。正stage固定順palindromic S2と原点解析的logに対する有限step相対形式保証。η0の定数32を既存監査で追跡 | 任意の負stage高次PF・乱択平均信号への自動移送、独立priorityの確定 |
| RF1 | raw K0 BCH辞書・SOS行和K_SOSを用いる相対係数と、定数512を含む四次有限剰余。RF0との小さい方を使用 | leading係数だけで有限step全体を認証したという主張 |
| EW2 | hΛ<1/4・η<1等の下、条件付きexact ground spectral outputのenergy-window tailをη²W(W+d)/((1−η)d²)で評価。W=U−L、d>0 | trial determinantがgroundである、state preparation/readoutを含む実QPE保証 |
| SPEC-01〜03 | exact binary64 intake・finite/Hermiticity・sector/mode/context、PSD cap、scalar ledger、L/f_j/Λと固定S2のgenerator/checker | 元電子積分・DF誤差・物理model対応の認証 |
| SPEC-04〜07 | determinant U/W、K0辞書とdyadic sqrt上側witness、centered capとgrouped相殺を残すCbar2、RF0/RF1/EW2対強化対照の有理step enclosureと独立checker | 実際の最適step・量子総資源の認定 |

正本は [RF0/RF1監査](../qc_relative_form_finite_step_audit_20261017/mathematical_audit.md)、[EW2・強い対照の監査](../qc_relative_form_ew2_step_audit_20261017/mathematical_audit.md)、[SPEC-04〜07のAPI・独立性](../qc_a2_spec04_07_synthetic_20261018/source_api_and_scope.md)、[Phase A接続・scopeと高速化](../qc_a2_pilot_phase_a_20261018/source_guard_and_resource_design.md)。独立priority・新規性は既存の未確定状態を保持する。

## 固定DF pilotと強い対照

科学SOURCE S `8d483f9bfa5983bb52fef55887e2f9f73a790511`、正式C `a7f8307123f6ba5bbb8d8b84e124400ac04a60ce`、結果payload `817d27b5a9af076b6205c3d54a7b60f646c6ad6f`、最終公開 `c1b80c0fabc1706d69585d993cbecfa12e4d09a6` を固定参照する。

対象はH4／1.00 Å／STO-3G／保存DF rank12、保存binary64 full DFを正確係数とした固定Hamiltonian。N=4・保存8 modes・全12 blocks、one_body→block0〜11の正stage S2、U用trial modes0〜3。PR-tail split/δ探索は行っていない。d=1/1000Ha、α_spec=1/10、sqrt_bits=96、root幅2^-40Ha^-1、max_bisections=256、ENTRYWISE_RE_IM_ABS_SUM_V1を使用した。

| route | 保存結果の十分step境界・概数（Ha^-1） | 保存判定 |
|---|---|---|
| RF0 | 1.2993e-5 | ENCLOSED |
| RF1 | 1.7096e-4 | ENCLOSED |
| EW2 | 1.7096e-4 | ENCLOSED |
| CENTERED_CBAR2 | 1.6858e-2 | ENCLOSED |

独立checker受理、`COUNTER_LARGER_SUFFICIENT_STEP`。約99倍というレビュー上の概数は十分step境界の比較に限る。RF1とEW2は同じmin選択に由来する値で、別々の独立実験ではない。初回science pipeline1回・修正0・retry0。過去人工test134 methods PASSは人工検査であって134分子の再現実験ではない。

[既存結果報告](../qc_a2_phase_b_execution_20261018/results_and_stop.md) と [既存summary](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/saved_results_summary.json) の値を引用する。ここではrawの科学値の再集計・比の再計算・root再評価をしない。現証拠法での実用上の負の結果を、定理の誤り・一般NO-GO・真のPF biasや元分子energyの保証へ拡大しない。

## 証拠取得費用と独立性

| 保存stage | 生成wall秒 | checker wall秒 |
|---|---|---|
| U | 0.002511394 | 1.704099742 |
| K_SOS | 0.029035078 | 0.021150763 |
| 構造付きCbar2 | 218.457223173 | 116.447777066 |

全run約343秒の大部分は構造付きCbar2の取得・検証である。これをA2側だけに課金しない。共通前処理・入出力・cacheを含む単独routeの総費用は別実行しておらず、表から再構成したとも主張しない。query/T/CX/depth/state-prep・合成・readout・償却と古典秒の換算やbreak-evenは未評価。cheap certificate目的への変更は今回未採択。

checkerはbit/CAR/occupation-matrix・trace・正分母不等式などの別実装経路を使い、generator flagを受理証拠にしない。一方、NPZ→raw adapter、schema/hash、Python整数/Fraction、既知数式は共有する。完全に独立した別runtime・形式証明・platform certificate・外部再実験・immutable CIと扱わない。full Fock occupation/Pauli展開は指数的であり、大規模系への安価な適用を実証していない。

## レビュー判断の保存範囲

完了GPTレビュー§§5–7は、同じ認証値を固定したd/α探索や同Λ・同適用域での単独trial/K_SOS改善を現経路の救済策とする根拠がないと判断した。これはレビューの紙上診断として保存し、Codexが新たな定理やcounterfactual certificateを作ったとはしない。鋭いΛ・構造保持の相対係数/有限剰余が得られた後の結果は未計算である。

現行経路への追加投資を見送る一方、理論・実装・負の結果・費用・限界を成果として保持する。論文執筆・新しいpriority調査・改善機構探索は今回実施しない。
