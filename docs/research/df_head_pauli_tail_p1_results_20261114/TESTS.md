# P1の技術・数値検査

専用Python `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B` を使用。

| 実行 | 結果 | 確認内容 |
|---|---|---|
| `tests/check_df_head_pauli_tail_p1.py` | 6 PASS | B辞書付きidentity、exactκ保全、固定R/κ区別、full/sector mean、独立raw、同normalizer共有、異なるRの比較抑止、foreign ID/law/SOURCE拒否、normalizer scope失敗の保存 |
| `tests/check_pr_rpe_signal_provider.py` | 10 PASS | 既存非可換/signed PF/K2–6、逐次照合、scalar、独立raw、旧全13人工batch、failure/precision/identity/保存SOURCE経路 |
| `tests/check_pr_rpe_signal_plan.py` | 7 PASS | 既存B metadata/全schedule、qRPE/qPF、source/保存/未資格保持 |
| caller `--preflight` | PASS | P0辞書hash/元binding/file bytes/固定条件のみ。実配列load・新schedule/normalizer/meanなし |
| 承認済みrun01 | 6件の局所数値gateすべて通過 | Pauli commonκ4件＋LD1/m0 fixed-R1＋同full256 1。normalizer5unique、保存DF4row受入、2同条件対照PASS |

計23技術test PASS。新6件は人工3mode/1粒子のcomplex非可換factorを使い、`np.load`とnominal N関数を禁止した。旧回帰も人工/mock入力であり実H4の新計算ではない。既存ファイルのDeprecationWarningはそのまま保持し、無関係なsourceを変更していない。

技術テストとmetadata preflightの後に05447d1d4へcommitし、P1科学run01を一回実行した。実行時間6.0985秒、retry/自動修正/fallback/再正規化/閾値変更0。head/exact-tail/power/norm/closure/raw rangeのgate1e−9、reference pair1e−10、state1e−12を維持した。

full256/36のexact/corrected/raw差は2.2888e−16/2.7756e−17/3.4332e−16、normalization差は0。DF同Rのexact/corrected差0/2.7756e−17、raw差は辞書依存として保存した。Rが異なる4共通κ候補ではDFとの数値差を算出していない。

これは著者の人工・局所数値検査であり、独立科学レビュー・外部再現・全RPEのRMSE/十分N保証ではない。
