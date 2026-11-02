# N3幅比較v2：新結果のGPT科学判断への引継ぎ

2026-10-13 JST。次の担当はGPT（レビュー開始の利用者承認待ち）。本資料はレビューの論点と証拠の引継ぎであり、科学レビューや研究継続GOを実施したものではない。

## 新たに取得した結果

[一次結果](n3_width_comparison_v2_results_20261013.md)の固定6stress Hamiltonianはfull5/6、ν2/3、一外部mode、非負density、候補2mode/H、ε_proj=.01。
化学geometry/basis、DF rank/policy、L_D、PF窓は適用なし。
NORM・OCC・同値SDP・SOSが全6Hを縮約でき、OCCの追加H縮約0。同じ入力の3timing反復を独立6H以上の入力数に数えない。
有限保存幅では、OCCはcheap ORよりH最小幅5/6・同mode10/12でstrictに狭い。同値SDPより全6Hで狭い保存幅と短いcold medianを取得した。
SOSとのraw同mode幅は6対6、H最小幅はOCC5対SOS1。包含する理想条件と、有限solver/有理修復で取得した証拠を区別する。
NORMは目標.01で同じ採否を、OCCより短いcold取得medianで得た。

## 判断を求める論点

1. 同じ理想OCC条件の同値SDPに対し、今回の有限証拠取得費用の差は研究を継続する根拠になるか。既知Fantope/SOSの特殊化としての位置付け、有限精度・修復の影響を含め、算法差のclaimを判断する。
2. 目標.01で追加H縮約が0の中、保存幅の改善をどのtaskと結び付ける価値があるか。今回の固定保存幅・取得費用で判別できる範囲と、別途必要になる新しいtask/精度契約を分ける。
3. density-aware参照の理想包含と、有限予算でのSOS取得結果をどう評価するか。未知最適幅や一般SOS/RDM族の優劣を今回のsolver出力から確定しない。
4. N3を中心研究候補として継続、対象限定、技術記録として整理、または別方向へ戻すか。今回のstress collection、小系checker費用、U/c取得費用、import/IO/OR計測の限界を含めて判断する。

## 不可欠な資料

- [一次結果とresource/失敗の記述](n3_width_comparison_v2_results_20261013.md)
- [固定数学・baseline・error契約](n3_width_comparison_v2_contract_20261012.md)、[結果前設計](n3_width_comparison_v2_preregistration_20261012.md)、[文献追補](n3_width_comparison_v2_literature_amendment_20261012.md)
- [実行計画](n3_width_comparison_v2_execution_plan_20261012.json)、[利用者実行認可](n3_width_comparison_v2_execution_authorization_20261013.md)
- [全係数入力](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)、[launch](../../artifacts/n3_width_comparison_v2/2026-10-12/run1/launch_receipt.json)、[6H結果summary](../../artifacts/n3_width_comparison_v2/2026-10-12/run1/result.json)
- [個別出力SHA索引](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/run_output_inventory.json)、[独立保存監査](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/independent_saved_audit.json)
- [全集計](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/summary.json)、[504 rows CSV](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/mode_method_records.csv)、[厳密比較区間](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/pairwise_saved_intervals.json)
- [source checker](../../src/trottertracks/representation_exploration/n3_width_check_v2.py)、[generator](../../src/trottertracks/representation_exploration/n3_width_generator_v2.py)、[消費済み認可STOP](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/AUTHORIZATION_CONSUMED_MANDATORY_STOP.json)

GitHubへcommit/pushし、固定URLとremote取得証跡を別publication recordに揃える。local絶対pathだけを公開取得の証拠にしない。
旧primary `NO_ADDITIONAL_ACCEPTANCE_OBSERVED`、旧2-base gate、N1停止、N2保留を保つ。今回の計算認可を追加実行やレビュー開始へ流用しない。

固定GitHub URLとremote取得・保存照合の証跡は[結果公開記録](n3_width_comparison_v2_results_publication_20261013.md)に揃えた。
