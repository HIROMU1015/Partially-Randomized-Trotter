# 旧新の正確性・保存P0価格照合

[条件とSOURCE](REPORT.md)。固定2 path×Strong/Reference×X/Y＝8 wrapper、全7資源＝56値。旧新両実装で次を独立に検査し、run02_pairedの16実行すべてPASSした。

- 最終native wordの全opcode／wire／clbit／parameter bitを保持するcanonical hash。signed zeroも落とさない。
- 各roundの入力／出力size、logical 1Q置換数、CX相殺数、固定点の反復回数。
- 旧binary64 global phaseのfloat.hex一致、MPFR160 phase ledgerの全record一致。
- T0の全7資源と保存P0価格。新しい価格補正／threshold変更は0。

| Wrapper | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4_reference_X | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| q4_reference_Y | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| q4_symmetric_directional_X | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| q4_symmetric_directional_Y | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| q8_reference_X | 42294 | 14620 | 5068 | 3108 | 11235 | 85157 | 17 |
| q8_reference_Y | 42294 | 14620 | 5068 | 3108 | 11235 | 85157 | 17 |
| q8_symmetric_directional_X | 42316 | 16204 | 10479 | 7996 | 24370 | 86571 | 9 |
| q8_symmetric_directional_Y | 42316 | 16204 | 10479 | 7996 | 24370 | 86571 | 9 |

表の全値は旧＝新＝保存P0。各56値の旧／新／保存値と残差0は[CSV](../../../artifacts/track_d_stream_bridge_fast/2026-11-10/run02_paired/RESOURCE_COMPARISON.csv)に保存した。保存P0値は既存q4/q8のCOMPARISON.jsonから再利用し、新P0やcompiled QPYの再生成はしていない。予測アルゴリズムへ価格補正として与えていない。

人工・技術回帰は新fast7＋旧53＋keyword adapter1＋stream integration2＝最終63件PASS。fastの7 test methodには長さ0〜3の259 native word、全chunk cut、nested cancellation、signed zero／special angle、classical dependency、不正入力、可変frontierの検査が含まれる。これらを別の科学標本数として二重計上しない。

固定54保存人工raw QPYでも、旧新のword／phase／各round／全7資源と保存P0が一致。[人工54結果](../../../artifacts/track_d_stream_bridge_fast/2026-11-10/technical/artificial54.json)。新P0は0。小人工phase期待値の初回terms数不一致は失敗logを保全し、旧optimize_with_ledgerをoracleにした期待値へ修正した。最適化器／adapterの意味や評価閾値は変更していない。

このcorrectnessは旧bridge・固定保存結果の厳密再現である。新しいH4全入力simulation／CAR／observable／reference leakage検証は行っていない。相対χやactual adjointはnative列とphaseに含まれるものを保ち、一般P0／全law／長qの作用・価格保証へ格上げしない。
