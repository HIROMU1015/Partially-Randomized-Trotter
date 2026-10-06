# Track B scripts

## R0 fixed exact symbolic checks

[check_rte_reallocation_symbolic.py](check_rte_reallocation_symbolic.py): standalone stdlib Fraction/free-word checker, A80/B9 fixtures and negative witnesses, 15s CPU/256 MiB AS/30s wall cap. [R0 packet](../../../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md). No sampling/science/circuit operations.


[BS-0.5設計監査](../../../docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)はdocs-only。
新runner/source/testsなし、old science replayなし、mandatory STOP。次scopeの判断はGPTへ戻す。

[SP-1後block合成仕様](../../../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はAPI設計と未承認pilot案のみ。
新script/科学runner/testsは作成・実行していない。既存SP-1/0.5はconsumed、追加run未認可。
RUN_READY=false、mandatory STOP、具体domain/辞書/対照/会計はGPT review待ち。以下は既存履歴。

SP-1の[一回結果とGPT handoff](../../../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)を公開。
science run1／retry0、mandatory STOP、marker consumed。旧runnerは再実行しない。
[audit_sp1_saved_result.py](audit_sp1_saved_result.py)はstdlibだけで保存identity/field/predicateを照合する。
matrix/guard/coefficients/Bernstein/resourceを再計算せず、source/runnerのimport・新合成は0。
[保存audit／evidence manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)、PASS。
追加scienceは未認可。以下のpending/preparation記述は結果前履歴として保持する。

[SP-1 runner](run_sp1_wrapper_pilot.py)は[採用契約](../../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)用。
`plan --source-commit <full S>`はsource-bound静的ledgerだけで、coefficient/resource/signal採点を行わない。
`run`は別review後の新S→direct authorization-only Aと明示指示を要求する。現在はpendingで拒否する。
登録science sweep0、59 focused testsはlocal pass。全結果STOP、次stage自動認可なし。
旧SP-0.5一回markerを再利用しない。

[SP-0.5 one-shot結果・GPT handoff](../../../docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)は
PRIMITIVE_TRADEOFF_EXISTS、mandatory STOP。[audit_sp05_saved_result.py](audit_sp05_saved_result.py)は
sourceのalgorithmをimportせず保存fieldを照合する。既存auditは同一性を確認して保持し、未保存時だけfresh出力を作る。
23 keys／16 rowsの科学計算は完了。旧run／PAI／Jの再評価やwrapper pilotを実行しない。


[SP-0.5 runner](run_sp05_synthesis_economics.py)はplan（合成0）と、別承認後のみのrunを分離する。
[結果前source review](../../../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)、
[contract／preparation](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/)。
pygridsynth一つ、catalogue一つ、8 target／23 keys。登録計測未実行。全outcomeでSTOP、retry0。


BM-0.5の限定記号監査は[audit_bm05_symbolic_equivalence.py](audit_bm05_symbolic_equivalence.py)。
standard-library Fractionと抽象非可換wordsだけを使い、degree3までの三経路を比較する。
physical inputs、行列、Hamiltonian、state、science provider、circuitは扱わない。
stdoutのJSONが[保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json)。
本文・scopeは[BM-0.5 packet](../../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
このscriptの追加はscience runnerの実行承認ではない。

既存BF science/recovery runnerの契約・result・STOPは
[Track B index](../../../docs/tracks/algorithm_codesign/README.md)を参照する。
過去のone-shot markerやauthorizationをBMへ流用しない。
