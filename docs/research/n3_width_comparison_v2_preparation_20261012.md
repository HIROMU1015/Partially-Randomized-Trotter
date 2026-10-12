# N3幅比較 v2：比較実装・未使用6入力・科学実行準備

2026-10-12レビュー系列。[受領した完了済み科学レビュー](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/review_intake.md)第14節を一つの作業として実装した。
現在は **SOURCE_AND_PLAN_PREPARED_SCIENCE_EXECUTION_NOT_AUTHORIZED_MANDATORY_STOP**。
科学batchの明示認可を代行しない。

## 成果物

- [数学・対照・証拠契約](n3_width_comparison_v2_contract_20261012.md)：同値SDP、固定SOS包含／残差、full-H lower、同mode/H endpoint、誤差区間の訂正。
- [固定6入力の比較設計](n3_width_comparison_v2_preregistration_20261012.md)：3family×full5/6、ν2/3、候補2mode、ε_proj=.01、kernel/cold×3 timing repetition。
- [文献追補](n3_width_comparison_v2_literature_amendment_20261012.md)：2020検証付きRDM、2026weighted SOSの対応と未取得範囲。
- [source](../../src/trottertracks/representation_exploration/n3_width_generator_v2.py)、[独立checker](../../src/trottertracks/representation_exploration/n3_width_check_v2.py)、[pipeline](../../src/trottertracks/representation_exploration/n3_width_pipeline_v2.py)、[CAR assembly](../../src/trottertracks/representation_exploration/n3_width_algebra_v2.py)、[contract](../../src/trottertracks/representation_exploration/n3_width_contract_v2.py)、[実行gate](../../src/trottertracks/representation_exploration/n3_width_execution_v2.py)。
- [runner](../../scripts/run_n3_width_comparison_v2.py)、[saved verifier](../../scripts/verify_n3_width_comparison_v2.py)、[literalのみの入力準備](../../scripts/prepare_n3_width_inputs_v2.py)、[tests](../../tests/test_n3_width_comparison_v2.py)。
- [未使用係数pack](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)、[identity](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_identity_audit.json)、[依存pin](../../scripts/n3_width_comparison_v2_requirements.txt)。

## 検証と費用設計

syntheticだけで7methodの証拠を取得・独立検査し、CARの短い全word、SOS包含、最大active5のGram25、singular/縮退、ν1/νmを確認した。
改変・非PSD・float/bool/oracle、cache identity、c強化時の証拠継承、finite unknown、source binding、selector/OR attribution、認可拒否、出力予約を検査。
2 spawn workers・thread1・AS3GiBの実経路とserialの証拠が一致した。速度倍率は主張しない。
登録packをtestで読むと拒否するguardを入れた。新6入力にはU/c、幅、採否、solver/固有値を適用していない。
mode削除時はdensityとmode順序／νも出力し、元H fallbackは削減0。全法に同じcheap ORを与え、そのcoverage集計の費用を独立運用測定と混同しない。

既存venvを変更せず、専用`/tmp/prt-n3-width-comparison-venv`にNumPy1.26.4、SciPy1.14.1、CVXPY1.6.5、SCS3.2.8と依存を固定。
支配workloadはSDP、Gram有理PSDと残差検査。CSCの係数map、universal CAR template、exact matrix SHA cacheを使用。
CPU32／RAM約64GiBを確認。独立Hを2workers、numeric thread1で並列化し、実行前available8GiB・aggregate RSS6GiBを要件とした。
worker AS3GiB、出力・log・call・iteration・bit上限を適用し、wall/CPU time capを設けない。
RSSはboundary／process highwater／sampled aggregate。孤立method peakや瞬間全peakを測定したとは言わない。
有理SOScheckerは小さいν-sectorを再構成するため、一般fillingでの多項式スケーリングや大系効率を認定しない。

## 今回の科学的境界

3 stress families、full5/6→active4/5、固定ν2/3、一候補一外部mode、非負density、ε_proj=.01。
化学geometry/basis、DF rank/policy、L_D、PF時間窓は適用なし。今回のprimary科学結果は未取得。
新sourceのsynthetic数値計算と、新6入力のscience計算0を分ける。原S1の旧族scan／旧batch再実行／N1・N2再開は0。
新しいRDM物理oracle、一般二体、実分子、orbital rotation、多mode逐次削除、compile/shotsを入れていない。
元primary NO_ADDITIONAL_ACCEPTANCE_OBSERVED、旧2-base未達、各STOPと旧source/成果物を保持する。
既存global manifest description欠落FAILを新sourceや科学証拠全体のPASSへ読み替えない。

## 次の担当と認可地点

完了済み科学レビュー第14節の「科学実行の直前には別途ユーザー認可を得る」に従って停止する。
次に必要なのは、固定source/input/planに対する**一回の新比較pilotの明示実行認可**。
同じ比較方針をsource修正・synthetic PASS・公開だけを理由に詳細GPTレビューへ戻さない。
実行認可後はsource/input/planとcapsを維持しattempt1/retry0で実行、独立保存監査、結果公開後にMandatory STOP。
その新結果を受けてGPTが取得効率・taskの価値・中心テーマを判断する。今回中心テーマ・新規性・量子総資源優位は未採択。
