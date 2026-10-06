# 2026-10-06：Track B R1一回実行・保存値監査・GPTへSTOP

利用者が固定source `d43d64a821a0249a0dfab12a2472bd3a72fdee74`でR1を一回実行し、
終了後mandatory STOPする明示指示を出した。source review PASS準備記録P
`09fe527274e8de5c0d808ada5890fa06a59f78ea`は保持し、実行HEADには使わなかった。
Sから独立した`track-b-r1-one-shot-execution-20261006`を作り、authorization JSONとreceiptだけを変えた
直接子A=`411f08f768244fe87b600d82308c3851847fe9e4`を公開した。

clean HEAD=A、source critical26 path、旧証拠66 path、固定runtimeのidentityを確認し、
固定runnerを一回呼び出した。126 synthesis calls、264 rows、132 controlled tasksを完了。
terminalは`R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`。全controlled task適格、failureなし、retry0。
science終了後mandatory STOPし、追加synthesis/target/circuit/scienceは行わなかった。

結果は2-qubit finite polynomial、m=3/K=2、x={1/8,1/4}、σ=±1、三native precisionの登録比較。
primary distinct-basis controlledではB²改善が全対照で残る一方、native/shot資源にはtrade-offがある。
G_Tはordinary比12 groups中LOWER8/HIGHER4、PTSC-K0比LOWER4/HIGHER8。
これを科学GO、最良algorithm、DF改善、I0実取得優位へ読み替えない。

保存field専用auditは126sequence、264row、2,904event、192対照pairのidentity/算術を確認しPASS。
strict matrix guard、native回路、Bernstein shot ceiling、exact signalは再計算しなかった。
原resultとmarkerを保持し、全264row CSV、summary、validation、GPT review依頼を公開する。
source/preregistration、共有API、Track A、過去STOPは変更しない。

現在の入口は[結果照合](../../tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)と
[GPT review依頼](../../tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。
研究方針・RQ・新規性・論文着地点・追加検証の必要性/範囲はGPT側へ戻す。
**mandatory STOP、run1/retry0、追加science未認可。**
