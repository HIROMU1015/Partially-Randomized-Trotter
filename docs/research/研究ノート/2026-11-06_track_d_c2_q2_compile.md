# Track D C2：固定q2 compileの記録

2026-11-06 JST。[実行前PLAN](../track_d_c2_q2_compile_20261106/PLAN.md)／[結果と保存物](../track_d_c2_q2_compile_20261106/RESULT.md)。

SOURCE `e0befa619a14b956ad64349a86ec2f261962883c`。同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/LD2/S2/s1/r32/K0/h=δ=.1、保存sample00→01・q2・step境界cut。C1 raw bytesを保持し同P0で4 wrapper compile/postcheck PASS。X/Y各価格はsymmetric→ReferenceでRZ10773→11114、CX4100→3860、depth6096→3014、qubits9→17。最大postcheck成分差1.84962e−12。固定一fixture・local binary64 evidenceで、全law母平均・長q・N・全RPE費用の保証ではない。元H厳密ground仮定・準備費除外、sampling/ground solve/DF refit/N評価/retry/resume0、mandatory STOP。

C1のstrong symmetric／Reference rawを再生成せず、actual A†／χと全system入力のobservableを同精度でcompile前後に照合した。q1のevent16 cutとq2のstep境界cutは異なるため、削減幅の変化をq単独へ帰属しない。RZとqubitの増加も保持し、主資源指標・共通energy精度・成功確率は新採択していない。次担当はユーザー／GPT：継続の要否と次の明示許可範囲を判断する。新科学計算・詳細レビューへ進まない。
