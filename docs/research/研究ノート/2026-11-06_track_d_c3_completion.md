# Track D C3 q4/q8の完了と中断補完

[補完結果](../track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)／[事前PLAN](../track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md)。同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=δ=.1、固定q4=保存00..03/cut2・q8=00..07/cut4。回路SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535`、補完SOURCE `88f0496349c056d0736f08487889f942c1aa4d37`。raw両q・全8 P0 wrapper postcheck・最終paired/χ照合PASS。X/Y各価格はsymmetric→Referenceでq4のRZ19867→20092・CX7686→6998・depth11116→5583、q8のRZ39610→39602・CX15360→13776・depth22259→10093、qubitsはいずれも9→17。強い対照のq−1境界融合、Reference内部q−2融合とglobal cutの1境界損失を含む。元H厳密ground仮定・準備費除外。run02中断後は保存回路の最後のpostcheckだけを補完し、新compile0・旧結果保全。q/path/cut/融合条件の差を含む固定fixture・local binary64証拠で、全law母平均・長q・N・全RPE優位は未評価。新sampling/ground solve/DF refit/N評価0、mandatory STOP、後続計算・詳細科学レビューへ進まない。

run02は8compile・7postcheck後に中断し原因未確定。利用者の修正指示で別SOURCE・別出力の保存回路補完を行い、旧run01/02を保全した。旧SOURCE/STOPや閾値を変更せず、後続研究へは進まない。次の担当はユーザー／GPT（固定fixture証拠の採否、次の科学範囲の判断）。Codexは公開確認後mandatory STOP。
