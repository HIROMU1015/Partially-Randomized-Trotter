# 2026-11-02 P3 Stage 4：条件付き会計準備・未実行

ユーザーはrun04をStage 3価格評価の完了と判断し、Stage 4を推奨するとともに別途科学実行承認と旧STOP維持を指定した。
基点は`e44e87a306d05c8549f1a044b7fd056c2e1055a3`。
[限定実行案](../h4_p3_stage4_conditional_accounting_contract_20261102.md)を作成し、Stage 2/3の保存JSON8件を固定refとbytes照合した。
H4/案A/5精度/M_B・M_C、finite-bit gap=0/compiled ρ=0、nominal-reference直接bias、
保存engineering不確かさに基づくB上界候補、cos/sin共分散を含むN/G/SEを対象とする案。
既存会計関数とscalar丸め照合を再利用し、旧1%guard/rho探索は移植しない。

今回の新bias上界・B上界・N/G・価格再集計・sampling/compileは0。科学コード・artifact・manifest・全STOPを変更しない。
次の担当はユーザー：案の科学範囲を明示許可するか判断。Codexは許可後に関連実装・人工検査・SOURCE固定・限定会計・公開をまとめて行う。
