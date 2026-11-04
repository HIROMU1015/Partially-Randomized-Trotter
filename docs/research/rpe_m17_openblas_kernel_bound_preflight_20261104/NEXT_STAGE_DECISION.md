# 現行kernel上界：次段判断

2026-11-04 JST。**CURRENT_KERNEL_BOUND_DERIVED**。
導入済みbinaryの限定経路について、実際のdispatchとFP modeを確認し、保守的な係数を導出した。
これは実H4科学結果や全体PASSではない。

1. **導出できた上界**：C-contiguous complex128 GEMVT/DOT、complex scale/subtract、
   実glibc非FMA hypotの丸め誤差。相対項だけでなくgradual underflowの加法項を保持する。
   reorderingやFMAの有無を保守的な式で覆い、既存KernelBoundsへ接続した。
2. **実環境との対応**：人工実行をgdbで追跡し、Prescott GEMVT/DOT、AVX2/FMA ufunc、
   CPython→hypotとworkerのMXCSRを確認した。C連続70次元GEMVTは出力をthread分担し、
   部分内積のthread間合算ではない。F/stridedや別binary/FP modeは認定範囲外。
3. **全support guard**：有限性とabsの上界を使う判定方法まで整理した。
   実H4のuniform norm/signal/guardは未評価で、成立を確認したわけではない。
   平均誤差だけが得られても完了不明ならU_exec=nullを保持する。
4. **科学実行可能か**：**まだ不可**。既存--runの停止は維持した。
   runtimeのprofile/layout/mode確認、保存context・誤差pass・finite/guard・結果保存をつなぐ統合SOURCEが残る。
5. **不足する証拠・作業**：将来実行processの全参加threadで算術modeが維持されること、
   実contextのshape/strideと全support有限性、guardの数値判定。
   今回のprobeの有限標本正常終了を、その証拠へ格上げしない。
6. **条件変更の必要性**：現時点の推奨はA（現行kernel維持）。B/Cへの変更は不要と判断するが、
   将来の保守的boundで十分条件を認定できるかは不明。条件変更は実行していない。
7. **次の担当と許可**：次段案はCodexによる薄い統合SOURCEと人工検査の完成・公開固定。
   ユーザーはその具体的SOURCEに対し、新しい実H4単一セルの科学実行を別途許可する。
   重要科学レビューも別許可。今回の成果公開後はmandatory STOPし、自動で次段へ進まない。

| 選択肢 | 今回の判断・限界 |
|---|---|
| A：現行kernel | **推奨**。導出した狭いprofileの係数を使い、科学SOURCEへruntime条件とguardを接続する。実H4の実用性は未評価 |
| B：未確認の算術仮定を置く条件付き評価 | 今回は採用しない。別環境・未対応経路で使うなら、実環境保証と区別した研究scopeのユーザー判断が必要 |
| C：別kernel | 今回は実装・切替なし。演算順を固定する利点はあるが別Model IIであり、元実装との差を新たにboundする必要。改善や費用優位は未評価 |

保存科学条件：linear H4/1.00 Å/STO-3G/legacy DF12/B state/4電子sector70、
L_D3/単一δ=.02/m17 q131072/r32 K2、extract_identity_phase/cutoff1e-12、N411と保存α/β/ε。
ideal categoricalの局所十分条件、旧SOURCE/results/status/STOPを変更しない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、旧INCONSISTENT/u_num=null、U_exec=null。

- [実経路・観察・binary対応](KERNEL_EXECUTION_PATH.md)
- [導出式・underflow/overflow・abs/guard](ROUNDOFF_BOUND_DERIVATION.md)
- [接続実装・残作業](IMPLEMENTATION_REPORT.md)
- [人工44検証と保全記録](VALIDATION.txt)

公開branch：`codex/rpe-m17-openblas-kernel-bound-preflight-20261104`。
base：`728c90a2d4d4335ef5b7237808c15b182e694bb9`。
公開commitとremote SHA/tree/成果物照合は最終チャットに記載する。
