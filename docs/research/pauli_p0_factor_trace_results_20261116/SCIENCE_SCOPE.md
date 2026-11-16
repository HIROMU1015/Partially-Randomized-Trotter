# LD7・8・9の保存operator再解析と実factor係数追跡

`49a6d7adeed6c5dd3b2624db1cf7ea397f8e7050`のNEXT_CHECKSに対するユーザーの明示承認範囲。
旧P0を再開せず、独立した診断SOURCEとfresh `run01/`を使用する。

§10.5照合：対象carrier `9fa9525b3`には保存operator/JW/dictionary/metadata/resultが揃う。
LD7/8/9は元full Frobenius gateのSTOPを保持。同じ実factor係数追跡の完了成果は関連Git履歴に
見つからず、B2の公開`133265772`等はLD4/6のnormal intake準備で本診断とは異なる。
PASS LD4/6は既存対照資料だけを使用し、今回の新数値対象には加えない。

保存Linear H4/1Å/STO-3G/tol-only DF10、H/state/snapshot/weighted-Hermitian completionを維持。
P0 `fd111f989`、M1 `5e4f00487`、collector `75fb0ba11`、save-only loader `155529e20`を再利用。
original fragment orderとsigned weights、scalarを保存値のまま使用する。

1. 保存operator再解析をLD9/8/7の各1件：true tail norm、絶対/相対残差、trace identity/traceless分解。
   保存JWから行列を再構成して保存Aと照合。新reference/meanは不要。
2. 保存factorを同じloaderで1回復元。stateは保存配列であり生成しない。
3. LD9の残存factor1個→LD8の2個→LD7の3個の順で各1 trace。
   元defaultの係数は保存JWと全項・順序・値で完全一致を確認する。
   one-body、積、weight、Fermion集積、JW集積のdefaultと局所no-filter shadowを記録する。
   default Fermion residualをno-filter JWへ渡すbridgeも同一trace内で用い、誤差vectorを分解する。
4. original gate 1e-9、global EQ_TOLERANCEを保持。
   defaultの既知のgate超過は診断値として保全し、shadow結果で元P0判定を上書きしない。
   SOURCE・入力・元JWとの不一致、非finite、shadowと保存tailの不整合は部分出力を保存してSTOP。

本番辞書採用、canonical/schedule生成、signal、normalization、価格較正、compile、Q3、N/Gは行わない。
診断callerには段階log・所要時間・例外・終了codeを保存する。大きなevent列や新しい計算frameworkは使わない。

実行前人工テスト：factor trace6、operator分解4、実callerの全3split mock/部分STOP/再開拒否2、計12 PASS。
caller人工検査でbasis fixtureとpath演算順を修正し、科学実行前に再検証した。本番P0の変更なし。
