# 新A V3：実施範囲

基準commit `43bfc5918703e8dcd020be4fcf96bb8dfd99f909`、SERVER 2、2026-11-07 JST。専用branch `codex/pr-reference-rotation-resources-v3-20261107`、独立出力 `artifacts/pr_reference_rotation_v3/2026-11-07/`。

固定条件はLinear H4、1Å、STO-3G、DF12、元H厳密ground供給・準備費除外、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、canonical finite-bit law、保存18round q/r、精度1.5936e-4 Ha、失敗率.05、位相cap.4rad。compiler A/Qiskit1.3.0/RZ-SX-X-CX/全結合、B′4-CX RTE回転、reference F/G・vacuum補正・2-register・8-CX CSWAP、D融合・basis共有を維持した。

登録14-CX baseline、B′、referenceの3方式を比較する。shot scheduleは保存値（8598/8598/8606）を再利用し、誤差・alpha・alias条件を変更しない。

許可範囲は保存local catalogue/raw/QPY・SOURCEの読取、分類・角度key・multiplicityの再解析、既存有理会計の拡張、必要な検査実装と人工test、文書・公開まで。新SCF/DF/ground solve、sampling、回路build/native compile、Clifford+T合成、strict guardの再評価、precision最適化は0。保存QPYは読み込んで角度を集計するだけで、simulationやDAG depth評価は行わない。

分類では解析的SOURCE由来と実nativeのbinary64角を区別する。近さによるClifford/T認定はしない。数値phase・betaでexact symbolic branchが保存されていないもの、trajectory依存補正を解決できないものは未確定に残す。解析的pi由来の型を参考内訳として別記するが、丸め差を消したexact native演算・無償Clifford・実T価格へ読み替えない。

既存SOURCE/raw/result/STOPと他研究の状態を保全する。新しい承認証明・grant管理基盤は作らない。独立14 QPY読取taskを実affinityに応じて並列化し、共有設定・他利用者process/fileを変更しない。

未分類や保証不足を含め結果を保存し、V3の限定範囲完了でmandatory STOP。V5、他の回路・角度分解、T合成、候補探索を自動開始しない。次の科学判断はGPT／ユーザー。
