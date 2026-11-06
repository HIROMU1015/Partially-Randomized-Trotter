# 新A B′：V1・必要最小限V2の実施範囲

基準commit `da9fceb944dae27f397e48734be30c6864e15f44`。独立branch `codex/pr-reference-strong-baseline-bprime-20261107`、SERVER 2、2026-11-07。ユーザーのV1/V2明示実行許可の範囲で実施する。旧SOURCE、入力、raw、結果、STOPは変更しない。

固定条件はLinear H4 1Å/STO-3G/DF12、元H groundを各shotで供給・準備費除外、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、canonical finite-bit law、18round q=2^mと保存r、精度1.5936e-4 Ha、失敗率.05、位相cap.4rad。Qiskit1.3.0/compiler A・全結合・RZ/SX/X/CX、近似commutation/angle cutoffなし。

変更はRTE occurrenceの中央controlled-RZZだけ。旧builderの出力から当該nodeの実角thetaとcontrol/supportを読み、`CX(a,b); CRZ(theta)(c,b); CX(a,b)`へ置換する。D、scalar、product、basis共有、event順序・符号、X/Y wrapper、reference F/G・位相計算法・16 CSWAPを固定する。

V1-A：保存r6種×order0/2×符号の24角、actual inverse、全3qubit operator（branch別fitなし）、native4CX/2RZ、既存r2 occurrenceの全512次元ordinary作用を確認する。

V1-B：B′の4有限context、初段/最終消去の境界、同一/異なるsupport・CRZ/CZ/Pとの局所境界を確認する。既存64 path（旧48＋16）のevent列だけを再構成し、B′のみX/Y計128 whole compile。旧baseline/referenceの保存価格を再利用する。新sampling・長r追加条件なし。

全supportの相殺集合一致を前提にしない。保護されたCRZ、D/frame/productを残し、全ZZ rotationのparity CXをすべて除外する安全なCX下界と、境界消去を無視する上界を構成する。RZは旧baselineからZZ回転あたり22を除く。reference RZ下界には全product-ZのRZを除外する。これらは比較のための境界であり、下界を実行可能な実価格とは呼ばない。

V1-C/D：同じfloor-CDFと有理確率・basis-run式・既存D会計を再利用。新kernelの解析的厳密性と局所lowering一致を確認後、既存192-bit局所区間を再利用し、B′のraw disk/半径/位相/整数shotを再計算する。同じalpha/alias/精度・成功条件で期待総CX/RZを比較する。恣意的な価値閾値は導入しない。

V2：context・fence・安全なB′下界の補遺、構造定数336/49/24/20の導出とbuilder witness、lambda/実角のbit一致、platform/Python/wheelの記録。既存q4・q1 pathのSOURCE構築だけを追加観測し、V4の長r価格は取得しない。真空位相方式・精度を変更しない。

correctness不成立、安全な価格境界なし、全support成功保証なし、既存保証の本質的前提の破れ、または限定範囲完了でmandatory STOP。V3〜V6、7-CX CSWAP、T合成、別PF/law/gadget、旧研究再開は対象外。共有設定・他者のprocess/fileを変更しない。独立task数と実資源に応じた並列計算を行う。

Claudeレビューは受領原文を `CLAUDE_REVIEW_RECEIVED.md` に保存する。その補題・予測・価値判断を自動採択しない。レビュー中の閾値や7-CX CSWAP提案より、今回のユーザー指示の限定範囲を優先する。
