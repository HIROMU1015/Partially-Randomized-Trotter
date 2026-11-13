# 診断後の判断（科学実行前版）

結果を取得してから更新する。現在、scalar不一致の原因候補と修正提案を区別し、guard・expected scalar・元演算順序を変更しない。

A descriptorのtail scalarとDのtotal scalarが別quantityである場合、source-boundなHamiltonian constantとtail identity係数を区別して接続する最小変更を提案する。診断の新値を自動採用せず、歴史的request/descriptor/STOPを書換えない。

ユーザーが結果を確認し、必要な技術修正・契約の訂正・T2再実行を別途判断する。今回の診断は1回でSTOPし、sampling/compile/価格/Gへ進まない。
