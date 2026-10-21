# 理論Gに必要な元保存入力

元パスがこの環境にないことをユーザーが確認。以下27候補について、同じHamiltonian/state/DF/preparation hashの
順序付きGaussian basis操作、元対角係数とbinary64生成profile、scalar receiptが必要。
PRはさらに完全なcomponent law・原native order probabilitiesが必要。H6/H8 PRは同じ新探索A n8の軸別shot原表が必要。
候補を推定再生成したり、旧cohortから復元したりしない。

|系|候補|役割|DF rank|科学SOURCE|
|---|---|---|---:|---|
|H2|B2_p1_q1_2nd_r2_K4|PR|3|de72e5d4e48177734c0ddd79dffac14222410658|
|H2|B1_p3_q1_2nd|PF|3|cf99e4452df96291d7e9d37d76d2e91be5cbc413|
|H2|B1_p3_q2_2nd|PF|3|cf99e4452df96291d7e9d37d76d2e91be5cbc413|
|H2|B2_p1_q4_2nd_r2_K4|PR|3|de72e5d4e48177734c0ddd79dffac14222410658|
|H2|B1_p3_q4_2nd|PF|3|cf99e4452df96291d7e9d37d76d2e91be5cbc413|
|H4|B2_p3_q1_2nd_r4_K2|PR|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B0I_p5_q1_2nd|PF|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B2_p3_q2_2nd_r4_K2|PR|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B0_p7_q2_2nd|PF|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B2_p2_q4_2nd_r4_K2|PR|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B0_p7_q4_2nd|PF|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B2_p2_q8_2nd_r2_K4|PR|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H4|B0_p7_q8_2nd|PF|10|84db3909639b34ec521004c3d8f7ef9b5e6ad851|
|H6|B2_p4_q2_2nd_r4_K2|PR|19|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H6|B0I_p7_q1_2nd|PF|19|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H6|B2_p4_q4_2nd_r4_K2|PR|19|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H6|B0_p11_q2_2nd|PF|19|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H6|B0_p11_q4_2nd|PF|19|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H6|B2_p4_q8_2nd_r4_K2|PR|19|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H6|B0_p11_q8_2nd|PF|19|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H8|B2_p8_q2_2nd_r4_K4|PR|28|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H8|B0I_p11_q1_2nd|PF|28|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H8|B2_p8_q4_2nd_r4_K2|PR|28|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H8|B0_p15_q4_2nd|PF|28|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H8|B2_p6_q8_2nd_r4_K4|PR|28|8601b921342fc1cc83780a830e1f65bf64fe9e93|
|H8|B0_p15_q8_2nd|PF|28|19f2c5a9409db69fd8bd6500eb4228fefa00145c|
|H8|B2_p7_q16_2nd_r4_K4|PR|28|8601b921342fc1cc83780a830e1f65bf64fe9e93|

各候補のHamiltonian/state hashと欠測fieldはraw output/theory_tasks.json、各精度のshot保存有無はfigure_data.jsonにある。
これらの入力がない限り、新しい理論Gは取得できない。新しい数学モデルが必要だと判断した結果ではない。
