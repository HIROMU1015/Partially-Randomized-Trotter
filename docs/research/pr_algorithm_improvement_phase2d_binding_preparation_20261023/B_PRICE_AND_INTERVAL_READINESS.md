# Price・intervalの境界：basic Bでは未確定を保持

| 取得内容 | basic Bでの扱い |
|---|---|
| SOURCE上のliteral grammar、run full/support、phase/D融合policy、compiler候補 | 静的policy identityとして固定可能 |
| exact coefficient／lambda／L／canonical lawとbase angle | 11caseの範囲で将来binding可能 |
| basis full/supportのactual entry/exit recipe price | gate recipe生成が必要なためPRICE_PROVIDER_UNBOUND |
| Qiskit circuitをbuildして得るliteral RZ数 | Bでは禁止 |
| compiled RZ/CX/depth | Bでは禁止、将来F2等の別契約 |
| scalar interval B/N/Phi/Jの資格 | Bでは呼ばずUNPRICED_FALLBACK／SHOT_COUNT_UNRESOLVED |

I pricing.recipe_pricesはBasisProvider.basisを呼んでGaussian/Pauli回路を作る。この経路をBから起動しない。未取得full/singleton priceを0にせずnull/空mapと明示statusにする。空literal_prices mapは「無料の回路」ではない。F1承認前にrecipe provider bindingと必要callを別途具体化する。

Iのintervals.pyはmpmath scalar API、sqrt/log、endpoint、ceilを使う。basic Bにはmpmath import／qualification／shot count callを**0件**と固定する。入力bindingに不要なので、5 CONTEXTSへ資格probeを潜ませない。F1で認証付きselectorを使う前に、source/envを固定したsqrt/log enclosure、endpoint rounding、finite-bit接続、ceil境界、score範囲の検査を別契約へ登録する必要がある。

資格がない場合はUNPRICED_FALLBACKを保持する。N式のceil区間が不確定ならSHOT_COUNT_UNRESOLVEDでSTOP。通常の二precision一致を厳密interval certificateの代用にしない。CERTIFIED_PRICE／CERTIFIED_SHOT_COUNTを発行しない。

基本Bはprice確定を成功条件にしないので、build追加を不可欠とは判断しない。本契約はB_SCOPE_CHANGE_REQUIRES_APPROVALを現在発生させていない。もし今後B内で価格を完結する方針へ変えるなら、recipe対象、build件数、出力、意味論、旧Bとの差分を提示して**別明示承認**を受ける必要がある。今回の契約で追加buildを承認したとは扱わない。

truncation toleranceは既存residual metadataからのAPI用nextafter規則で、price/interval資格やsignal bias allowance .025を認証する値ではない。F2 K0についてもshot数・law・biasをF1から流用しない。
