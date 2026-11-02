# F1へ渡すものと残条件

B入力bindingは完了、**F1は未承認・未実行**。input-only bundle SHAは`9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0`。private保存済みphysical/catalog／4arm／5lawを再生成せず再利用する。価格map／score mapは空であり、無料を意味しない。

固定P0の4 mode・2粒子・LD1・残差density-square二fragmentの合成model。分子geometry／AO basis／分子DF rankは非該当。identityはfaithful_identity_in_tail。4armのF1用lawはT=1/5、q2、delta=1/10（この一点のみ）、r2、h=1/20、K2。catalogはT=delta=1/20、q=r1、K2。

保存LはSS14／SP58／PS26／PP51、sum L=149。B出力のF1 Eは2,758／195,170／17,602／132,702、合計 **348,232**。component probe/build descriptorは **447**、motif controlled build96、full wrapper64／compile request64、catalog event68／controlled build136。

以下は既存P0台帳と保存Lを対応させた整数descriptorだけ。数値case payload、event、trajectory、IRを列挙していない。unitsの異なる行を合計して「科学call総数」と呼ばない。

| F1 family | 件数descriptor | 単位 |
|---|---:|---|
| F1.H | 4 | logical cases |
| F1.LAW | 348,232 | event records |
| F1.COMPONENT | 447 | builds and corresponding operator checks |
| F1.MOTIF | 96 | builds and corresponding operator checks |
| F1.WRAPPER | 64 | wrapper builds and raw signal checks |
| F1.COMPILE | 64 | compile requests; cache-hit aliases retain all logical records |
| F1.MEAN | 4 | logical cases |
| F1.CATALOG.EVENT | 136 | builds and corresponding operator checks |
| F1.CATALOG.MEAN | 2 | logical cases |
| F1.PRICE.ENUM | 68 | event price records |
| F1.PRICE.DP | 1 | logical cases |
| F1.PRICE.COMPARE | 1 | logical cases |
| F1.PRICE.ARMS | 4 | logical cases |
| F1.SELECTOR | 1 | logical cases |
| F1.EXHAUSTIVE | 1 | logical cases |
| F1.REFERENCE | 506 | precision evaluations; shared exact-identical reference may alias with recorded cache hit |

独立referenceの506は事前登録式2*(4+32+68+149)への整数代入であり、新しいreference計算ではない。68event／各Eの実体は未生成。正本と対応は[F1_COUNT_DESCRIPTORS.json](F1_COUNT_DESCRIPTORS.json)。F1最終payload inventoryはnullのまま。

F2のQ descriptorは **13,674**、U_F2は`UNRESOLVED_UNTIL_IR_BINDING`。F2数値lawやactual circuitを作っていない。

残る条件は、F1に必要な全SOURCE/interface・専用広域environment、literal recipe price provider、必要なbuild範囲と件数、interval/ceil資格・shot count、完全case payloadと参照／cache／output契約の固定。現状態はPRICE_PROVIDER_UNBOUND／UNPRICED_FALLBACK／SHOT_COUNT_UNRESOLVEDを保持する。これらの準備・資格検査は今回開始していない。

次の担当はユーザー。B結果を確認してF1前準備の範囲を判断する。Codexはその明示指示後だけ必要な準備を行い、F1科学実行には別承認を必要とする。F1後もSTOPし、F2はさらに別承認。GPT/Claude詳細レビューを自動依頼しない。
