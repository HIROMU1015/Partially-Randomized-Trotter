# H4 S0 → canonical B V2：A/Cへの引渡し

固定A公開`7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`、科学SOURCE`7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`。B基点`af284d853890155bacd19e61850c196bd1d5c660`、controller/integrationは6a9d7fd36から不変。C受動入口`bv2_mean_price.py`は公開`1db4d6e7ebce872c069e337b76632046f60d8b56`とbyte一致で取り込んだ。

## 正本とID

- [CONFIGURATION_V2](run01/CONFIGURATION_V2.json)：S0 `inputs`を既存`generate(I.a_spec(inputs,'PR'))`へ渡し新規生成した全13roundの正本。
- [IDENTITY_MAP](run01/IDENTITY_MAP.json)：旧V1 ID・新V2 ID・A source_request_id・元physical ID・全schedule一致の対応。
- [ORIGINAL_S0_N_HANDOFF](run01/ORIGINAL_S0_N_HANDOFF.json)：Aの元packetを**byte無変更**で保存。V1 ID・科学SOURCE・status・全null・nominalを変更しない。
- [EVIDENCE_PLAN](run01/EVIDENCE_PLAN.json)：既存plannerの26軸に、S0診断、旧HP/価格catalogの存在情報、C受動domain照会を付加した技術view。
- [C_REQUESTS](run01/C_REQUESTS.json)：26件の価格要求metadata。価格値ではない。要求mean、実price law、law距離資格、共分散/CIはnull。

V1 configuration ID：`6cc89360a60b6f5b6554d044193a28f4c92cafa0208e77efb4d1f2ce9f80181c`。

V2 configuration ID：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。

変わるのはschema/ID体系と`split.deterministic_scope`の旧`A_SOURCE_BOUND_PF_BASELINE`→`PREFIX`という明示化。元physical candidate、全T/q/R/event、signed配分、λとSOURCE、H/state/snapshot、prior/decoder、law/profile/error policyは一致した。差分がこれ以外ならadapterは例外を返す。元packetのIDを付け替えた結果をV2正本にはしていない。

κはS0のexact文字列を使用し、近似24.305…から再定義しない。exact値はV2 `spec.A_input.kappa_paper`と`spec.kappa.value`。h/ΔPFもS0のbinary64 exact値を維持する。εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski、finite_B_diagnosticを元policyのまま保存する。

## Codex A：次のmetadata packet

BはAの新V2 N packetを作っていない。plannerへは十分Nを一切供給せず、元V1 packetを診断参照としてだけ接続した。各rowの`S0_diagnostic`はm0/m6のnominal64/38又は未評価nullを表示し、N packetとして使わない。

Aが新V2 packetを公開する際は、上のV2正本から全`identity/law_binding`を使用し、次を保つ。

1. configuration ID、M12/共通κ、全13round・元physical/source_request ID、policyが完全一致すること。
2. nominal Nはm0=64/64、m6=38/38のみ。他11roundはnull/NOT_EVALUATED。
3. 十分N_X/Yは26軸すべてnull、global bound=null、全体/finite-K/signed mapping資格はUNQUALIFIED、status/source/evidence_kindを元S0から変更しないこと。
4. 新packetを作る技術adapterの来歴は別に記録できるが、元科学SOURCEと局所診断を科学的PASSへ変えないこと。

B側の`h4_s0_v2_intake.validate_A_v2_packet(config, supplied_packet, original_packet)`はこの受入条件を検査するだけで、packetを生成・変換しない。新S0十分Nを受け入れる機能ではなく、今後別の科学証拠が供給される場合にはその固定scopeに対応する更新が必要。テスト内のmock incoming packetは保存fixture検査専用で、Aの公開結果として出力していない。

**Aの科学的不足**：11roundのnominal診断、全13roundの十分N・finite-K/normalization・state/ground仮定・law/実装・alias/全decoderを含むglobal RMSE証拠。新科学計算又は保存値再解析は別許可が必要。既存mean/referenceがSOURCE/physical recipe/law/精度まで一致するなら先に再利用可否を確認し、旧HP NをRMSE十分Nとして採用しない。

## Codex C：固定された価格domain

全requestにH4 DF10/元H/state/snapshot、S2/p1/K2、共通κ/M12/h.1、各T/実Δ/q/R/event、signed weights/allocation、law、compiler/control/measurement/full wrapper/P0を保存した。特に：

| round | 元physical ID | qPF | R_outer | RTE event位置 | 必要な軸 |
|---|---|---:|---:|---:|---|
| m0 | `53e00d8b07d1547d19a7` | 1 | 2 | 2 | X/Y |
| m6 | `2a8c6718220eec59d419` | 64 | 128 | 8192 | X/Y |

event数はRTE位置のmetadataでありcompiled gate数や古典標本数ではない。Cが価格法・calibration/holdout/必要標本数・compile上限を決める。Bから新価格、S再計数、校正、fit、予測は行わない。

固定したB/C保存catalogでは、この13候補に同じq/Rの保存Cは存在しない。m0..6は旧118候補内のphysical recipeと一致するが、V2全schedule/law/SOURCEの資格とは別。m7以降はR255/510/1020/2040/4080/8160で旧menu候補を代入できない。C1のH4 q8/q16のR4/8は、今回m3 R16/m4 R32と不一致。有限cohort、MODEL false、元role/SOURCEを維持し、C1/S0aから価格を補完しない。

Cの`V2MeanPriceLookup`へ供給recordゼロで照会した26軸はすべてMISSING/null。これはH4 recipeがmetadataで記述可能というだけで、価格modelがdomain適格になったことではない。m1は固定R4のsaved-only入口だがq2の対応保存値はなく、他Rには固定R4モデルを移さない。新requestの実price law・idealとの距離も未知。新C recordは完全V2 identity/law、明示quantity、DIRECT/MODEL、cohort、domain/資格/SOURCE/uncertaintyを持つ必要がある。

**Cの科学的不足**：全26軸を覆う平均compiled価格と、採る方式に必要な全law・長q/大R・rare/境界・共有校正/共分散の資格。26軸の不足を26新compileという計画へ機械的に変換しない。保存追加catalogを発見した場合も元SOURCE/physical/law/全体contextを照合してから扱い、価格法の有限科学scopeはCが提案しユーザーが判断する。

## 比較・実行境界

今回の実configurationはPRひとつのみ。Full PF/truncated PF/Full-H RTEのcoverageは欠測のまま、B0はtruncated、DF B3は全H純RTEではない。PF formula/幅の比較menuと純RTE physical bridgeは未完成。4方式比較の完了、PRの優位、実G/最適値を表示しない。

plannerの`qualified_expectation`は既存の厳格な不足判定modeとして使い、今回`price_policy_adoption=null`を明示した。有限cohort scenarioの正式採用は別判断。新科学計算の許可やS0の旧許可を拡張していない。

次：**A**がV2診断packetへ技術追従、**C**がこの固定domainの価格検証計画を具体化、**ユーザー**が残るRMSE資格/価格policyと各有限科学scopeを判断・承認。十分N/Cと全体資格が揃い、実G scopeが明示承認された後だけ、**B**が既存cost/selectへ進める。今回は公開照合後**mandatory STOP**。
