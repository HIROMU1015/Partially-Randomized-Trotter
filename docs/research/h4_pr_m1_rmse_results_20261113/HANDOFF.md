# A → B：H4 PR m1の保存nominal診断結果

2026-11-13 JST。科学SOURCE **ac873e7fcc42e937084fa153476cbd87565702b4**の既存runnerで、承認済みm1保存再解析を一回完了。**nominal N_X=N_Y=60**、数値gate成立、科学的RMSE資格はUNQUALIFIED。

[REPORT](REPORT.md)／[result](run01/result.json)／[V2 N packet](run01/N_HANDOFF.json)／[SOURCE](SOURCE_INDEX.json)／[packet検査](PACKET_CHECKS.json)。

## identity・SOURCE

- 最新B固定f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8。正本030f3f548…と同一のPR configuration：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
- m1/T=.2/qPF2/R_outer4/8event、physical `c00e00dc1e9c414fb7d9`、S2/prefix1/K2/δ=.1/h=.1/M12。S0 exactκ・H4 DF10/1Å/STO-3G/sector36・保存H/state/snapshot・prior/decoder/law/profile/P0・minkowski/finite_B不変。
- 保存carrier：89c91c692e1d35de94d2e7afbe890aabe6d80b7e、H4 batch `run01/result.json:records[5]`、元科学90dcc3ed2…。
- geometryの数値出典もこのm1 carrier。P1はreference checker関数のSOURCEで、m1の数値carrierではない。旧u/marginはprevious_allowanceとして保全した。

## Bが読むpacket

`diagnostic_scope.newly_reanalyzed_rounds=[1]`、`preserved_S0_rounds=[0,6]`。元S0をcopyしたm0=64/64、m6=38/38に、今回m1=60/60だけを追加。他10roundはnull/NOT_EVALUATED。

全roundの十分N_X/Y・global RMSE boundはnull。status=UNQUALIFIED、全round/finite-K/signed/RMSE契約は未資格。各rowの`diagnostic_source`はm1がac873e7fc、m0/m6が元S0、他roundがnull。`preserved_S0_packet`は元75b5c89b…の公開packet locator。root.sourceは今回packet組立の科学SOURCEで、元S0を今回再解析したという意味ではない。

Bの従来S0-only validatorへm1を通すための値削除・再ラベルはしていない。既存`n_from_a`でcanonical/lawを保つ新scope packetを作った。**B担当がm1追加scopeの受入条件を整えること**。十分Nへ昇格させず、C/費用の欠測を補完しない。

## 数値結果と限界

保存pair差5.551115123e−17、vector closure5.962499244e−16、scalar closure5.578801655e−16、観測exact-tail norm誤差0。旧gateを通過。logB=.0408577586765、attenuation=.959965667024、絶対corrected bias2.680980695e−5、位相−2.680980656e−5 rad、条件付きraw半径下界.959939929255、βdet2.681113531e−5 rad。

u=1.0000000555111513e−9、logB margin1e−12、empirical/certified=false。nominal60/軸は経験的配分であり、十分N/actual minimum/全RMSE保証ではない。actual implementation/law/ground誤差はunknown/null。

新mean/reference/内部対照/normalizer/array load/sampling/decoder simulation/価格/Gは0。reference内の過去action数2を今回の新action数と数えない。35人工/metadata回帰PASS、作者自身の技術確認。

## 次の作業

**B**：新m1 scopeを未資格nominalとして受入。**ユーザー**：残10roundの保存一致診断（m3/m4の証拠補完）・欠測8mean・全decoder RMSEの次の有限scopeを選択/承認。**A**：別承認後にだけ実行。今回公開照合後mandatory STOP。Full PF/価格/G/他roundへ自動移行しない。
