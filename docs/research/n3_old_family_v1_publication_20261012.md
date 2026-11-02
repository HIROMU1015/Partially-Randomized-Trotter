# N3旧条件族診断 v1：source・plan公開とremote確認

2026-10-12 JST。完了済み科学レビューの限定方針を実装し、専用HIROMU1015 branch `n3-old-family-v1-preparation-20261012`へ公開した。
**準備完了、新S1診断0、別の明示実行指示待ち、Mandatory STOP。** 同内容の科学レビューを再要求しない。

- frozen source13 `1aa12ab0b777b28f1428d9bb8417af93c4376aae`
- source fingerprint `46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488`
- immutable preparation payload `23bf77c6cfffe798d54e5340d59ef5be26e9e9ae`
- plan SHA `ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5`
- input fingerprint `81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d`
- [実装・独立検証・準備報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/docs/research/n3_old_family_v1_preparation_20261012.md)
- [数学・比較・resource契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/docs/research/n3_old_family_v1_mathematical_contract_20261012.md)
- [固定一回execution plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/docs/research/n3_old_family_v1_execution_plan_20261012.json)
- [入力索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/artifacts/n3_old_family_v1/2026-10-12/inputs/index.json)
- [authorization template（false）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/artifacts/n3_old_family_v1/2026-10-12/preparation/authorization_template.json)
- [準備STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/artifacts/n3_old_family_v1/2026-10-12/preparation/MANDATORY_STOP.json)
- [分割inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23bf77c6cfffe798d54e5340d59ef5be26e9e9ae/artifacts/n3_old_family_v1/2026-10-12/preparation/publication_inventory/index.json)

GitHubから別bareへfetchし、59 blobs / 7,544,501 bytesをexportしてlocal byte/SHA/sizeに一致PASS。
fetch前に新payloadは取得cacheへ存在しなかった。過去GitHub取得objectsは再利用し、local object copy/alternatesは使わない。
取得したsource/input/planをidentity-only preflightで確認PASS。S1の新M/κ/μ/W/PSD/numeric score/採否を評価していない。
23 synthetic testsはarchiveのbyteとsource identityを取得照合し、remoteで再実行していない。same local environmentでありimmutable CI/外部科学再現ではない。

対象はS1二例、star、実対称one-body＋非負density、一外部mode、full10→active9、ν4、元δgoal=.005。
化学geometry/basis/DF rank/L_D/PF窓なし。新診断のδは保存OCCの正の有理上端そのものを固定する。
sourceのprimal/dual/未判定、NORMの別条件、FIVE包含はsyntheticと静的sourceで確認し、実S1への診断は未実施。
旧source/result/STOP、2882保護tree entries、旧manifest98 entries/他fieldを保持した。
rootでは外部の配置整理が同時進行し、1293 filesは同path/byte、51文書は同byteの移動先を確認、5索引等の外部編集は保持した。
rootのpath/status同一性や外部編集内容不変を主張せず、復元・削除・root commit/pushはしていない。
既存global manifestのdescription欠落FAILは保持。今回entry schema/prefix保全は別にPASS。

利用者がこの固定planでのS1二例・一回診断を明示指示した後、専用execution branchで新authorizationを作成する。
source/input/target/capsを変更せずattempt1/retry0。旧saved-width認可の流用と未認可templateの実行は拒否する。
回収／厳密分離／未判定の新結果を得た後はGPT科学判断へ戻す。現在は中心テーマnull、次段false、Mandatory STOP。

[compact publication receipt](../../artifacts/n3_old_family_v1/2026-10-12/preparation/publication_receipt.json)、[remote identity preflight](../../artifacts/n3_old_family_v1/2026-10-12/preparation/remote_preflight.json)。
新input/新OCC/U,c/真ground-gap/E_forced/quantum compile/旧batch再実行/N1-N2再開は0。
