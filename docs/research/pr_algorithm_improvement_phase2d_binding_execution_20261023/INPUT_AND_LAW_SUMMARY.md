# 生成済みinput／arm／law

固定P0の4 mode・2粒子・LD1・残差density-square二fragmentの合成model。分子geometry／AO basis／分子DF rankは非該当。identityはfaithful_identity_in_tail。4armのF1用lawはT=1/5、q2、delta=1/10（この一点のみ）、r2、h=1/20、K2。catalogはT=delta=1/20、q=r1、K2。

この結果は**binding済み・F1未検証**。科学input hashとSOURCE／環境hashを区別する。

| 保存対象 | SHA256 |
|---|---|
| inputs/physical.json | `d796385d5bc5f52fa2fbffe9d4f327b738055dae0420f4d255b40d862ea14658` |
| inputs/catalog.json | `f85781dbbd7b14e445e551b4094bbba7392ac2f78b27c2a26c621440e38c4dd8` |
| handoff/f1_bundle.json | `9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0` |
| manifest.json | `39a37d99ca9fe7d8d9112b6e94b3946b77bf4a5aded36f08e005b663e687902e` |
| STOP.json | `484139aba19a4125b09a518b77dd03977af3188c4835769b41e1cd41058ece34` |

| arm | canonical L | exact lambda | 将来F1 E |
|---|---:|---|---:|
| SS | 14 | `848/5445` | 2,758 |
| SP | 58 | `7180484777623/35269733400000` | 195,170 |
| PS | 26 | `1118879908/5636255625` | 17,602 |
| PP | 51 | `182339128250602481/876205986856200000` | 132,702 |

全4armでI/u/s/v coverageを固定規則によりbindingした。exact zeroだけを除外し、SのoriginとP fragment間collect、faithful_identity_in_tailを保持した。catalogは別の4label、exact lambda **`43/120`**であり、第五のphysical armではない。

5件のlawはK2、orders[0,2]、canonical probability／normalization／angle／configを保存した。以下は保存済みbinary64 hexのB_totalをそのまま表示する。

| context | B_total（binary64 hex） | law file SHA256 |
|---|---|---|
| SS | `0x1.000fe59c1d666p+0` | `763305d49a31e9f25e8642f666ffcc7913756014de978e7783a011341ae8ee2e` |
| SP | `0x1.001b2ad32c519p+0` | `905bcf032d5016c397af6f19a4c357ca20cee85322d990a8b3dfa1a325c384ac` |
| PS | `0x1.0019d47f49a8fp+0` | `7515c2c89c08ccc249f40ead40a0026b0be8196d60bb1f5c1bed32702655841b` |
| PP | `0x1.001c62a1cb571p+0` | `6b1e581c79d1b28cd4a507b30da3be626ac324f428d9bce7a0cb0e7e62185124` |
| catalog | `0x1.0015096f4f030p+0` | `a1dc5cf3021fbbb7df61ad2e2e4a04c57321c0dd14b57802cb1f9c45db6965b4` |

全component record、分布確率、theta0/theta2、API toleranceはprivate原本と[公開metadata要約](INPUT_LAW_SUMMARY.json)で追跡できる。公開要約はcomponent raw自体を含まない。toleranceは固定`nextafter(existing_residual_bound,+inf)`であり、bias allowance .025や物理signal精度の証明ではない。

F2 K0の数値law、イベント列、状態vector、Fock/Gamma operator、回路は生成していない。canonical JSONのfile bytes／hash／producerは[全出力台帳](OUTPUT_HASH_MANIFEST.json)。rawはprivate保全、内部実pathは非公開引継ぎに記録する。
