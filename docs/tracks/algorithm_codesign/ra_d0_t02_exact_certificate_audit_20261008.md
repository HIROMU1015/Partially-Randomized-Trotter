# RA-D0 T0.2 single-point exact certificate audit

2026-10-08 JST。限定分類は **`T02_FULL_CERT_PASS`**。
T0.1 fixed dyadic pointに対する指定weight移動を一回実施し、
sampler・B2 numerical membership・finite mean・confidence・workspaceを全てPASSした。
これは保存済み一件について固定certificateを満たすB2 lawを構成できたという事後診断である。
RA-D0 v3の成功、B2 minimum/最適性、B2/B3優劣、新規性、汎用repairの完成を意味しない。
全処理後mandatory STOPし、数値実装の一般化と研究継続判断はGPT側へ戻す。

## 固定入力と範囲

- source S：`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`。
- authorization A：`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`。
- v3 result R：`35f8b949079f15d0348bc082b916324870da7246`。
- T0 audit：`72192b3475d59f5c56370cb4068d5659789f0ef4`。
- T0.1 audit / branch基点：`5cf56e4a5949d64c24eac127bac0c223d431df87`。
- branch：`track-b-ra-d0-t02-exact-certificate-20261008`。
- worktree：`.worktrees/track-b-ra-d0-t02-exact-certificate-20261008`。
- 対象：`P1_ANCHORS:1/8:767135:minimum:T`一件のみ。

saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x=1/8、n=767135。
分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。
同じsaved tableの既存三precisionを保持し、他x/n/objectiveを評価しない。
T0.1 continuous pointは読み取り専用で、変更も再projectionも行わない。

[fixed shift contract](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/fixed_shift_contract_v1.json)
はT0.2診断前に保存した。SHA256は
`e74862fd14c23147fd4431117cea5b3c15fe8cdc2b973316feb3afe476b4cdd8`。
delta自体は既に失敗した点を調べた後に利用者が選んだ値であり、事前登録された新science実験ではない。
このcontractは本診断の限定仕様で、RA-D0実行authorizationではない。

[新しい独立script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t02_exact_certificate.py)
はstdlib/Fractionだけを使用し、旧T0/T0.1 scriptやv3 kernelをimport・変更しない。
静的ASTでstdlib import、weight移動の単一call site、重複dict keyがないことを確認した後、
次を一回だけ呼び出した。

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/tracks/algorithm_codesign/audit_ra_d0_t02_exact_certificate.py
```

## 対象列identityと一回の移動

保存tableのIDから解決した対象はordinaryの`O2:1e-4`と`O2:1e-6`。
unique candidate tableのzero-based indexは13/14、B2 variable indexは4/5。
profileから独立に再構築したvariable順序とT0.1保存順序は一致した。

両列はprototype O2、degree、ideal a/b、direction ratio、logical event labels/words/phasesが一致する。
D intervalsはcanonical JSON bytesとexact Fractionの双方で一致する。
D intervals SHA256は`d0b96b84af079a5b44f9807635f9650ac5b8634c2aaf86c7f7965823097a9197`。
workspace peakは両方1で、保存workspace sourceも同一。
destinationのsaved synthesis-bias上界dはsourceよりstrictに小さい。

| 列 | d upper（近似表示） | T cost | CX cost | 1Q cost |
|---|---:|---:|---:|---:|
| O2:1e-4 | 5.4081281094682018×10⁻⁵ | 2397/16 | 81/16 | 24427/64 |
| O2:1e-6 | 7.4053913772922095×10⁻⁷ | 1717/8 | 81/16 | 35137/64 |

入力lawはT0.1の保存integer count receiptから独立に再構築し、q/y/zと全certificate値が一致した。
source weightがdelta以上であることを確認して、delta=2⁻⁴⁰=1/1099511627776をsourceからdestinationへ一回移した。
common denominatorは2⁶⁰、移動countsは2²⁰=1,048,576。
追加の量子化は0回。largest-remainder計算もprojectionも行わない。
y、latent z、他のq、全group massは不変。inactive PTSC_K0/Aのq/zは引き続き0。

## 五certificateのexact再評価

判定は全て保存exact Fractionから行った。下表の小数は表示用であり、判定に使っていない。

| point | sampler | membership | mean margin / 判定 | confidence margin / 判定 | workspace |
|---|---|---|---|---|---|
| T01_PROJECTED_FIXED_DYADIC | PASS | PASS | +9.8463786198957500×10⁻¹³ / PASS | −9.1608554379127114×10⁻¹⁹ / FAIL | PASS、1 |
| T02_FIXED_WEIGHT_SHIFT | PASS | PASS | +9.8463786198957500×10⁻¹³ / PASS | +4.7597036654694233×10⁻¹⁷ / PASS | PASS、1 |

全qは非負、q総和1、countsは整数で総和2⁶⁰。yも固定gridにあり、latent sum=y。
全8 groupについて、両interval endpointsでmembership residual/tau/marginを再評価し、前後でexactに一致した。

| active group | residual upper（前後共通） | 固定tau（前後共通） |
|---|---:|---:|
| ordinary/O0 | 1.3816868054283835×10⁻¹⁹ | 3.0391410822573796×10⁻¹⁸ |
| ordinary/O2 | 1.4800226041303912×10⁻¹⁹ | 2.6054762855654438×10⁻¹⁸ |

inactive六groupのresidualは0。
Dq−ytの全degree lower/upper endpointsとxi upperは前後でexactに一致した。
xi upperの近似表示は3.0827480856777202×10⁻¹⁹。
固定mean capはy×10⁻¹²、confidenceはe=1/200、n=767135、既存ell/kappa upperを保持した。

confidence式全体を前後で再計算し、margin差が
`delta * (d_source - d_destination)`とexactに一致することを確認した。
既存合成誤差上界を使い、operator/matrix評価は行わない。
afterの五certificate PASSに加え、このidentityと下記resource identityの成立により`T02_FULL_CERT_PASS`とした。

## 診断pointのresource会計

固定tableのT/CX/1Q costsから`G_Q=2*n*(C_Q*q+h_Q)`を再計算した。
h_T=h_CX=0、h_1Q=5/2を保持する。
全resource vectorの前後値を
[resource delta](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/resource_delta_v1.json)へ保存した。

| 座標 | exact増分 G_after−G_before | 近似表示 |
|---|---|---:|
| T | 795518995/8796093022208 | +9.0440038889028074×10⁻⁵ |
| CX | 0 | 0 |
| 1Q | 4108007925/17592186044416 | +2.3351321516429380×10⁻⁴ |

増分は各座標で`2*n*delta*(C_destination−C_source)`とexactに一致した。
これらは診断pointの費用であり、新B2 minimum、最適解、資源性能改善の証拠として扱わない。

## 保存証拠と不変性

[input identity](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/input_identity_v1.json)、
[weight shift result](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/weight_shift_result_v1.json)、
[certificate comparison](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/certificate_comparison_v1.json)、
[verification](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/verification_v1.json)、
[evidence manifest](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/evidence_manifest_v1.json)を一組として読む。
全exact値、counts、source/tool/contract identities、禁止操作の回数を保存した。
source-boundなlocal保存値診断で、immutable CI・外部再現ではない。

v3/T0/T0.1の125 protected入力をT0.1 commitとbyte/hash照合し、処理前後で不変と確認した。
旧marker SHA256は`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`のまま。
既存overview/index/共有source/Track Aは変更せず、本書と
[GPT handoff](ra_d0_t02_gpt_handoff_20261008.md)を新しい入口とする。
schema、入力、column identity、exact reconstruction上のtechnical blocking issueはない。

元v3の`D0_TECHNICAL_INCONCLUSIVE`、T0の`T0_TECHNICAL_INCONCLUSIVE`、
T0.1の`T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED`は変更しない。
weight移動1回、量子化0回、solver/registered/synthetic LP/B2 minimum再取得/B3/Farkas/RA-D0 runnerは0回。
authorization/marker作成変更、v3/T0/T0.1修正、denominator/tolerance/tau変更、別delta/precision pair探索、
iterative repair、new synthesis/angle/precision、IS/CTS、science/circuit/matrix/trajectory/DF/molecule/NPZ/GPU/retryも0回。

**mandatory STOP。RA-D0 v4 source、authorization、新one-shotは作成しない。
次の数値実装一般化・研究継続の必要性と範囲はGPT側で判断する。**
