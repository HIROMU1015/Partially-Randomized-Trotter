# G5 固定辞書の終了認証・成果整理・GPT引継ぎ

2026-10-10 JST。最終技術分類：**`G5_DIGITAL_SIX_VERTEX_CLASS_EXCLUDED_BY_SAVED_CTS_LAW`**。
G4 review §13で指示された保存証拠の確認を完了し、**mandatory STOP**。
現固定toy・同辞書の実用優位実験主線を区切るというGPT判断を記録し、限定理論成果を保持する。
Track B全体の終了、次methodの採択、論文化採否は決めていない。

## sourceとinput

- branch：`track-b-g5-fixed-dictionary-closure-20261010`。
- worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-g5-fixed-dictionary-closure-20261010`。
- base G4：`a221588f42ef3e58f63373915de95607a4ca36be`、direct child G5 source：`407400d5e12aeadda8bb5ca475c4f80c82bff531`。
- [受領GPT review](../../research/track_b_G4_scientific_review_20261010.md)のみrootからbyte-exact snapshot。
  origin、size46702、SHA `50f1e38b4aff01c0746365179e46f3d641ada561a30b14cfdaa32491f79c1caf`を[identity](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/review_input_identity.json)へ記録した。
- [結果前scope/proof](g5_fixed_dictionary_closure_scope_20261010.md)、[scope JSON](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/scope_v1.json)。
  利用者の「こんな感じで進めていく」とreview §13に基づくG5限定技術認可であり、新science authorizationを作っていない。
- one-shot marker SHA256：`26a8ccc820d82bc59fa1f0b7a33b7ad536346d5119819192ca7dc4dc47e55b62`。new G5 markerのみ。旧source/contract/result/authorization/marker/STOPは不変。

対象は2-qubit distinct_basis controlled、p=(3/4,1/4)、sigma=+1、**既知development x=1/4のみ**、
R=3/4 ZI+1/4 V† IZ V、V=exp(-i pi XX/16)、|00〉、finite P3(-ixR)、workspace1 beyond2 systems。
分子、geometry、DF rank、L_D、PF delta windowは該当しない。新held-outや外部再現ではない。
accuracy axis1/200、alpha axis1/5280、log10560、precisions1e-3/1e-4/1e-6を保持した。

## 確認結果

4×7形式係数系の全35 rational basesを独立に消去し、6頂点の完全性とmeanを照合した。
頂点active group数2/3/3/4/4/3、**252 pure precision profiles、三軸756 prices**。
profile最小Phi区間は既存G2と重なり、固定r²を厳密に超えた。
G5はG1/G2/G3 evaluatorをimportせず、G4の有理interval/root/logとsaved table provenance checkerを共有する。
「独立」は頂点・price・CTS scalar coefficient verifierの再実装という意味で、別機関の再現ではない。

全21 columnsの108 saved event/precision conditionsをR1 rawへbindした。
三rについて各108条件で0<=d<=1とC<=r²(1-d)²をexact rationalで満たす。
ideal class全体と、対応する**非負digital係数・あるideal cへのL1距離をbiasに全額支払うclass**へ下界を延長できた。
これは旧residual/tolerance-only K3全体の証明ではない。

exp(37/4)<10560を60-term有理級数＋tailで確認。
Cauchy-Schwarzと共通sufficient-shot policyからG_native>37r²。
同classのtotal1Qはnative1Q以上。下表CTSの1Qはreadoutを含むため比較は保守的である。

| 座標 | 固定r | class policy下界 | 同一保存CTS law費用（表示丸め） | 下界−CTS |
|---|---:|---:|---:|---:|
| T | 2150 | 171,032,500 | 164,669,575.769880 | 6,362,924.230120 |
| CX | 350 | 4,532,500 | 3,712,301.362350 | 820,198.637650 |
| total1Q (native由来下界) | 3500 | 453,250,000 | 431,125,042.097893 | 22,124,957.902107 |

比較の三座標は全て`CTS_selected_complete_laws.json['1/4/1Q']`の同一law。
axisごとのwinnerを合成せず、再proposal探索・再選択していない。
CTSはshots/axis970447、全4 rotation precision1e-4、positive q、2^60分母、workspace1。
12取得済みG4 synthesis列のidentity/count/error metadataを照合し、matrix guardは再実行していない。
half-angle radicalの768bit有理区間から独立にfull P3 scalar coefficients・CTS ideal coefficient intervalsを確認。
signed rotation、negative real corrections、controlled相対位相、angle bias、coefficient L1、strict synthesis bias、
m2/range、整数Bernstein inequality、readout、三資源のexact fractionを照合した。
[raw result](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/result_v1.json)に全bridge rows・confidence certificate、
[全profiles](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/independent_profiles.json)、
[表示CSV](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/resource_bounds_display.csv)へ保存する。符号判断にfloatを使わない。

## 到達範囲

同一dictionaryの固定IID group shares・非負degree matching・任意precision配分・任意full-support event ISに対して、
同一confidence policyで同CTS lawが三資源下界を下回る。
新dictionary、任意LCU、別confidence/range/variance/stratification、真の物理必要shot下界、
whole-circuit最適compile費用、旧K3全体、一般involution/DF/分子/PR-QPE全体へは延長しない。
G4-Aの限定B2 separationとG4-Bの指定J1 pointへの優越は元resultのまま保持する。
G5はG4 reviewからの新post-hoc終了認証として別記録に保存した。

## 検証・provenance・資源

[14 focused tests](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/focused_tests.json)はoff-domain x=1/3とsynthetic costsのみ。
full suiteは実行していない。stdlib Python runtimeとsource/test SHAをscopeで固定した。
保存確認wall0.792741s、CPU0.792056s、peak RSS45,285,376bytes。
wall600s/CPU480s/RSS512MiB/output16MiB caps内、technical failureなし、run1/retry0。
新solver/registered LP/synthesis/angle/Hamiltonian/matrix/circuit/trajectory/DF/molecule/NPZ/GPU/quantum measurement=0。
35 coefficient systemsは形式有理消去であり、registered LPやquantum matrix評価ではない。
[provenance](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/post_execution_provenance.json)で
G4 HEAD時点851保護pathを照合。本文保持のindex追記のみ許可し、旧科学artifact・APIは変更しない。
Track A及びroot dirty worktreeは編集・stageせず、必要なreview一つだけを選んでsnapshotした。
[manifest](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)にscope/result/marker/STOP/source/資料のSHAを保存する。

## 成果整理と次の担当

[claim/evidence map](g5_claim_evidence_map_20261010.md)でR0、G1、G4-A、G4-B、G5とtechnical failuresを分けた。
[static access inventory](g5_static_access_inventory_20261010.md)は、明示p/Q/V/control/rotation、word全列挙、
取得済みnative cost/error表、dyadic full-support proposal、Pauli情報を使う箇所を記録する。
R0のO(m) coefficient arithmeticはこれらの取得費用を含まない。
一般I0のaccess契約・non-enumerative取得法・sampler実装・実classical比較・Pauli取得下界・general DF odd wrapperはMISSING。

**mandatory STOP**。同toy再探索・全面v4・新input/synthesis/LP/DFへ進まない。
現routeの記録確定、新しい構成/理論研究を採択する根拠、RQ・新規性・論文着地点はGPT/利用者へ戻す。
重要なGPT reviewを新たに開始する場合は利用者の開始承認が必要で、Codexから自動開始しない。
GitHub引継ぎ資料はこの独立branchへ必要最小限commit/pushし、公開HEAD全SHAと固定URLを利用者へ報告する。
