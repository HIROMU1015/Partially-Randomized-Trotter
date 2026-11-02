# Track A：H4 geometry / 固定anchor / H6 / H8の条件対応

2026-10-13。異なる条件の削減率を単一scaling curveへ接続しない。
この表は固定資料の転記であり、同じcompiler labelだけから回路費用の完全同一性を認定しない。
provenanceの各40文字commit・blob SHAで参照版を固定した。

|条件|H4 geometry|H4固定anchor|H6独立確認|H8 feasibility / core計画|
|---|---|---|---|---|
|モデル/距離/基底|linear H4、6距離.70/.80/.90/1.10/1.40/1.60 Å、STO-3G|linear H4、1.00 Å、STO-3G|linear H6、1.00 Å、STO-3G|linear H8、1.00 Å、STO-3G|
|modes/sector|8/36|8/36|12/400 (Na=Nb=3)|16/4900 (Na=Nb=4)|
|T / delta|.8 / .8/q、q1/2/4/8|同左|同左|同左|
|DF政策|legacy固定rank12、元凍結入力の順序/係数を保持|legacy固定rank12、保存1Å入力|tol-only1e-8、cutoff0、rank19、signed generation order、weighted Hermitian projection|tol-only1e-8、cutoff0、rank28、signed generation order、weighted Hermitian projection|
|prefix|0/3/4/5/6/9/12、PM-1 p4/5を含む|PR p3、full PF p12、16固定panel|B0/B2 p5/10/15、B1 full19|feasibility PRp7/full28、core B0/B2p7/14/21/full28|
|PF次数|S2のみ|PR S2、full PF S2/global Yoshida S4|B0/B2 S2、full S2/global Yoshida S4|同H6の登録設計|
|finite RTE|r1/2/4/8、K2/4|q1/2/4ではr4、q8ではr2、K2/4の固定8PR|探索r1/2/4、K2/4；確認はp5、q1/2/4 r4・q8 r2、K2/4|feasibilityは4移植構成、core探索r1/2/4、K2/4|
|control|ordinary-controlled Z/ZZとbranch phase|symmetric_directional、ordinary感度4 wrappers|symmetric_directional、ordinary感度4 wrappers|symmetric_directional、feasibility ordinary anchorsを別に記録|
|Gaussian lowering|`h4_geometry/gaussian_structure.py`のstructured JW 2-mode rotations+phases；隣接同basis境界を再利用|frozen native JW Givens `BogoliubovTransform(U.T)`、raw concatenation|同native cost path、raw concatenation|同native cost path、raw concatenation；coreのjoint-branch/hash再利用は別実装closure|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation_degree1、topologyなし|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし|同左、固定compiler hash1239304c…|同H6 hash、core source/profile bindingは別|
|compiler/環境identity|compiler fbe36b72…、environment6abf53c3…；旧host bit一致未確認|各anchor固定profileを参照；labelが同じだけでgeometry compiler FPと同一視しない|compiler1239304c…、profile SHA4e32bb9b…、Python3.11.0rc1等；旧host bit一致未確認|feasibility/core固定profile；旧host bit一致未確認|
|reference / u|元保存exp(-iE*T) target、明示u減算なし；全DF/stateの厳密保証を追加しない|full-action numerical reference/probes、経験的u=1e-11と10×感度；certificateではない|full DF reference、経験的u floor1e-12・safety10、B_upper；certificateではない|full DF numerical action、経験的u/B_upper、core split/CSR/Horner closure；certificateではない|
|shot会計|axis alpha=.025、h=epsilon/sqrt2-bias、ceil(2B² log80/h²)|同alpha、hからuを減算、B_upper|同alpha、hからuを減算、B_upper|core同H6、feasibility20件だけでshot優位を評価しない|
|state preparation|P=0、signal精度；energy/RPE総資源ではない|P=0、signal精度|P=0、signal精度；common P感度は別|P=0をprimaryとする計画、common P感度は別|
|候補/標本|6×218=1308、PR各32 paired、det各1；同じ標本で選抜と評価|16固定候補、PR8×32 paired、det8×1；固定panel内point selection|探索92候補/592groups、探索PR8本、上位union固定8候補を別seed32本で確認|feasibility8cases/20compile、PR1本ずつ；core92候補、探索8本、最大8unionを独立32本確認する承認済み設計|
|数値結果の範囲|302点×6、元集合B2点最小；B0+Iは未登録感度|4精度のfixed-panel PR/full-PF比；B0不在|4精度の確認集合PR/full-PF比と別B0比較|feasibilityはcompile可能性/単一wrapper値；core最終shot mapはこのH4資料に未収録|
|希少次数|events未保存、法則逆算のみ|固定公開metadataの範囲に限定|観測metadataあり、別条件付き8本pilotもあり|feasibilityは44 eventすべてorder0；coreの結果を補完していない|

H4 geometryのcost seriesを、旧dense H4やnative anchorの費用系列へ混ぜない。
anchorとH6でもDF rank政策、prefix、母集団・selection、uが異なる。
H6の希少次数pilot（別seed8本/16compile、0.00734%の局所補正）はH4/H8へ流用しない。
H8 feasibility資料の当時未承認というstatusと、後日のcore資料のone-shot承認を区別する。
H8実行・準備・source・grantはこのH4 taskから変更しない。

固定参照：

- H4 geometry `7c4483378047b57f514c47dad98905e3f2a9a756`（科学SOURCE d0726e…）。
- H4 anchor `0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`（SOURCE 217c97…）。
- H6確認 `321cd8c3066194fe98741cbebbcdba1515d1deaa`（科学5a903a…、実行87b6b9…）。
- H6 pilot `0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`（pilot SOURCE 843265…）。
- H8 feasibility `a55da5c2b012912c485184845aa73ca053ac00b9`（SOURCE 9bfea8…）。
- H8 core文書 `eaeeefc38be454375d141fabfb8ae229498175b9`（SOURCE cb8a56…）。

全文blob bytes/SHAは
[provenance](../../artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/provenance_v1.json)。
H6/H8の文書がこの専用branchにない場合も、上記固定commitのGitHubから参照できる。
