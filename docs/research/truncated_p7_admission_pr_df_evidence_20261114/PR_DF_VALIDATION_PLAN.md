# PR-DF：既存証拠の再利用と必要時だけ行う追加検証

**未承認・未実行の計画。実データの平均・残差係数・モデル予測・Gを今回計算しない。**

## SOURCE対応と資格

|既存資産|固定SOURCE / scope|現在使えるもの・残る不足|
|---|---|---|
|全13round構造平均|C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`、`h4_structural_schedule_results_20261113/run01/COVERAGE.json`、実装de2427d4|LD1/S2/.1/K2の26 E[S]・数値区間。compiled Cではない。現A/B canonical/law_bindingとのmetadata bridgeは未受入|
|C1 normal校正/holdout|C `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、`r_price_phase1_results_20261111/run01/`、凍結7eeabf0a|H4 prefix1 q8/16 R4/8、旧RZZ、元役割・有限cohort。共分散は旧校正のnominal量、全law/長q/R資格false。現scheduleに一致しない|
|C1 forced rare|同C1 `RARE_DIAGNOSTICS.json`|合法order0/order2対応pathの別unit。全law重み付き寄与を未評価、新ZZへ流用しない|
|P0 primitive検証|D `442cafa14f75aeee9e04696df54aaaff1f53260e`（既存law調査d72のHANDOFFから参照）|新ZZの局所作用・lowering。全wrapperの期待価格cohortではない|
|P1 paired C/S|D `39ddd59edf54a3a901e5302207b2765ddbd03fbb:docs/research/d_p1_efficient_czz_20261113/run02/RESULT.json`|LD1 m0/m6各n2、DEVELOPMENT参照、同path/axis。元run01失敗・run02成功を保全。新しい独立holdoutへ読み替えない|
|P2 paired C/S|同D `d_p2_direct_pilot_20261113/run01/RESULT.json`、科学SOURCE ea0706ab|LD1 m0/m6各fresh n4。P1 replayは別参照で、P1/fresh/replayをpoolしない。全law平均/CI/2%・95%/長roundは未資格|
|law対応|C `d72dc4e6228bb380f4d1d000e5e030794fce7164:docs/research/finite_k_sampler_law_alignment_20261113/`|生成SOURCE対応は確認済み。実H4距離、固定seed PCG64 joint-law、F0は未評価|
|価格の技術経路|C628 `rpe_mean_price_model.py`、`paper_nominal_cost.py`、guard、B eec|contextを合わせたE[S]＋単一残差の実装、算術前guard、tier管理を再利用。実値の結合は今回未実行|

新ZZは`CONTROLLED_ZZ_CX_CRZ_CX_V1`、旧C1は`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`。compiler hashが同じでも、両者のC−S係数・native/controlled処理を同一視しない。

P1/P2は同H4 DF10/S2/prefix1/Δ.1/K2、共通exactκ、m0 q1/R2・m6 q64/R128、scalar c+s_R一回、symmetric-directional/full measured X/Y/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。同じSOURCEを示すだけで、別prefix/R/τ/normalization law/軸・wrapperに値を移植しない。

## 第1水準・第2水準への接続

- **finite cohort**：m0/m6のC/S/残差標本が保存されていることと、母期待値が資格されたことを分ける。Cのconsumer guard/metadata bridge受入後でも、第1水準は明示したFINITE_COHORT_SCENARIO。現在C2はこの再受入/再平均をしていない。
- **構造平均のみ**：価格は欠測。構造値をFULL_WRAPPER_MEAN_COMPILED_PRICEへリラベルしない。
- **E[S]+残差model**：既存STRUCTURAL_PLUS_RESIDUALで計算できても、外部資格がなければUNQUALIFIED_MODEL。C1 affineとP2残差、local connected-cluster補正を重ねない。Accounting内の既存相殺を二重計上しない。
- **旧C1の直接cohort/model**：元domainではFINITE_COHORT_SCENARIO / UNQUALIFIED_MODELという区別ができる。現scheduleの値を生成できるという意味ではない。
- **第2水準の正式G**：全対象軸の適格な平均C又は根拠付き区間、law/implementation/identity資格、適格Nと同じtarget、共有誤差を含む精度予算が必要。現52軸のqualified E[C]はすべて未取得。G_paperも今回未計算。

## 最小の次段階

まずBが実際に比較するPR-DF候補・N又はN集合・意思決定margin/許容regret・価格CIのfamily/coverageを指定する。PF用の観測値基準2%と、旧PR pilotの2%/95%案は独立で、いずれも本件では採択しない。

1. **追加compile 0の優先作業**：元ES law/identityと現consumerのphysical/source mapping、P1/P2のdigest/axis/scalar/profileと新ZZのbindingを照合する。元のqualified=false/nullを保つ。必要なprice値のconsumer整形・残差平均の再利用/再解析は、技術metadata照合とは分けてユーザーの有限科学承認に含める。
2. **既観測2セルで精度が必要な場合**：P1/P2はdevelopmentのまま保持し、独立確認cohortを設計する。n4の分散やplug-in nをそのまま精度保証にしない。必要n、seed、cohort、同時CI、rare/tail条件を実行前に固定する。今回nを新計算・選定していない。
3. **残差modelが必要と判断された場合のみ**：既存C `3ae2e605:…/VALIDATION_PLAN.md`の小さいbootstrapを再利用する。LD1 m0/m6を受入後の校正入力として再利用し、m3 q8/R16 (`045843b2429bece6f29b`)を第3校正、m4 q16/R32 (`3df0cfd3356c358aeda8`)を凍結後holdoutとする候補。新しいpairedセルは2個で、各セルはn個のpath×X/Y。3係数基底(1,q,qR)の識別性・law/rareを承認後に確認し、ここではfitしない。
4. **LD2を残す場合のみ**：元のm0 q1/R1、m3 q8/R5、m6 q64/R40を校正、m4 q16/R10をholdoutとする4セル案を再利用できる。LD2にはE[S]がないため、Full C modelの検証か、別scopeでの構造平均取得かを先に選ぶ。LD1のE[S]/係数を移植しない。
5. **長round**：上記のholdoutでもm7〜m12は未資格。Bの候補と誤差予算から、必要な移送点か有効な誤差囲いを選ぶ。必要なら既存Dのm7 q128/R255、normal2 path×X/Y最大4 wrapper案を負荷診断として再利用できるが、平均価格認定用nではない。旧m12巨大T2、固定40/16 wrapper、全round compileを当然の前提にしない。

q/Rはschedule上で連動する。別R/κの最適化を本当に行う場合だけ、固定qでRを変える対照等を追加する。上記の小さいfixed-schedule案だけで独立したR移送を資格しない。

## law・rare・共有誤差

ideal law PのE[S]とsampler Qの平均残差を足すと、Pをtargetにする場合は `E_Q[C−S]−E_P[C−S]`、Q targetなら `E_P[S]−E_Q[S]` のlaw systematicが残る。生成SOURCEの一致、binary64/CDFの実選択確率、固定seed PCG64列、理想IIDモデルを分ける。距離を0と置かない。

必要なら既存F0（保存component確率324と13 order分布の有限距離診断）を別途承認する。今回実行しない。F0だけでは固定seed IID、価格range、rare/tail・全law CIは保証されない。

P1/P2のorder2未観測、C1のforced診断を0寄与の証拠にしない。新ZZ/同lawのlegal conditional/forced検証が必要なら、normal cohortと別unitにし、解析weight・位置/境界class・不確かさを示す。既存C1の別lowering価格を移すことも、forced pathをnormal平均へpoolすることもしない。

構造側の数値区間、残差有限cohort CI、係数誤差、law距離、rare、長q/R systematic、X/Y/round/候補共有誤差を別欄で保持する。未知covarianceを独立又は0としない。根拠のある同時誤差界b_iと固定非負Nが得られた場合だけ `|Ghat−G|≤Σ N_i b_i` を使う。今回b_i・n・Gは評価していない。

追加科学実行では候補/role/n/seed/cohort、SOURCE/law/scalar/compiled条件、保存値再解析・fit/freeze・独立holdout・rare/law診断のcall範囲を明示する。SOURCE/identity/精度不一致、欠測、holdout不合格では部分保存STOP。閾値変更・再fit・別lowering・追加compileへ自動移行しない。
