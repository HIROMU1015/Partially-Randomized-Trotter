# B6b：B/C統合仕様とcoverage

2026-11-08 JST。実行SOURCE `13114b2bd9db7dcf1ee7f027ab202f72786e2552`。[代数と共通N条件](MEASUREMENT_EQUIVALENCE.md)／[coverage JSON](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/coverage13.json)／[CSV](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/coverage13.csv)。NはB4-2の保存値の転記で、新しいshot数や全13-round費用を算定していない。

## B側のhandoffと固定identity

B4-2公開07d18bfe、実行local SOURCE8e472b50、公開SOURCE6da5afba、同一tree767d68f537d0c99f76b83fab398be2c900c55fe5の対応を保持。[B4 metadata](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/B4_handoff_metadata.json)と[43固定入力binding](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/input_bindings.json)を保存した。B4-2元handoffのN共通化未認定欄はfalseのまま、新B6bの理想条件付きassessmentを別記する。

固定条件はH4 linear1Å/STO-3G/Nα=Nβ=2、元H、number-CDF8/元frame順LD2、ordered S2/s1/r32/K0/h=.1、exact original-H ground仮定、状態準備除外。元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law記述hash `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5` を一致確認。law hashは分布距離/IID証明ではない。

energy .00159Ha/failure .01/βtotalπ/6/βstatπ/12/αaxis.01/26/prior[-10,10]/13-roundは不変。λ_R・r32と保存normalization、X=Re z/Y=Im z、φ=−hE/E=−φ/h、連続dyadic最近branch decoder、元H state/sector36/Qiskit bit-reverseをB4/固定SOURCEから対応させた。C側は全system入力のobservable検証であり、保存own-DF groundを元H stateとして流用しない。

|側/項目|契約上必要な内容|
|---|---|
|B round/axis|m0..12、q=2^m、t=.1q、axis X/Y、保存N_m,b・α・target/prior/decoder/source|
|B signal|B1元H operator誤差とその資格、raw attenuation A_m、scalar/identity一回、ideal P/implemented Q・fresh trial前提|
|C identity|同H/CDF/sector/frame/PR/h/q/law、同じunitary ordered pathとX/Y意味、同じ準備費除外|
|C price|方式、q/m/axis、full one-interrogation scope、6資源、peak幅は別、P0/topology/precision・SOURCE|
|C sample/model|whole-path単位、n/cohort、direct有限標本/理論期待値/明示conditional model、model係数/適用域/仮定とversion|
|C uncertainty|平均価格推定量の共分散、paired方法・XY/資源・共有calibration/round依存、bias/law不明はnull|

q8ではsampling23725、compile/post23725と44a58の限定引継ぎを区別。q32ではsampling/raw35bfdとcompile/post94e94を区別した。q1はCDF8/r32生成4df8の保存cohort、adapter/compile2dd012である。[価格catalog](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/price_catalog_metadata.json)。基点C7-3の実装・強いbaselineは不変。

## 全13-round coverage

|m|q|N_X|N_Y|価格分類|n|cut|母平均資格|
|---|---|---|---|---|---|---|---|
|0|1|512|512|DIRECT_COMPILED_FINITE_SAMPLE|32|event16|UNQUALIFIED|
|1|2|512|512|DIRECT_COMPILED_FINITE_SAMPLE|16|step1|UNQUALIFIED|
|2|4|512|512|DIRECT_COMPILED_FINITE_SAMPLE|16|step2|UNQUALIFIED|
|3|8|512|512|DIRECT_COMPILED_FINITE_SAMPLE|32|step4|UNQUALIFIED|
|4|16|513|513|DIRECT_COMPILED_FINITE_SAMPLE|16|step8|UNQUALIFIED|
|5|32|514|514|DIRECT_COMPILED_FINITE_SAMPLE|8|step16|UNQUALIFIED|
|6|64|516|516|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|7|128|521|521|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|8|256|530|530|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|9|512|549|549|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|10|1024|589|589|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|11|2048|679|679|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|
|12|4096|910|910|EXPLORATORY_PROXY_ONLY|—|model only|UNQUALIFIED|

q1の保存32本はDIRECT_COMPILED_FINITE_SAMPLEだが、q≥2のcomplete-step-cut proxyへ移送するとSCOPE_MISMATCH。信号の代数が同じことと、価格のcut/融合/標本lawが同じことは別。q2〜32の88本もXYを176独立標本に数えない。

元C6a affine係数はFROZEN_PROXY_AVAILABLEとして保存versionを指せるが、長q/母平均保証はない。C7-4公開 `64072ba81bb70db1df1b34e94adb08e78926f24b` を固定objectで確認し、SOURCEdd31e1e2・使用価格hash・4モデル仕様/係数参照・独立検証false/coverage falseを別版metadataとして採録した。[C7-4 metadata](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/C7_4_version_metadata.json)。新fit/価格再解析やC7-4 worktreeへのアクセスはない。

q64〜4096はC7-4のq-only探索予測が存在するが、独立実compileではなくEXPLORATORY_PROXY_ONLY。run/lane版は未来feature lawが不足している。全13-rowのPRICE_UNQUALIFIEDは「保証付き期待価格が揃っていない」の意味であり、未取得価格を0にしたり構造boundありと推定しない。raw論理register幅9/17はSOURCEで定義され、保存compileでも同じだが、routing/物理幅保証ではない。全RPE数値totalはnullを維持する。

## 最小実装とguard

新 `track_d_b6b_bc` はB4 handoff/CSVとC価格のmetadata・identity/P0/cut/単位を照合する薄いadapter。B4-1のradius/phase/N計算やdecoderを再実装・再実行しない。未来hook `aggregate_registered` は、全13×2軸×2方式の52期待価格recordと共通ideal N条件が明示される場合に、既存 `c1_stage3_accuracy.aggregate_totals` へ委譲する。選択された関数のASTがB4-2版と同一であることを確認した。

`rpe_resource_accounting.build_rpe_resource_summary` は旧DF/PF candidateを作成し直す多数のfieldと別guarantee schemaを要求するため、B4の元H handoffに架空のPF値を詰めて流用しない。既存の薄い総和関数にexpected scopeを明示して接続し、full-wrapperだけを許すguardを先に置いた。元の関数や研究契約は変更していない。

|guard/条件|扱い|
|---|---|
|欠損/重複round-axis-method、q不一致|算定前に拒否|
|H/CDF/law/PR/scope/P0/SOURCE/標本単位の不一致|拒否。kernel/event単位をfull-wrapperへ混ぜない|
|finite sampleを母平均と偽装、mean missing|拒否。explicit conditional modelは仮定を要求|
|actual保証の未知|ideal条件付き会計の絶対禁止理由にしない。actual claimは別途拒否|
|保存B4 NX=NY|現hookはこの固定scopeを確認。一般の異軸/方式別Nへ無断拡張しない|
|peak qubitsを加算metricへ入れる|拒否。幅制約として別扱い|
|不確かさmissing|None。0や独立block-diagonalをdefaultにしない|
|実H4 total|今回runnerは呼ばない。空価格表が全13-round guardで拒否されることだけ確認|

共通NならΔG_g=Σ_m,b N_m,b E[CRef−CSym]。資源ごとに分離し、attenuationを価格へ掛けず、q/r/nclassicalをshot数へ追加乗算しない。depth×shotsは累積depthであり実機時間ではない。RZをTへ換算しない。

不確かさのhookは各metricの52×52 Cov(推定期待価格_i,推定期待価格_j)を使い、符号付きweight wでwᵀΣwを計算する。これはsingle-path価格分散、物理的な総支出のrun-to-run分散、量子測定分散とは異なる。共有calibrationのround間/方式間off-diagonalを落とさない。cross-resource共分散は保存元のfull covariance/モデルSOURCEを保持し、現hookが返すのはmetric別の対角量のみ。将来複合指標を作るならCov(G_g,G_h)=Σ_i,j N_i N_j Cov(C_i,g,C_j,h)のjoint伝播が別途必要であり、未取得cross-resourceを0としない。

## 今回の実行と次の評価

人工13-round/52価格recordだけで既存総和への接続・paired恒等式・欠損拒否・共有共分散を試した。全6資源を同時に処理し、qubit加算/attenuation二重会計/actual昇格を拒否した。実H4の価格とNの乗算、新しいN/price統計/モデルfitは0。

**条件付きの理想モデル数値評価はactual保証完成より先に実行可能**である。ただし全qの価格estimand/仮定/versionとclaim水準をユーザー/GPTが採択し、今回禁止したH4 N×価格の数値算定に新しい明示許可が必要。[次段階案](NEXT_EXECUTION_DRAFT.md)。
