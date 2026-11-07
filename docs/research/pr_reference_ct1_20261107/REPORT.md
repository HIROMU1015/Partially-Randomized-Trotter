# 新A CT-1：限定Clifford+T合成・correctness pilot

2026-11-07 JST、SERVER 2。基準 `f5f16fad64748236b0160284dc6c3017974b0132`、独立branch `codex/pr-reference-ct1-20261107`。

**`PASS_LOCAL_CORRECTNESS`、`PASS_PILOT_JOINT_CORRECTNESS`。保存q1/r1のrare/nonrare各1本について、strict word・actual adjoint・phase・局所operator・whole状態を確認し、保存N=192/axisを維持する誤差予算条件を満たした。指定範囲を完了してmandatory STOPした。** 全supportのCT実装・vacuum価格、CT版RPE全体の成功保証、全18round総Tは未資格／未取得。

[SCOPE](SCOPE.md)、[固定対象・入力・raw・検証結果](../../../artifacts/pr_reference_ct1/2026-11-07/README.md)。本作業は著者の限定技術検証で、独立科学レビュー・外部再現・新規性認定ではない。

## 固定条件・参照資産

Linear H4、1.00Å、STO-3G、DF12。元Hの厳密groundをshot毎に供給し準備費除外。Partial S2、LD3、h=.02、finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law。energy1.5936e-4 Ha、全失敗率.05、phase cap .4 rad。Qiskit1.3.0/compiler A/native rz,sx,x,cx/全結合。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合/basis共有/scalar/event/XYを不変更。

正式共通指示2026-10-25、親/worktree AGENTS、PROJECT_MAP、CT-0の3文書/成果物、V3 `224440f740051ed5794e8db87a57908fc02dd02f`、V5補強 `a095a225ceae383027b300e1f31ec5ccbb1522fc`、C2-B pilot `78004943ed5b718c12841e80f55fddf1d1644364` / strong baseline `602ca4b67b1037e221e910a2733444630c5add62` の関連SOURCE・保存結果を確認した。別PFのT会計・最適性は流用しない。C2-B採用規約は `verified_final` で、初回結果や旧LP価格を今回のbaselineに使っていない。

## Stage A：合成前に固定した対象

q=1、r=1、j=0の2本を、**保存Reference X QPYがある各stratumの最小代表**として選んだ。CT価格を観測してからtrajectoryを選んでいない。

| stratum | seed | 固定event |
|---|---:|---|
| nonrare | 4216572573356262833 | order0、rotation `df-fragment-5:Z3Z7`、productなし |
| rare | 4841711708811669773 | order2、rotation `df-fragment-3:Z0Z4`、products `df-fragment-3:Z2Z3`,`df-fragment-3:Z1Z4` |

元event hash・実vacuum binary key・role/key・precisionは `PREDECLARED.json` に合成前保存。既存保存eventを再構成しただけで新samplingなし。Reference Xは保存2 QPYをbyte hashで再利用した。**B′のq1 native QPYは未保存**なので、B′ X/Y×2本=4、Reference Y×2本=2、合計6回だけ新native compile。全8構成のmetricは保存X/Y価格と一致した。

whole8構成の全RZ置換に必要なkeyと、保存local構造の最小代表を合併・重複除去した。**268 positive key：既存97、新規171**。whole依存閉包のkeyは全置換に必須で、任意のtrajectory・重複構造を追加していない。

一様precisionはCT-0 q1のB′/Reference×X/Yの4十分条件を用い、保存仕様e=-13…-6のうち条件を満たす最も緩い **e=-9、epsilon=10^-9** を固定した。これは候補gridの性能探索ではない。

| arm/axis | 全support RZ数上界L | 一様epsilonのCT-0十分境界（概数） |
|---|---:|---:|
| B′ X | 9940 | 3.3816e-9 |
| B′ Y | 9941 | 3.3813e-9 |
| Reference X | 10098 | 3.3287e-9 |
| Reference Y | 10099 | 3.3284e-9 |

保存N維持の追加raw disk安全端点はB′9.507345864357587e-5、Reference9.507345578376295e-5。正確なbinary端点と4条件はPREDECLAREDに保存した。epsilonは両arm・両軸へ共通、全18round用の精度ではない。

## Stage B：strict word

評価対象はcompiler後の**実binary64 RZ**。positive keyはabs(theta).hex、負角は同じwordのactual adjoint。2π正規化・理想πへのsnap・projective phase fitなし。`up_to_phase=False`、seed17、pygridsynth2.0.0、既存exact cyclotomic word matrixと外向きstrict Frobenius guardを再利用した。

既存97件はspec/key/sequence hash/PASS/guard方式/guard≤epsilon/T,T†/Wを照合して再利用。CT-0のguard方式metadataがない4 legacy record（e=-13）には今回も資格を与えず、保存archiveを変更していない。e=-9の採用recordは全てguard方式を持つ。

新規171件はPREDECLAREDのmissing keyだけを独立cacheへ取得し、**全件PASS**。実CPU affinity64に合わせ64 worker、失敗時は新規投入を止める構成。新合成法・自動precision変更・結果を見たkey追加は0。`KEY_REUSE_AND_SYNTHESIS.json` とraw archiveにword/spec/hash/guard/T/T†/W/route/roleを保存した。

## Stage C：correctness

| 検査 | 結果・証拠 |
|---|---|
| 全268 positive word | 100桁区間でstrict guardを再確認、全件epsilon以下 |
| 全268 actual adjoint | Wを含む正word×実adjointがcyclotomic ring上でexact identity |
| 全268×±の既存emitter | 逆順展開・global W・負角inverse・実T数を照合。phase fitなし |
| 21 local構造 | 1～8 qubit全作用空間のoperator比較PASS。D Gaussian primitive、D0 frame/実SOURCE inverse、RZZ/CRZ代表、RTE order0/2 Z/ZZ、scalar/event/vacuum P、CSWAP、H/Sdg |
| whole8構成 | native対CTを保存ground+vacuum入力と決定論的full-joint probeで比較、状態・X/Y信号がstrict上界と整合 |
| 数値評価器 | 人工3-qubit全空間でQiskit OperatorとRZ/SX/X/CX両方向/global phaseを比較しPASS |

local名`D_rzz`/`D_crz`は保存catalogueから選んだRZZ/CRZテンプレート代表であり、全D係数を個別に局所対角化した意味ではない。**実際のD half-step角・scalar・inverseを含む全whole RZ key**は、全268 keyのstrict検査とwholeのtelescopingで覆う。q1には反復境界のcontrolled D融合がないため、新しい境界条件は追加していない。

localの実測spectral誤差は、例えばD0 frame約9.0623e-10、CSWAP約6.2573e-10。これらはbinary64数値診断であり、別途保存したstrict区間上界の代用ではない。検査の絶対数値許容幅1e-10をepsilon/物理誤差保証へ加え替えていない。

wholeの全joint spectral保証は、**各実出現のstrict guardを加えるunitary telescoping**に基づく。有限状態の数値一致からnorm保証を推論していない。Reference17-qubit dense unitaryは作らず、物理wordを各native RZ位置の同一2×2作用として評価した。これはsimulationの圧縮であり、gate fusion・別gadget・新しい合成アルゴリズムではない。268 wordの実emitterとの一致を別に確認している。展開した巨大H/T whole QPYは保存せず、native QPY＋word archive＋位置を定義するSOURCEが合成後回路の表現である。

全wordのexact Wを数学的phase bookkeepingとして保持した。Qiskit emitterのglobal_phaseはfloat表現の数値診断としてexact wordと比較した。wholeではnativeの元global phaseを変えず、全局所wordのexact Wを含む作用を使う。SOURCE inverseをcompileしたnativeが、forward nativeのinverseと同一であるとは仮定しない。戻り側nativeの実binary targetを合成し、C2-BのBDB†重みは流用していない。

Referenceのreset8は同じ初期参照準備で、保存ground入力では真空。full-joint probeは**reset後のunitary部分**への人工検査であり、reset自体をunitaryと扱っていない。測定observable・wire順序はnativeと同じ。

## 誤差・保存shot条件

U_N→U_CTの追加operator上界をEとし、raw diskへ `d_syn <= 2sqrt(2) E`、`d_total=d_old+d_syn` と接続した。旧compiler・SOURCE・vacuum算術diskはd_oldに含め**一度だけ**使用。rho0とPF phase・alias・既存alphaを変更していない。

| arm | 検査した上界 | 追加raw disk上界 | 連続shot必要量上界 | 保存N/axis |
|---|---|---:|---:|---:|
| B′ | finite pilotの最大guard和 | 2.3622869323353764e-7 | 191.87745625980253 | 192、条件成立 |
| Reference | finite pilotの最大guard和 | 2.7435696081137044e-7 | 191.87750550987187 | 192、条件成立 |
| B′ | 全support L×epsilonという**条件式** | 2.811739404710188e-5 | 191.91347201153908 | 192、条件式成立 |
| Reference | 同上 | 2.8564285532811777e-5 | 191.91404936491847 | 192、条件式成立 |

全行でrho_lower>0、beta_stat>0。X/Y両軸の大きい方の誤差を使い、alpha=1/720、36軸の予算和.05、phase .4とenergy精度を維持する**予算条件**が通過した。新しいN scheduleやalpha探索は0。

**これはCT版のcanonical law全体のRPE成功保証ではない。** 今回取得したのは2 trajectoryと有限local keyのwordだけ。全supportの全位相keyで同じguard policyを実装したとは認定していない。有限pilot最大guardをPR mean全体の上界へ外挿せず、全support L×epsilonの計算は「全key policyが別途成立した場合の十分条件」として明示した。`ERROR_AND_SHOTS.json` の全law CT保証は `NOT_QUALIFIED`。

## Stage D：有限pilot T-count

実wordのT/T†を各物理出現に課金した。共通cacheやactual adjointでもgate出現を無料にしない。表は**それぞれの保存trajectory・1 axis/shot**の値。X/Yはこのpilotで同数だった。サンプル平均・期待T・最小T・勝者・改善率は算出していない。

| stratum | arm | T+T† /axis/shot | うちT† |
|---|---|---:|---:|
| nonrare | B′ | 64046 | 11180 |
| nonrare | Reference | 77882 | 16928 |
| rare | B′ | 63930 | 11182 |
| rare | Reference | 77766 | 16828 |

全8行とsigned出現数は `PILOT_T_COUNTS.json`。この有限価格から全support期待価格・全18round総Tを作らない。literal binaryの固定π由来RZにも同じstrict規約を適用し、fast pathのT=0はepsilon内近似の実word価格であってexact型への昇格ではない。両armへ同じprecision・word共有・新規key取得機会を与えた。

## 再利用・実装・技術修正

`track_c_h4_synthesis.acquire/strict_guard/exact_word_matrix`、`track_c2_b_verify.lower_native`、CT-0のrecord照合とq1 headroom、B′の`build_saved/reconstructed`、transferの`assemble`、pilotの`reference_core/wrapper/ground`、compiler Aと区間endpoint helperを再利用した。旧runner/STOP済みjobは再開しない。

新[CT-1 SOURCE](../../../src/trottertracks/pr_reference_ct1.py)は有限target固定・whitelist取得・局所/whole評価・会計だけ。適合する既存高速matrix-action kernelが焦点を絞ったSOURCE調査で見つからなかったため、小さなNumba gate評価器を加え、人工Qiskit比較で検査した。共有環境を変更せず、既存専用venvを利用した。

Stage Aの計算・保存は完了したが、最後のstdoutでjson.dumpsへflushを渡した技術的誤記があった。表示だけ修正し、6 compileや対象固定は再実行していない。Stage C初回はword/人工検査の後、任意依存threadpoolctl欠如でlocal診断前に停止した。依存を外し、21 local workerに対応して専用processのBLASをfloor(64/21)=3 threadとして未完了検査を継続した。通過済みword/人工検査・合成は再実行せず、target/precisionも不変更。初回/継続logと各実行SOURCEを保存した。科学的不整合やguard failureの再試行ではない。

## 公開前の技術確認

SOURCE hashと実行版、結果schema/STOP、PREDECLAREDが新合成より前であること、2 raw archiveと元出力の全byte一致、6文書リンク、旧manifest全entryの不変、索引/概要の追記、SOURCE/文書差分の空白を確認した。raw stderr logの末尾空白は元byteのまま保全した。これらの公開チェックで科学計算・合成を再実行していない。

## 終了・未資格事項

**mandatory STOP**。旧SOURCE/raw/result/grant/ledger/marker/STOPを保全。全18round総T、全support vacuum価格、CT全lawの成功保証、T-depth/FT depth/space-time、最適precision/最小T/論理qubit最適性、研究上の優越・新規性は未資格のまま。旧native depthの40.995952%をT-depthへ移さない。

次担当は**GPT／ユーザー**：この有限correctnessと費用tradeoffを評価し、全support word価格の資格方法・追加範囲の要否を判断する。Codexは自動的に新key・precision・trajectory・18round評価へ進まない。
