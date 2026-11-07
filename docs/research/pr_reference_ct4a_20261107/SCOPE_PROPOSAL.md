# 新A CT-4B：限定科学実行scopeの提案（未承認）

2026-11-07 JST。固定参照 **`c21a43f444f3b7936de36acd0fbd38556ae594d0`**。[CT-4A preflight](PREFLIGHT.md)。**本書は提案であり、実行承認・実行済みSOURCE・PREDECLAREDではない。CT-4B未開始。** ユーザー添付の「CT-4AからCT-4Bへ自動進行しない」に従い、明示承認まで停止する。

## 1. 承認へ提示する有限条件

| 項目 | 提案する固定値/規則 |
|---|---|
| 科学条件 | H4/1.00 Å/STO-3G/DF12、厳密元H ground供給・準備費除外、Partial S2/LD3/h=.02/K2、ordinary control、保存finite-bit law・RPE条件 |
| round | m7、q128/r1だけ |
| trajectory | 保存raw/q128_r1_rare0_0.json、raw/q128_r1_rare1_0.jsonの各j0。seed/event hashはPREFLIGHTの固定表 |
| 方式/軸 | B′4-CXと既存Reference、各X/Y、計8 native構成 |
| precision | 一様epsilon_G=10^-11、exponent=-11、両arm/軸/stratum共通 |
| word policy | 既存pygridsynth2.0.0・strict・seed17、同一specの保存word再利用。新規取得は固定missingだけ |
| 誤差/shot | CT-0 m7の既存disk、alpha=1/720、phase cap .4、保存N=199/axisを維持。再最適化なし |
| 目的量 | 各有限trajectory/軸の物理1shot T+T†とT†、可能なrole内訳。全support期待値を作らない |
| 旧CT-1比較 | q1/e=-9の保存価格は参考値。q128/e=-11との違いをq単独の効果とは解釈しない |

native CX/RZ/depth、追加8qubit、reset/測定とTは別指標で報告する。登録14-CX方式のcompile/新T価格は今回取得しない。

## 2. exact key確定前に必要な準備の境界

**重要な未確定事項：q128 whole QPYとliteral histogramが保存されていないため、静的preflightだけでは完全なmissing key一覧を出せない。** 新規T価格を得る前に入力/回路/precisionを固定し、その固定8構成だけを再compileして実key表を作る必要がある。旧CT-1にもnative準備後にkey集合を保存する構造がある。

添付§6を「native準備を含む一切の科学処理より前に、全literal key文字列の列挙が必須」と解釈する場合、現資料ではその条件を満たせない。そこで、ユーザーへ提示する承認範囲を、**固定8構成のnative準備を含み、実key一覧の保存後に限ってmissing合成へ進む条件付き範囲**として明示する。この順序を含む承認なしには準備compileも開始しない。別のqへ切り替えたり、静的に判明したように装ったkey表を作ったりしない。

承認された場合の順序は次とする。段階ごとの再承認を追加する意図はなく、初回承認がこの有限順序全体を含むことを求める。

1. **native準備前**：上表のq/r、2 event配列・seed/hash、8構成、SOURCE/compiler、e=-11、保存N、local対象選択規則、STOPを独立入力記録に固定する。旧pickle/行列を再利用し、DF/SCF/ground solveやevent抽選は行わない。
2. **native準備のみ**：固定8構成のbuild/compileを各1回実施し、保存native metricと比較する。Referenceの保存vacuum binary角、actual inverse、reset8、measurement1も確認する。T合成・T価格集計はまだ行わない。
3. **合成前の固定**：8 nativeの全literal signed RZと物理多重度を読み、positive lookup/specへ対応させる。local検証はその依存閉包に含まれるkeyだけを使う。保存e=-11 recordを照合し、reused/missing/未資格を分け、正確なkey一覧・role・入力/word provenanceをPREDECLARED相当へ保存する。
4. **固定missingのみ取得**：旧cacheはread-only、新規cacheは独立領域。同じ取得規則を両armに適用する。最後にcorrectness・誤差/保存shot条件・有限T会計を行ってSTOPする。

準備compileで保存価格/phaseと本質的不一致、対象外SOURCE/精度や別trajectoryが必要なら、その時点でSTOP。合成開始後の新key追加やprecision変更を禁止する。入力記録は科学的再現性のためで、ユーザー承認の証明書ではない。

## 3. 必要最小限の回路・実装

8構成はrare0/1×B′/Reference×X/Y。q128 QPYがないため、native再compileは8回。XからYを推測したり、保存CX/depthだけを使ってliteral角を再現したことにしない。新しいlocal native compile、新gadget、F/G split変更、CSWAP削減、新fusion探索は行わない。

既存 `bp.reconstructed/build_saved`、`transfer.assemble`、`pilot.reference_core/wrapper`、`compile_reference` を固定入力で再利用する。Referenceはsource Fのactual inverseを用い、forward nativeのinverseと同一と仮定しない。保存common.pkl.xzを独立領域に読み、旧SOURCE・cache・STOPを書き換えない。

CT-1のq1固定sample選択/出力先/endpointをq128へそのまま使えないため、別namespaceの小さなadapterとrunnerが必要。record/guard/emitter/計数関数を再利用し、合成法・回路builder・汎用管理基盤を再実装しない。必要な人工検査もCT-1の位相/順序/adjoint/controlled relative phaseに対応するものへ限定する。

**全expanded CT whole QPY、17-qubit dense全operator、q128のwhole state simulationは本提案の必須実行に含めない。** native streamの全位置と実wordへの対応、strict局所guard、同じjoint演算子に対するtelescopingがoperator保証を与える。局所検証でその接続が成立しない場合はSTOPし、巨大simulationを自動追加しない。

## 4. Word再利用・合成・correctness

再利用recordは、同一positive binary角/e=-11/backend/seed/strict spec、PASS、guard_method、保存guard上界、word hash、W、T/T†の整合を確認する。旧legacy e=-13の未資格metadataを補完しない。CT-1 e=-9 wordをe=-11要求へ自動再ラベルせず、cross-precisionの価格選択も行わない。

新規取得は合成前に固定したmissing keyのみ。既存 `acquire` のepsilon/4要求、exact-word strict_guard、必要時の同一word高精度guard、受理条件を再利用する。guard failure/phase不一致/長さ拒否が出たら停止し、別epsilon・別backend・SK fallback・新keyへ進まない。有限keyでもbackend完了の全域保証はないというCT-2/3Aの未資格を維持する。

取得後に少なくとも以下を確認する。

- 全採用wordのstrict binary RZ guardとW付きexact行列。Frobenius上界をspectral上界として使い、projective fitなし。
- 各keyの±実角、actual word adjoint、逆順emitter、実T/T†数の対応。WはT=0でもphaseから落とさない。
- CT-1と同じ局所full-space検査を再利用し、今回新しい反復D境界/inverseを保存native loweringで照合する。追加のlocal対象は**固定8 wholeに実際に出るパラメータ**に限定し、別角度keyを合成しない。
- 固定local検査族はDの端/融合境界、Gaussian frameとactual SOURCE inverse、RTE中央Z/ZZ、product、scalar/event/vacuum P、CSWAP、X/Y wrapper。既存catalogueの対応recordを使い、同じ型の全科学runnerを再実行しない。
- B′のcontrolled相対phaseは同じjoint operatorの局所targetで検査し、control branch別のphase fitや粒子数sector射影を行わない。戻り側native targetとforward wordのactual adjointは区別する。
- 8 nativeのRZ histogram、qubit、global phase、reset/measure、保存metricとword置換先が一致すること。未対応命令/欠損keyを0扱いしない。

局所検査で使うrecord名・実角・順序は合成前のPREDECLAREDへ保存する。既存q1の21 local検査を丸ごと再実行することや、priceを見て代表を追加することはしない。

## 5. 有限T会計とrole

native回路cの角度thetaの物理出現数をn_c(theta)、positive lookup a=abs(theta).hex、採用wordの総T数t(a)、うちT†数d(a)とする。

\[
T_c=\sum_\theta n_c(\theta)t(a),\qquad
T^\dagger_c=\sum_{\theta\ge0}n_c(\theta)d(a)
+\sum_{\theta<0}n_c(\theta)(t(a)-d(a)).
\]

同じwordの物理出現を1回にまとめず、key共有は取得の重複削減にだけ用いる。SX/X/CXは既存のexact Clifford lowering、reset/measurementは旧規約でT寄与なし。piに近いRZも、実binary targetのstrict recordが必要。native後の新Clifford+T fusion/最適化は入れない。

出力は8行、各stratum/axisにB′/ReferenceのT+T†、T†、保存native CX/depth/RZ、qubit9/17、strict joint誤差上界、保存N条件を対応させる。T増加も減少もそのまま記録する。stratumを平均して期待Tにせず、全18roundへ乗算しない。

role内訳はD/境界、frame/inverse、RTE中央、product、scalar/event、vacuum P、CSWAP、wrapper。V3 `saved_path_parts` のsigned histogramを固定8構成だけで照合できるが、**full nativeのsigned key histogramとの一致を確認した部分だけ**採用する。D融合差分は会計上のsigned補正として別記し、負の物理gate数とはしない。対応が未確定ならnative全体の直接計数を正本とし、role attributionは未資格/未割当として残す。恣意的に差額を特定roleへ押し付けない。

## 6. 誤差・保存shot条件

各physical RZ位置のstrict guard上界をepsilon_jとし、同じjoint unitary/reset/observableで

`epsilon_CT ≤ sum_j epsilon_j`、`d_syn ≤ 2sqrt(2)epsilon_CT`。

CT-0 m7の旧diskに一度だけ加えて、`d_total=d_old+d_syn`、`rho_lower=rho0-d_total`、`beta_stat=.4-beta_PF-asin(d_total/rho0)`とする。旧compiler/vacuum算術/normalizationを重複加算したり取り除いたりしない。期待gate数をworst-case個数にしない。

保存Lとepsilon=10^-11の十分条件、および取得した有限wordのguard合算について、正半径/正位相統計幅と保存N=199/axisの条件を既存192-bit区間helperで確認する。alpha=1/720、全失敗予算36/720=.05、energy精度/alias規約を変更しない。新Nやalpha/precision最適化は行わない。

これは有限pilotと「同じ局所精度を全supportで実装できたなら保存Nを保つ」という条件の確認である。**有限8回路のword成功から、canonical law全体のCT RPE成功保証を認定しない。** 他roundをCT化したとも扱わない。

## 7. 資源・保存・STOP

実行時にSERVER 2の実際の空き資源・適用制御を再確認する。有限8 compile、独立missing key、局所operatorを適切に並列化し、重複build/guard/word行列を避ける。科学的な入力と演算順序を保ち、CPU/RAM/workerへ根拠のない少数上限を置かない。共有設定、他ユーザー、他Trackのprocess/fileは操作しない。現在の空き量を将来の予約として扱わない。

CT-4Bを承認された場合だけ、独立領域にREPORT/SCOPE/PREDECLARED、SOURCE/INPUTS/ENVIRONMENT、保存/新wordとguard、8 nativeと検証・価格・role会計、未資格事項、STOP.jsonを保全する。旧recordはread-only参照または由来を保つ複製、新規は独立cacheへ。承認証明書やgrant管理を作らない。

次のいずれかでmandatory STOPする。

1. 固定8構成と限定検証/会計を完了。
2. strict guard、actual adjoint、W/relative phase、保存native入力に本質的不一致。
3. 両armの同一条件・取得機会や物理出現の会計を維持できない。
4. 旧誤差/保存shot条件が成立しない。
5. 合成開始後の対象外key、別precision、別trajectory/q/r、scope外compile/検証が必要。
6. 科学条件、compiler、F/G/gadget、lawの変更、または追加未承認計算が必要。

失敗時にepsilon/対象を変えて再探索せず、PASS/FAIL両方の証拠を保存する。有限取得の完了時間や全入力停止性は保証していない。通常の共有資源競合だけで正常な計算を不必要に停止・再起動しない。

終了後に独立branchへcommit/pushしremote SHA/tree/主要内容を照合する。**CT-3B、全support価格保証、全18round総T、最小T、T-depth/FT depth/space-time、precision/alpha/shot最適化へ自動進行しない。**

現在はCT-4Aの停止点。次担当は**ユーザー**による本scope・準備順序・一様precisionの承認判断。承認された場合だけ**Codex**が有限実行し、結果後の科学判断を**GPT／ユーザー**へ返す。
