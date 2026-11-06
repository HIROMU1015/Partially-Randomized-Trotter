# Track D C6b：独立q8 paired評価の実行前契約案

2026-11-07 JST。**未採択・未実行。本文は実行許可ではない。** [C6b-0 REPORT](REPORT.md)で確認した固定資料を基点とする。ユーザーが具体的scopeを明示して許可した後だけCodexが実装・科学実行する。承認証明・新grant・専用承認DBは要求しない。旧契約・SOURCE・input・raw/result・grant/ledger/marker/STOPを保全する。

## 1. 実行前に採択する目的と統計条件

|項目|固定すべき内容／現段階の候補|
|---|---|
|科学的な対象|同じlawのq8 whole-pathにおけるReference−strong symmetricの資源差、保存モデルのq8精度。全13-roundの費用・成功率は含めない|
|一次命題|paired予測精度12量を候補にする。direct優劣を一次にする案も別候補。モデル精度と優劣の同時claimにはjoint coverageまたは予算配分が必要|
|modelのestimand|population平均で校正した線の残差か、固定した保存モデルhat f8の条件付き誤差かを明記。前者の合成calibration floorを後者へ機械的に課さず、後者の評価で将来qの係数不確かさが解消したともしない|
|保持する資源|RZ/CX count、各depth、total depth、size、両方式の絶対価格・差・covariance。qubitsは別の幅W_max条件。主資源や重みは無断で新設しない|
|価格幅|旧候補1単位＋5%×事前固定価格規模、または用途に根拠があるユーザー指定幅。pairedのscaleはsymmetric規模で、差そのものでは割らない。具体値または決定規則をholdout前に固定|
|古典信頼度|alpha_C=.05は候補。paired12、absolute24、joint36のどれを確認的に扱うか固定。モデル/優劣/absoluteの各claimへ全体予算内で配分する。X/Y・資源の依存を保ち、観測後にfamilyを縮小しない|
|推論資格|ideal/implemented law、pseudo-streamの扱い、IID/tail/support/varianceの根拠、固定nまたは有効な逐次手順。名目的normalを有効CIと偽らない|
|標本数|保証されたnは未決。旧n32は探索的call予算候補のみ。確認的なら採択した統計手順で必要n/余裕を正当化する範囲の別途数値評価を先に許可。必要数を得られないまま確認的samplingへ進まない|
|追加校正|原則として自動追加しない。必要ならk2/k4、seed/ID、旧各16本の扱い、pooling/fit/共分散更新とsample数選択の統計条件をscopeへ明記。全モデルを凍結してから独立holdoutを生成|

統計条件を正当化できない場合は、ユーザーが**探索的q8診断だけ**を許可する選択肢を残す。その場合は有限標本のcorrectness・paired価格・名目的診断までを成果とし、確認的モデルPASS・全law母平均保証を出さない。n32が少なくとも必要／十分とは認定しない。

## 2. 固定SOURCE・入力・law・回路

- 起点はC6a公開 `e9d4e2f602d5f9e014507ca7e391719bab017b88`、科学実行SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`。将来の最小拡張は新しい専用revisionへ分離し、元ファイル・旧runを変更しない。
- linear H4/1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元順LD2、S2/s1/r32/K0、各h=δ=.1。元H厳密ground仮定、状態準備費除外。packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5` を維持。新fit・ground solveなし。
- q8はcanonical lawによるfreshな8 occurrence。各h-stepの32 ordered eventsを保持し、短path反復・旧32 cohort巡回・旧C3 q8 fixtureを代用しない。
- 予約済み文字列 `track_d_c6a_q2_q4_paired_price_pilot_v1:independent_holdout:q8:unused` を独立domainの起点候補にする。これはC6a固定記録では未使用で、本C6b-0でも未使用。将来scope内で未使用性を確認し、master・完全seed key・PRNG版・sample ID・occurrence順序・SOURCE bindingを生成前に固定。今回seedを作らない。
- 同sampleの同じwhole-pathを両方式・X/Yで共有。price用共通PR-roleを使い、method/axisで別trajectoryを引かない。量子shot用domainとは別。
- cutはcomplete step4後。V=U7…U0=A B、B=前4 step、A=後4 step。whole VにReference gadget一つ、A†は同じAのactual inverse。scalar/identity各step一回・signed角度・analytic全pathχ、weight/normalization/attenuationの会計を保持し、unitaryへweightを掛けない。
- strong対照はuncontrolled Gaussian/directional/head relative fusion/tail共有/合法step融合/phase集約。Referenceにも同等の合法最適化、追加register/CSWAP/χとcutの融合損失を含める。弱い対照、fenceによる別価格政策、節約値の合算、結果依存cut最適化は不可。
- P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1.0/initial-zero false/topology-free/位相保持。component abs1e−10、observable Frobenius2e−10、rtol0、global phase fitなし。routing/T合成・実機noiseは対象外。

## 3. 最小実装・許可するcallの指定案

現在のC6a FreshRequest/bindingはq2/q4・sample<16に限定されるため、そのguardを無断で広げない。q8/採択n向けのopt-in request・seed schema・価格資格だけを新revisionで追加し、C3 `prepare_steps`、C5 `wrapper_for/_checked_price`、C6a canonical sampling/CAR cache、corrected native bridge、StrongBuilder、Reference gadget、P0/checker/state-actionを再利用する。新sampler/compiler/state simulatorは作らない。

将来許可scopeに含める人工/回帰、sampling、raw/P0/postcheck、paired統計・必要なモデル診断の範囲を明記する。古い科学runの再起動はしない。SOURCE/input/law・契約整合性、人工回帰の後、sample ID/seedと全eventを回路使用前に保存・bindingする。イベントを再samplingで置換しない。

call予算はraw n_H path、各PASS pathの最大4 full wrappers、合計最大4 n_H compile。探索案n_H=32なら最大128 wrapperという予算で、現在の許可ではない。追加校正を採択する場合のみraw k2+k4・最大4(k2+k4) compileを別枠へ明記。既存C6aの128 compileを価格校正のため再実行しない。

rawで独立CAR/AB/actual adjoint/unitarity/vacuum ray/χ/scalar/sign/normalization/strong direct control/complex coherence/X/Yを確認。**そのpathのraw PASS後だけ**同pathのcompileに進む。compile後はCAR/raw/paired X/Y/位相/leakageとSOURCE/inputを照合する。失敗後の閾値緩和・phase fittingなし。

## 4. 保存・判定・STOP

|項目|固定規則案|
|---|---|
|標本単位と成否|n_H whole-path。X/Y・event・occurrenceを独立nに加算しない。予定/完了/失敗/欠損ID、全pathの結果を保全し、都合のよいcomplete-case平均にしない|
|モデル凍結|係数・校正ID/共分散・source・fit法・価格幅・family・推論法・n/停止規則をholdout生成前に固定。holdoutをcalibrationへ混ぜない|
|価格記録|6資源を別々のfull-wrapper native/DAG単位で保持。絶対価格も保存・不確かさを報告し、未資格なら未資格。peak widthは別記|
|技術PASS|全予定pathのraw/compile後/paired X/Y/χ、identityと公平性が成立した有限domainの記録。統計モデルの成功と分ける|
|統計/モデル成功|採択されたestimand・qualified同時推論で、事前固定の幅/優劣規則を満たした場合だけ。そのq8/law/P0/scopeに限定|
|不成立と未資格|幅/優劣の不成立、intervalが境界を跨ぐ未決、前提不足の未資格を区別。名目的normalだけなら確認的PASSにしない|
|停止理由|SOURCE/input/law/seed binding/順序/phase/AB/adjoint/vacuum不一致、correctness不成立、強い対照を維持不能、compile/profile/scope不一致、calibration混入、統計前提不足、未許可の追加数・別q・閾値/モデル変更が必要な場合は依存処理・claimをSTOP|
|失敗後|証拠と全IDを保存。無断retry/resume、seed/cut変更、holdoutに合わせた再fit・閾値緩和なし。次の判断をユーザー／GPTへ戻す|
|外挿|q8で通ってもq>8・全13-round・N共通利用・全RPE費用へ移送しない。全law CX定数差の証明とはしない|

重い計算は実空き資源・task別メモリを確認し、cache/共通context・path/方式/axis並列を活用する。SOURCE固定前に、余剰CPUの再配分とpath別raw PASS後compileのpipeline化を検討する。CPU/RAM/workerに恣意的な低い上限を設けず、共有OS/venv/scheduler/他jobへ変更しない。実行中SOURCEの書換え・正常計算の不要な中断を避ける。

公開対象は安全な科学要約・資源統計・SOURCE/索引・契約/STOP。C6aの非公開raw archiveを追加公開しない。将来新payloadの公開範囲も別に明示し、内部CPU/RAM・絶対path・資格情報・非公開rawを除外する。専用branchへcommit/pushしremote SHA/tree/公開bytes照合後mandatory STOP。

**次に必要な許可：ユーザー／GPTが第1節の未確定事項を選び、Codexに実装・人工検証・必要な限定数値設計・sampling/compile/postcheck/統計の対象とcall上限を明示する。** この草案の公開だけで科学実行を開始せず、B4-2再計算・B6・長q・全RPE・詳細科学レビューへ進まない。
