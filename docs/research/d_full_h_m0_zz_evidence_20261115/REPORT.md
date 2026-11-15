# D1：Full-H κ2 m0・ZZ対象0件の保存証拠照合

**保存証拠だけで両軸の再判定を提案できる。ただし、ZZ実績条件を適用外にするだけでは不十分。元runのparity戻り値とlegacy回路は未保存であり、対象0件での固定SOURCEの分岐同一性を代替証拠として認める規則案が必要である。** 今回、保存event・headなしのcontext・生成SOURCE・軸別QPYを接続して置換前の対象0件を確認し、既存の読取関数でscalar/event位相とX/Y一致を確認した。元runの未保存戻り値をPASSに補完していない。正式採択・単価登録・総額算出は未実施、C_0X=C_0Y=nullのまま。

開始点はC `426b8c4f14b4a00869c73f7b2ae4ecdc638a0107` の `docs/research/full_h_m0_reference_total_20261115/REPORT.md` / `run01/RESULT.json`。受入0/2・保存24軸接続・総額null・人工/回帰18件完了を引き継ぎ、再実施していない。C branchをmergeしたという意味ではなく、固定済み成果を読取参照してD1既存branchへ今回の差分だけを公開する。

## 固定対象・証拠

H4 linear 1.00 Å/STO-3G/保存tol-only DF10、one-body込みFull-H pure RTE・L_D=0・finite K2・κ2・P0、FULL_H_2a9a68a11d08af234a8c、m0/T=.1/q=1/R=9。cohort `D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1`、index0、seed `15612141344395026302`。独立path1、同じpathのX/Y。symmetric-directional、新ZZ lowering、測定付きfull controlled Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を変更しない。

| 役割 | 固定commit・path・selector |
|---|---|
| 実行SOURCE/入力 | `7f85f20e38af4df1a649ac239e57bfe76d0899ee`、`scripts/run_full_h_m0_reference.py`、`docs/research/d_full_h_m0_reference_20261115/FROZEN_JOB.json` |
| 元結果 | `b25d7da6dedebc4cd22e8d6f00e6de75d7072292`、`docs/research/d_full_h_m0_reference_20261115/`（以下「元結果root」） |
| 元event | 元結果root `run01/NATIVE_TRIAL/events_PRIVATE.json`、`events[0][0..8].application_sequence[0]` |
| 元準備・数値検査 | 元結果root `run01/H4_context.json`、`run01/RESULT.json.child.context_check`、`run01/NATIVE_TRIAL/result.json.checks[0..6]` |
| 軸別locator正本 | `74cd9c2625a6338793f1beef7b434c765a3a7e67`、元結果root `HANDOFF_C.json.axis_rows[0]`=X、`[1]`=Y |
| 採択済み価格規則 | `4a1cb0d7364fece475437749129fdce740fd8fc4`、`docs/research/full_h_m0_price_rule_20261115/PROPOSAL.json` |

[EVIDENCE.json](EVIDENCE.json) は各入力の固定SHA/pathと内容hash、9 applicationのselector、軸ごとの元receipt/QPY locator、今回の読取検査を保存する。[read_saved_evidence.py](read_saved_evidence.py) は固定Git blobと保存QPYだけを読み、元P1の純粋なphase/XY読取関数をASTから再利用する。科学runnerやproviderをimportして実行しない。

## 1. `replaced_controlled_zz > 0` が意味すること

`7f85f20e3:scripts/run_full_h_kappa2_p1.py:332` の `pilot` は、new evolutionと同じrequestsによる未compile legacy evolutionを作り、`01e068af7e06d9f1f5621a5b3800c30eb32ecb06:src/trottertracks/efficient_controlled_zz.py:58` の `instruction_parity` を呼ぶ。

- **対象が存在するか**：本来は置換前のcontrolled/nonidentity/rotation/support長2から独立に求める。置換後counterだけでは、対象0と処理漏れを区別できない。
- **存在する対象を正しく全処理したか**：`instruction_parity` はlegacyの各3-wire `crzz` を同じwire/角度の`cx,crz,cx`と比較し、その他の全命令順・wire・parameter、qubit数・global phaseを照合する。途中不一致はFAIL、末尾の余剰/欠落もFAIL。成功時だけ置換数を返す。
- **このpathで実装を1回以上通ったか**：`parity.status == PASS` に加えた `replaced_controlled_zz > 0` がcoverage条件である。対象0なら整合性が保たれていてもこの条件は成立しない。置換一般の正しさや全law coverageの資格とは別である。

元の複合条件には、lowering ID、新側crzz残存0、one-body実績>0、scalar値・event位相、native PASS・index・X/Y価格キーも含まれる。P2はさらにX/Y位相・同一U・q/R配分を要求する。今回のSTOPだけから、これらがすべてPASSだったとは読まない。

## 2. counterとは独立な対象0件の根拠

元eventの全applicationを読むと、rotation/nonidentity各9件、central supportは順に `[2],[6],[2],[3],[5],[5],[6],[0],[2]`。product/identity applicationは0、one-body6件、DF fragment0のZが3件。orderはすべて0だが、空のeventではない。対象述語 `controlled and not identity and role=rotation and len(support)=2` を満たすapplicationは0件である。controlled=True・ancilla8は固定native samplerの `make_df_partial_s2_step_request` の引数から確認する。

9 central supportだけでwrapper全体に結論を広げてはいない。以下を合わせて範囲を閉じた。

1. `9b83b96b83f62142c9d87858dc24a94db7ac5d61:scripts/h46_full_h_pure_rte.py:59` の `prepare_full_h` はdeterministic_blocksを空にする。保存contextでも0。従って、tail外の決定論headから対象が加わる経路がない。
2. `af284d853890155bacd19e61850c196bd1d5c660:scripts/h46_long_time_native.py:38` の `compose` は上のtailをpartial-nativeへ渡し、scalarを総Tで最後に一回付ける。`src/trottertracks/resource_applicability/ax2b_native_df_v5.py:169` は空blocksのときtailだけをcomposeする。
3. `c31ef424175c02379eafebe8fc6c7548c1d9aabe:src/trotterlib/df_rte_qiskit.py:214` の `_build` でapplicationを全走査する。basis changeはsystemだけに非制御で付加し、中心の非identity rotationだけが `_append_central_application` へ進む。同file:472でsupport長2だけがcontrolled RZZになる。basis再利用は中心applicationを削除・追加しない。
4. `01e068af…:make_builder_class` が変更するのはcontrolled rotation/support長2の分岐だけ。今回の9件はすべて元1-wire処理へ委譲される。`7f85f20e3:scripts/full_h_p2_acceleration.py` のmemoizationは同じsupport数/角度のgateをcopyし、newとlegacy双方の1-wire経路で有効。fast writerはfingerprintの直列化を変えるだけである。
5. 保存wrapper全命令の種類はXが`crz,h,measure,p,xx_plus_yy`、Yはこれに`sdg`。3-wire命令・opaqueな別subcircuitはなく、ancillaとの2-wire命令は9個のCRZだけ。`xx_plus_yy` はsystem間basis gateであり置換対象のcontrolled ZZではない。basis metadataの全wireも0..7内だった。
6. 同wrapper builder:259が外側に追加するのはH・YのSdg・H・測定のみ。保存QPYでも、ancilla8→cbit0の最後の測定1件と一致する。従って、X/Yの測定部分からも対象が加わらない。

これは**固定SOURCE・保存context/event/QPYに基づく対象数0の構造的根拠**であり、「置換counterを観測して0だった」という主張ではない。元counter自体は未保存。別path、別head、別SOURCE、未知のopaque gateにこの推論を自動流用しない。

## 3. 保存済み・今回照合・未確認の区分

| 項目 | 元runに保存されたもの | 今回の読取技術照合 | 残る限界 |
|---|---|---|---|
| preparation/H/state/scalar/component順 | child.context_check=PASS、H4_context、component_order | 固定提案とのpreparation/scalar一致、head0を確認 | H4再構築・新simulationなし。元検査を再実施していない |
| normal path/cohort/seed/q/R | normal_sampler_calls=1、conditional=false、event/seed、native index0 | 1 request/9事象・固定seed一致 | 再samplingせず、母law/IID/CIの資格を付けない |
| ZZ対象数 | 元eventとcontext、元STOP説明 | 独立な対象述語＋全生成経路で0 | original replaced counter戻り値は未保存 |
| parity | 複合gate STOP、legacy compose完了stage | 対象0なので新builderの変更分岐は不使用、他処理は同じSOURCE/requestsという分岐同一性の根拠 | **元parity.status未保存、legacy QPY未保存。実際の2回路比較は未実施** |
| lowering ID/残存crzz | 固定実行SOURCEはwith-swapを使用 | 保存wrapperにcrzzなし。対象分岐不使用を確認 | captured.builder文字列は未保存。SOURCEと到達経路の根拠であり元runtime markerの実測ではない |
| one-body実績 | 保存application列 | one-body6件、元>0条件と整合 | 対象0のZZ実装を通ったとは言わない |
| scalar/event位相 | scalar値は保存、個別gate戻り値は未保存 | **現在の読取検査PASS**。各軸のancilla命令10件=CRZ9＋scalar P1。全event phase=+1、event-phase P0件。角度`0.033147781341681615`のscalar PがU末尾index594 | 元run gate PASSへの書換えではない |
| X/Y | それぞれ元QPY/compile JSON/fingerprint、XY gate開始stage | **現在の読取検査PASS**。Uは両軸594命令で完全同一、global phase0。XはH-U-H-measure、YはH-U-Sdg-H-measure | 元XY戻り値は未保存。元STOPを変更しない |
| 数値検査 | native resultの7件すべてPASS、phase_alignment=false | 保存error<toleranceを確認。low/high/occupation、X/Y wrapper、X/Y compiled-vector | 再simulationなし。元の限られたstate検査であり全unitary/全law保証ではない |
| 軸別receipt/compiler | 2 compiler JSONと4 QPY、native各軸metrics、個別hash/fingerprint | 元HANDOFFの各hash一致、各axis自身のrecord/compiler/metrics対応 | RZ再計数・fingerprint再生成なし。元publication確認とnativeのmutationチェック完了経路を再利用 |

X fingerprint `b23187eaf78d4fd7a3614460a1c03f22ea83311425478375d4a71a21a25fc332`、Y `5c853c3469cb65c688bce01e3a9bb9c4ca8d6b6666ae3d0929944f58aca0b16c`。元観測RZは各1062という引用だけで、単価欄はnull。96倍・総額・PF順位を判断根拠にしていない。

## Cへ渡す提案上の不足と充足

対象0ならZZ使用実績は適用外にできるという提案は技術的に区別可能である。しかし旧gateをそのまま要求するなら、未保存parity/markerが残り、coverageだけを外しても無条件受入にはできない。

保存証拠のみで両軸再判定を提案する場合は、(a) 対象0の独立確認、(b) 非ZZ処理の固定SOURCE分岐同一性をparityの代替根拠とする明示的な規則、(c) 現在の保存QPYに対するscalar/event・X/Y読取検査、(d) 元7数値検査・軸別receipt・固定入力bindingを併せる。**(b)はcoverage条件の適用外化とは別の証拠方式の変更案**である。Cはこの区別を規則案に残す。元parityの回収・再buildで補う作業は今回は行わない。

## 実施範囲・保全

新sampling/再抽選/build/compile/simulation/RZ再計数/総額集計/18 test再実施はすべて0。小さい固定JSON/QPYとSOURCEを再利用し、独立な読取探索をまとめて実行した。新しい性能frameworkや速度倍率の主張はない。

読取helper初回はproviderの純粋関数に必要な定数 `_PHASE_VALIDATION_ATOL` の取り込み不足で終了した。固定SOURCEのliteralを取り込む最小修正後、読取照合を完了した。科学取得のretryではなく、原run/価格/科学SOURCEには一切変更しない。

元D7/F1、Cの426b記録、元D1取得root/STOP/PRICES、元gate SOURCEを保全。ロードマップ本体は変更せず、「対象0根拠と残条件の証拠区分が揃い、価格未採用の規則案へ進める」をHANDOFF差分とする。本担当だけで実施した技術照合であり、別reviewerによる独立レビューではない。公開照合後mandatory STOP。
