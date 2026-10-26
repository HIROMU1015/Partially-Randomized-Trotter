# C1：H2／旧H4／新H4／H6の保存結果統合・baseline再評価

2026-10-26 JST。起点 `3a8356a7843d3cb46bb8069955640381b490ac4c`。

**新H4のCDF8／XDF8では、共通precision条件でもCDFの条件付き総RZが11.85%／11.99%小さい。旧CDF8／XDF10の約27%は別条件の履歴として保持する。** H2の約36%削減、H6の約2%増加と合わせても、原子数に対する単調な性能変化は示さない。分解・rank政策・codec・native精度・shots取得手続き・合成誤差coverageが揃っていない。

次の技術的推奨は**H4 XDF9の表現・own-ground／alias・PFだけを調べる95-call実験**。[A／B設計・判断根拠](research_c_c1_next_minimal_experiment_selection_20261026.md)と[人工検証・未承認契約](research_c_c1_h4_xdf9_preparation_20261026.md)を別記する。推奨は正式採択・科学実行許可ではない。

本書はユーザーが許可した保存scalarの転記、SOURCE・来歴の技術監査、次期設計である。新しい実分子計算・保存配列の数値再解析・shots再算定・費用再集計・重要な科学レビューは実施していない。科学call0。

## 共通条件と比較の単位

全系は直鎖・隣接1.00 Å・STO-3G、中性singlet。Nα=NβはH2=1、H4=2、H6=3、物理sector次元は4／36／400。通常number CDF/XDF、元g補正κ・核scalar一回・元frame順を保持。決定論全frame ordered symmetric S2、s=1主対象／4対照、δ=.1、m=0..12、kernel時間h=.1/s、coherent反復n=s·2^m。PR splitは非適用。

CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、β_total=π/6、β_stat=π/12、候補単位failure=.01、round×X/Yのα_axis=.01/26。各DFのsector基底状態oracle、状態準備費用除外、ideal IID Born law、head＋kernel反復＋axis readoutのserial interrogation資源モデルに条件付ける。4候補同時・事後選択・実機noise・状態準備込みの成功保証は付けない。

DF sector誤差はHa、PFと合成のsignal誤差は無次元、RPEの統計誤差は確率／位相である。PFと合成は登録したsignal／位相十分条件へ各一回接続し、DF norm＋guardはWeylと最終RPE区間のenergy判定へ渡す。異種誤差の無根拠な加算、tensor残差からのsector保証、optimizer successからのDF予算認定はしない。

## 分解・初期化・rank選定

| 条件 | H2 | 旧H4 | 新H4 | H6 |
|---|---|---|---|---|
| CDF | ffsim0.0.83・通常C-DF・JAX x64解析gradient | 同左 | 保存CDF8を変更せず再利用 | 同左、修正codecの新1fit最終値 |
| 初期化codec | modified-Cholesky→旧real codec＋2D logm bridge | 同左 | 旧fitのJ/Uを再利用、再初期化なし | 保存XDF14→real_schur_column_gauge_v1 |
| optimizer | L-BFGS-B、initializer tol1e-8、options=None | 同solver／tol、options={} | optimizerなし | 同solver／tol、options=None |
| CDF rank政策 | requested1..3、各1fit、first-passing2で停止 | 上限8を事前指定した1fit | 同じrank8 | 旧codec ladder未達を保持し、修正rank14だけ1fit |
| 保存終了例 | nit5／nfev7、projected gradient停止 | nit68／nfev74、relative reduction停止 | 同じfitの保存値 | nit1／nfev3／njev3、relative reduction停止 |
| XDF生成 | ffsim非最適化modified-Cholesky | OpenFermion1.6.1 eigh由来の保存λ/G | 同じλ/G元順prefix8 | ffsim非最適化modified-Cholesky |
| XDF rank政策 | max3分解→prefix first-passing3 | df_rank=None／df_tol1e-8生成のrank10を対照に再利用 | 元順1..9の保存評価で1..7 NOT_MET、8が最初のPASS | max21／actual18分解→prefix first-passing14 |

`tol=1e-8`はffsim初期分解条件であり、SciPy minimizeの停止tolやsector誤差保証ではない。CDFの目的関数はtensor Frobenius残差の二乗／2であり、sector normそのものではない。旧codecの使用だけを理由に独立検証済みH2/H4最終表現のPASSを取り消さず、修正codecなら低rankや別の資源結果になるとも推測しない。H6ではtensor残差低下とsector誤差増加が併存し、最終DF予算はPASSした。

## DF予算とPF／合成資格

以下は保存値の転記。全8 armが登録CA/8内だが、CDF/XDFのDF近似誤差は一致しない。

| 比較・arm | rank | sector norm [Ha] | guard [Ha] | norm＋guard [Ha] | 予算 |
|---|---:|---:|---:|---:|---|
| H2 CDF | 2 | 1.3176231587153952e-6 | 2.4289938062665934e-12 | 1.3176255877092015e-6 | PASS |
| H2 XDF | 3 | 4.8652059505012625e-16 | 3.0465959187204976e-12 | 3.0470824393155477e-12 | PASS |
| 旧H4 CDF | 8 | 7.666027761392501e-5 | 1.881549094133745e-9 | 7.666215916301914e-5 | PASS |
| 旧H4 XDF | 10 | 3.3751558354542566e-15 | 1.4812598092146554e-9 | 1.4812631843704909e-9 | PASS |
| 新H4 CDF | 8 | 7.666027761392501e-5 | 1.881549094133745e-9 | 7.666215916301914e-5 | PASS |
| 新H4 XDF | 8 | 9.533753411107529e-5 | 1.480439843662034e-9 | 9.533901455091894e-5 | PASS |
| H6 CDF | 14 | .00010689213236556284 | 1.1384527736890936e-6 | .00010803058513925193 | PASS |
| H6 XDF | 14 | .00010247385485275444 | 1.1384527781887395e-6 | .00010361230763094318 | PASS |

H4は`binary64_operation_scale_guard8_v1`、H2/H6は`sector_operation_scale_guard8_proposal_v1`で、同じguard式ではない。工程guardを厳密な浮動小数点enclosureに昇格させず、数値floor近傍のXDF誤差を0と認定しない。

| 比較 | CDF PF最大 s1／s4 | XDF PF最大 s1／s4 | 保存13round資格とcoverage |
|---|---|---|---|
| H2 | .0132802154112／.000829829308650 | .0160182315159／.00100095521050 | 4候補52roundの登録位相条件PASS。controlled full32の全入力／出力数値行列、unitary telescoping＋Frobenius |
| 旧H4 | .0691369307091／.00432133825219 | .068430410／.0042770929 | 4候補52round PASS。controlled full512、保存s1残差と追加s4 reference、三角不等式／Frobenius surrogate。新H4と同じnorm-growth処理ではない |
| 新H4 | 上と同じ保存CDF PF | .0684279507275／.00427693921042 | 4候補52round PASS。full512全入力／出力の直接compiled/reference Frobeniusと両数値行列・shellのnorm growth。XDF8 PF26は専用再評価済み |
| H6 | .1222267558487／.00764308172000 | .1222311163101／.00764335475930 | PF52／合成104行PASS。ancilla×sector400の全800入力、全8192出力・leakageを含む。全8192入力global保証ではない。literal/shared exact-IR alias4組 |

PF最大値は既報の丸め表記を引用し、今回maxを再計算していない。[旧監査の合成距離表](research_c_c1_h2_h4_h6_comparison_audit_20261026.md)も保持。新H4の直接P0/reference FrobeniusはCDF s1/s4が3.289713005970301e-11／2.690823087680708e-11、XDF8が1.5766389176057384e-10／1.5875367826104e-10。H6は全sector基底資格B取得前の有限probe資格Aを別に残している。

全例で数学的telescopingの条件と、入力となるreference／行列／normの数値精度資格を区別する。資格Bは丸めを囲った厳密Cではない。未知のreference形成・expm・Operator・norm誤差を0とは認めない。有限probeだけで全operator上界を主張しない。

## 保存shots・条件付き総資源16候補

[CSV全16行](../../artifacts/research_c_c1_integration_update/2026-10-26/saved_comparison.csv)にRZ depth、CX depth、size、DF各値、元scalar／CSVのpath、科学SOURCE、結果公開commit、参照snapshotを保存した。以下はそこからの転記で、新たな費用集計ではない。depthはserial予定実行の加重和で、全回路のglobal再compile最小depthではない。

| 比較・arm | s | 算定shots | energy評価 [Ha] | 条件付き総RZ | 総CX | 総depth |
|---|---:|---:|---:|---:|---:|---:|
| H2 CDF2 | 1 | 13352 | 0.0012796349488257434 | 3990125812 | 2694155960 | 4893901728 |
| H2 CDF2 | 4 | 13294 | 0.0012796349488257434 | 15699690593 | 10600614368 | 19255632763 |
| H2 XDF3 | 1 | 13364 | 0.0012783173262851165 | 6274985632 | 4194706740 | 7593362128 |
| H2 XDF3 | 4 | 13294 | 0.0012783173262851165 | 24589446313 | 16437661520 | 29755608387 |
| 旧H4 CDF8 | 1 | 13606 | 0.0013549794824010532 | 111798946393 | 67828478608 | 117389854735 |
| 旧H4 CDF8 | 4 | 13314 | 0.0013549794824010532 | 408641594551 | 247923144128 | 429077084077 |
| 旧H4 XDF10 | 1 | 13600 | 0.0012783188045012185 | 153454725962 | 87693651360 | 149364852770 |
| 旧H4 XDF10 | 4 | 13314 | 0.0012783188045012185 | 560693479327 | 320492595456 | 545181461829 |
| 新H4 CDF8 | 1 | 13606 | 0.0013549794824010532 | 113219727909 | 68160609352 | 118146374763 |
| 新H4 CDF8 | 4 | 13314 | 0.0013549794824010532 | 413834761607 | 249137131232 | 431842276925 |
| 新H4 XDF8 | 1 | 13600 | 0.001373656337788953 | 128445454592 | 71444533608 | 120790723934 |
| 新H4 XDF8 | 4 | 13314 | 0.001373656337788953 | 470217718215 | 261546777184 | 442194889173 |
| H6 CDF14 | 1 | 13880 | 0.0013863479083772862 | 520491751108 | 306659760960 | 507573487852 |
| H6 CDF14 | 4 | 13326 | 0.0013863479083772862 | 1761144891569 | 1037643479376 | 1717292262491 |
| H6 XDF14 | 1 | 13880 | 0.0013819296308689773 | 510393076228 | 301300464192 | 498694652908 |
| H6 XDF14 | 4 | 13326 | 0.0013819296308689773 | 1727240281697 | 1019642229424 | 1687651106555 |

全16候補の**保存数値CA判定**は`NUMERICAL_PILOT_CA_PASS`。energy欄は実測energy誤差ではなく、登録した条件付き十分式の数値評価。shots欄はHoeffding十分式による予定X/Y測定数であり、量子samplingを実施した件数ではない。

既報の同s比較はH2がCDF36.412192%／36.152728%削減、旧H4がCDF/XDF比.728547／.728815（約27%削減）、新H4が11.853846%／11.990819%削減、H6が1.978607%／1.962935%増加。今回比率・百分率を再算定していない。

| shots取得手続き | 再利用と限界 |
|---|---|
| H2 | PF＋数値合成誤差から各roundの十分Nを算定。独自の初回小分子経路 |
| 旧H4 | 先行の楽観的Nを固定し、後から合成誤差込みのmarginを検査。Nは再最適化しなかった |
| 新H4 | full512合成profileを含め既存solverでNを算定し、固定N検査も実施。旧XDF10のNをrank8へ直接転用していない |
| H6 | 資格Bのsector-complete合成profileから新たにNを算定。両armのround／axis N一致は既報の観測 |

共通のfailure配分でも手続き全体は同一ではない。新旧H4で候補別shots合計の保存値が一致していることから、roundごとの一致や、shots手続き変更の因果効果0を推論しない。

## H4の約27%→約12%を何に帰属できるか

| 変更・不変点 | 保存証拠で確認できること | この監査では判定できないこと |
|---|---|---|
| XDF10→元順prefix8 | 最初のDF予算PASSは8。2 frameを除き、DF誤差は数値floorから約9.53e-5 Haへ変わった | rankだけを変えた統制実験としての削減寄与率、XDF全体の最適rank |
| XDF Hamiltonian／PF | truncated Hamiltonian・own-groundが変わり、XDF8専用ground／alias／PF26を資格付けた。PF最大値も別の保存値 | DF精度改善／悪化からPF・shots・費用への単調な関係 |
| CDF8 target | 元fit export、κ・scalar・frame順、独立CAR、保存PF／groundは同じtargetとして再利用 | native費用まで同一という主張。旧CDF PF kernelのbitwise一致証明もない |
| Gaussian精度 | 旧isclose／arccos経路からexact-zero／LAPACK zlartg／atan2へ変更し、両armへ適用。U.T・gate種・消去順序を維持 | 実H4におけるGaussian修正単独の寄与。修正効果とrank効果の分離実験はない |
| P0 phase | 公称P0は同じ。共通の公式definition展開に加え、H6由来task-local MPFR160 declared-phase加算精度を両armへ適用 | 実H4のgate／費用効果が必ず0という主張。人工回路のbody一致は実H4の因果ablationではない |
| 合成・shots | 新full512誤差とnorm growthで13roundを再資格付け、Nをsolverで取得。旧固定N margin方式と区別 | 旧新の総費用差をshotsの変更だけへ帰属すること |
| 共有・fusion | 両arm同一の既存完全同basis run共有・fsum角度結合。一般相対Gaussian融合は未導入 | 未観測のliteral/shared追加節約、将来の相対融合後にも同じ順位が保たれること |

native費用自体が変わった直接の保存証拠は次の通り（単一h-step）。旧s4は別hでのP0結果でありs1から外挿していない。

| native対象 | RZ | CX | total depth |
|---|---:|---:|---:|
| 旧CDF8 s1／s4 | 12118／12118 | 7352／7352 | 12724／12724 |
| 新CDF8 s1／s4 | 12272／12272 | 7388／7388 | 12806／12806 |
| 旧XDF10 s1／s4 | 16659／16627 | 9520／9504 | 16215／16167 |
| 新XDF8 s1／s4 | 13944／13944 | 7756／7756 | 13113／13113 |

旧値は保存cost candidate JSONの`calibration.diagnostics[0].metrics`、新値は[new native_metrics.csv](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/native_metrics.csv)。**CDF8の分解が同じでもnative費用が変わるため、27%→12%をrank変更だけの因果効果とは断定できない。** GaussianとP0を独立に切り替えた実分子計算は今回行っていない。

P0公称条件はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、backend・target・coupling・layout等指定なし。H2／旧H4は旧NativeBuilder Gaussian、H6／新H4はprecision providerとstable-phase wrapper。隣接rotation tupleの完全一致だけを共有し、別basis間の相対回転合成はない。H2 literalは未測定、旧H4とH6の保存literal/shared費用は一致、新H4はsharedだけを再評価した。

## 支持できる主張と未支持の主張

支持できるのは、指定した保存候補・共通CA数値十分条件・own-DF oracle・理想IID・状態準備除外・serial費用の下で、H2と新H4はCDFが小さく、H6同rank14ではXDFが小さかったこと。新H4では保存順first-passing XDF8と共通precisionの両armを用いてもCDFの条件付き総RZ優位が残った。これは旧rank10より強いbaselineとの比較である。

支持できないのは、厳密CA保証、無条件／実機／状態準備込みの総資源優位、CDFまたはXDF全体の最適性、同rank＝同DF誤差、純粋な分解表現だけの因果効果、原子数に対する単調な優位低下、first-passing rank＝最小総費用rankである。未測定の相対Gaussian融合を費用へ加えない。H6の小差をH2/H4と同一精度・同一rank政策によるスケーリング則とは扱わない。

## SOURCE・公開結果・参照来歴

13必須GitHub資料を固定refで取得し、各Git blobがローカル起点の同pathと一致した。[evidence_map](../../artifacts/research_c_c1_integration_update/2026-10-26/evidence_map.json)と[転記utility](../../artifacts/research_c_c1_integration_update/2026-10-26/transcribe_saved.py)に入力hash／転記元を残す。後日のsnapshotを科学SOURCEと混同しない。下表のSOURCEは最終CA／資源評価のSOURCEであり、fit／PF／nativeのSOURCEは対応報告で別に固定されている。

| 比較 | CA／費用科学SOURCE | 結果公開commit・報告 |
|---|---|---|
| H2 | `33c380d8aa792e5119bb3adec5dd07d1e1377214` | `cc0cdb52570993fa4f14f6eb199b583d0411d585`・[H2結果](research_c_c1_h2_execution_20261025.md) |
| 旧H4 | `002d0dc2e1dafd8cb71735c43253fa054e675f6d` | `6202f036457940532aeaa2873cd6d9e6f123c0ca`・[旧精度／費用結果](research_c_c1_stage3_deterministic_accuracy_execution_20261025.md) |
| 新H4 | `18045333a1438740160d7b9e44e78139ad59f2b5` | `3a8356a7843d3cb46bb8069955640381b490ac4c`・[新CA／費用結果](research_c_c1_h4_ca_resources_execution_20261026.md) |
| H6 | `55b3d86817308a8d676221982aae13e1a6e3bed4` | `5385c6ef01879ee8a70734bfc0d5ae8999dd6186`・[H6 CA／費用結果](research_c_c1_h6_ca_resources_execution_20261026.md) |

H4の追加証拠：[prefix1..9](research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)、[XDF8 PF](research_c_c1_h4_xdf8_pf_execution_20261026.md)、[両arm共通native SOURCE862a97d…／公開f4de9f62…](research_c_c1_h4_native_fair_execution_20261026.md)。H6：[修正fit](research_c_c1_h6_corrected_fit_execution_20261025.md)、[保存rank14 PF](research_c_c1_h6_saved_pf_execution_20261025.md)、[stable-phase native](research_c_c1_h6_native_stable_phase_execution_20261025.md)、[全sector基底合成](research_c_c1_h6_synthesis_bound_execution_20261026.md)。[旧監査](research_c_c1_h2_h4_h6_comparison_audit_20261026.md)／[旧A・B設計](research_c_c1_minimal_experiment_design_20261026.md)は書き換えず履歴として保全した。旧監査の「H4低prefix未評価」は当時の状態で、今回までの保存結果が更新した。

## 残る判断とSTOP

ユーザー／研究Cは次期案の正式採否と、固定95-call範囲の科学実行可否を判断する。Codexは今回、統合・技術準備・人工テスト・公開照合まで完了してSTOPする。XDF9のPFが数値適格でも、native／合成／CA／shots／総費用の追加準備・別許可が必要。重要な科学レビューも別途依頼・許可が必要で、本書はそのレビュー記録ではない。
