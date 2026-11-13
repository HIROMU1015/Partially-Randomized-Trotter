# A mean → B risk/whole-shot → C/D価格統合

2026-11-13 JST。[REPORT](REPORT.md)／[risk契約](RISK_AND_N_EVALUATION_PLAN.md)／[有限案](FINITE_SCIENCE_PLAN.md)／[readiness](SOURCE_AND_READINESS.json)。今回は設計・metadata照合のみ。新producer/optimizerやC統合moduleを実装・取込みしていない。

## 1. Aの入力：そのまま使える部分

- 公開bundle：`91349e703e0d09ef702ccd108745dfaf1acc29f7:docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json`とLD2。
- config：bundle.configuration_sourceが指す`a9abf8de75bf3432fcd714e868e5c14c8884b8f8`のCONFIGURATION_LD1/LD2。
- numerical implementation：`fefd548ecd56f60d72031b3d385f6f5e6ee95d06`。mean.sourceのraw carrierは`16210932e5dabe7d12d41f35fea0e0a0d7a9b271`。code/sourceと値のcarrierを同一fieldへ潰さない。

各meanのm/identity/law_binding、raw/corrected/reference/exact_tail、finite normalization、bias、uncertainty、reference/state evidenceを保全する。rawはX=Re/Y=Imの±1 mean。correctedで観測確率を作らず、B²やprice sample数を追加乗算しない。LD1/LD2で元physical candidate、split/λ/R/lawはそれぞれ別に維持する。

今回Bの既存prepareへshots=[]で接続済み。N未指定とphysical qualification未取得をblockerのまま残した。inputの数値gateを再判定せず、原値から新しい率・位相・半径・Nを算出していない。

## 2. 評価targetと最小caller

現bundle.targetには`reference_solver_energy_ha`があり、`energy_ha`はない。`truth_proxy_adopted_for_new_simulation=false`、ground資格はfalse。現在の`W.diagnose`はsummary用targetにenergy_haを要求するので、A bundleを直接渡して動くと報告しない。静的SOURCE上はsimulate_preparedの後にこのlookupを行うため、条件付き実行を許すとMC後に失敗する可能性がある。新callerは評価target/契約の検査を**MC呼出し前**に完了させる。今回それを実MCで試していない。

最小案は、原packetを保全したまま採択した`evaluation_target={kind, energy_ha又はtarget域, source, original_target_source, binding, qualification_scope}`を**統計契約/新run側に別記**すること。初回proxyを採択するかはユーザー判断。W.prepare→既存S.simulate_prepared→S.summarize(target_energy_ha=採択scalar)の薄いcallerなら、元mean.targetやdecoderを変更せず対応できる。

再利用が多くなる場合だけW.diagnoseへ同じ明示evaluation_target引数を追加する最小API案を検討する。未指定時の旧有効入力の結果は保ち、欠測targetはsimulation前に検出する。solver値への暗黙fallbackは追加しない。今回はどちらも未実装。mean.targetを書換えたり、prepare済み入力のtargetだけを後から差し替えて検証を迂回したりしない。

## 3. Cの統合形状への受動mapping

C基準：`2f76b5750dd4c4823c122ab04e7629f731fd03a3`、価格接続：`5cc7c07e112ce97205948465c94441cba0042416`。公開module/testの選択取込みと依存版確認が必要だが、今回branch mergeや他worktree変更は不要だった。

Cの`I.signal_input(packet)`は`configuration_id,target,source,means`を保持する。A bundleにはtop-level `source`とmean `provider_binding`がないので、将来は次の**資格を変えない受入mapping**をrisk実行前に用意する。

1. 正本config/物理recipeからI.plan相当のrequest/price_contextをmetadataとして照合する。AのSignalBatch/produce_signalsを呼んでmeanを再生成しない。
2. packet.sourceにはrequest.signal_sourceと一致する元numerical_sourceを保持し、bundle source/carrier/configuration_sourceを別に保存する。実装SOURCEと数値carrierの違いを隠さない。
3. mean.provider_bindingには確認したphysical_context、**mean.sourceと一致するcarrier参照**、元input/reference SOURCEを記録する。対応の根拠が不足する場合はpendingで返す。古いmeanの数値資格を付ける操作ではない。
4. risk producerが実際に消費するpacketを`I.signal_input`で記録し、evaluationにそのまま結び付ける。後から旧S3/旧HP Nへ新bindingを付けて資格を作らない。

source envelopeやbindingの追加は受入viewとして原A値/target/qualificationを保全し、rawが同じであることと科学的適用資格を分ける。現在のA meanはUNQUALIFIEDなので、C I.compareのphysical_qualification gateは依然blockする。A/B診断を行うためにそのgateを緩めない。所望claimに適合する外部受入資格が将来必要である。

## 4. N出力producer：既存機能と最小差分

| 部分 | 再利用/不足 |
|---|---|
| mean/shot受入 | W.prepare/S.prepare、canonical/whole-vector/source/law/target検査は既存 |
| 観測とdecoder | 元PCG64 binomial・exact count→binary64・R3 scalar互換・全trial/statusを再利用 |
| A診断 | 経験的summaryとCP。evaluation_target＋有限caller/消費signal bindingだけが最小追加。十分Nはnull |
| B区間 | bounded-loss envelopeは既存。conditional CIや特定estimatorのall-outcome riskを求めるなら、採択methodに限った小追加/人工検証が必要 |
| C科学的N資格 | 未実装。採択した全risk上側評価、元Minkowski各項、state/target/raw/law/数値移送、finite-K/signed条件と同時被覆が必要 |
| 出力形状 | 既存`configuration_id,allocation,statistical_contract,source,n_packet,signal_input`を使う。診断はn_packet=None、evaluated shot vectorとsummaryは別欄 |
| 正式packet | 全26軸exact N、rmse_evidence/source/error_contract/statistical_contract等を既存B条件で受入。CI fieldの存在やPASS flagだけで生成しない |

資格producerは証拠から所望claimの成立/未達/未評価を分類する部分で、承認証明機構ではない。unknownやCIの閾値跨ぎをinfeasibleへ短絡しない。userが水準を採択する前に万能なproducer/optimizerを作る必要はない。

最小callerを実装する際の人工検査は、target/契約欠測でMCが呼ばれないこと、原A mean/targetが不変なこと、scalar/R3のcount/trial/summary規則、LD/source/law混在拒否、診断N/G=nullに絞る。条件付きCIを追加する場合だけ、p_L=0・zero/tie・同時区間の人工例を追加する。既存の大きなsuiteを新機能と無関係に再実行しない。

現configのpolicy科学statusはPROPOSED_NOT_ADOPTED。将来正式Nを出す際、policy/error contractの採択がidentityへ影響するなら既存生成規則と物理同一性bridgeを用い、元A config/packetのID・statusを直接変更しない。

## 5. D価格と後のcost-aware比較

AのD_PRICE_REQUESTSは2構成×全13round×X/Yのmetadataで、平均E[C]は未取得。C5cc7の`price_requests`/P.bind_resultsはD `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`のrequest/contextを使う。Bがsampling/compile/残差fitを代行しない。

将来の価格はphysical identity、law、compiler/control/full wrapper/測定軸/P0、lowering、SOURCE、DIRECT/MODEL/cohort/domain/区間資格を照合する。Dのpath C/S/Dをmeanとして使わず、有限cohortを全lawへ昇格しない。`price_source`、physical_context、d_request_id/d_context_key、profile_extensionと同時interval contractは元C/B/D仕様を維持する。

価格未取得中に可能なのはrisk-vs-whole-N診断とcoverage。shot総数最小をG最小と表示しない。適格N/meanと資格付き価格が揃い、実Gの許可があるときだけW.compare→I.compare/selectの既存exact積和を使う。未知systematic/covariance/CI、未検証外挿、C1 MODEL false、全law未資格はnull/falseのまま。

## 次の担当・STOP

**ユーザー＋A/B**がrisk水準/target/loss・配分/precision/confidence/有限scopeを採択。**B**が必要最小caller/mappingを人工検証し、別許可後に評価する。**C**は既存統合とsignal_input、**D**は適格な同domain価格を担当する。

本納品は他チャットへの送信や作業変更を含まない。既存SOURCE/input/raw/結果/科学契約/grant/ledger/marker/KEEP/STOPを保全し、commit/push・remote照合後mandatory STOP。新MC・保存再解析・N/G最適化へ自動進行しない。
