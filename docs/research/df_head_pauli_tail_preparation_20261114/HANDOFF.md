# A → B/C/D：DF head＋Pauli tailの接続差分

調査・人工4testのみ。productionコード・実H4配列・保存科学値再解析・新λ/mean/B/N/MSE/C/G・回路/samplingは0。[REPORT](REPORT.md)／[条件付き有限案](SCIENCE_PLAN.md)。旧Full PF/P3/O1の科学成果・STOPは保持する。

## 共通identity（案、未採択）

既存B canonical生成を正本とし、同H/state/snapshot・sector・保存completion/fragment order、DF head prefix/one-body、formula/actual幅・全round qPF/qRPE/R/allocation、K/共通κとmapper定義、scalar方針、target/prior/decoder、compiler/control/P0を保つ。

新たに識別すべき物理条件はtail dictionaryである。`DF_ROTATED_Z_ZZ`と`GLOBAL_COLLECTED_JW_PAULI`、label/係数の正本SOURCEとhash、qubit/spin順、s_R、λとSOURCE、係数処理・理想law/実sampler law、event emit/lowering規約を`split`/`law`とprice contextに結びつける。

現legacy physical CIDはsystem/method/formula/prefix/T/q/R/Kから作り、辞書を直接含まない。Rまで一致するとDFとPauliが同じCIDを持ち得る。**旧CID単独で同じcandidateと判定せず、DF結果をPauliへ再ラベルしない。** Bが辞書付きcanonical identity/physical recipeの扱いを決め、Aは独自ID生成器を追加しない。旧IDは旧SOURCEのまま保つ。完全configurationとdictionary law/contextをcache key・受入に含める必要がある。

## Aの変更案

- 既存one-body/weighted factor→JW部品を保存prefix後へ適用するmatrix-only辞書adapter。元H−headを別の元分子Hから作らず、現保存DF-Hの同じtailを使う。
- 全256でoperator identityと個別label sector作用を確認。sector保存が成立する**合計tail/headのideal mean**だけを36で評価し、個々の射影Pauliをinvolutionとして使わない。
- `pr_rpe_signal_provider`のdirect head、paired finite polynomial、独立raw/corrected、exact-tail、scalar一回、matrix power/cache・数値allowanceを再利用。新Pauli λ/scalar/sourceを供給するfactoryと辞書受入だけが主な差分。
- 現`PauliSignalAdapter.ld`はPauli項prefixなのでDF LD1/LD2へ流用しない。旧M1全体runnerや固定P3 VBARを呼ばない。K0の使用は現provider制限変更が必要で、今回は提案のみ。

## Bの受入

raw complex mean、corrected/exact-tail/reference、logB/B/attenuation、数値不確かさ、full/reduced/sector evidence、state/target/law資格を既存mean packetへ対応させる。X/Yの理想pは(1+Re/Im raw)/2、normalizationを測定確率へ二度掛けない。fresh IID trajectory/shot、postselectionなしを固定し、actual samplerの誤差は未取得のまま保持する。

ce0517deのcontrollerは非負偶数Kとsource-bound split/lawを受け取れるが、A provider/integrationの固定DF recipeと旧law bindingを新Pauliとして通すことはしない。新dictionaryのcanonical構成とcarrier受入差分はB管理で定める。既存decoder/risk/状態集約DP・shot allocation・N packetを再実装しない。

固定R・同Kなら同Aのcorrected polynomialは共通、raw/Bは辞書λで変わる。共通κならRも変わるのでfinite-K biasも再評価が必要。旧Hoeffding Nやnominal Nの流用は不可。raw riskが含むPF/finite-K効果にδE等を重ねて加算しない。二時点pilotで全13roundのRMSE/N資格を付与しない。

## C/Dの価格

`append_pauli_event`はsign・product順・order phaseを保つprimitive。`h4_d2_pauli_control.central_control_body/_rotation`は支持上のbasis changeとparity ladder＋CRZ、逆順uncomputeを行う。aggregate phaseはancillaへ一回転送し、identity rotationは現D2の対象外なので抽出scalarとして扱う。

現`h4_structural_kernel.Model`は324成分/9basisとDF native wordを固定する。basis切替項を0にするだけでPauli期待価格にはならない。Pauli labelのsupport・X/Y/Z basis、product gateとphase、head入口/出口・隣接簡約からSOURCE付きtransition/rewardを作り、既存MeanKernel合成/powerを利用できるかCが判断する。構造Sをcompiled Cと同一視しない。

現Dの`NativeRuntime.context/sample`はDF preparation/hash/symbolic_tailへdispatchする。Pauli lawを偽装してこのtailへ渡さず、head/control/wrapperと価格保存部分だけを再利用する新contextが必要。P3/O1価格は別prefix・law・identity・K・q/R・compiler seed/loweringの有限cohortで、新平均Cとしては受け入れない。compiler hashだけでlowering一致としない。

旧Pauli D2は`diag(I,U)`へのoperator照合があるが、現在のDF head/controlとの混成wrapperが同じRe/+Im coherenceを与えることは未検証である。current `symmetric_directional`というprofile名だけで代用せず、scalar/order phaseを含むwhole controlled作用をC/Dの次scopeで照合する。

P0/P1では価格取得0。次の最短candidateはLD1/m0 full path X/Yの2 compileを構造・native接続確認に使う案だが、別承認・C/DによるSOURCE/seed/lowering固定が必要。全RPEのE[C]にはhead境界とnormal-law全supportを含むmodelとdomain検証が別途必要で、巨大長round回路を最初から作らない。

## 次の判断とSTOP

- ユーザー：LD1のみ又はLD1/LD2のP0 operator検証、必要ならP1限定meanまでの一括範囲を選び、明示承認する。辞書・K0・新κ・新shot policyの採択とは区別する。
- Codex A：承認された場合だけ小adapter/runnerを固定し、同H/scalar/sector/meanを有限検証する。
- Codex B/C/D：公開仕様を参照し、担当するcanonical受入・risk・構造価格・lowering差分を別の指示範囲で扱う。今回他担当へツールで依頼・変更していない。

価格単価やλだけで改善を認定せず、G=ΣN E[C]でFull PFと強いTruncated PFを含めて比較する。今は十分N/global RMSE/C/G/winnerは未評価/null。公開・remote照合後mandatory STOP。
