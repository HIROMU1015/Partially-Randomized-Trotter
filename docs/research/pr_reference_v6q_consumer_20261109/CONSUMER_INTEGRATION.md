# 外側consumerの接続と入力契約

[REPORT](REPORT.md)の固定科学条件・基点commitを使用する。新実装は[pr_reference_v6q_consumer.py](../../../src/trottertracks/pr_reference_v6q_consumer.py)。今回実行したのは人工fixtureだけである。

## 1. 接続構造

```text
固定cache bytes → hash確認 → 既存common() → FixedInputs（同一prep）
                                          │
保存18round → bind_round(m/q/r/bits) ───────┤
                                          ↓
canonical_source(metadata, prep) [実producerは未実装]
  → DeclaredCanonicalEvents(cfg, dist, events, law, producer_source)
  → cache/prep/tail/λ/metadata整合を確認
  → 既存reference_axes_for_saved_round
  → 旧assemble → 旧F/G → m17のみB1 → 旧core → 旧X/Y wrapper
  → X/Yそれぞれ固定Compiler A
  → 17q/1clbit回路へreset(8…15)、その後nativeをcompose
```

canonical_sourceを省略するとNotImplementedErrorで停止する。既存NumPy conditional samplerへ自動fallbackしない。law文字列が一致してもproducerの分布を認定せず、返却provenanceは常に`NOT_ESTABLISHED_BY_THIS_INTERFACE`である。これは承認証明やgrantではなく、科学入力の契約とその未資格を示すだけである。

## 2. Cache・cfg・event provenance

元科学ref `f2638a7845685a119619382b850441f5d8d33713`のcommon.pkl.xzを入力正本とする。解凍pickle SHA256は`68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。factoryは実ファイルのhashを確認し、coldな既存common loaderへpathを一時的に渡して読み、そのprocessの環境変数を元に戻す。contextをshot間で再利用し、毎shotのhash/unpickleや別cacheへの再bindをしない。

別のprepがcommonのLRUに入ったprocessはfactoryで拒否し、既存cacheを消去しない。consumer時にも実builderが使うcommon()の返却objectが同じかを確認する。ファイル・SOURCEは実行中も固定されるという既存条件を保持する。手作業でcontextを偽装した場合や、prepの後からの書換えを防ぐ認証基盤を構築したものではない。

supplierへ渡すprepはこのcontextと同一である。返却側でも同一性を照合し、cfgのtail ID/hash・λを固定prepへ結び付ける。既存callerがr/h/K、distのorder0/2・step整合、event長q*rを検査し、旧builderはbasis ID/hashとSOURCE順序を検査する。supplierのeventsは同一objectを渡し、再抽出・並べ替え・再sampling・角度再計算を追加しない。

**まだ保証できないもの**：単一event列、同じ長さ、law/sourceという申告だけではfresh independent bitsや生成確率は証明できない。canonical producerには次がSOURCE全体で必要である。

- 保存bitsでのinteger floor CDFを使い、order／componentの各drawにfresh independent uniform bitsを使用すること。
- 同じ固定component順とnominal float probabilityをexact rationalとして正規化する既存規約を保持すること。
- 既存_make_eventのrotation index／ordered products、符号・scalar、angle／normalizationを保持すること。finite-bit近似diskの代わりに独自IS weight等を追加しないこと。
- RPEの各shot／axis／roundの独立性、測定範囲・normalizationと既存保証の対応を維持すること。

integer_cdf／draw_cdfは再利用候補であるが、それらからevent列までのproducerは今回実装していない。PCG64のchoice、rare/nonrare条件付き価格cohort、他Trackの別条件samplerは同一lawの証拠ではない。

## 3. Compiler／reset・relative phase

既存[transfer.sample](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_qualification.py#L227)にはreset helperが独立関数としてなかったため、その4行の規約を_compile_and_resetへ移した。旧関数は変更しない。

compile_reference(axis_circuit)の後、QuantumCircuit(17,1)へreset(range(8,16))を置き、nativeをinplace composeする。**処理順ではcompileの後、実行順ではinterrogationの前**である。system register 0…7とancilla16をresetしない。nativeの17q/1clbit形状を確認する。resetをCompiler Aへ再入力せず、新pass・新control・追加inverse・角度最適化はない。

旧F/G、actual SOURCE inverse、D融合、basis共有、2 register・2 CSWAP cascade、relative P位置・wire、X/Y wrapperを同じ関数が生成する。m17の明示opt-in時のP角だけが違う。composeに伴うnativeの共通joint scalarと、信号を変えるrelative Pを混同しない。追加位相操作はなく、既存compile/resetと同じ経路である。

固定Compiler AはP/RZをinverse cancellation対象にせず、symbolic P→1 RZを保つ。最終cancelもCX/Xのみ。resetの挿入はRZを増減させないので、既存contextでのRZ単価不変性を維持する。別compiler、追加control、wire/register変更、resetのlower前移動、gate列再構成、angle cutoffは対象外。T-countや実機時間の不変性は主張しない。

## 4. Q3-N・V6-Pを移送する条件

Q3-Nの13 basis／135 generator／ℓ=65とnormal域、旧M/B_normは、同じcache・SOURCE文法・適格event生成と旧correctly-rounded binary64 add/sub/remainderモデルを前提とする。providerとcallerはbyte不変更で、binary入力取得・exact半角／F+G和・最後1回RNをそのまま使う。δ_B1≤B_norm+2uMを別precision／別periodへ流用しない。

consumer側のcompile/resetが旧経路であることは静的・人工に確認できた。残るcanonical producerのSOURCE資格が成立し、同じfresh independent law／測定normalization／固定compiler環境で運用される場合に、V6-Pの626 shots/axisと総RZの既存条件付き評価へ接続できる。今回はそれらの数値を再計算していない。

producer資格が未成立のため、現時点で実装全体のRPE成功や総RZ優越を無条件に認定しない。適格event列の全support位相保証と、その列を選ぶ法則のcoherent mean保証は別である。新しいcache、sampling法、phase policy、compiler context、CT emitter等への保証拡張も行わない。
