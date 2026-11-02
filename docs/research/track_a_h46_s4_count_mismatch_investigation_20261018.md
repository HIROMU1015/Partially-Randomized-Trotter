# H4/H6 S4：理論モデルとnative explicit loweredの8行差

状態：**TECHNICAL_INVESTIGATION_COMPLETED / mandatory STOP**。
原因機構と発生段階をsource・保存総数・人工fixtureから特定した。
実H4/H6のblock/qubit別zero判定と、係数統一後の実総数一致は未検証である。
元の8行の値とUNATTRIBUTED監査は履歴として変更していない。

[資料入口・機械可読証拠](../../artifacts/resource_applicability/track_a_h46_s4_count_mismatch/2026-10-18/README.md)。

## 範囲と来歴

起点REVIEW `7e454b91185ed7b0f92a3790f800ba5d65d5f156`、
元の科学SOURCE `536a3671c0ec2939ff3886e40467db26a27ec65a`、
native provider `217c971dd537b1cc7fbc13742bc0f6c263ff7f45`。
専用branch `track-a-h46-s4-count-mismatch-investigation-20261018`。

H4はlinear 1 Å/STO-3G/legacy DF12/8 modes/T=.8、full PF p12/q1/S4。
H6はlinear 1 Å/STO-3G/tol-only DF19/12 modes/T=.8、full PF p19/q8/S4。
いずれもsymmetric-directional、full Hadamard wrapper、state preparation除外。
保存compiled設定はQiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。
今回Qiskitはimportせず、保存compiled値だけを参照した。

必須5監査JSONと元報告を参照し、same-event全84行のprovenance digestを照合した。
NATIVE_MIXEDの対象は28行で、そのうち8行不一致・20行一致。
原因調査に用いた9 source blobは元closureのSHA-256と一致した。
実H4/H6のprimitive再生成・理論費用再計算・回路生成・compile・signal・samplingは全て0。
Hamiltonian/DF/stateの科学array、保存preparation内の実diagonalを読んでいない。

## 8 metric行：native − theory

|系・候補|軸|metric|理論NATIVE_MIXED|native explicit lowered|差|
|---|---|---|---:|---:|---:|
|H4 B1_p12_q1_4th|cosine|RZ|48,272|48,274|+2|
|同上|cosine|CX|21,444|21,446|+2|
|同上|sine|RZ|48,273|48,275|+2|
|同上|sine|CX|21,444|21,446|+2|
|H6 B1_p19_q8_4th|cosine|RZ|1,521,189|1,521,173|−16|
|同上|cosine|CX|667,264|667,248|−16|
|同上|sine|RZ|1,521,190|1,521,174|−16|
|同上|sine|CX|667,264|667,248|−16|

保存raw命令差もH4 **+4**、H6 **−32**で、SX/X差は両系・両軸とも0。
これらは保存値の差の整理であり、新しい実系counter評価ではない。

## 発生段階とsourceで確認した経路

1. counter `pf_schedule`とnative `deterministic_control_plan`は、同じblock順、
   正・負・正のS2合成、forward uncontrolled/reverse directionalを用いる。
   PFのNATIVE_MIXEDはAと同じで、Gaussian境界相殺はしない。
2. ただし負のYoshida係数のbinary64値が異なる。
   `a=2**(1/3)`として、nativeは`-a/(2-a)`、counterは`-a*(1/(2-a))`。
   nativeは`-0x1.b3d16dd72c671p+0`、counterは`-0x1.b3d16dd72c672p+0`。
   正係数は両者`0x1.59e8b6eb96339p+0`。
3. 時間の演算順もnativeが`T*weight/q/2`、counterが`T/q*weight/2`。
   登録q=1/8、T=.8については正のstage時間はbit一致し、負のstageだけ異なる。
   q1の負時間はnative `-0x1.5ca78b1289ec1p-1`、counter `-0x1.5ca78b1289ec2p-1`。
   q8では同じ仮数で指数が−4となる。一般のqにこの限定結論を拡張しない。
4. `from_live`は同じnative preparationのdiagonal・lambda・basis operationsをcounterへ渡し、
   `factory`とnative `_append_block`は**同じ**`ops.df_squared_diagonal_primitives`を使う。
   この関数は`alpha`、各`beta/2`を順番に`rz_angles`へ加算し、
   `_is_effectively_zero(..., atol=0)`で厳密0のRZを除く。
   代数的に0の和でも、時間の末尾bitが変わると浮動小数点の残差が変わり、保持項数が変わる。
5. 各appenderは渡されたRZリストを全て保持する。directionalは1項をCX–RZ–CX、
   uncontrolledはRZへ写す。RZZの厳密0判定とphaseの非zero判定も両側で対応する。
   counter `lower_gate`はangleに応じたpruningを行わない。
   `raw_lowered`もこの**同一counterのgate_cost**をnative命令へ適用する。

したがって、問題は**S4時間の生成 → DF diagonal primitiveの厳密zero選別**の段階にある。
ここまででpretranspileの語の長さが異なる。Qiskitのlowering/opt1へ帰属させない。
一般的なA/B仕様差、sine wrapperのSdg、最終scalar Pの数え方とも区別できる。

## 保存総数によるRZ保持数の切り分け

同じbasis、A構成、scalar/wrapperを前提に、等時間のforward/reverseを1組とする。
全組にわたる、保持RZ項数・非zero RZZ項数・非zero phase数の正味差をそれぞれ
`da, db, dg`とすると、元の明示lowering規則から次が成り立つ。

```text
ΔRZ  = 2 da + 2 db + dg
ΔCX  = 2 da + 6 db
Δraw = 4 da + 6 db + dg
```

保存差を代入すると、H4は`(da,db,dg)=(+1,0,0)`、H6は`(-8,0,0)`。
H6のq=8の反復を考えると1外側step当たり正味−1項である。
これにより**RZ primitive保持数の正味差**を特定できる。
ただし、異なるblock間でRZZ/phaseやRZの増減が相殺している可能性までは排除しない。
実block・qubitごとの発生箇所、各角度のhex、全wordのbit同一性は保存されていない。
本節は保存総数とsource規則からの代数的帰属で、実系を再生成して得た数値証明ではない。

## 人工fixtureと回帰テスト

元sourceの変更なしの関数本体をGit blobからASTで抽出した。
依存はNumPyと、numeric pathだけを記録するmock circuit。
symbolic ParameterExpressionは未検証であり、Qiskit実回路を検証したとは扱わない。
実H4/H6由来のdiagonalを使用せず、人工のidentity-basis、lambda=1を用いた。

|人工diagonal η|q|Δraw|ΔRZ|ΔCX|結果|
|---|---:|---:|---:|---:|---|
|(1,2,−3)|1|+4|+2|+2|H4保存差と同じ署名|
|(1,2,−3)|8|+32|+16|+16|反復による増幅|
|(1,5,−6)|1|−4|−2|−2|逆方向も再現|
|(1,5,−6)|8|−32|−16|−16|H6保存差と同じ署名|

人工対照で**時間だけ**をnative値へ揃えると、全4 fixtureのraw/RZ/CX/SX/X差が0となる。
S2のq1/3/8ではschedule一致。explicit zero RZ保持、zero RZZ省略、phase、
ancilla、符号、両軸wrapper差、非finite拒否、angle非依存loweringを確認した。
全**16 unittest PASS**（保存rowのmetadata照合を含む、0.171秒）。
小さいfixtureのためworker並列化による実益はなく、単一processで実施した。
人工fixtureの一致は、実H4/H6の係数統一後の一致やsignal精度の認定ではない。

## 分類と最小変更案（未実装）

|対象|判定|
|---|---|
|counterのS4 schedule|native referenceを再現する用途ではbinary64互換性の不整合。人工で再現する実装整合上の問題|
|共通primitive producer|正確な0判定という既存規約。入力時間差に敏感だが、そのこと自体をproducerのバグとはしない|
|native生成器|本調査で修正が必要な誤りは特定していない|
|explicit lowering/counting|元の共通規則を使用。8行差をQiskit optimizerの効果とは扱わない|
|A/B・compiledとの差|それぞれモデル仕様・別compiler処理の問題であり、本8行差と分離|

将来の最小案は、counterの係数とstage時間の作り方をnativeと同じ式・演算順へ揃えること。
共有のcanonical helperを使う設計も候補だが、元モジュールを今回変更していない。
epsilonで微小RZを落として数を揃えたり、native側を逆に変更したりしない。
同じ実数式であることは、有限精度wordの同一性を保証しない。
将来修正時も新SOURCE・新profile/closureで結果を区別し、旧値を置き換えない。

## 不足証拠と追加数値検証の最小案

**今回は実施せず、別途明示承認が必要。** 実8行のblock別完全帰属と、修正案の実系受入が残る。

1. 保存preparationをhash照合後、full PF2構成の負stageについてのみ、
   各blockをcounter/nativeの2時間で元producerへ渡す。
   one-bodyを各最大1つ含め、H4最大13・H6最大20 block、**最大66 primitive評価**。
   q反復と両軸の同じ計算は再利用し、新trajectory/control/reference/compileは0。
2. block ID・time/angleのhex・RZ保持index・RZZ非zero index・phase有無を保存し、
   正味`da=+1/-8`をblock/qubitまで照合する。最大block数はmetadataから実行前に確定する。
3. 正stageのbit一致は静的に示したため再評価しない。
   保存総数と局所差の照合までとし、全wrapper生成やQiskit transpileは不要。
   追加対象を自動で広げず、証拠が不一致ならSTOPする。
4. 修正counter全体の新しい理論費用表の発行はさらに別scopeとして承認・SOURCE固定する。
   本案は有効grantを発行せず、承認・runtime authorizationともfalse。

66評価のwall/RSSは未測定。最大12-modeのprimitive listだけを扱う小規模な案で、
実行直前の空き資源・非干渉を確認し、shared環境や既存上限を変更しない。
今回のmockが短いため、重い科学campaignの速度予測には使わない。

## 科学的結論への影響と保全

この差は「NATIVE_MIXEDが常にnative explicit wordと完全一致する」という解釈を否定する。
数学上同じS4でも、counter/nativeの浮動小数点scheduleを揃える必要がある。
一方、保存された実compiled gate数・14 wrapper fingerprint・shot・signal・母集団は変更していない。
本調査はPR/PF比や順位、統計的信頼性を再計算しておらず、並行担当のH4統計調査を代行しない。
モデルがcompiledより大きいことをバグと判定していない。
人工でこの差を除けることは、Qiskit lowering/opt1後の差を説明し尽くしたことにもならない。
旧compiler出力のbit完全同一性、母平均・希少次数の上界は引き続き未認定。

元SOURCE・全結果・grant/ledger/marker/STOPは不変。新規grant・科学workerは0。
既存venv・共有設定・他jobは変更せず、作業・logはhome配下のみ。
primary agentによる技術自己検査であり、別担当reviewやGPT/Claude詳細科学レビューではない。

次の担当：**ユーザー**が、上記最大66 primitive評価による実block別確認を行うか判断する。
承認する場合のみ**Codex**が新しい実行範囲を固定して実施する。
今回の調査は公開確認後に**mandatory STOP**する。
