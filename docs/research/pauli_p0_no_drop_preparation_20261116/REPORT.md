# H4 PR-Pauli：途中微小係数除去を行わないP0と通常候補準備

**LD7・8・9は、通常入口で新規生成した辞書について、それぞれ66/66のP0検査をPASSした。**
基準configurationと各13round scheduleを生成し、同要求の保存再利用も確認した。
信号・価格・Q3の取得は行っていない。

対象はLinear H4、1.00 Å、STO-3G、保存tol-only DF10、保存weighted-Hermitian completion、
同H/state/snapshot、full256／sector(2,2)36。旧P0の失敗・辞書・STOPは変更しない。
原因診断 `bf7b4dc17455f20eec3d646e7e461fed51237917` の採択後に、
[固定範囲](SCIENCE_SCOPE.md)だけを一回実施した。

## SOURCEと再利用

- 数値builder・今回の科学SOURCE：`25a08afafa32af2ca52bca4f63233c38b3af0e6e`。
- 保存registryとimmutable locator優先を含む技術API SOURCE：`9f00ddd86a44b5d1443f29b5f02dbf1db4ef9ab8`。
- 固定P0本体 `fd111f989`、元M1 one-body、JW関数、Pauli collector、saved-input loader、
  descriptor、P1変換、B canonical mapperを再利用。旧stage1 runnerや固定P0 moduleのbytesは不変。
- §10.5照合：旧P0 carrier `9fa9525b3` と原因追跡の保存値をSOURCE付きで利用。
  no-drop shadowの演算子一致は、今回の通常入口による辞書生成・全P0検査の代用にはしていない。
  D2最新確認 `69ef01cdc` のP0停止規則とB/B2の既存生成器に干渉していない。
- 各LD自身の保存descriptorを再利用。LD8は旧P0のFIXED_INPUTSのdescriptorであり、
  旧失敗辞書や他LDの係数を採用したものではない。

## 局所修正

[新builder](../../../src/trottertracks/pauli_p0_no_drop.py)の局所Fermion/Qubit subclassの加算だけで、
one-body、weight付き二乗項のFermion部分和、JW後のPauli部分和の非zero係数を保持する。
元のfactor順・積・weight・JW順・scalar規則を維持し、global class/toleranceを変更しない。
厳密に0となった項だけ除去する。最終の非zero実係数にも正のcutoffを設けていない。

`EQ_TOLERANCE=1e-8`、絶対Frobenius gate `1e-9`、既存Hermitian係数のimaginary許容 `1e-12` は不変。
raw JWはidentity込み481/527/549項。既存collectorが許容内の虚部を実数規約へ変換し、
実部が厳密に0の項を除いた最終非identity辞書は全LDで360項となる。
raw JW全係数も保存した。これは新しい微小係数cutoffの導入ではない。

## 実P0の結果

相対残差は保存DF tail自身のFrobenius normを分母とする。
旧値は既存記録から引用し、元builderを再計算していない。
新値は今回のP0が生成した最終辞書の残差である。

|LD|旧絶対残差|旧相対残差|新絶対残差|新相対残差|P0 PASS|
|---:|---:|---:|---:|---:|---:|
|7|1.1129429626958517e-07|0.00010969972181847215|3.3566476639643337e-19|3.3085551310515895e-16|66/66|
|8|9.5636036061347108e-07|0.0029674095265175074|6.4659218943850571e-20|2.0062561161368657e-16|66/66|
|9|1.3255266620668722e-06|1.0089061758470879|4.3324896928382836e-22|3.2976142487263515e-16|66/66|

|LD|非identity項数（旧→新）|λ_P（旧→新）|新JW identity|保持した保存scalar s_R|
|---:|---:|---:|---:|---:|
|7|184 → 360|0.00042858697343968894 → 0.00042859681055870485|4.5302064712666724e-05|4.5302064712666724e-05|
|8|101 → 360|0.00013740747868204949 → 0.00013779703426245883|1.2226628190470538e-05|1.2226628190470539e-05|
|9|1 → 360|1.098752230101463e-08 → 5.4738243453522907e-07|5.1933511224578921e-08|5.1933511224578927e-08|

λ_Pは最終Pauli辞書の非identity係数L1であり、DF descriptorのλ_Rとは区別する。
元と新のscalar/λ/hash/判定、tail norm、全round identityは[COMPARISON.json](COMPARISON.json)に保存した。
新scalarとの差はLD7=0、LD8≈−1.69e−21、LD9≈−6.62e−24で既定gate内。s_RとHamiltonian定数cは上書きしない。

|LD|旧dictionary SHA256|新dictionary SHA256|
|---:|---|---|
|7|`c115b1ada31d289b53f0125695f3c5f0ea0b0f4efdf57134eaa991cb441a6ac1`|`6fa807053d2fe0fc1d0015b1045eec934da87209fcb24a1d0c9982c0c6e77856`|
|8|`98f18e37ee51fac7ff1904da4bb06988c0c7c82d146c19f2cd83486197b99509`|`1096fb942167fc4f774ffe7e978ddf412a809544cff85b3d38f2d96d49750adb`|
|9|`f964d81e23210950473bd707768975d80361b2ab26a8d0652a6ef86c1189feb3`|`49053fbaa0904c27478cc28be93c43402276b6a45d7ea69c830249e555068d90`|

各66検査は共通51＋対象分割15。全block投影・Hermitian性・保存state lift/norm、full-H分割、
Pauli和、identity/scalar、sector内一致、head/tail/Pauli和のsector保存、確率正規化を含む。
個別Pauli labelは各312個がsector外へ出得るが、合計のsector漏れ残差は固定gate内。
個別eventをsectorに射影した別lawへ置換していない。
過去LD2の証拠はhistorical controlとして出所だけ残し、今回のcontrol PASSと偽装していない。

P0本体はLD7/8/9で約2.216/2.066/1.985秒、全process exit 0。
各保存入力復元1回・full拡張1回・辞書1件。mean/reference/normalizer/eigensolve/sampling/compile/価格/Q3/N/Gは0。
数値gateの失敗・科学retry・再正規化・閾値変更は0。

## configurationと再利用

[通常入口](../../../scripts/prepare_pauli_candidate.py)がB2 mapperを使い、
S2・ΔPF=binary64 .1・K2・h=binary64 .1・M12・元exactκ・P=0と元prior/decoder/誤差契約を保持した。
各roundはqPF=2^m、実幅h、今回のPauli λ_PによるR_outer=1、allocation=[1]。
これは生成されたscheduleであり、信号精度や価格の科学的採択ではない。

|LD|canonical configuration ID|辞書/P0の固定commit|
|---:|---|---|
|7|`925e613adacd4fef37623e0ec426979367dc65d5aaa91e717b92503c9a75272d`|`7ba6fe757a3decb0eee3b9147100e143841ad9c9`|
|8|`9766ce5ec41202db0aab8bb083822cce779c69d522f22e38357102453697ff6e`|`c96bca4d0af0c4db8e213a2c65488f1a2e9e4d47`|
|9|`7a73dbd84662ce0387c044f50a3d4bde0f1939f9ce98cef04b80dab85f5e522e`|`f5bf35c9033abc645c34f23e4ec79bee15a5f3f2`|

全configurationの固定commitは `c034dfa024e7134077001d23ca08bb60a424877e`。
[登録locator](../../../configs/pauli_candidate_preparation/h4_no_drop_splits.json)は辞書・P0・descriptor・configurationを個別に指定する。
旧SOURCE辞書と新辞書はbuilder policy・SOURCE・payload hashで識別し、旧cacheのsplitsを保持したまま
新cacheのsplit_versionsへ保存する。同builder IDで異なる固定辞書が競合する場合も黙って置換しない。

再利用検査はcache付きprepare再入力3件、default/cacheなし3件、新cache3件の計9件PASS。
D2の実際のA_BASE/A_DESIGN request形も各LDで通し、計6件PASS。
全件で同じ辞書・configuration・13round schedule・固定locatorを返し、P0/配列load/signal leafは0。
[再利用記録](reuse_fixed/CHECKS.json)、[D2 request形](reuse_d2/CHECKS.json)。
旧辞書のmeanや価格は新辞書へ転記せず、全LDでsignalは13件REQUEST、価格26軸未取得と返す。

初回の再利用技術検査では、同一のlocal cache configが固定locatorより先に選ばれた。
数値/ID/scheduleの相違はなく、immutable locator優先に最小修正して回帰検査した。
[初回記録](reuse/FAILED_CHECK.json)を保持し、実P0は再実行していない。

## テストと残る範囲

人工・保存metadata回帰は初回55件＋builder version7件、locator修正後の既存12件がPASS
（重複を除く63 test method）。新builder作者の人工検査、Assets担当者の人工検査、およびA親agentの実行・確認であり、独立科学レビューではない。
新旧cache分離、固定SOURCE不一致/改変/FAIL拒否、tiny最終項、全66gate、停止時部分出力、
旧入口保全、固定locator優先を確認した。[TECHNICAL_TESTS.json](TECHNICAL_TESTS.json)。

**configuration_complete=true、signal_13round_complete=false、price_26axis_complete=false**。
P0のbinary64 operator検査は厳密な演算子証明、sampler law資格、十分N、global RMSE、G/winnerの認定ではない。
`dictionary_eligible_for_configuration=true` と、従来の未資格flagは別の意味で保持する。
後段はD2/B2が共通探索で選んだ候補に対して、新辞書hashとlawを照合して処理する。
[HANDOFF](HANDOFF.md)の後はmandatory STOP。
