# H8 非相殺COST_COMPARISON：一回実行結果

承認されたcost stageを一回完了した。1,900行・5精度×2倍率×旧32/184の20比較、driver/launcher native exit0・全回収・最終gate PASS。
非相殺184基準のPR削減率（倍率1）は16.62%、34.68%、35.13%、39.28%、−29.09%である。
ε=.0005の負値もそのまま保存した。最良費用は全条件でモデル推定を含み、PRは確認n32の有限標本点推定である。
厳密bias保証・全格子実測・母平均優位を主張せず、研究方針は判断しない。mandatory STOP。

## 固定条件・来歴

linear H8/隣接1.00 Å/STO-3G/保存tol-only DF rank28/16 spin modes/sector4900/T=.8。
q=1/2/4/8、ε=.05/.01/.005/.001/.0005、allowance倍率1/10、α_axis=.025、P=0。
state preparation除外のshot-weighted full measured Hadamard wrapper RZ。
旧32＋追加152の決定論184候補とPR確認union6・各n32を比較し、探索n8は混合しない。
非相殺biasはB0/B0I:abs(discard)+abs(PF)、full PF:abs(PF)、PR:abs(outer_PF)+abs(finite_RTE)の経験的診断。
保存combined u・normalization・headroom・shot政策は凍結D0のまま。完成済みshotを再利用した。

- SOURCE：`ac932974a0f250be0df5720e1b6dad06f9ff9535`。readiness REVIEW/remote HEAD確認：`4d923ebac693a221fc5ecb3f73cdc55b29318a42`。
- plan canonical SHA256：`f171099baeb4fdd56fb40748292db1c70074479b711702f360c9d88921fc96ad`。
- 一回grant canonical SHA256：`0e3c65c017fcc3512a536f77d18a9251acc07626840d51895a00fd5ff8c15344`（消費済み。原本とlockはprivateに保持）。
- input contract byte SHA256：`8960fdc1df16b71fd1afe73a409daa7ba05e26b36a7a5b8bee788b13fc213ce0`。入力46件/11,281,117 bytesを実行前後に照合。
- config canonical SHA256：`3f5c9e7e5f97360db632a8d838daf15f4ba786843614c12174b87295084b28af`。
- B1 model：`8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`、compiler：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。
- [結果bundle](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/README.md)、[全20比較のexact CSV](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/comparison_summary_exact_v1.csv)、[全1900行](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/accounting_rows_v1.csv)。
- [準備131人工検証](track_a_h8_noncancellation_cost_readiness_20261017.md)。今回、同じ人工campaignは繰り返していない。

元planの未承認templateを変更せず、このユーザー承認を新しい有効grant/config/sealに結合して一回実行した。
SOURCE checkoutは正確なSOURCE SHA、結果・資料は別commitに分離する。
完成済みshot `3f902229cef9da26cb6feeff09eb35860b6115a6`/SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`、通常Phase C `6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086`/SOURCE `931755c4c84aa0751dbde7ddf84cb5187b5172ae`を保全。
B2 `fef2db7a79886ab682b652da7266b463c98c7d3a` の軸別SOURCE・最終登録4軸PASS・native/reap・2%未満・compiler/profileを結合した。
保存3軸＋新規1軸の最終受入を保持し、過去STOP・grant・ledger・未settle予約を変更していない。

## 通常biasとの削減率比較

削減率は100×(1−G_PR/G_det)。負値はPR点費用が決定論的最良費用より高いことを表す。
差は「非相殺−通常」のpercentage points。通常Phase Cは保存結果を参照し、再実行していない。
`USUAL_SIGNED_SAVED`と`EMPIRICAL_NONCANCELLATION`を分離し、厳密bias保証はnull。

|ε|倍率|通常 旧32 %|非相殺 旧32 %|通常 184 %|非相殺 184 %|差 旧32 pp|差 184 pp|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0.05|1|28.767874|39.480266|-7.136909|16.624251|10.712392|23.761161|
|0.05|10|28.767874|39.480266|-7.136909|16.624251|10.712392|23.761161|
|0.01|1|52.717979|52.720260|33.092838|34.684864|0.002281|1.592026|
|0.01|10|52.717979|52.720260|33.092838|34.684864|0.002281|1.592026|
|0.005|1|52.733612|52.738917|31.306608|35.125323|0.005305|3.818715|
|0.005|10|52.733572|52.738900|31.306549|35.125267|0.005328|3.818718|
|0.001|1|56.120547|56.147848|22.961345|39.283346|0.027301|16.322001|
|0.001|10|56.120424|56.147870|22.961037|39.283314|0.027446|16.322277|
|0.0005|1|23.833234|23.833234|-51.140301|-29.091008|0.000000|22.049293|
|0.0005|10|23.832459|23.832459|-51.141772|-29.092101|0.000000|22.049671|

## 最良費用とpaired SE

以下は表示用に丸めた値。正確な分子/分母と全44の適格PR候補条件のSEはCSV/JSONへ保存した。
SEは費用と同じRZ単位であり、信頼区間・選択後winner保証・母平均保証ではない。

|ε|倍率|旧32 G|184 G（モデル）|PR n32 G|PR paired SE|
|---:|---:|---:|---:|---:|---:|
|0.05|1|2.26206918e+09|1.64196218e+09|1.36899826e+09|7.43743052e+06|
|0.05|10|2.26206918e+09|1.64196218e+09|1.36899826e+09|7.43743052e+06|
|0.01|1|1.28579554e+11|9.30750250e+10|6.07920792e+10|2.60335063e+08|
|0.01|10|1.28579554e+11|9.30750250e+10|6.07920792e+10|2.60335063e+08|
|0.005|1|6.27225316e+11|4.56932489e+11|2.96433476e+11|1.26944215e+09|
|0.005|10|6.27225626e+11|4.56932489e+11|2.96433731e+11|1.26944324e+09|
|0.001|1|3.50268024e+13|2.52978475e+13|1.53600067e+13|4.54798615e+10|
|0.001|10|3.50270313e+13|2.52979870e+13|1.53600993e+13|4.54801357e+10|
|0.0005|1|1.86504990e+14|1.10042382e+14|1.42054819e+14|4.20613979e+11|
|0.0005|10|1.86506889e+14|1.10043689e+14|1.42057710e+14|4.20622539e+11|

|ε|通常184 winner|非相殺184 winner|非相殺旧32 winner|PR winner|
|---:|---|---|---|---|
|0.05|B0I_p11_q1_2nd|B0I_p15_q1_2nd|B0I_p21_q1_2nd|B2_p7_q2_2nd_r4_K2|
|0.01|B0_p15_q4_2nd|B0I_p15_q4_2nd|B0I_p21_q4_2nd|B2_p7_q4_2nd_r4_K4|
|0.005|B0_p15_q4_2nd|B0I_p15_q4_2nd|B0I_p21_q4_2nd|B2_p7_q4_2nd_r4_K4|
|0.001|B0_p15_q8_2nd|B0I_p15_q4_4th|B0I_p21_q8_2nd|B2_p7_q8_2nd_r4_K4|
|0.0005|B0I_p15_q4_4th|B0I_p16_q4_4th|B1_p28_q4_4th|B2_p7_q8_2nd_r4_K4|

winnerは倍率1/10で同じ。IDのB1はfull PF対照、B2はPRを表し、研究stage名とは別である。
ε=.05の旧32ではB0I_p21_q1_2ndとB0_p21_q1_2ndがexact tieで、lexical ID規則に従い前者を記録した。
184側winnerの単価は全条件モデル推定を含む。p15のS2はB1 fit範囲内、p15/p16のS4は登録4軸以外のモデル推定である。
旧32 winnerは今回すべて保存実測単価を使用するが、全登録候補の費用が実測済みという意味ではない。
ε=.0005の旧full PF単価は元runtime/profileのORIGINAL_OR_UNKNOWNという限定を維持する。

## 適格性・候補変更・費用来歴

|ε|旧32 通常→非相殺|184 通常→非相殺|PR6 通常→非相殺|
|---:|---:|---:|---:|
|0.05|24→21|156→140|6→6|
|0.01|13→13|105→105|6→6|
|0.005|12→12|90→90|6→6|
|0.001|6→6|60→60|2→2|
|0.0005|6→6|57→57|2→2|

適格候補数は両allowance倍率で同じ。全1900行は適格948・不適格952・未解決0・適格費用欠測0。
適格費用の区分はモデル条件付き788行、保存実測116行、PR有限標本平均44行。

ε=.05では通常winner B0I_p11_q1_2nd（184）とB0I_p14_q1_2nd（旧32）が非相殺で不適格となり、別候補へ切り替わった。
ε≤.01は適格集合が同じでも保存shot重みが候補ごとに変わるため、最良候補・最小費用は変わる。
ε=.001では184 winnerがB0 p15/q8 S2からB0I p15/q4 S4へ、ε=.0005ではB0I p15/q4 S4からp16/q4 S4へ変わった。
単価政策は固定のままであり、結果改善は新しいcompileや再fitによるものではない。
[各winnerの保存shot比・旧winner適格性・単価来歴](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/winner_provenance_v1.json)を保存した。

PRのpaired再現gateは適格44条件をREPRODUCED、union6×n32のpairing gateは6候補PASS。
他16のPR候補条件は不適格であり、費用0とは扱わない。
paired SEは凍結D0 weighted_metricで非相殺shot重みから本runで計算した。
選択されたPR winnerでは保存shot重みが通常biasと同じになり、費用・SEの値も一致した。通常SEをコピーした結果ではない。

## 科学的限定

- 全条件の強化184 winnerはモデル条件付き。B2登録4軸PASSは全prefix/q格子における2%未満誤差の保証ではない。
- S2のprefix14〜21外を外挿と表示。未実測S4の精度は全格子保証なし。モデル係数・候補・精度・shot政策を変更していない。
- PRは保存確認n32のpaired有限標本。選択後winner・母平均の優位・厳密bias certificateは認定しない。
- 確認trajectoryのTaylor次数4は観測0、次数2は4。探索側の次数4観測1を確認へ混合せず、未観測層の母平均費用を0と扱わない。
- 旧prefix19のruntime SHA原欄欠測、共有reference/scalar、数値exact_truncated、経験的u、旧B0取得STOPなどの来歴限定を結果JSONへ引き継いだ。

## native終了・資源・保全

実行回数1、再実行0。SOURCE23/input46の事後SHA一致、one-shot lock保持、全owned process回収、残存0。
[終了receipt](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/native_completion_v1.json)と[資源観測](../../artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/resource_receipt_v1.json)を保存した。
launcher wall 1.396095秒、driver CPU 0.818726秒、driver peak RSS 64.613MiB。
固定SOURCEはdriver単独wall開始点を記録せず、fresh gateを含む監視区間1.175528秒を保存している。
正確なdriver単独wallはnullとし、後から推定補完しない。launcherのwallと混同しない。
CPUは実際の許可範囲から[0]を選び、自runのdriverだけに適用。OOM増加0、PSIは観測のみ。
実行前の実効空きメモリ468.15GiB、filesystem/block/inode/quota gate PASS。
容量・inode数の数値snapshotは実行後観測と明記し、起動前の未保存値を補完しない。

既存SOURCE・shot結果・通常Phase C・B2原STOP/grant/ledger、無効grant、他worktreeと共有環境を変更しなかった。
新signal/compile/sampling/SCF/DF/state/fit/shot stage再実行/通常Phase C再実行はすべて0。
有効grant原本・消費済みlock・private原raw・認証情報は公開せず、digestと来歴だけを結合する。

次の担当はユーザー/GPT：本結果の限定を踏まえて研究方針・後続検証を判断する。
Codexは追加科学実行・compile・sampling・model変更・科学reviewを開始せずmandatory STOP。
