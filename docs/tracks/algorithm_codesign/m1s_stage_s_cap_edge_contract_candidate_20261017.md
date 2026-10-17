# M1-S Stage S：cap直下の楽観的shot headroom — 実行契約候補

2026-10-17 JST。基点`fb5519bf3f44392dc050feb9db44057dbc5207b1`。
`SOURCE_PREPARED_CANDIDATE_NOT_ADOPTED / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / MANDATORY_STOP`。

## 1. 今回の範囲と採択境界

ユーザーのStage S技術準備指示に従い、保存raw-H理想信号のTask Aに限ったcap-edge診断SOURCE・未採択契約・schema・guard・人工テストを準備する。正式契約採択、新科学計算、保存信号の再解析、実際の整数shot予測、native compile、RPE simulation、Stage C／Task Bへの移行は含めない。

[GPT Stage Sレビュー原本][review]をbyte保存した。原本SHA256は`f815e886d9e4ff458a2a1398c3b41c4be2e4187e357ba58069eea9c3136b492e`。レビューの科学的推奨と今回の技術準備承認を区別する。`selected_science=null`、`adopted=false`、`coverage_model_adopted=false`、`contract_grants_execution=false`、`scientific_execution_authorized=false`、`run_ready=false`を維持する。

## 2. 保存scopeと全48対照

linear H4、隣接距離1.00 Å、STO-3G、8 modes、保存DF rank12、G3/G6/G9、δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1。保存state・元norm²、非冗長Pauli-tail、同じ概念的独立label数のshared ±／ordinary／independent ±。元108方式行＝72非zero＋36zeroのidentityを保持する。

|既存層|対照数|将来のStage S処理|
|---|---:|---|
|G3、T=1・4：π/3限定gate成立|16|cap-edgeの主対象|
|G6、T=4：π/3限定gate成立|8|cap-edgeの主対象|
|G6、T=1：限定gate不成立|8|`CARRIED_PI3_NOT_SATISFIED_NOT_EVALUATED`。新shot評価なし|
|G9、絶対改善未解像|16|`G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED`。相対比による救済なし|

上表はユーザー指示と固定完了報告から転記したmetadataで、今回の数値再集計ではない。24主対象は既知の結果を用いた**事後的な探索層**であり、独立holdoutではない。全48対照の旧π3区分・旧absolute contrast区分・新Stage S区分を別列で報告する。最良cellだけを報告しない。36zeroはinventoryのみで再計算しない。

## 3. 数学的対象

保存V1の方式行から、exact rationalとして保存された

\[
 A_m^-=\max(0,\widehat A_m-r_{A,m}),\qquad
 b_m^+=\min(\pi,|\widehat b_m|+r_{b,m})
\]

の構成を継承する。対象行では保存`A_lower`と`unclipped_bias_rad`を使用し、後者がcapより小さいことを確認するためminのclipは作用しない。新しくraw信号やHamiltonianを再構成しない。biasは`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`に由来し、モデル差abs(T)Bを再加算しない。

保存π区間を[p_L,p_U]として、真の開端点γ_c=π/3のenclosure [p_L/3,p_U/3]を使う。**端点を許容γとして採用するのではない。** 新πや新γ*を計算せず、保存πの条件付きcoverageを引き継ぐ。

各主対象で直接trade-off

\[
 1\ge a=A_s^->c=A_c^->0,\qquad 0\le v=b_c^+<u=b_s^+<\pi/3
\]

と、保存`REVIEW_TRADEOFF_DIRECT`、`SHARED_LOWER`→`SHARED_HIGHER`、`UNIQUE_TRADEOFF_BOUNDARY`、π3 gateの`SATISFIED`を確認する。逆trade-offやdominanceへラベル交換して定理を転用しない。入力前提の不一致はtechnical STOP、正のmarginの算術未解像は`ARITHMETIC_UNRESOLVED`とする。

共通confidenceとX/Y等配分に対し

\[
 M_m(\gamma)=A_m^-\sin(\gamma-b_m^+),\quad
 K=4\ln(2/\alpha_{axis}),\quad S_m(\gamma)=K/M_m(\gamma)^2.
\]

連続shot比は**対照／shared**方向で

\[
 R_{cont}(\gamma)=S_c/S_s=(M_s/M_c)^2,
 \quad R_{cont}^{cap}=\lim_{\gamma\uparrow\pi/3}R_{cont}(\gamma).
\]

上のtrade-offではd(log R)/dγ=2[cot(γ−u)−cot(γ−v)]>0。従ってR_cont^capは指定された十分設計の共有有利側におけるsupremumであり、一般に開領域内で達成しない。単価・固定費差・整数化を省いた連続proxyの楽観的headroomであって、native価格や物理RPEの改善ではない。

## 4. ceilと開端点を混同しない

γがcapへ下から近づくとS_m(γ)はS_m(γ_c)へ**上から**厳密減少する。そのため

\[
 n_m^{cap-}=\lim_{\gamma\uparrow\gamma_c}\lceil S_m(\gamma)\rceil
 =\lfloor S_m(\gamma_c)\rfloor+1,\qquad N_m^{cap-}=2n_m^{cap-}.
\]

端点のSが整数ならceil(S(cap))は1少ない。SOURCEの`open_cap_integer`はfloor+1を使う。Sの有効区間[l,u]には整数区間[floor(l)+1,floor(u)+1]を対応させ、**両方式とも単一整数に解像した場合だけ**主整数判定を出す。上限が整数に触れてfloorが異なる場合も未解像。結果に合わせたprecision増加・epsilon・grid細分・再試行はしない。

|新Stage S区分|意味|
|---|---|
|`CAP_EDGE_STRICT_INTEGER_WITNESS`|n_s^cap- < n_c^cap-を確認。capより厳密に小さいγの非空開領域に十分shot設計の整数削減が存在|
|`CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED`|端点直下でtieまたはsharedが小さくない。より小さなγに削減がないとは結論しない|
|`ARITHMETIC_UNRESOLVED`|正のmargin、連続比>1または整数境界を採択された固定算術で解像できない|
|carry 2区分|旧G6 T1不成立・G9未解像を保持し、新数値を付けない|
|`NOT_EVALUATED`|未実行またはtechnical STOP後の未評価prefix|

`NO_INTEGER_SAVING_ANYWHERE`や科学的GOを出力する経路はない。全γ領域を完全分類するSOURCE、区間細分、実用γの選定は実装しない。

## 5. 科学条件の推奨（全て次期契約では未採択）

|項目|候補|採否の担当|
|---|---|---|
|Task／γ|保存raw-H理想Task A、γ→π/3^-のcap screen、固定γ/gridなし|ユーザー|
|confidence|`PER_METHOD_ROW`、α_row=1/20（0.05）|ユーザー。72行・全48対照の同時95%ではない|
|統計|fresh-IID、Hoeffding、X/Y等配分、α_axis=α_row/2|ユーザー。物理IID・物理meanを認証しない|
|coverage|`SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY`、`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`|ユーザー。旧platform条件を新区間算術が認証するものではない|
|算術|p=128 bits＋8 guard bitsのoutward dyadic interval、固定精度・refinement0|Codexの技術候補。正式契約で明示採択|
|結果用途|記述的screenのみ、研究GOなし、one-shot・failure STOP・retryなし|ユーザー|

native metric、HD合成、state preparation価格はStage Cへ延期し、Stage Sへ0を代入しない。Task Aではraw→目的位相差は定義上0だが、Task Bのstate品質・prior・全round・implementation/SPAMが0という意味ではない。G9の絶対差未解像や旧v5の未保存full-H/HD直接照合UNDETERMINEDも維持する。

## 6. SOURCE・guard・保存

新namespace：`src/trottertracks/algorithm_codesign/m1s_stage_s_v1`。runnerは`run_m1s_stage_s_v1.py`。既存SOURCEは変更しない。一般的なrational sin/log kernelを旧V1から新namespaceへ明示コピーし、旧evaluator／guardをimportしない。独立H4数値再現SOURCEではない。

順序は、契約metadata読込→正式採択確認→別execution authorization・S/C/A commit／bytes／interpreter照合→入力bytes identity→専用output新規作成・exclusive one-shot消費→science JSON decode→評価→結果保全・STOPである。採択前はGitや科学入力へのアクセスより先に拒否する。正式採択だけでもauthorityがなければ拒否する。新authorization／grant／markerは今回発行しない。

結果schemaは48行を要求し、carry行への新R/S/n値、scientific_GO、global impossibilityを禁止する。入力・SOURCE・C/Aとfailure、未評価prefix、exact interval端点、cap-open、nとN、resource実測を保存する。nは一軸、N=2nはX/Y合計。数値的なPASSと研究GOを別物にする。

## 7. resource候補と性能設計

最新ユーザー指示（2026-10-17）に従い、**CPU core／RAM／GPU／worker／CPU time／wall timeの人工上限は置かない**。旧CPU1・memory2GiB候補や旧V1時間capは履歴として保全し、新SOURCEへ転用しない。GPU禁止ではなく、exact rational scalar kernelにGPU実装がないためCPU backendを選ぶ。

12個の独立cell jobへbatch化し、各cellの3方式marginをcacheして2対照で共有する。worker数=min(利用可能affinity CPU数,独立job数)。job数12は科学scopeから決まる仕事量で、恣意的worker capではない。BLAS／OpenMP／NumPy／Qiskitをimportせず、共有環境の設定や他プロセスに触らない。人工並列テストの2workerはテストfixtureの2jobに由来し、本番制限ではない。

input各16MiB、result各16MiB、failure/log各16MiB。書込先は新output namespaceのみ。call/instruction候補は最大48対照、評価24対照、12cell、36margin、72sin端点call、log1call、各series512項、aggregate bound steps100000、exact rational65536 bits、fraction field1024、文字列25000、refinement0。これらは算術・入出力の有限化でありCPU/RAM枠ではない。resource候補も正式契約採択の対象である。

失敗時は当該Poolが所有する子workerだけを終了し、他ユーザーのプロセスにsignalを送らない。すでに完了したcellのprefix、未評価行、failureを保全しretryしない。実H4入力のruntime・RSS・数値correctness・整数解像性は未検証。

## 8. 次の担当と必要な承認

1. **ユーザー（次）**：Stage S最終契約候補のcap-open規則、confidence、coverage、p128、resource・one-shot/STOP条件を正式採択するか判断する。
2. **Codex（採択承認後）**：採択記録と別の正式machine契約Cを固定・公開し、科学計算せずSTOPする。
3. **ユーザー（さらに別）**：固定S/Cに対するStage S one-shot科学実行を明示承認する。
4. **Codex（実施承認後）**：新しい専用authorityを固定して一度だけ評価し、全48報告・結果/failure・公開監査を保存してSTOPする。

今回その順序を自動進行しない。**保存信号の再解析・実shot予測・native compile・新規科学計算0件。mandatory STOP。**

[review]: ../../research/m1s_stage_s_scientific_conditions_review_20261017.md

## 固定SOURCEとmachine候補

SOURCE S：`7a338c4eba778c4ce1f189cfa8c8f050afb5df08`（9ファイル）。[source manifest](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/source_manifest.json)、[machine契約候補](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/execution_contract_candidate.json)、[結果schema](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/result_schema.json)。Sをancestorとして候補を固定し、将来の正式Cと別実施承認Aのcommit順序・bytes一致をguardで照合する。今回は候補のみで、C/Aは発行しない。
