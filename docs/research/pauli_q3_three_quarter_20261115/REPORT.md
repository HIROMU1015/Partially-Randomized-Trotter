# Pauli Q3 3/4配分の限定2評価とPF shortlist受入

2026-11-15 JST。B1＝**B担当のPauli Q3**、B2＝PF 6候補のmetadata受入・次回計画。DのPauli B1-W/V/CAL価格検証は別作業で、今回の価格callは0。

## B1：結果

保存H4/1.00 Å/STO-3G/DF10/sector(2,2)、PR-Pauli LD1/LD2、K2、元κ・全13round・保存stateの下で、各1配分だけを評価した。両軸共通の新vectorは`[48,45,42,39,36,33,29,26,23,19,16,13,9]`、allocation IDは`PAULI_LD1_USER_SCALE_3O4_01`／`PAULI_LD2_USER_SCALE_3O4_01`。元control各要素のceil(3N/4)で、十分N・最適Nではない。

ユーザー採択済みZ1＋T1/Q3、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、ideal IID・`EXACT_AFFINE_OF_BINARY64_RAW`を使用。十進εE=`0.00159360010199040`からτ=εE²をexact有理数で固定し、既存`q3_summary`で判定した。

| 構成 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 |
|---|---:|---:|---|
| LD1 | 6.207999522801600473833666e−7 | 6.207999522802494898227110e−7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| LD2 | 6.207498269405979019554498e−7 | 6.207498269406694956690684e−7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |

τの表示は約2.539561285063813282041692e−6 Ha²。判定は表示値ではなくexact端点による。[SUMMARY.json](SUMMARY.json)は表示用、exact端点・全SOURCE・入力は[LD1](run01/RESULT_LD1.json)／[LD2](run01/RESULT_LD2.json)が正本。**今回は2 science call、各1回で終了し、retry・追加配分なし。** 元controlのcaf1417結果を再計算・再filterしていない。

## 上界・確率・数値診断

既存first-unsafe lemmaの上界はLD1約7.738157021903122e−6、LD2約7.377553516364259e−6 Ha²。今回の最良有効上界は両方ともtotal状態集約DPであり、既存kernelの**有効上界のmin**を採用した。lemma超過だけを精度未達と判定していない。各手法の上界を加算していない。

| 診断 | LD1 | LD2 |
|---|---:|---:|
| bias²区間の上端 Ha²（表示） | 9.5388318093947613e−9 | 9.4812923727143418e−9 |
| Var区間の上端 Ha²（表示） | 6.1126112047085473e−7 | 6.1126853456795515e−7 |
| 追跡外残余mass（表示） | 4.5299857326926341e−21 | 3.9078961773526891e−21 |
| 残余loss寄与上界 Ha²（表示） | 8.9442439344410778e−20 | 7.1593713618602983e−20 |
| principal guard不成立として残余へ会計したmass（表示） | 1.0540794294826604e−24 | 3.1382414357366639e−27 |
| 全countに基づくany-zero確率（表示） | 2.0161372612904071e−8 | 2.0141439813198215e−8 |

zeroはHOLD、tieはargminとして全事象lossへ残す。tieの追跡prefix確率・zero確率・各roundの残余はSUMMARYとrawに全13件保存。例えばm10のtie追跡massはLD1約9.3876292683815897e−8、LD2約9.3791053343810031e−8であり、残余を含む全履歴の厳密tie率と同一視しない。total decoderには未定義出力がない。raw内の`partial_decoder.unresolved_probability_bounds`（約1.14e−7）は**旧partial規則の参照診断**であり、新decoderの未解決率ではない。

first-unsafeの13分割とall-goodの和がexactに1、各roundのgood＋unsafeがexactに1であることを確認した。unsafeは安全集合から外れる事象であり、実際の誤branch・zero・tieとは別。DPは既存kernel/backendの各round exact質量保存検査を通過し、残余lossも捨てていない。有限binary64閉包検査は両方`VERIFIED_FINITE_BINARY64_SUPPORT`、検査組912,226。ρ=1は変更していない。

非終端frontierは8192以下。m12終端の保持状態数はLD1 15,854／LD2 15,808で、次roundへのfrontier制限とは別。各構成の遷移53,006,168組は固定上限141,559,036組以内。数値allowanceは元のper-transition 1e−12 radというengineering allowanceであり、厳密certificateではない。

## SOURCE・再利用・実行負荷

- 科学caller/INPUTS固定：**f54b41a2e311e44124c9e180e26b01fe11d4eabd**。B基盤9c0e12d87、total kernel710aa69b、accelerator9e294778、C c628e9e8、A Pauli7142476を再利用。
- 新callerは既存accelerated runner→`rpe_total_binding.evaluate`を各1回呼ぶ薄い接続。元INPUTSからallocationだけを変更し、採択された外部model閾値を追加。保存canonicalの歴史的`PROPOSED`フィールドを改竄せず、現在の採択条件を`adopted_evaluation`に明示した。元target/raw/lawと旧科学結果は不変。
- canonical total ID：LD1 `9fa3c47dbdef3b7d08d55cc02c9451cbb400ebbcf807b817db1abb72c5cd946c`、LD2 `285bf4812e07a519f266ed88b83293bc4675493c5a9cc03985a6312363c6b8f0`。旧→新bridgeとSOURCEは元INPUTS・結果の`deterministic_input`に保全。
- 2構成並列、各8worker・outcome batch8は既存人工benchmarkのplateauと完了実行で確認したbackend設定を再利用。恒常的なCPU/RAM上限ではなく共有設定も変更していない。
- [COMPLETION](run01/COMPLETION.json)：全体wall **163.259秒**。LD1 wall162.256秒／process-tree CPU261.495秒／観測peak process-tree RSS4,033,644 KiB。LD2 wall160.680秒／CPU256.241秒／RSS4,030,656 KiB。RSSは定期観測値で厳密peakではなく、parent/child個別ru_maxrssもrawへ保存。科学process終了確認済み。
- [範囲とcount-only preflight](SCOPE_AND_PREFLIGHT.md)、[INPUTS](INPUTS.json)、run01のstages・coarse/lemma/DP・資源観測を保存。事前上限はraw確率を評価せずcount supportから算出した。

## B2：6完成packet

A REPORT1adb7e65、数値SOURCEfb9a1878、carrier06a15ee4の`PACKET_INDEX`と**AによるB9c0e12d87 intake6/6 PASS**を再利用した。新たなpacket生成やintake再実装はしていない。[PF_PACKET_ACCEPTANCE.json](PF_PACKET_ACCEPTANCE.json)が受入viewである。

`accept_completed_pf_shortlist.py`は元canonical/total-view、13round、SOURCE、law、physical candidate、signed scheduleをC0b4b2d94の既存ALIGNMENTと照合するだけ。raw/reference/normalization/target/scalar/actual幅は元packetとAの保全証拠を参照し、変換・再解析しない。Cの旧coverageを上書きせず、完成情報だけをSOURCE付きで重ねた。6/6 metadata PASS。MSE/N/Gは未評価/null、数値allowance・UNQUALIFIEDも元のまま。

[PF_FINITE_PLAN.md](PF_FINITE_PLAN.md)に、deterministic B=1に対応する既存paper型scheduleを出発点とした**提案**を記載。Pauli配分を移植していない。6候補の新MSE・DP・N評価は0件。

## 技術検証・資格・ロードマップ影響

追加5＋既存canonical8＋高速DP7＝**20 test method PASS**。[TESTS.txt](TESTS.txt)。人工worker・exact閾値境界・欠測/矛盾・SOURCE/law/physical混在拒否・Pauli/C guard・旧kernel保全を確認した。テストで実H4 riskを起動していない。

ユーザー採択済みのQ3モデル契約による今回2vectorの判定と、ground認証・実sampler law移送・数値certificate・物理的RMSE保証を区別する。bias²＋Varは診断で、AのδE/δE²を加算していない。旧Minkowskiは別参照として保全。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。価格がない状態でshot数だけのwinnerを作っていない。

統合担当への影響報告：①Pauliの3/4 whole-vector2件が採択Q3閾値を満たした、②PF shortlist6構成の信号不足・total metadata接続待ちはA完成packetと受入証拠で解消、③PF6構成のQ3/MSE・必要Nと全方式の適格価格/Gは引き続き未完了。旧「Q3契約未採択」の判断待ちは今回ユーザー指示で解消したが、物理的資格は解消していない。**ロードマップ本体は編集していない。** API・A/C引渡しは[HANDOFF](HANDOFF.md)。

公開・remote照合後**mandatory STOP**。新shot探索、PF/PR-DF/Full-Hの新risk、十分N認定、価格・Gへ自動進行しない。
