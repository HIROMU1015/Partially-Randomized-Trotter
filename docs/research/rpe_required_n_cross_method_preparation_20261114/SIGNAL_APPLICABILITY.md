# 保存signal・decoder・riskの適用可能性

保存ファイルの存在、metadata接続、科学的risk資格を分離する。今回のraw確認はcopy・SOURCE・identityの一致確認のみで、確率・MSE・N・誤差率を再計算していない。

| 系列 | 全13round / 入力 | total/Q3接続 | 不足・次の担当 |
|---|---|---|---|
| PR-Pauli LD1/LD2 | A7142476の各13raw、辞書/hash/λ/scalar/bit順、per-round SOURCE、finite-K normalization・uncertaintyあり | eecbcのPauli intake→T.create/bind。caf1417の各1control診断が完了。再実行なし | **ユーザー/B**：主推定量・risk契約・閾値・有限縮小案の判断と許可。physical資格は未取得 |
| PR-DF LD1/LD2 | A91349のB_MEANS各13、元configuration a9abf8d、数値fefd548、raw carrier1621093。rawとcarrierの同一性、canonical law/identityを確認 | 旧caller2906の受入実績と旧DF DPあり。`A_FIXED_GRID_510FBACD_V1`のためT.createが拒否 | **B**：A_specと旧decoder契約を固定するgeneratorに対し、版付きdecoder-only bridgeが必要。現generic ceil-gridやPauli辞書/κ/scheduleへ変換しない。現在total canonical IDはnull |
| Full PF | 同条件H_eff 71f04be7/878000d8、Full PF各公式の二時点mean・保存catalogの限定signalあり。確認したSOURCE内に正式全13raw packetは未同定 | C.prepare＋T.create/bindを人工13roundでPASS。科学input不足のまま実packetを作らない | **A/C**：採択menuの全spectral成分・early actual幅・scalar/target/sourceを持つ13rawを提供。保存固有系利用でも新mean算出には別科学許可。BはH_effを再実装しない |
| Truncated PF p7/B0I/S2/.1 | 5fb2c32の`truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/`にPLAN＋B_SIGNAL_INPUT、13raw/reference、B=1、scalar一回・law=deterministic、数値SOURCE5a172629→5cc7c07e | 今回、保存packetのSOURCE照合→C.prepare→新total canonicalのmetadata PASS | **ユーザー/B**：opt-inのtarget/loss/whole-vectorと科学範囲。**A/C**の切断＋PF固有値shiftを全RPE MSEに代用・二重加算しない |
| Full-H κ2/K2/qdyadic | ce6aacfの`d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/`にPLAN＋B_SIGNAL_INPUT、13raw/reference、finite normalizer/scalar/one-body込みlaw、数値SOURCE5ccc1dfc→5cc7c07e | 今回、保存packetのSOURCE照合→C.prepare→新total canonicalのmetadata PASS | **ユーザー/B**：ideal IID/model target/loss/vector/有限risk範囲の別採択。実sampler移送・ground・数値certificateは未取得 |

## 元/新canonical（decoder契約だけのview）

| 系列 | 元ID | opt-in total ID |
|---|---|---|
| Full-H κ2 | `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e` | `265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612` |
| Truncated p7/B0I | `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc` | `b8ceb77562c4648f2ae51d7beebdc4069edb0f273d316dfcaee500d1d33015fb` |

これらはmetadata生成IDで、主推定量採択・MSE評価結果を示さない。全13roundの元physical ID、元/新canonical candidate、law、qPF/R/event_count/TとSOURCEは[READINESS](READINESS.json)を参照する。Full-HではqPF=null、物理qはgroupingであり、s=qRを保持する。

Full-H R448の元IDは`cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`。κ2とはR、event数、normalizer、raw、physical IDが別物で、R448 packetのκ2への結合は拒否テスト済み。共通referenceがあってもlaw/raw/価格を共有しない。DF B3とも区別する。

PR-DFの元IDはLD1 `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、LD2 `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。旧DPと今回Pauli total結果は同じcaseとして上書き・比較しない。

## 重複を避ける再利用境界

- Pauli caf1417の2完了結果、同run/旧停止runのcoarse・lemma、Aの全26roundを再評価・再取得しない。
- PR-DF旧DP e7ce2380は元partial推定量の結果として保全。total decoderの直接risk結果として流用できない理由は推定量・loss・辞書/lawの違い。
- Full-H κ2とp7はすでに13rawがあるので、次のrisk許可を得てもsignal生成は不要。保存normalization/uncertainty/referenceをそのまま使用する。
- Full PFのH_eff固有値shiftは、状態の全スペクトル重みやfinite-shot branch/aliasのMSEを代替しない。二時点検証から未取得11roundを補完しない。699178のscreenには数値gate STOPもあるため、その保存値を無条件に適格13roundへ昇格しない。
- 固定price guard・receiptとC plannerは再利用。新canonical IDだけを理由に旧価格receiptを再発行しない。law/cohort/full-wrapper/compiler/lowering/sourceが合う場合だけ元receipt付きbridgeを使う。適格E[C]なしでは正式G/winnerはUNKNOWN。
