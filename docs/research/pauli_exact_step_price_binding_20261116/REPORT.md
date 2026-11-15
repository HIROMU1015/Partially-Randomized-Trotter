# PR-Pauliの厳密ΔPF=1/5と価格接続UNKNOWN

**m0/m1は、保存ΔPF=binary64 0.1の同roundと同じ物理recipeへ接続できる。m2～m12は保留する。** ただしこれはrecipeのmetadata照合であり、別configurationの価格receiptを付け替えた受入ではない。厳密1/5をbinary64 0.2に置換するとm1からqが変わるため、その価格を流用できない。

Cのexact guardを緩和する根拠はなかった。今回、既存guardと同じ検査からround・不一致fieldを返す`pauli_round_binding`を追加し、通常の`plan`の例外にも`m=2 q=3 fields=delta`を表示するようにした。条件・tolerance・law・SOURCE・価格資格は変更していない。人工5＋既存回帰34＝39 test PASS。新しい実signal・sampling・compile・fit・Q3はすべて0件。

## SOURCEと、UNKNOWNまでの実経路

| 段階 | 固定参照・確認した関数 |
|---|---|
| Bの具体例 | `ff9d25d62ada75fc56e89540459c15d49157b90c` : `docs/research/pauli_common_entry_intake_20261116/HANDOFF.md` |
| Bの通常intake | `bd42dd1e95eb1d4152216d62792ad3421b803f7c` : `rpe_pauli_signal_intake.accept_saved` → `bind_packet` → C `plan` → `provider_binding_check` |
| BがロードするC | `c628e9e8447c11c984064b922cbce6c52dc72d48` : `rpe_optimization_pipeline._pauli_plan`。Bの`rpe_c_interfaces`が依存を接続する。C全体の差替えはしていない |
| canonical schedule | A/Bが参照する`ce0517de6e7b58cf4bf4c69f99f998d86099c406` : `pr_rpe_rmse_controller.generate`、`_assemble`。T=h·2^m、q=ceil(T/要求Δ)、実step=T/qを有理数で固定 |
| A physical job | `714247619afb4714fd663eb15c0470ef32bdc7af` : `df_head_pauli_tail_p1.job`。Bのqをそのまま保持し、Tと実stepをfloatへ変換 |
| A signal provider | 同系の`pr_rpe_signal_provider.evaluate`は、canonicalのT/実stepとphysicalのbinary64値をexact比較。`_macros`はpiece時間をT·w/qの順で計算 |
| Aの後続修正 | `4e5b8bef634e3f071fe63df9683d7245df984d98` : `pauli_signal_entry.round_support`に`ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT`を追加 |
| Aの固定metadata結果 | `08812cd30cf46c73346695cc20b6552c8359314a` : `docs/research/pauli_signal_entry_20261116/PLANS.json`。同commitの`docs/research/pr_pauli_m5_k2_composition_20261116/REPORT.md`に旧HANDOFFの訂正あり |
| Dの物理回路入口 | `8b28dce02cc5b5da1a8f4f1718f0b9f855759978` : `pauli_tail_price_connection.round_request`、`sample_steps`、`compose`。metadataだけを読取。実行していない |

Bの厳密1/5例は、**人工carrierによるsignal intakeの確認**である。人工値を受け入れたことは、Aの実H4 providerがその条件で平均signalを算出した証拠ではない。実際、Aの修正版入口はm2～m12をUNSUPPORTEDとする。signal-only投影の成功を根拠に、Cのexactなprovider identity制約を外すことはできない。

§10.5としてC plannerの関連Git履歴、Bの上記固定成果、Aの初版・修正版と保存LD3/LD5 packetを照合した。同等のC round別診断は確認範囲に無く、Aで既に済んだ整数化・未対応判定を再利用した。他担当のprivate実行領域・process・SOURCEは変更していない。

## 全roundの差分

以下は保存H4 metadata（h=binary64 0.1、S2、finite K2）の照合。LD3/LD5ともqは共通だがRは異なるため、[ROUND_BINDINGS.json](ROUND_BINDINGS.json)では別々に記録した。表の実stepは表示値であり、正本はJSONの有理数とbinary64 hexである。

| m | T表示 | q（厳密1/5） | 実step表示 | q（binary64 0.2） | 厳密1/5のC recipe／A入口 |
|---|---:|---:|---:|---:|---|
| 0 | 0.1 | 1 | 0.1 | 1 | MATCH／保存0.1と同一recipe |
| 1 | 0.2 | 2 | 0.1 | 1 | MATCH／保存0.1と同一recipe |
| 2 | 0.4 | 3 | 0.13333333333333333 | 2 | UNKNOWN／実step非binary64-exact |
| 3 | 0.8 | 5 | 0.16 | 4 | 同上 |
| 4 | 1.6 | 9 | 0.17777777777777778 | 8 | 同上 |
| 5 | 3.2 | 17 | 0.18823529411764706 | 16 | 同上 |
| 6 | 6.4 | 33 | 0.19393939393939394 | 32 | 同上 |
| 7 | 12.8 | 65 | 0.19692307692307692 | 64 | 同上 |
| 8 | 25.6 | 129 | 0.19844961240310077 | 128 | 同上 |
| 9 | 51.2 | 257 | 0.1992217898832685 | 256 | 同上 |
| 10 | 102.4 | 513 | 0.19961013645224174 | 512 | 同上 |
| 11 | 204.8 | 1025 | 0.1998048780487805 | 1024 | 同上 |
| 12 | 409.6 | 2049 | 0.19990239141044414 | 2048 | 同上 |

m1ではT=3602879701896397/18014398509481984が1/5より大きい。厳密1/5ならceil(T/(1/5))=2、binary64 0.2なら比は1となる。同じ表示「0.2」でもouter反復数・head回数・tail grouping・event数が違い、同じ回路とは扱えない。m1以降の全12roundで、この2つの要求幅の物理keyは異なる。m0だけは同じkeyになる。

m2ではcanonical実stepは3602879701896397/27021597764222976であり、A jobのstepは`0x1.1111111111111p-3`。Dの`float(T)/q`も同じhexになるが、**丸め後のoperandが一致することと、現行signal SOURCEのcanonical identity条件を満たすことは別**である。A provider自身も後者で拒否する。今回、このroundを価格接続済みに昇格していない。

## piece・物理回路への変換

対象のS2はsigned pieceが1個、w=1.0、allocation=[R]。Aの物理時間は`T*1.0/q`、Dは`float(T)/q`であり、Tが元binary64のままなので今回のmetadataでは各roundで同じbinary64 operandとなる。event時間はstep/R。qとRを再整数化しない。piece時間・head半時間・event時間のhex、R、event数を52行に保存した。

Dは順序付きDF head→Pauli tail→逆headを各outerでcomposeし、scalarは総Tで一回だけancilla phaseへ適用する。tailは元辞書・lawとD2 central rotation control（`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`）を使う。nativeは`af284d853890155bacd19e61850c196bd1d5c660`の`ax2b_native_df_v5.build_partial_native`。測定付きwrapper、X/Y、compiler、scalar SOURCEの既存guardも維持する。T/qだけの一致でそれらの証拠を代替しない。4th(m5_best)は11 pieceであり、このS2用D adapterに読み替えない。

## 接続してよい範囲と不足証拠

- **同じ値の表現差**：有理数で書いたbinary64のexact値と、そのfloat表現は既存比較で通る。厳密1/5のm0/m1は、元Δ=0.1の保存signalとA自身の`physical_key`が一致し、保存計画のrebindと整合した。Cのround recipe検査も通る。元carrierのID/SOURCEを保持した明示bridgeが必要であり、異なるcanonical IDの価格receiptを自動付替えしない。
- **別回路**：厳密1/5とbinary64 0.2のm1～m12はqが違う。保存価格・係数を移送する根拠はない。元のq/R/実step/piece/law/scalar/wrapper/軸を満たす証拠が必要。
- **同じ丸め後operandでもSOURCE不足**：厳密1/5のm2～m12は、現行A signal providerでは未対応。必要なのは、q・lawを保持したexact canonical T/qと物理binary64との対応を定義・検証するAのSOURCE、そのSOURCEに結び付くsignal record、同じ物理recipeの価格証拠である。単にCだけをfloat比較に変更しても解決しない。今回はAのSOURCEも科学条件も変更していない。

whole-schedule `plan`は従来どおりm2で停止する。診断のmatched roundを使って、部分的に正式Gへ進むことはない。既存比較カタログ・価格・Q3・モデル資格を更新していない。

## 再実行と検証

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
scripts/trace_pauli_step_binding.py \
  --output docs/research/pauli_exact_step_price_binding_20261116/ROUND_BINDINGS.json
```

保存PLANSのconfiguration ID・job metadata・rebind判定と再構成したmetadataが52/52で一致した。保存signal値の数値演算は行っていない。新人工5件は表現差、m1のq差、1 ULP差、丸めだけの偽一致、通常plannerの保留round表示・SOURCE保持を確認。既存`check_paper_g_price_routes.py`34件はplanner→provider binding→D request→価格guardの人工接続、および保存LD1/LD2 context照合を再利用。[TESTS.txt](TESTS.txt)参照。作者自身の技術検査であり、独立科学レビューではない。

原C/B/Aの固定SOURCE・入力・結果・旧STOPは保全した。新科学計算は行わず、公開照合後mandatory STOP。
