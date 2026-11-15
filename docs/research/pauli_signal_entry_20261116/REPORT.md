# PR-Pauliのconfigurationから13 roundの信号packetへ進む共通の入口

想定読者：B2（configurationを作る側）、B（packetを取り込む側）、統合担当。呼出し方と出力の欄は [HANDOFF.md](HANDOFF.md)。
検証結果の正本は [VALIDATION.json](VALIDATION.json)、人工例の計画は [PLANS.json](PLANS.json)。

## 冒頭の答え

| 項目 | 内容 |
|---|---|
| 作ったもの | B canonicalなPR-Pauli configurationを1つ受け取り、roundごとに「保存信号の再利用」「既存providerへの取得要求」「未対応」を決める入口。取得後は、LD1〜LD5と同じMEAN_HANDOFFを組み立てる。SOURCE `93a5ef3b9`（branch `codex/pr-r-dependent-n-model-20261111`） |
| 分割専用の作業 | 不要になった。LD3・LD5のような専用runner・intakeを書き足さずに、同じ入口で計画・取得・組立てができる。辞書・演算子・λ・s_Rは、configurationの記述子にある保存場所から読む |
| 保存LD3・LD5 | 手修正なしで、公開済みpacketと同じ行（記録・mean_source・identity）に到達した。Bの `bind_packet`（`b6d4dea49`）もPASS。LD1・LD2も同じ |
| κ・Δ_PFを変えた人工例 | 新規の取得要求になったのは、物理条件（R・event数・q・実step幅）が変わったroundだけ。LD3・LD5の6例すべてで、要求したroundと変わったroundが一致した |
| providerの対応 | 出力の各roundに、検証済み（2nd・K=2）／providerの範囲内だがPauliでは未検証（K=4、4th(m5_best)）／未対応（理由付き）を付けた |
| 計算 | 新しい平均値・compile・Q3は0件。新規roundは、人工のstubと、平均値の直前で止める確認（`--dry-context`）でだけ通した |

## 入口の動き

**入力**：次のどれでもよい。
- B canonical configurationのfile、またはGitの場所（commit:path）
- MEAN_HANDOFF（`--configuration-id` で1つ選ぶ）
- B2のNEEDSの候補（`configuration_id` と `signal_source` を持つ行）
- B specのfile。Bの `generate` でcanonical configurationにしてから使う（B2が分割やparameterを変えた場合）

**roundごとの照合**：保存された平均値の記録と、次の物理条件をすべて比べる。
- H・状態・snapshot（compile profileは除く）
- 辞書の中身：sha256、種類、写像、係数・scalarの扱い、event空間
- 分割：L_D、randomized/deterministicの範囲、headのPF方式、λ_P、保存s_R、1体項の扱い
- law：model、event model、scalarの扱い
- 公式、signed weights、pieceごとの配分とその規則
- T、q、実step幅、K、R_outer、event数
- 数値計算の種類（DIRECT_EXPM、paired Taylor、samplingなし）

configuration・候補のID、保存場所、κ、compile profileは比べない。いずれも平均値に入らないためである。
保存記録は、記録を算出したproviderのfile（provider本体、`pf_power_stability`、`h4_four_method_screen_mean`、`rte`）がP1 kernel `e8641b84f` とbyte一致する場合に限って使う。

**判定**

| 判定 | 意味 | 出力 |
|---|---|---|
| REUSE（同一identity） | 同じconfigurationの保存記録 | 元のcarrierをそのまま指す |
| REUSE（re-bound） | 物理条件が同じで、configurationが違う | 信号・数値・SOURCEは変えず、identityだけをこのconfigurationのcanonical値に付け替えた記録を、新しいcarrierに書く。元の場所・IDは `reuse` 欄に残す |
| REQUEST | 一致する保存記録がなく、既存providerで計算できる | 具体的な取得要求。provider・kernel・context（共通演算子と、このL_Dの `Pauli_nonidentity`）・参照値（保存H4のround）・実行経路を書く。保存roundとの差分の欄も付ける |
| UNSUPPORTED | 既存providerで計算できない | 理由のcode |

providerの対応区分は、全roundに付ける。

| 区分 | 条件 |
|---|---|
| `SUPPORTED_VALIDATED_FOR_PAULI` | 2nd・K=2。LD1/LD2/LD3/LD5で実際に使った条件 |
| `SUPPORTED_NOT_VALIDATED_FOR_PAULI` | 4th(m5_best)、またはK=4・6。providerの範囲内だが、Pauli tailで計算したことがない |
| `UNSUPPORTED` | 理由を出す。例：公式・Kが範囲外、構造がproviderの構造と違う、このroundのTに保存参照値が無い、Pauli辞書でない、λ・s_R・Hが保存値と違う |

**取得と組立て**
- `--execute`：REQUESTのroundだけを、LD3 stage 2の経路で計算する。経路は変更していない：`S3.load_context`（LD2でP1 `load_contexts` と照合）、P1 `job`、保存H4参照値、固定provider `evaluate`。
  - 計算は `--approve-new-means` を付けた場合に限る。無い場合は、re-boundのcarrierだけを書いて止まる。
  - 最初の失敗で止まり、再試行・再開はしない。
- `--assemble`：保存carrierと、今回commitしたcarrierから、MEAN_HANDOFFとROUND_SUMMARYを作る。Bの `bind_packet` と同じ条件を確かめてから出力する。

## 検証

### 1. 保存LD1・LD2・LD3・LD5（再利用だけで組み立てる）

| L_D | configuration | 13 round | 公開packetとの比較 | Bの条件（入口内） | Bの `bind_packet`（b6d4dea49） |
|---|---|---|---|---|---|
| 1 | `d905d219…` | すべてREUSE（同一identity） | 一致 | PASS | PASS |
| 2 | `d7dd4f52…` | 同上 | 一致 | PASS | PASS |
| 3 | `dd5b881d…` | 同上 | 一致 | PASS | PASS |
| 5 | `4d6e2c9c…` | 同上 | 一致 | PASS | PASS |

- 比較は各行のすべての欄で行った。違うのは `reused_without_reanalysis` だけである（今回は全行が再利用なのでTrue）。
- `mean_source` は元のcarrierのまま。例：LD3は `26bac4f32`／`8c8ce542e`、LD5は `1e3985e7e`／`c67469ad4`。
- LD3・LD5の `configuration_source` は、公開packet中のconfigurationを指す。中身は、公開packetの `configuration_source` と一致した。

**自分の保存信号を隠した場合**（`tests/check_pauli_signal_entry.py`）：LD3・LD5とも、13 roundすべてが検証済みの取得要求になった。
- 各要求の物理条件は、公開済み記録の物理条件と一致した。
- 入口が汎用に読んだ辞書itemは、分割専用の読込みと一致した（LD3は `L.dictionary_item`、LD5は `F.dictionary_item`）。演算子の場所も同じである。

### 2. 人工例：B2がparameterを変えたconfiguration

元のLD3・LD5 specを編集し、Bの `generate` で新しいconfigurationを作った。「変わったround」は、canonical行の (T, q_PF, 実step幅, R, event数, 配分, weights, 公式, K) を比べて、入口とは独立に数えた。

| 例 | 変わったround | 取得要求 | re-bound再利用 | 未対応 |
|---|---|---|---|---|
| LD3 κ×3/2 | m4–m12 | m4–m12 | m0–m3 | なし |
| LD3 κ×2/3 | m3–m12 | m3–m12 | m0–m2 | なし |
| LD3 Δ_PF=1/5 | m2–m12 | m2–m12 | m0–m1 | なし |
| LD3 Δ_PF=binary64の0.2 | m1–m12 | m1–m12 | m0 | なし |
| LD5 κ×3/2 | m10–m12 | m10–m12 | m0–m9 | なし |
| LD5 κ×2/3 | m12 | m12 | m0–m11 | なし |
| LD5 Δ_PF=1/5 | m2–m12 | m2–m12 | m0–m1 | なし |
| LD5 Δ_PF=binary64の0.2 | m1–m12 | m1–m12 | m0 | なし |

- 8例すべてで、取得要求したroundと変わったroundが一致した。
- κの例では、要求roundの差分の欄が R_outer・event数・配分になった。Δ_PFの例では、実step幅・q・event数になった。
- Δ_PFの2つの例でm1の扱いが分かれるのは、実step幅の違いによる。
  - 正確な1/5：m1のT（binary64の0.2）が1/5よりわずかに大きいので、q_PFは2、実step幅は0.1のままになり、再利用になった。
  - binary64の0.2：q_PF=1、実step幅0.2になり、取得要求になった。
  - 入口は、名目のΔではなく、Bが決めた実step幅で照合している。

**providerの対応区分**

| 例 | 結果 |
|---|---|
| K=4、公式 4th(m5_best)（LD3・LD5） | 13 roundすべてが取得要求。区分は `SUPPORTED_NOT_VALIDATED_FOR_PAULI` |
| h=1/20（LD3・LD5） | m1–m12は、保存のm0–m11と物理条件が一致した（同じbinary64のT・q・実step幅・R・event数）。round番号をずらしたre-bound再利用になった。m0（T=0.05）は保存参照値が無く、`NO_SAVED_REFERENCE_AT_THIS_ROUND_T` で未対応。providerには参照値の新規計算の経路が無い |
| DF tailのconfiguration（Bがcanonical化した元のLD2） | 全roundが未対応。理由は `NOT_PAULI_DICTIONARY` |

**B2の入力形式**：B2のrun02 NEEDS（`a4b4d0e1f`）の候補行を、そのまま入力にできた。
- L_D=2・3・5：保存configurationに解決した。
- L_D=4（configuration無し）：`CONFIGURATION_AND_SCHEDULE_REQUIRED_FIRST` で止まり、先に必要なstage 1（P0辞書とconfigure）の経路を示した。

### 3. 取得から組立てまでの通し（人工stub）

LD3・LD5の κ×3/2 と Δ_PF=1/5 の4例で、`--execute` から `--assemble` まで通した。
- 取得要求のroundには、人工stubが答えた。記録には `SYNTHETIC_STUB_NOT_A_MEAN` の印を付け、providerは呼んでいない。
- stubが呼ばれたroundは、取得要求のroundと一致した。例：LD5 κ×3/2ではm10–m12の3回。
- 再利用roundの信号は、保存値と一致した。
- packetは13 roundそろった。入口内のBの条件も、Bの `bind_packet` もPASSした（後者は記録を行から渡した人工確認）。
- 承認なしで実行すると、re-boundのcarrierだけを書き、`STOP_REQUESTS_NOT_APPROVED` で止まった。stubの呼出しは0回だった。

### 4. 実経路の準備確認（平均値なし）

SOURCE `93a5ef3b9` で、LD5 κ×3/2 を `--execute --dry-context` で実行した。記録：[dry_context_LD5_kappa_x3_2/](dry_context_LD5_kappa_x3_2/)。
- code SOURCEのbyte照合：11 file一致。
- LD2でのcontext builderの照合（P1 `load_contexts` と比較）：一致。
- LD5のcontextを構築し、m10–m12の保存参照値（`bba150834` PHYSICAL_INPUTS rounds[10–12]）を確認した。
- 平均値の直前で止めた（`DRY_CONTEXT_READY_NO_MEAN`）。所要時間は約5秒。
- 平均値・正規化子の新規計算は0件。

### テスト

`tests/check_pauli_signal_entry.py`：6件PASS（約21秒）。[TESTS.txt](TESTS.txt)
- 既存の `check_df_head_pauli_tail_split_signals.py`・`check_df_head_pauli_tail_split_stage1.py` もPASSした。

## 残課題と担当

| 項目 | 担当 | 内容 |
|---|---|---|
| Bの取込み | B | `accept_saved` は、LD1/LD2・LD3・LD5の組立てSOURCEだけを受ける固定の一覧を持つ。新しいpacketは、`bind_packet` を同じ引数で呼べば取り込める（`_ld3_metadata_helpers`、LD2のtarget）。一覧を広げるか、`handoff_source` を受ける汎用の入口が必要 |
| 新しいL_D | A（stage 1） | 辞書が無いL_D（例：4、6–9）は、先に汎用のstage 1（`scripts/run_df_head_pauli_tail_split_stage1.py --p0/--configure --prefix P`、`d6385593e`）でP0辞書とconfigurationを作る。その後この入口に渡す |
| 未検証の公式・K | 利用者の判断 | 4th(m5_best)、K=4・6は、providerの範囲内だがPauli tailでは未検証。実計算の前に、検証の要否を決める必要がある |
| 保存参照値の無いT | — | h変更の初回roundなど、保存H4参照値が無いTは、既存providerでは計算できない（未対応として出す） |
| 実取得 | 利用者の承認 | REQUESTのroundの実計算は、新しい科学計算である。`--approve-new-means` は、承認を得てから付ける |

## ロードマップへの影響（統合担当へ）

- PR-Pauliの分割・parameterを変えるたびに必要だった、専用runner・intakeの作成が不要になった。B2の候補から信号までが、計画→（承認後の）取得→組立ての3段になった。
- 科学的な結論は変わらない。新しい平均値は無い。
- ロードマップ本体は編集していない。

## 高速化・再利用の手法としての記録

手法集 `docs/PERFORMANCE_METHODS.md` は本branchに無い（`codex/d-full-h-p2-caller-prep-20261114` にある）。次の追記案を統合担当へ渡す。

- 重複計算の削減：保存した信号平均値を物理keyで照合し、同じ条件のroundは計算しない。
  - 実装：`src/trottertracks/pauli_signal_entry.py`（`93a5ef3b9`）
  - 条件：kernel fileのbyte一致、物理keyの全項目一致
  - 根拠：保存4 configurationの再組立て、人工8例
  - 実対象：計算の削減量は、実取得を行っていないため未測定

## 範囲と保全

- 保存資料と人工入力だけを使った。新規の平均値・compile・Q3・価格・N・shot探索は0件。
- 既存のcode・原結果・packet・旧STOPは変更していない。LD3 stage 2の経路（`8c8ce542e`）とP1 kernel（`e8641b84f`）はbyte一致で使った。
- 候補の選び方とshot探索はB2の担当で、この入口は扱わない。
- 検査の性質：作成者自身による技術検査で、独立レビューではない。Bの `bind_packet` はBのcodeをそのまま呼んだ確認である。
- mandatory STOP。
