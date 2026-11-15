# B2向け：PR-DF信号共通入口

入口SOURCE：`2a5e085e56d917607debcc18cec4faf7fa5a3539`。

- API：`src/trottertracks/pr_df_signal_entry.py`
- CLI：`scripts/prepare_pr_df_signals.py`
- 検査：`tests/test_pr_df_signal_entry.py`
- 数値provider：`fefd548ecd56f60d72031b3d385f6f5e6ee95d06:src/trottertracks/pr_rpe_signal_provider.py`
- B canonical validator：`437ba4acdb82708b6fc98e3cd119a2a734f7cf64`（旧LD7/LD9 helperと同じloader）

新しい科学実行は0件。保存4構成52 roundを再利用し、LD7/LD9のbundleを元形式と完全一致で再構成した。入口17件＋既存価格入口17件PASS。

## B2から呼ぶ

```python
from trottertracks.pr_df_signal_entry import DFSignalEntry

entry = DFSignalEntry(repository_root)
plan = entry.plan(generated_configuration)
requests = [r for r in plan['rounds'] if r['decision'] == 'REQUEST']
unsupported = [r for r in plan['rounds'] if r['decision'] == 'UNSUPPORTED']
```

入力はB2のcanonical configuration本体、生成結果の `configuration` 欄を持つitem、または `{commit,path,selector}`。未生成のB specから候補を作る機能はない。

出力は各roundの `REUSE`／`REQUEST`／`UNSUPPORTED`。REUSEには元rawとbundleのlocator、REQUESTには元providerへそのまま渡す `job`（B identity/law/physical_context）と保存reference、UNSUPPORTEDには理由が入る。κの宣言・IDだけの変更は再利用する。LD7/LD9をR=1やevent数の一致で取り違えない。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python \
  scripts/prepare_pr_df_signals.py plan \
  --input docs/research/d_pr_df_signal_entry_20261116/EXAMPLE_PLAN_INPUT.json \
  --output /tmp/df-signal-plan.json
```

固定SOURCE viewは、既存の `d2-pr-df-ld7-signal-source-fefd-20261116` を既定で読む。別配置では `FixedRuntime(root, view=...)` または `--signal-view` を使う。viewのHEADとprovider/metadataの固定bytesを確認する。共有環境の更新は不要。検査環境は既存Python 3.11.0rc1／NumPy 1.26.4／SciPy 1.14.1。B2の新しいpackageと混在しないよう固定providerは別namespaceで読む。

## 対応公式と未対応

元providerの実装対応：`2nd`、`4th(m5_best)`、K=2/4/6、非空head/tailを持つ保存H4 DF10のPR-DF。元fragment順、one-body別head、prefix固有λ_R・s_R、cの別SOURCE、総Tでのscalar一回適用を保持する。S2/K2以外は今回H4の実信号で検証していない。

未知公式/K、Pauli、PF endpoints、H/state/snapshot・head/tail・scalar/law SOURCE不一致、保存Full-H referenceのないT、不正canonical構成等は未対応。binary64の物理T/stepとcanonical有理数が一致しないroundも未対応。入力scalarを丸めて合わせたり、guardを無効化したりしない。元B validatorを通した上で、正常roundのjobは元plannerと同一、precision不成立roundのみを分離する。

REQUESTはb>1やunitarity/closure等の数値gateを通った意味ではない。元providerは取得時にそれらを評価する。Pauli側のpiece用normalizer上限修正をDFへ無条件に移す処理はない。本入口には実信号を実行するCLI mode自体を設けていない。

## 取得後に既存intakeへ渡す

後続で実行する場合は、REQUESTのjobに対して元providerの `load_h4_context`（configurationごとに一回）、`saved_h4_reference`、`evaluate` を使う。その科学実行は今回行っていない。元providerのmacro cacheはprefix・公式・実step・R・K・piece/配分を含む。LD7のcontextをLD9へ置き換えずに流用しない。

実信号の実行は、既存LD7/LD9経路と同様、固定SOURCE viewのsrc/scriptsを先に置いた新しいworker processで行う。provider内には `trotterlib.rte` 等の絶対importもあるため、B2が既に読み込んだ別版の数値moduleと混在させない。今回の別namespace化と検査はmetadata接続の確認であり、将来workerの数値import graphを検査済みにするものではない。

1. 取得済みrawを固定locatorで渡す：`entry.stage(config, fresh_output_directory, obtained={m: raw_locator, ...})`。REUSEを上書きするobtainedは拒否する。κ aliasは数値欄を保持したcarrierへidentityだけをrebindし、元SOURCEを残す。
2. `CONFIGURATION.json`、`raw/m0.json`〜`raw/m12.json` とSTAGINGをcommitで固定する。欠測・不一致・gate失敗では `PARTIAL_OR_INVALID_NO_BUNDLE`。失敗recordは別名で保全し、再実行しない。
3. CLI `bundle` に `{"carrier":{"commit":"固定SHA","directory":"公開出力先"}}` を渡す。13個のrawとconfigurationを読み、`B_MEANS.json` を生成する。既存rawが既に単一commitなら、`EXAMPLE_BUNDLE_INPUT.json` のように `configuration_source`＋`raw_sources` でもよい。
4. bundleを公開し、既存Q3 caseへ `signal_packet_kind="A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED"` とその固定locatorを渡す。元 `accept_saved`／total bindingを使う。Q3のAST guard・worker・評価契約には変更がない。

intakeが要求する「全rawが一つのpublication commit」を緩めない。元rawが複数commitに分かれる場合やidentityが変わった場合にstageを使う。configuration/rawのtop-level carrierが必要で、selector付きrecordを直接bundleに使わない。原結果は上書きしない。

## C/B2が今使える固定bundle

公開commit：`0bcd1df98c7896fb91896cc23e8d8c6105611408`。

| 分割 | path | 状態 |
|---|---|---|
| LD7 | `docs/research/d_pr_df_signal_entry_20261116/B_MEANS_LD7.json` | 保存13 meanと完全一致、元intake受入 |
| LD9 | `docs/research/d_pr_df_signal_entry_20261116/B_MEANS_LD9.json` | 保存13 meanと完全一致、元intake受入 |

configuration、元raw carrier、数値SOURCE、reference、targetは各bundleからたどれる。[PUBLISHED_INTAKE_CHECK.json](PUBLISHED_INTAKE_CHECK.json) は公開blobを元 `27e70efe533862f38f4080aabb9cdd3ad1749fe9:src/trottertracks/rpe_signal_statistics_caller.py` の `accept_saved` で読んだ結果で、各13 mean・`statistics_executed=False`。D2自身の技術受入であり、Cの科学的受入やQ3適合の再認定ではない。

価格は `7cd8ea139d703573f0da25a9ee5951f6930f0a25` で公開済みのPriceRulesを使う。B2の `47f86eb20` のMODEL接続も確認済み。本入口は価格を作らず、既存の係数・model/direct/不足区分を変更しない。

## 重複照合・引継ぎ

DFの元planner/providerとLD7/LD9 caller、B2の生成・価格接続、AのPauli共通入口とprecision修正、関連branch履歴・対象jobを確認した。同等DF入口・同一取得jobは確認範囲で未同定。A/B2の作業には介入していない。

次の利用前には、新しく公開された同一物理条件のmean、未完了job、固定kernel/descriptor/referenceの変更と出力先の競合だけを差分確認する。saved inventoryはentry生成時の固定リストなので、新しいbundleを追加する際はfresh entryで再構築する。不明な保存物を取得要求へ自動変換しない。

ロードマップへの差分：B2のPR-DF生成候補をround別の信号取得要求へ接続でき、取得後も旧intakeを使える。新しい条件の信号取得・数値gate・Q3・価格較正は後続。ロードマップ本体、固定SOURCE・原結果・旧STOPは保全。今回の技術整備・公開で終了する。
