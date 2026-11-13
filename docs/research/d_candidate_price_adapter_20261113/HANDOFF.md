# A/C/B向け：candidate contextとprice requestの接続

2026-11-13 JST。[REPORT](REPORT.md)／[初回取得計画](FIRST_ACQUISITION_PLAN.md)／[SOURCE・再現](SOURCES.md)。例はAPIの説明で、実科学実行の承認ではない。

## 1. 受動metadata API

```python
from trottertracks import candidate_price_adapter as D

request = D.from_b(
    config, m, physical_candidate,
    input_source=input_reference, execution_source=new_execution_commit,
    axes=('X', 'Y'), preparation_hash=expected_preparation_hash_or_none,
)
# A821c pr_rpe_signal_plan.plan のjobを受ける場合：
request = D.from_a_job(
    config, a_job, input_source=input_reference,
    execution_source=new_execution_commit, axes=('X', 'Y'),
)
pending = D.pending_price_records(request)
```

configは既存B V2 controller又はsource-bound adapterが生成した全schedule。Dはconfigを生成・選択せず、B._validateを使う。A job内のB_configuration_id/B_candidate_id/B_identity/B_law_bindingとphysical_contextを照合する。signal値やNは読まない。

physical_candidateは元carrierのid規則、system/method/formula/prefix/T/δ/q/R/Kを持ち、B rowのstructureと一致させる。q_RPEとq_PF、piece allocation、signed順序を区別する。input_sourceは既存input/loaderのSOURCE・hash参照、execution_sourceは今回のadapterを含む新commitを指す。承認証明書ではない。

B profileにlowering_idがない旧schemaでは、元identityを上書きせず`physical_context.price_profile.lowering_id`に新ZZを明示する。既存A fixed-gridのone-body指定省略はそのSOURCE定義に従ってDETERMINISTICとして解決し、generic B candidateは明示指定を要求する。

Bのexact時刻/δとphysical rowのbinary64が一致しなければ拒否する。別の丸め付きbridgeを黙って作らない。未対応は探索候補の不適格性ではなく、価格経路のOUTSIDE_DOMAINとして扱う。

## 2. 科学実行を別途許可された場合のAPI

callerは元H/state/snapshot・input bytesとprovider SOURCEを既存loaderで復元・照合し、対応する準備済みHamiltonianを渡す。グローバルなpath・共有venvを書換えず、専用processのPYTHONPATH/src/scriptsと指定providerを利用する。NativeRuntimeは実際に使用するprovider moduleのbytesと固定c31ef424を照合し、異なれば停止する。

```python
runtime = D.NativeRuntime(
    repo_root, restored_hamiltonian,
    binding=config['spec']['binding'],
    input_source=input_reference, execution_source=new_execution_commit,
)
# ここからが科学的なpreparation / sampling / price acquisition：
ctx = runtime.context(request)
path = runtime.sample(request, cohort=fixed_cohort, index=fixed_index)
result = D.Adapter(repo_root, runtime, cache_root=read_only_cache).acquire(
    request, path, cohort=fixed_cohort, index=fixed_index,
    directory=fresh_path_directory, conditional=False,
)
```

`context`自体がbasis/preparationを構築する科学入口である。今回の実H4でこの例を実行していない。新grant/auth機構は要求しないが、既存個別SOURCE・実行範囲・STOP条件と将来の明示科学承認は必要である。

元sample_requestsのseedはcandidate/cohort/index/outer/pieceを含む。既存pathを明示復元してacquireへ渡す場合は、**同じruntimeのctx.prep**を使う元DFPartialS2StepRequestを用い、seed/roleとconditionalを正しく指定する。異なるprepare object、K、time、tail hash、control、groupingを拒否する。供給pathをnormalと指定しただけで全law標本として認定する機能はない。

各Runtime/Adapterは単一threadのprocess専用。既存lowering/hash contextは可逆なmodule swapを使うため、thread並列は避け、独立pathをprocessで並列化する。並列度は実利用可能資源とrun依存に応じて選び、任意のCPU/RAM上限を課さない。共有環境や他ユーザーのprocess/file/settingsを変更しない。

## 3. 保存ファイルと価格の意味

| file / field | 用途 |
|---|---|
| events_PRIVATE.json | そのpathのevent列・seed・order law・conditional。元C1形式、rawは新directoryのみ |
| STRUCTURAL.json | 同eventのS、digest。compiled meanではない |
| cosine/sine.started.json、軸.json | 個別wrapper fingerprintとcompiler/6指標/cache hit。指定されていない軸はcompileしない |
| result.json、TECHNICAL_TIMING.json | 元fast Backend形式。単軸/逆順ではaxesを付けて配列の対応を明示 |
| CANDIDATE_PATH.json | 元request/B identity、resolved preparation/tail/scalar/λ/fragmentとkey、軸別S/C/D、6指標、cache reference、seed/law、input/execution SOURCE |
| mean_compiled_price / confidence_interval | 常にnull、qualified_expectation=false。path≠母平均 |

cacheは旧cacheのfingerprint/compiler一致を読取再利用するだけ。CANDIDATE_PATHは新physical requestの実回路照合結果であり、旧P1/P2の平均やdevelopment roleを書換えた記録ではない。実回路が違えば同じevent数でもcache hitしない。新しいcache書込管理layerは追加していない。

## 4. 平均価格・区間への後続接続

- **A/探索担当**：physical candidateとwhole scheduleを供給。Dの対応範囲外の候補は探索空間に残す。
- **C**：必要な場合だけ同context/lawのE[S]・数値enclosure・featureを供給。固定L_D1の26値を別L_Dへ移さない。
- **D**：許可範囲でpath価格を取得。平均/CIを評価する場合は既存paired_residuals/cohort_summary等を再利用し、context・law・profile・roleを揃えた両軸pathだけを渡す。片軸応答は勝手に補完しない。
- **B**：現在のpending_price_recordsは既存平均価格recordのquantity/measurement/identity/lawを持つが、mean=null・NOT_EVALUATED・law資格nullなので最終Cへ使えない。path出力からmean欄を埋める自動fallbackはない。

後のfinite-cohort meanはfinite scenario、全law E[C]やCIは別の資格である。十分な価格区間をBへ渡すにはtarget ideal/sampler law、同時被覆、rare、S数値誤差、model/window/fit誤差等の必要項目を別の承認scopeで評価する。[価格探索HANDOFF](../d_rpe_price_search_design_20261113/HANDOFF.md)のsidecar仕様を保持し、現在根拠のないintervalを生成しない。

outer-window法をこのadapterへ接続して正式modelへ昇格する処理はない。Sやevent数をC上下界としない。N/decoder、signal pipeline、候補探索・実G計算も呼ばない。

## 次の担当

**ユーザー/B/探索担当**が最初のcandidate/round/axes・精度/confidence又はpilot目的・有限path数を固定する。**D**がそのSOURCE/入力/seed・compile予算を具体化して承認後だけ実行する。**C**の新feature評価も必要時のみ別途許可範囲で行う。今回は公開照合後mandatory STOP。
