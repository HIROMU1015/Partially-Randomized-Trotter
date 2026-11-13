# B/C/D HANDOFF：固定H4全13roundの構造価格経路

[REPORT](REPORT.md)／[静的coverage](prepared/COVERAGE.json)／[未実行SCIENCE_PLAN](SCIENCE_PLAN.json)／[数値契約の適用条件](NUMERICAL_ADDENDUM.md)。技術SOURCE `de2427d4a31d6b4b39d2c8eb8b63e3c972ffb89e`、branch `codex/r-variable-compiled-price-design-20261111`を継続。

## 現時点のデータ

- configurationは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。B最新公開branchは`3216d27a69ac8532d2a22e78f5f94491d202d238`と確認、B正本030f3f54のconfiguration JSON bytes/SHAを照合した。
- 固定H4 DF10/S2/L_D1/δ.1/K2/h.1/M12/P0、元H/state/snapshot/324 component/9 basis・preparation hashを保持。exactκはAの保存文字列をcopyする。R=255/510/1020/2040/4080/8160を丸めない。
- m0/m6の既存四値は83d219cc6のK1と全record一致でcopy。新しい物理処理のselectorはそれらを拒否する。他11round/22軸のmean S/interval、新law確率/角度はnull。
- 今回scientific --executeは起動していない。準備CLI、人工/mock8＋回帰18＝26 method PASSだけで、他Rの物理PASSではない。

## 最小取込みとAPI

新`src/trottertracks/h4_structural_schedule.py`、`scripts/run_h4_structural_schedule.py`、対応testと、旧coreへの3hookをまとめて取り込む。旧coreのhookは、explicit Tを使うsuffix、同γ16式のenclosure評価関数指定、partial wire kernel保存。旧callerのdefaultを維持し、元Accounting/MeanKernel/closure/provider/B controllerは不変。

`prepare(repo_root)`は固定GitのB configurationと保存K1を読み、metadataを返すだけ。物理Context・law API・S・mean・sampler・compiler・fit・N/Gを呼ばない。Bはこのmetadataを読むためにscience backendを起動する必要はない。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/project-python -B scripts/run_h4_structural_schedule.py --out /fresh/metadata-directory
```

科学入口`execute(repo_root, source, fresh_out)`は、**将来の明示許可後だけ**使用する。CLIでは`--execute --source <固定技術SOURCE>`を指定する。新grant/承認証明のファイルは不要。計画の具体的commandはSCIENCE_PLAN.jsonにあり、今回は実行していない。

## 物理K0とK1の扱い

同preparationを1回復元し、元APIから残11lawを取得する。全componentの元metadata・SOURCE演算、phase/resetの拡張確認を共通で行い、law別にwire state graph・parity/閉包・mass/longdouble/区間・scalar/XY terminalを検査する。新Rでも旧state数やK0 PASSをそのまま採用しない。

batchはm1,2,3,4,5,7,8,9,10,11,12。共通条件/あるroundのK0がUNKNOWN/FAILなら当該K1と後続roundを開始しない。完成wire kernelはpartial、完成roundはそのまま保存し、STOPを残す。PASSとkernel/state保存後にのみ当該roundのcosine/sine二値を評価する。追加の近似・threshold変更・law置換・自動retryはない。

計算は既存MeanKernel/Band・matrix二乗法で、全event列を作らない。新しい外向きinterval powerは元γ16式をO(log(qR))回の固定精度乗算で囲み、巨大FractionのO(qR)bit保持を避ける。物理Zの加算順やlaw、既存mass tolerance1e−10は不変。実remaining roundの性能・数値区間は未測定。

将来の出力は共通LAWS/検査metadata、各mのK0・kernel NPZ・状態辞書・K1、全体COVERAGE・STOP。m0/m6の四値はSAVED_VALUESとしてcopyするだけ。成功時はコピー4＋新22＝26の構造mean recordが同configurationへ結び付く。これは予定で、今回22値は欠測。

## B/Dとの科学的な境界

```text
quantity = FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT
expectation = IDEAL_FINITE_K_SOURCE_LAW
mean_compiled_C = null
qualified_for_compiled_price = false
full_rpe_resources = null
```

FULL_WRAPPER_MEAN_COMPILED_PRICEとしてBのC providerへ渡さない。full wrapper・軸・scalar/P0の構造数であり、normalization Bを価格へ乗算しない。実sampler対ideal法則の距離・compiler差・rare寄与/長q systematic・共有covariance/total CI、全N/RMSE/Gは別に未資格のまま。

Codex DがE[C−S]を担当する。Cから渡すのは同identity/law/source/軸/unitと保存構造kernel/meanの出典。Dのfit・compiled校正・有限cohort統計をCで再実装/再解析しない。ideal E[S]と有限cohort Dの意味・不確かさが一致しないままE[C]と認定しない。

## 次の担当とSTOP

**ユーザー**はSCIENCE_PLANの残11round一括実行を承認するか判断する。**Codex C**は許可後に固定SOURCE/metadataと数値契約のまま一括処理する。**Codex B**は13round identity/欠測を参照し、**Codex D**は別の承認scope内で残差を扱う。A/B/Dの作業領域は変更していない。

元m0/m6 K0/K1、失敗run01、C1/C2/S0a/old SOURCE/input/raw/結果/KEEP/STOPは不変。今回の技術成果を公開・remote SHA/tree/主要成果物照合後mandatory STOP。新科学計算へは進まない。
