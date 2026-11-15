# C・統合担当向け：PR-DF LD9共通V0完了

**LD9は共通V0で適合。LD7と全13 R=1・予定event数qRは同じで、条件付きモデルMSEはLD9が約4.037E-12 Ha²低かった。** 今回新規configuration1、signal13 round、Q3一回、compile0。LD7は保存結果の再利用だけ。LD9の価格・総費用は未評価のまま。

| 分割 | MSE lower (Ha²) | MSE upper (Ha²) | 判定 | 今回の取得 |
|---|---:|---:|---|---|
| LD7 | 4.28414160318294721301353E-7 | 4.28414160318294721365383E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 保存V0再利用 |
| LD9 | 4.28410123342266259264443E-7 | 4.28410123342266259328471E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 新規1 call |

共通X/Y vector：`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。各軸別の整数列は同一で、poolしない。モデルMSE差のexact区間はSUMMARY参照。全R1でqRは各roundのqと等しいことはmapperの出力で確認したもので、samplingや価格の観測ではない。

## Cが読む固定入口

**join commit：`778118926c85158f947928c1e744d565464aab17`**

- `docs/research/d_pr_df_ld9_v0_results_20261116/C_HANDOFF.json`：configuration/source・original/total configuration ID・binding ID→V0配分→Q3原結果とMSE判定。signal packetとfixed Q3 inputsのlocatorを含む。axis_prices/total_model_cost/formal_E_C/Gはnull。
- 同directory `SUMMARY.json`：LD7保存V0とLD9新規V0のexact MSE、R/event列、MSE差の区間、件数/資格。
- `ROUND_INDEX.json`：全13のnormalizer b/hex/τ、raw/corrected/exact-tailと原raw locator。X/Y共有の複素信号。
- `SOURCE_LOCATORS.json`：全40桁SOURCE/input/raw locator。表の短縮SHAはここで解決可能。

| 内容 | 固定commit | path（当directoryから） |
|---|---|---|
| 生成SOURCE | 303ed2ef705d0a2b24cff1c9118058386e247ca1 | ../../../scripts/run_pr_df_ld9_v0.py（repo pathはscripts/run_pr_df_ld9_v0.py） |
| 生成前入力 | 04090592a | GENERATION_INPUTS.json |
| LD9 configuration・scalar・両R列 | 33a308bd7693511d798b48836284729d3ac86f9f | generation/CONFIGURATION.json、SCALAR.json、LD7_LD9_SCHEDULE.json、SCHEDULE.csv |
| 13信号の原raw | 2f717a8071e6d3ae31fd57fa38dff7021395bb22 | signals/m0.json〜m12.json、signals/RESULT.json |
| 全13 B bundle | f117269c14507efc38b64d226670823a17718653 | B_MEANS_LD9.json |
| Q3観測前固定条件 | 54b10d2473df0fdc2c5bfc7991e4071428a88a53 | Q3_INPUTS.json |
| 一回のQ3原結果・API・閉包・DP・資源 | 560ef11e734644ea9d0df00fc49bd90a75dda2a8 | q3/LD9_V0/RESULT.jsonほか、q3/COMPLETION.json |
| LD7保存共通V0 | 0defa286c53a1b9acf513be453907bcad83102b3 | ../d_pr_df_ld7_v0_results_20261116/SUMMARY.json rows[2] |

LD9 configuration ID：`ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833`。one-body別head＋DF0..8 head、DF9 tail。保存H4 linear1Å/STO-3G/tol-only DF10、元H/state/snapshot・fragment順、S2/Δ=.1/K2/共通paperκを維持。LD9 λ_R=1.5605698403540042e-07、s_R=5.193351122457893e-08。c=2.29310124732とは別SOURCE、c+s_R=2.2931012992535114を総Tで一回だけ適用。

## 実行・gate・資格

全13のpiece b=1.0000000000000004（hex0x1.0000000000002p+0）、元b>1 gateを通過。元provider fefd548e、binary64・NumPy1.26.4/SciPy1.14.1、丸め・normalization・guardの変更0。数値gateFAIL0/技術停止0/未取得0。b=1の場合の最初のroundでの停止・診断保存は人工入口検査済みだが、実データの失敗とは混同しない。bが1に近いことを理由に新しい資格や高精度保証を付与していない。

Q3は3.11.0rc1、元worker・AST guard・decoder・target・all-event loss・exact τ=15136964828156312001/5960464477539062500000000 Ha²。元intakeが13 raw/identity/law/normalizerを照合してから一回実行。API==RESULT・閉包・count scope・exit0を確認した。MSE適合は保存ideal finite-RTE raw IIDモデルの条件付き判定で、ground/物理全誤差/厳密数値資格やC自身の独立受入とは別。詳細資源はREPORTと原RESOURCE_FINAL。

LD7 rowのQ3_disposition等は元実行の履歴をそのまま保持している。今回のLD7再評価は0で、今回のreuse/new件数はSUMMARY.countsに示す。Cは価格未評価を0 RZに置換せずnullとして扱う。

読取例（exact MSEの大きな整数に対応）：

```python
import json, subprocess, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
joined = json.loads(subprocess.check_output([
    "git", "show",
    "778118926c85158f947928c1e744d565464aab17:docs/research/d_pr_df_ld9_v0_results_20261116/C_HANDOFF.json"
]))
```

これは保存JSONの読取だけで、Q3/fit/価格取得を起動しない。

## ロードマップ差分とSTOP

LD9を「descriptorあり・未評価」から「現行configurationと13信号、共通V0 Q3適合」へ進められる。LD7/LD9の予定event数が同じで精度側にわずかなモデル差があることまで判明した。より大きなheadと異なるtailのcompiled単価は分からないため、低費用性やLD7/LD9の費用順位は未判定。LD9固有価格を調べるかは別指示で決める。価格モデル/配分探索は今回は未実施、LD7係数の移送もしない。

§10.5ではf89・0def・5c5の調査を再利用し、その後の同H/state/snapshot/split/law/条件の成果・job・SOURCE・出力競合を差分照合。新13 physical IDと旧archiveの一致0、同等現行信号/Q3・未解決対象0。PR-Pauli LD3/決定論PF/Y4/LD7のC joinを別条件と区別し、他worktree/jobは非介入。後続実行前は今回公開後の差分を確認する。

ロードマップ本体は統合担当が更新。元SOURCE/raw/旧STOP、LD7の較正・holdout・最小適合配分を保全。厳密E[C]/CI/G/winner・全L_D最適性は未認定。commit/push・remote一致確認後mandatory STOP。追加信号・price fit・配分探索・価格compile・retryは行わない。
