# 次の限定科学計画：保存certificateによるsingleton期待転送

2026-11-05 JST。技術SOURCE準備済み・未実行。
本計画を含む公開commit SHA（最終報告に記載）への別途明示科学実行許可が必要。
旧b0c354ce0の1 pass・18/36結果・STOPを変更しない。

## 固定方法と入力

[分析・証明](SUPPORT_BOUND_ANALYSIS.md)のη_r閉形式を採用する。
componentのd_lには旧218個のoperator上端をそのまま使い、新しいprojector差を計算しない。
component lawは保存uniform53 grid mass、colorは(source basis ID,hash)、
order mass/θは保存6rのK2 law。model τ、probability64へ置換しない。
θのsinだけを96-bit Arbで外向き評価し、残りはexact Fraction。

[FIXED_BINDINGS.json](FIXED_BINDINGS.json)に固定commit/path/blob/SHA256。
旧6原本はb0c354ce071af71a6ce857b11f8acc0c129a05b8のVALIDATION.json経由でbyte確認する。
RESULTからd_l、boundary/lowering/旧uniform上端、DESCRIPTORSから全componentのsource colorを読む。
component grid massは元m17 categorical_laws.json、ID順は元source_replay.jsonに対応させる。
old base.static_inputsにより保存N/α/β/ε・18tuple・signal τ/θ・law/policyと旧SOURCEを照合する。
旧229 ballsはbyte保全するが、新しいmatrix decode・norm評価はしない。

科学条件はlinear H4／1Å／STO-3G／保存legacy DF12、L_D3・δ=.02・K2、
8qubit・N4 sector70・保存normalized physical ray、identity抽出/cutoff1e-12、
保存m0..17/q2^m/r/N/α/β/ε/component/basis/sign/support/lawと同じ。

## 新しい科学call・精度

| 操作 | 固定範囲 |
|---|---|
| exact component/color集約・D0/D1/D2 | 218成分を1 pass（source color別probabilityと3 moment） |
| sin絶対値上端 | 保存6r×2order＝12、96 bit |
| singleton η_r | 6r各1 |
| mean追加上界 | 18round各1 |
| strict十分条件 | 保存36軸各1 |
| numerical precision / pass | Arb96 bit / 1 |
| H/state load、preparation、support再登録、basis生成 | 全て0 |
| 新component/projector/column差、mean matrix、CDF/atan、sampling/shot | 全て0 |
| trajectory、長q build/compile、D/費用/noise/状態準備 | 全て0 |

これは保存科学結果から新しい誤差上界とstrict判定を得る科学的再解析であり、
default technical inspectionや前回の実行許可には含まれない。
今回この表の数値評価は未実施。

## 評価と判定

D_j=Σ_l p_l d_l(1−h_b)^j（j=0,1,2）、h_b=Σ_(l:b_l=b) p_l。
r=1はη1=w0 s0 D0+w2[(1+s2)D1+D2]。
r≥2はηr=[2w0s0+w2(1+s2)]D1+
[(r−2)w0s0+w2((2r−1)+(r−1)s2)]D2。
s_nは保存binary64 θの|sin|に対するcertified上端、0..1に入れる。
符号相殺・実装event平均の積分解は使わない。
独立uniform53とrun_le_1・identity無しの前提を満たす場合だけ適用する。

Δ_mean=min(2,旧Δ_uniform,Δ_boundary+q min(2,ηr)+Δ_lowering)。
これはIID平均専用であり任意word/固定PCG64への新上界と扱わない。
旧I_cat/U_catのπ64項は再加算しない。
旧U_cat＋Δ_mean＋保存shot半幅上端 < 保存εをexact Fractionで比較する。
等号・missing evidenceはPASSとしない。N/α/β/εは変更・再計算しない。

全36軸strictなら
CONDITIONAL_IDEAL_CONTROLLED_IID_MEAN_LOCAL_CONDITIONS_CONFIRMED。
不足ならUNDETERMINED。旧18/36結果は別の一様上界結果としてそのまま保全する。
成功時もRPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=nullを維持する。

## 実行入口・資源・STOP

defaultはbyte/schema/literal検査だけで、新H4評価0。

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
 docs/research/rpe_support_bound_sharpness_20261105/run_weighted_transfer.py
~~~

以下は公開SOURCEへの別途明示許可後、Codexが一度だけ使う予定のcommand。
**今回は起動しない。**

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
 docs/research/rpe_support_bound_sharpness_20261105/run_weighted_transfer.py \
 --run --output artifacts/rpe_support_weighted_transfer/2026-11-05/run01
~~~

SOURCEがcommit済みであること、固定入力/Git objectが揃うこと、fresh outputを確認する。
既存出力へ上書きしない。RESULT.jsonにstage・moment・sine・ηr・18追加上界・36判定・開始数・停止理由を保存する。
計算量はO(218＋6＋18)のscalar処理で、q/r回の回路は生成しない。
共通のexact集約を一度だけ行い6rで共有する。12個のscalar sinにはspawn/GPUを選ばない。
これは逐次依存と小さいscalar workloadによる実装選択で、CPU/RAM/worker上限を課すものではない。
起動時のCPU affinity/MemAvailable/RSSを記録し、共有環境・他jobを変更しない。
未測定のspeedup倍率は主張しない。

input/hash/SOURCE/policy/ID順・law coverage不一致、非有限値、negative/missing bound、
runtime/正規化/IID条件不足はUNKNOWN/STOP。途中結果を保存し新候補へ進まない。
strict不成立だけなら固定36比較を完了してUNDETERMINED/STOP。
追加precision、retry/resume、projection/DPのH4評価、basis再生成、sampling、D研究へ自動拡張しない。
成功・失敗を問わず原本・公開SHAを照合してmandatory STOP。
承認証明書・新grant発行・approval text保存は不要。

担当：ユーザーが公開SHAと本固定範囲へ別途明示許可し、
Codexがその後に限って1 passを実行・報告する。
