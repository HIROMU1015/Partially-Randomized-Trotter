# A → C/B：追加幅Full PFの局所候補

[REPORT](REPORT.md)／[C packet](C_HANDOFF.json)／[5点表](CELL_SUMMARY.csv)／[全13round metadata](ROUND_SCHEDULE.csv)／[SOURCE_INDEX](SOURCE_INDEX.json)。固定計画d48d8732、科学caller a3005d72349e9a91c3229215b189e6cf8bed0000、旧結果3ae055faを保持した。保存H4 DF10/sector36/Full10、同H/state/snapshot/target・h.1/M12/P0・元予算/decoder/profile。

## Cへの3候補

| label | canonical configuration ID | 局所δE（Ha） | m12 qPF / pieces |
|---|---|---:|---|
| S2_16h_over_7 | `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d` | +6.99614984259e−4 | 1792 / 1792 |
| Y4_16h_over_3 | `a799697fc8ca36b024acec26af43313777948ead432e66804c670649a18afe76` | −5.90726632746e−4 | 768 / 2304 |
| M5_16h | `70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd` | −1.04921622697e−5 | 256 / 2816 |

これは局所予算に基づく比較優先候補で、全RPE RMSE資格・G winnerではない。S2/8h/3、Y4/32h/5は局所予算超過として同packet内に残し、結果を削除・科学的に全RPE FAILと再ラベルしない。

`configurations[].constructor`から既存C/D/B APIで完全configurationを再生成できる。全5配置の65行にexact T/actual幅・qRPE/qPF・canonical/physical candidate・対応固有系SOURCE・保存XY価格statusを保持した。初回planのexact h-relative有理数を表示floatへ置換しない。特に32h/5をfloat .64に置換するとceilが変わる条件がある。

early actual幅は元PLANどおり。S2/16h/7はm0=h、m1–3=2h、m≥4=16h/7。Y4/16h/3はm0=h、m1=2h、m2–3=4h、m≥4=16h/3。M5/16hはm0–3=h/2h/4h/8h、m≥4=16h。対応旧cellはSOURCE参照のみで再計算していない。M5/1.6の分枝窓は物理回路変更ではない。

3候補ともm12の価格は旧正本catalog照合域外。一部早期のEXACT_METADATA_MATCHは旧locatorの転記で、新価格受入ではない。full measured Hadamard X/Y wrapper、scalar、fragment/signed順、control/native/profile、compiler seed等を合わせる。q/piece比例の価格や別loweringの証拠を無条件に使わない。今回新compile/fit/予測/G0。

## Bへ渡す意味

元H固有系SOURCE付きcacheに対し、全5新stepの最大trace overlap groupはeffective最低group0だった。保存stateの全spectral重み・元basis overlap・残差は各`run01/cell_XX/EIGENSYSTEM.json`。新complex meanは直接とspectral各10で、各rowの`new_round_mean_available`は同一physical T/q/stepへの照合結果を参照する。Y4の共有m4を含めても新しい科学callは増えない。

M5/.8のwindow整合検査PASS、M5/1.6はwindow条件下で整数枝−1を1モードに適用して整合した。厳密なphysical branch/ground同定とは区別する。solver energyでbranchを合わせていない。

δEだけで全RPE energy bias・RMSE上界を認定しない。early exception、全spectral成分、保存state/prior/decoder/loss・shot統計の契約が必要である。直接raw由来riskへPF δEを二重加算しない。十分N/global RMSE/Gはnull、数値誤差rigorous flagはfalseを維持する。

## 次の作業とSTOP

- ユーザー：この3候補の価格/risk接続について、次に実行する有限範囲を決定・承認する。
- Codex C：承認された場合だけ、同exact scheduleの全round価格coverageを確認し、不足価格を取得する。
- Codex B：承認された場合だけ、同候補のstate/branch/shot条件と全RPE riskを評価し、同じN/Cの範囲で比較する。
- Codex A：今回の成果公開後mandatory STOP。M5/3.2や追加幅探索、N/Gへ自動進行しない。
